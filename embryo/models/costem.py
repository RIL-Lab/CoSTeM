"""CoSTeM: Complementary Spatial-Temporal pattern Mining for embryo grading.

Reference: Y. Sun et al., "Time-Lapse Video-Based Embryo Grading via Complementary
Spatial-Temporal Pattern Mining", MICCAI 2025 (arXiv:2506.04950).
"""

import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from transformers import CLIPModel

# make the repository root importable no matter where the module is imported from
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from embryo.utils.tools import AverageMeter, GatherMeter, ClassificationReporter
from embryo.data.embryo_dataset import EmbryoDataset, mmcv_collate, SubsetRandomSampler
from embryo.models.modules import BypassAdaptationNetwork, MixtureOfCrossAttentiveExperts, TemporalSelectionBlock, TemporalTransformer
from embryo.models.base_model import BaseModel
from embryo.models.clip_image_encoder import build_image_encoder


def build_image_encoder_state_dict(ckpt_path):
    """Build the state dict used to initialize the frozen CLIP ViT-B/16 image encoder.

    The expected checkpoint is a dict containing a ``state_dict`` key whose keys are
    prefixed with ``vision_model.``. When the file does not exist, the weights of
    ``openai/clip-vit-base-patch16`` are downloaded from the Hugging Face hub instead.
    """
    if ckpt_path and os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    else:
        print(f"[WARNING] Pretrained checkpoint '{ckpt_path}' was not found, "
              "downloading openai/clip-vit-base-patch16 from the Hugging Face hub instead.")
        state_dict = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").state_dict()

    has_vision_prefix = any("vision_model." in key for key in state_dict)
    new_state_dict = {}
    for key, value in state_dict.items():
        if has_vision_prefix and "vision_model." not in key:
            continue
        new_key = key.replace("vision_model.", "")
        if new_key not in new_state_dict:
            new_state_dict[new_key] = value

    return new_state_dict


class CoSTeM(BaseModel):
    """Complementary Spatial-Temporal pattern Mining network for embryo grading.

    Frames are encoded by a frozen CLIP ViT-B/16 image encoder, whose intermediate
    features are modulated by a Bypass Adaptation Network. The resulting tokens are split
    into two complementary streams:

    * **Morphological branch** (patch tokens): Mixture of Cross-Attentive Experts (MCAE)
      followed by a Temporal Selection Block (TSB) mines the static *spatial* pattern;
    * **Morphokinetic branch** (class tokens): a Temporal Transformer mines the dynamic
      *temporal* pattern.

    The two patterns are concatenated and fed to an MLP classifier.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        self.build_branches(config)

    def build_branches(self, config):
        self.config = config
        num_classes = config.num_classes

        # frozen pretrained image encoder, intermediate features are read from these layers
        self.adaptation_layers = [3, 6, 9, 12]
        self.image_encoder = build_image_encoder(config)
        self.loaded_keys, self.missing_keys = self.load_pretrained_encoder()

        length = config.frame_seq_len

        # bypass adaptation: one MLP adaptor + channel reduction per selected layer
        self.bypass_adaptation = BypassAdaptationNetwork(n_scale=len(self.adaptation_layers), embed_dim=768, hidden_dim=192, reduction=4)

        # morphological branch: spatial pattern mining
        self.mcae = MixtureOfCrossAttentiveExperts(seq_len=length, input_dim=768, num_experts=8, num_heads=8, dropout=0.3)
        self.tsb = TemporalSelectionBlock(seq_len=length, dropout=0.3, pe=True, n_experts=8, embed_dim=768, n_layers=2, mlp_ratio=1.5)

        # morphokinetic branch: temporal pattern mining on every second frame
        self.temporal_transformer = TemporalTransformer(length=length // 2, embed_dim=768 // 2, layers=4, mlp_ratio=1.5, dropout=0.5, pe=True)
        self.ch_reduce = nn.Linear(768, 384, bias=False)

        self.classifier = nn.Sequential(
                                nn.Linear(768 + 384, 256),
                                nn.GELU(),
                                nn.Linear(256, num_classes)
                                        )

    def forward(self, batch: torch.Tensor):
        imgs = batch["imgs"]

        b, t, c, h, w = imgs.shape
        imgs = imgs.view(b * t, c, h, w)

        output_dict = self.image_encoder(pixel_values=imgs, output_hidden_states=True)

        hidden_states = output_dict["hidden_states"]
        selected_feats = [hidden_states[idx] for idx in self.adaptation_layers]

        # patch tokens -> morphological branch, class tokens -> morphokinetic branch
        patch_tokens, cls_tokens = self.bypass_adaptation(selected_feats)
        embed_dim = patch_tokens.size(-1)

        # spatial pattern: per-frame expert selection + key frame selection
        spatial_feats, spatial_attn = self.mcae(patch_tokens)
        spatial_feats = spatial_feats.view(b, t, embed_dim)
        spatial_pattern, temporal_attn = self.tsb(spatial_feats)
        spatial_pattern = torch.mean(spatial_pattern, dim=1, keepdim=False)

        # temporal pattern: global temporal modelling of the class token sequence
        cls_tokens = cls_tokens.view(b, -1, embed_dim)
        cls_tokens = self.ch_reduce(cls_tokens)[:, ::2, :]
        temporal_pattern = self.temporal_transformer(cls_tokens)
        temporal_pattern = torch.mean(temporal_pattern, dim=1, keepdim=False)

        out = torch.cat([spatial_pattern, temporal_pattern], dim=-1)
        logits = self.classifier(out)

        return dict(
                    logits=logits,
                    diversity_loss = self.tsb.diversity_loss,
                    spatial_attn=spatial_attn,
                    temporal_attn=temporal_attn
                    )

    def load_pretrained_encoder(self):
        ckpt_path = getattr(self.config, "pretrained_ckpt", None)
        new_state_dict = build_image_encoder_state_dict(ckpt_path)

        self.image_encoder.load_state_dict(new_state_dict, strict=False)

        # collect the parameters that were not initialized from the pretrained weights
        loaded_keys = set(new_state_dict.keys())
        model_keys = set(self.image_encoder.state_dict().keys())
        missing_keys = sorted(list(model_keys - loaded_keys))

        return loaded_keys, missing_keys

    def freeze_image_encoder(self):
        loaded_keys = ["image_encoder." + key for key in self.loaded_keys]
        for name, param in self.named_parameters():
            if name in loaded_keys:
                param.requires_grad = False

    def build_dataloader(self, logger=None, world_size=1, global_rank=0):
        # pixel mean / std of the embryo frames, measured on the training split
        img_norm_cfg = dict(mean=[122.52, 124.34, 121.65], std=[75.94, 76.18, 75.70], to_bgr=False)

        train_pipeline = [
            dict(type='DecordInit'),
            dict(type='SampleFrames', 
                 sample_start=self.config.sample_start,
                 sample_end=self.config.sample_end,
                 interval=self.config.sample_stride,
                 time_points=self.config.time_points,
                 num_frames=self.config.num_frames,
                 jitter=self.config.frame_jitter,
                 jitter_range=self.config.jitter_range),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(256, 256), keep_ratio=False),
            dict(type='RandomCrop', crop_size=self.config.input_size),
            dict(type='Flip', flip_ratio=0.5, direction='horizontal'),
            dict(type='Imgaug', transforms=[dict(type='Rotate', rotate=(-30, 30))]),
            dict(type='ColorJitter', p=0.5, brightness=0.8, contrast=0.4, saturation=0.2),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'quality', "time_slots", "grading", "mask"], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'quality', "time_slots", "grading", "mask"]),
        ]

        train_data = EmbryoDataset(root_path=self.config.root_path, ann_file=self.config.train_file,
                                   pipeline=train_pipeline, sample_ratio=1.0)

        sampler_train = torch.utils.data.DistributedSampler(
                train_data, num_replicas=world_size, rank=global_rank, shuffle=True
            )
        collate_fn = partial(mmcv_collate, samples_per_gpu=self.config.batch_size)

        train_loader = DataLoader(
            train_data, sampler=sampler_train,
            batch_size=self.config.batch_size,
            num_workers=12,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn
        )

        val_pipeline = [
            dict(type='DecordInit'),
            dict(type='SampleFrames', 
                 sample_start=self.config.sample_start,
                 sample_end=self.config.sample_end,
                 interval=self.config.sample_stride,
                 time_points=self.config.time_points,
                 num_frames=self.config.num_frames,
                 jitter=False,
                 jitter_range=None),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(256, 256), keep_ratio=False),
            dict(type='CenterCrop', crop_size=self.config.input_size),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'quality', "embryo_ID", "effective_duration", "year", "female_age", "time_slots", "grading", "mask"], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'quality', "embryo_ID", "effective_duration", "year", "female_age", "time_slots", "grading", "mask"])
        ]

        val_data = EmbryoDataset(root_path=self.config.root_path, ann_file=self.config.val_file,
                                 pipeline=val_pipeline)
        indices = np.arange(global_rank, len(val_data), world_size)
        sampler_val = SubsetRandomSampler(indices)
        val_loader = DataLoader(
            val_data, sampler=sampler_val,
            batch_size=self.config.batch_size,
            num_workers=12,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(mmcv_collate, samples_per_gpu=self.config.batch_size),
        )

        return train_data, val_data, train_loader, val_loader

    def build_optimizer(self, logger=None):
        self.freeze_image_encoder()

        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=self.config.weight_decay)

        return optimizer

    def build_scheduler(self, optimizer, n_iter_per_epoch, logger=None):
        num_steps = int(self.config.epochs * n_iter_per_epoch)
        warmup_steps = int(self.config.warmup_epochs * n_iter_per_epoch)

        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=0,
            warmup_lr_init=self.config.warmup_init_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

        return lr_scheduler

    @staticmethod
    def train_one_epoch(epoch, model: nn.Module, criterion, optimizer, lr_scheduler, train_loader, config, logger, writer):
        model.train()
        optimizer.zero_grad()

        num_steps = len(train_loader)
        batch_time = AverageMeter()
        tot_loss_meter = AverageMeter()

        start = time.time()
        end = time.time()

        for idx, batch_data in enumerate(train_loader):
            images = batch_data["imgs"].cuda(non_blocking=True)
            label_key = "quality" if model.module.config.task == "Evaluation" else "grading"
            label_id = batch_data[label_key].cuda(non_blocking=True)
            label_id = label_id.reshape(-1)
            time_slots = batch_data["time_slots"].cuda(non_blocking=True)
            mask = batch_data["mask"].cuda(non_blocking=True)

            batch = dict(imgs=images, time_slots=time_slots, mask=mask)

            outputs = model(batch)
            logits = outputs["logits"]
            total_loss = criterion(logits, label_id) + config.div_loss_weight * outputs["diversity_loss"]
            total_loss = total_loss / config.accumulation_steps

            total_loss.backward()

            if config.accumulation_steps > 1:
                if (idx + 1) % config.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)

            else:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)

            torch.cuda.synchronize()

            tot_loss_meter.update(total_loss.item(), len(label_id))
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.print_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f"Train: [{epoch + 1}/{config.epochs}][{idx}/{num_steps}]\t"
                    f"eta {datetime.timedelta(seconds=int(etas))}\t"
                    f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                    f"tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t"
                    f"mem {memory_used:.0f}MB"
                )

        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        dist.barrier()
        if dist.get_rank() == 0:
            tot_loss_meter.sync()
            writer.add_scalar("Train loss", tot_loss_meter.avg, epoch)

    @staticmethod
    def validate(epoch, model, criterion, val_loader, config, logger, writer):
        model.eval()

        gather_meter = GatherMeter()
        acc_meter = AverageMeter()
        tot_loss_meter = AverageMeter()
        print("start validation")
        with torch.no_grad():
            for idx, batch_data in enumerate(val_loader):
                images = batch_data["imgs"].cuda(non_blocking=True)
                label_key = "quality" if model.module.config.task == "Evaluation" else "grading"
                label_id = batch_data[label_key].cuda(non_blocking=True)
                label_id = label_id.reshape(-1)
                time_slots = batch_data["time_slots"].cuda(non_blocking=True)
                mask = batch_data["mask"].cuda(non_blocking=True)

                batch = dict(imgs=images, time_slots=time_slots, mask=mask)

                outputs = model(batch)
                logits = outputs["logits"]

                loss = criterion(logits, label_id) + config.div_loss_weight * outputs["diversity_loss"]
                probs = torch.softmax(logits, dim=-1)
                _, inds = probs.topk(1, dim=-1)

                acc = 0
                for i in range(images.size(0)):
                    if inds[i] == label_id[i]:
                        acc += 1

                tot_loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(float(acc) / images.size(0) * 100, images.size(0))
                gather_meter.add_batch(inds, label_id)

        acc_meter.sync()
        logger.info(f"Acc@1: {acc_meter.avg:.3f}")

        gather_meter.sync()
        preds = gather_meter.preds
        labels = gather_meter.labels
        reporter = ClassificationReporter(preds, labels)
        report = reporter.get_dict_classification_report()
        full_report = reporter.get_str_classification_report()
        logger.info(f"\n {full_report}")

        dist.barrier()
        if dist.get_rank() == 0:
            writer.add_scalar("ValLoss", tot_loss_meter.avg, epoch)
            writer.add_scalar("Accuracy", report["acc"], epoch)
            writer.add_scalar("Precision", report["pre"], epoch)
            writer.add_scalar("Recall", report["rec"], epoch)
            writer.add_scalar("F1-score", report["f1"], epoch)

        return report
