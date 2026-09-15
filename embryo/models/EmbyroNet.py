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
from embryo.utils.tools import AverageMeter, GatherMeter, ClsReortor
from embryo.data.embryo_dataset import MyEmbryoDataset, mmcv_collate, SubsetRandomSampler
from embryo.models.modules import MultiframeIntegrationTransformer, AdaFeatSelection, MSFeatureModulation, FeatureSelectionModule
from embryo.models.base_model import BaseModel
from embryo.models.custom_clip import build_backbone


def build_clip_vision_state_dict(ckpt_path):
    """Build the state dict used to initialize the frozen CLIP ViT-B/16 backbone.

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


class EQENet(BaseModel):
    """Embryo Quality Evaluation Network.

    A frozen CLIP ViT-B/16 encodes every sampled frame, multi-scale hidden states are
    modulated and then spatially / temporally selected, and a Multiframe Integration
    Transformer aggregates the frame-level cls tokens before the final classifier.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        self.adaptive_query_init(config)

    def adaptive_query_init(self, config):
        self.config = config
        num_classes = config.num_classes

        # frame encoder: frozen CLIP ViT-B/16, multi-scale hidden states [3, 6, 9, 12]
        self.backbone = build_backbone(config)
        self.loaded_keys, self.missing_keys = self.load_pretrained_backbone()

        self.ms_feat_idx = [3, 6, 9, 12]
        length = config.frame_seq_len

        self.ms_feat_aggregator = MSFeatureModulation(n_scale=len(self.ms_feat_idx), embed_dim=768, hidden_dim=192, reduction=4)
        self.temporal_selector = AdaFeatSelection(seq_len=length, dropout=0.3, pe=True, n_queries=8, embed_dim=768, n_layers=2, mlp_ratio=1.5)
        self.spatial_selection = FeatureSelectionModule(seq_len=length, input_dim=768, num_experts=8, num_heads=8, dropout=0.3)

        # temporal modelling runs on every second frame, hence length // 2
        self.mit = MultiframeIntegrationTransformer(length=length // 2, embed_dim=768 // 2, layers=4, mlp_ratio=1.5, dropout=0.5, pe=True)
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

        output_dict = self.backbone(pixel_values=imgs, output_hidden_states=True) # bt c

        scale_idx = self.ms_feat_idx
        hidden_states = output_dict["hidden_states"]
        selected_feats = [hidden_states[idx] for idx in scale_idx]

        ms_feats, ms_cls = self.ms_feat_aggregator(selected_feats)
        embed_dim = ms_feats.size(-1)

        ms_feats, spatial_attn = self.spatial_selection(ms_feats)
        ms_feats = ms_feats.view(b, t, embed_dim)

        spatial_out, temporal_attn = self.temporal_selector(ms_feats)
        spatial_out = torch.mean(spatial_out, dim=1, keepdim=False)

        ms_cls = ms_cls.view(b, -1, embed_dim)
        ms_cls = self.ch_reduce(ms_cls)[:, ::2, :]
        temporal_out = self.mit(ms_cls) # b t c
        temporal_out = torch.mean(temporal_out, dim=1, keepdim=False) # b c    

        out = torch.cat([spatial_out, temporal_out], dim=-1)

        logits = self.classifier(out)
        
        return dict(
                    logits=logits,
                    temporal_loss = self.temporal_selector.diversity_loss * 0.1,
                    spatial_attn=spatial_attn,
                    temporal_attn=temporal_attn
                    )  

    def load_pretrained_backbone(self):
        ckpt_path = getattr(self.config, "pretrained_ckpt", None)
        new_state_dict = build_clip_vision_state_dict(ckpt_path)

        self.backbone.load_state_dict(new_state_dict, strict=False)

        # collect the parameters that were not initialized from the pretrained weights
        loaded_keys = set(new_state_dict.keys())
        model_keys = set(self.backbone.state_dict().keys())
        missing_keys = sorted(list(model_keys - loaded_keys))

        return loaded_keys, missing_keys

    def set_trainable_params(self):
        loaded_keys = ["backbone." + key for key in self.loaded_keys]
        for name, param in self.named_parameters():
            if name in loaded_keys:
                param.requires_grad = False

    def build_dataloader(self, logger=None, world_size=1, global_rank=0):
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
            
        
        train_data = MyEmbryoDataset(root_path=self.config.root_path, ann_file=self.config.train_file, 
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
        
        val_data = MyEmbryoDataset(root_path=self.config.root_path, ann_file=self.config.val_file,
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
        self.set_trainable_params()

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
    def train_one_epoch(epoch, model: nn.Module, criterion, optimizer, lr_scheduler, train_loader, config, logger, writter):
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
            total_loss = criterion(logits, label_id) + outputs["temporal_loss"]
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
            writter.add_scalar("Train loss", tot_loss_meter.avg, epoch)

    @staticmethod
    def validate(epoch, model, criterion, val_loader, config, logger, writter):
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

                loss = criterion(logits, label_id) + outputs["temporal_loss"]
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
        cls_reportor = ClsReortor(preds, labels)
        report = cls_reportor.get_dict_classification_report()
        full_report = cls_reportor.get_str_classification_report()
        logger.info(f"\n {full_report}")

        dist.barrier()
        if dist.get_rank() == 0:
            writter.add_scalar("ValLoss", tot_loss_meter.avg, epoch)
            writter.add_scalar("Accuracy", report["acc"], epoch)
            writter.add_scalar("Precision", report["pre"], epoch)
            writter.add_scalar("Recall", report["rec"], epoch)
            writter.add_scalar("F1-score", report["f1"], epoch)

        return report
