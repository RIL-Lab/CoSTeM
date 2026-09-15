import os
import sys
import random
import argparse
import datetime
import yaml
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard.writer import SummaryWriter

# make the repository root importable no matter where the script is launched from
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from embryo.utils.logger import create_logger
from embryo.models.costem import CoSTeM
from embryo.utils.tools import save_checkpoint
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # task arguments
    parser.add_argument("--task", type=str, choices=["Evaluation", "Grading"], default="Grading")
    parser.add_argument("--num_classes", type=int, default=3, help="class number of the dataset.")

    # data arguments
    parser.add_argument("--root_path", type=str, default="../data/embryo_videos", help="The folder path that saves raw videos: root_path/year/F12345/embryo_1.avi")
    parser.add_argument("--train_file", type=str, default="train.xlsx", help="training annotaion file, relative to root_path.")
    parser.add_argument("--val_file", type=str, default="val.xlsx", help="validation annotation file, relative to root_path.")
    parser.add_argument("--sample_start", type=float, default=16, help="start form which time point to sample frames.")
    parser.add_argument("--sample_end", type=float, default=144, help="End at which time point to sample frames.")
    parser.add_argument("--sample_stride", type=float, default=2, help="The time stride for sampling frames in a video.")
    parser.add_argument("--time_points", type=int, nargs="+", default=None, help="Specific time points to sample frames.")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--frame_seq_len", type=int, default=64)
    parser.add_argument("--frame_jitter", type=bool, default=False)
    parser.add_argument("--jitter_range", type=int, default=None)

    # training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_init_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--div_loss_weight", type=float, default=0.1,
                        help="Weight lambda of the diversity loss on the temporal experts.")
    parser.add_argument("--print_freq", type=int, default=25)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--arch", type=str, default="ViT-B/16")
    parser.add_argument("--output_dir", type=str, default="../experiments")
    parser.add_argument("--exp_name", type=str, required=True)

    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for DistributedDataParallel")
    parser.add_argument("--world_size", type=int, default=1, help="world size, only used when RANK/WORLD_SIZE are not set by the launcher")
    default_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pretrained_models", "clip_vit_base_patch16.ckpt")
    parser.add_argument("--pretrained_ckpt", type=str, default=default_ckpt,
                        help="CLIP ViT-B/16 checkpoint used to initialize the frozen image encoder. "
                             "If the file does not exist, the weights are downloaded from openai/clip-vit-base-patch16.")

    args = parser.parse_args()

    return args


def main(args):
    # set current device according to the local rank
    device = torch.device(f"cuda:{dist.get_rank()}")

    model = CoSTeM(args)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)

    # creating working dir based on the exp_name and current time strings
    # one example will be --exp_name resnet50.linear_probe   ----> exp/resnet50/linear_probe
    exp_time = datetime.datetime.now()
    exp_time = exp_time.strftime("%Y_%m_%d_%H_%M_%S")
    exp_sub_names = args.exp_name.split(".")
    working_dir = os.path.join(args.output_dir, *exp_sub_names, exp_time)   
    os.makedirs(working_dir, exist_ok=True)

    # create logger to record the training info
    # The training info will only be printed on the main process
    # The logging file will be saved in the same folder with checkpoints
    logger = create_logger(
        output_dir=working_dir,
        dist_rank=dist.get_rank(),
        exp_name=f"{args.exp_name}"
    )
    logger.info(f"Logger:{logger.name} has been created.")

    # In the main process, save the config into working dir
    if dist.get_rank() == 0:
        config_for_saving = vars(args)
        with open(os.path.join(working_dir, "config.yaml"), mode="w") as f:
            yaml.dump(config_for_saving, f, allow_unicode=True)

    logger.info("Model has been constructed!")

    # build the dataset and dataloader
    train_data, val_data, train_loader, val_loader = model.module.build_dataloader(logger=logger, world_size=dist.get_world_size(), global_rank=dist.get_rank())

    logger.info("Dataset and dataloader have been constructed!")
    
    # build criterion based on whether or not to use label soomthing
    if args.task == "Grading":
        loss_weights = torch.tensor([1.0, 2.25, 3.24], device=device)
    else:
        loss_weights = torch.tensor([1.0, 1.33], device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    
    # build optimizer and scheduler
    optimizer = model.module.build_optimizer(logger=logger)
    scheduler = model.module.build_scheduler(optimizer, n_iter_per_epoch=len(train_loader), logger=logger)

    # log trainable parameters
    total_params = sum(p.numel() for p in model.module.parameters()if p.requires_grad)  
    logger.info(f"Total trainable parameters: {total_params}")  

    
    # training loop
    start_epoch, max_f1 = 0, 0.0
    writer = SummaryWriter(log_dir=f"{working_dir}") if dist.get_rank() == 0 else None
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.module.train_one_epoch(epoch=epoch, model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=scheduler,
                               train_loader=train_loader, config=args, logger=logger, writer=writer)
        
        report = model.module.validate(epoch=epoch, criterion=criterion, model=model, val_loader=val_loader, config=args, logger=logger, writer=writer)
        f1_score = report["f1"]
        logger.info(f"Macro F1-score of the {args.arch} on the {len(val_data)} test videos is {f1_score:.4f}.")

        is_best = f1_score > max_f1
        max_f1 = max(max_f1, f1_score)
        logger.info(f"Max F1-score: {max_f1:.4f}")
        
        # if on the main process and meets the save frequency, save model to the disk.
        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs - 1):
            save_checkpoint(args, epoch, model, max_f1, optimizer, scheduler,
                        logger, working_dir=working_dir, is_best=is_best)

    if writer is not None:
        writer.close()



if __name__ == "__main__":
    args = parse_args()

    print(args)

    # should ensure that world_size and rank are included in the environment, and in some environment
    # use local_rank rather than rank
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

    else:
        os.environ['RANK'] = str(args.local_rank)  
        os.environ['WORLD_SIZE'] = str(args.world_size) 

    thread = 4 
    torch.set_num_threads(int(thread))

    # init process group
    torch.cuda.set_device(args.local_rank)   
    torch.distributed.init_process_group(backend='nccl', init_method="env://", world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # fix the random seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True


    main(args)

    