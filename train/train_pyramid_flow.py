import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import datetime
import numpy as np
import time
import torch
import logging
import json
import math
import random
import diffusers
import transformers
from pathlib import Path
from packaging import version
from copy import deepcopy

from dataset import (
    ImageTextDataset,
    LengthGroupedVideoTextDataset,
    create_image_text_dataloaders,
    create_length_grouped_video_text_dataloader
)

from pyramid_dit import (
    PyramidDiTForVideoGeneration,
    JointTransformerBlock,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)

from trainer_misc import (
    init_distributed_mode, 
    setup_for_distributed, 
    create_optimizer,
    train_one_epoch_with_fsdp,
    constant_scheduler,
    cosine_scheduler,
)

from trainer_misc import (
    is_sequence_parallel_initialized,
    init_sequence_parallel_group,
    get_sequence_parallel_proc_num,
    init_sync_input_group,
    get_sync_input_group,
)

from collections import OrderedDict

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, 
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
    CPUOffload,
    StateDictType,
)

from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block

import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate import FullyShardedDataParallelPlugin
from diffusers.utils import is_wandb_available
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from diffusers.optimization import get_scheduler

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser('Pyramid-Flow Multi-process Training script', add_help=False)
    parser.add_argument('--task', default='t2v', type=str, choices=["t2v", "t2i"], help="Training image generation or video generation")
    parser.add_argument('--batch_size', default=4, type=int, help="The per device batch size")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--iters_per_epoch', default=2000, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--ema_update', action='store_true')
    parser.add_argument('--ema_decay', default=0.9999, type=float, metavar='MODEL', help='ema decay rate')
    parser.add_argument('--load_ema_model', default='', type=str, help='The ema model checkpoint loading')
    parser.add_argument('--model_name', default='pyramid_flux', type=str, help="The Model Architecture Name", choices=["pyramid_flux", "pyramid_mmdit"])
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained dit weight path')
    parser.add_argument('--model_variant', default='diffusion_transformer_384p', type=str, help='The dit model variant', choices=['diffusion_transformer_768p', 'diffusion_transformer_384p', 'diffusion_transformer_image'])
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or fp16", choices=['bf16', 'fp16'])
    parser.add_argument('--load_model_ema_to_cpu', action='store_true')

    # FSDP condig
    parser.add_argument('--use_fsdp', action='store_true')
    parser.add_argument('--fsdp_shard_strategy', default='zero2', type=str, choices=['zero2', 'zero3'])

    # The training manner config
    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--use_temporal_causal', action='store_true', default=True)
    parser.add_argument('--interp_condition_pos', action='store_true', default=True)
    parser.add_argument('--sync_video_input', action='store_true', help="whether to sync the video input")
    parser.add_argument('--load_text_encoder', action='store_true', help="whether to load the text encoder during training")
    parser.add_argument('--load_vae', action='store_true', help="whether to load the video vae during training")

    # Sequence Parallel config
    parser.add_argument('--use_sequence_parallel', action='store_true')
    parser.add_argument('--sp_group_size', default=1, type=int, help="The group size of sequence parallel")
    parser.add_argument('--sp_proc_num', default=-1, type=int, help="The number of process used for video training, default=-1 means using all process. This args indicated using how many processes for video training")

    # Model input config
    parser.add_argument('--max_frames', default=16, type=int, help='number of max video frames')
    parser.add_argument('--frame_per_unit', default=1, type=int, help="The number of frames per training unit")
    parser.add_argument('--schedule_shift', default=1.0, type=float, help="The flow matching schedule shift")
    parser.add_argument('--corrupt_ratio', default=1/3, type=float, help="The corruption ratio for the clean history in AR training")

    # Dataset Cconfig
    parser.add_argument('--anno_file', default='', type=str, help="The annotation jsonl file")
    parser.add_argument('--resolution', default='384p', type=str, help="The input resolution", choices=['384p', '768p'])

    # Training set config
    parser.add_argument('--dit_pretrained_weight', default='', type=str, help='The pretrained dit checkpoint')  
    parser.add_argument('--vae_pretrained_weight', default='', type=str,)
    parser.add_argument('--not_add_normalize', action='store_true')
    parser.add_argument('--use_temporal_pyramid', action='store_true', help="Whether to use the AR temporal pyramid training for video generation")
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_checkpointing_ratio', type=float, default=0.75, help="The ratio of transformer blocks used for gradient_checkpointing")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--video_sync_group', default=8, type=int, help="The number of process that accepts the same input video, used for temporal pyramid AR training. \
        This contributes to stable AR training. We recommend to set this value to 4, 8 or 16. If you have enough GPUs, set it equals to max_frames (16 for 5s, 32 for 10s), \
            make sure to satisfy `max_frames % video_sync_group == 0`")

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_beta1', default=0.9, type=float, metavar='BETA1',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--opt_beta2', default=0.999, type=float, metavar='BETA2',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--logging_dir', type=str, default='log', help='path where to tensorboard log')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    # Distributed Training parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--global_step', default=0, type=int, metavar='N', help='The global optimization step')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training', type=str)

    return parser.parse_args()


def build_model_runner(args):
    model_dtype = args.model_dtype
    model_path = args.model_path
    model_name = args.model_name
    model_variant = args.model_variant

    print(f"Load the {model_name} model checkpoint from path: {model_path}, using dtype {model_dtype}")
    sample_ratios = [1, 2, 1]  # The sample_ratios of each stage
    assert args.batch_size % int(sum(sample_ratios)) == 0, "The batchsize should be diivided by sum(sample_ratios)"

    runner = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype,
        model_name=model_name,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_ratio=args.gradient_checkpointing_ratio,
        return_log=True,
        model_variant=model_variant,
        timestep_shift=args.schedule_shift,
        stages=[1, 2, 4],      # using 3 stages
        stage_range=[0, 1/3, 2/3, 1],
        sample_ratios=sample_ratios,     # The sample proportion in a training batch
        use_mixed_training=True,
        use_flash_attn=args.use_flash_attn,
        load_text_encoder=args.load_text_encoder,
        load_vae=args.load_vae,
        max_temporal_length=args.max_frames,
        frame_per_unit=args.frame_per_unit,
        use_temporal_causal=args.use_temporal_causal,
        corrupt_ratio=args.corrupt_ratio,
        interp_condition_pos=args.interp_condition_pos,
        video_sync_group=args.video_sync_group,
    )
    
    if args.dit_pretrained_weight:
        dit_pretrained_weight = args.dit_pretrained_weight
        print(f"Loading the pre-trained DiT checkpoint from {dit_pretrained_weight}")
        runner.load_checkpoint(dit_pretrained_weight)

    if args.vae_pretrained_weight:
        vae_pretrained_weight = args.vae_pretrained_weight
        print(f"Loading the pre-trained VAE checkpoint from {vae_pretrained_weight}")
        runner.load_vae_checkpoint(vae_pretrained_weight)

    return runner


def auto_resume(args, accelerator):
    if len(args.resume) > 0:
        path = args.resume
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint does not exist. Starting a new training run."
        )
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])
        initial_global_step = global_step
    
    return initial_global_step


def build_fsdp_plugin(args):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP if args.fsdp_shard_strategy == 'zero2' else ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        auto_wrap_policy=ModuleWrapPolicy([FluxSingleTransformerBlock, FluxTransformerBlock, JointTransformerBlock, T5Block, CLIPEncoderLayer]),
        cpu_offload=CPUOffload(offload_params=False),
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
    return fsdp_plugin


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # Initialize the Environment variables throught MPI run
    init_distributed_mode(args, init_pytorch_ddp=False)   # set `init_pytorch_ddp` to False, since the accelerate will do later

    if args.use_fsdp:
        fsdp_plugin = build_fsdp_plugin(args)
    else:
        fsdp_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.model_dtype,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
    )

    # To block the print on non main process
    setup_for_distributed(accelerator.is_main_process)

    # If uses the sequence parallel 
    if args.use_sequence_parallel:
        assert args.sp_group_size > 1, "Sequence Parallel needs group size > 1"
        init_sequence_parallel_group(args)
        print(f"Using sequence parallel, the parallel size is {args.sp_group_size}")

    if args.sp_proc_num == -1:
        args.sp_proc_num = accelerator.num_processes    # if not specified, all processes are used for video training

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    device = accelerator.device

    # building model
    runner = build_model_runner(args)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if runner.vae:
        logger.info(f"Rank {args.rank}: Casting VAE to {weight_dtype}", main_process_only=False)
        runner.vae.to(dtype=weight_dtype)

    if runner.text_encoder:
        logger.info(f"Rank {args.rank}: Casting TextEncoder to {weight_dtype}", main_process_only=False)
        runner.text_encoder.to(dtype=weight_dtype)

    # building dataloader
    global_rank = accelerator.process_index
    anno_file = args.anno_file

    if args.task == 't2i':
        # For image generation training
        if args.resolution == '384p':
            image_ratios = [1/1, 3/5, 5/3]
            image_sizes = [(512, 512), (384, 640), (640, 384)]
        else:
            assert args.resolution == '768p'
            image_ratios = [1/1, 3/5, 5/3]
            image_sizes = [(1024, 1024), (768, 1280), (1280, 768)]

        image_text_dataset = ImageTextDataset(
            anno_file, 
            add_normalize=not args.not_add_normalize,
            ratios=image_ratios, 
            sizes=image_sizes,  
        )

        train_dataloader = create_image_text_dataloaders(
            image_text_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            multi_aspect_ratio=True,
            epoch=args.seed,
            sizes=image_sizes,
            use_distributed=True,
            world_size=accelerator.num_processes,
            rank=global_rank,
        )

    else:
        assert args.task == 't2v'
        # For video generation training
        video_text_dataset = LengthGroupedVideoTextDataset(
            anno_file, 
            max_frames=args.max_frames,
            resolution=args.resolution,
            load_vae_latent=not args.load_vae,
            load_text_fea=not args.load_text_encoder,
        )

        if args.sync_video_input:
            assert args.sp_proc_num % args.video_sync_group == 0, "The video_sync_group should be divided by world size"
            assert args.max_frames % args.video_sync_group == 0, "The video_sync_group should be divided by num_frames"

            train_dataloader = create_length_grouped_video_text_dataloader(
                video_text_dataset, 
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_frames=args.max_frames,
                world_size=args.sp_proc_num // args.video_sync_group,
                rank=global_rank // args.video_sync_group,
                epoch=args.seed,
                use_distributed=True,
            )
        else:
            train_dataloader = create_length_grouped_video_text_dataloader(
                video_text_dataset, 
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_frames=args.max_frames,
                world_size=args.sp_proc_num,
                rank=global_rank,
                epoch=args.seed,
                use_distributed=True,
            )

    accelerator.wait_for_everyone()
    logger.info("Building dataset finished")

    # building ema model
    model_ema = deepcopy(runner.dit) if args.ema_update else None
    if model_ema:
        model_ema.eval()

    # set the ema model not update by gradient
    if model_ema:
        model_ema.to(dtype=weight_dtype)
        for param in model_ema.parameters():
            param.requires_grad = False

    # report model details
    n_learnable_parameters = sum(p.numel() for p in runner.dit.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in runner.dit.parameters() if not p.requires_grad)
    logger.info(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    logger.info(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    # `accelerate` 0.16.0 will have better support for customized saving
    # Register Hook to load and save model_ema
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if model_ema:
                    model_ema_state = model_ema.state_dict()
                    torch.save(model_ema_state, os.path.join(output_dir, 'pytorch_model_ema.bin'))

        def load_model_hook(models, input_dir):
            if model_ema:
                model_ema_path = os.path.join(input_dir, 'pytorch_model_ema.bin')
                if os.path.exists(model_ema_path):
                    model_ema_state = torch.load(model_ema_path, map_location='cpu')
                    load_res = model_ema.load_state_dict(model_ema_state)
                    print(f"Loading ema weights {load_res}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Create the Optimizer
    optimizer = create_optimizer(args, runner.dit)
    logger.info(f"optimizer: {optimizer}")

    # Create the LR scheduler
    num_training_steps_per_epoch = args.iters_per_epoch
    args.max_train_steps = args.epochs * num_training_steps_per_epoch
    warmup_iters = args.warmup_epochs * num_training_steps_per_epoch

    if args.warmup_steps > 0:
        warmup_iters = args.warmup_steps

    logger.info(f"LRScheduler: {args.lr_scheduler}, Warmup steps: {warmup_iters * args.gradient_accumulation_steps}")

    if args.lr_scheduler == 'cosine':
        lr_schedule_values = cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        ) 
    elif args.lr_scheduler == 'constant_with_warmup':
        lr_schedule_values = constant_scheduler(
            args.lr, args.epochs, num_training_steps_per_epoch, 
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
    else:
        raise NotImplementedError(f"Not Implemented for scheduler {args.lr_scheduler}")

    # Wrap the model, optmizer, and scheduler with accelerate
    logger.info(f'before accelerator.prepare')

    if fsdp_plugin is not None:
        logger.info(f'show fsdp configs:')
        print('accelerator.state.fsdp_plugin.use_orig_params', accelerator.state.fsdp_plugin.use_orig_params)
        print('accelerator.state.fsdp_plugin.sync_module_states', accelerator.state.fsdp_plugin.sync_module_states)
        print('accelerator.state.fsdp_plugin.forward_prefetch', accelerator.state.fsdp_plugin.forward_prefetch)
        print('accelerator.state.fsdp_plugin.mixed_precision_policy', accelerator.state.fsdp_plugin.mixed_precision_policy)
        print('accelerator.state.fsdp_plugin.backward_prefetch', accelerator.state.fsdp_plugin.backward_prefetch)

    # Only wrapping the trained dit and huge text encoder
    runner.dit, optimizer = accelerator.prepare(runner.dit, optimizer)

    # Load the VAE and EMAmodel to GPU
    if runner.vae:
        runner.vae.to(device)

    if runner.text_encoder:
        runner.text_encoder.to(device)

    logger.info(f'after accelerator.prepare')
    logger.info(f'{runner.dit}')

    if model_ema and (not args.load_model_ema_to_cpu):
        model_ema.to(device)

    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.basename(args.output_dir), config=vars(args))

    # Report the training info
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info("LR = %.8f" % args.lr)
    logger.info("Min LR = %.8f" % args.min_lr)
    logger.info("Weigth Decay = %.8f" % args.weight_decay)
    logger.info("Batch size = %d" % total_batch_size)
    logger.info("Number of training steps = %d" % (num_training_steps_per_epoch * args.epochs))
    logger.info("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    # Auto resume the checkpoint
    initial_global_step = auto_resume(args, accelerator)
    first_epoch = initial_global_step // num_training_steps_per_epoch

    # Start Train!
    start_time = time.time()
    accelerator.wait_for_everyone()

    for epoch in range(first_epoch, args.epochs):
        train_stats = train_one_epoch_with_fsdp(
            runner, 
            model_ema,
            accelerator,
            args.model_dtype,
            train_dataloader,
            optimizer,
            lr_schedule_values,
            device, 
            epoch, 
            args.clip_grad,
            start_steps=epoch * num_training_steps_per_epoch,
            args=args,
            print_freq=args.print_freq,
            iters_per_epoch=num_training_steps_per_epoch,
            ema_decay=args.ema_decay,
            use_temporal_pyramid=args.use_temporal_pyramid,
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                if accelerator.sync_gradients:
                    global_step = num_training_steps_per_epoch * (epoch + 1)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path, safe_serialization=False)
                    logger.info(f"Saved state to {save_path}")

            accelerator.wait_for_everyone()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch, 'n_parameters': n_learnable_parameters}

        if args.output_dir and accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
