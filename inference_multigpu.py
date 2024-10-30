import os
import torch
import sys
import argparse
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.utils import export_to_video
from pyramid_dit import PyramidDiTForVideoGeneration
from trainer_misc import init_distributed_mode, init_sequence_parallel_group
import PIL
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process Script', add_help=False)
    parser.add_argument('--model_name', default='pyramid_mmdit', type=str, help="The model name", choices=["pyramid_flux", "pyramid_mmdit"])
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16")
    parser.add_argument('--model_path', default='/home/jinyang06/models/pyramid-flow', type=str, help='Set it to the downloaded checkpoint dir')
    parser.add_argument('--variant', default='diffusion_transformer_768p', type=str,)
    parser.add_argument('--task', default='t2v', type=str, choices=['i2v', 't2v'])
    parser.add_argument('--temp', default=16, type=int, help='The generated latent num, num_frames = temp * 8 + 1')
    parser.add_argument('--sp_group_size', default=2, type=int, help="The number of gpus used for inference, should be 2 or 4")
    parser.add_argument('--sp_proc_num', default=-1, type=int, help="The number of process used for video training, default=-1 means using all process.")

    return parser.parse_args()


def main():
    args = get_args()

    # setup DDP
    init_distributed_mode(args)

    assert args.world_size == args.sp_group_size, "The sequence parallel size should be DDP world size"

    # Enable sequence parallel
    init_sequence_parallel_group(args)

    device = torch.device('cuda')
    rank = args.rank
    model_dtype = args.model_dtype

    if args.model_name == "pyramid_flux":
        assert args.variant != "diffusion_transformer_768p", "The pyramid_flux does not support high resolution now, \
            we will release it after finishing training. You can modify the model_name to pyramid_mmdit to support 768p version generation"
    
    model = PyramidDiTForVideoGeneration(
        args.model_path,
        model_dtype,
        model_name=args.model_name,
        model_variant=args.variant,
    )

    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device)
    model.vae.enable_tiling()

    if model_dtype == "bf16":
        torch_dtype = torch.bfloat16 
    elif model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # The video generation config
    if args.variant == 'diffusion_transformer_768p':
        width = 1280
        height = 768
    else:
        assert args.variant == 'diffusion_transformer_384p'
        width = 640
        height = 384

    if args.task == 't2v':
        prompt = "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors"

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
            frames = model.generate(
                prompt=prompt,
                num_inference_steps=[20, 20, 20],
                video_num_inference_steps=[10, 10, 10],
                height=height,
                width=width,
                temp=args.temp,
                guidance_scale=7.0,         # The guidance for the first frame, set it to 7 for 384p variant
                video_guidance_scale=5.0,   # The guidance for the other video latent
                output_type="pil",
                save_memory=True,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
                cpu_offloading=False,       # If OOM, set it to True to reduce memory usage
                inference_multigpu=True,
            )
        if rank == 0:
            export_to_video(frames, "./text_to_video_sample.mp4", fps=24)

    else:
        assert args.task == 'i2v'

        image_path = 'assets/the_great_wall.jpg'
        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height))

        prompt = "FPV flying over the Great Wall"

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
            frames = model.generate_i2v(
                prompt=prompt,
                input_image=image,
                num_inference_steps=[10, 10, 10],
                temp=args.temp,
                video_guidance_scale=4.0,
                output_type="pil",
                save_memory=True,         # If you have enough GPU memory, set it to `False` to improve vae decoding speed
                cpu_offloading=False,       # If OOM, set it to True to reduce memory usage
                inference_multigpu=True,
            )

        if rank == 0:
            export_to_video(frames, "./image_to_video_sample.mp4", fps=24)

    torch.distributed.barrier()


if __name__ == "__main__":
    main()