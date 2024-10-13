import os
import torch
import sys
import argparse
from diffusers.utils import export_to_video
from pyramid_dit import PyramidDiTForVideoGeneration
from trainer_misc import init_distributed_mode, init_sequence_parallel_group
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process Script', add_help=False)
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16")
    parser.add_argument('--model_path', required=True, type=str, help='Path to the downloaded checkpoint directory')
    parser.add_argument('--variant', default='diffusion_transformer_768p', type=str,)
    parser.add_argument('--task', default='t2v', type=str, choices=['i2v', 't2v'])
    parser.add_argument('--temp', default=16, type=int, help='The generated latent num, num_frames = temp * 8 + 1')
    parser.add_argument('--sp_group_size', default=2, type=int, help="The number of GPUs used for inference, should be 2 or 4")
    parser.add_argument('--sp_proc_num', default=-1, type=int, help="The number of processes used for video training, default=-1 means using all processes.")
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt for video generation")
    parser.add_argument('--image_path', type=str, help="Path to the input image for image-to-video")
    parser.add_argument('--video_guidance_scale', type=float, default=5.0, help="Video guidance scale")
    parser.add_argument('--guidance_scale', type=float, default=9.0, help="Guidance scale for text-to-video")
    parser.add_argument('--resolution', type=str, default='768p', choices=['768p', '384p'], help="Model resolution")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated video")
    return parser.parse_args()

def main():
    args = get_args()

    # setup DDP
    init_distributed_mode(args)

    assert args.world_size == args.sp_group_size, "The sequence parallel size should match DDP world size"

    # Enable sequence parallel
    init_sequence_parallel_group(args)

    device = torch.device('cuda')
    rank = args.rank
    model_dtype = args.model_dtype

    model = PyramidDiTForVideoGeneration(
        args.model_path,
        model_dtype,
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
    if args.resolution == '768p':
        width = 1280
        height = 768
    else:
        width = 640
        height = 384

    try:
        if args.task == 't2v':
            prompt = args.prompt
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
                frames = model.generate(
                    prompt=prompt,
                    num_inference_steps=[20, 20, 20],
                    video_num_inference_steps=[10, 10, 10],
                    height=height,
                    width=width,
                    temp=args.temp,
                    guidance_scale=args.guidance_scale,
                    video_guidance_scale=args.video_guidance_scale,
                    output_type="pil",
                    save_memory=True,
                    cpu_offloading=False,
                    inference_multigpu=True,
                )
            if rank == 0:
                export_to_video(frames, args.output_path, fps=24)

        elif args.task == 'i2v':
            if not args.image_path:
                raise ValueError("Image path is required for image-to-video task")
            image = Image.open(args.image_path).convert("RGB")
            image = image.resize((width, height))

            prompt = args.prompt

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
                frames = model.generate_i2v(
                    prompt=prompt,
                    input_image=image,
                    num_inference_steps=[10, 10, 10],
                    temp=args.temp,
                    video_guidance_scale=args.video_guidance_scale,
                    output_type="pil",
                    save_memory=True,
                    cpu_offloading=False,
                    inference_multigpu=True,
                )
            if rank == 0:
                export_to_video(frames, args.output_path, fps=24)

    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Error during video generation: {e}")
        raise
    finally:
        torch.distributed.barrier()

if __name__ == "__main__":
    main()

