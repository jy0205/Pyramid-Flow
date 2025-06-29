import os
import argparse
import gradio as gr
import torch
from PIL import Image
from diffusers.utils import export_to_video
from pyramid_dit import PyramidDiTForVideoGeneration

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Web Interface for Pyramid Flow")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_name", type=str, default="pyramid_flux", choices=["pyramid_flux", "pyramid_mmdit"], 
                        help="Model architecture to use")
    parser.add_argument("--model_dtype", type=str, default="bf16", choices=["bf16", "fp32"], 
                        help="Model data type")
    parser.add_argument("--share", action="store_true", help="Share the Gradio interface")
    return parser.parse_args()

def resize_crop_image(img, tgt_width, tgt_height):
    """Resize and crop the image to the target size while maintaining aspect ratio"""
    ori_width, ori_height = img.width, img.height
    scale = max(tgt_width / ori_width, tgt_height / ori_height)
    resized_width = round(ori_width * scale)
    resized_height = round(ori_height * scale)
    img = img.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - tgt_width) / 2
    top = (resized_height - tgt_height) / 2
    right = (resized_width + tgt_width) / 2
    bottom = (resized_height + tgt_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    return img

def initialize_model(model_path, model_name, model_dtype, variant):
    """Initialize the Pyramid Flow model"""
    print(f"Initializing {model_name} model from {model_path} with {model_dtype} precision...")
    
    model = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype,
        model_name=model_name,
        model_variant=variant,
    )
    
    # Enable CPU offloading to reduce memory usage
    model.enable_sequential_cpu_offload()
    model.vae.enable_tiling()
    
    print("Model initialized successfully!")
    return model

def generate_text_to_video(
    prompt, 
    negative_prompt,
    resolution, 
    duration, 
    guidance_scale, 
    video_guidance_scale,
    seed,
    model,
    progress=gr.Progress()
):
    """Generate video from text prompt"""
    progress(0, desc="Preparing")
    
    # Set resolution based on selection
    if resolution == "768p":
        width, height = 1280, 768
        variant = "diffusion_transformer_768p"
    else:
        width, height = 640, 384
        variant = "diffusion_transformer_384p"
    
    # Set random seed if provided
    if seed is not None and seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    # Determine torch dtype
    if model.model_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    progress(0.1, desc="Generating video")
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        frames = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=[20, 20, 20],
            video_num_inference_steps=[10, 10, 10],
            height=height,
            width=width,
            temp=duration,
            guidance_scale=guidance_scale,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            save_memory=True,
            callback=lambda i, t: progress((i + 1) / t),
        )
    
    # Save video to temporary file
    output_path = "output_video.mp4"
    export_to_video(frames, output_path, fps=24)
    
    return output_path

def generate_image_to_video(
    image, 
    prompt, 
    resolution, 
    duration, 
    video_guidance_scale,
    seed,
    model,
    progress=gr.Progress()
):
    """Generate video from input image and text prompt"""
    progress(0, desc="Preparing")
    
    # Set resolution based on selection
    if resolution == "768p":
        width, height = 1280, 768
        variant = "diffusion_transformer_768p"
    else:
        width, height = 640, 384
        variant = "diffusion_transformer_384p"
    
    # Resize and crop image to match target resolution
    image = resize_crop_image(image, width, height)
    
    # Set random seed if provided
    if seed is not None and seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    # Determine torch dtype
    if model.model_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    progress(0.1, desc="Generating video")
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        frames = model.generate_i2v(
            prompt=prompt,
            input_image=image,
            num_inference_steps=[10, 10, 10],
            temp=duration,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            save_memory=True,
            callback=lambda i, t: progress((i + 1) / t),
        )
    
    # Save video to temporary file
    output_path = "output_video.mp4"
    export_to_video(frames, output_path, fps=24)
    
    return output_path

def main():
    args = parse_args()
    
    # Initialize model
    model_384p = None
    model_768p = None
    
    def get_model(resolution):
        nonlocal model_384p, model_768p
        
        if resolution == "768p":
            if model_768p is None:
                model_768p = initialize_model(
                    args.model_path, 
                    args.model_name, 
                    args.model_dtype,
                    "diffusion_transformer_768p"
                )
            return model_768p
        else:
            if model_384p is None:
                model_384p = initialize_model(
                    args.model_path, 
                    args.model_name, 
                    args.model_dtype,
                    "diffusion_transformer_384p"
                )
            return model_384p
    
    # Create Gradio interface
    with gr.Blocks(title="Pyramid Flow Video Generation") as demo:
        gr.Markdown(
            """
            # 🎬 Pyramid Flow Video Generation
            
            Generate high-quality videos from text prompts or images using Pyramid Flow.
            
            - For text-to-video: Enter a detailed prompt describing the video you want to generate
            - For image-to-video: Upload an image and provide a prompt for the motion
            
            [GitHub Repository](https://github.com/jy0205/Pyramid-Flow) | [Paper](https://arxiv.org/abs/2410.05954)
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("Text to Video"):
                with gr.Row():
                    with gr.Column():
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors",
                            lines=3
                        )
                        t2v_negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="blurry, low quality, distorted, bad anatomy",
                            lines=2
                        )
                        t2v_resolution = gr.Dropdown(
                            choices=["384p", "768p"],
                            value="384p",
                            label="Resolution"
                        )
                        t2v_duration = gr.Slider(
                            minimum=4, 
                            maximum=16, 
                            value=8, 
                            step=1, 
                            label="Duration (frames)"
                        )
                        t2v_guidance_scale = gr.Slider(
                            minimum=1.0, 
                            maximum=15.0, 
                            value=7.0, 
                            step=0.1, 
                            label="Guidance Scale"
                        )
                        t2v_video_guidance_scale = gr.Slider(
                            minimum=1.0, 
                            maximum=10.0, 
                            value=5.0, 
                            step=0.1, 
                            label="Video Guidance Scale"
                        )
                        t2v_seed = gr.Number(
                            label="Seed (0 for random)", 
                            value=0
                        )
                        t2v_generate = gr.Button("Generate Video")
                    
                    with gr.Column():
                        t2v_output = gr.Video(label="Generated Video")
            
            with gr.TabItem("Image to Video"):
                with gr.Row():
                    with gr.Column():
                        i2v_image = gr.Image(
                            label="Input Image", 
                            type="pil"
                        )
                        i2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="FPV flying over the landscape",
                            lines=2
                        )
                        i2v_resolution = gr.Dropdown(
                            choices=["384p", "768p"],
                            value="384p",
                            label="Resolution"
                        )
                        i2v_duration = gr.Slider(
                            minimum=4, 
                            maximum=16, 
                            value=8, 
                            step=1, 
                            label="Duration (frames)"
                        )
                        i2v_video_guidance_scale = gr.Slider(
                            minimum=1.0, 
                            maximum=7.0, 
                            value=4.0, 
                            step=0.1, 
                            label="Video Guidance Scale"
                        )
                        i2v_seed = gr.Number(
                            label="Seed (0 for random)", 
                            value=0
                        )
                        i2v_generate = gr.Button("Generate Video")
                    
                    with gr.Column():
                        i2v_output = gr.Video(label="Generated Video")
        
        # Set up event handlers
        t2v_generate.click(
            fn=generate_text_to_video,
            inputs=[
                t2v_prompt,
                t2v_negative_prompt,
                t2v_resolution,
                t2v_duration,
                t2v_guidance_scale,
                t2v_video_guidance_scale,
                t2v_seed,
                t2v_resolution
            ],
            outputs=t2v_output,
            preprocess=lambda *args: [*args[:-1], get_model(args[-1])]
        )
        
        i2v_generate.click(
            fn=generate_image_to_video,
            inputs=[
                i2v_image,
                i2v_prompt,
                i2v_resolution,
                i2v_duration,
                i2v_video_guidance_scale,
                i2v_seed,
                i2v_resolution
            ],
            outputs=i2v_output,
            preprocess=lambda *args: [*args[:-1], get_model(args[-1])]
        )
        
        # Add examples
        gr.Examples(
            examples=[
                [
                    "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors",
                    "",
                    "384p",
                    8,
                    7.0,
                    5.0,
                    42
                ],
                [
                    "Beautiful, snowy Tokyo city is bustling. The camera moves through the bustling city street, following several people enjoying the beautiful snowy weather and shopping at nearby stalls.",
                    "",
                    "384p",
                    8,
                    7.0,
                    5.0,
                    0
                ],
            ],
            inputs=[t2v_prompt, t2v_negative_prompt, t2v_resolution, t2v_duration, t2v_guidance_scale, t2v_video_guidance_scale, t2v_seed],
            outputs=[t2v_output],
            fn=generate_text_to_video,
            preprocess=lambda *args: [*args[:-1], get_model(args[2])]
        )
    
    # Launch the interface
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()