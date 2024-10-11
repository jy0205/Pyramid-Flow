import os
import gradio as gr
import torch
from PIL import Image
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from pyramid_dit import PyramidDiTForVideoGeneration

# Configuration
model_repo = "rain1011/pyramid-flow-sd3"  # Replace with the actual model repository on Hugging Face
model_dtype = 'bf16'
variant = 'diffusion_transformer_768p'  # For high resolution

# Get the current working directory and create a folder to store the model
current_directory = os.getcwd()
model_path = os.path.join(current_directory, "pyramid_flow_model")  # Directory to store the model

# Download the model if not already present
def download_model_from_hf(model_repo, model_dir):
    if not os.path.exists(model_dir):
        print(f"Downloading model from {model_repo} to {model_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            repo_type='model'
        )
        print("Model download complete.")
    else:
        print(f"Model directory '{model_dir}' already exists. Skipping download.")

# Download model from Hugging Face if not present
download_model_from_hf(model_repo, model_path)

# Initialize model and move to CUDA
torch.cuda.set_device(0)
model = PyramidDiTForVideoGeneration(
    model_path,
    model_dtype,
    model_variant=variant,
)
model.vae.to("cuda")
model.dit.to("cuda")
model.text_encoder.to("cuda")

# Set torch_dtype based on model_dtype
if model_dtype == "bf16":
    torch_dtype = torch.bfloat16 
elif model_dtype == "fp16":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

# Function to generate text-to-video
def generate_text_to_video(prompt, temp, guidance_scale, video_guidance_scale):
    model.vae.enable_tiling()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
        frames = model.generate(
            prompt=prompt,
            num_inference_steps=[20, 20, 20],
            video_num_inference_steps=[10, 10, 10],
            height=768,
            width=1280,
            temp=temp,
            guidance_scale=guidance_scale,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            save_memory=True,  # If you have enough GPU memory, set it to `False` to improve vae decoding speed
        )

    video_path = "./text_to_video_sample.mp4"
    export_to_video(frames, video_path, fps=24)
    return video_path

# Function to generate image-to-video
def generate_image_to_video(image, prompt, temp, guidance_scale, video_guidance_scale):
    image = image.resize((1280, 768))
    model.vae.enable_tiling()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
        frames = model.generate_i2v(
            prompt=prompt,
            input_image=image,
            num_inference_steps=[10, 10, 10],
            temp=temp,
            guidance_scale=guidance_scale,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            save_memory=True,  # If you have enough GPU memory, set it to `False` to improve vae decoding speed
        )

    video_path = "./image_to_video_sample.mp4"
    export_to_video(frames, video_path, fps=24)
    return video_path

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Text-to-Video and Image-to-Video Generation")

    with gr.Tab("Text-to-Video"):
        text_prompt = gr.Textbox(label="Prompt", placeholder="Enter a text prompt for the video", lines=2)
        temp_slider = gr.Slider(1, 31, value=16, step=1, label="Temperature")
        guidance_scale_slider = gr.Slider(1.0, 15.0, value=9.0, step=0.1, label="Guidance Scale")
        video_guidance_scale_slider = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Video Guidance Scale")
        text_video_output = gr.Video(label="Generated Video")
        
        generate_text_button = gr.Button("Generate Text-to-Video")
        generate_text_button.click(
            generate_text_to_video,
            inputs=[text_prompt, temp_slider, guidance_scale_slider, video_guidance_scale_slider],
            outputs=text_video_output,
        )

    with gr.Tab("Image-to-Video"):
        image_input = gr.Image(type="pil", label="Input Image")  # Removed `source="upload"`
        image_prompt = gr.Textbox(label="Prompt", placeholder="Enter a text prompt for the video", lines=2)
        image_temp_slider = gr.Slider(1, 31, value=16, step=1, label="Temperature")
        image_guidance_scale_slider = gr.Slider(1.0, 15.0, value=7.0, step=0.1, label="Guidance Scale")
        image_video_guidance_scale_slider = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="Video Guidance Scale")
        image_video_output = gr.Video(label="Generated Video")
        
        generate_image_button = gr.Button("Generate Image-to-Video")
        generate_image_button.click(
            generate_image_to_video,
            inputs=[image_input, image_prompt, image_temp_slider, image_guidance_scale_slider, image_video_guidance_scale_slider],
            outputs=image_video_output,
        )

# Launch Gradio app
demo.launch(share=True)

