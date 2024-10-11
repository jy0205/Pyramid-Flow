import os
import uuid
import gradio as gr
import torch
import PIL
from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download


# Configuration
model_repo = "rain1011/pyramid-flow-sd3"  # Replace with the actual model repository on Hugging Face
model_dtype = 'bf16'
variant = 'diffusion_transformer_768p'  # For high resolution version
width = 1280
height = 768

# variant = 'diffusion_transformer_384p'  # For low resolution version
# width = 640
# height = 384

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
model.vae.enable_tiling()

# Set torch_dtype based on model_dtype
if model_dtype == "bf16":
    torch_dtype = torch.bfloat16 
elif model_dtype == "fp16":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32


def resize_crop_image(img: PIL.Image, tgt_width, tgt_height):
    ori_width, ori_height = img.width, img.height
    scale = max(tgt_width / ori_width, tgt_height / ori_height)
    resized_width = round(ori_width * scale)
    resized_height = round(ori_height * scale)
    img = img.resize((resized_width, resized_height))

    left = (resized_width - tgt_width) / 2
    top = (resized_height - tgt_height) / 2
    right = (resized_width + tgt_width) / 2
    bottom = (resized_height + tgt_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    return img


# Function to generate text-to-video
def generate_text_to_video(prompt, temp, guidance_scale, video_guidance_scale):

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
        frames = model.generate(
            prompt=prompt,
            num_inference_steps=[20, 20, 20],
            video_num_inference_steps=[10, 10, 10],
            height=height,
            width=width,
            temp=temp,
            guidance_scale=guidance_scale,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            save_memory=True,  # If you have enough GPU memory, set it to `False` to improve vae decoding speed
            cpu_offloading=False, # If you do not have enough GPU memory, set it to `True` to reduce memory usage (will increase inference time)
        )

    video_path = f"{str(uuid.uuid4())}_text_to_video_sample.mp4"
    export_to_video(frames, video_path, fps=24)
    return video_path


# Function to generate image-to-video
def generate_image_to_video(image, prompt, temp, video_guidance_scale):

    image = resize_crop_image(image, width, height)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
        frames = model.generate_i2v(
            prompt=prompt,
            input_image=image,
            num_inference_steps=[10, 10, 10],
            temp=temp,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            save_memory=True,  # If you have enough GPU memory, set it to `False` to improve vae decoding speed
            cpu_offloading=False, # If you do not have enough GPU memory, set it to `True` to reduce memory usage (will increase inference time)
        )

    video_path = f"{str(uuid.uuid4())}_image_to_video_sample.mp4"
    export_to_video(frames, video_path, fps=24)
    return video_path


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
# Pyramid Flow Video Generation Demo

Pyramid Flow is a training-efficient **Autoregressive Video Generation** model based on **Flow Matching**. It is trained only on open-source datasets within 20.7k A100 GPU hours.

[[Paper]](https://arxiv.org/abs/2410.05954) [[Project Page]](https://pyramid-flow.github.io) [[Code]](https://github.com/jy0205/Pyramid-Flow) [[Model]](https://huggingface.co/rain1011/pyramid-flow-sd3)
"""
    )

    with gr.Tab("Text-to-Video"):
        with gr.Row():
            with gr.Column():
                text_prompt = gr.Textbox(label="Prompt (Less than 128 words)", placeholder="Enter a text prompt for the video", lines=2)
                temp_slider = gr.Slider(1, 31, value=16, step=1, label="Duration")
                guidance_scale_slider = gr.Slider(1.0, 15.0, value=9.0, step=0.1, label="Guidance Scale")
                video_guidance_scale_slider = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Video Guidance Scale")
                txt_generate = gr.Button("Generate Video")
            with gr.Column():
                txt_output = gr.Video(label="Generated Video")
        gr.Examples(
            examples=[
                ["A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors", 16, 9, 5],
                ["Beautiful, snowy Tokyo city is bustling. The camera moves through the bustling city street, following several people enjoying the beautiful snowy weather and shopping at nearby stalls. Gorgeous sakura petals are flying through the wind along with snowflakes", 16, 9, 5],
                ["Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. Shallow focus and light smoke. vivid colours", 31, 9, 5],
            ],
            inputs=[text_prompt, temp_slider, guidance_scale_slider, video_guidance_scale_slider],
            outputs=[txt_output],
            fn=generate_text_to_video,
            cache_examples='lazy',
        )

    with gr.Tab("Image-to-Video"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")  # Removed `source="upload"`
                image_prompt = gr.Textbox(label="Prompt (Less than 128 words)", placeholder="Enter a text prompt for the video", lines=2)
                image_temp_slider = gr.Slider(2, 16, value=16, step=1, label="Duration")
                image_video_guidance_scale_slider = gr.Slider(1.0, 7.0, value=4.0, step=0.1, label="Video Guidance Scale")
                img_generate = gr.Button("Generate Video")
            with gr.Column():
                img_output = gr.Video(label="Generated Video")
        gr.Examples(
            examples=[
                ['assets/the_great_wall.jpg', 'FPV flying over the Great Wall', 16, 4]
            ],
            inputs=[image_input, image_prompt, image_temp_slider, image_video_guidance_scale_slider],
            outputs=[img_output],
            fn=generate_image_to_video,
            cache_examples='lazy',
        )

    txt_generate.click(generate_text_to_video,
                       inputs=[text_prompt, temp_slider, guidance_scale_slider, video_guidance_scale_slider],
                       outputs=txt_output)

    img_generate.click(generate_image_to_video,
                       inputs=[image_input, image_prompt, image_temp_slider, image_video_guidance_scale_slider],
                       outputs=img_output)


# Launch Gradio app
demo.launch(share=True)