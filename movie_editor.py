import gradio as gr
from huggingface_hub import snapshot_download
import torch
from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import load_image, export_to_video
import tempfile
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Download and set up the model
model_path = os.path.expanduser('~/.cache/huggingface/hub/pyramid-flow')
snapshot_download("rain1011/pyramid-flow-sd3", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')

torch.cuda.set_device(0)
model_dtype, torch_dtype = 'bf16', torch.bfloat16

model = PyramidDiTForVideoGeneration(
    model_path,
    model_dtype,
    model_variant='diffusion_transformer_384p',
    save_memory=True,
)

model.vae.to("cuda")
model.dit.to("cuda")
model.text_encoder.to("cuda")
model.vae.enable_tiling()

# Function to generate video from text
def generate_video_from_text(prompt, num_inference_steps, guidance_scale, temp, resolution):
    width, height = resolution.split('x')
    width, height = int(width), int(height)
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        frames = model.generate(
            prompt=prompt,
            num_inference_steps=[num_inference_steps] * 3,
            video_num_inference_steps=[num_inference_steps // 2] * 3,
            height=height,
            width=width,
            temp=temp,
            guidance_scale=guidance_scale,
            video_guidance_scale=guidance_scale,
            output_type="pil",
            save_memory=True,
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        export_to_video(frames, tmp_file.name, fps=24)
        return tmp_file.name

# Function to generate video from image and text
def generate_video_from_image_and_text(image, prompt, num_inference_steps, guidance_scale, temp, resolution):
    width, height = resolution.split('x')
    width, height = int(width), int(height)
    image = image.convert("RGB").resize((width, height))
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        frames = model.generate_i2v(
            prompt=prompt,
            input_image=image,
            num_inference_steps=[num_inference_steps] * 3,
            temp=temp,
            video_guidance_scale=guidance_scale,
            output_type="pil",
            save_memory=True,
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        export_to_video(frames, tmp_file.name, fps=24)
        return tmp_file.name

# Function to extract the last frame from a video
def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Unable to open video file")
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set the frame position to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    # Read the last frame
    ret, frame = cap.read()
    
    if not ret:
        raise ValueError("Unable to read the last frame")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    last_frame = Image.fromarray(frame_rgb)
    
    cap.release()
    return last_frame

# Function to extend video
def extend_video(input_video, prompt, num_inference_steps, guidance_scale, temp, resolution):
    # Extract the last frame from the input video
    last_frame = extract_last_frame(input_video)
    
    # Generate new video segment from the last frame
    new_segment = generate_video_from_image_and_text(last_frame, prompt, num_inference_steps, guidance_scale, temp, resolution)
    
    # Concatenate the original video and the new segment
    original_clip = VideoFileClip(input_video)
    new_segment_clip = VideoFileClip(new_segment)
    final_clip = concatenate_videoclips([original_clip, new_segment_clip])
    
    # Save the final video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        final_clip.write_videofile(tmp_file.name, codec="libx264")
        return tmp_file.name

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Video Generation with Pyramid-DiT")
    
    with gr.Tab("Text to Video"):
        text_input = gr.Textbox(label="Enter your prompt")
        with gr.Row():
            text_num_inference_steps = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Num Inference Steps")
            text_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=9.0, step=0.1, label="Guidance Scale")
            text_temp = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Length 16 is about 5 seconds")
        text_resolution = gr.Radio(["640x384", "1280x768"], label="Resolution", value="640x384")
        text_button = gr.Button("Generate Video")
        text_output = gr.Video(label="Generated Video")
        
        text_button.click(
            generate_video_from_text, 
            inputs=[text_input, text_num_inference_steps, text_guidance_scale, text_temp, text_resolution],
            outputs=text_output
        )
    
    with gr.Tab("Image to Video"):
        image_input = gr.Image(type="pil", label="Upload an image")
        image_text_input = gr.Textbox(label="Enter your prompt")
        with gr.Row():
            image_num_inference_steps = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Num Inference Steps")
            image_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=9.0, step=0.1, label="Guidance Scale")
            image_temp = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Length 16 is about 5 seconds")
        image_resolution = gr.Radio(["640x384", "1280x768"], label="Resolution", value="640x384")
        image_button = gr.Button("Generate Video")
        image_output = gr.Video(label="Generated Video")
        
        image_button.click(
            generate_video_from_image_and_text, 
            inputs=[image_input, image_text_input, image_num_inference_steps, image_guidance_scale, image_temp, image_resolution],
            outputs=image_output
        )
    
    with gr.Tab("Extend Video"):
        extend_video_input = gr.Video(label="Upload a video to extend")
        extend_text_input = gr.Textbox(label="Enter your prompt for the extension")
        with gr.Row():
            extend_num_inference_steps = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Num Inference Steps")
            extend_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=9.0, step=0.1, label="Guidance Scale")
            extend_temp = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Length 16 is about 5 seconds")
        extend_resolution = gr.Radio(["640x384", "1280x768"], label="Resolution", value="640x384")
        extend_button = gr.Button("Extend Video")
        extend_output = gr.Video(label="Extended Video")
        
        extend_button.click(
            extend_video,
            inputs=[extend_video_input, extend_text_input, extend_num_inference_steps, extend_guidance_scale, extend_temp, extend_resolution],
            outputs=extend_output
        )

demo.launch()
