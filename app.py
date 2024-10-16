import os
import uuid
import gradio as gr
import torch
import PIL
from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
import threading

# Disabling parallelism to avoid deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global model cache
model_cache = {}

# Lock to ensure thread-safe access to the model cache
model_cache_lock = threading.Lock()

# Configuration
model_repo = "rain1011/pyramid-flow-sd3"  # Replace with the actual model repository on Hugging Face
model_dtype = "bf16" if torch.cuda.is_available() else "fp32"  # Support bf16 and fp32

variants = {
    'high': 'diffusion_transformer_768p',  # For high-resolution version
    'low': 'diffusion_transformer_384p'    # For low-resolution version
}
required_file = 'config.json'  # Ensure config.json is present
width_high = 1280
height_high = 768
width_low = 640
height_low = 384
cpu_offloading = torch.cuda.is_available()  # enable cpu_offloading by default

# Get the current working directory and create a folder to store the model
current_directory = os.getcwd()
model_path = os.path.join(current_directory, "pyramid_flow_model")  # Directory to store the model

# Download the model if not already present
def download_model_from_hf(model_repo, model_dir, variants, required_file):
    need_download = False
    if not os.path.exists(model_dir):
        print(f"[INFO] Model directory '{model_dir}' does not exist. Initiating download...")
        need_download = True
    else:
        # Check if all required files exist for each variant
        for variant_key, variant_dir in variants.items():
            variant_path = os.path.join(model_dir, variant_dir)
            file_path = os.path.join(variant_path, required_file)
            if not os.path.exists(file_path):
                print(f"[WARNING] Required file '{required_file}' missing in '{variant_path}'.")
                need_download = True
                break

    if need_download:
        print(f"[INFO] Downloading model from '{model_repo}' to '{model_dir}'...")
        try:
            snapshot_download(
                repo_id=model_repo,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                repo_type='model'
            )
            print("[INFO] Model download complete.")
        except Exception as e:
            print(f"[ERROR] Failed to download the model: {e}")
            raise
    else:
        print(f"[INFO] All required model files are present in '{model_dir}'. Skipping download.")

# Download model from Hugging Face if not present
download_model_from_hf(model_repo, model_path, variants, required_file)

# Function to initialize the model based on user options
def initialize_model(variant):
    print(f"[INFO] Initializing model with variant='{variant}', using bf16 precision...")

    # Determine the correct variant directory
    variant_dir = variants['high'] if variant == '768p' else variants['low']
    base_path = model_path  # Pass the base model path

    print(f"[DEBUG] Model base path: {base_path}")

    # Verify that config.json exists in the variant directory
    config_path = os.path.join(model_path, variant_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"[ERROR] config.json not found in '{os.path.join(model_path, variant_dir)}'.")
        raise FileNotFoundError(f"config.json not found in '{os.path.join(model_path, variant_dir)}'.")

    if model_dtype == "bf16":
        torch_dtype_selected = torch.bfloat16
    if model_dtype == "fp16":
        torch_dtype_selected = torch.float16
    else:
        torch_dtype_selected = torch.float32

    # Initialize the model
    try:
        model = PyramidDiTForVideoGeneration(
            base_path,                # Pass the base model path
            model_dtype=model_dtype,  # Use bf16
            model_variant=variant_dir,  # Pass the variant directory name
            cpu_offloading=cpu_offloading,  # Pass the CPU offloading flag
        )

        # Always enable tiling for the VAE
        model.vae.enable_tiling()

        # Remove manual device placement when using CPU offloading
        # The components will be moved to the appropriate devices automatically
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            # Manual device replacement when not using CPU offloading
            if not cpu_offloading:
                model.vae.to("cuda")
                model.dit.to("cuda")
                model.text_encoder.to("cuda")
        elif torch.mps.is_available():
            model.vae.to("mps")
            model.dit.to("mps")
            model.text_encoder.to("mps")
        else:
            print("[WARNING] CUDA is not available. Proceeding without GPU.")

        print("[INFO] Model initialized successfully.")
        return model, torch_dtype_selected
    except Exception as e:
        print(f"[ERROR] Error initializing model: {e}")
        raise

# Function to get the model from cache or initialize it
def initialize_model_cached(variant):
    key = variant
    
    # Check if the model is already in the cache
    if key not in model_cache:
        with model_cache_lock:
            # Double-checked locking to prevent race conditions
            if key not in model_cache:
                model, dtype = initialize_model(variant)
                model_cache[key] = (model, dtype)
    
    return model_cache[key]

def resize_crop_image(img: PIL.Image.Image, tgt_width, tgt_height):
    ori_width, ori_height = img.width, img.height
    scale = max(tgt_width / ori_width, tgt_height / ori_height)
    resized_width = round(ori_width * scale)
    resized_height = round(ori_height * scale)
    img = img.resize((resized_width, resized_height), resample=PIL.Image.LANCZOS)

    left = (resized_width - tgt_width) / 2
    top = (resized_height - tgt_height) / 2
    right = (resized_width + tgt_width) / 2
    bottom = (resized_height + tgt_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    return img

# Function to generate text-to-video
def generate_text_to_video(prompt, temp, guidance_scale, video_guidance_scale, resolution, progress=gr.Progress()):
    progress(0, desc="Loading model")
    print("[DEBUG] generate_text_to_video called.")
    variant = '768p' if resolution == "768p" else '384p'
    height = height_high if resolution == "768p" else height_low
    width = width_high if resolution == "768p" else width_low

    def progress_callback(i, m):
        progress(i/m)

    # Initialize model based on user options using cached function
    try:
        model, torch_dtype_selected = initialize_model_cached(variant)
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return f"Model initialization failed: {e}"

    try:
        print("[INFO] Starting text-to-video generation...")
        with torch.no_grad(), torch.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch_dtype_selected):
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
                cpu_offloading=cpu_offloading,
                save_memory=True,
                callback=progress_callback,
            )
        print("[INFO] Text-to-video generation completed.")
    except Exception as e:
        print(f"[ERROR] Error during text-to-video generation: {e}")
        return f"Error during video generation: {e}"

    video_path = f"{str(uuid.uuid4())}_text_to_video_sample.mp4"
    try:
        export_to_video(frames, video_path, fps=24)
        print(f"[INFO] Video exported to {video_path}.")
    except Exception as e:
        print(f"[ERROR] Error exporting video: {e}")
        return f"Error exporting video: {e}"
    return video_path

# Function to generate image-to-video
def generate_image_to_video(image, prompt, temp, video_guidance_scale, resolution, progress=gr.Progress()):
    progress(0, desc="Loading model")
    print("[DEBUG] generate_image_to_video called.")
    variant = '768p' if resolution == "768p" else '384p'
    height = height_high if resolution == "768p" else height_low
    width = width_high if resolution == "768p" else width_low

    try:
        image = resize_crop_image(image, width, height)
        print("[INFO] Image resized and cropped successfully.")
    except Exception as e:
        print(f"[ERROR] Error processing image: {e}")
        return f"Error processing image: {e}"

    def progress_callback(i, m):
        progress(i/m)

    # Initialize model based on user options using cached function
    try:
        model, torch_dtype_selected = initialize_model_cached(variant)
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return f"Model initialization failed: {e}"

    try:
        print("[INFO] Starting image-to-video generation...")
        with torch.no_grad(), torch.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch_dtype_selected):
            frames = model.generate_i2v(
                prompt=prompt,
                input_image=image,
                num_inference_steps=[10, 10, 10],
                temp=temp,
                video_guidance_scale=video_guidance_scale,
                output_type="pil",
                cpu_offloading=cpu_offloading,
                save_memory=True,
                callback=progress_callback,
            )
        print("[INFO] Image-to-video generation completed.")
    except Exception as e:
        print(f"[ERROR] Error during image-to-video generation: {e}")
        return f"Error during video generation: {e}"

    video_path = f"{str(uuid.uuid4())}_image_to_video_sample.mp4"
    try:
        export_to_video(frames, video_path, fps=24)
        print(f"[INFO] Video exported to {video_path}.")
    except Exception as e:
        print(f"[ERROR] Error exporting video: {e}")
        return f"Error exporting video: {e}"
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

    # Shared settings
    with gr.Row():
        resolution_dropdown = gr.Dropdown(
            choices=["768p", "384p"],
            value="768p",
            label="Model Resolution"
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
                ["A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors", 16, 9.0, 5.0, "768p"],
                ["Beautiful, snowy Tokyo city is bustling. The camera moves through the bustling city street, following several people enjoying the beautiful snowy weather and shopping at nearby stalls. Gorgeous sakura petals are flying through the wind along with snowflakes", 16, 9.0, 5.0, "768p"],
                ["Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. Shallow focus and light smoke. vivid colours", 31, 9.0, 5.0, "768p"],
            ],
            inputs=[text_prompt, temp_slider, guidance_scale_slider, video_guidance_scale_slider, resolution_dropdown],
            outputs=[txt_output],
            fn=generate_text_to_video,
            cache_examples='lazy',
        )

    with gr.Tab("Image-to-Video"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                image_prompt = gr.Textbox(label="Prompt (Less than 128 words)", placeholder="Enter a text prompt for the video", lines=2)
                image_temp_slider = gr.Slider(2, 16, value=16, step=1, label="Duration")
                image_video_guidance_scale_slider = gr.Slider(1.0, 7.0, value=4.0, step=0.1, label="Video Guidance Scale")
                img_generate = gr.Button("Generate Video")
            with gr.Column():
                img_output = gr.Video(label="Generated Video")
        gr.Examples(
            examples=[
                ['assets/the_great_wall.jpg', 'FPV flying over the Great Wall', 16, 4.0, "768p"]
            ],
            inputs=[image_input, image_prompt, image_temp_slider, image_video_guidance_scale_slider, resolution_dropdown],
            outputs=[img_output],
            fn=generate_image_to_video,
            cache_examples='lazy',
        )

    # Update generate functions to include resolution options
    txt_generate.click(
        generate_text_to_video,
        inputs=[text_prompt, temp_slider, guidance_scale_slider, video_guidance_scale_slider, resolution_dropdown],
        outputs=txt_output
    )

    img_generate.click(
        generate_image_to_video,
        inputs=[image_input, image_prompt, image_temp_slider, image_video_guidance_scale_slider, resolution_dropdown],
        outputs=img_output
    )

# Launch Gradio app
demo.launch(share=True)
