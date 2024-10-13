import os
import uuid
import gradio as gr
import subprocess
import tempfile
import shutil

def run_inference_multigpu(gpus, variant, model_path, temp, guidance_scale, video_guidance_scale, resolution, prompt):
    """
    Runs the external multi-GPU inference script and returns the path to the generated video.
    """
    # Create a temporary directory to store inputs and outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_video = os.path.join(tmpdir, f"{uuid.uuid4()}_output.mp4")
        
        # Path to the external shell script
        script_path = "./scripts/app_multigpu_engine.sh"  # Updated script path

        # Prepare the command
        cmd = [
            script_path,
            str(gpus),
            variant,
            model_path,
            't2v',  # Task is always 't2v' since 'i2v' is removed
            str(temp),
            str(guidance_scale),
            str(video_guidance_scale),
            resolution,
            output_video,
            prompt  # Pass the prompt directly as an argument
        ]

        try:
            # Run the external script
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error during video generation: {e}")

        # After generation, move the video to a permanent location
        final_output = os.path.join("generated_videos", f"{uuid.uuid4()}_output.mp4")
        os.makedirs("generated_videos", exist_ok=True)
        shutil.move(output_video, final_output)

        return final_output

def generate_text_to_video(prompt, temp, guidance_scale, video_guidance_scale, resolution, gpus):
    model_path = "./pyramid_flow_model"  # Use the model path as specified
    # Determine variant based on resolution
    if resolution == "768p":
        variant = "diffusion_transformer_768p"
    else:
        variant = "diffusion_transformer_384p"
    return run_inference_multigpu(gpus, variant, model_path, temp, guidance_scale, video_guidance_scale, resolution, prompt)

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
        gpus_dropdown = gr.Dropdown(
            choices=[2, 4],
            value=4,
            label="Number of GPUs"
        )
        resolution_dropdown = gr.Dropdown(
            choices=["768p", "384p"],
            value="768p",
            label="Model Resolution"
        )

    with gr.Tab("Text-to-Video"):
        with gr.Row():
            with gr.Column():
                text_prompt = gr.Textbox(
                    label="Prompt (Less than 128 words)",
                    placeholder="Enter a text prompt for the video",
                    lines=2
                )
                temp_slider = gr.Slider(1, 31, value=16, step=1, label="Duration")
                guidance_scale_slider = gr.Slider(1.0, 15.0, value=9.0, step=0.1, label="Guidance Scale")
                video_guidance_scale_slider = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Video Guidance Scale")
                txt_generate = gr.Button("Generate Video")
            with gr.Column():
                txt_output = gr.Video(label="Generated Video")
        gr.Examples(
            examples=[
                [
                    "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors",
                    16,
                    9.0,
                    5.0,
                    "768p",
                    4
                ],
                [
                    "Beautiful, snowy Tokyo city is bustling. The camera moves through the bustling city street, following several people enjoying the beautiful snowy weather and shopping at nearby stalls. Gorgeous sakura petals are flying through the wind along with snowflakes",
                    16,
                    9.0,
                    5.0,
                    "768p",
                    4
                ],
                [
                    "Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. Shallow focus and light smoke. vivid colours",
                    31,
                    9.0,
                    5.0,
                    "768p",
                    4
                ],
            ],
            inputs=[text_prompt, temp_slider, guidance_scale_slider, video_guidance_scale_slider, resolution_dropdown, gpus_dropdown],
            outputs=[txt_output],
            fn=generate_text_to_video,
            cache_examples='lazy',
        )

    # Update generate function for Text-to-Video
    txt_generate.click(
        generate_text_to_video,
        inputs=[
            text_prompt,
            temp_slider,
            guidance_scale_slider,
            video_guidance_scale_slider,
            resolution_dropdown,
            gpus_dropdown
        ],
        outputs=txt_output
    )

# Launch Gradio app
demo.launch(share=True)
