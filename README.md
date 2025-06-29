# Pyramid Flow Extensions

This repository contains extensions and utilities for the [Pyramid Flow](https://github.com/jy0205/Pyramid-Flow) text-to-video generation model.

## Features

### 1. Simple Web Interface
A user-friendly web interface for generating videos from text prompts or images without writing code.

```bash
python simple_interface.py --model_path /path/to/model --share
```

### 2. Batch Video Processing
Process multiple video generation requests efficiently, with support for parallel processing.

```bash
python batch_processor.py --model_path /path/to/model --create_sample
python batch_processor.py --model_path /path/to/model --batch_file sample_batch.json
```

### 3. Video Enhancement
Post-processing tools to improve the quality of generated videos.

```bash
python video_enhancer.py --input_video generated_video.mp4 --output_video enhanced_video.mp4 --color_grading cinematic --interpolate
```

### 4. Prompt Optimization
Tools to analyze and optimize prompts for better video generation results.

```bash
python prompt_optimizer.py --prompt "A mountain landscape" --analyze
python prompt_optimizer.py --prompt "A mountain landscape" --variations 3
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pyramid-flow-extensions
cd pyramid-flow-extensions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Pyramid Flow model:
```python
from huggingface_hub import snapshot_download
model_path = 'PATH'   # The local directory to save downloaded checkpoint
snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
```

## Usage Examples

### Simple Web Interface

```bash
python simple_interface.py --model_path /path/to/model
```

This launches a Gradio web interface with tabs for text-to-video and image-to-video generation.

### Batch Processing

Create a sample batch file:
```bash
python batch_processor.py --model_path /path/to/model --create_sample
```

Process a batch of requests:
```bash
python batch_processor.py --model_path /path/to/model --batch_file sample_batch.json --workers 1
```

### Video Enhancement

```bash
python video_enhancer.py --input_video generated_video.mp4 --enhance_quality --temporal_smoothing --color_grading cinematic
```

### Prompt Optimization

Analyze a prompt:
```bash
python prompt_optimizer.py --prompt "A mountain landscape" --analyze
```

Generate optimized variations:
```bash
python prompt_optimizer.py --prompt "A mountain landscape" --variations 3
```

## Memory Optimization

All tools support memory optimization techniques:

1. VAE tiling is enabled by default
2. CPU offloading is used to reduce GPU memory usage
3. Batch processing allows efficient use of resources

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU with at least 8GB VRAM
- See requirements.txt for full dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.