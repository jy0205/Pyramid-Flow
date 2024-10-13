#!/bin/bash

# Usage:
# ./scripts/app_multigpu_engine.sh GPUS VARIANT MODEL_PATH TASK TEMP GUIDANCE_SCALE VIDEO_GUIDANCE_SCALE RESOLUTION OUTPUT_PATH [IMAGE_PATH] PROMPT

GPUS=$1
VARIANT=$2
MODEL_PATH=$3
TASK=$4
TEMP=$5
GUIDANCE_SCALE=$6
VIDEO_GUIDANCE_SCALE=$7
RESOLUTION=$8
OUTPUT_PATH=$9
shift 9
# Now the remaining arguments are $@

if [ "$TASK" == "t2v" ]; then
    PROMPT="$1"
    IMAGE_ARG=""
elif [ "$TASK" == "i2v" ]; then
    IMAGE_PATH="$1"
    PROMPT="$2"
    IMAGE_ARG="--image_path $IMAGE_PATH"
else
    echo "Invalid task: $TASK"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent directory of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to include the project root directory
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Adjust the path to app_multigpu_engine.py
PYTHON_SCRIPT="$SCRIPT_DIR/app_multigpu_engine.py"

torchrun --nproc_per_node="$GPUS" \
    "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --variant "$VARIANT" \
    --task "$TASK" \
    --model_dtype bf16 \
    --temp "$TEMP" \
    --sp_group_size "$GPUS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --video_guidance_scale "$VIDEO_GUIDANCE_SCALE" \
    --resolution "$RESOLUTION" \
    --output_path "$OUTPUT_PATH" \
    --prompt "$PROMPT" \
    $IMAGE_ARG

