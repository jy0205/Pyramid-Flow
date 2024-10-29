#!/bin/bash

# This script is used for batch extract the text features for video generation training
# Since the T5 will cost a lot of GPU memory, pre-extract the text features will save the training memory

GPUS=8  # The gpu number
MODEL_NAME=pyramid_flux     # The model name, `pyramid_flux` or `pyramid_mmdit`
MODEL_PATH=/PATH/pyramid-flow-miniflux  # The downloaded ckpt dir. IMPORTANT: It should match with model_name, flux or mmdit (sd3)
ANNO_FILE=annotation/video_text.jsonl   # The video-text annotation file path


torchrun --nproc_per_node $GPUS \
    tools/extract_text_features.py \
    --batch_size 1 \
    --model_dtype bf16 \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --anno_file $ANNO_FILE