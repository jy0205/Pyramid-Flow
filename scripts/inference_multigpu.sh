#!/bin/bash

# This scripts using 2 gpus to inference. Now only tested at 2GPUs and 4GPUs
# You can set it to 4 to further reduce the generating time
# Requires nproc_per_node == sp_group_size
# Replace the model_path to your downloaded ckpt dir

GPUS=4 # should be 2 or 4
MODEL_NAME=pyramid_mmdit    # or pyramid_flux
VARIANT=diffusion_transformer_768p
MODEL_PATH=PATH
TASK=t2v    # i2v for image-to-video

torchrun --nproc_per_node $GPUS \
    inference_multigpu.py \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --variant $VARIANT \
    --task $TASK \
    --model_dtype bf16 \
    --temp 16 \
    --sp_group_size $GPUS