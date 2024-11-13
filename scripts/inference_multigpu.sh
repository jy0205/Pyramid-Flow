#!/bin/bash

# This scripts using 2 gpus to inference.
# Now only supports 2GPUs and 4GPUs for pyramid-flow-sd3; and 2GPUs or 3 GPUs for pyramid-flow-miniflux
# You can set it to 4 to further reduce the generating time
# Requires nproc_per_node == sp_group_size
# Replace the model_path to your downloaded ckpt dir

GPUS=2 # should be 2 for pyramid_flux, and 2 or 4 for pyramid_mmdit
MODEL_NAME=pyramid_flux    # or pyramid_mmdit
VARIANT=diffusion_transformer_768p   # or diffusion_transformer_384p
MODEL_PATH=/home/jinyang06/models/pyramid-flow-miniflux   # Replace with your checkpoint path
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