#!/bin/bash

# This script is used for Pyramid-Flow Image and Video Generation Training (without using Temporal Pyramid and autoregressive training)
# Since the design of spatial pyramid and temporal pyramid are decoupled, we can only use the spatial pyramid flow
# to train with full-sequence diffusion, which is also more effective than the normal flow matching training strategy


GPUS=8  # The gpu number
TASK=t2i   # t2i or t2v
SHARD_STRATEGY=zero2   # zero2 or zero3
MODEL_NAME=pyramid_flux     # The model name, `pyramid_flux` or `pyramid_mmdit`
MODEL_PATH=/PATH/pyramid-flow-miniflux  # The downloaded ckpt dir. IMPORTANT: It should match with model_name, flux or mmdit (sd3)
VARIANT=diffusion_transformer_image  # The DiT Variant, diffusion_transformer_image or diffusion_transformer_384p

OUTPUT_DIR=/PATH/output_dir    # The checkpoint saving dir
NUM_FRAMES=8         # e.g., 8 for 2s, 16 for 5s, 32 for 10s
BATCH_SIZE=4         # It should satisfy batch_size % 4 == 0
RESOLUTION="768p"    # 384p or 768p
ANNO_FILE=annotation/image_text.jsonl  # The annotation file path


torchrun --nproc_per_node $GPUS \
    train/train_pyramid_flow.py \
    --num_workers 8 \
    --task $TASK \
    --use_fsdp \
    --fsdp_shard_strategy $SHARD_STRATEGY \
    --use_flash_attn \
    --load_text_encoder \
    --load_vae \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --model_dtype bf16 \
    --model_variant $VARIANT \
    --schedule_shift 1.0 \
    --gradient_accumulation_steps 1 \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_frames $NUM_FRAMES \
    --resolution $RESOLUTION \
    --anno_file $ANNO_FILE \
    --frame_per_unit 1 \
    --lr_scheduler constant_with_warmup \
    --opt adamw \
    --opt_beta1 0.9 \
    --opt_beta2 0.95 \
    --seed 42 \
    --weight_decay 1e-4 \
    --clip_grad 1.0 \
    --lr 1e-4 \
    --warmup_steps 1000 \
    --epochs 20 \
    --iters_per_epoch 2000 \
    --report_to tensorboard \
    --print_freq 40 \
    --save_ckpt_freq 1