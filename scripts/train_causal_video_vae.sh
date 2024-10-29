#!/bin/bash

# This script is used for Causal VAE Training
# It undergoes a two-stage training
# Stage-1: image and video mixed training
# Stage-2: pure video training, using context parallel to load video with more video frames (up to 257 frames)

GPUS=8  # The gpu number
VAE_MODEL_PATH=PATH/vae_ckpt   # The vae model dir
LPIPS_CKPT=vgg_lpips.pth    # The LPIPS VGG CKPT path, used for calculating the lpips loss
OUTPUT_DIR=/PATH/output_dir    # The checkpoint saving dir
IMAGE_ANNO=annotation/image_text.jsonl   # The image annotation file path
VIDEO_ANNO=annotation/video_text.jsonl   # The video annotation file path
RESOLUTION=256     # The training resolution, default is 256
NUM_FRAMES=17     # x * 8 + 1, the number of video frames
BATCH_SIZE=2  


# Stage-1
torchrun --nproc_per_node $GPUS \
    train/train_video_vae.py \
    --num_workers 6 \
    --model_path $VAE_MODEL_PATH \
    --model_dtype bf16 \
    --lpips_ckpt $LPIPS_CKPT \
    --output_dir $OUTPUT_DIR \
    --image_anno $IMAGE_ANNO \
    --video_anno $VIDEO_ANNO \
    --use_image_video_mixed_training \
    --image_mix_ratio 0.1 \
    --resolution $RESOLUTION \
    --max_frames $NUM_FRAMES \
    --disc_start 250000 \
    --kl_weight 1e-12 \
    --pixelloss_weight 10.0 \
    --perceptual_weight 1.0 \
    --disc_weight 0.5 \
    --batch_size $BATCH_SIZE \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --seed 42 \
    --weight_decay 1e-3 \
    --clip_grad 1.0 \
    --lr 1e-4 \
    --lr_disc 1e-4 \
    --warmup_epochs 1 \
    --epochs 100 \
    --iters_per_epoch 2000 \
    --print_freq 40 \
    --save_ckpt_freq 1


# Stage-2
CONTEXT_SIZE=2  # context parallel size, GPUS % CONTEXT_SIZE == 0
NUM_FRAMES=33   # 17 * CONTEXT_SIZE + 1
VAE_CKPT_PATH=stage1_path   # The stage-1 trained ckpt

torchrun --nproc_per_node $GPUS \
    train/train_video_vae.py \
    --num_workers 6 \
    --model_path $VAE_MODEL_PATH \
    --model_dtype bf16 \
    --pretrained_vae_weight $VAE_CKPT_PATH \
    --use_context_parallel \
    --context_size $CONTEXT_SIZE \
    --lpips_ckpt $LPIPS_CKPT \
    --output_dir $OUTPUT_DIR \
    --video_anno $VIDEO_ANNO \
    --image_mix_ratio 0.0 \
    --resolution $RESOLUTION \
    --max_frames $NUM_FRAMES \
    --disc_start 250000 \
    --kl_weight 1e-12 \
    --pixelloss_weight 10.0 \
    --perceptual_weight 1.0 \
    --disc_weight 0.5 \
    --batch_size $BATCH_SIZE \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --seed 42 \
    --weight_decay 1e-3 \
    --clip_grad 1.0 \
    --lr 1e-4 \
    --lr_disc 1e-4 \
    --warmup_epochs 1 \
    --epochs 100 \
    --iters_per_epoch 2000 \
    --print_freq 40 \
    --save_ckpt_freq 1