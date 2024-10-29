# Pyramid Flow's VAE Training Guide

This is the training guide for a [MAGVIT-v2](https://arxiv.org/abs/2310.05737) like continuous 3D VAE, which should be quite flexible. Feel free to build your own video generative model on this part of VAE training code. Please refer to [another document](https://github.com/jy0205/Pyramid-Flow/blob/main/docs/DiT) for DiT finetuning.

## Hardware Requirements

+ VAE training: At least 8 A100 GPUs.


## Prepare the Dataset

The training of our causal video vae uses both image and video data. Both of them should be arranged into a json file, with `video` or `image` field. The final training annotation json file should look like the following format:

```
# For Video
{"video": video_path}

# For Image
{"image": image_path}
```

## Run Training

The causal video vae undergoes a two-stage training. 
+ Stage-1: image and video mixed training
+ Stage-2: pure video training, using context parallel to load video with more video frames

The VAE training script is `scripts/train_causal_video_vae.sh`, run it as follows:

```bash
sh scripts/train_causal_video_vae.sh
```

We also provide a VAE demo `causal_video_vae_demo.ipynb` for image and video reconstruction.


## Tips

+ For stage-1, we use a mixed image and video training. Add the param `--use_image_video_mixed_training` to support the mixed training. We set the image ratio to 0.1 by default. 
+ Set the `resolution` to 256 is enough for VAE training.
+ For stage-1, the `max_frames` is set to 17. It means we use 17 sampled video frames for training.
+ For stage-2, we open the param `use_context_parallel` to distribute long video frames to multiple GPUs. Make sure to set `GPUS % CONTEXT_SIZE == 0` and `NUM_FRAMES=17 * CONTEXT_SIZE + 1`