# Pyramid Flow's DiT Finetuning Guide

This is the finetuning guide for the DiT in Pyramid Flow. We provide instructions for both autoregressive and non-autoregressive versions. The former is more research oriented and the latter is more stable (but less efficient without temporal pyramid). Please refer to [another document](https://github.com/jy0205/Pyramid-Flow/blob/main/docs/VAE) for VAE finetuning.

## Hardware Requirements

+ DiT finetuning: At least 8 A100 GPUs.


## Prepare the Dataset

The training dataset should be arranged into a json file, with `video`, `text` fields. Since the video vae latent extraction is very slow, we strongly recommend you to pre-extract the video vae latents to save the training time. We provide a video vae latent extraction script in folder `tools`. You can run it with the following command:

```bash
sh scripts/extract_vae_latent.sh
```

(optional) Since the T5 text encoder will cost a lot of GPU memory, pre-extract the text features will save the training memory. We also provide a text feature extraction script in folder `tools`. You can run it with the following command:

```bash
sh scripts/extract_text_feature.sh
```

The final training annotation json file should look like the following format:

```
{"video": video_path, "text": text prompt, "latent": extracted video vae latent, "text_fea": extracted text feature}
```

We provide the example json annotation files for [video](https://github.com/jy0205/Pyramid-Flow/blob/main/annotation/video_text.jsonl) and [image](https://github.com/jy0205/Pyramid-Flow/blob/main/annotation/image_text.jsonl)) training in the `annotation` folder. You can refer them to prepare your training dataset.


## Run Training
We provide two types of training scripts: (1) autoregressive video generation training with temporal pyramid. (2) Full-sequence diffusion training with pyramid-flow for both text-to-image and text-to-video training. This corresponds to the following two script files. Running these training scripts using at least 8 GPUs:

+ `scripts/train_pyramid_flow.sh`: The autoregressive video generation training with temporal pyramid.

```bash
sh scripts/train_pyramid_flow.sh
```

+ `scripts/train_pyramid_flow_without_ar.sh`: Using pyramid-flow for full-sequence diffusion training.

```bash
sh scripts/train_pyramid_flow_without_ar.sh
```


## Tips

+ For the 768p version, make sure to add the args:  `--gradient_checkpointing`
+ Param `NUM_FRAMES` should be set to a multiple of 8
+ For the param `video_sync_group`, it indicates the number of process that accepts the same input video, used for temporal pyramid AR training. We recommend to set this value to 4, 8 or 16. (16 is better if you have more GPUs)
+ Make sure to set `NUM_FRAMES % VIDEO_SYNC_GROUP == 0`, `GPUS % VIDEO_SYNC_GROUP == 0`, and `BATCH_SIZE % 4 == 0`
