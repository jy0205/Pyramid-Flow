import torch
import os
import gc
import sys
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import math
import random
import PIL
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union
from accelerate import Accelerator, cpu_offload
from diffusion_schedulers import PyramidFlowMatchEulerDiscreteScheduler
from video_vae.modeling_causal_vae import CausalVideoVAE

from trainer_misc import (
    all_to_all,
    is_sequence_parallel_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_group_rank,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_rank,
)

from .mmdit_modules import (
    PyramidDiffusionMMDiT,
    SD3TextEncoderWithMask,
)

from .flux_modules import (
    PyramidFluxTransformer,
    FluxTextEncoderWithMask,
)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def build_pyramid_dit(
    model_name : str,
    model_path : str,
    torch_dtype,
    use_flash_attn : bool,
    use_mixed_training: bool,
    interp_condition_pos: bool = True,
    use_gradient_checkpointing: bool = False,
    use_temporal_causal: bool = True,
    gradient_checkpointing_ratio: float = 0.6,
):
    model_dtype = torch.float32 if use_mixed_training else torch_dtype
    if model_name == "pyramid_flux":
        dit = PyramidFluxTransformer.from_pretrained(
            model_path, torch_dtype=model_dtype,
            use_gradient_checkpointing=use_gradient_checkpointing, 
            gradient_checkpointing_ratio=gradient_checkpointing_ratio,
            use_flash_attn=use_flash_attn, use_temporal_causal=use_temporal_causal,
            interp_condition_pos=interp_condition_pos, axes_dims_rope=[16, 24, 24],
        )
    elif model_name == "pyramid_mmdit":
        dit = PyramidDiffusionMMDiT.from_pretrained(
            model_path, torch_dtype=model_dtype, use_gradient_checkpointing=use_gradient_checkpointing, 
            gradient_checkpointing_ratio=gradient_checkpointing_ratio,
            use_flash_attn=use_flash_attn, use_t5_mask=True, 
            add_temp_pos_embed=True, temp_pos_embed_type='rope', 
            use_temporal_causal=use_temporal_causal, interp_condition_pos=interp_condition_pos,
        )
    else:
        raise NotImplementedError(f"Unsupported DiT architecture, please set the model_name to `pyramid_flux` or `pyramid_mmdit`")

    return dit


def build_text_encoder(
    model_name : str,
    model_path : str,
    torch_dtype,
    load_text_encoder: bool = True,
):
    # The text encoder
    if load_text_encoder:
        if model_name == "pyramid_flux":
            text_encoder = FluxTextEncoderWithMask(model_path, torch_dtype=torch_dtype)
        elif model_name == "pyramid_mmdit":
            text_encoder = SD3TextEncoderWithMask(model_path, torch_dtype=torch_dtype)
        else:
            raise NotImplementedError(f"Unsupported Text Encoder architecture, please set the model_name to `pyramid_flux` or `pyramid_mmdit`")
    else:
        text_encoder = None

    return text_encoder


class PyramidDiTForVideoGeneration:
    """
        The pyramid dit for both image and video generation, The running class wrapper
        This class is mainly for fixed unit implementation: 1 + n + n + n
    """
    def __init__(self, model_path, model_dtype='bf16', model_name='pyramid_mmdit', use_gradient_checkpointing=False, 
        return_log=True, model_variant="diffusion_transformer_768p", timestep_shift=1.0, stage_range=[0, 1/3, 2/3, 1],
        sample_ratios=[1, 1, 1], scheduler_gamma=1/3, use_mixed_training=False, use_flash_attn=False, 
        load_text_encoder=True, load_vae=True, max_temporal_length=31, frame_per_unit=1, use_temporal_causal=True, 
        corrupt_ratio=1/3, interp_condition_pos=True, stages=[1, 2, 4], video_sync_group=8, gradient_checkpointing_ratio=0.6, **kwargs,
    ):
        super().__init__()

        if model_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif model_dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.stages = stages
        self.sample_ratios = sample_ratios
        self.corrupt_ratio = corrupt_ratio

        dit_path = os.path.join(model_path, model_variant)

        # The dit
        self.dit = build_pyramid_dit(
            model_name, dit_path, torch_dtype, 
            use_flash_attn=use_flash_attn, use_mixed_training=use_mixed_training,
            interp_condition_pos=interp_condition_pos, use_gradient_checkpointing=use_gradient_checkpointing,
            use_temporal_causal=use_temporal_causal, gradient_checkpointing_ratio=gradient_checkpointing_ratio,
        )

        # The text encoder
        self.text_encoder = build_text_encoder(
            model_name, model_path, torch_dtype, load_text_encoder=load_text_encoder,
        )
        self.load_text_encoder = load_text_encoder

        # The base video vae decoder
        if load_vae:
            self.vae = CausalVideoVAE.from_pretrained(os.path.join(model_path, 'causal_video_vae'), torch_dtype=torch_dtype, interpolate=False)
            # Freeze vae
            for parameter in self.vae.parameters():
                parameter.requires_grad = False
        else:
            self.vae = None
        self.load_vae = load_vae
        
        # For the image latent
        if model_name == "pyramid_flux":
            self.vae_shift_factor = -0.04
            self.vae_scale_factor = 1 / 1.8726
        elif model_name == "pyramid_mmdit":
            self.vae_shift_factor = 0.1490
            self.vae_scale_factor = 1 / 1.8415
        else:
            raise NotImplementedError(f"Unsupported model name : {model_name}")

        # For the video latent
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986

        self.downsample = 8

        # Configure the video training hyper-parameters
        # The video sequence: one frame + N * unit
        self.frame_per_unit = frame_per_unit
        self.max_temporal_length = max_temporal_length
        assert (max_temporal_length - 1) % frame_per_unit == 0, "The frame number should be divided by the frame number per unit"
        self.num_units_per_video = 1 + ((max_temporal_length - 1) // frame_per_unit) + int(sum(sample_ratios))

        self.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
            shift=timestep_shift, stages=len(self.stages), 
            stage_range=stage_range, gamma=scheduler_gamma,
        )
        print(f"The start sigmas and end sigmas of each stage is Start: {self.scheduler.start_sigmas}, End: {self.scheduler.end_sigmas}, Ori_start: {self.scheduler.ori_start_sigmas}")
        
        self.cfg_rate = 0.1
        self.return_log = return_log
        self.use_flash_attn = use_flash_attn
        self.model_name = model_name
        self.sequential_offload_enabled = False
        self.accumulate_steps = 0
        self.video_sync_group = video_sync_group
        
    def _enable_sequential_cpu_offload(self, model):
        self.sequential_offload_enabled = True
        torch_device = torch.device("cuda")
        device_type = torch_device.type
        device = torch.device(f"{device_type}:0")
        offload_buffers = len(model._parameters) > 0
        cpu_offload(model, device, offload_buffers=offload_buffers)
    
    def enable_sequential_cpu_offload(self):
        self._enable_sequential_cpu_offload(self.text_encoder)
        self._enable_sequential_cpu_offload(self.dit)

    def load_checkpoint(self, checkpoint_path, model_key='model', **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        dit_checkpoint = OrderedDict()
        for key in checkpoint:
            if key.startswith('vae') or key.startswith('text_encoder'):
                continue
            if key.startswith('dit'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[1:])
                dit_checkpoint[new_key] = checkpoint[key]
            else:
                dit_checkpoint[key] = checkpoint[key]

        load_result = self.dit.load_state_dict(dit_checkpoint, strict=True)
        print(f"Load checkpoint from {checkpoint_path}, load result: {load_result}")

    def load_vae_checkpoint(self, vae_checkpoint_path, model_key='model'):
        checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
        checkpoint = checkpoint[model_key]
        loaded_checkpoint = OrderedDict()
        
        for key in checkpoint.keys():
            if key.startswith('vae.'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[1:])
                loaded_checkpoint[new_key] = checkpoint[key]

        load_result = self.vae.load_state_dict(loaded_checkpoint)
        print(f"Load the VAE from {vae_checkpoint_path}, load result: {load_result}")

    @torch.no_grad()
    def add_pyramid_noise(
        self, 
        latents_list,
        sample_ratios=[1, 1, 1],
    ):
        """
        add the noise for each pyramidal stage
            noting that, this method is a general strategy for pyramid-flow, it 
            can be used for both image and video training.
            You can also use this method to train pyramid-flow with full-sequence 
            diffusion in video generation (without using temporal pyramid and autoregressive modeling)

        Params:
            latent_list: [low_res, mid_res, high_res] The vae latents of all stages
            sample_ratios: The proportion of each stage in the training batch
        """
        noise = torch.randn_like(latents_list[-1])
        device = noise.device
        dtype = latents_list[-1].dtype
        t = noise.shape[2]

        stages = len(self.stages)
        tot_samples = noise.shape[0]
        assert tot_samples % (int(sum(sample_ratios))) == 0
        assert stages == len(sample_ratios)
        
        height, width = noise.shape[-2], noise.shape[-1]
        noise_list = [noise]
        cur_noise = noise
        for i_s in range(stages-1):
            height //= 2;width //= 2
            cur_noise = rearrange(cur_noise, 'b c t h w -> (b t) c h w')
            cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
            cur_noise = rearrange(cur_noise, '(b t) c h w -> b c t h w', t=t)
            noise_list.append(cur_noise)

        noise_list = list(reversed(noise_list))   # make sure from low res to high res
        
        # To calculate the padding batchsize and column size
        batch_size = tot_samples // int(sum(sample_ratios))
        column_size = int(sum(sample_ratios))
        
        column_to_stage = {}
        i_sum = 0
        for i_s, column_num in enumerate(sample_ratios):
            for index in range(i_sum, i_sum + column_num):
                column_to_stage[index] = i_s
            i_sum += column_num

        noisy_latents_list = []
        ratios_list = []
        targets_list = []
        timesteps_list = []
        training_steps = self.scheduler.config.num_train_timesteps

        # from low resolution to high resolution
        for index in range(column_size):
            i_s = column_to_stage[index]
            clean_latent = latents_list[i_s][index::column_size]   # [bs, c, t, h, w]
            last_clean_latent = None if i_s == 0 else latents_list[i_s-1][index::column_size]
            start_sigma = self.scheduler.start_sigmas[i_s]
            end_sigma = self.scheduler.end_sigmas[i_s]
            
            if i_s == 0:
                start_point = noise_list[i_s][index::column_size]
            else:
                # Get the upsampled latent
                last_clean_latent = rearrange(last_clean_latent, 'b c t h w -> (b t) c h w')
                last_clean_latent = F.interpolate(last_clean_latent, size=(last_clean_latent.shape[-2] * 2, last_clean_latent.shape[-1] * 2), mode='nearest')
                last_clean_latent = rearrange(last_clean_latent, '(b t) c h w -> b c t h w', t=t)
                start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent
            
            if i_s == stages - 1:
                end_point = clean_latent
            else:
                end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

            # To sample a timestep
            u = compute_density_for_timestep_sampling(
                weighting_scheme='random',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )

            indices = (u * training_steps).long()   # Totally 1000 training steps per stage
            indices = indices.clamp(0, training_steps-1)
            timesteps = self.scheduler.timesteps_per_stage[i_s][indices].to(device=device)
            ratios = self.scheduler.sigmas_per_stage[i_s][indices].to(device=device)

            while len(ratios.shape) < start_point.ndim:
                ratios = ratios.unsqueeze(-1)

            # interpolate the latent
            noisy_latents = ratios * start_point + (1 - ratios) * end_point

            last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=device) * self.corrupt_ratio

            # [stage1_latent, stage2_latent, ..., stagen_latent], which will be concat after patching
            noisy_latents_list.append([noisy_latents.to(dtype)])
            ratios_list.append(ratios.to(dtype))
            timesteps_list.append(timesteps.to(dtype))
            targets_list.append(start_point - end_point)     # The standard rectified flow matching objective

        return noisy_latents_list, ratios_list, timesteps_list, targets_list

    def sample_stage_length(self, num_stages, max_units=None):
        max_units_in_training = 1 + ((self.max_temporal_length - 1) // self.frame_per_unit)
        cur_rank = get_rank()

        self.accumulate_steps = self.accumulate_steps + 1
        total_turns =  max_units_in_training // self.video_sync_group
        update_turn = self.accumulate_steps % total_turns

        # # uniformly sampling each position
        cur_highres_unit = max(int((cur_rank % self.video_sync_group + 1) + update_turn * self.video_sync_group), 1)
        cur_mid_res_unit = max(1 + max_units_in_training - cur_highres_unit, 1)
        cur_low_res_unit = cur_mid_res_unit

        if max_units is not None:
            cur_highres_unit = min(cur_highres_unit, max_units)
            cur_mid_res_unit = min(cur_mid_res_unit, max_units)
            cur_low_res_unit = min(cur_low_res_unit, max_units)

        length_list = [cur_low_res_unit, cur_mid_res_unit, cur_highres_unit]
        
        assert len(length_list) == num_stages

        return length_list

    @torch.no_grad()
    def add_pyramid_noise_with_temporal_pyramid(
        self, 
        latents_list,
        sample_ratios=[1, 1, 1],
    ):
        """
        add the noise for each pyramidal stage, used for AR video training with temporal pyramid
        Params:
            latent_list: [low_res, mid_res, high_res] The vae latents of all stages
            sample_ratios: The proportion of each stage in the training batch
        """
        stages = len(self.stages)
        tot_samples = latents_list[0].shape[0]
        device = latents_list[0].device
        dtype = latents_list[0].dtype

        assert tot_samples % (int(sum(sample_ratios))) == 0
        assert stages == len(sample_ratios)

        noise = torch.randn_like(latents_list[-1])
        t = noise.shape[2]

        # To allocate the temporal length of each stage, ensuring the sum == constant
        max_units = 1 + (t - 1) // self.frame_per_unit

        if is_sequence_parallel_initialized():
            max_units_per_sample = torch.LongTensor([max_units]).to(device)
            sp_group = get_sequence_parallel_group()
            sp_group_size = get_sequence_parallel_world_size()
            max_units_per_sample = all_to_all(max_units_per_sample.unsqueeze(1).repeat(1, sp_group_size), sp_group, sp_group_size, scatter_dim=1, gather_dim=0).squeeze(1)
            max_units = min(max_units_per_sample.cpu().tolist())

        num_units_per_stage = self.sample_stage_length(stages, max_units=max_units)   # [The unit number of each stage]

        # we needs to sync the length alloc of each sequence parallel group
        if is_sequence_parallel_initialized():
            num_units_per_stage = torch.LongTensor(num_units_per_stage).to(device)
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(num_units_per_stage, global_src_rank, group=get_sequence_parallel_group())
            num_units_per_stage = num_units_per_stage.tolist()

        height, width = noise.shape[-2], noise.shape[-1]
        noise_list = [noise]
        cur_noise = noise
        for i_s in range(stages-1):
            height //= 2;width //= 2
            cur_noise = rearrange(cur_noise, 'b c t h w -> (b t) c h w')
            cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
            cur_noise = rearrange(cur_noise, '(b t) c h w -> b c t h w', t=t)
            noise_list.append(cur_noise)

        noise_list = list(reversed(noise_list))   # make sure from low res to high res

        # To calculate the batchsize and column size
        batch_size = tot_samples // int(sum(sample_ratios))
        column_size = int(sum(sample_ratios))

        column_to_stage = {}
        i_sum = 0
        for i_s, column_num in enumerate(sample_ratios):
            for index in range(i_sum, i_sum + column_num):
                column_to_stage[index] = i_s
            i_sum += column_num

        noisy_latents_list = []
        ratios_list = []
        targets_list = []
        timesteps_list = []
        training_steps = self.scheduler.config.num_train_timesteps

        # from low resolution to high resolution
        for index in range(column_size):
            # First prepare the trainable latent construction
            i_s = column_to_stage[index]
            clean_latent = latents_list[i_s][index::column_size]   # [bs, c, t, h, w]
            last_clean_latent = None if i_s == 0 else latents_list[i_s-1][index::column_size]
            start_sigma = self.scheduler.start_sigmas[i_s]
            end_sigma = self.scheduler.end_sigmas[i_s]

            if i_s == 0:
                start_point = noise_list[i_s][index::column_size]
            else:
                # Get the upsampled latent
                last_clean_latent = rearrange(last_clean_latent, 'b c t h w -> (b t) c h w')
                last_clean_latent = F.interpolate(last_clean_latent, size=(last_clean_latent.shape[-2] * 2, last_clean_latent.shape[-1] * 2), mode='nearest')
                last_clean_latent = rearrange(last_clean_latent, '(b t) c h w -> b c t h w', t=t)
                start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent
            
            if i_s == stages - 1:
                end_point = clean_latent
            else:
                end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

            # To sample a timestep
            u = compute_density_for_timestep_sampling(
                weighting_scheme='random',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )

            indices = (u * training_steps).long()   # Totally 1000 training steps per stage
            indices = indices.clamp(0, training_steps-1)
            timesteps = self.scheduler.timesteps_per_stage[i_s][indices].to(device=device)
            ratios = self.scheduler.sigmas_per_stage[i_s][indices].to(device=device)
            noise_ratios = ratios * start_sigma + (1 - ratios) * end_sigma

            while len(ratios.shape) < start_point.ndim:
                ratios = ratios.unsqueeze(-1)

            # interpolate the latent
            noisy_latents = ratios * start_point + (1 - ratios) * end_point

            # The flow matching object
            target_latents = start_point - end_point

            # pad the noisy previous
            num_units = num_units_per_stage[i_s]
            num_units = min(num_units, 1 + (t - 1) // self.frame_per_unit)
            actual_frames = 1 + (num_units - 1) * self.frame_per_unit

            noisy_latents = noisy_latents[:, :, :actual_frames]
            target_latents = target_latents[:, :, :actual_frames]

            clean_latent = clean_latent[:, :, :actual_frames]
            stage_noise = noise_list[i_s][index::column_size][:, :, :actual_frames]

            # only the last latent takes part in training
            noisy_latents = noisy_latents[:, :, -self.frame_per_unit:] 
            target_latents = target_latents[:, :, -self.frame_per_unit:]

            last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=device) * self.corrupt_ratio

            if num_units == 1:
                stage_input = [noisy_latents.to(dtype)]
            else:
                # add the random noise for the last cond clip
                last_cond_latent = clean_latent[:, :, -(2*self.frame_per_unit):-self.frame_per_unit]

                while len(last_cond_noisy_sigma.shape) < last_cond_latent.ndim:
                    last_cond_noisy_sigma = last_cond_noisy_sigma.unsqueeze(-1)

                # We adding some noise to corrupt the clean condition
                last_cond_latent = last_cond_noisy_sigma * torch.randn_like(last_cond_latent) + (1 - last_cond_noisy_sigma) * last_cond_latent

                # concat the corrupted condition and the input noisy latents
                stage_input = [noisy_latents.to(dtype), last_cond_latent.to(dtype)]

                cur_unit_num = 2
                cur_stage = i_s

                while cur_unit_num < num_units:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_num += 1
                    cond_latents = latents_list[cur_stage][index::column_size][:, :, :actual_frames]
                    cond_latents = cond_latents[:, :, -(cur_unit_num * self.frame_per_unit) : -((cur_unit_num - 1) * self.frame_per_unit)]
                    cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(cond_latents.to(dtype))

                if cur_stage == 0 and cur_unit_num < num_units:
                    cond_latents = latents_list[0][index::column_size][:, :, :actual_frames]
                    cond_latents = cond_latents[:, :, :-(cur_unit_num * self.frame_per_unit)]

                    cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(cond_latents.to(dtype))

            stage_input = list(reversed(stage_input))
            noisy_latents_list.append(stage_input)
            ratios_list.append(ratios.to(dtype))
            timesteps_list.append(timesteps.to(dtype))
            targets_list.append(target_latents)     # The standard rectified flow matching objective
        
        return noisy_latents_list, ratios_list, timesteps_list, targets_list
    
    @torch.no_grad()
    def get_pyramid_latent(self, x, stage_num):
        # x is the origin vae latent
        vae_latent_list = []
        vae_latent_list.append(x)

        temp, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
        for _ in range(stage_num):
            height //= 2
            width //= 2
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = torch.nn.functional.interpolate(x, size=(height, width), mode='bilinear')
            x = rearrange(x, '(b t) c h w -> b c t h w', t=temp)
            vae_latent_list.append(x)

        vae_latent_list = list(reversed(vae_latent_list))
        return vae_latent_list

    @torch.no_grad()
    def get_vae_latent(self, video, use_temporal_pyramid=True):
        if self.load_vae:
            assert video.shape[1] == 3, "The vae is loaded, the input should be raw pixels"
            video = self.vae.encode(video).latent_dist.sample() # [b c t h w]

        if video.shape[2] == 1:
            # is image
            video = (video - self.vae_shift_factor) * self.vae_scale_factor
        else:
            # is video
            video[:, :, :1] = (video[:, :, :1] - self.vae_shift_factor) * self.vae_scale_factor
            video[:, :, 1:] =  (video[:, :, 1:] - self.vae_video_shift_factor) * self.vae_video_scale_factor
        
        # Get the pyramidal stages
        vae_latent_list = self.get_pyramid_latent(video, len(self.stages) - 1)

        if use_temporal_pyramid:
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.add_pyramid_noise_with_temporal_pyramid(vae_latent_list, self.sample_ratios)
        else:
            # Only use the spatial pyramidal (without temporal ar)
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.add_pyramid_noise(vae_latent_list, self.sample_ratios)
        
        return noisy_latents_list, ratios_list, timesteps_list, targets_list

    @torch.no_grad()
    def get_text_embeddings(self, text, rand_idx, device):
        if self.load_text_encoder:
            batch_size = len(text)   # Text is a str list
            for idx in range(batch_size):
                if rand_idx[idx].item():
                    text[idx] = ''
            return self.text_encoder(text, device)   # [b s c]
        else:
            batch_size = len(text['prompt_embeds'])

            for idx in range(batch_size):
                if rand_idx[idx].item():
                    text['prompt_embeds'][idx] = self.null_text_embeds['prompt_embed'].to(device)
                    text['prompt_attention_mask'][idx] = self.null_text_embeds['prompt_attention_mask'].to(device)
                    text['pooled_prompt_embeds'][idx] = self.null_text_embeds['pooled_prompt_embed'].to(device)

            return text['prompt_embeds'], text['prompt_attention_mask'], text['pooled_prompt_embeds']

    def calculate_loss(self, model_preds_list, targets_list):
        loss_list = []
    
        for model_pred, target in zip(model_preds_list, targets_list):
            # Compute the loss.
            loss_weight = torch.ones_like(target)

            loss = torch.mean(
                (loss_weight.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss_list.append(loss)

        diffusion_loss = torch.cat(loss_list, dim=0).mean()

        if self.return_log:
            log = {}
            split="train"
            log[f'{split}/loss'] = diffusion_loss.detach()
            return diffusion_loss, log
        else:
            return diffusion_loss, {}

    def __call__(self, video, text, identifier=['video'], use_temporal_pyramid=True, accelerator: Accelerator=None):
        xdim = video.ndim
        device = video.device

        if 'video' in identifier:
            assert 'image' not in identifier
            is_image = False
        else:
            assert 'video' not in identifier
            video = video.unsqueeze(2)  # 'b c h w -> b c 1 h w'
            is_image = True

        # TODO: now have 3 stages, firstly get the vae latents
        with torch.no_grad(), accelerator.autocast():
            # 10% prob drop the text
            batch_size = len(video)
            rand_idx = torch.rand((batch_size,)) <= self.cfg_rate
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.get_text_embeddings(text, rand_idx, device)
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.get_vae_latent(video, use_temporal_pyramid=use_temporal_pyramid)

        timesteps = torch.cat([timestep.unsqueeze(-1) for timestep in timesteps_list], dim=-1)
        timesteps = timesteps.reshape(-1)

        assert timesteps.shape[0] == prompt_embeds.shape[0]

        # DiT forward
        model_preds_list = self.dit(
            sample=noisy_latents_list,
            timestep_ratio=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled_prompt_embeds,
        )

        # calculate the loss
        return self.calculate_loss(model_preds_list, targets_list)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        temp,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(temp),
            int(height) // self.downsample,
            int(width) // self.downsample,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def sample_block_noise(self, bs, ch, temp, height, width):
        gamma = self.scheduler.config.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma)
        block_number = bs * ch * temp * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',b=bs,c=ch,t=temp,h=height//2,w=width//2,p=2,q=2)
        return noise

    @torch.no_grad()
    def generate_one_unit(
        self,
        latents,
        past_conditions, # List of past conditions, contains the conditions of each stage
        prompt_embeds,
        prompt_attention_mask,
        pooled_prompt_embeds,
        num_inference_steps,
        height,
        width,
        temp,
        device,
        dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        is_first_frame: bool = False,
    ):
        stages = self.stages
        intermed_latents = []

        for i_s in range(len(stages)):
            self.scheduler.set_timesteps(num_inference_steps[i_s], i_s, device=device)
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                height *= 2; width *= 2
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)
                # Fix the stage
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]   # the original coeff of signal
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                bs, ch, temp, height, width = latents.shape
                noise = self.sample_block_noise(bs, ch, temp, height, width)
                noise = noise.to(device=device, dtype=dtype)
                latents = alpha * latents + beta * noise    # To fix the block artifact

            for idx, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                if is_sequence_parallel_initialized():
                    # sync the input latent
                    sp_group_rank = get_sequence_parallel_group_rank()
                    global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
                    torch.distributed.broadcast(latent_model_input, global_src_rank, group=get_sequence_parallel_group())
                
                latent_model_input = past_conditions[i_s] + [latent_model_input]

                noise_pred = self.dit(
                    sample=[latent_model_input],
                    timestep_ratio=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                )

                noise_pred = noise_pred[0]
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if is_first_frame:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + self.video_guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

            intermed_latents.append(latents)

        return intermed_latents

    @torch.no_grad()
    def generate_i2v(
        self,
        prompt: Union[str, List[str]] = '',
        input_image: PIL.Image = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 4.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        if self.sequential_offload_enabled and not cpu_offloading:
            print("Warning: overriding cpu_offloading set to false, as it's needed for sequential cpu offload")
            cpu_offloading=True
        device = self.device if not cpu_offloading else torch.device("cuda")
        dtype = self.dtype
        if cpu_offloading:
            # skip caring about the text encoder here as its about to be used anyways.
            if not self.sequential_offload_enabled:
                if str(self.dit.device) != "cpu":
                    print("(dit) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                    self.dit.to("cpu")
                    torch.cuda.empty_cache()
            if str(self.vae.device) != "cpu":
                print("(vae) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                self.vae.to("cpu")
                torch.cuda.empty_cache()

        width = input_image.width
        height = input_image.height

        assert temp % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(prompt, str):
            batch_size = 1
            prompt = prompt + ", hyper quality, Ultra HD, 8K"   # adding this prompt to improve aesthetics
        else:
            assert isinstance(prompt, list)
            batch_size = len(prompt)
            prompt = [_ + ", hyper quality, Ultra HD, 8K" for _ in prompt]

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)
        
        negative_prompt = negative_prompt or ""

        # Get the text embeddings
        if cpu_offloading and not self.sequential_offload_enabled:
            self.text_encoder.to("cuda") 
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.text_encoder.to("cpu")
            self.vae.to("cuda")
            torch.cuda.empty_cache()

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp+1)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if is_sequence_parallel_initialized():
            # sync the prompt embedding across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(pooled_prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(prompt_attention_mask, global_src_rank, group=get_sequence_parallel_group())

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by defalut, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = temp // self.frame_per_unit
        stages = self.stages

        # encode the image latents
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        input_image_tensor = image_transform(input_image).unsqueeze(0).unsqueeze(2)   # [b c 1 h w]
        input_image_latent = (self.vae.encode(input_image_tensor.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample() - self.vae_shift_factor) * self.vae_scale_factor  # [b c 1 h w]

        if is_sequence_parallel_initialized():
            # sync the image latent across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(input_image_latent, global_src_rank, group=get_sequence_parallel_group())

        generated_latents_list = [input_image_latent]    # The generated results
        last_generated_latents = input_image_latent

        if cpu_offloading:
            self.vae.to("cpu")
            if not self.sequential_offload_enabled:
                self.dit.to("cuda")
            torch.cuda.empty_cache()
        
        for unit_index in tqdm(range(1, num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)
        
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            # prepare the condition latents
            past_condition_latents = []
            clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
            
            for i_s in range(len(stages)):
                last_cond_latent = clean_latents_list[i_s][:,:,-self.frame_per_unit:]

                stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
        
                # pad the past clean latents
                cur_unit_num = unit_index
                cur_stage = i_s
                cur_unit_ptx = 1

                while cur_unit_ptx < cur_unit_num:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_ptx += 1
                    cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                    cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
            
                stage_input = list(reversed(stage_input))
                past_condition_latents.append(stage_input)

            intermed_latents = self.generate_one_unit(
                latents[:,:,(unit_index - 1) * self.frame_per_unit:unit_index * self.frame_per_unit],
                past_condition_latents,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                num_inference_steps,
                height,
                width,
                self.frame_per_unit,
                device,
                dtype,
                generator,
                is_first_frame=False,
            )
    
            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents

        generated_latents = torch.cat(generated_latents_list, dim=2)

        if output_type == "latent":
            image = generated_latents
        else:
            if cpu_offloading:
                if not self.sequential_offload_enabled:
                    self.dit.to("cpu")
                self.vae.to("cuda")
                torch.cuda.empty_cache()
            image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
            if cpu_offloading:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                # not technically necessary, but returns the pipeline to its original state

        return image

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]]="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        save_memory: bool = True,
        cpu_offloading: bool = False, # If true, reload device will be cuda.
        inference_multigpu: bool = False,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        if self.sequential_offload_enabled and not cpu_offloading:
            print("Warning: overriding cpu_offloading set to false, as it's needed for sequential cpu offload")
            cpu_offloading=True
        device = self.device if not cpu_offloading else torch.device("cuda")
        dtype = self.dtype
        if cpu_offloading:
            # skip caring about the text encoder here as its about to be used anyways.
            if not self.sequential_offload_enabled:
                if str(self.dit.device) != "cpu":
                    print("(dit) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                    self.dit.to("cpu")
                    torch.cuda.empty_cache()
            if str(self.vae.device) != "cpu":
                print("(vae) Warning: Do not preload pipeline components (i.e. to cuda) with cpu offloading enabled! Otherwise, a second transfer will occur needlessly taking up time.")
                self.vae.to("cpu")
                torch.cuda.empty_cache()


        assert (temp - 1) % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(prompt, str):
            batch_size = 1
            prompt = prompt + ", hyper quality, Ultra HD, 8K"        # adding this prompt to improve aesthetics
        else:
            assert isinstance(prompt, list)
            batch_size = len(prompt)
            prompt = [_ + ", hyper quality, Ultra HD, 8K" for _ in prompt]

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)

        if isinstance(video_num_inference_steps, int):
            video_num_inference_steps = [video_num_inference_steps] * len(self.stages)

        negative_prompt = negative_prompt or ""

        # Get the text embeddings
        if cpu_offloading and not self.sequential_offload_enabled:
            self.text_encoder.to("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(prompt, device)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = self.text_encoder(negative_prompt, device)
        if cpu_offloading:
            if not self.sequential_offload_enabled:
                self.text_encoder.to("cpu")
                self.dit.to("cuda")
            torch.cuda.empty_cache()

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            # guidance_scale_list = torch.linspace(max_guidance_scale, min_guidance_scale, temp).tolist()
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if is_sequence_parallel_initialized():
            # sync the prompt embedding across multiple GPUs
            sp_group_rank = get_sequence_parallel_group_rank()
            global_src_rank = sp_group_rank * get_sequence_parallel_world_size()
            torch.distributed.broadcast(prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(pooled_prompt_embeds, global_src_rank, group=get_sequence_parallel_group())
            torch.distributed.broadcast(prompt_attention_mask, global_src_rank, group=get_sequence_parallel_group())

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by default, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = 1 + (temp - 1) // self.frame_per_unit
        stages = self.stages

        generated_latents_list = []    # The generated results
        last_generated_latents = None

        for unit_index in tqdm(range(num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)
            
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            if unit_index == 0:
                past_condition_latents = [[] for _ in range(len(stages))]
                intermed_latents = self.generate_one_unit(
                    latents[:,:,:1],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    num_inference_steps,
                    height,
                    width,
                    1,
                    device,
                    dtype,
                    generator,
                    is_first_frame=True,
                )
            else:
                # prepare the condition latents
                past_condition_latents = []
                clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
                
                for i_s in range(len(stages)):
                    last_cond_latent = clean_latents_list[i_s][:,:,-(self.frame_per_unit):]

                    stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
            
                    # pad the past clean latents
                    cur_unit_num = unit_index
                    cur_stage = i_s
                    cur_unit_ptx = 1

                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage - 1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                
                    stage_input = list(reversed(stage_input))
                    past_condition_latents.append(stage_input)

                intermed_latents = self.generate_one_unit(
                    latents[:,:, 1 + (unit_index - 1) * self.frame_per_unit:1 + unit_index * self.frame_per_unit],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    video_num_inference_steps,
                    height,
                    width,
                    self.frame_per_unit,
                    device,
                    dtype,
                    generator,
                    is_first_frame=False,
                )

            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents

        generated_latents = torch.cat(generated_latents_list, dim=2)

        if output_type == "latent":
            image = generated_latents
        else:
            if cpu_offloading:
                if not self.sequential_offload_enabled:
                    self.dit.to("cpu")
                self.vae.to("cuda")
                torch.cuda.empty_cache()
            image = self.decode_latent(generated_latents, save_memory=save_memory, inference_multigpu=inference_multigpu)
            if cpu_offloading:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                # not technically necessary, but returns the pipeline to its original state

        return image

    def decode_latent(self, latents, save_memory=True, inference_multigpu=False):
        # only the main process needs vae decoding
        if inference_multigpu and get_rank() != 0:
            return None

        if latents.shape[2] == 1:
            latents = (latents / self.vae_scale_factor) + self.vae_shift_factor
        else:
            latents[:, :, :1] = (latents[:, :, :1] / self.vae_scale_factor) + self.vae_shift_factor
            latents[:, :, 1:] = (latents[:, :, 1:] / self.vae_video_scale_factor) + self.vae_video_shift_factor

        if save_memory:
            # reducing the tile size and temporal chunk window size
            image = self.vae.decode(latents, temporal_chunk=True, window_size=1, tile_sample_min_size=256).sample
        else:
            image = self.vae.decode(latents, temporal_chunk=True, window_size=2, tile_sample_min_size=512).sample

        image = image.mul(127.5).add(127.5).clamp(0, 255).byte()
        image = rearrange(image, "B C T H W -> (B T) H W C")
        image = image.cpu().numpy()
        image = self.numpy_to_pil(image)
        
        return image

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @property
    def device(self):
        return next(self.dit.parameters()).device

    @property
    def dtype(self):
        return next(self.dit.parameters()).dtype

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def video_guidance_scale(self):
        return self._video_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0
