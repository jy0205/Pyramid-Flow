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

from .modeling_pyramid_mmdit import PyramidDiffusionMMDiT
from .modeling_text_encoder import SD3TextEncoderWithMask


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


class PyramidDiTForVideoGeneration:
    """
        The pyramid dit for both image and video generation, The running class wrapper
        This class is mainly for fixed unit implementation: 1 + n + n + n
    """
    def __init__(self, model_path, model_dtype='bf16', use_gradient_checkpointing=False, return_log=True,
        model_variant="diffusion_transformer_768p", timestep_shift=1.0, stage_range=[0, 1/3, 2/3, 1],
        sample_ratios=[1, 1, 1], scheduler_gamma=1/3, use_mixed_training=False, use_flash_attn=False, 
        load_text_encoder=True, load_vae=True, max_temporal_length=31, frame_per_unit=1, use_temporal_causal=True, 
        corrupt_ratio=1/3, interp_condition_pos=True, stages=[1, 2, 4], **kwargs,
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
        if use_mixed_training:
            print("using mixed precision training, do not explicitly casting models")
            self.dit = PyramidDiffusionMMDiT.from_pretrained(
                dit_path, use_gradient_checkpointing=use_gradient_checkpointing, 
                use_flash_attn=use_flash_attn, use_t5_mask=True, 
                add_temp_pos_embed=True, temp_pos_embed_type='rope', 
                use_temporal_causal=use_temporal_causal, interp_condition_pos=interp_condition_pos,
            )
        else:
            print("using half precision")
            self.dit = PyramidDiffusionMMDiT.from_pretrained(
                dit_path, torch_dtype=torch_dtype, 
                use_gradient_checkpointing=use_gradient_checkpointing, 
                use_flash_attn=use_flash_attn, use_t5_mask=True,
                add_temp_pos_embed=True, temp_pos_embed_type='rope', 
                use_temporal_causal=use_temporal_causal, interp_condition_pos=interp_condition_pos,
            )

        # The text encoder
        if load_text_encoder:
            self.text_encoder = SD3TextEncoderWithMask(model_path, torch_dtype=torch_dtype)
        else:
            self.text_encoder = None

        # The base video vae decoder
        if load_vae:
            self.vae = CausalVideoVAE.from_pretrained(os.path.join(model_path, 'causal_video_vae'), torch_dtype=torch_dtype, interpolate=False)
            # Freeze vae
            for parameter in self.vae.parameters():
                parameter.requires_grad = False
        else:
            self.vae = None
        
        # For the image latent
        self.vae_shift_factor = 0.1490
        self.vae_scale_factor = 1 / 1.8415

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
        self.sequential_offload_enabled = False
        
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
            self.scheduler.set_timesteps(num_inference_steps[i_s], i_s, device=device, dtype=dtype)
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
        num_channels_latents = self.dit.config.in_channels
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
        num_channels_latents = self.dit.config.in_channels
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
