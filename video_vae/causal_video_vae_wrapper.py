import torch
import os
import torch.nn as nn
from collections import OrderedDict
from .modeling_causal_vae import CausalVideoVAE
from .modeling_loss import LPIPSWithDiscriminator
from einops import rearrange
from PIL import Image
from IPython import embed

from utils import (
    is_context_parallel_initialized,
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_context_parallel_rank,
    get_context_parallel_group_rank,
)

from .context_parallel_ops import (
    conv_scatter_to_context_parallel_region,
    conv_gather_from_context_parallel_region,
)


class CausalVideoVAELossWrapper(nn.Module):
    """
        The causal video vae training and inference running wrapper
    """
    def __init__(self, model_path, model_dtype='fp32', disc_start=0, logvar_init=0.0, kl_weight=1.0, 
        pixelloss_weight=1.0, perceptual_weight=1.0, disc_weight=0.5, interpolate=True, 
        add_discriminator=True, freeze_encoder=False, load_loss_module=False, lpips_ckpt=None, **kwargs,
    ):
        super().__init__()

        if model_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif model_dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.vae = CausalVideoVAE.from_pretrained(model_path, torch_dtype=torch_dtype, interpolate=False)
        self.vae_scale_factor = self.vae.config.scaling_factor

        if freeze_encoder:
            print("Freeze the parameters of vae encoder")
            for parameter in self.vae.encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.vae.quant_conv.parameters():
                parameter.requires_grad = False

        self.add_discriminator = add_discriminator
        self.freeze_encoder = freeze_encoder

        # Used for training
        if load_loss_module:
            self.loss = LPIPSWithDiscriminator(disc_start, logvar_init=logvar_init, kl_weight=kl_weight,
                pixelloss_weight=pixelloss_weight, perceptual_weight=perceptual_weight, disc_weight=disc_weight, 
                add_discriminator=add_discriminator, using_3d_discriminator=False, disc_num_layers=4, lpips_ckpt=lpips_ckpt)
        else:
            self.loss = None

        self.disc_start = disc_start

    def load_checkpoint(self, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        vae_checkpoint = OrderedDict()
        disc_checkpoint = OrderedDict()

        for key in checkpoint.keys():
            if key.startswith('vae.'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[1:])
                vae_checkpoint[new_key] = checkpoint[key]
            if key.startswith('loss.discriminator'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[2:])
                disc_checkpoint[new_key] = checkpoint[key]

        vae_ckpt_load_result = self.vae.load_state_dict(vae_checkpoint, strict=False)
        print(f"Load vae checkpoint from {checkpoint_path}, load result: {vae_ckpt_load_result}")

        if self.add_discriminator:
            disc_ckpt_load_result = self.loss.discriminator.load_state_dict(disc_checkpoint, strict=False)
            print(f"Load disc checkpoint from {checkpoint_path}, load result: {disc_ckpt_load_result}")

    def forward(self, x, step, identifier=['video']):
        xdim = x.ndim
        if xdim == 4:
            x = x.unsqueeze(2)   #  (B, C, H, W) -> (B, C, 1, H , W)

        if 'video' in identifier:
            # The input is video
            assert 'image' not in identifier
        else:
            # The input is image
            assert 'video' not in identifier
            # We arrange multiple images to a 5D Tensor for compatibility with video input
            # So we needs to reformulate images into 1-frame video tensor 
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = x.unsqueeze(2)  # [(b t) c 1 h w]

        if is_context_parallel_initialized():
            assert self.training, "Only supports during training now"
            cp_world_size = get_context_parallel_world_size()
            global_src_rank = get_context_parallel_group_rank() * cp_world_size
            # sync the input and split
            torch.distributed.broadcast(x, src=global_src_rank, group=get_context_parallel_group())
            batch_x = conv_scatter_to_context_parallel_region(x, dim=2, kernel_size=1)
        else:
            batch_x = x

        posterior, reconstruct = self.vae(batch_x, freeze_encoder=self.freeze_encoder, 
                    is_init_image=True, temporal_chunk=False,)

        # The reconstruct loss
        reconstruct_loss, rec_log = self.loss(
            batch_x, reconstruct, posterior, 
            optimizer_idx=0, global_step=step, last_layer=self.vae.get_last_layer(),
        )

        if step < self.disc_start:
            return reconstruct_loss, None, rec_log

        # The loss to train the discriminator
        gan_loss, gan_log = self.loss(batch_x, reconstruct, posterior, optimizer_idx=1, 
            global_step=step, last_layer=self.vae.get_last_layer(),
        )

        loss_log = {**rec_log, **gan_log}

        return reconstruct_loss, gan_loss, loss_log

    def encode(self, x, sample=False, is_init_image=True, 
            temporal_chunk=False, window_size=16, tile_sample_min_size=256,):
        # x: (B, C, T, H, W) or (B, C, H, W)
        B = x.shape[0]
        xdim = x.ndim

        if xdim == 4:
            # The input is an image
            x = x.unsqueeze(2)

        if sample:
            x = self.vae.encode(
                x, is_init_image=is_init_image, temporal_chunk=temporal_chunk,
                window_size=window_size, tile_sample_min_size=tile_sample_min_size,
            ).latent_dist.sample()
        else:
            x = self.vae.encode(
                x, is_init_image=is_init_image, temporal_chunk=temporal_chunk,
                window_size=window_size, tile_sample_min_size=tile_sample_min_size,
            ).latent_dist.mode()

        return x

    def decode(self, x, is_init_image=True, temporal_chunk=False, 
            window_size=2, tile_sample_min_size=256,):
        # x: (B, C, T, H, W) or (B, C, H, W)
        B = x.shape[0]
        xdim = x.ndim

        if xdim == 4:
            # The input is an image
            x = x.unsqueeze(2)

        x = self.vae.decode(
            x, is_init_image=is_init_image, temporal_chunk=temporal_chunk,
            window_size=window_size, tile_sample_min_size=tile_sample_min_size,
        ).sample

        return x

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def reconstruct(
        self, x, sample=False, return_latent=False, is_init_image=True, 
        temporal_chunk=False, window_size=16, tile_sample_min_size=256, **kwargs
    ):
        assert x.shape[0] == 1
        xdim = x.ndim
        encode_window_size = window_size
        decode_window_size = window_size // self.vae.downsample_scale

        # Encode
        x = self.encode(
            x, sample, is_init_image, temporal_chunk, encode_window_size, tile_sample_min_size,
        )
        encode_latent = x

        # Decode
        x = self.decode(
            x, is_init_image, temporal_chunk, decode_window_size, tile_sample_min_size
        )
        output_image = x.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)

        # Convert to PIL images
        output_image = rearrange(output_image, "B C T H W -> (B T) C H W")
        output_image = output_image.cpu().permute(0, 2, 3, 1).numpy()
        output_images = self.numpy_to_pil(output_image)

        if return_latent:
            return output_images, encode_latent
        
        return output_images

    # encode vae latent
    def encode_latent(self, x, sample=False, is_init_image=True, 
            temporal_chunk=False, window_size=16, tile_sample_min_size=256,):
        # Encode
        latent = self.encode(
            x, sample, is_init_image, temporal_chunk, window_size, tile_sample_min_size,
        )
        return latent

    # decode vae latent
    def decode_latent(self, latent, is_init_image=True, 
        temporal_chunk=False, window_size=2, tile_sample_min_size=256,):
        x = self.decode(
            latent, is_init_image, temporal_chunk, window_size, tile_sample_min_size
        )
        output_image = x.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        # Convert to PIL images
        output_image = rearrange(output_image, "B C T H W -> (B T) C H W")
        output_image = output_image.cpu().permute(0, 2, 3, 1).numpy()
        output_images = self.numpy_to_pil(output_image)
        return output_images
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype