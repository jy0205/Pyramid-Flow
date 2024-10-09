from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import numpy as np
import math

from diffusers.models.activations import get_activation
from einops import rearrange


def get_1d_sincos_pos_embed(
    embed_dim, num_frames, cls_token=False, extra_tokens=0,
):
    t = np.arange(num_frames, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, t)  # (T, D)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        sample_proj_bias=True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, act_fn="silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = get_activation(act_fn)
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepConditionEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)
        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = timesteps_emb + pooled_projections
        return conditioning


class CombinedTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class PatchEmbed3D(nn.Module):
    """Support the 3D Tensor input"""

    def __init__(
        self,
        height=128,
        width=128,
        patch_size=2,
        in_channels=16,
        embed_dim=1536,
        layer_norm=False,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        temp_pos_embed_type='rope',
        pos_embed_max_size=192,   # For SD3 cropping
        max_num_frames=64,
        add_temp_pos_embed=False,
        interp_condition_pos=False,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        self.add_temp_pos_embed = add_temp_pos_embed

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None

        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)

            if add_temp_pos_embed and temp_pos_embed_type == 'sincos':
                time_pos_embed = get_1d_sincos_pos_embed(embed_dim, max_num_frames)
                self.register_buffer("temp_pos_embed", torch.from_numpy(time_pos_embed).float().unsqueeze(0), persistent=True)

        elif pos_embed_type == "rope":
            print("Using the rotary position embedding")

        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

        self.pos_embed_type = pos_embed_type
        self.temp_pos_embed_type = temp_pos_embed_type
        self.interp_condition_pos = interp_condition_pos

    def cropped_pos_embed(self, height, width, ori_height, ori_width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        ori_height = ori_height // self.patch_size
        ori_width = ori_width // self.patch_size

        assert ori_height >= height, "The ori_height needs >= height"
        assert ori_width >= width, "The ori_width needs >= width"

        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        if self.interp_condition_pos:
            top = (self.pos_embed_max_size - ori_height) // 2
            left = (self.pos_embed_max_size - ori_width) // 2
            spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
            spatial_pos_embed = spatial_pos_embed[:, top : top + ori_height, left : left + ori_width, :]   # [b h w c]
            if ori_height != height or ori_width != width:
                spatial_pos_embed = spatial_pos_embed.permute(0, 3, 1, 2)
                spatial_pos_embed = torch.nn.functional.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear')
                spatial_pos_embed = spatial_pos_embed.permute(0, 2, 3, 1)
        else:
            top = (self.pos_embed_max_size - height) // 2
            left = (self.pos_embed_max_size - width) // 2
            spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
            spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

        return spatial_pos_embed

    def forward_func(self, latent, time_index=0, ori_height=None, ori_width=None):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        bs = latent.shape[0]
        temp = latent.shape[2]

        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # (BT)CHW -> (BT)NC

        if self.layer_norm:
            latent = self.norm(latent)

        if self.pos_embed_type == 'sincos':
            # Spatial position embedding, Interpolate or crop positional embeddings as needed
            if self.pos_embed_max_size:
                pos_embed = self.cropped_pos_embed(height, width, ori_height, ori_width)
            else:
                raise NotImplementedError("Not implemented sincos pos embed without sd3 max pos crop")
                if self.height != height or self.width != width:
                    pos_embed = get_2d_sincos_pos_embed(
                        embed_dim=self.pos_embed.shape[-1],
                        grid_size=(height, width),
                        base_size=self.base_size,
                        interpolation_scale=self.interpolation_scale,
                    )
                    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
                else:
                    pos_embed = self.pos_embed

            if self.add_temp_pos_embed and self.temp_pos_embed_type == 'sincos':
                latent_dtype = latent.dtype
                latent = latent + pos_embed
                latent = rearrange(latent, '(b t) n c -> (b n) t c', t=temp)
                latent = latent + self.temp_pos_embed[:, time_index:time_index + temp, :]
                latent = latent.to(latent_dtype)
                latent = rearrange(latent, '(b n) t c -> b t n c', b=bs)
            else:
                latent = (latent + pos_embed).to(latent.dtype)
                latent = rearrange(latent, '(b t) n c -> b t n c', b=bs, t=temp)

        else:
            assert self.pos_embed_type == "rope", "Only supporting the sincos and rope embedding"
            latent = rearrange(latent, '(b t) n c -> b t n c', b=bs, t=temp)
        
        return latent

    def forward(self, latent):
        """
        Arguments:
            past_condition_latents (Torch.FloatTensor): The past latent during the generation
            flatten_input (bool): True indicate flatten the latent into 1D sequence
        """

        if isinstance(latent, list):
            output_list = []
            
            for latent_ in latent:
                if not isinstance(latent_, list):
                    latent_ = [latent_]

                output_latent = []
                time_index = 0
                ori_height, ori_width = latent_[-1].shape[-2:]
                for each_latent in latent_:
                    hidden_state = self.forward_func(each_latent, time_index=time_index, ori_height=ori_height, ori_width=ori_width)
                    time_index += each_latent.shape[2]
                    hidden_state = rearrange(hidden_state, "b t n c -> b (t n) c")
                    output_latent.append(hidden_state)

                output_latent = torch.cat(output_latent, dim=1)
                output_list.append(output_latent)

            return output_list
        else:
            hidden_states = self.forward_func(latent)
            hidden_states = rearrange(hidden_states, "b t n c -> b (t n) c")
            return hidden_states