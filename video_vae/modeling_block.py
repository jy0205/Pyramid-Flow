# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from diffusers.utils import logging
from diffusers.models.attention_processor import Attention
from .modeling_resnet import (
    Downsample2D, ResnetBlock2D, CausalResnetBlock3D, Upsample2D,
    TemporalDownsample2x, TemporalUpsample2x,
    CausalDownsample2x, CausalTemporalDownsample2x,
    CausalUpsample2x, CausalTemporalUpsample2x,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_input_layer(
    in_channels: int,
    out_channels: int,
    norm_num_groups: int,
    layer_type: str,
    norm_type: str = 'group',
    affine: bool = True,
):
    if layer_type == 'conv':
        input_layer = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    
    elif layer_type == 'pixel_shuffle':
        input_layer = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
        )
    else:
        raise NotImplementedError(f"Not support input layer {layer_type}")

    return input_layer


def get_output_layer(
    in_channels: int,
    out_channels: int,
    norm_num_groups: int,
    layer_type: str,
    norm_type: str = 'group',
    affine: bool = True,
):
    if layer_type == 'norm_act_conv':
        output_layer = nn.Sequential(
            nn.GroupNorm(num_channels=in_channels, num_groups=norm_num_groups, eps=1e-6, affine=affine),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
        )

    elif layer_type == 'pixel_shuffle':
        output_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )
    
    else:
        raise NotImplementedError(f"Not support output layer {layer_type}")

    return output_layer


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int = None,
    temb_channels: int = None,
    add_spatial_downsample: bool = None,
    add_temporal_downsample: bool = None,
    resnet_eps: float = 1e-6,
    resnet_act_fn: str = 'silu',
    resnet_groups: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    resnet_time_scale_shift: str = "default",
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
    norm_affline: bool = True,
    norm_layer: str = 'layer',
):
    
    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_spatial_downsample=add_spatial_downsample,
            add_temporal_downsample=add_temporal_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    elif down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_spatial_downsample=add_spatial_downsample,
            add_temporal_downsample=add_temporal_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int = None,
    temb_channels: int = None,
    add_spatial_upsample: bool = None,
    add_temporal_upsample: bool = None,
    resnet_eps: float = 1e-6,
    resnet_act_fn: str = 'silu',
    resolution_idx: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    resnet_time_scale_shift: str = "default",
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
    interpolate: bool = True,
    norm_affline: bool = True,
    norm_layer: str = 'layer',
) -> nn.Module:
    
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_spatial_upsample=add_spatial_upsample,
            add_temporal_upsample=add_temporal_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
            interpolate=interpolate,
        )

    elif up_block_type == "UpDecoderBlockCausal3D":
        return UpDecoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_spatial_upsample=add_spatial_upsample,
            add_temporal_upsample=add_temporal_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
            interpolate=interpolate,
        )

    raise ValueError(f"{up_block_type} does not exist.")



class UNetMidBlock2D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                # Spatial attention
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        t = hidden_states.shape[2]

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, 'b c t h w -> b t c h w')
                hidden_states = rearrange(hidden_states, 'b t c h w -> (b t) c h w')
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(hidden_states, '(b t) c h w -> b t c h w', t=t)
                hidden_states = rearrange(hidden_states, 'b t c h w -> b c t h w')

            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CausalUNetMidBlock2D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        resnets = [
            CausalResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                # Spatial attention
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                CausalResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None,
            is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        t = hidden_states.shape[2]

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, 'b c t h w -> b t c h w')
                hidden_states = rearrange(hidden_states, 'b t c h w -> (b t) c h w')
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(hidden_states, '(b t) c h w -> b t c h w', t=t)
                hidden_states = rearrange(hidden_states, 'b t c h w -> b c t h w')

            hidden_states = resnet(hidden_states, temb, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        return hidden_states


class DownEncoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_spatial_downsample: bool = True,
        add_temporal_downsample: bool = False,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                CausalResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    CausalDownsample2x(
                        out_channels, use_conv=True, out_channels=out_channels,
                    )
                ]
            )
        else:
            self.downsamplers = None

        if add_temporal_downsample:
            self.temporal_downsamplers = nn.ModuleList(
                [
                    CausalTemporalDownsample2x(
                        out_channels, use_conv=True, out_channels=out_channels,
                    )
                ]
            )
        else:
            self.temporal_downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor, is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        if self.temporal_downsamplers is not None:
            for temporal_downsampler in self.temporal_downsamplers:
                hidden_states = temporal_downsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        return hidden_states
    

class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_spatial_downsample: bool = True,
        add_temporal_downsample: bool = False,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        if add_temporal_downsample:
            self.temporal_downsamplers = nn.ModuleList(
                [
                    TemporalDownsample2x(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding,
                    )
                ]
            )
        else:
            self.temporal_downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        if self.temporal_downsamplers is not None:
            for temporal_downsampler in self.temporal_downsamplers:
                hidden_states = temporal_downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_spatial_upsample: bool = True,
        add_temporal_upsample: bool = False,
        temb_channels: Optional[int] = None,
        interpolate: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels, interpolate=interpolate)])
        else:
            self.upsamplers = None

        if add_temporal_upsample:
            self.temporal_upsamplers = nn.ModuleList([TemporalUpsample2x(out_channels, use_conv=True, out_channels=out_channels, interpolate=interpolate)])
        else:
            self.temporal_upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None, scale: float = 1.0, is_image: bool = False,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb, scale=scale)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        
        if self.temporal_upsamplers is not None:
            for temporal_upsampler in self.temporal_upsamplers:
                hidden_states = temporal_upsampler(hidden_states, is_image=is_image)

        return hidden_states


class UpDecoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_spatial_upsample: bool = True,
        add_temporal_upsample: bool = False,
        temb_channels: Optional[int] = None,
        interpolate: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                CausalResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_upsample:
            self.upsamplers = nn.ModuleList([CausalUpsample2x(out_channels, use_conv=True, out_channels=out_channels, interpolate=interpolate)])
        else:
            self.upsamplers = None

        if add_temporal_upsample:
            self.temporal_upsamplers = nn.ModuleList([CausalTemporalUpsample2x(out_channels, use_conv=True, out_channels=out_channels, interpolate=interpolate)])
        else:
            self.temporal_upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None,
        is_init_image=True, temporal_chunk=False,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        
        if self.temporal_upsamplers is not None:
            for temporal_upsampler in self.temporal_upsamplers:
                hidden_states = temporal_upsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        return hidden_states

