from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from collections import deque
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch import Tensor

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
    cp_pass_from_previous_rank,
)


def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_odd(n):
    return not divisible_by(n, 2)


class CausalGroupNorm(nn.GroupNorm):

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = super().forward(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class CausalConv3d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = 1,
            pad_mode: str ='constant',
            **kwargs
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 3)
    
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop('dilation', 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = (stride, 1, 1)
    
        time_pad = dilation * (time_kernel_size - 1)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.temporal_stride = stride[0]
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
        self.cache_front_feat = deque()

    def _clear_context_parallel_cache(self):
        del self.cache_front_feat
        self.cache_front_feat = deque()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def context_parallel_forward(self, x):
        cp_rank = get_context_parallel_rank()
        if self.time_kernel_size == 3 and ((cp_rank == 0 and x.shape[2] <= 2) or (cp_rank != 0 and x.shape[2] <= 1)):
            # This code is only for training 8 frames per GPU (except for cp_rank=0, 9 frames) with context parallel
            # If you do not have enough GPU memory, you can set the total frames = 8 * CONTEXT_SIZE + 1, enable each GPU
            # only forward 8 frames during training
            x = cp_pass_from_previous_rank(x, dim=2, kernel_size=2)   # pass one latent
            trans_x = cp_pass_from_previous_rank(x[:, :, :-1], dim=2, kernel_size=2)   # pass one latent
            x = torch.cat([trans_x, x[:, :,-1:]], dim=2)
        else:
            x = cp_pass_from_previous_rank(x, dim=2, kernel_size=self.time_kernel_size)
        
        x = F.pad(x, self.time_uncausal_padding, mode='constant')

        if cp_rank != 0:
            if self.temporal_stride == 2 and self.time_kernel_size == 3:
                x = x[:,:,1:]

        x = self.conv(x)
        return x

    def forward(self, x, is_init_image=True, temporal_chunk=False):
        # temporal_chunk: whether to use the temporal chunk

        if is_context_parallel_initialized():
            return self.context_parallel_forward(x)
        
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        if not temporal_chunk:
            x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        else:
            assert not self.training, "The feature cache should not be used in training"
            if is_init_image:
                # Encode the first chunk
                x = F.pad(x, self.time_causal_padding, mode=pad_mode)
                self._clear_context_parallel_cache()
                self.cache_front_feat.append(x[:, :, -2:].clone().detach())
            else:
                x = F.pad(x, self.time_uncausal_padding, mode=pad_mode)
                video_front_context = self.cache_front_feat.pop()
                self._clear_context_parallel_cache()

                if self.temporal_stride == 1 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context, x], dim=2)
                elif self.temporal_stride == 2 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context[:,:,-1:], x], dim=2)

                self.cache_front_feat.append(x[:, :, -2:].clone().detach())
        
        x = self.conv(x)
        return x
