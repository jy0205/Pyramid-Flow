from typing import Any, Dict, List, Optional, Union

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version

from .modeling_normalization import AdaLayerNormContinuous
from .modeling_embedding import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from .modeling_flux_block import FluxTransformerBlock, FluxSingleTransformerBlock

from trainer_misc import (
    is_sequence_parallel_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    all_to_all,
)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)


class PyramidFluxTransformer(ModelMixin, ConfigMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        axes_dims_rope: List[int] = [16, 24, 24],
        use_flash_attn: bool = False,
        use_temporal_causal: bool = True,
        interp_condition_pos: bool = True,
        use_gradient_checkpointing: bool = False,
        gradient_checkpointing_ratio: float = 0.6,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=axes_dims_rope)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = use_gradient_checkpointing
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio

        self.use_temporal_causal = use_temporal_causal
        if self.use_temporal_causal:
            print("Using temporal causal attention")

        self.use_flash_attn = use_flash_attn
        if self.use_flash_attn:
            print("Using Flash attention")

        self.patch_size = 2   # hard-code for now

        # init weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize all the conditioning to normal init
        nn.init.normal_(self.time_text_embed.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.time_text_embed.timestep_embedder.linear_2.weight, std=0.02)
        nn.init.normal_(self.time_text_embed.text_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.time_text_embed.text_embedder.linear_2.weight, std=0.02)
        nn.init.normal_(self.context_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm1_context.linear.weight, 0)
            nn.init.constant_(block.norm1_context.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.norm.linear.weight, 0)
            nn.init.constant_(block.norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    @torch.no_grad()
    def _prepare_image_ids(self, batch_size, temp, height, width, train_height, train_width, device, start_time_stamp=0):
        latent_image_ids = torch.zeros(temp, height, width, 3)

        # Temporal Rope
        latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(start_time_stamp, start_time_stamp + temp)[:, None, None]

        # height Rope
        if height != train_height:
            height_pos = F.interpolate(torch.arange(train_height)[None, None, :].float(), height, mode='linear').squeeze(0, 1)
        else:
            height_pos = torch.arange(train_height).float()

        latent_image_ids[..., 1] = latent_image_ids[..., 1] + height_pos[None, :, None]

        # width rope
        if width != train_width:
            width_pos = F.interpolate(torch.arange(train_width)[None, None, :].float(), width, mode='linear').squeeze(0, 1)
        else:
            width_pos = torch.arange(train_width).float()

        latent_image_ids[..., 2] = latent_image_ids[..., 2] + width_pos[None, None, :]

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
        latent_image_ids = rearrange(latent_image_ids, 'b t h w c -> b (t h w) c')

        return latent_image_ids.to(device=device)

    @torch.no_grad()
    def _prepare_pyramid_image_ids(self, sample, batch_size, device):
        image_ids_list = []

        for i_b, sample_ in enumerate(sample):
            if not isinstance(sample_, list):
                sample_ = [sample_]

            cur_image_ids = []
            start_time_stamp = 0

            train_height = sample_[-1].shape[-2] // self.patch_size
            train_width = sample_[-1].shape[-1] // self.patch_size

            for clip_ in sample_:
                _, _, temp, height, width = clip_.shape
                height = height // self.patch_size
                width = width // self.patch_size
                cur_image_ids.append(self._prepare_image_ids(batch_size, temp, height, width, train_height, train_width, device, start_time_stamp=start_time_stamp))
                start_time_stamp += temp

            cur_image_ids = torch.cat(cur_image_ids, dim=1)
            image_ids_list.append(cur_image_ids)

        return image_ids_list

    def merge_input(self, sample, encoder_hidden_length, encoder_attention_mask):
        """
            Merge the input video with different resolutions into one sequence
            Sample: From low resolution to high resolution
        """
        if isinstance(sample[0], list):
            device = sample[0][-1].device
            pad_batch_size = sample[0][-1].shape[0]
        else:
            device = sample[0].device
            pad_batch_size = sample[0].shape[0]

        num_stages = len(sample)
        height_list = [];width_list = [];temp_list = []
        trainable_token_list = []

        for i_b, sample_ in enumerate(sample):
            if isinstance(sample_, list):
                sample_ = sample_[-1]
            _, _, temp, height, width = sample_.shape
            height = height // self.patch_size
            width = width // self.patch_size
            temp_list.append(temp)
            height_list.append(height)
            width_list.append(width)
            trainable_token_list.append(height * width * temp)

        # prepare the RoPE IDs, 
        image_ids_list = self._prepare_pyramid_image_ids(sample, pad_batch_size, device)
        text_ids = torch.zeros(pad_batch_size, encoder_attention_mask.shape[1], 3).to(device=device)
        input_ids_list = [torch.cat([text_ids, image_ids], dim=1) for image_ids in image_ids_list]
        image_rotary_emb = [self.pos_embed(input_ids) for input_ids in input_ids_list]  # [bs, seq_len, 1, head_dim // 2, 2, 2]

        if is_sequence_parallel_initialized():
            sp_group = get_sequence_parallel_group()
            sp_group_size = get_sequence_parallel_world_size()
            concat_output = True if self.training else False
            image_rotary_emb = [all_to_all(x_.repeat(1, 1, sp_group_size, 1, 1, 1), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output) for x_ in image_rotary_emb]
            input_ids_list = [all_to_all(input_ids.repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output) for input_ids in input_ids_list]

        hidden_states, hidden_length = [], []
    
        for sample_ in sample:
            video_tokens = []

            for each_latent in sample_:
                each_latent = rearrange(each_latent, 'b c t h w -> b t h w c')
                each_latent = rearrange(each_latent, 'b t (h p1) (w p2) c -> b (t h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
                video_tokens.append(each_latent)

            video_tokens = torch.cat(video_tokens, dim=1)
            video_tokens = self.x_embedder(video_tokens)
            hidden_states.append(video_tokens)
            hidden_length.append(video_tokens.shape[1])

        # prepare the attention mask
        if self.use_flash_attn:
            attention_mask = None
            indices_list = []
            for i_p, length in enumerate(hidden_length):
                pad_attention_mask = torch.ones((pad_batch_size, length), dtype=encoder_attention_mask.dtype).to(device)
                pad_attention_mask = torch.cat([encoder_attention_mask[i_p::num_stages], pad_attention_mask], dim=1)
                
                if is_sequence_parallel_initialized():
                    sp_group = get_sequence_parallel_group()
                    sp_group_size = get_sequence_parallel_world_size()
                    pad_attention_mask = all_to_all(pad_attention_mask.unsqueeze(2).repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0)
                    pad_attention_mask = pad_attention_mask.squeeze(2)

                seqlens_in_batch = pad_attention_mask.sum(dim=-1, dtype=torch.int32)
                indices = torch.nonzero(pad_attention_mask.flatten(), as_tuple=False).flatten()

                indices_list.append(
                    {
                        'indices': indices,
                        'seqlens_in_batch': seqlens_in_batch,
                    }
                )
            encoder_attention_mask = indices_list
        else:
            assert encoder_attention_mask.shape[1] == encoder_hidden_length
            real_batch_size = encoder_attention_mask.shape[0]

            # prepare text ids
            text_ids = torch.arange(1, real_batch_size + 1, dtype=encoder_attention_mask.dtype).unsqueeze(1).repeat(1, encoder_hidden_length)
            text_ids = text_ids.to(device)
            text_ids[encoder_attention_mask == 0] = 0

            # prepare image ids
            image_ids = torch.arange(1, real_batch_size + 1, dtype=encoder_attention_mask.dtype).unsqueeze(1).repeat(1, max(hidden_length))
            image_ids = image_ids.to(device)
            image_ids_list = []
            for i_p, length in enumerate(hidden_length):
                image_ids_list.append(image_ids[i_p::num_stages][:, :length])

            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()
                concat_output = True if self.training else False
                text_ids = all_to_all(text_ids.unsqueeze(2).repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output).squeeze(2)
                image_ids_list = [all_to_all(image_ids_.unsqueeze(2).repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output).squeeze(2) for image_ids_ in image_ids_list]
            
            attention_mask = []
            for i_p in range(len(hidden_length)):
                image_ids = image_ids_list[i_p]
                token_ids = torch.cat([text_ids[i_p::num_stages], image_ids], dim=1)
                stage_attention_mask = rearrange(token_ids, 'b i -> b 1 i 1') == rearrange(token_ids, 'b j -> b 1 1 j')  # [bs, 1, q_len, k_len]
                if self.use_temporal_causal:
                    input_order_ids = input_ids_list[i_p][:,:,0]
                    temporal_causal_mask = rearrange(input_order_ids, 'b i -> b 1 i 1') >= rearrange(input_order_ids, 'b j -> b 1 1 j')
                    stage_attention_mask = stage_attention_mask & temporal_causal_mask
                attention_mask.append(stage_attention_mask)

        return hidden_states, hidden_length, temp_list, height_list, width_list, trainable_token_list, encoder_attention_mask, attention_mask, image_rotary_emb

    def split_output(self, batch_hidden_states, hidden_length, temps, heights, widths, trainable_token_list):
        # To split the hidden states
        batch_size = batch_hidden_states.shape[0]
        output_hidden_list = []
        batch_hidden_states = torch.split(batch_hidden_states, hidden_length, dim=1)

        if is_sequence_parallel_initialized():
            sp_group_size = get_sequence_parallel_world_size()
            if self.training:
                batch_size = batch_size // sp_group_size

        for i_p, length in enumerate(hidden_length):
            width, height, temp = widths[i_p], heights[i_p], temps[i_p]
            trainable_token_num = trainable_token_list[i_p]
            hidden_states = batch_hidden_states[i_p]

            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()

                if not self.training:
                    hidden_states = hidden_states.repeat(sp_group_size, 1, 1)

                hidden_states = all_to_all(hidden_states, sp_group, sp_group_size, scatter_dim=0, gather_dim=1)

            # only the trainable token are taking part in loss computation
            hidden_states = hidden_states[:, -trainable_token_num:]

            # unpatchify
            hidden_states = hidden_states.reshape(
                shape=(batch_size, temp, height, width, self.patch_size, self.patch_size, self.out_channels // 4)
            )
            hidden_states = rearrange(hidden_states, "b t h w p1 p2 c -> b t (h p1) (w p2) c")
            hidden_states = rearrange(hidden_states, "b t h w c -> b c t h w")
            output_hidden_list.append(hidden_states)

        return output_hidden_list

    def forward(
        self,
        sample: torch.FloatTensor, # [num_stages]
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        pooled_projections: torch.Tensor = None,
        timestep_ratio: torch.LongTensor = None,
    ):
        temb = self.time_text_embed(timestep_ratio, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_length = encoder_hidden_states.shape[1]

        # Get the input sequence
        hidden_states, hidden_length, temps, heights, widths, trainable_token_list, encoder_attention_mask, attention_mask, \
                image_rotary_emb = self.merge_input(sample, encoder_hidden_length, encoder_attention_mask)
        
        # split the long latents if necessary
        if is_sequence_parallel_initialized():
            sp_group = get_sequence_parallel_group()
            sp_group_size = get_sequence_parallel_world_size()
            concat_output = True if self.training else False
            
            # sync the input hidden states
            batch_hidden_states = []
            for i_p, hidden_states_ in enumerate(hidden_states):
                assert hidden_states_.shape[1] % sp_group_size == 0, "The sequence length should be divided by sequence parallel size"
                hidden_states_ = all_to_all(hidden_states_, sp_group, sp_group_size, scatter_dim=1, gather_dim=0, concat_output=concat_output)
                hidden_length[i_p] = hidden_length[i_p] // sp_group_size
                batch_hidden_states.append(hidden_states_)

            # sync the encoder hidden states
            hidden_states = torch.cat(batch_hidden_states, dim=1)
            encoder_hidden_states = all_to_all(encoder_hidden_states, sp_group, sp_group_size, scatter_dim=1, gather_dim=0, concat_output=concat_output)
            temb = all_to_all(temb.unsqueeze(1).repeat(1, sp_group_size, 1), sp_group, sp_group_size, scatter_dim=1, gather_dim=0, concat_output=concat_output)
            temb = temb.squeeze(1)
        else:
            hidden_states = torch.cat(hidden_states, dim=1)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing and (index_block <= int(len(self.transformer_blocks) * self.gradient_checkpointing_ratio)):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    temb,
                    attention_mask,
                    hidden_length,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                    attention_mask=attention_mask,
                    hidden_length=hidden_length,
                    image_rotary_emb=image_rotary_emb,
                )

        # remerge for single attention block
        num_stages = len(hidden_length)
        batch_hidden_states = list(torch.split(hidden_states, hidden_length, dim=1))
        concat_hidden_length = []

        if is_sequence_parallel_initialized():
            sp_group = get_sequence_parallel_group()
            sp_group_size = get_sequence_parallel_world_size()
            encoder_hidden_states = all_to_all(encoder_hidden_states, sp_group, sp_group_size, scatter_dim=0, gather_dim=1)

        for i_p in range(len(hidden_length)):

            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()
                batch_hidden_states[i_p] = all_to_all(batch_hidden_states[i_p], sp_group, sp_group_size, scatter_dim=0, gather_dim=1)

            batch_hidden_states[i_p] = torch.cat([encoder_hidden_states[i_p::num_stages], batch_hidden_states[i_p]], dim=1)

            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()
                batch_hidden_states[i_p] = all_to_all(batch_hidden_states[i_p], sp_group, sp_group_size, scatter_dim=1, gather_dim=0)

            concat_hidden_length.append(batch_hidden_states[i_p].shape[1])

        hidden_states = torch.cat(batch_hidden_states, dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing and (index_block <= int(len(self.single_transformer_blocks) * self.gradient_checkpointing_ratio)):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    encoder_attention_mask,
                    attention_mask,
                    concat_hidden_length,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    encoder_attention_mask=encoder_attention_mask,      # used for 
                    attention_mask=attention_mask,
                    hidden_length=concat_hidden_length,
                    image_rotary_emb=image_rotary_emb,
                )

        batch_hidden_states = list(torch.split(hidden_states, concat_hidden_length, dim=1))

        for i_p in range(len(concat_hidden_length)):
            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()
                batch_hidden_states[i_p] = all_to_all(batch_hidden_states[i_p], sp_group, sp_group_size, scatter_dim=0, gather_dim=1)
            
            batch_hidden_states[i_p] = batch_hidden_states[i_p][:, encoder_hidden_length :, ...]

            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()
                batch_hidden_states[i_p] = all_to_all(batch_hidden_states[i_p], sp_group, sp_group_size, scatter_dim=1, gather_dim=0)
            
        hidden_states = torch.cat(batch_hidden_states, dim=1)
        hidden_states = self.norm_out(hidden_states, temb, hidden_length=hidden_length)
        hidden_states = self.proj_out(hidden_states)

        output = self.split_output(hidden_states, hidden_length, temps, heights, widths, trainable_token_list)

        return output