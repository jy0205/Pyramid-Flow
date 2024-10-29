from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from einops import rearrange

from diffusers.utils import deprecate
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, SwiGLU

from .modeling_normalization import (
    AdaLayerNormContinuous, AdaLayerNormZero, 
    AdaLayerNormZeroSingle, FP32LayerNorm, RMSNorm
)

from trainer_misc import (
    is_sequence_parallel_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    all_to_all,
)

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except:
    flash_attn_func = None
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_func = None


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class SequenceParallelVarlenFlashSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        qkv_list = []
        num_stages = len(hidden_length)

        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        # To sync the encoder query, key and values
        sp_group = get_sequence_parallel_group()
        sp_group_size = get_sequence_parallel_world_size()
        encoder_qkv = all_to_all(encoder_qkv, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]

        output_hidden = torch.zeros_like(qkv[:,:,0])
        output_encoder_hidden = torch.zeros_like(encoder_qkv[:,:,0])
        encoder_length = encoder_qkv.shape[1]
        
        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            # get the query, key, value from padding sequence
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            qkv_tokens = all_to_all(qkv_tokens, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, pad_seq, 3, nhead, dim]

            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(concat_qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x_.shape[0] for x_ in qkv_list]
        qkv = torch.cat(qkv_list, dim=0)
        query, key, value = qkv.unbind(1)

        cu_seqlens = torch.cat([x_['seqlens_in_batch'] for x_ in encoder_attention_mask], dim=0)
        max_seqlen_q = cu_seqlens.max().item()
        max_seqlen_k = max_seqlen_q
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = cu_seqlens_q.clone()

        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )

        # To merge the tokens
        i_sum = 0;token_sum = 0
        for i_p, length in enumerate(hidden_length):
            tot_token_num = token_lengths[i_p]
            stage_output = output[token_sum : token_sum + tot_token_num]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, encoder_length + length * sp_group_size)
            stage_encoder_hidden_output = stage_output[:, :encoder_length]
            stage_hidden_output = stage_output[:, encoder_length:]
            stage_hidden_output = all_to_all(stage_hidden_output, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
            output_hidden[:, i_sum:i_sum+length] = stage_hidden_output
            output_encoder_hidden[i_p::num_stages] = stage_encoder_hidden_output
            token_sum += tot_token_num
            i_sum += length

        output_encoder_hidden = all_to_all(output_encoder_hidden, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
        output_hidden = output_hidden.flatten(2, 3)
        output_encoder_hidden = output_encoder_hidden.flatten(2, 3)

        return output_hidden, output_encoder_hidden


class VarlenFlashSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        output_hidden = torch.zeros_like(query)
        output_encoder_hidden = torch.zeros_like(encoder_query)
        encoder_length = encoder_query.shape[1]

        qkv_list = []
        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(concat_qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x_.shape[0] for x_ in qkv_list]
        qkv = torch.cat(qkv_list, dim=0)
        query, key, value = qkv.unbind(1)

        cu_seqlens = torch.cat([x_['seqlens_in_batch'] for x_ in encoder_attention_mask], dim=0)
        max_seqlen_q = cu_seqlens.max().item()
        max_seqlen_k = max_seqlen_q
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = cu_seqlens_q.clone()

        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )

        # To merge the tokens
        i_sum = 0;token_sum = 0
        for i_p, length in enumerate(hidden_length):
            tot_token_num = token_lengths[i_p]
            stage_output = output[token_sum : token_sum + tot_token_num]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, encoder_length + length)
            stage_encoder_hidden_output = stage_output[:, :encoder_length]
            stage_hidden_output = stage_output[:, encoder_length:]   
            output_hidden[:, i_sum:i_sum+length] = stage_hidden_output
            output_encoder_hidden[i_p::num_stages] = stage_encoder_hidden_output
            token_sum += tot_token_num
            i_sum += length

        output_hidden = output_hidden.flatten(2, 3)
        output_encoder_hidden = output_encoder_hidden.flatten(2, 3)

        return output_hidden, output_encoder_hidden


class SequenceParallelVarlenSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        # To sync the encoder query, key and values
        sp_group = get_sequence_parallel_group()
        sp_group_size = get_sequence_parallel_world_size()
        encoder_qkv = all_to_all(encoder_qkv, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]
        encoder_length = encoder_qkv.shape[1]

        i_sum = 0
        output_encoder_hidden_list = []
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            qkv_tokens = all_to_all(qkv_tokens, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = concat_qkv_tokens.unbind(2)   # [bs, tot_seq, nhead, dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2)   # [bs, tot_seq, nhead, dim]

            output_encoder_hidden_list.append(stage_hidden_states[:, :encoder_length])

            output_hidden = stage_hidden_states[:, encoder_length:]
            output_hidden = all_to_all(output_hidden, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
            output_hidden_list.append(output_hidden)

            i_sum += length

        output_encoder_hidden = torch.stack(output_encoder_hidden_list, dim=1)  # [b n s nhead d]
        output_encoder_hidden = rearrange(output_encoder_hidden, 'b n s h d -> (b n) s h d')
        output_encoder_hidden = all_to_all(output_encoder_hidden, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
        output_encoder_hidden = output_encoder_hidden.flatten(2, 3)
        output_hidden = torch.cat(output_hidden_list, dim=1).flatten(2, 3)

        return output_hidden, output_encoder_hidden


class VarlenSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        encoder_length = encoder_query.shape[1]
        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        output_encoder_hidden_list = []
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = concat_qkv_tokens.unbind(2)   # [bs, tot_seq, nhead, dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2).flatten(2, 3)   # [bs, tot_seq, dim]

            output_encoder_hidden_list.append(stage_hidden_states[:, :encoder_length])
            output_hidden_list.append(stage_hidden_states[:, encoder_length:])
            i_sum += length

        output_encoder_hidden = torch.stack(output_encoder_hidden_list, dim=1)  # [b n s d]
        output_encoder_hidden = rearrange(output_encoder_hidden, 'b n s d -> (b n) s d')
        output_hidden = torch.cat(output_hidden_list, dim=1)

        return output_hidden, output_encoder_hidden


class SequenceParallelVarlenFlashAttnSingle:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, heads, scale, 
            hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        qkv_list = []
        num_stages = len(hidden_length)

        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        output_hidden = torch.zeros_like(qkv[:,:,0])

        sp_group = get_sequence_parallel_group()
        sp_group_size = get_sequence_parallel_world_size()
    
        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            # get the query, key, value from padding sequence
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            qkv_tokens = all_to_all(qkv_tokens, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]

            if image_rotary_emb is not None:
                qkv_tokens[:,:,0], qkv_tokens[:,:,1] = apply_rope(qkv_tokens[:,:,0], qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x_.shape[0] for x_ in qkv_list]
        qkv = torch.cat(qkv_list, dim=0)
        query, key, value = qkv.unbind(1)

        cu_seqlens = torch.cat([x_['seqlens_in_batch'] for x_ in encoder_attention_mask], dim=0)
        max_seqlen_q = cu_seqlens.max().item()
        max_seqlen_k = max_seqlen_q
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = cu_seqlens_q.clone()

        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )

        # To merge the tokens
        i_sum = 0;token_sum = 0
        for i_p, length in enumerate(hidden_length):
            tot_token_num = token_lengths[i_p]
            stage_output = output[token_sum : token_sum + tot_token_num]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, length * sp_group_size)
            stage_hidden_output = all_to_all(stage_output, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
            output_hidden[:, i_sum:i_sum+length] = stage_hidden_output
            token_sum += tot_token_num
            i_sum += length

        output_hidden = output_hidden.flatten(2, 3)

        return output_hidden


class VarlenFlashSelfAttnSingle:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, heads, scale, 
            hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        output_hidden = torch.zeros_like(query)

        qkv_list = []
        num_stages = len(hidden_length)        
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            qkv_tokens = qkv[:, i_sum:i_sum+length]

            if image_rotary_emb is not None:
                qkv_tokens[:,:,0], qkv_tokens[:,:,1] = apply_rope(qkv_tokens[:,:,0], qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x_.shape[0] for x_ in qkv_list]
        qkv = torch.cat(qkv_list, dim=0)
        query, key, value = qkv.unbind(1)

        cu_seqlens = torch.cat([x_['seqlens_in_batch'] for x_ in encoder_attention_mask], dim=0)
        max_seqlen_q = cu_seqlens.max().item()
        max_seqlen_k = max_seqlen_q
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = cu_seqlens_q.clone()

        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )

        # To merge the tokens
        i_sum = 0;token_sum = 0
        for i_p, length in enumerate(hidden_length):
            tot_token_num = token_lengths[i_p]
            stage_output = output[token_sum : token_sum + tot_token_num]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, length)
            output_hidden[:, i_sum:i_sum+length] = stage_output
            token_sum += tot_token_num
            i_sum += length

        output_hidden = output_hidden.flatten(2, 3)

        return output_hidden


class SequenceParallelVarlenAttnSingle:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, heads, scale, 
            hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        num_stages = len(hidden_length)        
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        # To sync the encoder query, key and values
        sp_group = get_sequence_parallel_group()
        sp_group_size = get_sequence_parallel_world_size()

        i_sum = 0
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            qkv_tokens = all_to_all(qkv_tokens, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]
            
            if image_rotary_emb is not None:
                qkv_tokens[:,:,0], qkv_tokens[:,:,1] = apply_rope(qkv_tokens[:,:,0], qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = qkv_tokens.unbind(2)   # [bs, tot_seq, nhead, dim]
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2)   # [bs, tot_seq, nhead, dim]

            output_hidden = stage_hidden_states
            output_hidden = all_to_all(output_hidden, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
            output_hidden_list.append(output_hidden)

            i_sum += length

        output_hidden = torch.cat(output_hidden_list, dim=1).flatten(2, 3)

        return output_hidden


class VarlenSelfAttnSingle:

    def __init__(self):
        pass

    def __call__(
            self, query, key, value, heads, scale, 
            hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        num_stages = len(hidden_length)        
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            
            if image_rotary_emb is not None:
                qkv_tokens[:,:,0], qkv_tokens[:,:,1] = apply_rope(qkv_tokens[:,:,0], qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = qkv_tokens.unbind(2)
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2).flatten(2, 3)   # [bs, tot_seq, dim]

            output_hidden_list.append(stage_hidden_states)
            i_sum += length

        output_hidden = torch.cat(output_hidden_list, dim=1)

        return output_hidden


class Attention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.query_dim = query_dim
        self.use_bias = bias
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        self.scale = dim_head**-0.5
        self.heads = out_dim // dim_head if out_dim is not None else heads


        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # set attention processor
        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_length: List = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            hidden_length=hidden_length,
            image_rotary_emb=image_rotary_emb,
        )


class FluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(self, use_flash_attn=False):
        self.use_flash_attn = use_flash_attn

        if self.use_flash_attn:
            if is_sequence_parallel_initialized():
                self.varlen_flash_attn = SequenceParallelVarlenFlashAttnSingle()
            else:
                self.varlen_flash_attn = VarlenFlashSelfAttnSingle()
        else:
            if is_sequence_parallel_initialized():
                self.varlen_attn = SequenceParallelVarlenAttnSingle()
            else:
                self.varlen_attn = VarlenSelfAttnSingle()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        hidden_length: List = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(query.shape[0], -1, attn.heads, head_dim)
        key = key.view(key.shape[0], -1, attn.heads, head_dim)
        value = value.view(value.shape[0], -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.use_flash_attn:
            hidden_states = self.varlen_flash_attn(
                query, key, value, 
                attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, encoder_attention_mask,
            )
        else:
            hidden_states = self.varlen_attn(
                query, key, value, 
                attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, attention_mask,
            )

        return hidden_states


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, use_flash_attn=False):
        self.use_flash_attn = use_flash_attn

        if self.use_flash_attn:
            if is_sequence_parallel_initialized():
                self.varlen_flash_attn = SequenceParallelVarlenFlashSelfAttentionWithT5Mask()
            else:
                self.varlen_flash_attn = VarlenFlashSelfAttentionWithT5Mask()
        else:
            if is_sequence_parallel_initialized():
                self.varlen_attn = SequenceParallelVarlenSelfAttentionWithT5Mask()
            else:
                self.varlen_attn = VarlenSelfAttentionWithT5Mask()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        hidden_length: List = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(query.shape[0], -1, attn.heads, head_dim)
        key = key.view(key.shape[0], -1, attn.heads, head_dim)
        value = value.view(value.shape[0], -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            encoder_hidden_states_query_proj.shape[0], -1, attn.heads, head_dim
        )
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            encoder_hidden_states_key_proj.shape[0], -1, attn.heads, head_dim
        )
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            encoder_hidden_states_value_proj.shape[0], -1, attn.heads, head_dim
        )

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        if self.use_flash_attn:
            hidden_states, encoder_hidden_states = self.varlen_flash_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, encoder_attention_mask,
            )
        else:
            hidden_states, encoder_hidden_states = self.varlen_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, attn.heads, attn.scale, hidden_length, 
                image_rotary_emb, attention_mask,
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, use_flash_attn=False):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxSingleAttnProcessor2_0(use_flash_attn)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        encoder_attention_mask=None,
        attention_mask=None,
        hidden_length=None,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb, hidden_length=hidden_length)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_attention_mask, 
            attention_mask=attention_mask,
            hidden_length=hidden_length,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6, use_flash_attn=False):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0(use_flash_attn)
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_mask: torch.FloatTensor = None,
        hidden_length: List = None,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb, hidden_length=hidden_length)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, 
            attention_mask=attention_mask,
            hidden_length=hidden_length,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states
