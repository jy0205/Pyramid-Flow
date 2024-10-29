from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except:
    flash_attn_func = None
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_func = None

from trainer_misc import (
    is_sequence_parallel_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    all_to_all,
)

from .modeling_normalization import AdaLayerNormZero, AdaLayerNormContinuous, RMSNorm


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


class VarlenFlashSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

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
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

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


class SequenceParallelVarlenFlashSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

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
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

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


class VarlenSelfAttentionWithT5Mask:

    """
        For chunk stage attention without using flash attention
    """

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

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
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

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


class SequenceParallelVarlenSelfAttentionWithT5Mask:
    """
        For chunk stage attention without using flash attention
    """

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

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
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

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


class JointAttention(nn.Module):
    
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
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only=None,
        use_flash_attn=True,
    ): 
        """
            Fixing the QKNorm, following the flux, norm the head dimension
        """
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.use_bias = bias
        self.dropout = dropout

        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only

        self.scale = dim_head**-0.5
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        elif qk_norm == 'rms_norm':
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
    
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)

            if qk_norm is None:
                self.norm_add_q = None
                self.norm_add_k = None
            elif qk_norm == "layer_norm":
                self.norm_add_q = nn.LayerNorm(dim_head, eps=eps)
                self.norm_add_k = nn.LayerNorm(dim_head, eps=eps)
            elif qk_norm == 'rms_norm':
                self.norm_add_q = RMSNorm(dim_head, eps=eps)
                self.norm_add_k = RMSNorm(dim_head, eps=eps)
            else:
                raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        self.use_flash_attn = use_flash_attn

        if flash_attn_func is None:
            self.use_flash_attn = False

        # print(f"Using flash-attention: {self.use_flash_attn}")
        if self.use_flash_attn:
            if is_sequence_parallel_initialized():
                self.var_flash_attn = SequenceParallelVarlenFlashSelfAttentionWithT5Mask()
            else:
                self.var_flash_attn = VarlenFlashSelfAttentionWithT5Mask()
        else:
            if is_sequence_parallel_initialized():
                self.var_len_attn = SequenceParallelVarlenSelfAttentionWithT5Mask()
            else:
                self.var_len_attn = VarlenSelfAttentionWithT5Mask()
    

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,   # [B, L, S]
        hidden_length: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        # This function is only used during training
        # `sample` projections.
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(query.shape[0], -1, self.heads, head_dim)
        key = key.view(key.shape[0], -1, self.heads, head_dim)
        value = value.view(value.shape[0], -1, self.heads, head_dim)

        if self.norm_q is not None:
            query = self.norm_q(query)

        if self.norm_k is not None:
            key = self.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            encoder_hidden_states_query_proj.shape[0], -1, self.heads, head_dim
        )
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            encoder_hidden_states_key_proj.shape[0], -1, self.heads, head_dim
        )
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            encoder_hidden_states_value_proj.shape[0], -1, self.heads, head_dim
        )

        if self.norm_add_q is not None:
            encoder_hidden_states_query_proj = self.norm_add_q(encoder_hidden_states_query_proj)

        if self.norm_add_k is not None:
            encoder_hidden_states_key_proj = self.norm_add_k(encoder_hidden_states_key_proj)

        # To cat the hidden and encoder hidden, perform attention compuataion, and then split
        if self.use_flash_attn:
            hidden_states, encoder_hidden_states = self.var_flash_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, self.heads, self.scale, hidden_length, 
                image_rotary_emb, encoder_attention_mask,
            )
        else:
            hidden_states, encoder_hidden_states = self.var_len_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, self.heads, self.scale, hidden_length, 
                image_rotary_emb, attention_mask,
            )

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        if not self.context_pre_only:
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class JointTransformerBlock(nn.Module):
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

    def __init__(
        self, dim, num_attention_heads, attention_head_dim, qk_norm=None, 
        context_pre_only=False, use_flash_attn=True,
    ):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        self.attn = JointAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim // num_attention_heads,
            heads=num_attention_heads,
            out_dim=attention_head_dim,
            qk_norm=qk_norm,
            context_pre_only=context_pre_only,
            bias=True,
            use_flash_attn=use_flash_attn,
        )
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, 
        encoder_attention_mask: torch.FloatTensor, temb: torch.FloatTensor, 
        attention_mask: torch.FloatTensor = None, hidden_length: List = None, 
        image_rotary_emb: torch.FloatTensor = None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb, hidden_length=hidden_length)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb,
            )

        # Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, attention_mask=attention_mask, 
            hidden_length=hidden_length, image_rotary_emb=image_rotary_emb,
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
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states