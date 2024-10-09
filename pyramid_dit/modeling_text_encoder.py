import torch
import torch.nn as nn
import os

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from typing import Any, Callable, Dict, List, Optional, Union


class SD3TextEncoderWithMask(nn.Module):
    def __init__(self, model_path, torch_dtype):
        super().__init__()
        # CLIP-L
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(os.path.join(model_path, 'text_encoder'), torch_dtype=torch_dtype)

        # CLIP-G
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer_2'))
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(model_path, 'text_encoder_2'), torch_dtype=torch_dtype)

        # T5
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer_3'))
        self.text_encoder_3 = T5EncoderModel.from_pretrained(os.path.join(model_path, 'text_encoder_3'), torch_dtype=torch_dtype)
    
        self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 128,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)
        prompt_embeds = self.text_encoder_3(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return pooled_prompt_embeds

    def encode_prompt(self, 
        prompt, 
        num_images_per_prompt=1, 
        clip_skip: Optional[int] = None,
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=0,
        )
        pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=1,
        )
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
        )
        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds

    def forward(self, input_prompts, device):
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.encode_prompt(input_prompts, 1, clip_skip=None, device=device)

        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds