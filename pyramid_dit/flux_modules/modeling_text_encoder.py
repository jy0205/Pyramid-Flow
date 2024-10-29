import torch
import torch.nn as nn
import os

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from typing import Any, Callable, Dict, List, Optional, Union


class FluxTextEncoderWithMask(nn.Module):
    def __init__(self, model_path, torch_dtype):
        super().__init__()
        # CLIP-G
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'), torch_dtype=torch_dtype)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(model_path, 'text_encoder'), torch_dtype=torch_dtype)

        # T5
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer_2'))
        self.text_encoder_2 = T5EncoderModel.from_pretrained(os.path.join(model_path, 'text_encoder_2'), torch_dtype=torch_dtype)

        self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), attention_mask=prompt_attention_mask, output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
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
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(self, 
        prompt, 
        num_images_per_prompt=1, 
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        batch_size = len(prompt)

        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )

        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
        )

        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds

    def forward(self, input_prompts, device):
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.encode_prompt(input_prompts, 1, device=device)

        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds