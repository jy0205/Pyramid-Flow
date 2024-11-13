from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)

from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from .modeling_enc_dec import (
    DecoderOutput, DiagonalGaussianDistribution, 
    CausalVaeDecoder, CausalVaeEncoder,
)
from .modeling_causal_conv import CausalConv3d

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


class CausalVideoVAE(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # encoder related parameters
        encoder_in_channels: int = 3,
        encoder_out_channels: int = 4,
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        encoder_down_block_types: Tuple[str, ...] = (
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
        ),
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        encoder_spatial_down_sample: Tuple[bool, ...] = (True, True, True, False),
        encoder_temporal_down_sample: Tuple[bool, ...] = (True, True, True, False),
        encoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        encoder_act_fn: str = "silu",
        encoder_norm_num_groups: int = 32,
        encoder_double_z: bool = True,
        encoder_type: str = 'causal_vae_conv',
        # decoder related
        decoder_in_channels: int = 4,
        decoder_out_channels: int = 3,
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_spatial_up_sample: Tuple[bool, ...] = (True, True, True, False),
        decoder_temporal_up_sample: Tuple[bool, ...] = (True, True, True, False),
        decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        decoder_type: str = 'causal_vae_conv',
        sample_size: int = 256,
        scaling_factor: float = 0.18215,
        add_post_quant_conv: bool = True,
        interpolate: bool = False,
        downsample_scale: int = 8,
    ):
        super().__init__()

        print(f"The latent dimmension channes is {encoder_out_channels}")
        # pass init params to Encoder

        self.encoder = CausalVaeEncoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            down_block_types=encoder_down_block_types,
            spatial_down_sample=encoder_spatial_down_sample,
            temporal_down_sample=encoder_temporal_down_sample,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            act_fn=encoder_act_fn,
            norm_num_groups=encoder_norm_num_groups,
            double_z=True,
            block_dropout=encoder_block_dropout,
        )

        # pass init params to Decoder
        self.decoder = CausalVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            spatial_up_sample=decoder_spatial_up_sample,
            temporal_up_sample=decoder_temporal_up_sample,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            interpolate=interpolate,
            block_dropout=decoder_block_dropout,
        )

        self.quant_conv = CausalConv3d(2 * encoder_out_channels, 2 * encoder_out_channels, kernel_size=1, stride=1)
        self.post_quant_conv = CausalConv3d(encoder_out_channels, encoder_out_channels, kernel_size=1, stride=1)
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size

        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / downsample_scale) 
        self.encode_tile_overlap_factor = 1 / 4
        self.decode_tile_overlap_factor = 1 / 4
        self.downsample_scale = downsample_scale

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True,
        is_init_image=True, temporal_chunk=False, window_size=16, tile_sample_min_size=256,
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)

        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict, is_init_image=is_init_image, 
                temporal_chunk=temporal_chunk, window_size=window_size)

        if temporal_chunk:
            moments = self.chunk_encode(x, window_size=window_size)
        else:
            h = self.encoder(x, is_init_image=is_init_image, temporal_chunk=False)
            moments = self.quant_conv(h, is_init_image=is_init_image, temporal_chunk=False)
    
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @torch.no_grad()
    def chunk_encode(self, x: torch.FloatTensor, window_size=16):
        # Only used during inference
        # Encode a long video clips through sliding window
        num_frames = x.shape[2]
        assert (num_frames - 1) % self.downsample_scale == 0
        init_window_size = window_size + 1
        frame_list = [x[:,:,:init_window_size]]

        # To chunk the long video 
        full_chunk_size = (num_frames - init_window_size) // window_size
        fid = init_window_size
        for idx in range(full_chunk_size):
            frame_list.append(x[:, :, fid:fid+window_size])
            fid += window_size

        if fid < num_frames:
            frame_list.append(x[:, :, fid:])

        latent_list = []
        for idx, frames in enumerate(frame_list):
            if idx == 0:
                h = self.encoder(frames, is_init_image=True, temporal_chunk=True)
                moments = self.quant_conv(h, is_init_image=True, temporal_chunk=True)
            else:
                h = self.encoder(frames, is_init_image=False, temporal_chunk=True)
                moments = self.quant_conv(h, is_init_image=False, temporal_chunk=True)

            latent_list.append(moments)

        latent = torch.cat(latent_list, dim=2)
        return latent

    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight
    
    @torch.no_grad()
    def chunk_decode(self, z: torch.FloatTensor, window_size=2):
        num_frames = z.shape[2]
        init_window_size = window_size + 1
        frame_list = [z[:,:,:init_window_size]]

        # To chunk the long video 
        full_chunk_size = (num_frames - init_window_size) // window_size
        fid = init_window_size
        for idx in range(full_chunk_size):
            frame_list.append(z[:, :, fid:fid+window_size])
            fid += window_size

        if fid < num_frames:
            frame_list.append(z[:, :, fid:])

        dec_list = []
        for idx, frames in enumerate(frame_list):
            if idx == 0:
                z_h = self.post_quant_conv(frames, is_init_image=True, temporal_chunk=True)
                dec = self.decoder(z_h, is_init_image=True, temporal_chunk=True)
            else:
                z_h = self.post_quant_conv(frames, is_init_image=False, temporal_chunk=True)
                dec = self.decoder(z_h, is_init_image=False, temporal_chunk=True)

            dec_list.append(dec)

        dec = torch.cat(dec_list, dim=2)
        return dec

    def decode(self, z: torch.FloatTensor, is_init_image=True, temporal_chunk=False, 
            return_dict: bool = True, window_size: int = 2, tile_sample_min_size: int = 256,) -> Union[DecoderOutput, torch.FloatTensor]:
        
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)

        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, is_init_image=is_init_image, 
                    temporal_chunk=temporal_chunk, window_size=window_size, return_dict=return_dict)

        if temporal_chunk:
            dec = self.chunk_decode(z, window_size=window_size)
        else:
            z = self.post_quant_conv(z, is_init_image=is_init_image, temporal_chunk=False)
            dec = self.decoder(z, is_init_image=is_init_image, temporal_chunk=False)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True, 
            is_init_image=True, temporal_chunk=False, window_size=16,) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.encode_tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.encode_tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                if temporal_chunk:
                    tile = self.chunk_encode(tile, window_size=window_size)
                else:
                    tile = self.encoder(tile, is_init_image=True, temporal_chunk=False)
                    tile = self.quant_conv(tile, is_init_image=True, temporal_chunk=False)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.FloatTensor, is_init_image=True, 
            temporal_chunk=False, window_size=2, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.decode_tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.decode_tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                if temporal_chunk:
                    decoded = self.chunk_decode(tile, window_size=window_size)
                else:
                    tile = self.post_quant_conv(tile, is_init_image=True, temporal_chunk=False)
                    decoded = self.decoder(tile, is_init_image=True, temporal_chunk=False)
                row.append(decoded)
            rows.append(row)
        result_rows = []

        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
        freeze_encoder: bool = False,
        is_init_image=True, 
        temporal_chunk=False,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample

        if is_context_parallel_initialized():
            assert self.training, "Only supports during training now"

            if freeze_encoder:
                with torch.no_grad():
                    h = self.encoder(x, is_init_image=True, temporal_chunk=False)
                    moments = self.quant_conv(h, is_init_image=True, temporal_chunk=False)
                    posterior = DiagonalGaussianDistribution(moments)
                    global_posterior = posterior
            else:
                h = self.encoder(x, is_init_image=True, temporal_chunk=False)
                moments = self.quant_conv(h, is_init_image=True, temporal_chunk=False)
                posterior = DiagonalGaussianDistribution(moments)
                global_moments = conv_gather_from_context_parallel_region(moments, dim=2, kernel_size=1)
                global_posterior = DiagonalGaussianDistribution(global_moments)
            
            if sample_posterior:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()

            if get_context_parallel_rank() == 0:
                dec = self.decode(z, is_init_image=True).sample
            else:
                # Do not drop the first upsampled frame
                dec = self.decode(z, is_init_image=False).sample

            return global_posterior, dec

        else:
            # The normal training
            if freeze_encoder:
                with torch.no_grad():
                    posterior = self.encode(x, is_init_image=is_init_image, 
                            temporal_chunk=temporal_chunk).latent_dist
            else:
                posterior = self.encode(x, is_init_image=is_init_image, 
                        temporal_chunk=temporal_chunk).latent_dist
        
            if sample_posterior:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()

            dec = self.decode(z, is_init_image=is_init_image, temporal_chunk=temporal_chunk).sample

            return posterior, dec

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
