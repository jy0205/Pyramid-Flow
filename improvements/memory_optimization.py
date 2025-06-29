import torch
import gc
from contextlib import contextmanager
from typing import Optional, Generator

class MemoryOptimizer:
    """Memory optimization utilities for Pyramid Flow"""
    
    def __init__(self, model):
        self.model = model
        self.original_forward = None
        
    @staticmethod
    @contextmanager
    def memory_efficient_mode() -> Generator[None, None, None]:
        """Context manager for memory efficient inference"""
        try:
            # Clear cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            yield
            
        finally:
            # Clean up after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training"""
        if hasattr(self.model.dit, 'enable_gradient_checkpointing'):
            self.model.dit.enable_gradient_checkpointing()
        
        if hasattr(self.model.vae, 'enable_gradient_checkpointing'):
            self.model.vae.enable_gradient_checkpointing()
    
    def enable_cpu_offloading(self):
        """Enable CPU offloading for components not currently in use"""
        if hasattr(self.model, 'enable_sequential_cpu_offload'):
            self.model.enable_sequential_cpu_offload()
    
    def optimize_vae_tiling(self, tile_size: int = 512):
        """Optimize VAE tiling for memory efficiency"""
        if hasattr(self.model.vae, 'enable_tiling'):
            self.model.vae.enable_tiling()
            # Set custom tile size if supported
            if hasattr(self.model.vae, 'tile_sample_min_size'):
                self.model.vae.tile_sample_min_size = tile_size
    
    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
            stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return stats
    
    def clear_memory(self):
        """Clear GPU memory and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class AdaptiveInference:
    """Adaptive inference that adjusts parameters based on available memory"""
    
    def __init__(self, model, memory_threshold_gb: float = 8.0):
        self.model = model
        self.memory_threshold = memory_threshold_gb * 1024**3  # Convert to bytes
        self.optimizer = MemoryOptimizer(model)
    
    def get_optimal_settings(self, target_resolution: tuple, target_frames: int) -> dict:
        """Get optimal settings based on available memory"""
        available_memory = self._get_available_memory()
        
        settings = {
            'use_cpu_offloading': False,
            'enable_tiling': False,
            'batch_size': 1,
            'save_memory': True,
            'tile_size': 512
        }
        
        # Estimate memory requirements
        width, height = target_resolution
        estimated_memory = self._estimate_memory_usage(width, height, target_frames)
        
        if estimated_memory > available_memory * 0.8:  # Use 80% threshold
            settings['use_cpu_offloading'] = True
            settings['enable_tiling'] = True
            settings['save_memory'] = True
            
            if estimated_memory > available_memory * 0.9:
                settings['tile_size'] = 256  # Smaller tiles for very limited memory
        
        return settings
    
    def _get_available_memory(self) -> int:
        """Get available GPU memory in bytes"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        return 8 * 1024**3  # Default to 8GB for CPU
    
    def _estimate_memory_usage(self, width: int, height: int, frames: int) -> int:
        """Estimate memory usage for given parameters"""
        # Rough estimation based on resolution and frames
        base_memory = width * height * frames * 4 * 3  # 4 bytes per pixel, 3 channels
        model_memory = 4 * 1024**3  # Estimate 4GB for model weights
        
        return base_memory + model_memory