import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple
from PIL import Image
import cv2
from diffusers.utils import export_to_video
from pyramid_dit import PyramidDiTForVideoGeneration

class EnhancedPyramidFlow:
    """Enhanced version of Pyramid Flow with improved inference capabilities"""
    
    def __init__(self, model_path: str, model_name: str = "pyramid_flux", 
                 model_dtype: str = "bf16", device: str = "cuda"):
        self.device = device
        self.model_dtype = model_dtype
        
        # Initialize the base model
        self.model = PyramidDiTForVideoGeneration(
            model_path,
            model_dtype,
            model_name=model_name,
            model_variant='diffusion_transformer_768p',
        )
        
        # Enable optimizations
        self.model.vae.enable_tiling()
        if hasattr(self.model.vae, 'to'):
            self.model.vae.to(device)
        if hasattr(self.model.dit, 'to'):
            self.model.dit.to(device)
        if hasattr(self.model.text_encoder, 'to'):
            self.model.text_encoder.to(device)
            
        self.torch_dtype = torch.bfloat16 if model_dtype == "bf16" else torch.float32
        
    def enhance_prompt(self, prompt: str) -> str:
        """Enhance prompts with cinematic and quality keywords"""
        quality_keywords = [
            "high quality", "detailed", "cinematic", "professional",
            "vivid colors", "sharp focus", "well-lit"
        ]
        
        style_keywords = [
            "cinematic style", "film grain", "depth of field",
            "dynamic lighting", "atmospheric"
        ]
        
        # Add quality and style keywords if not present
        enhanced = prompt.lower()
        
        # Add quality keywords
        if not any(kw in enhanced for kw in ["quality", "detailed", "sharp"]):
            prompt += ", high quality, detailed"
            
        # Add cinematic style if not present
        if not any(kw in enhanced for kw in ["cinematic", "film", "movie"]):
            prompt += ", cinematic style"
            
        # Add lighting if not present
        if not any(kw in enhanced for kw in ["lighting", "lit"]):
            prompt += ", dynamic lighting"
            
        return prompt
    
    def adaptive_guidance_scheduling(self, num_steps: int, base_guidance: float = 7.0) -> List[float]:
        """Create adaptive guidance schedule that starts high and decreases"""
        schedule = []
        for i in range(num_steps):
            # Higher guidance at the beginning, lower at the end
            progress = i / (num_steps - 1)
            guidance = base_guidance * (1.0 - 0.3 * progress)  # Reduce by 30% over time
            schedule.append(max(guidance, 3.0))  # Minimum guidance of 3.0
        return schedule
    
    def generate_with_enhancement(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, artifacts, bad anatomy",
        width: int = 1280,
        height: int = 768,
        temp: int = 16,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 5.0,
        num_inference_steps: List[int] = [20, 20, 20],
        video_num_inference_steps: List[int] = [10, 10, 10],
        enhance_prompt_flag: bool = True,
        use_adaptive_guidance: bool = True,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Enhanced generation with multiple improvements"""
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Enhance prompt if requested
        if enhance_prompt_flag:
            prompt = self.enhance_prompt(prompt)
        
        # Use adaptive guidance if requested
        if use_adaptive_guidance:
            guidance_scale = self.adaptive_guidance_scheduling(
                max(num_inference_steps), guidance_scale
            )[0]  # Use first value for now
        
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=(self.model_dtype != 'fp32'), dtype=self.torch_dtype
        ):
            frames = self.model.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                video_num_inference_steps=video_num_inference_steps,
                height=height,
                width=width,
                temp=temp,
                guidance_scale=guidance_scale,
                video_guidance_scale=video_guidance_scale,
                output_type="pil",
                save_memory=True,
                **kwargs
            )
        
        return frames
    
    def interpolate_frames(self, frames: List[Image.Image], target_fps: int = 60) -> List[Image.Image]:
        """Interpolate frames to increase FPS using simple blending"""
        if len(frames) < 2:
            return frames
            
        interpolated = [frames[0]]
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Add original frame
            interpolated.append(current_frame)
            
            # Add interpolated frame (simple blend)
            current_array = np.array(current_frame)
            next_array = np.array(next_frame)
            
            # Simple linear interpolation
            blended = (current_array * 0.5 + next_array * 0.5).astype(np.uint8)
            interpolated.append(Image.fromarray(blended))
        
        # Add last frame
        interpolated.append(frames[-1])
        
        return interpolated
    
    def apply_post_processing(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply post-processing to enhance video quality"""
        processed_frames = []
        
        for frame in frames:
            # Convert to numpy array
            frame_array = np.array(frame)
            
            # Apply subtle sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(frame_array, -1, kernel * 0.1)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Enhance contrast slightly
            enhanced = cv2.convertScaleAbs(sharpened, alpha=1.05, beta=2)
            
            processed_frames.append(Image.fromarray(enhanced))
        
        return processed_frames
    
    def generate_video_with_enhancements(
        self,
        prompt: str,
        output_path: str,
        fps: int = 24,
        apply_interpolation: bool = False,
        apply_post_processing: bool = True,
        **generation_kwargs
    ):
        """Complete pipeline with all enhancements"""
        
        print(f"Generating video for prompt: '{prompt}'")
        
        # Generate frames
        frames = self.generate_with_enhancement(prompt, **generation_kwargs)
        
        # Apply post-processing
        if apply_post_processing:
            print("Applying post-processing...")
            frames = self.apply_post_processing(frames)
        
        # Apply frame interpolation
        if apply_interpolation:
            print("Interpolating frames...")
            frames = self.interpolate_frames(frames, target_fps=fps*2)
            fps = fps * 2
        
        # Export video
        print(f"Exporting video to {output_path}")
        export_to_video(frames, output_path, fps=fps)
        
        return frames

class PromptOptimizer:
    """Optimize prompts for better video generation"""
    
    def __init__(self):
        self.quality_terms = [
            "high quality", "4k", "detailed", "sharp", "crisp",
            "professional", "cinematic", "masterpiece"
        ]
        
        self.style_terms = [
            "cinematic style", "film grain", "movie trailer",
            "documentary style", "artistic", "atmospheric"
        ]
        
        self.lighting_terms = [
            "dramatic lighting", "soft lighting", "natural lighting",
            "golden hour", "studio lighting", "volumetric lighting"
        ]
        
        self.camera_terms = [
            "wide shot", "close-up", "medium shot", "aerial view",
            "tracking shot", "dolly zoom", "handheld camera"
        ]
    
    def analyze_prompt(self, prompt: str) -> dict:
        """Analyze prompt and suggest improvements"""
        prompt_lower = prompt.lower()
        
        analysis = {
            "has_quality_terms": any(term in prompt_lower for term in self.quality_terms),
            "has_style_terms": any(term in prompt_lower for term in self.style_terms),
            "has_lighting_terms": any(term in prompt_lower for term in self.lighting_terms),
            "has_camera_terms": any(term in prompt_lower for term in self.camera_terms),
            "word_count": len(prompt.split()),
            "suggestions": []
        }
        
        # Generate suggestions
        if not analysis["has_quality_terms"]:
            analysis["suggestions"].append("Add quality terms like 'high quality', 'detailed', or 'cinematic'")
        
        if not analysis["has_style_terms"]:
            analysis["suggestions"].append("Consider adding style terms like 'cinematic style' or 'film grain'")
        
        if not analysis["has_lighting_terms"]:
            analysis["suggestions"].append("Add lighting description like 'dramatic lighting' or 'soft lighting'")
        
        if analysis["word_count"] < 5:
            analysis["suggestions"].append("Consider adding more descriptive details")
        
        if analysis["word_count"] > 50:
            analysis["suggestions"].append("Consider shortening the prompt for better focus")
        
        return analysis
    
    def optimize_prompt(self, prompt: str, style: str = "cinematic") -> str:
        """Automatically optimize a prompt"""
        analysis = self.analyze_prompt(prompt)
        optimized = prompt
        
        # Add quality if missing
        if not analysis["has_quality_terms"]:
            optimized += ", high quality, detailed"
        
        # Add style based on preference
        if not analysis["has_style_terms"]:
            if style == "cinematic":
                optimized += ", cinematic style, film grain"
            elif style == "documentary":
                optimized += ", documentary style, realistic"
            elif style == "artistic":
                optimized += ", artistic, creative composition"
        
        # Add lighting if missing
        if not analysis["has_lighting_terms"]:
            optimized += ", dramatic lighting"
        
        return optimized.strip()

def create_prompt_variations(base_prompt: str, num_variations: int = 5) -> List[str]:
    """Create variations of a prompt for experimentation"""
    
    style_modifiers = [
        "cinematic style, shot on 35mm film",
        "documentary style, handheld camera",
        "artistic composition, creative angles",
        "professional cinematography, studio quality",
        "vintage film aesthetic, film grain"
    ]
    
    lighting_modifiers = [
        "dramatic lighting, high contrast",
        "soft natural lighting, golden hour",
        "studio lighting, professional setup",
        "atmospheric lighting, moody",
        "bright daylight, vivid colors"
    ]
    
    variations = []
    
    for i in range(num_variations):
        variation = base_prompt
        
        # Add style modifier
        if i < len(style_modifiers):
            variation += f", {style_modifiers[i]}"
        
        # Add lighting modifier
        if i < len(lighting_modifiers):
            variation += f", {lighting_modifiers[i]}"
        
        variations.append(variation)
    
    return variations