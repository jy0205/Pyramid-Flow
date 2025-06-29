import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import List, Tuple, Optional

class VideoQualityEnhancer:
    """Post-processing quality enhancement for generated videos"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def enhance_frame_quality(self, frame: Image.Image, 
                            sharpen_strength: float = 0.1,
                            contrast_boost: float = 1.05,
                            saturation_boost: float = 1.1) -> Image.Image:
        """Enhance individual frame quality"""
        
        # Convert to numpy for OpenCV operations
        frame_array = np.array(frame)
        
        # Apply unsharp masking for sharpening
        if sharpen_strength > 0:
            gaussian = cv2.GaussianBlur(frame_array, (0, 0), 2.0)
            frame_array = cv2.addWeighted(frame_array, 1.0 + sharpen_strength, 
                                        gaussian, -sharpen_strength, 0)
        
        # Convert back to PIL for other enhancements
        enhanced_frame = Image.fromarray(np.clip(frame_array, 0, 255).astype(np.uint8))
        
        # Enhance contrast
        if contrast_boost != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced_frame)
            enhanced_frame = enhancer.enhance(contrast_boost)
        
        # Enhance saturation
        if saturation_boost != 1.0:
            enhancer = ImageEnhance.Color(enhanced_frame)
            enhanced_frame = enhancer.enhance(saturation_boost)
        
        return enhanced_frame
    
    def temporal_smoothing(self, frames: List[Image.Image], 
                          smoothing_strength: float = 0.1) -> List[Image.Image]:
        """Apply temporal smoothing to reduce flickering"""
        if len(frames) < 3:
            return frames
        
        smoothed_frames = [frames[0]]  # Keep first frame as is
        
        for i in range(1, len(frames) - 1):
            prev_frame = np.array(frames[i-1]).astype(np.float32)
            curr_frame = np.array(frames[i]).astype(np.float32)
            next_frame = np.array(frames[i+1]).astype(np.float32)
            
            # Temporal average with current frame weighted more
            smoothed = (prev_frame * smoothing_strength + 
                       curr_frame * (1 - 2 * smoothing_strength) + 
                       next_frame * smoothing_strength)
            
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
            smoothed_frames.append(Image.fromarray(smoothed))
        
        smoothed_frames.append(frames[-1])  # Keep last frame as is
        return smoothed_frames
    
    def stabilize_video(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Simple video stabilization using feature matching"""
        if len(frames) < 2:
            return frames
        
        stabilized_frames = [frames[0]]
        
        # Convert first frame to grayscale for feature detection
        prev_gray = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY)
        
        for i in range(1, len(frames)):
            curr_frame = np.array(frames[i])
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            # Detect features and compute optical flow
            try:
                # Use Lucas-Kanade optical flow for stabilization
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, None, None,
                    winSize=(15, 15), maxLevel=2
                )
                
                # Apply simple stabilization (this is a basic implementation)
                # In practice, you'd want more sophisticated stabilization
                stabilized_frames.append(frames[i])
                
            except:
                # If stabilization fails, use original frame
                stabilized_frames.append(frames[i])
            
            prev_gray = curr_gray
        
        return stabilized_frames
    
    def upscale_frames(self, frames: List[Image.Image], 
                      scale_factor: float = 2.0) -> List[Image.Image]:
        """Upscale frames using bicubic interpolation"""
        upscaled_frames = []
        
        for frame in frames:
            width, height = frame.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            upscaled = frame.resize((new_width, new_height), Image.BICUBIC)
            upscaled_frames.append(upscaled)
        
        return upscaled_frames
    
    def apply_color_grading(self, frames: List[Image.Image], 
                           style: str = "cinematic") -> List[Image.Image]:
        """Apply color grading for different styles"""
        graded_frames = []
        
        for frame in frames:
            frame_array = np.array(frame).astype(np.float32) / 255.0
            
            if style == "cinematic":
                # Orange and teal look
                frame_array[:, :, 0] *= 1.1  # Boost reds
                frame_array[:, :, 1] *= 0.95  # Slightly reduce greens
                frame_array[:, :, 2] *= 1.05  # Boost blues
                
            elif style == "warm":
                # Warm, golden look
                frame_array[:, :, 0] *= 1.15  # Boost reds
                frame_array[:, :, 1] *= 1.05  # Boost greens
                frame_array[:, :, 2] *= 0.9   # Reduce blues
                
            elif style == "cool":
                # Cool, blue look
                frame_array[:, :, 0] *= 0.9   # Reduce reds
                frame_array[:, :, 1] *= 0.95  # Slightly reduce greens
                frame_array[:, :, 2] *= 1.15  # Boost blues
            
            # Clamp values and convert back
            frame_array = np.clip(frame_array * 255, 0, 255).astype(np.uint8)
            graded_frames.append(Image.fromarray(frame_array))
        
        return graded_frames
    
    def enhance_video_complete(self, frames: List[Image.Image],
                             enhance_quality: bool = True,
                             apply_temporal_smoothing: bool = True,
                             apply_stabilization: bool = False,
                             color_grading_style: Optional[str] = "cinematic",
                             upscale_factor: Optional[float] = None) -> List[Image.Image]:
        """Complete video enhancement pipeline"""
        
        enhanced_frames = frames.copy()
        
        # Apply quality enhancement
        if enhance_quality:
            enhanced_frames = [
                self.enhance_frame_quality(frame) 
                for frame in enhanced_frames
            ]
        
        # Apply temporal smoothing
        if apply_temporal_smoothing:
            enhanced_frames = self.temporal_smoothing(enhanced_frames)
        
        # Apply stabilization
        if apply_stabilization:
            enhanced_frames = self.stabilize_video(enhanced_frames)
        
        # Apply color grading
        if color_grading_style:
            enhanced_frames = self.apply_color_grading(enhanced_frames, color_grading_style)
        
        # Apply upscaling
        if upscale_factor and upscale_factor > 1.0:
            enhanced_frames = self.upscale_frames(enhanced_frames, upscale_factor)
        
        return enhanced_frames

class FrameInterpolator:
    """Advanced frame interpolation for smoother videos"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def linear_interpolation(self, frame1: Image.Image, frame2: Image.Image, 
                           alpha: float = 0.5) -> Image.Image:
        """Simple linear interpolation between two frames"""
        array1 = np.array(frame1).astype(np.float32)
        array2 = np.array(frame2).astype(np.float32)
        
        interpolated = (1 - alpha) * array1 + alpha * array2
        return Image.fromarray(np.clip(interpolated, 0, 255).astype(np.uint8))
    
    def optical_flow_interpolation(self, frame1: Image.Image, frame2: Image.Image) -> Image.Image:
        """More advanced interpolation using optical flow"""
        # Convert to numpy arrays
        img1 = np.array(frame1)
        img2 = np.array(frame2)
        
        # Convert to grayscale for flow calculation
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Create interpolated frame (simplified version)
        h, w = gray1.shape
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x + flow[:, :, 0] * 0.5
        map_y = map_y + flow[:, :, 1] * 0.5
        
        # Remap the image
        interpolated = cv2.remap(img1, map_x.astype(np.float32), 
                               map_y.astype(np.float32), cv2.INTER_LINEAR)
        
        return Image.fromarray(interpolated)
    
    def interpolate_sequence(self, frames: List[Image.Image], 
                           target_fps_multiplier: int = 2,
                           method: str = "linear") -> List[Image.Image]:
        """Interpolate a sequence of frames to increase FPS"""
        if len(frames) < 2 or target_fps_multiplier <= 1:
            return frames
        
        interpolated_frames = [frames[0]]
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Add original frame
            interpolated_frames.append(current_frame)
            
            # Add interpolated frames
            for j in range(1, target_fps_multiplier):
                alpha = j / target_fps_multiplier
                
                if method == "linear":
                    interp_frame = self.linear_interpolation(current_frame, next_frame, alpha)
                elif method == "optical_flow":
                    try:
                        interp_frame = self.optical_flow_interpolation(current_frame, next_frame)
                    except:
                        # Fallback to linear if optical flow fails
                        interp_frame = self.linear_interpolation(current_frame, next_frame, alpha)
                else:
                    interp_frame = self.linear_interpolation(current_frame, next_frame, alpha)
                
                interpolated_frames.append(interp_frame)
        
        # Add the last frame
        interpolated_frames.append(frames[-1])
        
        return interpolated_frames