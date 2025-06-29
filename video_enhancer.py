import os
import argparse
import torch
import numpy as np
import cv2
from typing import List, Optional, Union, Tuple
from PIL import Image, ImageEnhance, ImageFilter
from diffusers.utils import export_to_video
from pyramid_dit import PyramidDiTForVideoGeneration

class VideoEnhancer:
    """Post-processing enhancements for generated videos"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def enhance_frame_quality(self, frame: Image.Image, 
                            sharpen_strength: float = 0.3,
                            contrast_boost: float = 1.1,
                            saturation_boost: float = 1.1,
                            brightness_boost: float = 1.05) -> Image.Image:
        """Enhance individual frame quality"""
        
        # Convert to numpy for OpenCV operations
        frame_array = np.array(frame)
        
        # Apply unsharp masking for sharpening
        if sharpen_strength > 0:
            gaussian = cv2.GaussianBlur(frame_array, (0, 0), 3.0)
            frame_array = cv2.addWeighted(frame_array, 1.0 + sharpen_strength, 
                                        gaussian, -sharpen_strength, 0)
        
        # Convert back to PIL for other enhancements
        enhanced_frame = Image.fromarray(np.clip(frame_array, 0, 255).astype(np.uint8))
        
        # Enhance brightness
        if brightness_boost != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced_frame)
            enhanced_frame = enhancer.enhance(brightness_boost)
        
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
                          smoothing_strength: float = 0.2) -> List[Image.Image]:
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
    
    def interpolate_frames(self, frames: List[Image.Image], 
                          method: str = "blend") -> List[Image.Image]:
        """Interpolate frames to increase FPS"""
        if len(frames) < 2:
            return frames
            
        interpolated = []
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Add original frame
            interpolated.append(current_frame)
            
            # Add interpolated frame
            if method == "blend":
                # Simple linear interpolation
                current_array = np.array(current_frame)
                next_array = np.array(next_frame)
                blended = (current_array * 0.5 + next_array * 0.5).astype(np.uint8)
                interpolated.append(Image.fromarray(blended))
            elif method == "flow":
                # More advanced optical flow-based interpolation
                # This is a simplified version - real optical flow would be more complex
                current_array = np.array(current_frame)
                next_array = np.array(next_frame)
                
                # Convert to grayscale for flow calculation
                current_gray = cv2.cvtColor(current_array, cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(next_array, cv2.COLOR_RGB2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    current_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Create interpolated frame (simplified)
                h, w = current_gray.shape
                map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = map_x + flow[:, :, 0] * 0.5
                map_y = map_y + flow[:, :, 1] * 0.5
                
                # Remap the image
                interpolated_frame = cv2.remap(current_array, map_x.astype(np.float32), 
                                             map_y.astype(np.float32), cv2.INTER_LINEAR)
                
                interpolated.append(Image.fromarray(interpolated_frame))
        
        # Add last frame
        interpolated.append(frames[-1])
        
        return interpolated
    
    def enhance_video(self, 
                     input_frames: List[Image.Image],
                     enhance_quality: bool = True,
                     apply_temporal_smoothing: bool = True,
                     apply_color_grading: Optional[str] = "cinematic",
                     interpolate_frames: bool = False,
                     output_path: str = "enhanced_video.mp4",
                     fps: int = 24) -> str:
        """Apply all enhancements to a video"""
        
        frames = input_frames.copy()
        
        print(f"Enhancing video with {len(frames)} frames...")
        
        # Apply quality enhancement
        if enhance_quality:
            print("Applying quality enhancement...")
            frames = [self.enhance_frame_quality(frame) for frame in frames]
        
        # Apply temporal smoothing
        if apply_temporal_smoothing:
            print("Applying temporal smoothing...")
            frames = self.temporal_smoothing(frames)
        
        # Apply color grading
        if apply_color_grading:
            print(f"Applying {apply_color_grading} color grading...")
            frames = self.apply_color_grading(frames, apply_color_grading)
        
        # Apply frame interpolation
        if interpolate_frames:
            print("Interpolating frames...")
            frames = self.interpolate_frames(frames)
            fps = fps * 2  # Double FPS since we've doubled the frames
        
        # Export video
        print(f"Exporting enhanced video to {output_path}...")
        export_to_video(frames, output_path, fps=fps)
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Enhance Pyramid Flow generated videos")
    parser.add_argument("--input_video", type=str, required=True,
                       help="Path to input video file or directory of frames")
    parser.add_argument("--output_video", type=str, default="enhanced_video.mp4",
                       help="Path to save enhanced video")
    parser.add_argument("--enhance_quality", action="store_true", default=True,
                       help="Apply quality enhancement to individual frames")
    parser.add_argument("--temporal_smoothing", action="store_true", default=True,
                       help="Apply temporal smoothing to reduce flickering")
    parser.add_argument("--color_grading", type=str, default="cinematic",
                       choices=["cinematic", "warm", "cool", "none"],
                       help="Apply color grading style")
    parser.add_argument("--interpolate", action="store_true",
                       help="Interpolate frames to double FPS")
    parser.add_argument("--fps", type=int, default=24,
                       help="Output video FPS")
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = VideoEnhancer()
    
    # Load input video frames
    if args.input_video.endswith(('.mp4', '.avi', '.mov')):
        # Load from video file
        print(f"Loading video from {args.input_video}...")
        cap = cv2.VideoCapture(args.input_video)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        cap.release()
    else:
        # Assume it's a directory of frames
        print(f"Loading frames from directory {args.input_video}...")
        frame_files = sorted([f for f in os.listdir(args.input_video) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        frames = [Image.open(os.path.join(args.input_video, f)) for f in frame_files]
    
    print(f"Loaded {len(frames)} frames")
    
    # Process color grading argument
    color_grading = None if args.color_grading == "none" else args.color_grading
    
    # Enhance video
    output_path = enhancer.enhance_video(
        frames,
        enhance_quality=args.enhance_quality,
        apply_temporal_smoothing=args.temporal_smoothing,
        apply_color_grading=color_grading,
        interpolate_frames=args.interpolate,
        output_path=args.output_video,
        fps=args.fps
    )
    
    print(f"Video enhancement complete! Output saved to: {output_path}")

if __name__ == "__main__":
    main()