#!/usr/bin/env python3
"""
Enhanced Pyramid Flow Demo with Improvements
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enhanced_inference import EnhancedPyramidFlow, PromptOptimizer, create_prompt_variations
from memory_optimization import MemoryOptimizer, AdaptiveInference
from quality_enhancer import VideoQualityEnhancer, FrameInterpolator

def main():
    parser = argparse.ArgumentParser(description="Enhanced Pyramid Flow Video Generation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the Pyramid Flow model")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="enhanced_output.mp4",
                       help="Output video path")
    parser.add_argument("--model_name", type=str, default="pyramid_flux",
                       choices=["pyramid_flux", "pyramid_mmdit"],
                       help="Model architecture to use")
    parser.add_argument("--resolution", type=str, default="768p",
                       choices=["384p", "768p"],
                       help="Output resolution")
    parser.add_argument("--duration", type=int, default=16,
                       help="Video duration in frames (temp parameter)")
    parser.add_argument("--fps", type=int, default=24,
                       help="Output video FPS")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Enhancement options
    parser.add_argument("--enhance_prompt", action="store_true",
                       help="Automatically enhance the prompt")
    parser.add_argument("--optimize_memory", action="store_true",
                       help="Enable memory optimizations")
    parser.add_argument("--post_process", action="store_true",
                       help="Apply post-processing enhancements")
    parser.add_argument("--interpolate_frames", action="store_true",
                       help="Interpolate frames for smoother video")
    parser.add_argument("--color_grading", type=str, default=None,
                       choices=["cinematic", "warm", "cool"],
                       help="Apply color grading style")
    parser.add_argument("--upscale", type=float, default=None,
                       help="Upscale factor for output resolution")
    
    # Advanced options
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                       help="Guidance scale for generation")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0,
                       help="Video guidance scale")
    parser.add_argument("--create_variations", type=int, default=0,
                       help="Create N variations of the prompt")
    
    args = parser.parse_args()
    
    # Set resolution parameters
    if args.resolution == "768p":
        width, height = 1280, 768
    else:
        width, height = 640, 384
    
    print(f"🎬 Enhanced Pyramid Flow Video Generation")
    print(f"📝 Prompt: {args.prompt}")
    print(f"📐 Resolution: {width}x{height}")
    print(f"⏱️  Duration: {args.duration} frames")
    print(f"🎯 Output: {args.output}")
    
    # Initialize enhanced model
    print("\n🔧 Initializing enhanced model...")
    enhanced_model = EnhancedPyramidFlow(
        model_path=args.model_path,
        model_name=args.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Apply memory optimizations if requested
    if args.optimize_memory:
        print("💾 Applying memory optimizations...")
        memory_optimizer = MemoryOptimizer(enhanced_model.model)
        memory_optimizer.enable_cpu_offloading()
        memory_optimizer.optimize_vae_tiling()
        
        adaptive_inference = AdaptiveInference(enhanced_model.model)
        optimal_settings = adaptive_inference.get_optimal_settings(
            (width, height), args.duration
        )
        print(f"   Optimal settings: {optimal_settings}")
    
    # Optimize prompt if requested
    prompt = args.prompt
    if args.enhance_prompt:
        print("✨ Enhancing prompt...")
        prompt_optimizer = PromptOptimizer()
        analysis = prompt_optimizer.analyze_prompt(prompt)
        print(f"   Prompt analysis: {analysis}")
        
        prompt = prompt_optimizer.optimize_prompt(prompt, style="cinematic")
        print(f"   Enhanced prompt: {prompt}")
    
    # Create prompt variations if requested
    if args.create_variations > 0:
        print(f"🎭 Creating {args.create_variations} prompt variations...")
        variations = create_prompt_variations(prompt, args.create_variations)
        for i, variation in enumerate(variations):
            print(f"   Variation {i+1}: {variation}")
        
        # For demo, use the first variation
        prompt = variations[0]
    
    # Generate video
    print("\n🎥 Generating video...")
    with MemoryOptimizer.memory_efficient_mode():
        frames = enhanced_model.generate_with_enhancement(
            prompt=prompt,
            width=width,
            height=height,
            temp=args.duration,
            guidance_scale=args.guidance_scale,
            video_guidance_scale=args.video_guidance_scale,
            enhance_prompt_flag=False,  # Already enhanced above
            seed=args.seed
        )
    
    print(f"✅ Generated {len(frames)} frames")
    
    # Apply post-processing if requested
    if args.post_process or args.color_grading or args.upscale or args.interpolate_frames:
        print("\n🎨 Applying post-processing...")
        
        quality_enhancer = VideoQualityEnhancer()
        
        # Apply quality enhancements
        frames = quality_enhancer.enhance_video_complete(
            frames,
            enhance_quality=args.post_process,
            apply_temporal_smoothing=args.post_process,
            color_grading_style=args.color_grading,
            upscale_factor=args.upscale
        )
        
        # Apply frame interpolation
        if args.interpolate_frames:
            print("🔄 Interpolating frames...")
            interpolator = FrameInterpolator()
            frames = interpolator.interpolate_sequence(frames, target_fps_multiplier=2)
            fps = args.fps * 2
            print(f"   Interpolated to {len(frames)} frames at {fps} FPS")
        else:
            fps = args.fps
    else:
        fps = args.fps
    
    # Export final video
    print(f"\n💾 Exporting video to {args.output}...")
    from diffusers.utils import export_to_video
    export_to_video(frames, args.output, fps=fps)
    
    print(f"🎉 Video generation complete!")
    print(f"📁 Output saved to: {args.output}")
    print(f"📊 Final stats:")
    print(f"   - Frames: {len(frames)}")
    print(f"   - FPS: {fps}")
    print(f"   - Duration: {len(frames)/fps:.2f} seconds")
    
    # Show memory stats if optimization was used
    if args.optimize_memory:
        memory_stats = memory_optimizer.get_memory_stats()
        print(f"   - GPU Memory: {memory_stats.get('gpu_allocated', 0):.2f} GB")

if __name__ == "__main__":
    import torch
    main()