import os
import argparse
import json
import time
import torch
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video

class BatchVideoProcessor:
    """Process multiple video generation requests efficiently"""
    
    def __init__(self, model_path: str, model_name: str = "pyramid_flux", 
                 model_variant: str = "diffusion_transformer_768p",
                 model_dtype: str = "bf16",
                 max_workers: int = 1, 
                 output_dir: str = "batch_outputs"):
        self.model_path = model_path
        self.model_name = model_name
        self.model_variant = model_variant
        self.model_dtype = model_dtype
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Thread-local storage for models
        self.local = threading.local()
        
    def _get_model(self):
        """Get thread-local model instance"""
        if not hasattr(self.local, 'model') or self.local.model is None:
            print(f"Initializing model for thread {threading.current_thread().name}")
            self.local.model = PyramidDiTForVideoGeneration(
                self.model_path, 
                self.model_dtype,
                model_name=self.model_name,
                model_variant=self.model_variant,
            )
            
            # Apply optimizations
            self.local.model.vae.enable_tiling()
            self.local.model.enable_sequential_cpu_offload()
            
            # Set torch dtype
            if self.model_dtype == "bf16":
                self.local.torch_dtype = torch.bfloat16
            elif self.model_dtype == "fp16":
                self.local.torch_dtype = torch.float16
            else:
                self.local.torch_dtype = torch.float32
        
        return self.local.model, self.local.torch_dtype
    
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
    
    def process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video generation request"""
        start_time = time.time()
        
        try:
            # Extract parameters
            prompt = request['prompt']
            output_name = request.get('output_name', f"video_{int(time.time())}.mp4")
            settings = request.get('settings', {})
            
            # Get model instance
            model, torch_dtype = self._get_model()
            
            # Enhance prompt if requested
            if settings.get('enhance_prompt', True):
                prompt = self.enhance_prompt(prompt)
            
            # Set resolution
            if settings.get('resolution', '768p') == '768p':
                width, height = 1280, 768
            else:
                width, height = 640, 384
            
            # Set seed if provided
            seed = settings.get('seed')
            if seed is not None and seed > 0:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            
            # Generate video
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                frames = model.generate(
                    prompt=prompt,
                    num_inference_steps=[20, 20, 20],
                    video_num_inference_steps=[10, 10, 10],
                    height=height,
                    width=width,
                    temp=settings.get('duration', 16),
                    guidance_scale=settings.get('guidance_scale', 7.0),
                    video_guidance_scale=settings.get('video_guidance_scale', 5.0),
                    output_type="pil",
                    save_memory=True,
                )
            
            # Save video
            output_path = self.output_dir / output_name
            export_to_video(frames, str(output_path), fps=settings.get('fps', 24))
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'output_path': str(output_path),
                'processing_time': processing_time,
                'frames_generated': len(frames),
                'prompt': prompt,
                'original_request': request
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': processing_time,
                'original_request': request
            }
    
    def process_batch(self, requests: List[Dict[str, Any]], 
                     save_results: bool = True) -> List[Dict[str, Any]]:
        """Process a batch of video generation requests"""
        
        print(f"🎬 Processing batch of {len(requests)} requests...")
        print(f"👥 Using {self.max_workers} worker(s)")
        
        results = []
        
        if self.max_workers == 1:
            # Sequential processing
            for i, request in enumerate(requests):
                print(f"📹 Processing request {i+1}/{len(requests)}: {request['prompt'][:50]}...")
                result = self.process_single_request(request)
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"✅ Completed in {result['processing_time']:.2f}s")
                else:
                    print(f"❌ Failed: {result['error']}")
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all requests
                future_to_request = {
                    executor.submit(self.process_single_request, request): request 
                    for request in requests
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_request):
                    request = future_to_request[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'success':
                            print(f"✅ Completed: {request['prompt'][:50]}...")
                        else:
                            print(f"❌ Failed: {request['prompt'][:50]}... - {result['error']}")
                            
                    except Exception as e:
                        print(f"❌ Exception for {request['prompt'][:50]}...: {e}")
                        results.append({
                            'status': 'error',
                            'error': str(e),
                            'original_request': request
                        })
        
        # Save results if requested
        if save_results:
            results_file = self.output_dir / "batch_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📊 Results saved to {results_file}")
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        total_time = sum(r['processing_time'] for r in results)
        
        print(f"\n📈 Batch Processing Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📊 Average time: {total_time/len(results):.2f}s per video")
        
        return results
    
    def load_requests_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load batch requests from a JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def create_sample_batch(self, output_file: str = "sample_batch.json"):
        """Create a sample batch file for testing"""
        sample_requests = [
            {
                "prompt": "A serene mountain landscape at sunset",
                "output_name": "mountain_sunset.mp4",
                "settings": {
                    "duration": 16,
                    "resolution": "768p",
                    "guidance_scale": 7.0,
                    "video_guidance_scale": 5.0,
                    "fps": 24,
                    "seed": 42
                }
            },
            {
                "prompt": "A bustling city street at night with neon lights",
                "output_name": "city_night.mp4",
                "settings": {
                    "duration": 16,
                    "resolution": "768p",
                    "guidance_scale": 7.0,
                    "video_guidance_scale": 5.0,
                    "fps": 24,
                    "seed": 123
                }
            },
            {
                "prompt": "Ocean waves crashing on a rocky shore",
                "output_name": "ocean_waves.mp4",
                "settings": {
                    "duration": 16,
                    "resolution": "384p",
                    "guidance_scale": 7.0,
                    "video_guidance_scale": 5.0,
                    "fps": 24,
                    "seed": 456
                }
            },
            {
                "prompt": "A peaceful forest with sunlight filtering through trees",
                "output_name": "forest_sunlight.mp4",
                "settings": {
                    "duration": 16,
                    "resolution": "384p",
                    "guidance_scale": 7.0,
                    "video_guidance_scale": 5.0,
                    "fps": 24,
                    "seed": 789
                }
            }
        ]
        
        with open(output_file, 'w') as f:
            json.dump(sample_requests, f, indent=2)
        
        print(f"📝 Sample batch file created: {output_file}")
        return sample_requests

def main():
    """Example usage of batch processor"""
    parser = argparse.ArgumentParser(description="Batch Video Generation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the Pyramid Flow model")
    parser.add_argument("--model_name", type=str, default="pyramid_flux",
                       choices=["pyramid_flux", "pyramid_mmdit"],
                       help="Model architecture to use")
    parser.add_argument("--model_variant", type=str, default="diffusion_transformer_768p",
                       choices=["diffusion_transformer_768p", "diffusion_transformer_384p"],
                       help="Model variant to use")
    parser.add_argument("--batch_file", type=str, 
                       help="JSON file containing batch requests")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create a sample batch file")
    parser.add_argument("--output_dir", type=str, default="batch_outputs",
                       help="Output directory for generated videos")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker threads (use 1 for single GPU)")
    
    args = parser.parse_args()
    
    # Initialize batch processor
    processor = BatchVideoProcessor(
        model_path=args.model_path,
        model_name=args.model_name,
        model_variant=args.model_variant,
        max_workers=args.workers,
        output_dir=args.output_dir
    )
    
    if args.create_sample:
        # Create sample batch file
        sample_file = "sample_batch.json"
        requests = processor.create_sample_batch(sample_file)
        print(f"Created sample batch with {len(requests)} requests")
        
        if not args.batch_file:
            args.batch_file = sample_file
    
    if args.batch_file:
        # Load and process batch
        requests = processor.load_requests_from_file(args.batch_file)
        results = processor.process_batch(requests)
        
        print(f"\n🎉 Batch processing complete!")
        print(f"📁 Videos saved to: {args.output_dir}")
    else:
        print("Please provide --batch_file or use --create_sample")

if __name__ == "__main__":
    main()