import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from enhanced_inference import EnhancedPyramidFlow, PromptOptimizer
from quality_enhancer import VideoQualityEnhancer
from memory_optimization import MemoryOptimizer

class BatchVideoProcessor:
    """Process multiple video generation requests efficiently"""
    
    def __init__(self, model_path: str, model_name: str = "pyramid_flux", 
                 max_workers: int = 1, output_dir: str = "batch_outputs"):
        self.model_path = model_path
        self.model_name = model_name
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.quality_enhancer = VideoQualityEnhancer()
        self.prompt_optimizer = PromptOptimizer()
        
        # Thread-local storage for models
        self.local = threading.local()
        
    def _get_model(self):
        """Get thread-local model instance"""
        if not hasattr(self.local, 'model') or self.local.model is None:
            self.local.model = EnhancedPyramidFlow(
                self.model_path, 
                self.model_name
            )
            # Apply memory optimizations
            memory_optimizer = MemoryOptimizer(self.local.model.model)
            memory_optimizer.enable_cpu_offloading()
            memory_optimizer.optimize_vae_tiling()
        
        return self.local.model
    
    def process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video generation request"""
        start_time = time.time()
        
        try:
            # Extract parameters
            prompt = request['prompt']
            output_name = request.get('output_name', f"video_{int(time.time())}.mp4")
            settings = request.get('settings', {})
            
            # Get model instance
            model = self._get_model()
            
            # Enhance prompt if requested
            if settings.get('enhance_prompt', True):
                prompt = self.prompt_optimizer.optimize_prompt(prompt)
            
            # Generate video
            with MemoryOptimizer.memory_efficient_mode():
                frames = model.generate_with_enhancement(
                    prompt=prompt,
                    width=settings.get('width', 1280),
                    height=settings.get('height', 768),
                    temp=settings.get('duration', 16),
                    guidance_scale=settings.get('guidance_scale', 7.0),
                    video_guidance_scale=settings.get('video_guidance_scale', 5.0),
                    seed=settings.get('seed'),
                    enhance_prompt_flag=False  # Already enhanced above
                )
            
            # Apply post-processing if requested
            if settings.get('post_process', True):
                frames = self.quality_enhancer.enhance_video_complete(
                    frames,
                    enhance_quality=True,
                    apply_temporal_smoothing=True,
                    color_grading_style=settings.get('color_grading', 'cinematic')
                )
            
            # Save video
            output_path = self.output_dir / output_name
            from diffusers.utils import export_to_video
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
                    "color_grading": "warm"
                }
            },
            {
                "prompt": "A bustling city street at night with neon lights",
                "output_name": "city_night.mp4",
                "settings": {
                    "duration": 16,
                    "resolution": "768p",
                    "color_grading": "cool"
                }
            },
            {
                "prompt": "Ocean waves crashing on a rocky shore",
                "output_name": "ocean_waves.mp4",
                "settings": {
                    "duration": 20,
                    "resolution": "768p",
                    "color_grading": "cinematic"
                }
            },
            {
                "prompt": "A peaceful forest with sunlight filtering through trees",
                "output_name": "forest_sunlight.mp4",
                "settings": {
                    "duration": 16,
                    "resolution": "768p",
                    "color_grading": "warm"
                }
            }
        ]
        
        with open(output_file, 'w') as f:
            json.dump(sample_requests, f, indent=2)
        
        print(f"📝 Sample batch file created: {output_file}")
        return sample_requests

def main():
    """Example usage of batch processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Video Generation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the Pyramid Flow model")
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