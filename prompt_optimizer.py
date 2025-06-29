import argparse
import re
import random
from typing import List, Dict, Optional, Tuple, Union

class PromptOptimizer:
    """Optimize prompts for better video generation results"""
    
    def __init__(self):
        # Quality terms to enhance visual fidelity
        self.quality_terms = [
            "high quality", "detailed", "sharp", "crisp", "4K", "HD",
            "professional", "masterpiece", "photorealistic"
        ]
        
        # Style terms for cinematic look
        self.cinematic_terms = [
            "cinematic", "film grain", "movie quality", "professional cinematography",
            "dramatic lighting", "depth of field", "shallow focus", "bokeh"
        ]
        
        # Camera movement terms
        self.camera_terms = [
            "tracking shot", "dolly zoom", "aerial view", "establishing shot",
            "panning shot", "wide angle", "close-up", "medium shot"
        ]
        
        # Lighting terms
        self.lighting_terms = [
            "golden hour", "dramatic lighting", "volumetric lighting", 
            "soft lighting", "backlit", "studio lighting", "natural lighting"
        ]
        
        # Color terms
        self.color_terms = [
            "vibrant colors", "vivid", "colorful", "rich colors",
            "high contrast", "muted colors", "pastel colors"
        ]
        
        # Film emulation terms
        self.film_terms = [
            "shot on 35mm film", "Kodak Portra", "Fujifilm", "analog film",
            "vintage film look", "film grain", "celluloid"
        ]
        
        # Negative terms to avoid
        self.negative_terms = [
            "blurry", "low quality", "pixelated", "distorted", "low resolution",
            "poor lighting", "bad composition", "artifacts", "noise", "grainy",
            "oversaturated", "washed out", "bad anatomy", "deformed", "disfigured"
        ]
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and identify potential improvements"""
        prompt_lower = prompt.lower()
        
        analysis = {
            "has_quality_terms": any(term in prompt_lower for term in self.quality_terms),
            "has_cinematic_terms": any(term in prompt_lower for term in self.cinematic_terms),
            "has_camera_terms": any(term in prompt_lower for term in self.camera_terms),
            "has_lighting_terms": any(term in prompt_lower for term in self.lighting_terms),
            "has_color_terms": any(term in prompt_lower for term in self.color_terms),
            "has_film_terms": any(term in prompt_lower for term in self.film_terms),
            "has_negative_terms": any(term in prompt_lower for term in self.negative_terms),
            "word_count": len(prompt.split()),
            "suggestions": []
        }
        
        # Generate suggestions
        if not analysis["has_quality_terms"]:
            analysis["suggestions"].append("Add quality terms like 'high quality' or 'detailed'")
        
        if not analysis["has_cinematic_terms"]:
            analysis["suggestions"].append("Add cinematic terms like 'cinematic style' or 'film grain'")
        
        if not analysis["has_camera_terms"]:
            analysis["suggestions"].append("Consider specifying camera movement or shot type")
        
        if not analysis["has_lighting_terms"]:
            analysis["suggestions"].append("Add lighting description like 'dramatic lighting'")
        
        if not analysis["has_color_terms"]:
            analysis["suggestions"].append("Consider adding color terms like 'vibrant colors'")
        
        if not analysis["has_film_terms"]:
            analysis["suggestions"].append("Consider adding film emulation terms like 'shot on 35mm film'")
        
        if analysis["has_negative_terms"]:
            negative_found = [term for term in self.negative_terms if term in prompt_lower]
            analysis["suggestions"].append(f"Remove negative terms: {', '.join(negative_found)}")
        
        if analysis["word_count"] < 10:
            analysis["suggestions"].append("Prompt is quite short. Consider adding more descriptive details")
        
        return analysis
    
    def optimize_prompt(self, prompt: str, style: str = "cinematic", 
                       add_quality: bool = True, 
                       add_style: bool = True,
                       add_lighting: bool = True,
                       add_color: bool = True,
                       add_film: bool = False) -> str:
        """Automatically optimize a prompt based on analysis"""
        analysis = self.analyze_prompt(prompt)
        optimized = prompt
        
        # Add quality terms if missing and requested
        if add_quality and not analysis["has_quality_terms"]:
            quality_term = random.choice(self.quality_terms)
            optimized += f", {quality_term}"
        
        # Add style terms if missing and requested
        if add_style and not analysis["has_cinematic_terms"]:
            if style == "cinematic":
                style_term = random.choice(self.cinematic_terms)
                optimized += f", {style_term}"
        
        # Add lighting terms if missing and requested
        if add_lighting and not analysis["has_lighting_terms"]:
            lighting_term = random.choice(self.lighting_terms)
            optimized += f", {lighting_term}"
        
        # Add color terms if missing and requested
        if add_color and not analysis["has_color_terms"]:
            color_term = random.choice(self.color_terms)
            optimized += f", {color_term}"
        
        # Add film terms if missing and requested
        if add_film and not analysis["has_film_terms"]:
            film_term = random.choice(self.film_terms)
            optimized += f", {film_term}"
        
        return optimized.strip()
    
    def generate_variations(self, base_prompt: str, num_variations: int = 3) -> List[str]:
        """Generate variations of a prompt with different styles"""
        variations = []
        
        # First variation: cinematic
        cinematic = self.optimize_prompt(
            base_prompt, 
            style="cinematic",
            add_quality=True,
            add_style=True,
            add_lighting=True,
            add_color=True,
            add_film=True
        )
        variations.append(cinematic)
        
        # Second variation: natural
        natural = self.optimize_prompt(
            base_prompt,
            style="natural",
            add_quality=True,
            add_style=False,
            add_lighting=True,
            add_color=True,
            add_film=False
        )
        natural += ", natural lighting, documentary style"
        variations.append(natural)
        
        # Third variation: artistic
        artistic = self.optimize_prompt(
            base_prompt,
            style="artistic",
            add_quality=True,
            add_style=False,
            add_lighting=True,
            add_color=True,
            add_film=False
        )
        artistic += ", artistic, creative composition, stylized"
        variations.append(artistic)
        
        # Add more variations if requested
        if num_variations > 3:
            for i in range(num_variations - 3):
                # Mix and match different elements
                variation = base_prompt
                
                # Add random quality term
                variation += f", {random.choice(self.quality_terms)}"
                
                # Add random style elements
                categories = [
                    self.cinematic_terms,
                    self.camera_terms,
                    self.lighting_terms,
                    self.color_terms,
                    self.film_terms
                ]
                
                # Add 2-3 random elements from different categories
                for category in random.sample(categories, k=min(3, len(categories))):
                    variation += f", {random.choice(category)}"
                
                variations.append(variation)
        
        return variations[:num_variations]
    
    def create_negative_prompt(self) -> str:
        """Create a standard negative prompt for video generation"""
        return ", ".join([
            "blurry", "low quality", "distorted", "bad anatomy", "disfigured",
            "poorly drawn", "extra limbs", "strange colors", "pixelated",
            "low resolution", "watermark", "signature", "artifacts"
        ])

def main():
    parser = argparse.ArgumentParser(description="Prompt Optimizer for Video Generation")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt to optimize")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze the prompt without optimizing")
    parser.add_argument("--style", type=str, default="cinematic",
                       choices=["cinematic", "natural", "artistic"],
                       help="Style to optimize for")
    parser.add_argument("--variations", type=int, default=0,
                       help="Number of variations to generate (0 for none)")
    
    args = parser.parse_args()
    
    optimizer = PromptOptimizer()
    
    if args.analyze:
        analysis = optimizer.analyze_prompt(args.prompt)
        print("\n📝 Prompt Analysis:")
        print(f"Original prompt: \"{args.prompt}\"")
        print(f"Word count: {analysis['word_count']}")
        print("\nChecklist:")
        for key, value in analysis.items():
            if key.startswith("has_") and key != "has_negative_terms":
                print(f"✓ {key[4:].replace('_', ' ').title()}" if value else f"✗ {key[4:].replace('_', ' ').title()}")
        
        if analysis["has_negative_terms"]:
            print("⚠️ Contains negative terms that may hurt quality")
        
        if analysis["suggestions"]:
            print("\nSuggestions:")
            for suggestion in analysis["suggestions"]:
                print(f"• {suggestion}")
    else:
        optimized = optimizer.optimize_prompt(args.prompt, style=args.style)
        print("\n🔄 Prompt Optimization:")
        print(f"Original: \"{args.prompt}\"")
        print(f"Optimized: \"{optimized}\"")
        
        if args.variations > 0:
            print("\n🎭 Prompt Variations:")
            variations = optimizer.generate_variations(args.prompt, args.variations)
            for i, variation in enumerate(variations):
                print(f"{i+1}. \"{variation}\"")
        
        print("\n❌ Suggested Negative Prompt:")
        print(f"\"{optimizer.create_negative_prompt()}\"")

if __name__ == "__main__":
    main()