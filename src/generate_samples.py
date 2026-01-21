"""
ST2HE Sample Generation
Generate sample H&E images from spatial transcriptomics data with various configurations.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import ST2HEInference


def generate_samples(
    model_path: str,
    input_images: List[str],
    output_dir: str,
    prompt: str = "image of HE",
    direction: str = "a2b",
    image_prep: str = "no_resize",
    use_fp16: bool = False,
    num_variations: int = 1,
    save_comparison: bool = True
):
    """
    Generate sample H&E images with various configurations.
    
    Args:
        model_path: Path to the trained model
        input_images: List of paths to input images
        output_dir: Directory to save output images
        prompt: Text prompt for generation
        direction: Translation direction
        image_prep: Image preparation method
        use_fp16: Use float16 precision
        num_variations: Number of variations to generate per input
        save_comparison: Whether to save side-by-side comparison images
    """
    # Initialize inference model
    inference = ST2HEInference(
        model_path=model_path,
        prompt=prompt,
        direction=direction,
        image_prep=image_prep,
        use_fp16=use_fp16
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each input image
    for input_img_path in input_images:
        input_path = Path(input_img_path)
        if not input_path.exists():
            print(f"Warning: {input_img_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {input_path.name}...")
        
        # Load input image
        input_img = Image.open(input_path).convert('RGB')
        
        # Generate variations
        generated_images = []
        for i in range(num_variations):
            print(f"  Generating variation {i+1}/{num_variations}...")
            output_img = inference.predict(input_img)
            generated_images.append(output_img)
            
            # Save individual output
            output_file = output_path / f"{input_path.stem}_variation_{i+1}{input_path.suffix}"
            output_img.save(output_file)
            print(f"  Saved to {output_file}")
        
        # Save comparison image if requested
        if save_comparison and generated_images:
            comparison = create_comparison_image(input_img, generated_images[0])
            comparison_file = output_path / f"{input_path.stem}_comparison.png"
            comparison.save(comparison_file)
            print(f"  Comparison saved to {comparison_file}")


def create_comparison_image(input_img: Image.Image, output_img: Image.Image) -> Image.Image:
    """
    Create a side-by-side comparison image.
    
    Args:
        input_img: Input spatial transcriptomics image
        output_img: Generated H&E image
        
    Returns:
        PIL Image with side-by-side comparison
    """
    # Resize images to same height
    height = max(input_img.height, output_img.height)
    input_resized = input_img.resize(
        (int(input_img.width * height / input_img.height), height),
        Image.LANCZOS
    )
    output_resized = output_img.resize(
        (int(output_img.width * height / output_img.height), height),
        Image.LANCZOS
    )
    
    # Create comparison image
    comparison = Image.new('RGB', (input_resized.width + output_resized.width, height))
    comparison.paste(input_resized, (0, 0))
    comparison.paste(output_resized, (input_resized.width, 0))
    
    return comparison


def create_sample_grid(
    input_images: List[str],
    generated_images: List[List[str]],
    output_file: str,
    grid_size: Optional[Tuple[int, int]] = None
):
    """
    Create a grid of sample images for visualization.
    
    Args:
        input_images: List of input image paths
        generated_images: List of lists, where each inner list contains paths to generated images
        output_file: Path to save the grid image
        grid_size: Tuple of (rows, cols) for grid layout (auto-determined if None)
    """
    from PIL import Image
    
    if len(input_images) != len(generated_images):
        raise ValueError("Number of input images must match number of generated image lists")
    
    # Load all images
    all_images = []
    for inp_img, gen_imgs in zip(input_images, generated_images):
        all_images.append(Image.open(inp_img).convert('RGB'))
        for gen_img in gen_imgs:
            all_images.append(Image.open(gen_img).convert('RGB'))
    
    # Determine grid size
    if grid_size is None:
        cols = max(len(gen_imgs) + 1 for gen_imgs in generated_images)  # +1 for input
        rows = len(input_images)
    else:
        rows, cols = grid_size
    
    # Resize all images to same size
    img_size = (256, 256)
    all_images = [img.resize(img_size, Image.LANCZOS) for img in all_images]
    
    # Create grid
    grid_img = Image.new('RGB', (cols * img_size[0], rows * img_size[1]))
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < len(all_images):
                grid_img.paste(all_images[idx], (col * img_size[0], row * img_size[1]))
                idx += 1
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_file)
    print(f"Grid image saved to {output_file}")


def main():
    """Command-line interface for sample generation."""
    parser = argparse.ArgumentParser(
        description="Generate sample H&E images from spatial transcriptomics data"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained pix2pix turbo model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        nargs='+',
        help='Path(s) to input image(s) or directory containing images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output images'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="image of HE",
        help='Text prompt for the model (default: "image of HE")'
    )
    parser.add_argument(
        '--direction',
        type=str,
        default="a2b",
        choices=["a2b", "b2a"],
        help='Translation direction (default: "a2b")'
    )
    parser.add_argument(
        '--image_prep',
        type=str,
        default="no_resize",
        help='Image preparation method (default: "no_resize")'
    )
    parser.add_argument(
        '--use_fp16',
        action='store_true',
        help='Use Float16 precision for faster inference'
    )
    parser.add_argument(
        '--num_variations',
        type=int,
        default=1,
        help='Number of variations to generate per input (default: 1)'
    )
    parser.add_argument(
        '--no_comparison',
        action='store_true',
        help='Do not save comparison images'
    )
    
    args = parser.parse_args()
    
    # Collect input images
    input_images = []
    for inp in args.input:
        inp_path = Path(inp)
        if inp_path.is_dir():
            # Add all images in directory
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                input_images.extend(inp_path.glob(f"*{ext}"))
                input_images.extend(inp_path.glob(f"*{ext.upper()}"))
        else:
            input_images.append(inp_path)
    
    if not input_images:
        print("No input images found!")
        return
    
    print(f"Found {len(input_images)} input image(s)")
    
    # Generate samples
    generate_samples(
        model_path=args.model_path,
        input_images=[str(img) for img in input_images],
        output_dir=args.output_dir,
        prompt=args.prompt,
        direction=args.direction,
        image_prep=args.image_prep,
        use_fp16=args.use_fp16,
        num_variations=args.num_variations,
        save_comparison=not args.no_comparison
    )


if __name__ == "__main__":
    main()
