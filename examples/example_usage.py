"""
Example usage of ST2HE inference.
This script demonstrates how to use the ST2HE package for converting
spatial transcriptomics images to H&E images.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from inference import ST2HEInference


def example_single_image():
    """Example: Convert a single image."""
    print("Example 1: Single Image Conversion")
    print("-" * 40)
    
    # Initialize model
    # Update this path to point to your trained model checkpoint
    model_path = "/path/to/your/model/checkpoint.pkl"
    
    inference = ST2HEInference(
        model_path=model_path,
        prompt="image of HE",
        direction="a2b",
        image_prep="no_resize",
        use_fp16=False  # Set to True if you have a compatible GPU
    )
    
    # Convert image
    input_image = "/path/to/your/input/image.png"
    output_image = "/path/to/your/output/image.png"
    
    print(f"Converting {input_image}...")
    output_img = inference.predict(input_image)
    output_img.save(output_image)
    print(f"Saved to {output_image}\n")


def example_batch_processing():
    """Example: Process multiple images."""
    print("Example 2: Batch Processing")
    print("-" * 40)
    
    # Update this path to point to your trained model checkpoint
    model_path = "/path/to/your/model/checkpoint.pkl"
    
    inference = ST2HEInference(
        model_path=model_path,
        prompt="image of HE",
        direction="a2b"
    )
    
    input_dir = "/path/to/input/directory"
    output_dir = "/path/to/output/directory"
    
    print(f"Processing images in {input_dir}...")
    inference.predict_batch(input_dir, output_dir)
    print(f"Results saved to {output_dir}\n")


def example_custom_configuration():
    """Example: Using custom configuration."""
    print("Example 3: Custom Configuration")
    print("-" * 40)
    
    # Update this path to point to your trained model checkpoint
    model_path = "/path/to/your/model/checkpoint.pkl"
    
    # Custom configuration
    inference = ST2HEInference(
        model_path=model_path,
        prompt="high quality image of HE stained tissue",
        direction="a2b",
        image_prep="resize_512x512",  # Resize to 512x512
        use_fp16=True  # Use float16 for faster inference
    )
    
    input_image = "/path/to/your/input/image.png"
    output_image = "/path/to/your/output/image.png"
    
    print("Converting with custom configuration...")
    output_img = inference.predict(input_image)
    output_img.save(output_image)
    print(f"Saved to {output_image}\n")


if __name__ == "__main__":
    print("ST2HE Example Usage")
    print("=" * 40)
    print("\nNote: Update the paths in this script to match your setup.")
    print("Uncomment the example you want to run.\n")
    
    # Uncomment the example you want to run:
    # example_single_image()
    # example_batch_processing()
    # example_custom_configuration()
    
    print("Please uncomment one of the examples above to run it.")
