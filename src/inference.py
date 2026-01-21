"""
ST2HE Inference Module
Converts spatial transcriptomics data to H&E images using pix2pix turbo model.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from typing import Union, Optional

# Add pix2pix turbo model path to sys.path
# Update this path to point to your pix2pix-turbo repository location
PIX2PIX_TURBO_PATH = os.environ.get("PIX2PIX_TURBO_PATH", "path/to/pix2pix-turbo")
sys.path.insert(0, os.path.join(PIX2PIX_TURBO_PATH, "src"))

try:
    from cyclegan_turbo import CycleGAN_Turbo
    from my_utils.training_utils import build_transform
except ImportError as e:
    raise ImportError(
        f"Could not import pix2pix turbo modules. "
        f"Please ensure the model is available at {PIX2PIX_TURBO_PATH}. Error: {e}"
    )


class ST2HEInference:
    """
    Inference class for ST2HE (Spatial Transcriptomics to H&E) conversion.
    """
    
    def __init__(
        self,
        model_path: str,
        prompt: str = "image of HE",
        direction: str = "a2b",
        image_prep: str = "no_resize",
        use_fp16: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the ST2HE inference model.
        
        Args:
            model_path: Path to the trained pix2pix turbo model checkpoint
            prompt: Text prompt for the model (default: "image of HE")
            direction: Translation direction, either "a2b" or "b2a" (default: "a2b")
            image_prep: Image preparation method (default: "no_resize")
            use_fp16: Whether to use float16 precision for faster inference
            device: Device to run inference on (default: "cuda" if available, else "cpu")
        """
        self.model_path = model_path
        self.prompt = prompt
        self.direction = direction
        self.image_prep = image_prep
        self.use_fp16 = use_fp16
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize model
        print(f"Loading model from {model_path}...")
        self.model = CycleGAN_Turbo(pretrained_name=None, pretrained_path=model_path)
        self.model.eval()
        
        # Move model to device if needed (models from CycleGAN_Turbo are typically on CUDA by default)
        if device != "cuda" and hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        
        if hasattr(self.model, 'unet'):
            self.model.unet.enable_xformers_memory_efficient_attention()
        
        if use_fp16:
            self.model.half()
        
        # Build transform
        self.transform = build_transform(image_prep)
        
        print(f"Model loaded successfully. Using device: {self.device}")
    
    def predict(self, input_image: Union[str, Image.Image]) -> Image.Image:
        """
        Convert spatial transcriptomics image to H&E image.
        
        Args:
            input_image: Path to input image or PIL Image object
            
        Returns:
            PIL Image of the generated H&E image
        """
        # Load image if path is provided
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB')
        
        # Prepare input
        with torch.no_grad():
            input_img = self.transform(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(self.device)
            
            if self.use_fp16:
                x_t = x_t.half()
            
            # Run inference
            output = self.model(x_t, direction=self.direction, caption=self.prompt)
        
        # Convert output to PIL Image
        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)
        
        return output_pil
    
    def predict_batch(
        self,
        input_dir: str,
        output_dir: str,
        image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    ):
        """
        Process a batch of images from a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            image_extensions: Tuple of allowed image extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in image_files:
            print(f"Processing {img_path.name}...")
            try:
                output_img = self.predict(str(img_path))
                output_file = output_path / img_path.name
                output_img.save(output_file)
                print(f"Saved to {output_file}")
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")


def main():
    """Command-line interface for ST2HE inference."""
    parser = argparse.ArgumentParser(
        description="ST2HE: Convert spatial transcriptomics data to H&E images"
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
        help='Path to input image or directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output image or directory for output images'
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
    
    args = parser.parse_args()
    
    # Initialize model
    inference = ST2HEInference(
        model_path=args.model_path,
        prompt=args.prompt,
        direction=args.direction,
        image_prep=args.image_prep,
        use_fp16=args.use_fp16
    )
    
    # Check if input is a directory or file
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        # Batch processing
        inference.predict_batch(str(input_path), str(output_path))
    else:
        # Single image processing
        output_img = inference.predict(str(input_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_path)
        print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
