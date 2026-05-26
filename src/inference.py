"""ST2HE inference entry point built on a vendored pix2pix-turbo backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from PIL import Image as PILImage


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
            model_path: Path to the trained pix2pix-turbo weights file
            prompt: Text prompt for the model (default: "image of HE")
            direction: Translation direction. The vendored ST2HE path supports "a2b".
            image_prep: Image preparation method (default: "no_resize")
            use_fp16: Whether to use float16 precision for faster inference
            device: Device to run inference on (default: "cuda" if available, else "cpu")
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {self.model_path}")

        self.prompt = prompt
        self.direction = direction
        self.image_prep = image_prep
        self.use_fp16 = use_fp16

        if self.direction != "a2b":
            raise ValueError("The vendored pix2pix-turbo ST2HE inference only supports direction='a2b'.")

        runtime = self._load_runtime_dependencies()
        self._image_module = runtime["image"]
        self._torch = runtime["torch"]
        self._transforms = runtime["transforms"]

        if device is None:
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading model from {self.model_path}...")
        self.model = runtime["model_class"](
            pretrained_name=None,
            pretrained_path=str(self.model_path),
            device=self.device,
            enable_xformers=True,
        )
        self.model.set_eval()

        if use_fp16:
            self.model.half()

        self.transform = runtime["build_transform"](image_prep)
        print(f"Model loaded successfully. Using device: {self.device}")

    @staticmethod
    def _load_runtime_dependencies():
        from PIL import Image
        import torch
        from torchvision import transforms

        from st2he_vendor import Pix2Pix_Turbo, build_transform

        return {
            "build_transform": build_transform,
            "image": Image,
            "model_class": Pix2Pix_Turbo,
            "torch": torch,
            "transforms": transforms,
        }

    def predict(self, input_image: Union[str, Path, "PILImage"]) -> "PILImage":
        """
        Convert spatial transcriptomics image to H&E image.
        
        Args:
            input_image: Path to input image or PIL Image object
            
        Returns:
            PIL Image of the generated H&E image
        """
        # Load image if path is provided
        if isinstance(input_image, (str, Path)):
            input_image = self._image_module.open(input_image).convert("RGB")

        with self._torch.no_grad():
            input_img = self.transform(input_image)
            x_t = self._transforms.ToTensor()(input_img)
            x_t = self._transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            )(x_t).unsqueeze(0).to(self.device)

            if self.use_fp16:
                x_t = x_t.half()

            output = self.model(x_t, prompt=self.prompt)

        output_pil = self._transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize(
            (input_image.width, input_image.height),
            self._image_module.LANCZOS,
        )
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
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"Found {len(image_files)} images to process")

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
        help='Path to the trained pix2pix-turbo weights file'
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
        choices=["a2b"],
        help='Translation direction (only "a2b" is supported)'
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
