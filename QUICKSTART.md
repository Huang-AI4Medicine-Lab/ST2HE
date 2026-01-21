# Quick Start Guide

This guide will help you get started with ST2HE quickly.

## Prerequisites

1. Ensure you have the pix2pix-turbo model repository cloned/downloaded
2. Have your trained model checkpoint ready (e.g., `model_25001.pkl`)

## Installation

```bash
cd ST2HE
pip install -r requirements.txt

# Set the path to your pix2pix-turbo repository
export PIX2PIX_TURBO_PATH="/path/to/pix2pix-turbo"
```

## Quick Examples

### 1. Convert a Single Image

```bash
python src/inference.py \
    --model_path /path/to/your/model.pkl \
    --input /path/to/input_image.png \
    --output /path/to/output_image.png
```

### 2. Process Multiple Images

```bash
python src/inference.py \
    --model_path /path/to/your/model.pkl \
    --input /path/to/input/directory \
    --output /path/to/output/directory
```

### 3. Generate Samples with Comparisons

```bash
python src/generate_samples.py \
    --model_path /path/to/your/model.pkl \
    --input /path/to/input/directory \
    --output_dir /path/to/output/directory \
    --num_variations 2
```

## Using the Python API

```python
from src.inference import ST2HEInference

# Initialize
inference = ST2HEInference(
    model_path="/path/to/model.pkl",
    prompt="image of HE",
    direction="a2b"
)

# Convert image
output = inference.predict("/path/to/input.png")
output.save("/path/to/output.png")

# Batch process
inference.predict_batch(
    input_dir="/path/to/inputs",
    output_dir="/path/to/outputs"
)
```

## Common Issues

### Import Error
If you see import errors, make sure the pix2pix-turbo path is correct in `src/inference.py` (line 16).

### CUDA Out of Memory
- Use `--use_fp16` flag for faster inference with less memory
- Process images in smaller batches
- Reduce image size using `--image_prep resize_512x512`

### Model Not Found
Ensure your model checkpoint path is correct and the file exists.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/example_usage.py](examples/example_usage.py) for more examples
- Customize configuration in [configs/default_config.yaml](configs/default_config.yaml)
