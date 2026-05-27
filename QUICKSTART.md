# Quick Start Guide

This guide will help you get started with ST2HE quickly.

## Prerequisites

1. Install the Python dependencies from `requirements.txt`
2. Use one of the bundled ST2HE weights in `weights/` such as `weights/UnCondGen.pkl` or `weights/CondGen.pkl`

## Installation

```bash
cd ST2HE
pip install -r requirements.txt
```

## Quick Examples

### 0. Prepare Xenium Inputs

If you are starting from a Xenium sample directory, first generate ST2HE-ready PNG tiles:

```bash
python scripts/generate_xenium_inputs.py \
    --xenium-dir /path/to/xenium_sample \
    --tiles-csv /path/to/tiles.csv \
    --output-dir /path/to/dps_tiles
```

Expected inputs:
- `xenium_sample/outs/morphology.ome.tif`
- `xenium_sample/outs/transcripts.parquet`
- a CSV manifest with `cell_id`, `x_centroid`, `y_centroid`, and optional `tile_id`

Useful flags:
- `--dapi-only`
- `--qv-threshold 20`
- `--resample-scale 0.2125`
- `--tile-size 512`

### 1. Convert a Single Image

```bash
python src/inference.py \
    --model_path weights/UnCondGen.pkl \
    --input /path/to/input_image.png \
    --output /path/to/output_image.png \
    --prompt "dapi2he"
```

Use `weights/CondGen.pkl` with a tissue-specific prompt such as `"This is a breast H&E image"` for conditional generation.

### 2. Process Multiple Images

```bash
python src/inference.py \
    --model_path weights/UnCondGen.pkl \
    --input /path/to/input/directory \
    --output /path/to/output/directory \
    --prompt "dapi2he"
```

### 3. Generate Samples with Comparisons

```bash
python src/generate_samples.py \
    --model_path weights/UnCondGen.pkl \
    --input /path/to/input/directory \
    --output_dir /path/to/output/directory \
    --num_variations 2 \
    --prompt "dapi2he"
```

## Using the Python API

```python
from src.inference import ST2HEInference

# Initialize
inference = ST2HEInference(
    model_path="weights/UnCondGen.pkl",
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
If you see import errors, confirm that `requirements.txt` is installed in the Python environment you are using.

### CUDA Out of Memory
- Use `--use_fp16` flag for faster inference with less memory
- Process images in smaller batches
- Reduce image size using `--image_prep resize_512x512`

### Model Not Found
Ensure your model weights path is correct and the file exists.

### First Run Download
The first inference run may download the base `stabilityai/sd-turbo` weights if they are not already cached.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/example_usage.py](examples/example_usage.py) for more examples
- Customize configuration in [configs/default_config.yaml](configs/default_config.yaml)
