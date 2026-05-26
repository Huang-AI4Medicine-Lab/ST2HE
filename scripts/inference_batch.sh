#!/bin/bash
# Batch inference script for ST2HE conversion

# Default paths (update these to match your setup)
MODEL_PATH="${MODEL_PATH:-weights/UnCondGen.pkl}"
INPUT_DIR="${INPUT_DIR:-/path/to/input/images}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/st2he_results}"
PROMPT="${PROMPT:-dapi2he}"
DIRECTION="${DIRECTION:-a2b}"
IMAGE_PREP="${IMAGE_PREP:-no_resize}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "ST2HE Batch Inference"
echo "===================="
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Prompt: $PROMPT"
echo "Direction: $DIRECTION"
echo ""

# Run inference
python src/inference.py \
    --model_path "$MODEL_PATH" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --prompt "$PROMPT" \
    --direction "$DIRECTION" \
    --image_prep "$IMAGE_PREP"

echo ""
echo "Inference complete! Results saved to $OUTPUT_DIR"
