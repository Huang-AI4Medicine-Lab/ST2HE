# ST2HE Vendored Pix2Pix Design

## Goal

Make `st2he` self-contained for inference by vendoring the minimal pix2pix-turbo backbone used in revision analyses.

## Scope

- Vendor only the inference-critical subset of pix2pix-turbo.
- Remove the external `PIX2PIX_TURBO_PATH` dependency.
- Keep weights local to `weights/`.
- Preserve the existing `ST2HEInference` interface where practical.

## Included Components

- `src/st2he_vendor/pix2pix_turbo.py`
- `src/st2he_vendor/model.py`
- `src/st2he_vendor/image_prep.py`

## Excluded Components

- training scripts
- `data/`
- `outputs/`
- `wandb/`
- CycleGAN-specific code

## Runtime Behavior

- `inference.py` should import without immediately requiring runtime ML dependencies.
- Missing weights files should raise `FileNotFoundError` before model initialization.
- Inference should use the vendored pix2pix-turbo class and local weights.

## Weights Provenance

Reviewer-response scripts referenced `model_9501.pkl` weights from:

- `output/pix2pix_turbo/xenium_multi_tissue_512_color/checkpoints/`
- `output/pix2pix_turbo/xenium_multi_tissue_512_color_cond/checkpoints/`

## Open Constraints

- Base SD-Turbo weights still come from Hugging Face at runtime unless cached locally.
- Public weights distribution should use Git LFS or Releases if files grow.
