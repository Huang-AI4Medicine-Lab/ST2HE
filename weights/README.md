ST2HE expects pretrained pix2pix-turbo weights to live in this folder.

Included files:
- `UnCondGen.pkl` — bundled ST2HE weights file copied from the unconditioned review-time model below
- `CondGen.pkl` — bundled ST2HE weights file copied from the conditioned review-time model below

Revision provenance:
- `model_9501.pkl` under `output/pix2pix_turbo/xenium_multi_tissue_512_color/checkpoints/`
  was used by `revs/R2-M3-M4/03_run_inference.sh` and is the source of `UnCondGen.pkl`.
- `model_9501.pkl` under `output/pix2pix_turbo/xenium_multi_tissue_512_color_cond/checkpoints/`
  was used by `revs/R2-M5/01_run_inference.sh` and is the source of `CondGen.pkl`.

If you publish these weights on GitHub, prefer Git LFS or GitHub Releases once files exceed standard git-friendly sizes.
