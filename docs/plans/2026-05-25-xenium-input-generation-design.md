# Xenium Input Generation Design

## Goal

Add a reusable ST2HE utility that generates tiled model-input PNGs from Xenium-native data, using the Xenium breast pipeline as the behavioral reference.

## Scope

- Support Xenium-native inputs only in the first version.
- Generate tiled DAPI-plus-transcript images ready for ST2HE inference.
- Use a user-provided tile manifest rather than inventing a sampling policy.

## Input Contract

Required inputs:
- Xenium sample root containing `outs/transcripts.parquet`
- Xenium OME-TIFF DAPI image under `outs/*.ome.tif`
- Tile manifest CSV with `cell_id`, `x_centroid`, and `y_centroid`

Optional manifest fields:
- `tile_id`
- `label`

## Output Contract

- One PNG per manifest row
- Default tile size `512x512`
- Output naming from `tile_id` if present, otherwise `cell_id`

## Behavioral Reference

The implementation should mirror the existing breast-generation logic in:
- `revs/R2-M3-M4/02_generate_dps.py`

Key inherited behaviors:
- registered DAPI loading from OME-TIFF
- transcript filtering by QV threshold
- removal of control/blank probes
- centered tiles around manifest centroids
- per-tile randomized gene-color palettes

## Public CLI Shape

Single reusable script under `scripts/` with arguments for:
- Xenium data root
- tile manifest path
- output directory
- tile size
- QV threshold
- dot size
- DAPI resample scale
- RNG seed
- max tiles
- optional DAPI-only mode

## Non-Goals

- no generic non-Xenium tabular format yet
- no full-slide mosaic output
- no built-in tile selection or sampling policy

