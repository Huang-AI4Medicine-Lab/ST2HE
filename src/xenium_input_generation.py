"""Helpers for generating ST2HE-ready Xenium input tiles."""

import argparse
import colorsys
import functools
import math
import os
from numbers import Integral
from pathlib import Path
import re

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in environments without pandas
    pd = None


CONTROL_PATTERN = re.compile(r"^BLANK|^NegControlCodeword|^NegControlProbe", re.IGNORECASE)
REQUIRED_MANIFEST_COLUMNS = ("cell_id", "x_centroid", "y_centroid")
DEFAULT_TILE_SIZE = 512
DEFAULT_QV_THRESHOLD = 20
DEFAULT_RESAMPLE_SCALE = 0.2125
DEFAULT_PYRAMID_LEVEL = 0
DEFAULT_DPI = 64
DEFAULT_DOT_SIZE = 0.1


def _require_pandas():
    """Return pandas, or raise a clear import error for data-loading paths."""
    if pd is None:
        raise ImportError("pandas is required for Xenium input generation")
    return pd


def _load_numeric_stack():
    """Import numpy and scipy only when the heavy stack is needed."""
    import numpy as np
    from scipy import ndimage

    return np, ndimage


def _load_render_stack():
    """Import rendering dependencies lazily so helper tests stay lightweight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def _load_ome_stack():
    """Import OME-TIFF readers lazily for Xenium file loading."""
    import dask.array as da
    from tifffile import TiffFile
    import zarr

    return TiffFile, zarr, da


def _as_path(path_like):
    """Normalize a filesystem input to Path."""
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _is_missing_value(value):
    """Return True when a required manifest value is null-like."""
    if value is None:
        return True
    if pd is not None and pd.isna(value):
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _is_usable_identifier(value):
    """Return True when a manifest identifier can be used in an output name."""
    if _is_missing_value(value):
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def _decode_feature_name(value):
    """Return a string feature name for parquet payloads that store bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _coerce_image_2d(image):
    """Convert an image payload into a 2D numeric array."""
    np, _ = _load_numeric_stack()
    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D DAPI image after squeezing, found shape {array.shape}")
    return array


def _has_rows(records):
    """Return True when a transcript payload contains rows."""
    try:
        return len(records) > 0
    except TypeError:
        return False


def make_output_name(row):
    """Build the output image name from a manifest row."""
    for key in ("tile_id", "cell_id"):
        value = row.get(key)
        if _is_usable_identifier(value):
            return f"{value}.png"
    raise ValueError("row must include a usable tile_id or cell_id")


def resolve_xenium_paths(xenium_dir):
    """Return the canonical Xenium outs/ paths used by this CLI."""
    xenium_dir = _as_path(xenium_dir)
    outs_dir = xenium_dir / "outs"
    return str(outs_dir / "morphology.ome.tif"), str(outs_dir / "transcripts.parquet")


def find_xenium_files(xenium_dir):
    """Locate the registered DAPI image and transcript parquet for a Xenium sample."""
    dapi_path, transcripts_path = resolve_xenium_paths(xenium_dir)
    dapi_path = _as_path(dapi_path)
    transcripts_path = _as_path(transcripts_path)
    if not dapi_path.exists():
        outs_dir = dapi_path.parent
        dapi_candidates = sorted(outs_dir.glob("*.ome.tif"))
        if not dapi_candidates:
            raise FileNotFoundError(f"no registered Xenium OME-TIFF found under {outs_dir}")
        dapi_path = dapi_candidates[0]
    if not transcripts_path.exists():
        raise FileNotFoundError(f"missing Xenium transcripts parquet at {transcripts_path}")
    return dapi_path, transcripts_path


def load_manifest_csv(tiles_csv):
    """Load a manifest CSV without applying additional transformation."""
    pandas_module = _require_pandas()
    return pandas_module.read_csv(tiles_csv)


def filter_transcripts(transcripts, qv_threshold):
    """Drop control probes and low-quality transcripts."""
    feature_names = transcripts["feature_name"].map(_decode_feature_name)
    keep_mask = ~feature_names.str.contains(CONTROL_PATTERN, na=False)
    keep_mask &= transcripts["qv"] >= qv_threshold
    filtered = transcripts.loc[keep_mask].copy()
    filtered["feature_name"] = feature_names.loc[keep_mask]
    return filtered


def load_xenium_transcripts(transcripts_path, qv_threshold=DEFAULT_QV_THRESHOLD):
    """Load and filter Xenium transcripts from parquet."""
    pandas_module = _require_pandas()
    transcripts = pandas_module.read_parquet(transcripts_path)
    return filter_transcripts(transcripts, qv_threshold=qv_threshold)


def normalize_dapi_image(image, new_min=0, new_max=255):
    """Normalize a numeric 2D image to the requested range."""
    np, _ = _load_numeric_stack()
    image_array = _coerce_image_2d(image).astype(np.float32, copy=False)
    if image_array.size == 0:
        return image_array.astype(np.uint8)

    lo = float(image_array.min())
    hi = float(image_array.max())
    if hi == lo:
        return np.full_like(image_array, fill_value=int(new_min), dtype=np.uint8)

    scale = (new_max - new_min) / (hi - lo)
    normalized = (image_array - lo) * scale + new_min
    return normalized.astype(np.uint8)


def open_ome_tif(path, pyramid_level=DEFAULT_PYRAMID_LEVEL):
    """Open a Xenium OME-TIFF pyramid level as a dask array."""
    TiffFile, zarr, da = _load_ome_stack()
    with TiffFile(path) as tif:
        store = tif.series[0].aszarr(level=pyramid_level)
        array = zarr.open(store, mode="r")
    return da.from_array(array, chunks="auto")


def resample_image(image, scale):
    """Resample a DAPI image by the requested scale factor."""
    _, ndimage = _load_numeric_stack()
    if scale is None or scale == 1:
        return image
    return ndimage.zoom(image, scale)


def load_registered_dapi(
    dapi_path,
    scale=DEFAULT_RESAMPLE_SCALE,
    pyramid_level=DEFAULT_PYRAMID_LEVEL,
    open_image_fn=None,
    resample_fn=None,
    normalize_fn=None,
):
    """Load, resample, and normalize a Xenium DAPI image."""
    open_image_fn = open_image_fn or open_ome_tif
    resample_fn = resample_fn or resample_image
    normalize_fn = normalize_fn or normalize_dapi_image

    dapi_image = open_image_fn(dapi_path, pyramid_level=pyramid_level)
    if hasattr(dapi_image, "compute"):
        dapi_image = dapi_image.compute()
    dapi_image = _coerce_image_2d(dapi_image)
    dapi_image = resample_fn(dapi_image, scale)
    return normalize_fn(dapi_image)


def load_xenium_dapi_image(
    xenium_dir,
    resample_scale=DEFAULT_RESAMPLE_SCALE,
    pyramid_level=DEFAULT_PYRAMID_LEVEL,
):
    """Load the registered Xenium DAPI image from a sample directory."""
    dapi_path, _ = find_xenium_files(xenium_dir)
    return load_registered_dapi(
        dapi_path,
        scale=resample_scale,
        pyramid_level=pyramid_level,
    )


def validate_manifest(manifest):
    """Ensure the manifest includes the required Xenium-native columns."""
    missing_columns = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing_columns:
        raise ValueError(
            "manifest is missing required columns: " + ", ".join(missing_columns)
        )

    invalid_columns = []
    for column in REQUIRED_MANIFEST_COLUMNS:
        if column == "cell_id":
            has_invalid = manifest[column].map(lambda value: not _is_usable_identifier(value)).any()
        else:
            has_invalid = manifest[column].map(_is_missing_value).any()
        if has_invalid:
            invalid_columns.append(column)
    if invalid_columns:
        raise ValueError(
            "manifest has unusable values in required columns: " + ", ".join(invalid_columns)
        )
    return manifest


def validate_tile_size(tile_size):
    """Validate and return a positive even integer tile size."""
    if isinstance(tile_size, bool) or not isinstance(tile_size, Integral):
        raise TypeError("tile_size must be an even integer")
    if tile_size <= 0 or tile_size % 2 != 0:
        raise ValueError("tile_size must be a positive even integer")
    return tile_size


def tile_window(cx, cy, tile_size):
    """Return centered tile bounds for a positive even integer tile size."""
    tile_size = validate_tile_size(tile_size)
    half_size = tile_size / 2
    x1 = math.ceil(cx - half_size)
    x2 = math.ceil(cx + half_size)
    y1 = math.ceil(cy - half_size)
    y2 = math.ceil(cy + half_size)
    return x1, x2, y1, y2


def extract_transcript_window(transcripts, x1, x2, y1, y2):
    """Return transcripts whose locations fall within the inclusive tile bounds."""
    keep_mask = (
        (transcripts["x_location"] >= x1)
        & (transcripts["x_location"] <= x2)
        & (transcripts["y_location"] >= y1)
        & (transcripts["y_location"] <= y2)
    )
    return transcripts.loc[keep_mask].copy()


def create_gene_palette(genes):
    """Create one deterministic RGB color for each unique gene."""
    unique_genes = []
    seen = set()
    for gene in genes:
        if gene in seen:
            continue
        seen.add(gene)
        unique_genes.append(gene)

    palette = {}
    total = len(unique_genes)
    for index, gene in enumerate(unique_genes):
        hue = index / max(total, 1)
        palette[gene] = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return palette


def prepare_transcript_overlay_inputs(transcripts, x1, x2, y1, y2):
    """Extract tile transcripts and the matching per-tile palette."""
    tile_transcripts = extract_transcript_window(transcripts, x1=x1, x2=x2, y1=y1, y2=y2)
    palette = create_gene_palette(tile_transcripts["feature_name"].tolist())
    return tile_transcripts, palette


def save_tile_png(
    dapi_image,
    transcripts,
    x1,
    x2,
    y1,
    y2,
    palette,
    output_path,
    dot_size=DEFAULT_DOT_SIZE,
    dpi=DEFAULT_DPI,
):
    """Render and save one Xenium input tile as a PNG."""
    plt, sns = _load_render_stack()

    tile_width = x2 - x1
    tile_height = y2 - y1
    figure_size = (tile_width / dpi, tile_height / dpi)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=figure_size, dpi=dpi)
    plt.imshow(dapi_image, cmap="gray")
    if _has_rows(transcripts):
        sns.scatterplot(
            x="x_location",
            y="y_location",
            data=transcripts,
            s=dot_size,
            alpha=1,
            hue="feature_name",
            palette=palette,
            legend=False,
        )
    plt.axis("off")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return output_path


def render_tile(dapi_image, transcripts, x1, x2, y1, y2, palette, output_path, renderer_fn=None):
    """Render one tile using an injected renderer, or return render metadata."""
    if renderer_fn is not None:
        return renderer_fn(
            dapi_image=dapi_image,
            transcripts=transcripts,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            palette=palette,
            output_path=output_path,
        )
    return {
        "dapi_image": dapi_image,
        "transcripts": transcripts,
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
        "palette": palette,
        "output_path": output_path,
    }


def _image_shape(image):
    """Return image height and width for array-like image payloads."""
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    height = len(image)
    width = len(image[0]) if height else 0
    return height, width


def generate_tiles(
    manifest,
    dapi_image,
    output_dir,
    tile_size=DEFAULT_TILE_SIZE,
    transcripts=None,
    dapi_only=False,
    prepare_transcripts_fn=None,
    render_tile_fn=None,
):
    """Generate tile render jobs while keeping I/O and rendering injectable."""
    validate_manifest(manifest)
    tile_size = validate_tile_size(tile_size)
    image_height, image_width = _image_shape(dapi_image)
    prepare_transcripts_fn = prepare_transcripts_fn or prepare_transcript_overlay_inputs
    render_tile_fn = render_tile_fn or save_tile_png

    generated = []
    for row in manifest.to_dict("records"):
        x1, x2, y1, y2 = tile_window(row["x_centroid"], row["y_centroid"], tile_size=tile_size)
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            continue

        if dapi_only:
            tile_transcripts = []
            palette = {}
        else:
            if transcripts is None:
                raise ValueError("transcripts are required unless dapi_only is enabled")
            tile_transcripts, palette = prepare_transcripts_fn(
                transcripts=transcripts,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
            )

        output_path = os.path.join(output_dir, make_output_name(row))
        generated.append(
            render_tile(
                dapi_image=dapi_image,
                transcripts=tile_transcripts,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                palette=palette,
                output_path=output_path,
                renderer_fn=render_tile_fn,
            )
        )
    return generated


def generate_xenium_inputs(
    xenium_dir,
    tiles_csv,
    output_dir,
    tile_size=DEFAULT_TILE_SIZE,
    qv_threshold=DEFAULT_QV_THRESHOLD,
    resample_scale=DEFAULT_RESAMPLE_SCALE,
    pyramid_level=DEFAULT_PYRAMID_LEVEL,
    dot_size=DEFAULT_DOT_SIZE,
    dapi_only=False,
):
    """Generate ST2HE-ready Xenium tiles from a sample directory and manifest CSV."""
    _, transcripts_path = resolve_xenium_paths(xenium_dir)
    manifest = load_manifest_csv(tiles_csv)
    if hasattr(manifest, "columns"):
        manifest = validate_manifest(manifest)
    dapi_image = load_xenium_dapi_image(
        xenium_dir,
        resample_scale=resample_scale,
        pyramid_level=pyramid_level,
    )
    transcripts = None if dapi_only else load_xenium_transcripts(transcripts_path, qv_threshold=qv_threshold)
    os.makedirs(output_dir, exist_ok=True)
    renderer = None
    if dot_size != DEFAULT_DOT_SIZE:
        renderer = functools.partial(save_tile_png, dot_size=dot_size)
    return generate_tiles(
        manifest=manifest,
        dapi_image=dapi_image,
        output_dir=output_dir,
        tile_size=tile_size,
        transcripts=transcripts,
        dapi_only=dapi_only,
        render_tile_fn=renderer,
    )


def _parse_tile_size(value):
    """Parse CLI tile size using the shared tile size contract."""
    try:
        parsed = int(value)
        return validate_tile_size(parsed)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_arg_parser():
    """Build the Xenium-native CLI parser."""
    parser = argparse.ArgumentParser(description="Generate Xenium-native ST2HE input tiles.")
    parser.add_argument("--xenium-dir", required=True, help="Xenium sample directory containing outs/")
    parser.add_argument("--tiles-csv", required=True, help="CSV manifest with cell_id,x_centroid,y_centroid")
    parser.add_argument("--output-dir", required=True, help="Directory for generated PNG tiles")
    parser.add_argument("--tile-size", type=_parse_tile_size, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--qv-threshold", type=int, default=DEFAULT_QV_THRESHOLD)
    parser.add_argument("--resample-scale", type=float, default=DEFAULT_RESAMPLE_SCALE)
    parser.add_argument("--pyramid-level", type=int, default=DEFAULT_PYRAMID_LEVEL)
    parser.add_argument("--dot-size", type=float, default=DEFAULT_DOT_SIZE)
    parser.add_argument("--dapi-only", action="store_true", help="Render DAPI-only tiles without transcripts")
    return parser


def main(argv=None):
    """Parse CLI arguments and generate Xenium-native input tiles."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _, transcripts_path = resolve_xenium_paths(args.xenium_dir)
    manifest = load_manifest_csv(args.tiles_csv)
    if hasattr(manifest, "columns"):
        manifest = validate_manifest(manifest)
    dapi_kwargs = {"resample_scale": args.resample_scale}
    if args.pyramid_level != DEFAULT_PYRAMID_LEVEL:
        dapi_kwargs["pyramid_level"] = args.pyramid_level
    dapi_image = load_xenium_dapi_image(args.xenium_dir, **dapi_kwargs)
    transcripts = None
    if not args.dapi_only:
        transcripts = load_xenium_transcripts(
            transcripts_path,
            qv_threshold=args.qv_threshold,
        )
    os.makedirs(args.output_dir, exist_ok=True)
    generate_kwargs = {
        "manifest": manifest,
        "dapi_image": dapi_image,
        "output_dir": args.output_dir,
        "tile_size": args.tile_size,
        "transcripts": transcripts,
        "dapi_only": args.dapi_only,
    }
    if args.dot_size != DEFAULT_DOT_SIZE:
        generate_kwargs["render_tile_fn"] = functools.partial(save_tile_png, dot_size=args.dot_size)
    return generate_tiles(
        **generate_kwargs,
    )
