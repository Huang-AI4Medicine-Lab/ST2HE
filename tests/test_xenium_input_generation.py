"""Unit tests for Xenium input generation helpers."""

from pathlib import Path
from decimal import Decimal
import math
import sys
import unittest
from unittest import mock

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in environments without pandas
    pd = None

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import xenium_input_generation
from xenium_input_generation import (
    build_arg_parser,
    create_gene_palette,
    extract_transcript_window,
    filter_transcripts,
    generate_tiles,
    load_manifest_csv,
    load_xenium_transcripts,
    main,
    make_output_name,
    resolve_xenium_paths,
    tile_window,
    validate_manifest,
)


class TestXeniumInputGeneration(unittest.TestCase):
    """Test cases for Xenium input generation helpers."""

    def test_make_output_name_prefers_tile_id(self):
        """tile_id should be preferred when present."""
        row = {"tile_id": "tile_a", "cell_id": 17}
        self.assertEqual(make_output_name(row), "tile_a.png")

    def test_make_output_name_falls_back_to_cell_id(self):
        """cell_id should be used when tile_id is missing."""
        row = {"cell_id": 17}
        self.assertEqual(make_output_name(row), "17.png")

    @unittest.skipUnless(pd is not None, "pandas is required for pd.NA coverage")
    def test_make_output_name_falls_back_when_tile_id_is_pandas_na(self):
        """pandas missing tile_id values should fall back to cell_id."""
        row = {"tile_id": pd.NA, "cell_id": 17}
        self.assertEqual(make_output_name(row), "17.png")

    def test_make_output_name_falls_back_when_tile_id_is_nan(self):
        """NaN tile_id values should fall back to cell_id."""
        row = {"tile_id": math.nan, "cell_id": 17}
        self.assertEqual(make_output_name(row), "17.png")

    def test_make_output_name_falls_back_when_tile_id_is_nan_without_pandas(self):
        """NaN tile_id should still fall back when pandas is unavailable."""
        row = {"tile_id": math.nan, "cell_id": 17}
        with mock.patch.object(xenium_input_generation, "pd", None):
            self.assertEqual(make_output_name(row), "17.png")

    def test_make_output_name_falls_back_when_tile_id_is_decimal_nan_without_pandas(self):
        """Non-builtin NaN-like identifiers should be rejected without pandas."""
        row = {"tile_id": Decimal("NaN"), "cell_id": 17}
        with mock.patch.object(xenium_input_generation, "pd", None):
            self.assertEqual(make_output_name(row), "17.png")

    def test_make_output_name_falls_back_when_tile_id_is_empty(self):
        """Empty tile_id values should fall back to cell_id."""
        row = {"tile_id": "", "cell_id": 17}
        self.assertEqual(make_output_name(row), "17.png")

    def test_make_output_name_raises_when_no_usable_identifier_exists(self):
        """A row without a usable tile_id or cell_id should fail explicitly."""
        row = {"tile_id": math.nan, "cell_id": ""}
        with self.assertRaises(ValueError):
            make_output_name(row)

    @unittest.skipUnless(pd is not None, "pandas is required for transcript filtering")
    def test_filter_transcripts_removes_controls_and_low_qv_rows(self):
        """Control probes and low-quality transcripts should be excluded."""
        transcripts = pd.DataFrame(
            [
                {"feature_name": b"EPCAM", "qv": 25},
                {"feature_name": b"blank_1", "qv": 30},
                {"feature_name": "NegControlCodeword_7", "qv": 30},
                {"feature_name": "NegControlProbe_2", "qv": 30},
                {"feature_name": "VIM", "qv": 10},
            ]
        )

        filtered = filter_transcripts(transcripts, qv_threshold=20)

        self.assertEqual(filtered["feature_name"].tolist(), ["EPCAM"])
        self.assertEqual(filtered["qv"].tolist(), [25])

    def test_create_gene_palette_returns_one_color_per_gene(self):
        """Palette creation should return a single color for each unique gene."""
        palette = create_gene_palette(["EPCAM", "VIM", "EPCAM", "COL1A1"])

        self.assertEqual(set(palette), {"EPCAM", "VIM", "COL1A1"})
        self.assertEqual(len(palette), 3)

    @unittest.skipUnless(pd is not None, "pandas is required for transcript window extraction")
    def test_extract_transcript_window_respects_tile_bounds(self):
        """Window extraction should include only transcripts inside the tile bounds."""
        transcripts = pd.DataFrame(
            [
                {"feature_name": "A", "x_location": 10, "y_location": 10},
                {"feature_name": "B", "x_location": 20, "y_location": 20},
                {"feature_name": "C", "x_location": 21, "y_location": 20},
                {"feature_name": "D", "x_location": 20, "y_location": 21},
                {"feature_name": "E", "x_location": 9, "y_location": 15},
                {"feature_name": "F", "x_location": 15, "y_location": 9},
            ]
        )

        filtered = extract_transcript_window(transcripts, x1=10, x2=20, y1=10, y2=20)

        self.assertEqual(filtered["feature_name"].tolist(), ["A", "B"])

    @unittest.skipUnless(pd is not None, "pandas is required for tile generation")
    def test_generate_tiles_dapi_only_skips_transcript_overlay_input_preparation(self):
        """DAPI-only generation should not prepare transcript overlay inputs."""
        manifest = pd.DataFrame(
            [{"cell_id": "cell-1", "x_centroid": 16, "y_centroid": 16}]
        )
        dapi_image = [[0] * 32 for _ in range(32)]
        prepare_transcripts = mock.Mock(side_effect=AssertionError("should not be called"))
        render_tile = mock.Mock()

        generate_tiles(
            manifest,
            dapi_image,
            output_dir="/tmp/out",
            tile_size=16,
            dapi_only=True,
            prepare_transcripts_fn=prepare_transcripts,
            render_tile_fn=render_tile,
        )

        prepare_transcripts.assert_not_called()
        render_tile.assert_called_once()
        self.assertEqual(render_tile.call_args.kwargs["transcripts"], [])
        self.assertEqual(render_tile.call_args.kwargs["palette"], {})

    def test_tile_window_centers_on_centroid_for_parameterized_tile_size(self):
        """Tile bounds should center on the centroid using the requested tile size."""
        self.assertEqual(tile_window(100, 120, tile_size=512), (-156, 356, -136, 376))
        self.assertEqual(tile_window(100, 120, tile_size=128), (36, 164, 56, 184))

    def test_tile_window_rejects_non_integer_tile_size(self):
        """Tile size should be an integer number of pixels."""
        with self.assertRaisesRegex(TypeError, "tile_size must be an even integer"):
            tile_window(100, 120, tile_size=128.0)

    def test_tile_window_rejects_non_positive_or_odd_tile_size(self):
        """Tile size should be positive and even."""
        with self.assertRaisesRegex(ValueError, "tile_size must be a positive even integer"):
            tile_window(100, 120, tile_size=0)
        with self.assertRaisesRegex(ValueError, "tile_size must be a positive even integer"):
            tile_window(100, 120, tile_size=127)

    @unittest.skipUnless(pd is not None, "pandas is required for manifest validation")
    def test_validate_manifest_requires_centroid_columns(self):
        """Manifest validation should reject rows without Xenium centroid coordinates."""
        manifest = pd.DataFrame([{"cell_id": "cell-1"}])

        with self.assertRaisesRegex(
            ValueError,
            "manifest is missing required columns: x_centroid, y_centroid",
        ):
            validate_manifest(manifest)

    @unittest.skipUnless(pd is not None, "pandas is required for manifest validation")
    def test_validate_manifest_rejects_null_required_values(self):
        """Manifest validation should reject null ids and centroid coordinates."""
        manifest = pd.DataFrame(
            [
                {"cell_id": "cell-1", "x_centroid": 10.0, "y_centroid": 20.0},
                {"cell_id": None, "x_centroid": 11.0, "y_centroid": 21.0},
                {"cell_id": "cell-3", "x_centroid": math.nan, "y_centroid": 22.0},
                {"cell_id": "cell-4", "x_centroid": 13.0, "y_centroid": pd.NA},
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "manifest has unusable values in required columns: cell_id, x_centroid, y_centroid",
        ):
            validate_manifest(manifest)

    def test_build_arg_parser_accepts_required_args_and_defaults_tile_size(self):
        """Xenium-native parser should require path inputs and default tile size."""
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--xenium-dir",
                "/tmp/xenium",
                "--tiles-csv",
                "/tmp/tiles.csv",
                "--output-dir",
                "/tmp/out",
            ]
        )

        self.assertEqual(args.xenium_dir, "/tmp/xenium")
        self.assertEqual(args.tiles_csv, "/tmp/tiles.csv")
        self.assertEqual(args.output_dir, "/tmp/out")
        self.assertEqual(args.tile_size, 512)
        self.assertFalse(args.dapi_only)
        self.assertEqual(args.qv_threshold, 20)

    def test_resolve_xenium_paths_uses_standard_outs_layout(self):
        """Path resolution should target the standard Xenium outs payloads."""
        dapi_path, transcripts_path = resolve_xenium_paths("/tmp/xenium-sample")

        self.assertEqual(dapi_path, "/tmp/xenium-sample/outs/morphology.ome.tif")
        self.assertEqual(
            transcripts_path,
            "/tmp/xenium-sample/outs/transcripts.parquet",
        )

    @unittest.skipUnless(pd is not None, "pandas is required for manifest loading")
    def test_load_manifest_csv_reads_manifest_rows(self):
        """Manifest loading should parse a CSV into a DataFrame."""
        with unittest.mock.patch.object(pd, "read_csv", return_value="manifest") as read_csv:
            manifest = load_manifest_csv("/tmp/tiles.csv")

        self.assertEqual(manifest, "manifest")
        read_csv.assert_called_once_with("/tmp/tiles.csv")

    def test_build_arg_parser_rejects_invalid_tile_size(self):
        """CLI parsing should reject tile sizes that violate the tile contract."""
        parser = build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--xenium-dir",
                    "/tmp/xenium",
                    "--tiles-csv",
                    "/tmp/tiles.csv",
                    "--output-dir",
                    "/tmp/out",
                    "--tile-size",
                    "127",
                ]
            )

    def test_main_runs_generation_with_dapi_only_mode(self):
        """Main should load Xenium inputs and dispatch manifest-driven tile generation."""
        manifest = object()
        dapi_image = object()
        generated = [{"output_path": "/tmp/out/tile-1.png"}]

        with mock.patch.object(
            xenium_input_generation,
            "load_manifest_csv",
            return_value=manifest,
        ) as load_manifest, mock.patch.object(
            xenium_input_generation,
            "find_xenium_files",
            return_value=("/tmp/xenium/outs/morphology.ome.tif", "/tmp/xenium/outs/transcripts.parquet"),
        ) as find_files, mock.patch.object(
            xenium_input_generation,
            "load_xenium_dapi_image",
            return_value=dapi_image,
        ) as load_dapi, mock.patch.object(
            xenium_input_generation,
            "generate_tiles",
            return_value=generated,
        ) as generate, mock.patch.object(
            xenium_input_generation.os,
            "makedirs",
        ) as makedirs:
            result = main(
                [
                    "--xenium-dir",
                    "/tmp/xenium",
                    "--tiles-csv",
                    "/tmp/tiles.csv",
                    "--output-dir",
                    "/tmp/out",
                    "--dapi-only",
                ]
            )

        self.assertEqual(result, generated)
        load_manifest.assert_called_once_with("/tmp/tiles.csv")
        load_dapi.assert_called_once_with("/tmp/xenium", resample_scale=0.2125)
        makedirs.assert_called_once_with("/tmp/out", exist_ok=True)
        generate.assert_called_once_with(
            manifest=manifest,
            dapi_image=dapi_image,
            output_dir="/tmp/out",
            tile_size=512,
            transcripts=None,
            dapi_only=True,
        )

    def test_main_rejects_invalid_tile_size(self):
        """Main should fail fast when CLI tile size is invalid."""
        with self.assertRaises(SystemExit):
            main(
                [
                    "--xenium-dir",
                    "/tmp/xenium",
                    "--tiles-csv",
                    "/tmp/tiles.csv",
                    "--output-dir",
                    "/tmp/out",
                    "--tile-size",
                    "0",
                ]
            )

    @unittest.skipUnless(pd is not None, "pandas is required for transcript loading")
    def test_load_xenium_transcripts_applies_default_quality_filter(self):
        """Transcript loading should delegate to parquet IO then apply the shared filters."""
        transcripts = pd.DataFrame(
            [
                {"feature_name": b"EPCAM", "qv": 25},
                {"feature_name": b"BLANK_1", "qv": 30},
            ]
        )

        with mock.patch.object(pd, "read_parquet", return_value=transcripts) as read_parquet:
            filtered = load_xenium_transcripts("/tmp/transcripts.parquet", qv_threshold=20)

        read_parquet.assert_called_once_with("/tmp/transcripts.parquet")
        self.assertEqual(filtered["feature_name"].tolist(), ["EPCAM"])


if __name__ == "__main__":
    unittest.main()
