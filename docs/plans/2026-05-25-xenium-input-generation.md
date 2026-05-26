# Xenium Input Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable Xenium-native CLI that generates ST2HE-ready tiled input PNGs from DAPI and transcript coordinates using a manifest of tile centroids.

**Architecture:** Extract the reusable breast-reference logic into a small library module under `src/`, then expose a thin CLI in `scripts/`. Keep the first version manifest-driven, Xenium-specific, and test the non-heavy logic directly while isolating filesystem and parsing behavior.

**Tech Stack:** Python, pandas, numpy, scipy, matplotlib, seaborn, tifffile, zarr, dask, Pillow, unittest

---

### Task 1: Add failing tests for manifest and naming behavior

**Files:**
- Create: `tests/test_xenium_input_generation.py`
- Reference: `revs/R2-M3-M4/02_generate_dps.py`

**Step 1: Write the failing test**

```python
def test_make_output_name_prefers_tile_id():
    row = {"tile_id": "tile_a", "cell_id": 17}
    assert make_output_name(row) == "tile_a.png"


def test_make_output_name_falls_back_to_cell_id():
    row = {"cell_id": 17}
    assert make_output_name(row) == "17.png"
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: FAIL because the module/functions do not exist yet.

**Step 3: Write minimal implementation**

Create a new reusable module with `make_output_name`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: PASS for the new naming tests.

**Step 5: Commit**

```bash
git add tests/test_xenium_input_generation.py src/xenium_input_generation.py
git commit -m "test: add xenium tile naming coverage"
```

### Task 2: Add failing tests for transcript filtering and bounds math

**Files:**
- Modify: `tests/test_xenium_input_generation.py`
- Modify: `src/xenium_input_generation.py`

**Step 1: Write the failing test**

```python
def test_filter_transcripts_removes_controls_and_low_qv():
    df = pd.DataFrame(
        [
            {"feature_name": "EPCAM", "qv": 25},
            {"feature_name": "BLANK_1", "qv": 30},
            {"feature_name": "NegControlProbe_1", "qv": 30},
            {"feature_name": "VIM", "qv": 10},
        ]
    )
    kept = filter_transcripts(df, qv_threshold=20)
    assert kept["feature_name"].tolist() == ["EPCAM"]


def test_tile_window_centers_on_centroid():
    assert tile_window(100, 120, tile_size=512) == (-156, 356, -136, 376)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: FAIL because filter/bounds helpers are incomplete.

**Step 3: Write minimal implementation**

Add:
- transcript decoding helper
- control/blank filtering
- QV filtering
- tile window math parameterized by `tile_size`

**Step 4: Run test to verify it passes**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_xenium_input_generation.py src/xenium_input_generation.py
git commit -m "feat: add xenium transcript filtering helpers"
```

### Task 3: Add failing tests for manifest validation and CLI parsing

**Files:**
- Modify: `tests/test_xenium_input_generation.py`
- Modify: `src/xenium_input_generation.py`
- Create: `scripts/generate_xenium_inputs.py`

**Step 1: Write the failing test**

```python
def test_validate_manifest_requires_centroid_columns():
    manifest = pd.DataFrame({"cell_id": [1], "x_centroid": [10]})
    with self.assertRaises(ValueError):
        validate_manifest(manifest)
```

Add a parser smoke test:

```python
def test_build_arg_parser_accepts_required_args():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--xenium-dir", "sample",
            "--tiles-csv", "tiles.csv",
            "--output-dir", "out",
        ]
    )
    assert args.tile_size == 512
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: FAIL for missing validators/parser functions.

**Step 3: Write minimal implementation**

Add:
- manifest validation
- argument parser builder
- thin script entrypoint that calls the library module

**Step 4: Run test to verify it passes**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_xenium_input_generation.py src/xenium_input_generation.py scripts/generate_xenium_inputs.py
git commit -m "feat: scaffold xenium input generation cli"
```

### Task 4: Implement the reusable library around the breast reference pipeline

**Files:**
- Modify: `src/xenium_input_generation.py`
- Reference: `revs/R2-M3-M4/02_generate_dps.py`

**Step 1: Write the failing test**

Add focused tests for:
- palette creation returns one color per gene
- transcript window filtering respects tile bounds
- DAPI-only mode skips transcript overlay input preparation

**Step 2: Run test to verify it fails**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: FAIL on new helper behavior.

**Step 3: Write minimal implementation**

Implement:
- OME-TIFF opening
- DAPI resampling/normalization
- transcript window extraction
- palette creation
- tile rendering function
- top-level `generate_tiles(...)`

Keep heavy IO in clearly separated functions so unit tests can stub inputs.

**Step 4: Run test to verify it passes**

Run: `python -m unittest -q tests.test_xenium_input_generation`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_xenium_input_generation.py src/xenium_input_generation.py
git commit -m "feat: implement xenium tile generation library"
```

### Task 5: Wire docs, dependencies, and usage examples

**Files:**
- Modify: `README.md`
- Modify: `QUICKSTART.md`
- Modify: `examples/example_usage.py`
- Modify: `requirements.txt`

**Step 1: Write the failing test**

Use a lightweight documentation check in the plan sense: identify missing dependencies and missing script references before updating docs.

**Step 2: Run check to verify the gaps**

Run:
- `rg -n "generate_xenium_inputs|xenium input" README.md QUICKSTART.md examples/example_usage.py`
- `rg -n "pandas|matplotlib|seaborn|tifffile|zarr|dask" requirements.txt`

Expected: missing references/dependencies.

**Step 3: Write minimal implementation**

Update:
- dependency list for the new Xenium pipeline
- README quick example
- QUICKSTART data-prep example
- example usage comments or a short snippet

**Step 4: Run verification**

Run:
- `python -m py_compile src/xenium_input_generation.py scripts/generate_xenium_inputs.py`
- `python -m unittest -q tests.test_xenium_input_generation tests.test_inference`

Expected: PASS

**Step 5: Commit**

```bash
git add README.md QUICKSTART.md examples/example_usage.py requirements.txt src/xenium_input_generation.py scripts/generate_xenium_inputs.py tests/test_xenium_input_generation.py
git commit -m "feat: document xenium input generation workflow"
```

### Task 6: Final verification and repo review

**Files:**
- Review: `scripts/generate_xenium_inputs.py`
- Review: `src/xenium_input_generation.py`
- Review: `tests/test_xenium_input_generation.py`

**Step 1: Run the full verification suite**

Run:

```bash
python -m unittest -q tests.test_xenium_input_generation tests.test_inference
python -m py_compile src/xenium_input_generation.py scripts/generate_xenium_inputs.py src/inference.py
git diff --stat
```

Expected: tests pass, compile succeeds, diff is scoped.

**Step 2: Review error messages and CLI help**

Run:

```bash
python scripts/generate_xenium_inputs.py --help
```

Expected: clear Xenium-native arguments and defaults.

**Step 3: Commit**

```bash
git add .
git commit -m "feat: add reusable xenium input generation script"
```
