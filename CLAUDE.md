# fishtank — Claude Code Notes

## Project overview

`fishtank` is a Python library for processing MERFISH (Multiplexed Error-Robust Fluorescence In Situ Hybridization) imaging data. It covers the full analysis pipeline: reading raw images, illumination correction, spot detection, spot decoding, cell segmentation, and visualization.

Maintained by William Colgan (wcolgan@mit.edu). Source: https://github.com/jweissmanlab/fishtank

## Running tests

The project uses a `fishtank` conda environment. Use `conda run` or the direct path to pytest:

```bash
# Run all tests
conda run -n fishtank pytest tests/

# Run a specific test file
conda run -n fishtank pytest tests/test_read.py -v

# Run a specific test
conda run -n fishtank pytest tests/test_read.py::test_read_img -v

# Run slow tests (skipped by default)
conda run -n fishtank pytest tests/ --runslow
```

Direct path (faster, no conda overhead):
```bash
/lab/solexa_weissman/wcolgan/tools/miniconda3/envs/fishtank/bin/pytest tests/
```

`python -m pytest` and bare `pytest` do NOT work — pytest is only installed in the conda env.

Tests are configured in `pyproject.toml` (`[tool.pytest.ini_options]`): testpath is `tests/`, import mode is `importlib`.

## Code style

Uses `ruff` for linting and formatting. Line length is 120. Docstrings follow numpy convention. Run via:

```bash
conda run -n fishtank ruff check src/
conda run -n fishtank ruff format src/
```

## Package structure

```
src/fishtank/
  io/         — image reading (read_img, read_dax, read_xml, read_fov, read_mosaic, read_color_usage)
  correct/    — illumination correction
  filters/    — image filters (deconvolution, unsharp mask)
  decode/     — spot decoding (expectation maximization, logistic regression)
  seg/        — cell segmentation (cellpose wrapper, polygon utilities)
  pl/         — visualization (imshow)
  scripts/    — CLI entry points (fishtank CLI)
  utils.py    — mosaic creation, FOV format detection
  cli.py      — argparse CLI dispatcher
```

Top-level imports: `import fishtank as ft` → `ft.io`, `ft.correct`, `ft.decode`, `ft.seg`, `ft.filters`, `ft.pl`.

## Image format conventions

Raw images are MERFISH `.dax` files (flat binary, uint16) paired with `.xml` metadata files. The `.xml` contains: image shape, z-offsets, color order, frames-per-color, stage position, and transform flags (flip/transpose).

Output shape convention: `(Y, X)` for 2D, `(Z, Y, X)` for 3D single-channel, `(C, Z, Y, X)` for multi-channel 3D. `read_img` always squeezes length-1 axes.

### Frame ordering (non-sparse)

Frames in a `.dax` file are ordered color-inner, z-outer. For 5 colors and 5 z-planes:
- Frame 0–4: all colors at z[0]
- Frame 5–9: all colors at z[1]
- etc.

The reshape after loading uses Fortran (`order="F"`) to correctly map `(C, Z)`.

### Sparse / ragged acquisitions

When `frames_per_color` values differ across colors, the acquisition is ragged. `read_img` requires either `colors` or `z_slices` to be specified in this case, and uses `_reconstruct_sparse_frame_map` (XML-based) or a frame table CSV (explicit).

### Frame table CSV format

A frame table CSV maps each physical frame index to a `(color, z)` pair. Used for acquisitions with non-standard or ragged frame ordering. Pass via `read_img(..., frames="path/to/frame_table.csv")`.

Format: index column (frame index, 0-based), `color` column (wavelength as float), `z` column (z position as float). Rows with NaN `color` are skipped (blank frames). The table must have exactly one row per physical frame in the image — a mismatch raises `ValueError`. Example at `dev/blkf3_405f1_560f39_650f39_frame_table.csv`.

## Test data

Located in `tests/data/`:
- `merfish/` — two FOVs (`H0R1`, `H1R2`), 288×288 pixels, 5 colors, 5 z-planes
  - `H0R1/Conv_zscan_00.dax` + `.xml` — primary test image
  - `H0R1/frame_table.csv` — frame table matching the XML (used by frame table tests)
  - `color_usage.csv` — maps series names to color/bit assignments
- `corrections/` — illumination correction reference data
- `spots/` — detected spot coordinates
- `decoding/` — codebook and weights for decode tests
- `masks/` — segmentation masks (.npy)
- `polygons/`, `polygons_2d/` — GeoJSON polygon files

## dev/ directory

Contains development notebooks and real instrument data used for manual testing and prototyping:
- `hal-config-mf3-epi-common_000.dax/.xml` — real acquisition with non-standard frame ordering
- `blkf3_405f1_560f39_650f39_frame_table.csv` — frame table for the above (405 at z=0 only, 560/650 at z=1.0–20.0 in 0.5 steps)
- Various `.ipynb` notebooks for pipeline development
