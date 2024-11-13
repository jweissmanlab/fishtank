import argparse
import logging
import multiprocessing as mp
import warnings
from functools import partial
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

import fishtank as ft
from fishtank.utils import parse_index, parse_path


def get_parser():
    """Get parser for cellpose script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Input file directory")
    parser.add_argument("-o", "--output", type=parse_path, default="polygons.json", help="Output file path")
    parser.add_argument("--min_size", type=float, default=100, help="Minimum area/volume for a cell to be kept")
    parser.add_argument("--min_ioa", type=float, default=0.2, help="Minimum intersection over area for merging cells")
    parser.add_argument(
        "--fovs", type=parse_index, default=None, help="Fields of view to aggregate (e.g., 1 or 1,2,3 or 1:20:5)"
    )
    parser.add_argument(
        "--file_pattern", type=str, default="polygons_{fov}.json", help="Naming pattern for polygon files"
    )
    parser.add_argument("--cell_column", type=str, default="cell", help="Column containing cell ID")
    parser.add_argument("--z_column", type=str, default=None, help="Column containing z-slice. None for 2D polygons")
    parser.add_argument("--x_offset_column", type=str, default="x_offset", help="Column containing x-offset")
    parser.add_argument("--y_offset_column", type=str, default="y_offset", help="Column containing y-offset")
    parser.add_argument("--scale_factor", type=float, default=0.107, help="Factor for converting pixels to microns")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Tolerance from polygon simplification (microns)")
    parser.set_defaults(func=main)
    return parser


def _load_fov_polygons(
    fov,
    path,
    file_pattern="{fov}_cells.json",
    x_offset_column="x_offset",
    y_offset_column="y_offset",
    scale_factor=0.107,
    tolerance=0.5,
):
    """Load and rescale polygons for a single FOV."""
    path = Path(path)
    polygons = gpd.read_file(path / file_pattern.format(fov=fov)).reset_index(drop=True).assign(fov=fov)
    polygons.crs = None
    if len(polygons) == 0:
        return []
    if scale_factor != 1:
        polygons.geometry = polygons.geometry.affine_transform([scale_factor, 0, 0, scale_factor, 0, 0])
    if x_offset_column is not None:
        polygons.geometry = polygons.geometry.translate(
            xoff=polygons[x_offset_column][0], yoff=polygons[y_offset_column][0]
        )
    polygons.geometry = polygons.geometry.simplify(tolerance).make_valid()
    return polygons


def main(args):
    """Aggregate polygons from multiple FOVs"""
    # Setup
    logger = logging.getLogger("aggregate_polygons")
    logger.info(f"fishtank version: {ft.__version__}")
    if args.fovs is None:
        fovs = ft.io.list_fovs(args.input, file_pattern=args.file_pattern)
    else:
        fovs = args.fovs
    # Load Polygons
    logger.info("Loading polygons in parallel.")
    parallel_func = partial(
        _load_fov_polygons,
        path=args.input,
        file_pattern=args.file_pattern,
        x_offset_column=args.x_offset_column,
        y_offset_column=args.y_offset_column,
        scale_factor=args.scale_factor,
        tolerance=args.tolerance,
    )
    with mp.Pool(mp.cpu_count()) as pool:
        polygons = list(tqdm(pool.imap_unordered(parallel_func, fovs), total=len(fovs)))
    polygons = pd.concat(polygons)
    polygons["cell"] = (polygons["fov"] * 1e5 + polygons[args.cell_column]).rank(method="dense").astype(int)
    logger.info(f"Loaded {len(polygons[args.cell_column].unique())} polygons.")
    # Fix overlapping polygons
    logger.info("Fixing overlapping polygons.")
    polygons = ft.seg.fix_overlaps(
        polygons, min_ioa=args.min_ioa, cell=args.cell_column, z=args.z_column, fov="fov", tolerance=args.tolerance
    )
    logger.info(f"{polygons[args.cell_column].nunique()} polygons after fixing overlaps.")
    # Calculate polygon statistics
    logger.info("Calculating polygon statistics.")
    metadata = ft.seg.polygon_properties(polygons, cell=args.cell_column, z=args.z_column)
    if args.min_size > 0:
        if args.z_column is not None:
            metadata = metadata[metadata["volume"] > args.min_size].copy()
        else:
            metadata = metadata[metadata["area"] > args.min_size].copy()
    logger.info(f"{len(metadata)} polygons after removing polygons smaller than {args.min_size}.")
    # Save polygons
    logger.info("Saving polygons.")
    polygons = polygons.drop(columns=["x_offset", "y_offset"]).query("cell in @metadata.cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore CRS warning
        polygons.to_file(args.output, driver="GeoJSON")
    metadata.to_csv(str(args.output).replace(".json", "_metadata.csv"), index=False)
