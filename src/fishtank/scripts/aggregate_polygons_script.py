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

from ._utils import parse_index, parse_path


def _load_fov_polygons(
    fov,
    path,
    file_pattern="polygons_{fov}.json",
    x_offset_column="x_offset",
    y_offset_column="y_offset",
    scale_factor=0.107,
    tolerance=0.5,
    flip_horizontal=False,
    flip_vertical=False,
    img_size = (2304, 2304)
):
    """Load and rescale polygons for a single FOV."""
    path = Path(path)
    polygons = gpd.read_file(path / file_pattern.format(fov=fov)).reset_index(drop=True).assign(fov=fov)
    polygons.crs = None
    if len(polygons) == 0:
        return None
    if flip_horizontal or flip_vertical:
        polygons.geometry = polygons.geometry.scale(
            xfact=-1 if flip_horizontal else 1,
            yfact=-1 if flip_vertical else 1,
            origin=(0, 0)
        ).translate(xoff=img_size[0] if flip_horizontal else 0, yoff=img_size[1] if flip_vertical else 0)
    if scale_factor != 1:
        polygons.geometry = polygons.geometry.affine_transform([scale_factor, 0, 0, scale_factor, 0, 0])

    if x_offset_column is not None:
        polygons.geometry = polygons.geometry.translate(
            xoff=polygons[x_offset_column][0], yoff=polygons[y_offset_column][0]
        )
    polygons.geometry = polygons.geometry.simplify(tolerance).make_valid()
    return polygons


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
    parser.add_argument("--save_union", type=bool, default=False, help="Save polygons flattened (unioned) to 2D")
    parser.add_argument(
        "--flip_horizontal", type=bool, default=False, help="Flip polygons horizontally"
    )
    parser.add_argument(
        "--flip_vertical", type=bool, default=False, help="Flip polygons vertically"
    )
    parser.set_defaults(func=aggregate_polygons)
    return parser


def aggregate_polygons(
    input: str | Path,
    output: str | Path = "polygons.json",
    min_size: float = 100,
    min_ioa: float = 0.2,
    fovs: list[int] | slice | None = None,
    file_pattern: str = "polygons_{fov}.json",
    cell_column: str = "cell",
    z_column: str | None = None,
    x_offset_column: str = "x_offset",
    y_offset_column: str = "y_offset",
    scale_factor: float = 0.107,
    tolerance: float = 0.5,
    save_union: bool = False,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    **kwargs,
):
    """Aggregate polygons from multiple FOVs.

    fishtank aggregate-polygons -i input

    Parameters
    ----------
    input
        Input file directory.
    output
        Output file path.
    min_size
        Minimum area/volume for a cell to be kept.
    min_ioa
        Minimum intersection over area for merging cells.
    fovs
        Fields of view to aggregate (e.g., 1 or 1,2,3 or 1:20:5).
    file_pattern
        Naming pattern for polygon files.
    cell_column
        Column containing cell ID.
    z_column
        Column containing z-slice. None for 2D polygons.
    x_offset_column
        Column containing x-offset.
    y_offset_column
        Column containing y-offset.
    scale_factor
        Factor for converting pixels to microns.
    tolerance
        Tolerance from polygon simplification (microns).
    save_union
        Save polygons flattened (unioned) to 2D.
    """
    # Setup
    logger = logging.getLogger("aggregate_polygons")
    logger.info(f"fishtank version: {ft.__version__}")
    if fovs is None:
        fovs = ft.io.list_fovs(input, file_pattern=file_pattern)
    else:
        fovs = fovs
    # Load Polygons
    logger.info("Loading polygons in parallel.")
    parallel_func = partial(
        _load_fov_polygons,
        path=input,
        file_pattern=file_pattern,
        x_offset_column=x_offset_column,
        y_offset_column=y_offset_column,
        scale_factor=scale_factor,
        tolerance=tolerance,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
    )
    with mp.Pool(mp.cpu_count()) as pool:
        polygons = list(tqdm(pool.imap_unordered(parallel_func, fovs), total=len(fovs)))
    polygons = pd.concat(polygons)
    polygons["cell"] = (polygons["fov"] * 1e5 + polygons[cell_column]).rank(method="dense").astype(int) + 1
    logger.info(f"Loaded {len(polygons[cell_column].unique())} polygons.")
    # Fix overlapping polygons
    logger.info("Fixing overlapping polygons.")
    polygons = ft.seg.fix_overlaps(
        polygons, min_ioa=min_ioa, cell=cell_column, z=z_column, fov="fov", tolerance=tolerance
    )
    logger.info(f"{polygons[cell_column].nunique()} polygons after fixing overlaps.")
    # Calculate polygon statistics
    logger.info("Calculating polygon statistics.")
    metadata = ft.seg.polygon_properties(polygons, cell=cell_column, z=z_column)
    if min_size > 0:
        if z_column is not None:
            metadata = metadata[metadata["volume"] > min_size].copy()
        else:
            metadata = metadata[metadata["area"] > min_size].copy()
    logger.info(f"{len(metadata)} polygons after removing polygons smaller than {min_size}.")
    # Save polygons
    logger.info(f"Saving polygons to {output}")
    polygons = polygons.drop(columns=["x_offset", "y_offset"]).query("cell in @metadata.cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore CRS warning
        polygons.to_file(output, driver="GeoJSON")
    metadata.drop(columns=["x_offset", "y_offset", "global_z", "z"], errors="ignore", inplace=True)
    metadata.to_csv(str(output).replace(".json", "_metadata.csv"), index=False)
    # Save unioned polygons
    if save_union:
        union_path = str(output).replace(".json", "_union.json")
        logger.info(f"Saving unioned polygons to {union_path}")
        polygons_union = polygons.dissolve(by="cell").reset_index()[["cell", "fov", "geometry"]]
        polygons_union.geometry = polygons_union.geometry.buffer(-1.5).buffer(2)
        polygons_union.geometry = polygons_union.simplify(0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore CRS warning
            polygons_union.to_file(union_path, driver="GeoJSON")
    logger.info("Done")
