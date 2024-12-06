import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

import fishtank as ft

from ._utils import parse_path


def get_parser():
    """Get parser for align_spots script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Spots input file path")
    parser.add_argument("-p", "--polygons", type=parse_path, default="polygons.json", help="Polygons input file path")
    parser.add_argument("-o", "--output", type=parse_path, default="assigned_spots.csv", help="Output file path")
    parser.add_argument(
        "--max_dist", type=float, default=0, help="Maximum distance from polygon edge for spot assignment"
    )
    parser.add_argument("--cell_column", type=str, default="cell", help="Column containing cell ID in polygons")
    parser.add_argument("--subset", type=parse_path, default=None, help="Set of polygons to assign spots to")
    parser.add_argument("--x_column", type=str, default="global_x", help="Column containing x-coordinate in spots")
    parser.add_argument("--y_column", type=str, default="global_y", help="Column containing y-coordinate in spots")
    parser.add_argument(
        "--z_column", type=str, default=None, help="Column containing z-slice in spots. None for 2D polygons"
    )
    parser.add_argument(
        "--polygons_z_column",
        type=str,
        default=None,
        help="Column containing z-slice in polygons. Defaults to z_column",
    )
    parser.add_argument("--cell_fill", type=int, default=0, help="Fill value for unassigned cells")
    parser.add_argument(
        "--alignment", type=parse_path, default=None, help="File used to align spots space to polygons space"
    )
    parser.add_argument("--map_z", type=bool, default=False, help="Map spot z values to polygon z values")
    parser.set_defaults(func=assign_spots)
    return parser


def assign_spots(
    input: str | Path,
    polygons: str | Path = "polygons.json",
    output: str | Path = "assigned_spots.csv",
    max_dist: float = 0,
    cell_column: str = "cell",
    subset: str | Path = None,
    x_column: str = "global_x",
    y_column: str = "global_y",
    z_column: str = None,
    polygons_z_column: str = None,
    cell_fill: int = 0,
    alignment: str | Path = None,
    map_z: bool = False,
):
    """Assign spots to the nearest polygon.

    fishtank assign-spots -i input -p polygons -o output

    Parameters
    ----------
    input
        Spots input file path.
    polygons
        Polygons input file path.
    output
        Output file path.
    max_dist
        Maximum distance from polygon edge for spot assignment.
    cell_column
        Column containing cell ID in polygons.
    subset
        Set of polygons to assign spots to.
    x_column
        Column containing x-coordinate in spots.
    y_column
        Column containing y-coordinate in spots.
    z_column
        Column containing z-slice in spots. None for 2D polygons.
    polygons_z_column
        Column containing z-slice in polygons. Defaults to z_column.
    cell_fill
        Fill value for unassigned cells.
    alignment
        File used to align spots space to polygons space.
    map_z
        Map spot z values to polygon z values.
    """
    # Setup
    logger = logging.getLogger("assign_spots")
    logger.info(f"fishtank version: {ft.__version__}")
    if polygons_z_column is None:
        polygons_z_column = z_column
    # Load data
    logger.info("Loading spots.")
    spots = pd.read_csv(input, keep_default_na=False)
    logger.info("Loading polygons.")
    polygons = gpd.read_file(polygons)
    polygons = polygons.set_crs(None, allow_override=True)
    # Subset cells
    if subset is not None:
        logger.info("Subsetting polygons.")
        subset = pd.read_csv(subset, sep="\t", header=None)[0].values
        polygons = polygons.query(f"{cell_column} in @subset").copy()
    # Map z values
    if map_z:
        polygon_z_slices = sorted(polygons[polygons_z_column].unique())
        spot_z_slices = sorted(spots[z_column].unique())
        z_map = {spot_z_slices[i]: polygon_z_slices[i] for i in range(len(spot_z_slices))}
        spots[z_column] = spots[z_column].map(z_map)
    # Align spots
    if alignment is not None:
        logger.info("Adjusting spot coordinates based on alignment.")
        alignment = gpd.read_file(alignment)
        alignment = alignment.set_crs(None, allow_override=True)
        spots = ft.correct.spot_alignment(spots, alignment)
    # Assign spots
    logger.info("Assigning spots to polygons.")
    spots = ft.seg.assign_spots(
        spots,
        polygons,
        max_dist=max_dist,
        cell=cell_column,
        x=x_column,
        y=y_column,
        z=z_column,
        polygons_z=polygons_z_column,
    )
    # Fill unassigned cells
    if cell_fill is not None:
        spots[cell_column] = spots[cell_column].fillna(cell_fill).astype(int)
    # Save
    logger.info("Saving spot assignments.")
    spots.to_csv(output, index=False)
    logger.info("Done")
