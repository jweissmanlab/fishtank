import argparse
import json
from pathlib import Path

import pandas as pd
import tifffile

import fishtank as ft

from ._utils import parse_bool, parse_index, parse_list, parse_path


def get_parser():
    """Get parser for mosaic script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Input file directory")
    parser.add_argument("-o", "--output", type=parse_path, required=True, help="Output file path for mosaic TIFF")
    parser.add_argument("-z", "--z_slice", type=int, default=0, help="Z slice to use for mosaic (0-indexed)")
    parser.add_argument(
        "--file_pattern", type=str, default="{series}/Conv_zscan_{fov}.dax", help="Naming pattern for image files"
    )
    parser.add_argument("--downsample", type=int, default=4, help="Downsample factor for mosaic")
    parser.add_argument("--scale_factor", type=float, default=None, help="Scale factor in micron per pixel")
    parser.add_argument("--colors", type=parse_list, default=[405], help="Colors to include in mosaic (e.g 405,560)")
    parser.add_argument(
        "--fovs", type=parse_index, default=None, help="FOVs to include in mosaic (e.g. 1 or 1,2,3 or 1:20)"
    )
    parser.add_argument(
        "--positions", type=parse_path, default=None, help="TSV file with FOV positions (no header, columns: x, y)"
    )
    parser.add_argument("--flip_horizontal", type=parse_bool, default="False", help="Flip FOV images horizontally")
    parser.add_argument("--flip_vertical", type=parse_bool, default="False", help="Flip FOV images vertically")
    parser.set_defaults(func=mosaic)
    return parser


def mosaic(
    input: str | Path,
    output: str | Path,
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    downsample: int = 4,
    z_slice: int = 0,
    scale_factor: float = None,
    colors: list[int] | None = None,
    fovs: list[int] | None = None,
    positions: str | Path | None = None,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    **kwargs,
):
    """Create a mosaic from a set of image files.

    fishtank mosaic -i input -o output --colors 405

    Parameters
    ----------
    input
        Input file directory.
    output
        Output file path for mosaic TIFF.
    file_pattern
        Naming pattern for image files.
    downsample
        Downsample factor for mosaic.
    z_slice
        Z slice to use for mosaic (0-indexed).
    scale_factor
        Scale factor in micron per pixel.
    colors
        Colors to include in mosaic.
    fovs
        FOVs to include in mosaic.
    positions
        Path to a two-column (no header) TSV file with x and y positions in microns for each FOV.
        Row index corresponds to FOV number. When provided, overrides stage positions from image metadata.
    flip_horizontal
        Flip each FOV image horizontally before stitching.
    flip_vertical
        Flip each FOV image vertically before stitching.
    """
    if fovs is None:
        fovs = ft.io.list_fovs(input, file_pattern=file_pattern)
    positions_dict = None
    if positions is not None:
        pos_df = pd.read_csv(positions, sep=None, engine="python", header=None, names=["x", "y"])
        positions_dict = {i: (row["x"], row["y"]) for i, row in pos_df.iterrows()}
    result = ft.io.read_mosaic(
        input,
        fovs=fovs,
        file_pattern="{series}",
        series=file_pattern,
        z_slices=z_slice,
        downsample=downsample,
        microns_per_pixel=scale_factor,
        colors=colors,
        positions=positions_dict,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
    )
    description = json.dumps({"custom_metadata": {"bounds": result[1].tolist()}})
    tifffile.imwrite(output, result[0], description=description)
