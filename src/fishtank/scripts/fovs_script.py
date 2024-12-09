import argparse
from pathlib import Path

import fishtank as ft

from ._utils import parse_path


def get_parser():
    """Get parser for fovs script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=parse_path, required=True, help="Input file directory")
    parser.add_argument(
        "--file_pattern", type=str, default="{series}/Conv_zscan_{fov}.dax", help="Naming pattern for image files"
    )
    parser.set_defaults(func=fovs)
    return parser


def fovs(
    input: str | Path,
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    **kwargs,
):
    """List fields of view in an input directory.

    fishtank fovs -i input

    Parameters
    ----------
    input
        Input file directory.
    file_pattern
        Naming pattern for image files
    """
    fovs = ft.io.list_fovs(input, file_pattern=file_pattern)
    print(",".join(map(str, fovs)))
