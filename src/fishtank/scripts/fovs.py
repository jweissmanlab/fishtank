import argparse

import fishtank as ft
from fishtank.utils import parse_path


def get_parser():
    """Get parser for cellpose script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=parse_path, required=True, help="Input file directory")
    parser.add_argument(
        "--file_pattern", type=str, default="{series}/Conv_zscan_{fov}.dax", help="Naming pattern for image files"
    )
    parser.set_defaults(func=main)
    return parser


def main(args):
    """Segment cells using Cellpose"""
    fovs = ft.io.list_fovs(args.input, file_pattern=args.file_pattern)
    print(",".join(map(str, fovs)))
