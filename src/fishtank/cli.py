import argparse

import fishtank.scripts as scripts


def main():
    """Main entry point for the command line interface."""
    parser = argparse.ArgumentParser(prog="fishtank", description="Fishtank command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subcommands by importing the parsers from each script
    subparsers.add_parser(
        "cellpose", parents=[scripts.cellpose_script.get_parser()], help="Segment cells using Cellpose"
    )
    subparsers.add_parser(
        "aggregate-polygons",
        parents=[scripts.aggregate_polygons_script.get_parser()],
        help="Aggregate polygons from multiple FOVs",
    )
    subparsers.add_parser(
        "align-experiments",
        parents=[scripts.align_experiments_script.get_parser()],
        help="Align experiments using optical flow",
    )
    subparsers.add_parser(
        "detect-spots",
        parents=[scripts.detect_spots_script.get_parser()],
        help="Detect spots in an image and quantify their intensity",
    )
    subparsers.add_parser(
        "decode-spots",
        parents=[scripts.decode_spots_script.get_parser()],
        help="Decode spots using a specified strategy",
    )
    subparsers.add_parser(
        "assign-proseg",
        parents=[scripts.assign_proseg_script.get_parser()],
        help="Assign additional transcripts to polygons using ProSeg",
    )
    subparsers.add_parser(
        "assign-spots", parents=[scripts.assign_spots_script.get_parser()], help="Assign spots to the nearest polygon"
    )
    subparsers.add_parser("fovs", parents=[scripts.fovs_script.get_parser()], help="List FOVs in a directory")

    # Parse arguments and dispatch the function
    args = parser.parse_args()
    func = args.func
    kwargs = vars(args)
    kwargs.pop("func")
    func(**kwargs)


if __name__ == "__main__":
    main()
