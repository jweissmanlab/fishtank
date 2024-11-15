import argparse

from fishtank.scripts import aggregate_polygons, align_experiments, cellpose, fovs


def main():
    """Main entry point for the command line interface."""
    parser = argparse.ArgumentParser(prog="fishtank", description="Fishtank command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subcommands by importing the parsers from each script
    subparsers.add_parser("cellpose", parents=[cellpose.get_parser()], help="Segment cells using Cellpose")
    subparsers.add_parser(
        "aggregate-polygons", parents=[aggregate_polygons.get_parser()], help="Aggregate polygons from multiple FOVs"
    )
    subparsers.add_parser(
        "align-experiments", parents=[align_experiments.get_parser()], help="Align experiments using optical flow"
    )
    subparsers.add_parser("fovs", parents=[fovs.get_parser()], help="List FOVs in a directory")

    # Parse arguments and dispatch the function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
