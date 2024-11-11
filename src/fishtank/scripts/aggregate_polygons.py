import argparse


def parse_dict(arg):
    if arg is None:
        return None
    return dict(item.split("=") for item in arg.split(","))


def parse_list(arg):
    if arg is None:
        return None
    return arg.split(",")


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=str, required=True, help="Input image file path")
    parser.add_argument("--output", type=str, default="./cellpose_polygons", help="Output directory")
    return parser


def main(args):
    print("Aggregating polygons")
