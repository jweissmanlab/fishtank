import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

import fishtank as ft

from ._utils import parse_index, parse_path


def get_parser():
    """Get parser for decode_spots script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Inpute file path or directory")
    parser.add_argument("-s", "--strategy", type=parse_path, required=True, help="Decoding strategy file")
    parser.add_argument("-o", "--output", type=parse_path, default="decoded_spots", help="Output file path")
    parser.add_argument(
        "--file_pattern", type=str, default="spots_{fov}.csv", help="If input is a directory, naming pattern for files"
    )
    parser.add_argument(
        "--fovs", type=parse_index, default=None, help="Set of fields of view to process (e.g., 1 or 1,2,3 or 10:20)"
    )
    parser.add_argument("--color_usage", type=str, default="{input}/color_usage.csv", help="Path to color usage file")
    parser.add_argument(
        "--max_dist", type=float, default=1.7, help="Maximum distance between intensity and codebook vectors"
    )
    parser.add_argument("--min_snr", type=float, default=4, help="Minimum signal-to-noise for bits in the codebook")
    parser.add_argument("--normalize_colors", type=bool, default=True, help="Normalize intensities by color")
    parser.add_argument(
        "--filter_output", type=bool, default=True, help="Exclude spots with distance > max_dist from output"
    )
    parser.set_defaults(func=main)
    return parser


def _save_decoding_plot(path, suffix):
    path = path.with_stem(path.stem + "_" + suffix).with_suffix(".png")
    plt.savefig(path, dpi=600)
    plt.close()


def main(args):
    """Decode spots using specified strategy"""
    # Setup
    logger = logging.getLogger("decode_spots")
    logger.info(f"fishtank version: {ft.__version__}")
    # Load strategy
    strategies = pd.read_csv(args.strategy)
    em_codebooks = {}
    lr_weights = {}
    decoding_bits = set()
    for _, strategy in strategies.iterrows():
        strategy = strategy.to_dict()
        file = pd.read_csv(strategy["file"], index_col=0, keep_default_na=False)
        decoding_bits.update(set(file.columns.values))
        if strategy["method"] == "expectation_maximization":
            em_codebooks[strategy["name"]] = file
        elif strategy["method"] == "logistic_regression":
            lr_weights[strategy["name"]] = file
    # Load spots
    logger.info(f"Loading spots in {args.input}")
    if args.input.is_dir():
        if args.fovs is None:
            args.fovs = ft.io.list_fovs(args.input, args.file_pattern)
        spots = []
        for fov in args.fovs:
            fov_spots = pd.read_csv(args.input / args.file_pattern.format(fov=fov))
            if len(fov_spots) > 0:
                spots.append(fov_spots)
        spots = pd.concat(spots)
    else:
        spots = pd.read_csv(args.input)
    # Load color usage
    if "{input}" in args.color_usage:
        args.color_usage = args.color_usage.format(input=args.input)
    channels = ft.io.read_color_usage(args.color_usage)
    spots.drop(columns=set(channels.bit) - decoding_bits, inplace=True, errors="ignore")
    channels = channels.query("bit in @decoding_bits").copy()
    # Color normalization
    if args.normalize_colors:
        logger.info("Normalizing spot intensities by color")
        spots[channels.bit] = ft.correct.color_normalization(spots[channels.bit], channels.color)
    # EM decoding
    for name, codebook in em_codebooks.items():
        logger.info(f"Using EM to decode {name}")
        decoded, bit_performance = ft.decode.expectation_maximization(
            spots, codebook, max_dist=args.max_dist, min_snr=args.min_snr, plot=True
        )
        _save_decoding_plot(args.output, name)
        logger.info(f"Bit performance: \n{bit_performance}")
        spots[[name, f"{name}_dist", f"{name}_intensity", f"{name}_snr"]] = decoded
        if args.filter_output:
            spots = spots.query(f"{name}_dist <= {args.max_dist}").copy()
    # LR decoding
    for name, weights in lr_weights.items():
        logger.info(f"Using logistic regression to decode {name}")
        decoded, bit_performance = ft.decode.logistic_regression(spots, weights, plot=True)
        _save_decoding_plot(args.output, name)
        logger.info(f"Bit performance: \n{bit_performance}")
        spots[[name, f"{name}_prob", f"{name}_intensity", f"{name}_snr"]] = decoded
    # Save decoded spots
    logger.info(f"Saving decoded spots to {args.output}")
    spots.drop(columns=channels.bit).round(3).to_csv(args.output, index=False)
    logger.info("Done")
