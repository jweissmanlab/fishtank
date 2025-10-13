import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import fishtank as ft

from ._utils import parse_bool, parse_index, parse_path


def get_parser():
    """Get parser for decode_spots script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Inpute file path or directory")
    parser.add_argument("-s", "--strategy", type=parse_path, default=None, help="Decoding strategy file")
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
    parser.add_argument("--normalize_colors", type=parse_bool, default="True", help="Normalize intensities by color")
    parser.add_argument(
        "--filter_output", type=parse_bool, default="True", help="Exclude spots with distance > max_dist from output"
    )
    parser.add_argument(
        "--save_intensities", type=parse_bool, default="False", help="Include spot intensities in output"
    )
    parser.set_defaults(func=decode_spots)
    return parser


def _save_decoding_plot(path, suffix):
    path = path.with_stem(path.stem + "_" + suffix).with_suffix(".png")
    plt.savefig(path, dpi=600)
    plt.close()


def decode_spots(
    input: str | Path,
    strategy: str | Path | None = None,
    output: str | Path = "decoded_spots",
    file_pattern: str = "spots_{fov}.csv",
    fovs: list[int] | slice | None = None,
    color_usage: str = "{input}/color_usage.csv",
    max_dist: float = 1.7,
    min_snr: float = 4,
    normalize_colors: bool = True,
    filter_output: bool = True,
    save_intensities: bool = False,
    **kwargs,
):
    """Decode spots using specified strategy.

    fishtank decode-spots -i input -s strategy -o output

    Parameters
    ----------
    input
        Input file path or directory.
    strategy
        Decoding strategy file.
    output
        Output file path.
    file_pattern
        Naming pattern for files if input is a directory.
    fovs
        Set of fields of view to process.
    color_usage
        Path to color usage file.
    max_dist
        Maximum distance between intensity and codebook vectors.
    min_snr
        Minimum signal-to-noise for bits in the codebook.
    normalize_colors
        Normalize intensities by color.
    filter_output
        Exclude spots with distance > max_dist from output.
    save_intensities
        Include spot intensities in output.
    """
    # Setup
    logger = logging.getLogger("decode_spots")
    logger.info(f"fishtank version: {ft.__version__}")
    # Load strategy
    logger.info(f"Loading decoding strategy from {strategy}")
    em_codebooks = {}
    lr_weights = {}
    if strategy is not None:
        strategies = pd.read_csv(strategy)
        decoding_bits = set()
        for _, strategy in strategies.iterrows():
            strategy = strategy.to_dict()
            file = pd.read_csv(strategy["file"], index_col=0, keep_default_na=False)
            decoding_bits.update(set(file.columns.values))
            if strategy["method"] == "expectation_maximization":
                whitelist = strategy.get("whitelist", None)
                if whitelist is not None:
                    logger.info(f"Using whitelist for {strategy['name']}")
                    whitelist = pd.read_csv(whitelist, sep="\t", header=None)[0].values
                    file = file.loc[whitelist, :].copy()
                em_codebooks[strategy["name"]] = file
            elif strategy["method"] == "logistic_regression":
                lr_weights[strategy["name"]] = file
    # Load spots
    logger.info(f"Loading spots in {input}")
    if input.is_dir():
        if fovs is None:
            fovs = ft.io.list_fovs(input, file_pattern)
        spots = []
        for fov in fovs:
            fov_spots = pd.read_csv(input / file_pattern.format(fov=fov))
            if len(fov_spots) > 0:
                spots.append(fov_spots)
        spots = pd.concat(spots)
    else:
        spots = pd.read_csv(input)
    # Load color usage
    if "{input}" in color_usage:
        color_usage = color_usage.format(input=input)
    channels = ft.io.read_color_usage(color_usage)
    if strategy is not None:
        spots.drop(columns=set(channels.bit) - decoding_bits, inplace=True, errors="ignore")
        channels = channels.query("bit in @decoding_bits").copy()
    # Color normalization
    if normalize_colors:
        logger.info("Normalizing spot intensities by color")
        spots[channels.bit] = ft.correct.color_normalization(spots[channels.bit], channels.color)
    # EM decoding
    for name, codebook in em_codebooks.items():
        logger.info(f"Using EM to decode {name}")
        decoded, bit_performance = ft.decode.expectation_maximization(
            spots, codebook, max_dist=max_dist, min_snr=min_snr, plot=True
        )
        _save_decoding_plot(output, name)
        logger.info(f"Bit performance: \n{bit_performance}")
        spots[[name, f"{name}_dist", f"{name}_intensity", f"{name}_snr"]] = decoded
        if filter_output:
            spots = spots.query(f"{name}_dist <= {max_dist}").copy()
    # LR decoding
    for name, weights in lr_weights.items():
        logger.info(f"Using logistic regression to decode {name}")
        decoded, bit_performance = ft.decode.logistic_regression(spots, weights, plot=True)
        _save_decoding_plot(output, name)
        logger.info(f"Bit performance: \n{bit_performance}")
        spots[[name, f"{name}_prob", f"{name}_intensity", f"{name}_snr"]] = decoded
    # Save decoded spots
    logger.info(f"Saving decoded spots to {output}")
    if not save_intensities:
        spots.drop(columns=channels.bit, inplace=True, errors="ignore")
    spots.round(3).to_csv(output, index=False)
    logger.info("Done")
