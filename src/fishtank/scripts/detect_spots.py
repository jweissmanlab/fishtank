import argparse
import logging

import numpy as np
import pandas as pd
import skimage as ski

import fishtank as ft

from ._utils import parse_dict, parse_list, parse_path


class FOVLoggerAdapter(logging.LoggerAdapter):  # noqa: D101
    def process(self, msg, kwargs):  # noqa: D102
        fov_number = self.extra.get("fov", "Unknown FOV")
        return f"[FOV {fov_number}] {msg}", kwargs


def get_parser():
    """Get parser for detect_spots script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Image file directory")
    parser.add_argument("-f", "--fov", type=int, required=True, help="Field of view to process")
    parser.add_argument("--ref_series", type=str, required=True, help="Reference series for drift correction")
    parser.add_argument("--common_bits", type=parse_list, required=True, help="Common bits used for spot detection")
    parser.add_argument("--reg_bit", type=str, default="beads", help="Bit used for series registration")
    parser.add_argument("-o", "--output", type=parse_path, default="spots", help="Output file path")
    parser.add_argument(
        "--file_pattern", type=str, default="{series}/Conv_zscan_{fov}.dax", help="Naming pattern for image files"
    )
    parser.add_argument("--color_usage", type=str, default="{input}/color_usage.csv", help="Path to color usage file")
    parser.add_argument("--filter", type=str, default=None, help="Filter to apply to the image")
    parser.add_argument(
        "--filter_args", type=parse_dict, default={}, help="Additional filter arguments (e.g., key1=val1,key2=val2)"
    )
    parser.add_argument("--spot_min_sigma", type=int, default=2, help="Minimum sigma for spot detection")
    parser.add_argument("--spot_max_sigma", type=int, default=20, help="Maximum sigma for spot detection")
    parser.add_argument(
        "--spot_threshold_quantile", type=float, default=0.95, help="Quantile for spot intensity threshold"
    )
    parser.add_argument("--spot_radius", type=int, default=5, help="Spot radius for intensity quantification")
    parser.add_argument(
        "--exclude_bits",
        type=parse_list,
        default=["DAPI", "empty"],
        help="Bits to exclude from intensity quantification",
    )
    parser.add_argument(
        "--include_series",
        type=parse_list,
        default=None,
        help="Series to include in intensity quantification. None for all series",
    )
    parser.set_defaults(func=main)
    return parser


def _get_filter(name, filter_args):
    """Get a filter function by name"""
    if name is None:
        return lambda x: x
    elif hasattr(ft.filters, name):
        return lambda x: getattr(ft.filters, name)(x, channel_axis=0, **filter_args)
    elif hasattr(ski.filters, name):
        return lambda x: getattr(ski.filters, name)(x, channel_axis=0, **filter_args)
    else:
        raise ValueError(f"Filter {name} not found in fishtank.filters or skimage.filters")


def main(args):
    """Detect spots in an image and quantify their intensity"""
    # Setup
    logger = logging.getLogger("detect_spots")
    logger = FOVLoggerAdapter(logger, {"fov": args.fov})
    logger.info(f"fishtank version: {ft.__version__}")
    if "{input}" in args.color_usage:
        args.color_usage = args.color_usage.format(input=args.input)
    channels = ft.io.read_color_usage(args.color_usage)
    if args.include_series is not None:
        channels = channels[channels["series"].isin(args.include_series)]
    filter = _get_filter(args.filter, args.filter_args)
    # Load reference
    logger.info(f"Loading reference series {args.ref_series}")
    ref_channels = channels.query("series == @args.ref_series")
    ref_img, attr = ft.io.read_fov(args.input, args.fov, channels=ref_channels, file_pattern=args.file_pattern)
    reg_img = ref_img[ref_channels.bit == args.reg_bit].max(axis=(0, 1))
    # Get common image
    common_img = ref_img[ref_channels.bit.isin(args.common_bits)].squeeze()
    if common_img.ndim > 3:
        common_img = common_img.max(axis=0)
    logger.info(f"Applying {args.filter} filter")
    common_img = filter(common_img)
    threshold = np.quantile(common_img, args.spot_threshold_quantile)
    # Detect spots
    logger.info(f"Detecting spots with threshold {threshold}")
    common_img = np.pad(common_img, ((1, 1), (0, 0), (0, 0)), mode="constant")  # pad z axis
    positions = ski.feature.blob_log(
        common_img.astype(float), min_sigma=args.spot_min_sigma, max_sigma=args.spot_max_sigma, threshold=threshold
    )
    logger.info(f"Detected {positions.shape[0]} spots")
    spots = pd.DataFrame(positions[:, :3].astype(int), columns=["z", "y", "x"])
    spots["z"] = (spots["z"] - 1).clip(0, common_img.shape[0] - 3)
    del common_img
    # Intensity quantification
    logger.info("Quantifying spot intensities")
    filtered_channels = channels.query("bit not in @args.exclude_bits").copy()
    for series, series_channels in filtered_channels.groupby("series", sort=False):
        # Read series image
        logger.info(f"Processing series {series}")
        img, _ = ft.io.read_fov(args.input, args.fov, channels=series_channels, file_pattern=args.file_pattern)
        filtered_channels.loc[filtered_channels.series == series, "max_intensity"] = img.max(axis=(1, 2, 3))
        # Get the drift
        drift = ski.registration.phase_cross_correlation(
            reg_img, img[series_channels.bit == args.reg_bit].max(axis=(0, 1))
        )[0].astype(int)
        logger.info(f"Series drift: {drift}")
        filtered_channels.loc[filtered_channels.series == series, "x_drift"] = drift[1]
        filtered_channels.loc[filtered_channels.series == series, "y_drift"] = drift[0]
        # Remove the registration channel
        img = img[series_channels.bit != args.reg_bit]
        series_channels = series_channels[series_channels.bit != args.reg_bit]
        # Filter the image
        logger.info(f"Applying {args.filter} filter")
        for i in range(img.shape[0]):
            img[i] = filter(img[i])
        # Get spot intensities
        logger.info("Quantifying spot intensities")
        spots[series_channels.bit] = ft.decode.spot_intensities(
            img, spots.x - drift[1], spots.y - drift[0], spots.z, args.spot_radius
        )
    # Get global coordinates
    spots["global_x"] = spots["x"] * attr["micron_per_pixel"] + attr["stage_position"][0]
    spots["global_y"] = spots["y"] * attr["micron_per_pixel"] + attr["stage_position"][1]
    spots["global_z"] = spots["z"].apply(lambda x: attr["z_offsets"][x])
    spots["spot"] = spots.index.values
    # Save results
    logger.info(f"Saving results in {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    col_order = ["spot", "x", "y", "z", "global_x", "global_y", "global_z"]
    col_order = col_order + [x for x in spots.columns if x not in col_order]
    spots = spots[col_order]
    spots.to_csv(args.output / f"spots_{args.fov}.csv", index=False)
    filtered_channels.to_csv(args.output / f"channels_{args.fov}.csv", index=False)
