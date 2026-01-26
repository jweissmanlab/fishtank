import argparse
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as ski

import fishtank as ft

from ._utils import parse_bool, parse_dict, parse_list, parse_path, parse_int_or_str


class FOVLoggerAdapter(logging.LoggerAdapter):  # noqa: D101
    def process(self, msg, kwargs):  # noqa: D102
        fov_number = self.extra.get("fov", "Unknown FOV")
        return f"[FOV {fov_number}] {msg}", kwargs


def _get_filter(name, filter_args):
    """Get a filter function by name"""
    if name is None:
        return lambda x: x
    elif hasattr(ft.filters, name):
        return partial(getattr(ft.filters, name), **filter_args)
    elif hasattr(ski.filters, name):
        return partial(getattr(ski.filters, name) ** filter_args)
    else:
        raise ValueError(f"Filter {name} not found in fishtank.filters or skimage.filters")

def _get_reg_img(img, reg_bit, reg_color, channels, reg_z_slice=None, z_drift=False, clip_pct=None):
    """Return the registration image and initialize the drift vector."""
    if reg_color is not None:
        reg_channel = channels.query("color == @reg_color")
        if reg_channel.empty:
            raise ValueError(f"Registration color {reg_color} not found in color usage")
        reg_bit = reg_channel.bit.values[0]
    reg_img = img[channels.bit == reg_bit]
    if reg_img.size == 0:
        raise ValueError(f"Registration bit {reg_bit} not found in image channels")
    if reg_z_slice is not None:
        reg_img = reg_img[:, reg_z_slice].max(axis=0)
    elif z_drift:
        reg_img = reg_img.max(axis=0)
    else:
        reg_img = reg_img.max(axis=(0, 1))
    if clip_pct is not None:
        reg_img = np.minimum(reg_img / np.percentile(reg_img, clip_pct, axis=(-2, -1), keepdims=True), 1)
    return reg_img


def get_parser():
    """Get parser for detect_spots script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Image file directory")
    parser.add_argument("-f", "--fov", type=int, required=True, help="Field of view to process")
    parser.add_argument("--ref_series", type=parse_int_or_str, required=True, help="Reference series for drift correction")
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
        "--spot_threshold", type=int, default=1000, help="Minimum intensity threshold for spot detection"
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
    parser.add_argument("--z_drift", type=parse_bool, default=False, help="Do drift correction in z")
    parser.add_argument(
        "--reg_min_intensity", type=int, default=1000, help="Minimum intensity for registration channel to consider drift valid"
    )
    parser.add_argument(
        "--reg_color", type=int, default=None, help="Color name for registration channel (overrides reg_bit if provided)"
    )
    parser.add_argument(
        "--reg_z_slice", type=int, default=None, help="Z slice to use for registration (overrides z_drift if provided)"
    )
    parser.add_argument(
        "--reg_clip_pct", type=float, default=None, help="Percentile to clip registration image intensities"
    )
    parser.set_defaults(func=detect_spots)
    return parser


def detect_spots(
    input: str | Path,
    fov: int,
    ref_series: str | int,
    common_bits: list[str],
    reg_bit: str = "beads",
    output: str | Path = "spots",
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    color_usage: str = "{input}/color_usage.csv",
    filter: str = None,
    filter_args: dict = None,
    spot_min_sigma: int = 2,
    spot_max_sigma: int = 20,
    spot_threshold: int = 1000,
    spot_radius: int = 5,
    exclude_bits: list[str] = ["DAPI", "empty"],  # noqa: B006
    include_series: list[str] | None = None,
    z_drift: bool = False,
    reg_min_intensity: int = 1000,
    reg_color: int | None = None,
    reg_z_slice: int | None = None,
    reg_clip_pct: float | None = None,
    **kwargs,
):
    """Detect spots in an image and quantify their intensity.

    fishtank detect-spots -i input -f 1 --ref_series H0M1 --common_bits DAPI,empty -o spots

    Parameters
    ----------
    input
        Image file directory.
    fov
        Field of view to process.
    ref_series
        Reference series for drift correction.
    common_bits
        Common bits used for spot detection.
    reg_bit
        Bit used for series registration.
    output
        Output file path.
    file_pattern
        Naming pattern for image files.
    color_usage
        Path to color usage file.
    filter
        Filter to apply to the image.
    filter_args
        Additional filter arguments.
    spot_min_sigma
        Minimum sigma for spot detection.
    spot_max_sigma
        Maximum sigma for spot detection.
    spot_threshold
        Minimum intensity threshold for spot detection.
    spot_radius
        Spot radius for intensity quantification.
    exclude_bits
        Bits to exclude from intensity quantification.
    include_series
        Series to include in intensity quantification.
    z_drift
        Do drift correction in z.
    reg_min_intensity
        Minimum intensity for registration channel to consider drift valid.
    reg_color
        Color name for registration channel (overrides reg_bit if provided).
    reg_z_slice
        Z slice to use for registration (overrides z_drift if provided).
    reg_clip_pct
        Percentile to clip registration image intensities.
    """
    # Setup
    logger = logging.getLogger("detect_spots")
    logger = FOVLoggerAdapter(logger, {"fov": fov})
    logger.info(f"fishtank version: {ft.__version__}")
    if "{input}" in color_usage:
        color_usage = color_usage.format(input=input)
    channels = ft.io.read_color_usage(color_usage)
    if include_series is not None:
        channels = channels[channels["series"].isin(include_series)]
    filter_name = filter
    filter = _get_filter(filter, filter_args)
    # Get reference image
    logger.info(f"Loading reference series {ref_series}")
    ref_channels = channels.query("series == @ref_series")
    ref_img, attr = ft.io.read_fov(input, fov, channels=ref_channels, file_pattern=file_pattern)
    reg_img = _get_reg_img(
        ref_img, reg_bit, reg_color, ref_channels,
        reg_z_slice=reg_z_slice, z_drift=z_drift, clip_pct=reg_clip_pct
    )
    current_drift = np.zeros(3, dtype=int) if z_drift else np.zeros(2, dtype=int)
    # Get common image
    if len(ref_channels.query("bit in @common_bits")) == len(common_bits):
        common_img = ref_img[ref_channels.bit.isin(common_bits)].squeeze()
        if common_img.ndim > 3: # (C, Z, Y, X)
            common_img = common_img.max(axis=0) # (Z, Y, X)
    else:
        common_img = []
        logger.info("Getting common bit max projection")
        for series, series_channels in channels.query("bit in @common_bits").groupby("series", sort=False):
            logger.info(f"Loading series {series}")
            img, _ = ft.io.read_fov(input, fov, channels=series_channels, file_pattern=file_pattern)
            series_reg_img = _get_reg_img(
                img, reg_bit, reg_color, series_channels,
                reg_z_slice=reg_z_slice, z_drift=z_drift, clip_pct=reg_clip_pct
            )
            drift = ski.registration.phase_cross_correlation(reg_img, series_reg_img)[0].astype(int)
            logger.info(f"Series drift: {drift}")
            img = np.roll(img, shift=drift[:2], axis=(-2, -1))# apply drift correction
            common_img.append(img.max(axis=0))  # (Z, Y, X)
        common_img = np.stack(common_img).max(axis=0)  # (Z, Y, X)
    del ref_img
    logger.info(f"Applying {filter_name} filter")
    common_img = ski.util.apply_parallel(
        filter, common_img, chunks=(1, common_img.shape[1], common_img.shape[2]), dtype=common_img.dtype, channel_axis=0
    )
    # Detect spots
    logger.info(f"Detecting spots with threshold {spot_threshold}")
    common_img = np.pad(common_img, ((1, 1), (0, 0), (0, 0)), mode="constant")  # pad z axis
    positions = ski.feature.blob_log(
        common_img.astype(float),
        min_sigma=spot_min_sigma,
        max_sigma=spot_max_sigma,
        threshold=spot_threshold,
        num_sigma=4,
    )
    logger.info(f"Detected {positions.shape[0]} spots")
    spots = pd.DataFrame(positions[:, :3].astype(int), columns=["z", "y", "x"])
    max_z = common_img.shape[0] - 1
    spots["z"] = (spots["z"] - 1).clip(0, max_z - 2)
    del common_img
    # Intensity quantification
    logger.info("Quantifying spot intensities")
    filtered_channels = channels.query("bit not in @exclude_bits").copy()
    for series, series_channels in filtered_channels.groupby("series", sort=False):
        # Read series image
        logger.info(f"Loading series {series}")
        img, _ = ft.io.read_fov(input, fov, channels=series_channels, file_pattern=file_pattern)
        channel_max = img.max(axis=(1, 2, 3))
        filtered_channels.loc[filtered_channels.series == series, "max_intensity"] = channel_max
        # Get the drift
        series_reg_img = _get_reg_img(
            img, reg_bit, reg_color, series_channels,
            reg_z_slice=reg_z_slice, z_drift=z_drift, clip_pct=reg_clip_pct
        )
        drift = ski.registration.phase_cross_correlation(reg_img, series_reg_img)[0].astype(int)
        if z_drift:
            if np.abs(drift[0] - current_drift[0]) > 3:
                logger.warning(f"Large z drift detected: {drift[0]}. Using previous z drift: {current_drift[0]}")
                drift[0] = current_drift[0]
        if np.sum(np.abs(drift - current_drift)) > 100:
            logger.warning(f"Large drift detected: {drift}. Using previous drift: {current_drift}")
            drift = current_drift
        if channel_max[series_channels.bit == reg_bit] < reg_min_intensity:
            logger.warning(
                f"Low intensity for registration channel: {channel_max[series_channels.bit == reg_bit]}. Using previous drift: {current_drift}"
            )
            drift = current_drift
        current_drift = drift
        logger.info(f"Series drift: {drift}")
        filtered_channels.loc[filtered_channels.series == series, "x_drift"] = drift[-1]
        filtered_channels.loc[filtered_channels.series == series, "y_drift"] = drift[-2]
        if z_drift:
            filtered_channels.loc[filtered_channels.series == series, "z_drift"] = drift[0]
        # Remove the registration channel
        if reg_color is None:
            img = img[series_channels.bit != reg_bit]
            series_channels = series_channels[series_channels.bit != reg_bit]
        # Filter the image
        logger.info(f"Applying {filter_name} filter")
        for i in range(img.shape[0]):
            img[i] = ski.util.apply_parallel(
                filter, img[i], chunks=(1, img.shape[-2], img.shape[-1]), dtype=img.dtype, channel_axis=0
            )
        # Get spot intensities
        logger.info("Getting spot intensities")
        z = (spots.z if not z_drift else spots.z - drift[0]).clip(0, max_z)
        spots[series_channels.bit] = ft.decode.spot_intensities(
            img, spots.x - drift[-1], spots.y - drift[-2], z, spot_radius
        )
    # Clean up
    spots = spots.dropna().copy()
    spots["global_x"] = spots["x"] * attr["micron_per_pixel"] + attr["stage_position"][0]
    spots["global_y"] = spots["y"] * attr["micron_per_pixel"] + attr["stage_position"][1]
    spots["global_z"] = spots["z"].apply(lambda x: attr["z_offsets"][x])
    spots["spot"] = spots.reset_index().index.values
    spots["fov"] = fov
    col_order = ["fov", "spot", "x", "y", "z", "global_x", "global_y", "global_z"]
    col_order = col_order + [x for x in spots.columns if x not in col_order]
    spots = spots[col_order]
    filtered_channels["fov"] = fov
    # Save results
    logger.info(f"Saving results in {output}")
    output.mkdir(parents=True, exist_ok=True)
    spots.to_csv(output / f"spots_{fov}.csv", index=False)
    filtered_channels.to_csv(output / f"channels_{fov}.csv", index=False)
    logger.info("Done")
