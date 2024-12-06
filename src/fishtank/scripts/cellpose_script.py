import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import skimage as ski

import fishtank as ft

from ._utils import parse_dict, parse_index, parse_list, parse_path


class FOVLoggerAdapter(logging.LoggerAdapter):  # noqa: D101
    def process(self, msg, kwargs):  # noqa: D102
        fov_number = self.extra.get("fov", "Unknown FOV")
        return f"[FOV {fov_number}] {msg}", kwargs


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def get_parser():
    """Get parser for cellpose script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, required=True, help="Input file directory")
    parser.add_argument("-f", "--fov", type=int, required=True, help="Field of view to process")
    parser.add_argument("-o", "--output", type=parse_path, default="./cellpose_polygons", help="Output directory")
    parser.add_argument(
        "--channels", type=parse_list, required=True, help="Channel for segmentation (e.g., DAPI or PolyT,DAPI)"
    )
    parser.add_argument(
        "--file_pattern", type=str, default="{series}/Conv_zscan_{fov}.dax", help="Naming pattern for image files"
    )
    parser.add_argument("--corrections", type=parse_path, default=None, help="Path to image corrections directory")
    parser.add_argument("--color_usage", type=str, default="{input}/color_usage.csv", help="Path to color usage file")
    parser.add_argument(
        "--z_slices",
        type=parse_index,
        default=None,
        help="Z-slices to use for segmentation (e.g., 1 or 1,2,3 or 1:20:5)",
    )
    parser.add_argument("--model", type=str, default="nuclei", help="Cellpose model (e.g., cyto or nuclei)")
    parser.add_argument("--diameter", type=int, default=18, help="Cellpose diameter")
    parser.add_argument("--cellprob_threshold", type=float, default=-4, help="Cell probability threshold")
    parser.add_argument("--downsample", type=int, default=None, help="Downsampling factor")
    parser.add_argument("--do_3D", type=bool, default=False, help="Use 3D segmentation")
    parser.add_argument(
        "--model_args", type=parse_dict, default={}, help="Additional model arguments (e.g., key1=val1,key2=val2)"
    )
    parser.add_argument("--clear_border", type=bool, default=False, help="Remove cells touching the image border")
    parser.add_argument("--min_size", type=int, default=1000, help="Minimum area or volume for a cell to be kept")
    parser.add_argument("--filter", type=str, default=None, help="Filter to apply to the image before segmentation")
    parser.add_argument(
        "--filter_args", type=parse_dict, default={}, help="Additional filter arguments (e.g., key1=val1,key2=val2)"
    )
    parser.add_argument("--gpu", type=bool, default=False, help="Use GPU")
    parser.set_defaults(func=cellpose)
    return parser


def cellpose(
    input: str | Path,
    fov: int,
    channels: list[str],
    output: str | Path = "./cellpose_polygons",
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    corrections: str | Path = None,
    color_usage: str = "{input}/color_usage.csv",
    z_slices: list[int] | slice | None = None,
    model: str = "nuclei",
    diameter: int = 18,
    cellprob_threshold: float = -4,
    downsample: int = None,
    do_3D: bool = False,
    model_args: dict = None,
    clear_border: bool = False,
    min_size: int = 1000,
    filter: str = None,
    filter_args: dict = None,
    gpu: bool = False,
):
    """Segment cells using Cellpose.

    fishtank cellpose -i input -f 1 --channels DAPI -o cellpose_polygons

    Parameters
    ----------
    input
        Image file directory.
    fov
        Field of view to process.
    channels
        Channel for segmentation (e.g., DAPI or PolyT,DAPI).
    output
        Output directory.
    file_pattern
        Naming pattern for image files.
    corrections
        Path to image corrections directory.
    color_usage
        Path to color usage file.
    z_slices
        Z-slices to use for segmentation (e.g., 1 or 1,2,3 or 1:20:5).
    model
        Cellpose model (e.g., cyto or nuclei).
    diameter
        Cellpose diameter.
    cellprob_threshold
        Cell probability threshold.
    downsample
        Downsampling factor.
    do_3D
        Use 3D segmentation.
    model_args
        Additional model arguments (e.g., key1=val1,key2=val2).
    clear_border
        Remove cells touching the image border.
    min_size
        Minimum area or volume for a cell to be kept.
    filter
        Filter to apply to the image before segmentation.
    filter_args
        Additional filter arguments (e.g., key1=val1,key2=val2).
    gpu
        Use GPU.
    """
    # Import Cellpose
    from cellpose import models

    # Setup logger
    logger = logging.getLogger("cellpose")
    logger = FOVLoggerAdapter(logger, {"fov": fov})
    logger.info(f"fishtank version: {ft.__version__}")
    # Read image
    logger.info("Loading image")
    if "{input}" in color_usage:
        color_usage = color_usage.format(input=input)
    use_channels = channels
    channels = ft.io.read_color_usage(color_usage)
    channels = channels.set_index("bit").loc[use_channels, :].reset_index()
    img, attrs = ft.io.read_fov(input, fov, channels=channels, file_pattern=file_pattern, z_slices=z_slices)
    logger.info(f"Loaded image with shape: {img.shape}")
    # Correct illumination
    if corrections is not None:
        logger.info(f"Correcting illumination with {corrections}")
        img = ft.correct.illumination(img, channels.color.values, corrections)
    else:
        logger.info("No correction path provided, skipping illumination correction")
    # Apply filter
    if filter is not None:
        if hasattr(ft.filters, filter):
            logger.info(f"Applying {filter} filter from fishtank")
            if filter == "deconwolf":
                filter_args["gpu"] = gpu
                filter_args["z_step"] = round(attrs["z_offsets"][1] - attrs["z_offsets"][0], 3)
                filter_args["colors"] = attrs["colors"]
            img = getattr(ft.filters, filter)(img, **filter_args)
        elif hasattr(ski.filters, filter):
            logger.info(f"Applying {filter} filter from skimage")
            if len(channels) > 1:
                filter_args["channel_axis"] = 0
            img = getattr(ski.filters, filter)(img, **filter_args)
        else:
            raise ValueError(f"{filter} filter found in fishtank.filters or skimage.filters.")
    else:
        logger.info("No filter provided, skipping filtering")
    # Downsample image
    if downsample is not None:
        logger.info(f"Downsampling image by a factor of {downsample}")
        downscale = (1,) * (img.ndim - 2) + (downsample, downsample)
        img = ski.transform.downscale_local_mean(img, downscale)
    # Segment cells
    logger.info("Segmenting cells")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.Cellpose(model_type=model, gpu=gpu)
    masks = model.eval(
        img,
        diameter=diameter,
        channels=[0, 0] if len(channels) == 1 else [0, 1],
        cellprob_threshold=cellprob_threshold,
        do_3D=do_3D,
        **model_args,
    )[0]
    # Clear border
    if clear_border:
        logger.info("Clearing border")
        masks = ski.segmentation.clear_border(np.pad(masks, ((1, 1), (0, 0), (0, 0)), mode="constant"))[1:-1, :, :]
    # Convert to polygons
    logger.info("Converting to polygons")
    polygons = ft.seg.masks_to_polygons(masks)
    if downsample is not None:
        polygons.geometry = polygons.geometry.affine_transform([downsample, 0, 0, downsample, 0, 0])
    if "z" not in polygons.columns:
        polygons["z"] = z_slices[0]
    elif z_slices is not None:
        polygons["z"] = np.array(z_slices)[polygons["z"].values]
    z_mapping = {z: attrs["z_offsets"][i] for i, z in enumerate(sorted(polygons["z"].unique()))}
    polygons["global_z"] = polygons["z"].map(z_mapping)
    # Filter small cells
    logger.info(f"Filtering out small cells with size < {min_size}")
    polygons.index = polygons["cell"].values
    polygons["area"] = polygons.area
    polygons["area"] = polygons.groupby("cell")["area"].transform("sum")
    n_before = polygons["cell"].nunique()
    polygons = polygons[polygons["area"] > min_size].copy()
    n_after = polygons["cell"].nunique()
    logger.info(f"Removed {n_before - n_after} out of {n_before} total cells")
    # Save results
    logger.info(f"Saving polygons to {output} / polygons_{fov}.json")
    polygons["fov"] = fov
    polygons["x_offset"] = attrs["stage_position"][0]
    polygons["y_offset"] = attrs["stage_position"][1]
    output.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polygons.to_file(output / f"polygons_{fov}.json", driver="GeoJSON")
    logger.info("Done")
