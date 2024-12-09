import argparse
import logging
import multiprocessing as mp
import warnings
from functools import partial

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely as shp
import skimage as ski
from tqdm import tqdm

import fishtank as ft

from ._utils import parse_path


def get_parser():
    """Get parser for align_experiments script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--ref", type=parse_path, required=True, help="Reference image directory")
    parser.add_argument("-m", "--moving", type=parse_path, required=True, help="Moving image directory")
    parser.add_argument("--ref_series", type=str, required=True, help="Reference series to use for alignment")
    parser.add_argument("--moving_series", type=str, required=True, help="Moving series to use for alignment")
    parser.add_argument("--rotation", type=parse_path, default=None, 
                        help="File path to rotation matrix, moving relative to ref, if available")    
    parser.add_argument("-o", "--output", type=parse_path, default="alignment.json", help="Output file path")
    parser.add_argument("--color", type=int, default=405, help="Color channel to use for alignment")
    parser.add_argument("--z_offset", type=int, default=-3, help="Z offset for alignment")
    parser.add_argument(
        "--file_pattern", type=str, default="{series}/Conv_zscan_{fov}.dax", help="Naming pattern for image files"
    )
    parser.add_argument("--downsample", type=int, default=4, help="Image downsample factor")
    parser.add_argument("--filter_sigma", type=float, default=5, help="Sigma for unsharp mask filter")
    parser.add_argument("--attachment", type=float, default=15, help="Attachment factor for optical flow")
    parser.add_argument("--tile_size", type=int, default=100, help="Size of shift tiles in pixels")
    parser.set_defaults(func=main)
    return parser


def _get_slice_and_resolution(path, series, file_pattern, z_offset):
    """Get z-slice and resolution from xml file"""
    xml_path = path / ft.utils.determine_fov_format(path, series, file_pattern).format(series=series, fov=1)
    attrs = ft.io.read_xml(xml_path.with_suffix(".xml"))
    nearest_index = int(np.argmin(np.abs(np.array(attrs["z_offsets"]) - z_offset)))
    return nearest_index, attrs["micron_per_pixel"]


def _paste_image(image, target, target_bounds, source_bounds):
    """Paste an image into a target image given bounds."""
    x_offset = source_bounds[0] - target_bounds[0]
    y_offset = source_bounds[1] - target_bounds[1]
    target[y_offset : y_offset + image.shape[0], x_offset : x_offset + image.shape[1]] = image


def _combined_mosaic(ref_mosaic, moving_mosaic, ref_px, moving_px):
    """Combine two mosaics into a single mosaic."""
    combined_px = np.array([ref_px, moving_px])
    combined_px = np.concatenate([combined_px[:, :2].min(axis=0), combined_px[:, 2:].max(axis=0)])
    combined_mosaic = np.zeros(
        (3, combined_px[3] - combined_px[1], combined_px[2] - combined_px[0]), dtype=moving_mosaic.dtype
    )
    _paste_image(ref_mosaic, combined_mosaic[0], combined_px, ref_px)
    _paste_image(moving_mosaic, combined_mosaic[1], combined_px, moving_px)
    _paste_image(ref_mosaic, combined_mosaic[2], combined_px, ref_px)
    return combined_mosaic, combined_px

def _rotate_mosaic(mosaic, 
                   mosaic_bounds, 
                   rotation_matrix_filename,
                   rotation_center=[0,0],
                   micron_per_pixel: float = 0.107,
                   ):
    """Rotate the mosaic with a rotation matrix, centered around 0"""
    from cv2 import warpAffine, BORDER_CONSTANT
    from fishtank.utils import load_rotation
    # get original shape
    mosaic_height, mosaic_width = mosaic.shape
    # get rotation center:
    if rotation_center is None:
        center_y, center_x = mosaic_height // 2, mosaic_width // 2
    else:
        center_y, center_x = rotation_center
    # get rotation matrix    
    rotation = load_rotation(rotation_matrix_filename, center_y=center_y, center_x=center_x)
    # generate four corners:
    #bounds = np.array([offset_x, offset_y, offset_x + max_x, offset_y + max_y]) * micron_per_pixel
    corners = np.array([[mosaic_bounds[0], mosaic_bounds[1]],
                            [mosaic_bounds[2], mosaic_bounds[1]],
                            [mosaic_bounds[0], mosaic_bounds[3]],
                            [mosaic_bounds[2], mosaic_bounds[3]]])
    rotated_corners = (corners @ rotation[:,:2])
    print(corners, rotated_corners)
    
    # generate new bounds
    rotated_bounds = np.concatenate([np.min(rotated_corners, axis=0), np.max(rotated_corners, axis=0)])
    # generate new mosaic size:
    rotation_cosine, rotation_sine = np.abs(rotation[0,0]), np.abs(rotation[0,1])
    new_height = int(mosaic_height * rotation_cosine + mosaic_width * rotation_sine)
    new_width = int(mosaic_height * rotation_sine + mosaic_width * rotation_cosine)
    
    # generate new rotation center
    #rotation[0,2] += (new_width/2)-center_x
    #rotation[1,2] += (new_height/2)-center_y
    # rotate the mosaic
    rotated_mosaic = warpAffine(mosaic, rotation, 
                                (new_width, new_height), 
                                borderMode=BORDER_CONSTANT, 
                                borderValue=np.min(mosaic))
    return rotated_mosaic, rotated_bounds, rotation, (center_x, center_y)

def _save_mosaic(path, suffix, mosaic, dpi=600):
    """Save as a png file"""
    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(np.moveaxis(mosaic, 0, -1))
    ax.axis("off")
    path = path.with_stem(path.stem + suffix).with_suffix(".png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _compute_optical_flow(tile, attachment=15):
    """Compute optical flow for a tile."""
    tile, positions = tile
    return (ski.registration.optical_flow_tvl1(tile[0], tile[1], attachment=attachment), positions)


def main(args):
    """Align two experiments using optical flow."""
    # Setup
    logger = logging.getLogger("align_experiments")
    logger.info(f"fishtank version: {ft.__version__}")
    ref_z_slice, ref_resolution = _get_slice_and_resolution(args.ref, args.ref_series, args.file_pattern, args.z_offset)
    ref_resolution = args.downsample * ref_resolution
    logger.info(f"Reference z-slice: {ref_z_slice}")
    moving_z_slice, moving_resolution = _get_slice_and_resolution(
        args.moving, args.moving_series, args.file_pattern, args.z_offset
    )
    moving_resolution = args.downsample * moving_resolution
    logger.info(f"Moving z-slice: {moving_z_slice}")
    # Load mosaics
    _read_mosaic = partial(
        ft.io.read_mosaic,
        colors=args.color,
        downsample=args.downsample,
        file_pattern=args.file_pattern,
        filter=ft.filters.unsharp_mask,
        filter_args={"sigma": args.filter_sigma},
    )
    logger.info("Loading reference mosaic.")
    ref_mosaic, ref_bounds = _read_mosaic(args.ref, z_slices=ref_z_slice, series=args.ref_series)
    ref_mosaic = ski.exposure.rescale_intensity(
        ref_mosaic, in_range=(0, np.percentile(ref_mosaic, 98)), out_range=(0, 1)
    )
    logger.info("Loading moving mosaic.")
    moving_mosaic, moving_bounds = _read_mosaic(args.moving, z_slices=moving_z_slice, series=args.moving_series)
    moving_mosaic = ski.exposure.rescale_intensity(
        moving_mosaic, in_range=(0, np.percentile(moving_mosaic, 98)), out_range=(0, 1)
    )
    # Apply rotation matrix if available:
    if args.rotation is not None:
        logger.info(f"Rotating moving mosaic with rotation: {args.rotation}")
        moving_mosaic, moving_bounds, rotation_matrix, rotation_center = _rotate_mosaic(moving_mosaic, moving_bounds, args.rotation)
    # Combined mosaic
    moving_px = (moving_bounds[[0, 1, 0, 1]] / moving_resolution).astype(int)
    moving_px += np.array([0, 0, moving_mosaic.shape[1], moving_mosaic.shape[0]])
    ref_px = (ref_bounds[[0, 1, 0, 1]] / ref_resolution).astype(int)
    ref_px += np.array([0, 0, ref_mosaic.shape[1], ref_mosaic.shape[0]])
    combined_mosaic, combined_px = _combined_mosaic(ref_mosaic, moving_mosaic, ref_px, moving_px)
    # Coarse alignment
    logger.info("Performing coarse alignment with phase cross correlation.")
    coarse_shift = ski.registration.phase_cross_correlation(combined_mosaic[0], combined_mosaic[1])[0].astype(int)
    logger.info(f"Coarse shift: {coarse_shift}")
    shifted_moving_px = moving_px + np.array([coarse_shift[1], coarse_shift[0], coarse_shift[1], coarse_shift[0]])
    combined_mosaic, combined_px = _combined_mosaic(ref_mosaic, moving_mosaic, ref_px, shifted_moving_px)
    _save_mosaic(args.output, "_coarse", combined_mosaic)
    # Fine alignment
    tiles, positions = ft.utils.tile_image(combined_mosaic, tile_shape=(1000, 1000))
    logger.info(f"Computing optical flow with attachment {args.attachment}.")
    parallel_func = partial(_compute_optical_flow, attachment=args.attachment)
    with mp.Pool(mp.cpu_count()) as pool:
        flows = list(tqdm(pool.imap_unordered(parallel_func, zip(tiles, positions, strict=False)), total=len(tiles)))
    flows, positions = zip(*flows, strict=True)
    flow, _ = ft.utils.create_mosaic(flows, positions, micron_per_pixel=1)
    # Warp moving mosaic
    logger.info("Plotting fine alignment.")
    nr, nc = combined_mosaic[0].shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    warped = ski.transform.warp(combined_mosaic[1], np.array([row_coords + flow[0], col_coords + flow[1]]), mode="edge")
    combined_mosaic[1] = warped
    _save_mosaic(args.output, "_fine", combined_mosaic)
    # Calculate weighted average shift for each tile
    logger.info("Calculating mean shift for each tile.")
    weighted_flow = np.concatenate([flow, combined_mosaic[[0]]], axis=0)  # weight flow by moving mosaic
    tiles, positions = ft.utils.tile_image(weighted_flow, tile_shape=(args.tile_size, args.tile_size))
    tile_size_micron = args.tile_size * moving_resolution
    alignment = []
    for tile, position in zip(tiles, positions, strict=True):
        x_offset = (position[0] + combined_px[0] - shifted_moving_px[0]) * moving_resolution + moving_bounds[0]
        y_offset = (position[1] + combined_px[1] - shifted_moving_px[1]) * moving_resolution + moving_bounds[1]
        geometry = shp.geometry.box(x_offset, y_offset, x_offset + tile_size_micron, y_offset + tile_size_micron)
        if np.all(tile[2] == 0):
            continue
        tile_fine = np.average(tile[:2], weights=tile[2], axis=(1, 2))
        alignment.append(
            {
                "geometry": geometry,
                "x_shift": (coarse_shift[1] - tile_fine[1]) * moving_resolution,
                "y_shift": (coarse_shift[0] - tile_fine[0]) * moving_resolution,
                "rotation": rotation_matrix,
                "rotation_center": rotation_center,
            }
        )
    alignment = gpd.GeoDataFrame(alignment, geometry="geometry", crs=None)
    # Save alignment
    logger.info(f"Saving alignment tiles to {args.output}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore crs warning
        alignment.to_file(args.output, driver="GeoJSON")
    logger.info("Done")
