import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely as shp


def determine_fov_format(
    path: str | Path, series: int | str, file_pattern: str = "{series}/Conv_zscan_{fov}.dax", fov: int | str = 1
) -> str:
    """Determine FOV format.

    Parameters
    ----------
    path
        Path to files
    series
        Series name.
    file_pattern
        Naming pattern of fov files.
    fov
        Field of view number.

    Returns
    -------
    file_pattern
        a file pattern with the correct fov format.
    """
    path = Path(path)
    for format in ["{fov:01d}", "{fov:02d}", "{fov:03d}"]:
        if format in file_pattern:
            return file_pattern
        if os.path.exists(path / file_pattern.replace("{fov}", format).format(fov=fov, series=series)):
            return file_pattern.replace("{fov}", format)
    raise ValueError(f"Could not find file matching {file_pattern.format(series = series,fov = '[fov]')} in {path}")


def tile_polygons(
    polygons: gpd.GeoDataFrame, tile_shape: tuple = (500, 500), buffer: int = 25, cell: str = "cell"
) -> tuple[list[gpd.GeoDataFrame], list]:
    """Split polygons into tiles.

    Parameters
    ----------
    polygons
        a GeoDataFrame of cell outlines.
    tile_shape
        the shape of the tiles to divide the polygons into.
    buffer
        Add a buffer around the tiles to include polygons that overlap the edges.
    cell
        the name of the cell column in the GeoDataFrame.

    Returns
    -------
    tiles
        a list of GeoDataFrames of cell outlines for each tile.
    bounds
        The bounds for each tile.

    """
    centers = polygons.groupby(cell).first().geometry.centroid
    polygons["center"] = polygons[cell].map(centers)
    total_bounds = polygons.total_bounds
    x_splits = np.append(np.arange(total_bounds[0], total_bounds[2], tile_shape[0]), total_bounds[2])
    y_splits = np.append(np.arange(total_bounds[1], total_bounds[3], tile_shape[1]), total_bounds[3])
    tiles = []
    bounds = []
    for x in range(len(x_splits) - 1):
        for y in range(len(y_splits) - 1):
            bound = shp.geometry.box(x_splits[x], y_splits[y], x_splits[x + 1], y_splits[y + 1])
            buffered_bound = bound.buffer(buffer)
            subset = polygons[polygons.center.within(buffered_bound)].copy()
            if len(subset) == 0:
                continue
            subset["in_tile"] = subset.center.within(bound)
            tiles.append(subset.drop(columns="center"))
            bounds.append(bound)
    return tiles, bounds


def tile_image(image: np.ndarray, tile_shape: tuple = (1000, 1000)) -> tuple[list[np.ndarray], list]:
    """Split image into tiles.

    Parameters
    ----------
    image
        a (C,Y,X), (C,Z,Y,X) or (Y,X) image to split into tiles.
    tile_shape
        the shape of the tiles to divide the polygons into.

    Returns
    -------
    tiles
        a list of image tiles.
    positions
        a list of (x,y) positions for each tile.
    """
    tiles = []
    positions = []
    for i in range(0, image.shape[-2], tile_shape[0]):
        for j in range(0, image.shape[-1], tile_shape[1]):
            if len(image.shape) == 2:
                tile = image[i : i + tile_shape[0], j : j + tile_shape[1]]
            elif len(image.shape) == 3:
                tile = image[:, i : i + tile_shape[0], j : j + tile_shape[1]]
            elif len(image.shape) == 4:
                tile = image[:, :, i : i + tile_shape[0], j : j + tile_shape[1]]
            tiles.append(tile)
            positions.append((j, i))
    return tiles, positions


def create_mosaic(
    imgs: list[np.ndarray],
    positions: list,
    micron_per_pixel: float = 0.107,
) -> tuple[np.ndarray, np.ndarray]:
    """Create mosaic from images.

    Parameters
    ----------
    imgs
        a list (C,Y,X) or (Y,X) images to create a mosaic from.
    positions
        a list of (x,y) positions for each image.
    micron_per_pixel
        the micron per pixel conversion factor.

    Returns
    -------
    mosaic
        an image mosaic of all the images.
    bounds
        the bounds of the mosaic in microns.
    """
    # Convert positions from microns to pixels
    positions_pixels = [(int(x / micron_per_pixel), int(y / micron_per_pixel)) for x, y in positions]
    # Adjust positions with the offsets
    offset_x = min(pos[0] for pos in positions_pixels)
    offset_y = min(pos[1] for pos in positions_pixels)
    adjusted_positions = [(pos[0] - offset_x, pos[1] - offset_y) for pos in positions_pixels]
    # Calculate mosaic size, including maximum channel count
    max_x = max(pos[0] + img.shape[-1] for img, pos in zip(imgs, adjusted_positions, strict=True))
    max_y = max(pos[1] + img.shape[-2] for img, pos in zip(imgs, adjusted_positions, strict=True))
    # Initialize the canvas
    if len(imgs[0].shape) == 2:
        mosaic = np.zeros((max_y, max_x), dtype=imgs[0].dtype)
    elif len(imgs[0].shape) == 3:
        mosaic = np.zeros((imgs[0].shape[0], max_y, max_x), dtype=imgs[0].dtype)
    else:
        raise ValueError("Function does not support images with more than 3 dimensions")
    # Paste each image into its adjusted position on the canvas
    for img, (x, y) in zip(imgs, adjusted_positions, strict=True):
        if len(img.shape) == 2:
            mosaic[y : y + img.shape[-2], x : x + img.shape[-1]] = img
        elif len(img.shape) == 3:
            mosaic[:, y : y + img.shape[-2], x : x + img.shape[-1]] = img
    bounds = np.array([offset_x, offset_y, offset_x + max_x, offset_y + max_y]) * micron_per_pixel
    return mosaic, bounds
