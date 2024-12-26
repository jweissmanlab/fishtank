import logging
import multiprocessing as mp
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from fishtank.utils import tile_polygons


def _align_z(spots, polygons, spots_z, polygons_z, z="_z", logger=None):
    """Aligns z values between spots and polygons."""
    # Get unique z slices
    unique_spots_z = pd.DataFrame({spots_z: spots[spots_z].unique()})
    unique_polygons_z = pd.DataFrame({polygons_z: polygons[polygons_z].unique()})
    polygons_z_step = unique_polygons_z[polygons_z].diff().max()
    # Log
    if logger is not None:
        logger.info(
            f"{len(unique_spots_z)} unique {spots_z} values in spots range from {unique_spots_z[spots_z].min()} to {unique_spots_z[spots_z].max()}"
        )
        logger.info(
            f"{len(unique_polygons_z)} unique {polygons_z} values in polygons range from {unique_polygons_z[polygons_z].min()} to {unique_polygons_z[polygons_z].max()}"
        )

    # Get z mapping
    def map_closest_z(z_fine, z_coarse_values, threshold=1):
        closest_value = z_coarse_values[np.abs(z_coarse_values - z_fine).argmin()]
        return closest_value if abs(z_fine - closest_value) <= threshold else np.nan

    unique_spots_z[z] = unique_spots_z[spots_z].apply(
        lambda x: map_closest_z(x, unique_polygons_z[polygons_z].values, threshold=polygons_z_step / 2)
    )
    unique_spots_z[z] = unique_spots_z[z].fillna(unique_spots_z[spots_z])
    # Update spots and polygons
    spots = pd.merge(spots, unique_spots_z[[spots_z, z]], on=spots_z, how="left")
    polygons[z] = polygons[polygons_z]
    return spots, polygons


def _assign_nearest(tile, z="z", max_dist=5):
    """Assigns points to the nearest polygon in 3D space."""
    # Initialize distance and nearest polygon columns
    points, polygons = tile
    points["dist"] = np.inf
    points["nearest"] = pd.NA
    # Iterate over z slices
    for z_slice in points[z].unique():
        # Iterate through over neighboring z slices with increasing distance
        neighbor_z_dists = {
            int(abs(z_dist - z_slice)) for z_dist in polygons[z].unique() if abs(z_dist - z_slice) <= max_dist
        }
        for z_dist in sorted(neighbor_z_dists):
            # Filter to polygons within the neighboring z slices
            neighbor_slices = [z_slice + z_dist, z_slice - z_dist]  # noqa: F841
            filtered_polygons = polygons.query(f"{z} in @neighbor_slices")
            # Filter to points that could have closer polygon given current z_dist
            filtered_points = points.query(f"{z} == @z_slice & dist > @z_dist")
            # Perform nearest neighbor join within the specified max_dist
            if not filtered_points.empty and not filtered_polygons.empty:
                nearest = filtered_points.sjoin_nearest(
                    filtered_polygons,
                    how="left",
                    distance_col="dist",
                    lsuffix="point",
                    rsuffix="polygon",
                    max_distance=max_dist + 1e-6,
                )
                # Remove duplicate points and calculate 3D distance
                nearest = nearest[~nearest.index.duplicated(keep="first")]
                nearest["dist"] = (nearest["dist"] ** 2 + z_dist**2) ** 0.5
                # Update the best shape and distance in the original points DataFrame
                update_mask = nearest.index[nearest["dist"] < points.loc[nearest.index, "dist"]]
                points.loc[update_mask, "dist"] = nearest.loc[update_mask, "dist"]
                points.loc[update_mask, "nearest"] = nearest.loc[update_mask, "index_polygon"]
    points.loc[points["dist"] > max_dist, "dist"] = pd.NA
    points.loc[points.dist.isna(), "nearest"] = pd.NA
    return points[["dist", "nearest"]]


def _tile_points(points, bounds):
    """Split points into tiles based on bounds."""
    n_tiles = len(bounds)
    bounds = gpd.GeoDataFrame(geometry=bounds)
    points = gpd.sjoin(points, bounds, how="left").query("index_right.notna()")
    tiles = []
    for tile in range(n_tiles):  # noqa: B007
        tiles.append(points.query("index_right == @tile").drop(columns="index_right").copy())
    points.drop(columns="index_right", inplace=True)
    return tiles


def assign_spots(
    spots: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
    max_dist: float | int = 5,
    cell: str = "cell",
    x: str = "global_x",
    y: str = "global_y",
    z: str | None = None,
    polygons_z: str | None = None,
) -> pd.DataFrame:
    """Assigns spots to the nearest polygon

    Parameters
    ----------
    spots
        a DataFrame of spot coordinates. Can be 2D or 3D.
    polygons
        a GeoDataFrame of cell outlines. Can be 2D or 3D.
    max_dist
        the maximum distance to search for the nearest polygon.
    cell
        the name of the cell column in the polygons.
    x
        the name of the x column in the spots.
    y
        the name of the y column in the spots.
    z
        the name of the z column in the spots. If None, the polygons are assumed to be 2D.
    polygons_z
        the name of the z column in the polygons. If None, the spots z values will be used.

    Returns
    -------
    spots
        the spots DataFrame with `{cell}` and `{cell}_dist` columns added.
    """
    # Setup
    logger = logging.getLogger("assign_spots")
    logger.setLevel(logging.INFO)
    if polygons_z is None:
        polygons_z = z
    polygons.reset_index(drop=True, inplace=True)
    spots.reset_index(drop=True, inplace=True)
    has_layers = z is not None
    if not has_layers:
        logger.info("No z column provided. Assuming 2D.")
        z = "_z"
        spots["_z"] = 0
        polygons["_z"] = 0
    points = gpd.GeoDataFrame(spots[z], geometry=gpd.points_from_xy(spots[x], spots[y]))
    # Align z
    if has_layers:
        logger.info("Aligning z values between spots and polygons.")
        points, polygons = _align_z(points, polygons, z, polygons_z, "_z", logger=logger)
    # Split into tiles
    logger.info("Splitting polygons and spots into tiles.")
    polygon_tiles, bounds = tile_polygons(polygons)
    point_tiles = _tile_points(points, bounds)
    # Assign nearest polygons
    logger.info("Assigning spots to nearest polygon in parallel.")
    parallel_func = partial(_assign_nearest, z="_z", max_dist=max_dist)
    with mp.Pool(mp.cpu_count()) as pool:
        nearest = list(
            tqdm(
                pool.imap_unordered(parallel_func, zip(point_tiles, polygon_tiles, strict=True)), total=len(point_tiles)
            )
        )
    # Merge the nearest polygons back into the spots df
    nearest = pd.concat(nearest).query("nearest.notna()")
    # return nearest, polygons
    nearest[cell] = polygons.loc[nearest["nearest"], cell].values
    spots.drop(columns=[cell, f"{cell}_dist"], inplace=True, errors="ignore")
    spots = spots.merge(nearest[[cell, "dist"]], left_index=True, right_index=True, how="left")
    spots.rename(columns={"dist": f"{cell}_dist"}, inplace=True)
    if not has_layers:
        spots.drop(columns="_z", inplace=True)
    return spots
