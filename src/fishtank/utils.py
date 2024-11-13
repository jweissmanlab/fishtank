from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely as shp


def parse_dict(arg: str) -> dict:
    """Parse dictionary input in the form of key1=val1,key2=val2"""
    if arg is None:
        return {}
    return dict(item.split("=") for item in arg.split(","))


def parse_list(arg: str) -> list:
    """Parse list input separated by commas"""
    if arg is None:
        return None
    return arg.split(",")


def parse_index(arg: str) -> list:
    """Parse index input in the form of start:end:step"""
    if arg is None:
        return None
    if "," in arg:
        return parse_list(arg)
    if ":" in arg:
        return list(range(*map(int, arg.split(":"))))
    return [int(arg)]


def parse_path(arg: str) -> Path:
    """Parse path input"""
    if arg is None:
        return None
    return Path(arg)


def tile_polygons(
    polygons: gpd.GeoDataFrame, tile_shape: tuple = (500, 500), buffer: int = 25, cell: str = "cell"
) -> list:
    """Split polygons into non-overlapping tiles."""
    polygons.index = polygons[cell].values
    polygons["center"] = polygons.groupby(cell).first().geometry.centroid
    total_bounds = polygons.total_bounds
    x_splits = np.append(np.arange(total_bounds[0], total_bounds[2], tile_shape[0]), total_bounds[2])
    y_splits = np.append(np.arange(total_bounds[1], total_bounds[3], tile_shape[1]), total_bounds[3])
    polygon_tiles = []
    for x in range(len(x_splits) - 1):
        for y in range(len(y_splits) - 1):
            tile = shp.geometry.box(x_splits[x], y_splits[y], x_splits[x + 1], y_splits[y + 1])
            buffered_tile = tile.buffer(buffer)
            subset = polygons[polygons.center.within(buffered_tile)].copy()
            if len(subset) == 0:
                continue
            subset["in_tile"] = subset.center.within(tile)
            polygon_tiles.append(subset.drop(columns="center"))
    return polygon_tiles
