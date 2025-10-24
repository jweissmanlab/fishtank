import logging
import multiprocessing as mp
from functools import partial

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely as shp
from tqdm import tqdm

from fishtank.utils import tile_polygons


def _get_edge_polygons(polygons, fov="fov", cell="cell", buffer=-10):
    """Split the polygons into edge and interior sets."""
    fov_bounds = (
        polygons.groupby(fov).geometry.apply(lambda x: shp.geometry.box(*x.total_bounds)).reset_index(name="geometry")
    )
    edge_region = fov_bounds.overlay(fov_bounds, how="intersection").query(f"{fov}_1 != {fov}_2").union_all()
    polygons_2d = polygons.dissolve("cell").reset_index()
    edge_cells = polygons_2d.loc[polygons_2d.geometry.intersects(edge_region), cell]
    edge_polygons = polygons[polygons[cell].isin(edge_cells)].copy()
    interior_polygons = polygons[~polygons[cell].isin(edge_cells)].copy()
    return edge_polygons, interior_polygons


def _get_overlap_graph(polygons_2d, cell="cell", fov="fov"):
    """Create a graph of overlapping polygons."""
    overlaps = polygons_2d[[cell, fov, "geometry"]].sjoin(
        polygons_2d[[cell, fov, "geometry"]], how="inner", predicate="intersects"
    )
    overlaps = overlaps.query(f"{cell}_left != {cell}_right & {fov}_left != {fov}_right")
    overlap_graph = nx.Graph()
    overlap_graph.add_edges_from(list(overlaps[[f"{cell}_left", f"{cell}_right"]].itertuples(index=False, name=None)))
    return overlap_graph


def _combine_3d(polygon_1, polygon_2, z="global_z"):
    """intersection, union, and difference of 3D polygons."""
    merged = polygon_1.merge(polygon_2, how="outer", on=[z], suffixes=["_1", "_2"])
    merged = merged.fillna(shp.geometry.Polygon())
    merged["intersection"] = merged.geometry_1.intersection(merged.geometry_2)
    intersection = merged.loc[~merged["intersection"].is_empty, ["intersection", z]].rename(
        columns={"intersection": "geometry"}
    )
    merged["union"] = merged.geometry_1.union(merged.geometry_2)
    union = merged.loc[~merged["union"].is_empty, ["union", z]].rename(columns={"union": "geometry"})
    merged["difference"] = merged.geometry_1.difference(merged.geometry_2)
    difference = merged.loc[~merged["difference"].is_empty, ["difference", z]].rename(
        columns={"difference": "geometry"}
    )
    return intersection, union, difference


def _area_3d(polygon):
    """Area of a 3D polygon."""
    return float(np.sum(polygon.geometry.area))


def _get_z_mapping(df, z):
    """Get columns that are constant for each z-slice."""
    z_columns = [c for c in df.columns if c != z and df.groupby(z)[c].nunique().max() == 1]
    return df.groupby(z)[z_columns].first().reset_index()


def _fix_overlaps(polygons, min_ioa=0.2, cell="cell", z="global_z", fov="fov", tolerance=0.5):
    """Fix overlapping polygons by merging > min_ioa."""
    z_mapping = _get_z_mapping(polygons, z)
    polygons_2d = polygons.dissolve(cell).reset_index().drop(columns=[z])
    polygons_3d = {i: polygons.loc[polygons[cell] == i, ["geometry", z]] for i in polygons[cell].unique()}
    overlap_graph = _get_overlap_graph(polygons_2d, cell, fov)
    while len(overlap_graph.edges) > 0:
        edge = list(overlap_graph.edges)[0]
        polygon_1 = polygons_3d[edge[0]]
        polygon_2 = polygons_3d[edge[1]]
        intersection, union, difference = _combine_3d(polygon_1, polygon_2, z)
        intersection_area = _area_3d(intersection)
        # Do nothing if the intersection is empty
        if intersection_area == 0:
            overlap_graph.remove_edge(*edge)
        # Merge polygons if intersection area is greater than min_ioa
        elif intersection_area / min(_area_3d(polygon_1), _area_3d(polygon_2)) > min_ioa:
            for neighbor in list(overlap_graph.neighbors(edge[0])):
                if neighbor != edge[1]:
                    overlap_graph.add_edge(neighbor, edge[1])
            overlap_graph.remove_node(edge[0])
            del polygons_3d[edge[0]]
            union.geometry = union.geometry.simplify(tolerance).make_valid()
            polygons_3d[edge[1]] = union
        # Otherwise, subtract one polygon from the other
        else:
            overlap_graph.remove_edge(*edge)
            difference.geometry = difference.geometry.simplify(tolerance).make_valid()
            polygons_3d[edge[0]] = difference
    polygons_3d = gpd.GeoDataFrame(
        pd.concat(polygons_3d).reset_index(names=["cell", "drop"]), geometry="geometry"
    ).drop(columns="drop")
    polygons = polygons_3d.merge(polygons_2d.drop(columns="geometry"), on=cell, how="left")
    polygons = polygons[polygons.area > 1].copy()
    polygons = polygons.drop(columns=set(z_mapping.columns) - {z}).merge(z_mapping, on=z, how="left")
    return polygons


def fix_overlaps(
    polygons: gpd.GeoDataFrame,
    min_ioa: float = 0.2,
    cell: str = "cell",
    z: str | None = "global_z",
    fov: str = "fov",
    tile_shape: tuple = (500, 500),
    tolerance: float = 0.5,
    diameter: float = 20,
) -> gpd.GeoDataFrame:
    """Fix overlapping polygons from adjacent FOVs.

    Parameters
    ----------
    polygons
        a GeoDataFrame of cell outlines.
    min_ioa
        the minimum intersection over area to merge overlapping polygons.
    cell
        the name of the cell column in the GeoDataFrame.
    z
        the name of the z column in the GeoDataFrame. If None, the polygons are assumed to be 2D.
    fov
        the name of the fov column in the GeoDataFrame.
    tile_shape
        the shape of the tiles to divide the polygons into for parallel processing.
    tolerance
        the tolerance for simplifying the polygons.
    diameter
        Approximate diameter of the polygons.

    Returns
    -------
    polygons
        a GeoDataFrame of cell outlines.
    """
    # Setup
    logger = logging.getLogger("fix_overlaps")
    logger.setLevel(logging.INFO)
    if z is None:
        if polygons[cell].nunique() != len(polygons):
            raise ValueError(
                "2D polygons must have a unique cell identifier. If your polygons are 3D, please provide a z column."
            )
        z = "_z"
        polygons[z] = 0
    logger.info("Splitting polygons into edge and interior sets.")
    edge_polygons, interior_polygons = _get_edge_polygons(polygons, fov=fov, cell=cell, buffer=diameter / 2)
    logger.info("Splitting edge polygons into tiles.")
    # if theres no edge polygons just return the polygons
    try:
        edge_tiles, _ = tile_polygons(edge_polygons, tile_shape=tile_shape, buffer=diameter, cell=cell)
    except:
        logger.error("Error in tile_polygons. Please check the input polygons.")
        print(edge_polygons)
        print(polygons)
        raise ValueError(f"Error in tile_polygons, edge_polygons: {edge_polygons},length: {len(edge_polygons)}, interior_polygons length: {len(interior_polygons)}")

    logger.info("Fixing overlapping polygons in parallel.")
    parallel_func = partial(_fix_overlaps, min_ioa=min_ioa, cell=cell, z=z, fov=fov, tolerance=tolerance)
    with mp.Pool(mp.cpu_count()) as pool:
        edge_tiles = list(tqdm(pool.imap_unordered(parallel_func, edge_tiles), total=len(edge_tiles)))
    polygons = [tile.query("in_tile").drop(columns="in_tile") for tile in edge_tiles] + [interior_polygons]
    polygons = gpd.GeoDataFrame(pd.concat(polygons))
    if z == "_z":
        polygons = polygons.drop(columns=z)
    return polygons
