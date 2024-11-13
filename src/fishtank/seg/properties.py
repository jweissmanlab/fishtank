import geopandas as gpd
import pandas as pd


def polygon_properties(polygons: gpd.GeoDataFrame, cell: str = "cell", z: str | None = "global_z") -> pd.DataFrame:
    """Calculate geometric properties of polygons.

    Parameters
    ----------
    polygons
        a GeoDataFrame of cell outlines.
    cell
        the name of the cell column in the GeoDataFrame.
    z
        the name of the z column in the GeoDataFrame. If None, the polygons are assumed to be 2D.

    Returns
    -------
    properties
        a DataFrame of cell properties.
    """
    columns = polygons.columns
    polygons["area"] = polygons.area
    polygons["centroid_x"] = polygons.centroid.x * polygons["area"]
    polygons["centroid_y"] = polygons.centroid.y * polygons["area"]
    agg_fun = {col: "first" for col in set(columns) - {"geometry"}}
    agg_fun.update({"centroid_x": "sum", "centroid_y": "sum", "area": "sum"})
    if z is not None:
        polygons["centroid_z"] = polygons[z] * polygons["area"]
        polygons["n_layers"] = polygons.groupby(cell)[z].transform("count")
        agg_fun.update({"centroid_z": "sum", "n_layers": "first"})
    properties = polygons.groupby(cell).agg(agg_fun).reset_index(drop=True)
    polygons.drop(columns=set(polygons.columns) - set(columns), inplace=True)
    properties["centroid_x"] = properties["centroid_x"] / properties["area"]
    properties["centroid_y"] = properties["centroid_y"] / properties["area"]
    if z is not None:
        properties["centroid_z"] = properties["centroid_z"] / properties["area"]
        z_slices = polygons[z].unique()
        z_step = z_slices[1] - z_slices[0]
        properties["volume"] = properties["area"] * z_step
        properties.drop(columns="area", inplace=True)
    return properties
