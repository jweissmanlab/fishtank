import geopandas as gpd
import pandas as pd


def spot_alignment(
    spots: pd.DataFrame,
    alignment: gpd.GeoDataFrame,
    x: str = "global_x",
    y: str = "global_y",
    z: str | None = None,
) -> pd.DataFrame:
    """Adjust spot coordinates based on alignment.

    Parameters
    ----------
    spots
        a DataFrame of spot coordinates. Can be 2D or 3D.
    alignment
        a GeoDataFrame specifying the alignment between spots and polygons.
    x
        the name of the x column in the spots.
    y
        the name of the y column in the spots.
    z
        the name of the z column in the spots. If None, alignment is done in 2D.

    Returns
    -------
    spots
        the spots DataFrame with x and y columns shifted based on alignment.
    """
    spots = gpd.GeoDataFrame(spots, geometry=gpd.points_from_xy(spots[x], spots[y]))
    spots = gpd.sjoin(spots, alignment, how="left", predicate="within")
    if "rotation" in alignment.columns:
        rotation = alignment["rotation"].value_counts().idxmax()
        spots["geometry"] = spots["geometry"].rotate(-rotation, origin=(0, 0))
        spots[x] = spots.geometry.x
        spots[y] = spots.geometry.y
    spots["x_shift"] = spots["x_shift"].fillna(0)
    spots["y_shift"] = spots["y_shift"].fillna(0)
    spots[x] = spots[x] + spots["x_shift"]
    spots[y] = spots[y] + spots["y_shift"]
    spots.drop(columns=["index_right", "x_shift", "y_shift", "geometry", "rotation"], inplace=True, errors="ignore")
    return spots
