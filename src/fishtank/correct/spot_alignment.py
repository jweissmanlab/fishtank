import geopandas as gpd
import pandas as pd


def spot_alignment(
    spots: pd.DataFrame,
    alignment: gpd.GeoDataFrame,
    x: str = "global_x",
    y: str = "global_y",
    z: str | None = None,
    rotation: np.ndarray | None = None,
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
        the name of the z column in the spots. If None, aligment is done in 2D.
    rotation
        a 2D rotation matrix to apply to the spots before alignment.
        
    Returns
    -------
    spots
        the spots DataFrame with x and y columns shifted based on alignment.
    """
    spots = gpd.GeoDataFrame(spots, geometry=gpd.points_from_xy(spots[x], spots[y]))
    if rotation is not None:
        # apply rotation to these spots
        spots[[x,y]] = spots[[x,y]] @ rotation[:2,:2] # apply rotation
    elif 'rotation' in alignment.columns:
        rotation = alignment['rotation'].iloc[0]
        spots[[x,y]] = spots[[x,y]] @ rotation[:2,:2]
    
    spots = gpd.sjoin(spots, alignment, how="left", predicate="within")
    spots["x_shift"] = spots["x_shift"].fillna(0)
    spots["y_shift"] = spots["y_shift"].fillna(0)
    #print(np.max(spots["x_shift"]), np.min(spots["y_shift"]))
    
    spots[x] = spots[x] + spots["x_shift"]
    spots[y] = spots[y] + spots["y_shift"]
    spots.drop(columns=["index_right", "x_shift", "y_shift", "geometry"], inplace=True)
    return spots
