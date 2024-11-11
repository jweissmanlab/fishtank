import geopandas as gpd
import numpy as np
import shapely as shp
import skimage as ski
from rasterio.features import rasterize


def masks_to_polygons(masks: np.ndarray, tolerance: float = 0.5, id: str = "cell", z: str = "z") -> gpd.GeoDataFrame:
    """Covert segmentation masks to polygons.

    Parameters
    ----------
    masks
        a (Y,X) or (Z,Y,X) numpy array of segmentation masks.
    tolerance
        The maximum allowed geometry displacement. The higher this value, the smaller the number of vertices in the resulting geometry.
    id
        The name of the id column in the resulting GeoDataFrame
    z
        The name of the z column in the resulting GeoDataFrame

    Returns
    -------
    polygons
        a GeoDataFrame of cell outlines.
    """
    # Setup
    polygons = []
    cells = []
    layers = []
    has_layers = len(masks.shape) > 2
    if not has_layers:
        masks = np.expand_dims(masks, axis=0)
    # Transpose and flip masks
    masks = np.flip(masks.swapaxes(-1, -2), axis=-1)
    # Convert masks to polygons
    for i in np.unique(masks)[1:]:
        mask = masks == i
        for l in np.where(mask.any(axis=(1, 2)))[0]:
            # Pad mask to ensure contours are closed
            padded_mask = np.pad(mask[l], 1, mode="constant")
            contours = ski.measure.find_contours(padded_mask, 0.5)
            contours = [contour - 1 for contour in contours]
            if len(contours[0]) > 3:
                polygons.append(shp.geometry.Polygon(contours[0]).simplify(tolerance=tolerance, preserve_topology=True))
                cells.append(i)
                layers.append(l)
    polygons = gpd.GeoDataFrame({id: cells, "geometry": polygons}, crs=None)
    if has_layers:
        polygons[z] = layers
    return polygons


def polygons_to_masks(
    polygons: gpd.GeoDataFrame, bounds: list, shape: tuple, id: str = "cell", z: str = "z"
) -> np.ndarray:
    """Convert polygons to segmentation masks.

    Parameters
    ----------
    polygons
        A GeoDataFrame of polygons.
    bounds
        The total bounds in the polygons coordinate system.
    shape
        The shape of the resulting masks in the form (Z,X,Y) or (X,Y).
    id
        The name of the id column in the polygons GeoDataFrame
    z
        The name of the z column in the polygons GeoDataFrame

    Returns
    -------
    masks
        a numpy array with specified shape.
    """
    # Setup
    masks = []
    has_layers = len(shape) > 2
    if has_layers and z not in polygons.columns:
        raise ValueError(f"The polygons GeoDataFrame must have a '{z}' column.")
    if not has_layers:
        polygons[z] = 0
        shape = (1,) + shape
    # Get transform
    x_res = (bounds[2] - bounds[0]) / shape[-1]
    y_res = (bounds[3] - bounds[1]) / shape[-2]
    transform = [x_res, 0, bounds[0], 0, -y_res, bounds[3]]
    # Convert polygons to masks
    for l in range(shape[0]):
        layer_polygons = polygons[polygons[z] == l]
        mask = rasterize(
            list(zip(layer_polygons["geometry"], layer_polygons[id], strict=False)),
            out_shape=shape[1:],
            transform=transform,
        )
        masks.append(mask)
    if has_layers:
        masks = np.stack(masks, axis=0)
    else:
        masks = masks[0]
    return masks.astype(np.uint16)
