import geopandas as gpd
import numpy as np
import pytest
import shapely as shp
import skimage as ski
from geopandas.testing import assert_geodataframe_equal

import fishtank as ft


def test_polygons_to_masks(polygons, masks):
    new_masks = ft.seg.polygons_to_masks(polygons, bounds=(0, 0, 288, 288), shape=(10, 288, 288))
    assert np.mean(masks == new_masks) > 0.99
    assert set(np.unique(new_masks)) - {0} == set(polygons["cell"].unique())


def test_polygons_to_masks_2d(polygons, masks):
    polygons = polygons.query("z == 5").drop(columns="z")
    new_masks = ft.seg.polygons_to_masks(polygons, bounds=(0, 0, 288, 288), shape=(288, 288))
    assert np.mean(masks[5] == new_masks) > 0.99
    assert set(np.unique(new_masks)) - {0} == set(polygons["cell"].unique())


def test_masks_to_polygons(polygons, masks):
    new_polygons = ft.seg.masks_to_polygons(masks)
    intersection = polygons.union_all().intersection(new_polygons.union_all())
    assert intersection.area / new_polygons.union_all().area > 0.99
    assert set(np.unique(masks)) - {0} == set(new_polygons["cell"].unique())


def test_masks_to_polygons_2d(polygons, masks):
    new_polygons = ft.seg.masks_to_polygons(masks[5])
    intersection = polygons.query("z == 5").union_all().intersection(new_polygons.union_all())
    assert intersection.area / new_polygons.union_all().area > 0.99
    assert set(np.unique(masks[5])) - {0} == set(new_polygons["cell"].unique())


def _masks_to_polygons_full_scan(masks, tolerance=0.5):
    """Previous masks_to_polygons, which scanned the whole volume for each cell"""
    polygons, cells, layers = [], [], []
    masks = masks.swapaxes(-1, -2)
    for i in np.unique(masks)[1:]:
        mask = masks == i
        for l in np.where(mask.any(axis=(1, 2)))[0]:
            padded_mask = np.pad(mask[l], 1, mode="constant")
            contours = [c - 1 for c in ski.measure.find_contours(padded_mask, 0.5)]
            if len(contours[0]) > 3:
                polygons.append(shp.geometry.Polygon(contours[0]).simplify(tolerance=tolerance, preserve_topology=True))
                cells.append(i)
                layers.append(l)
    polygons = gpd.GeoDataFrame({"cell": cells, "geometry": polygons}, crs=None)
    polygons["z"] = layers
    return polygons


def _synthetic_masks():
    """Cells on the borders, split into pieces, spanning few layers, and with gaps in the labels"""
    masks = np.zeros((4, 60, 50), dtype=np.uint16)
    masks[:, :10, :10] = 1
    masks[1:3, 20:30, 40:50] = 3
    masks[2, 50:60, 0:8] = 7
    masks[0, 40:45, 20:25] = 7
    masks[3, 5:25, 15:35] = 9
    masks[3, 10:15, 20:30] = 0
    return masks


@pytest.mark.parametrize("source", ["fixture", "synthetic"])
def test_masks_to_polygons_matches_full_scan(masks, source):
    if source == "synthetic":
        masks = _synthetic_masks()
    expected = _masks_to_polygons_full_scan(masks)
    new_polygons = ft.seg.masks_to_polygons(masks)
    assert_geodataframe_equal(new_polygons, expected, check_less_precise=False)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
