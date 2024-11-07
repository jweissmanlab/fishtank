import numpy as np

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
