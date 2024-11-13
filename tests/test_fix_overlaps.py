import pytest

import fishtank as ft


def test_fix_overlaps_3d(overlapping_polygons):
    polygons = ft.seg.fix_overlaps(overlapping_polygons, diameter=80)
    assert polygons.cell.nunique() == 27


def test_fix_overlaps_2d(overlapping_polygons):
    overlapping_polygons = overlapping_polygons.dissolve("cell").reset_index().drop(columns=["global_z", "z"])
    polygons = ft.seg.fix_overlaps(overlapping_polygons, z=None, diameter=80)
    assert polygons.cell.nunique() == 26


if __name__ == "__main__":
    pytest.main(["-v", __file__])
