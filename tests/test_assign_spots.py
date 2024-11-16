import pandas as pd
import pytest

import fishtank as ft


def test_assign_spots_3d(polygons, spots):
    assigned = ft.seg.assign_spots(spots, polygons, max_dist=5, x="x", y="y", z="z")
    assert isinstance(assigned, pd.DataFrame)
    assert len(assigned) == len(spots)
    assert assigned.cell.nunique() == 7
    assert assigned.cell.notna().sum() == 40
    assert assigned.cell_dist.max() <= 5


def test_assign_spots_2d(polygons, spots):
    polygons = polygons.query("z == 5").copy()
    assigned = ft.seg.assign_spots(spots, polygons, max_dist=5, x="x", y="y")
    assert isinstance(assigned, pd.DataFrame)
    assert len(assigned) == len(spots)
    assert assigned.cell.nunique() == 6
    assert assigned.cell.notna().sum() == 36
    assert assigned.cell_dist.max() <= 5


if __name__ == "__main__":
    pytest.main(["-v", __file__])
