import pandas as pd
import pytest

import fishtank as ft


def test_polygon_properties_3d(polygons):
    properties = ft.seg.polygon_properties(polygons)
    assert isinstance(properties, pd.DataFrame)
    assert set(polygons.cell) == set(properties.cell)
    assert properties.loc[properties.cell == 169, "volume"].values[0] == pytest.approx(111.599, abs=1e-3)
    assert properties.loc[properties.cell == 169, "n_layers"].values[0] == 3
    assert properties.loc[properties.cell == 169, "centroid_y"].values[0] == pytest.approx(286.0564, abs=1e-3)
    assert properties.loc[properties.cell == 169, "centroid_z"].values[0] == pytest.approx(-4.80967, abs=1e-3)


def test_polygon_properties_2d(polygons):
    polygons = polygons.query("z == 1").copy()
    properties = ft.seg.polygon_properties(polygons, z=None)
    assert isinstance(properties, pd.DataFrame)
    assert set(polygons.cell) == set(properties.cell)
    assert properties.loc[properties.cell == 178, "area"].values[0] == pytest.approx(2743.5)
    assert properties.loc[properties.cell == 178, "centroid_y"].values[0] == pytest.approx(261.365, abs=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
