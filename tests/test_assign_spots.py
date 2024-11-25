import logging

import pandas as pd
import pytest

import fishtank as ft


def test_assign_spots_3d(polygons, spots):
    assigned = ft.seg.assign_spots(spots, polygons, max_dist=5, x="x", y="y", z="z")
    assert isinstance(assigned, pd.DataFrame)
    assert len(assigned) == len(spots)
    assert assigned.cell.nunique() == 9
    assert assigned.cell.notna().sum() == 41
    assert assigned.cell_dist.max() <= 5


def test_assign_spots_2d(polygons, spots):
    polygons = polygons.query("z == 5").copy()
    assigned = ft.seg.assign_spots(spots, polygons, max_dist=5, x="x", y="y")
    assert isinstance(assigned, pd.DataFrame)
    assert len(assigned) == len(spots)
    assert assigned.cell.nunique() == 6
    assert assigned.cell.notna().sum() == 30
    assert assigned.cell_dist.max() <= 5


@pytest.mark.slow
def test_assign_spots_script(caplog):
    parser = ft.scripts.assign_spots.get_parser()
    args = parser.parse_args(
        [
            "-i",
            "./tests/data/spots/spots_0.csv",
            "-p",
            "./tests/data/polygons/polygons_0.json",
            "-o",
            "./tests/output/assigned_spots.csv",
            "--max_dist",
            "5",
            "--x_column",
            "x",
            "--y_column",
            "y",
            "--z_column",
            "z",
        ]
    )
    with caplog.at_level(logging.INFO):
        ft.scripts.assign_spots.main(args)
    assert "10 unique z values in spots range from 0 to 9" in caplog.text
    assert "Splitting polygons and spots into tiles" in caplog.text
    assert "Saving spot assignments" in caplog.text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
