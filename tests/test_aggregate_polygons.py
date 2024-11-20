import logging

import pytest

import fishtank as ft


@pytest.mark.slow
def test_aggregate_polygons_3d(caplog):
    parser = ft.scripts.aggregate_polygons.get_parser()
    args = parser.parse_args(
        [
            "--input",
            "./tests/data/polygons",
            "--output",
            "./tests/output/polygons_3d.json",
            "--min_size",
            "2000",
            "--scale_factor",
            "1",
            "--min_ioa",
            ".2",
            "--z_column",
            "global_z",
        ]
    )
    with caplog.at_level(logging.INFO):
        ft.scripts.aggregate_polygons.main(args)
    assert "Loaded 33 polygons." in caplog.text
    assert "27 polygons after fixing overlaps." in caplog.text
    assert "17 polygons after removing polygons smaller than 2000.0." in caplog.text
    assert "Saving polygons to tests/output/polygons_3d.json" in caplog.text


@pytest.mark.slow
def test_aggregate_polygons_2d(caplog):
    parser = ft.scripts.aggregate_polygons.get_parser()
    args = parser.parse_args(
        [
            "--input",
            "./tests/data/polygons_2d",
            "--output",
            "./tests/output/polygons_2d.json",
            "--min_size",
            "500",
            "--scale_factor",
            "1",
            "--min_ioa",
            ".2",
        ]
    )
    with caplog.at_level(logging.INFO):
        ft.scripts.aggregate_polygons.main(args)
    assert "Loaded 20 polygons." in caplog.text
    assert "16 polygons after fixing overlaps." in caplog.text
    assert "14 polygons after removing polygons smaller than 500.0." in caplog.text
    assert "Saving polygons to tests/output/polygons_2d.json" in caplog.text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
