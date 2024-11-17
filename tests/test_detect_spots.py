import logging

import pytest

import fishtank as ft


@pytest.mark.slow
def test_detect_spots_3d(caplog):
    parser = ft.scripts.detect_spots.get_parser()
    args = parser.parse_args(
        [
            "-i",
            "./tests/data/merfish",
            "-o",
            "./tests/output/spots",
            "--fov",
            "0",
            "--ref_series",
            "H0R1",
            "--common_bits",
            "r52,r52",
            "--filter",
            "unsharp_mask",
            "--filter_args",
            "sigma=10",
        ]
    )
    with caplog.at_level(logging.INFO):
        ft.scripts.detect_spots.main(args)
    assert "Detecting spots with threshold 2310.0" in caplog.text
    assert "Detected 46 spots" in caplog.text
    assert "Series drift: [0 0]" in caplog.text
    assert "Saving results in tests/output/spots" in caplog.text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
