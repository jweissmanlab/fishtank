import logging

import pytest

import fishtank as ft


#@pytest.mark.slow
def test_cellpose_2d(caplog):
    parser = ft.scripts.cellpose_script.get_parser()
    args = parser.parse_args(
        [
            "--input",
            "./tests/data/merfish",
            "--output",
            "./tests/output/cellpose_2d",
            "--fov",
            "1",
            "--model",
            "cpsam",
            "--min_size",
            "100",
            "--z_slices",
            "3",
            "--filter",
            "unsharp_mask",
            "--channels",
            "DAPI",
            "--corrections",
            "./tests/data/corrections",
        ]
    )
    with caplog.at_level(logging.INFO):
        kwargs = vars(args)
        kwargs.pop("func")
        ft.scripts.cellpose(**kwargs)
    assert "Correcting illumination with tests/data/corrections" in caplog.text
    assert "Saving polygons to tests/output/cellpose_2d" in caplog.text


@pytest.mark.slow
def test_cellpose_3d(caplog):
    parser = ft.scripts.cellpose_script.get_parser()
    args = parser.parse_args(
        [
            "--input",
            "./tests/data/merfish",
            "--output",
            "./tests/output/cellpose_3d",
            "--fov",
            "1",
            "--model",
            "spsam",
            "--diameter",
            "20",
            "--downsample",
            "4",
            "--min_size",
            "500",
            "--filter",
            "deconwolf",
            "--channels",
            "DAPI",
            "--do_3D",
            "True",
        ]
    )
    with caplog.at_level(logging.INFO):
        kwargs = vars(args)
        kwargs.pop("func")
        ft.scripts.cellpose(**kwargs)
    assert "Applying deconwolf filter from fishtank" in caplog.text
    assert "Downsampling image by a factor of 4" in caplog.text
    assert "Saving polygons to tests/output/cellpose_3d" in caplog.text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
