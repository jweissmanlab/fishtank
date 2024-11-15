import logging

import pytest

import fishtank as ft


@pytest.mark.slow
def test_align_experiments(caplog):
    parser = ft.scripts.align_experiments.get_parser()
    args = parser.parse_args(
        [
            "--ref",
            "./tests/data/merfish",
            "--moving",
            "./tests/data/shifted_merfish",
            "--ref_series",
            "H0R1",
            "--moving_series",
            "H0M1",
            "--output",
            "./tests/output/alignment.json",
        ]
    )
    with caplog.at_level(logging.INFO):
        ft.scripts.align_experiments.main(args)
    assert "Reference z-slice: 4" in caplog.text
    assert "Coarse shift: [  0 -12]" in caplog.text
    assert "Calculating mean shift for each tile" in caplog.text
    assert "Created 2 records" in caplog.text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
