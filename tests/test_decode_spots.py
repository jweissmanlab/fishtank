import logging

import pytest

import fishtank as ft


@pytest.mark.slow
def test_assign_spots_script(caplog):
    parser = ft.scripts.decode_spots_script.get_parser()
    args = parser.parse_args(
        [
            "-i",
            "./tests/data/spots",
            "-s",
            "./tests/data/decoding/strategy.csv",
            "-o",
            "./tests/output/decoded_spots.csv",
            "--max_dist",
            "1.7",
        ]
    )
    with caplog.at_level(logging.INFO):
        kwargs = vars(args)
        kwargs.pop("func")
        ft.scripts.decode_spots(**kwargs)
    assert "Normalizing spot intensities by color" in caplog.text
    assert "Using EM to decode barcode" in caplog.text
    assert "Iteration 9: 17.71% of spots with distance < 1.7" in caplog.text
    assert "Using logistic regression to decode state" in caplog.text
    assert "Saving decoded spots to tests/output/decoded_spots.csv" in caplog.text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
