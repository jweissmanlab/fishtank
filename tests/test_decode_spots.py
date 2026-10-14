import logging
import shutil

import pandas as pd
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


def _spots_copy(tmp_path):
    spots_dir = tmp_path / "spots"
    shutil.copytree("./tests/data/spots", spots_dir)
    return spots_dir, sorted(spots_dir.glob("spots_*.csv"))[0]


def _decode(input, output):
    parser = ft.scripts.decode_spots_script.get_parser()
    args = parser.parse_args(["-i", str(input), "-s", "./tests/data/decoding/strategy.csv", "-o", str(output)])
    kwargs = vars(args)
    kwargs.pop("func")
    ft.scripts.decode_spots(**kwargs)


@pytest.mark.slow
def test_decode_spots_drops_nan(tmp_path, caplog):
    spots_dir, file = _spots_copy(tmp_path)
    spots = pd.read_csv(file)
    spots.loc[0, "r1"] = float("nan")
    spots.to_csv(file, index=False)
    with caplog.at_level(logging.INFO):
        _decode(spots_dir, tmp_path / "decoded.csv")
    assert "Dropped 1 spots with missing intensities" in caplog.text
    decoded = pd.read_csv(tmp_path / "decoded.csv")
    assert decoded["barcode_dist"].notna().all()


@pytest.mark.slow
def test_decode_spots_missing_bit(tmp_path):
    spots_dir, file = _spots_copy(tmp_path)
    for f in spots_dir.glob("spots_*.csv"):
        pd.read_csv(f).drop(columns="r1").to_csv(f, index=False)
    with pytest.raises(ValueError, match="r1"):
        _decode(spots_dir, tmp_path / "decoded.csv")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
