import pytest

import fishtank as ft


def test_imshow(img):
    ax = ft.pl.imshow(img[:3, 2], colors=["magenta", "green", "blue"])
    assert ax is not None


def test_imshow_percentile(img):
    ax = ft.pl.imshow(img[:3, 2], colors=["magenta", "green", "blue"], vmax=["p98", "p98", "p99"])
    assert ax is not None


def test_imshow_bad(img):
    with pytest.raises(ValueError):
        ft.pl.imshow(img[:3, 2], colors=["magenta", "green"])
    with pytest.raises(IndexError):
        ft.pl.imshow(img[:3, 2], colors=["magenta", "green", "blue"], vmax=["p98", "p98"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
