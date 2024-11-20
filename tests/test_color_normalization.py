import pytest

import fishtank as ft


def test_color_normalization(spots, spot_channels):
    channels = spot_channels.query("bit.isin(@spots.columns)")
    normalized = ft.correct.color_normalization(spots[channels.bit], channels.color)
    color_means = []
    for color in channels.color.unique():
        color_means.append(normalized[:, channels.color == color].mean())
    assert color_means[0] == pytest.approx(color_means[1])
    assert color_means[1] == pytest.approx(color_means[2])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
