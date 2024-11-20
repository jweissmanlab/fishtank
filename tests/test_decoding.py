import pytest

import fishtank as ft


@pytest.fixture()
def normalized_spots(spots, spot_channels):
    channels = spot_channels.query("bit.isin(@spots.columns)")
    spots[channels.bit] = ft.correct.color_normalization(spots[channels.bit], channels.color)
    yield spots


def test_expectation_maximization(normalized_spots, codebook):
    decoded, bit_performance = ft.decode.expectation_maximization(normalized_spots, codebook, max_dist=1.7)
    assert decoded.shape[0] == normalized_spots.shape[0]
    assert len(decoded.query("dist < 1.7")) == 23
    assert bit_performance.shape[0] == codebook.shape[1]


def test_logistic_regression(normalized_spots, weights):
    decoded, bit_performance = ft.decode.logistic_regression(normalized_spots, weights)
    assert decoded.shape[0] == normalized_spots.shape[0]
    assert len(decoded.query("prob > .5")) == 25
    assert bit_performance.shape[0] == weights.shape[1] - 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
