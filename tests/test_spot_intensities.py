import numpy as np
import pytest

import fishtank as ft


def test_spot_intensities_3d(img):
    intensities = ft.decode.spot_intensities(img, x=[10, 20], y=[10, 20], z=[0, 1], radius=5)
    assert intensities.shape == (2, 7)
    assert intensities[0, 0] == 22401
    assert intensities.dtype == np.float64


def test_spot_intensities_2d(img):
    intensities = ft.decode.spot_intensities(img[:, 0], x=[10, 20], y=[10, 20], radius=5)
    assert intensities.shape == (2, 7)
    assert intensities[0, 0] == 22401
    assert intensities.dtype == np.float64


if __name__ == "__main__":
    pytest.main(["-v", __file__])
