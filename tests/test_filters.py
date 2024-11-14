import pytest
import skimage as ski

import fishtank as ft


@pytest.mark.slow
def test_deconwolf(dapi_img):
    filtered = ft.filters.deconwolf(dapi_img, colors=405, z_step=0.6)
    assert ski.metrics.normalized_root_mse(dapi_img, filtered) > 0.3
    assert filtered.shape == dapi_img.shape
    assert filtered.dtype == dapi_img.dtype


def test_unsharp_mask(dapi_img):
    filtered = ft.filters.unsharp_mask(dapi_img, sigma=10)
    assert filtered.shape == dapi_img.shape
    assert filtered.dtype == dapi_img.dtype
    assert filtered.max() <= dapi_img.max()
    assert ski.metrics.normalized_root_mse(dapi_img, filtered) > 0.99


if __name__ == "__main__":
    pytest.main(["-v", __file__])
