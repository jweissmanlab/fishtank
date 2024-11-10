import pytest

import fishtank as ft


def test_correct_illumination(img, channels, corr_path):
    corrected = ft.correct.illumination(img, colors=channels["color"], corr_path=corr_path)
    assert corrected.shape == img.shape
    assert corrected.dtype == img.dtype
    assert corrected.max() == 65313


def test_correct_illumination_shapes(img, corr_path):
    corrected = ft.correct.illumination(img[3], colors=[405], corr_path=corr_path)
    assert corrected.shape == img[3].shape
    corrected = ft.correct.illumination(img[3, 0], colors=405, corr_path=corr_path)
    assert corrected.shape == img[3, 0].shape


if __name__ == "__main__":
    pytest.main(["-v", __file__])
