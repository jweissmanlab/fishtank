import pytest

import fishtank as ft


def test_create_mosaic(img_path):
    imgs = []
    positions = []
    for fov in [0, 1]:
        img, attrs = ft.io.read_fov(img_path, fov, "H0R1", colors=[405], z_slices=[3])
        imgs.append(img)
        positions.append(attrs["stage_position"])
    mosaic, bounds = ft.utils.create_mosaic(imgs, positions, micron_per_pixel=0.107)
    assert mosaic.shape == (288, 488)
    assert mosaic.dtype == img.dtype


if __name__ == "__main__":
    pytest.main(["-v", __file__])
