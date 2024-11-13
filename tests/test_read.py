import numpy as np
import pytest

import fishtank as ft


@pytest.fixture()
def xml_path(img_path):
    return img_path / "H0R1" / "Conv_zscan_01.xml"


@pytest.fixture()
def dax_path(img_path):
    return img_path / "H0R1" / "Conv_zscan_01.dax"


@pytest.fixture()
def color_usage_path(img_path):
    return img_path / "color_usage.csv"


def test_read_xml(xml_path):
    expected = {
        "objective": "obj1",
        "micron_per_pixel": 0.107,
        "flip_horizontal": False,
        "flip_vertical": True,
        "transpose": True,
        "x_pixels": 288,
        "y_pixels": 288,
        "stage_position": [400.1, -2494.35],
        "number_frames": 25,
        "z_offsets": [-15.0, -14.4, -13.8, -13.2, -12.6],
        "colors": [748, 637, 545, 477, 405],
    }
    attrs = ft.io.read_xml(xml_path, parse=True)
    for key in expected:
        assert attrs[key] == expected[key]


def test_list_fovs(img_path, polygons_path):
    fovs = ft.io.list_fovs(img_path)
    assert fovs == [0, 1]
    fovs = ft.io.list_fovs(polygons_path, file_pattern="polygons_{fov}.json")
    assert fovs == [0, 1]


def test_read_dax(dax_path):
    img = ft.io.read_dax(dax_path, shape=(288, 288))
    assert img.shape == (25, 288, 288)
    assert img.dtype == "uint16"
    with pytest.raises(ValueError):
        ft.io.read_dax(dax_path, shape=(100, 100))


def test_read_dax_frames(dax_path):
    img = ft.io.read_dax(dax_path, frames=[0, 5, 10], shape=(288, 288))
    assert img.shape == (3, 288, 288)
    img = ft.io.read_dax(dax_path, frames=5, shape=(288, 288))
    assert img.shape == (1, 288, 288)
    with pytest.raises(ValueError):
        img = ft.io.read_dax(dax_path, frames=[1, 100], shape=(288, 288))


def test_read_img(dax_path):
    img, attrs = ft.io.read_img(dax_path)
    assert img.shape == (5, 5, 288, 288)
    assert img.dtype == "uint16"
    img, attrs = ft.io.read_img(dax_path, z_slices=np.array([0, 1, 2]))
    assert img.shape == (5, 3, 288, 288)
    img, attrs = ft.io.read_img(dax_path, colors=[637, 545])
    assert img.shape == (2, 5, 288, 288)
    img, attrs = ft.io.read_img(dax_path, z_slices=1, colors=[637, 545])
    assert img.shape == (2, 288, 288)
    img, attrs = ft.io.read_img(dax_path, z_slices=[0, 1, 2], colors=405, z_project=True)
    assert img.shape == (288, 288)
    with pytest.raises(ValueError):
        img, attrs = ft.io.read_img(dax_path, z_slices=[0, 1, 2], colors="bad")
    with pytest.raises((ValueError, IndexError)):
        img, attrs = ft.io.read_img(dax_path, z_slices=100)


def test_read_color_usage(color_usage_path):
    channels = ft.io.read_color_usage(color_usage_path)
    assert channels.shape == (13, 3)
    assert channels["color"].nunique() == 5
    assert set(channels.columns) == {"color", "series", "bit"}


def test_read_fov(img_path, channels):
    # single series
    img, attrs = ft.io.read_fov(img_path, 1, "H0R1", file_pattern="{series}/Conv_zscan_{fov}.dax")
    assert img.shape == (5, 5, 288, 288)
    assert img.dtype == "uint16"
    assert attrs["objective"] == "obj1"
    assert attrs["colors"] == [748, 637, 545, 477, 405]
    # list of series
    img, attrs = ft.io.read_fov(img_path, 1, ["H0R1", "H1R2"], file_pattern="{series}/Conv_zscan_{fov}.dax")
    assert img.shape == (9, 5, 288, 288)
    assert attrs["colors"] == [748, 637, 545, 477, 405, 748, 637, 545, 477]
    # channels
    img, attrs = ft.io.read_fov(
        img_path,
        1,
        channels=channels.query("color == 748"),
        file_pattern="{series}/Conv_zscan_{fov}.dax",
        z_project=True,
    )
    assert img.shape == (3, 288, 288)
    assert attrs["colors"] == [748, 748, 748]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
