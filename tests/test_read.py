import numpy as np
import pandas as pd
import pytest

import fishtank as ft


@pytest.fixture()
def xml_path(img_path):
    return img_path / "H0R1" / "Conv_zscan_00.xml"


@pytest.fixture()
def dax_path(img_path):
    return img_path / "H0R1" / "Conv_zscan_00.dax"


@pytest.fixture()
def color_usage_path(img_path):
    return img_path / "color_usage.csv"


@pytest.fixture()
def frame_table_path(img_path):
    return img_path / "H0R1" / "frame_table.csv"


@pytest.fixture()
def frame_table_ragged_path(img_path):
    return img_path / "H0R1" / "frame_table_ragged.csv"



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
    img, attrs = ft.io.read_img(dax_path, z_slices=np.array([0, 2]))
    assert img.shape == (5, 2, 288, 288)
    assert attrs["z_offsets"] == [-15.0, -13.8]
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


def test_read_img_frame_table(dax_path, frame_table_path):
    # Frame table should give identical result to XML-based read
    img_xml, attrs_xml = ft.io.read_img(dax_path)
    img_ft, attrs_ft = ft.io.read_img(dax_path, frames=frame_table_path)
    assert img_ft.shape == (5, 5, 288, 288)
    assert np.array_equal(img_ft, img_xml)
    assert attrs_ft["z_offsets"] == attrs_xml["z_offsets"]
    assert attrs_ft["colors"] == attrs_xml["colors"]
    # Color selection
    img_ft, attrs = ft.io.read_img(dax_path, frames=frame_table_path, colors=[748, 405])
    assert img_ft.shape == (2, 5, 288, 288)
    # z-slice selection
    img_ft, attrs = ft.io.read_img(dax_path, frames=frame_table_path, z_slices=[0, 2])
    assert img_ft.shape == (5, 2, 288, 288)
    assert attrs["z_offsets"] == [-15.0, -13.8]
    # Frame count mismatch raises an error
    wrong_frame_table = frame_table_path.parent / "frame_table_wrong.csv"
    wrong_frame_table.write_text(",color,z\n0,748,-15.0\n1,637,-15.0\n")
    with pytest.raises(ValueError, match="Frame table has 2 rows but image has 25 frames"):
        ft.io.read_img(dax_path, frames=wrong_frame_table)
    wrong_frame_table.unlink()


def test_read_img_frame_table_inference(dax_path, frame_table_ragged_path):
    # Ragged table: 748 at z_index 0,1; 405 at z_index 0 only
    # colors specified → z_slices inferred (only z where ALL colors have frames)
    img, attrs = ft.io.read_img(dax_path, frames=frame_table_ragged_path, colors=[748, 405])
    assert img.shape == (2, 288, 288)  # both colors, 1 common z-plane (z_index 0)
    assert attrs["z_offsets"] == [-15.0]
    # single color → gets its own z range
    img, attrs = ft.io.read_img(dax_path, frames=frame_table_ragged_path, colors=[748])
    assert img.shape == (2, 288, 288)  # 1 color (squeezed), 2 z-planes
    assert attrs["z_offsets"] == [-15.0, -14.4]
    # z_slices specified → colors inferred (only colors present at ALL z_slices)
    img, attrs = ft.io.read_img(dax_path, frames=frame_table_ragged_path, z_slices=[0])
    assert img.shape == (2, 288, 288)  # 2 colors at z_index 0, 1 z-plane (squeezed)
    assert set(attrs["colors"]) == {748, 405}
    img, attrs = ft.io.read_img(dax_path, frames=frame_table_ragged_path, z_slices=[1])
    assert img.shape == (288, 288)  # only 748 at z_index 1, both axes squeezed
    assert attrs["colors"] == [748]
    # neither specified → error (ragged)
    with pytest.raises(ValueError, match="ragged"):
        ft.io.read_img(dax_path, frames=frame_table_ragged_path)
    # explicit invalid combination still errors
    with pytest.raises(ValueError, match="color/z combinations that do not exist"):
        ft.io.read_img(dax_path, frames=frame_table_ragged_path, colors=[748, 405], z_slices=[0, 1])


def test_read_color_usage(color_usage_path, img_path):
    channels = ft.io.read_color_usage(color_usage_path)
    assert channels.shape == (13, 3)
    assert channels["color"].nunique() == 5
    assert set(channels.columns) == {"color", "series", "bit"}
    # with frames column
    channels_ft = ft.io.read_color_usage(img_path / "color_usage_frames.csv")
    assert set(channels_ft.columns) == {"color", "series", "bit", "frames"}
    assert channels_ft.loc[channels_ft["series"] == "H0R1", "frames"].iloc[0] == "tests/data/merfish/H0R1/frame_table.csv"
    assert pd.isna(channels_ft.loc[channels_ft["series"] == "H1R2", "frames"].iloc[0])


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
    # list of series and colors
    img, attrs = ft.io.read_fov(
        img_path, 1, ["H0R1", "H1R2"], file_pattern="{series}/Conv_zscan_{fov}.dax", colors=[748, 477]
    )
    assert img.shape == (4, 5, 288, 288)
    assert attrs["colors"] == [748, 477, 748, 477]
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
    # channels with frames column: result should match reading without frame table
    channels_ft = ft.io.read_color_usage(img_path / "color_usage_frames.csv")
    img_ft, attrs_ft = ft.io.read_fov(
        img_path, 1, channels=channels_ft.query("series == 'H0R1'"), file_pattern="{series}/Conv_zscan_{fov}.dax"
    )
    img_noft, _ = ft.io.read_fov(img_path, 1, "H0R1", file_pattern="{series}/Conv_zscan_{fov}.dax")
    assert img_ft.shape == img_noft.shape
    assert np.array_equal(img_ft, img_noft)


def test_read_mosaic(img_path):
    mosaic, bounds = ft.io.read_mosaic(img_path, series="H0R1", colors=405, z_slices=[3], downsample=False)
    assert mosaic.shape == (288, 488)
    assert bounds[0] == pytest.approx(400, abs=1)
    assert bounds[2] == pytest.approx(452, abs=1)
    mosaic, bounds = ft.io.read_mosaic(img_path, series="H0R1", colors=405, z_slices=[3], downsample=4)
    assert mosaic.shape == (72, 122)
    assert bounds[0] == pytest.approx(400, abs=1)
    assert bounds[2] == pytest.approx(452, abs=1)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
