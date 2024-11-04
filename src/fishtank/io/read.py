import os
import pathlib
import re
import xml

import numpy as np
import pandas as pd
import skimage as ski


def _xml_to_dict(element):
    """Convert xml element to dictionary."""
    if len(element) == 0:
        return element.text
    return {child.tag: _xml_to_dict(child) for child in element}


def read_xml(path: str | pathlib.Path, parse: bool = True) -> dict:
    """Read and parse MERFISH formatted xml file.

    Parameters
    ----------
    path
        Path to xml file.
    parse
        If True, parse relevant fields.

    Returns
    -------
    attrs
        Dictionary of image attributes.
    """
    tree = _xml_to_dict(xml.etree.ElementTree.parse(path).getroot())
    if parse is False:
        return tree
    attrs = {}
    attrs["objective"] = tree["mosaic"]["objective"]
    attrs["micron_per_pixel"] = float(tree["mosaic"][attrs["objective"]].split(",")[1])
    for key in ["flip_horizontal", "flip_vertical", "transpose"]:
        attrs[key] = tree["mosaic"][key] == "True"
    attrs["x_pixels"] = int(tree["camera1"]["x_pixels"])
    attrs["y_pixels"] = int(tree["camera1"]["y_pixels"])
    attrs["stage_position"] = [float(i) for i in tree["acquisition"]["stage_position"].split(",")]
    attrs["number_frames"] = int(tree["acquisition"]["number_frames"])
    z_offsets = []
    for z_offset in tree["focuslock"]["hardware_z_scan"]["z_offsets"].split(","):
        if float(z_offset) not in z_offsets:
            z_offsets.append(float(z_offset))
    attrs["z_offsets"] = z_offsets
    colors_str = re.search(r"shutter_([\d_]+)_s", tree["illumination"]["shutters"]).group(1)
    attrs["colors"] = list(map(int, colors_str.split("_")))
    return attrs


def read_dax(path: str | pathlib.Path, shape=(2304, 2304), dtype="uint16") -> np.ndarray:
    """Read a MERFISH formatted dax file.

    Parameters
    ----------
    path
        Path to dax file.
    shape
        Shape of a single frame.

    Returns
    -------
    img
        Image as numpy array with shape (F, Y, X) where F is the number of frames.
    """
    img = np.fromfile(path, dtype=dtype, count=-1)
    n_frames = len(img) // np.prod(shape)
    if len(img) % np.prod(shape) != 0:
        raise ValueError(f"Image size {len(img)} is not divisible by frame size {np.prod(shape)}")
    return img.reshape(n_frames, *shape)


def read_img(path: str | pathlib.Path, plugin: str = None, **plugin_args) -> np.ndarray:
    """Read image file with paired attributes.

    Parameters
    ----------
    path
        Image file path.
    plugin
        Name of skimage plugin used to load image if not dax format.
    plugin_args
        Passed to the given plugin

    Returns
    -------
    img
        Image as numpy array, such that a gray image is (Y,X), a 3D gray image is (Z,Y,X), and multi-channel image is (C,Z,Y,X).
    attrs
        Dictionary of image attributes.
    """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    # Attempt to load attributes
    if os.path.exists(path.with_suffix(".xml")):
        attrs = read_xml(path.with_suffix(".xml"))
    else:
        attrs = {}
    # Load image
    if suffix == ".dax":
        if "x_pixels" in attrs.keys() and "y_pixels" in attrs.keys():
            shape = (attrs["x_pixels"], attrs["y_pixels"])
            img = read_dax(path, shape=shape, **plugin_args)
        else:
            img = read_dax(path)
    else:
        img = ski.io.imread(path, plugin=plugin, **plugin_args)
    # Reshape image if necessary
    if len(attrs["colors"]) > 1:
        z_slices = img.shape[0] // len(attrs["colors"])
        img = np.reshape(img, (len(attrs["colors"]), z_slices, *img.shape[1:]))
    # Apply transpose and flip operations
    if attrs["transpose"]:
        img = img.swapaxes(-1, -2)
    if attrs["flip_horizontal"]:
        img = np.flip(img, axis=-1)
    if attrs["flip_vertical"]:
        img = np.flip(img, axis=-2)
    return img, attrs


def read_color_usage(path: str | pathlib.Path) -> dict:
    """Read MERFISH color usage file.

    Parameters
    ----------
    path
        Path to color usage csv file.

    Returns
    -------
    channels
        DataFrame with columns "series", "color", and "bit" for each channel.
    """
    color_usage = pd.read_csv(path)
    color_usage = color_usage.rename(columns={color_usage.columns[0]: "series"})
    channels = color_usage.melt(id_vars="series", var_name="color", value_name="bit")
    channels["color_order"] = channels["color"].factorize()[0]
    channels["order"] = channels["series"].factorize()[0]
    channels = channels.sort_values(["order", "color_order"]).drop(columns=["color_order", "order"])
    channels = channels[~channels.bit.isna()].reset_index(drop=True)
    channels["color"] = channels["color"].astype(int)
    return channels


def _determine_fov_format(path, fov, series, file_pattern):
    """Determine the fov field format."""
    for format in ["{fov:01d}", "{fov:02d}", "{fov:03d}"]:
        if os.path.exists(path / file_pattern.replace("{fov}", format).format(fov=fov, series=series)):
            return file_pattern.replace("{fov}", format)
    raise ValueError(f"Could not find file matching {file_pattern.format(fov=fov,series=series)} in {path}")


def read_fov(
    path: str | pathlib.Path,
    fov: int | str,
    series: int | str | list = None,
    channels: pd.DataFrame = None,
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    z_project: bool = False,
    z_slices: list = None,
    ref_series: int | str = None,
):
    """Read a single field of view from a MERFISH dataset.

    Parameters
    ----------
    path
        Path to image files.
    fov
        Field of view number.
    series
        Series number.
    channels
        List of channel colors.
    file_pattern
        Pattern for image files.
    z_project
        If True, z-project the image.
    z_slices
        Slice indices to keep.

    Returns
    -------
    img
        Image as numpy array.
    attrs
        Dictionary of image attributes.
    """
    imgs = []
    attrs = []
    if series is not None:
        if isinstance(series, int) or isinstance(series, str):
            series = [series]
        file_pattern = _determine_fov_format(path, fov, series[0], file_pattern)
        for s in series:
            img, attr = read_img(path / file_pattern.format(series=s, fov=fov))
            imgs.append(img)
            attrs.append(attr)
    elif channels is not None:
        print("channels")
        print(channels)
        file_pattern = _determine_fov_format(path, fov, channels["series"].values[0], file_pattern)
        for s, s_channels in channels.groupby("series", sort=False):
            img, attr = read_img(path / file_pattern.format(series=s, fov=fov))
            imgs.append(img[np.isin(np.array(attr["colors"]), s_channels["color"])])
            attrs.append(attr)
    imgs = np.concatenate(imgs, axis=0)
    if ref_series is not None:
        attrs = attrs[np.where(series == ref_series)[0][0]]
    else:
        attrs = attrs[0]
    return imgs, attrs
