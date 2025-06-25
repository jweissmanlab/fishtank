import os
import pathlib
import re
from xml.etree import ElementTree as xml

import numpy as np
import pandas as pd
import skimage as ski

from fishtank.utils import create_mosaic, determine_fov_format


def _xml_to_dict(element):
    """Convert xml element to dictionary."""
    if len(element) == 0:
        return element.text
    return {child.tag: _xml_to_dict(child) for child in element}


def read_xml(path: str | pathlib.Path, parse: bool = True) -> dict:
    """Read MERFISH formatted xml file.

    Parameters
    ----------
    path
        Path to xml file.
    parse
        If True, parse relevant fields.

    Returns
    -------
    attrs
        a dictionary of image attributes.
    """
    tree = _xml_to_dict(xml.parse(path).getroot())
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
    shutters_str = tree["illumination"]["shutters"]
    if "shutter_" in  shutters_str:
        colors_str = re.search(r"shutter_([\d_]+)_s", shutters_str).group(1)
        attrs["colors"] = list(map(int, colors_str.split("_")))
    elif "f21" in shutters_str:
        color_str = re.search(r'(\d+f21(?:_\d+f21)*_)', shutters_str).group(1)
        attrs["colors"] = list(map(int, color_str.split("f21_")[:-1]))
    else:
        raise ValueError(f"Cannot parse colors from shutter string: {shutters_str}")
    return attrs


def list_fovs(path: str | pathlib.Path, file_pattern: str = "{series}/Conv_zscan_{fov}.dax") -> list:
    """List FOV files.

    Parameters
    ----------
    path
        Directory containing files.
    file_pattern
        Name pattern of files.

    Returns
    -------
    fovs
        a list of field of view numbers.
    """
    path = pathlib.Path(path)
    fovs = set()

    # List possible series
    if "{series}" in file_pattern:
        series_list = [d.name for d in path.iterdir() if d.is_dir()]
    else:
        series_list = [""]

    for series in series_list:
        fov_pattern = file_pattern.format(series=series, fov="*")
        prefix, suffix = fov_pattern.split("*")
        fov_files = list(path.glob(fov_pattern))
        for f in fov_files:
            fovs.add(int(str(f.relative_to(path))[len(prefix) : -len(suffix)]))
        if len(fovs) > 0:
            break

    return sorted(fovs)


def read_dax(
    path: str | pathlib.Path, frames: int | list = None, shape: tuple = (2304, 2304), dtype: str = "uint16"
) -> np.ndarray:
    """Read a MERFISH formatted dax file.

    Parameters
    ----------
    path
        Path to dax file.
    frames
        Indices of frames to read.
    shape
        Shape of a single frame.
    dtype
        Data type of the image.

    Returns
    -------
    img
        a numpy array with shape (F, Y, X) where F is the number of frames.
    """
    if frames is None:
        img = np.fromfile(path, dtype=dtype, count=-1)
        n_frames = len(img) // np.prod(shape)
        if len(img) % np.prod(shape) != 0:
            raise ValueError(f"Image size {len(img)} is not divisible by frame size {np.prod(shape)}")
    else:
        memmap = np.memmap(path, dtype=dtype, mode="r")
        if isinstance(frames, int):
            frames = [frames]
        try:
            img = np.array([memmap[frame * np.prod(shape) : (frame + 1) * np.prod(shape)] for frame in frames])
        except ValueError:
            raise ValueError(f"Frame indices {frames} are out of bounds")  # noqa: B904
        n_frames = len(frames)
    return img.reshape(n_frames, *shape)


def read_img(
    path: str | pathlib.Path,
    colors: int | str | list = None,
    z_slices: int | list = None,
    z_project: bool = False,
    shape: tuple = None,
    color_order: list = None,
    plugin: str = None,
    **plugin_args,
) -> tuple[np.ndarray, dict]:
    """Read image file.

    Parameters
    ----------
    path
        Image file path.
    colors
        List of color indices to read.
    z_slices
        List of z-slice indices to read.
    z_project
        If True, z-project the image.
    shape
        Shape of a single frame.
    color_order
        Order of colors in the image.
    plugin
        Name of skimage plugin used to load image if not dax format.
    plugin_args
        Passed to the given plugin

    Returns
    -------
    img
        a numpy array, such that a gray image is (Y,X), a 3D gray image is (Z,Y,X), and 3D multi-channel image is (C,Z,Y,X).
    attrs
        a dictionary of image attributes.
    """
    # Setup
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    frames = None
    z_max = None
    n_colors = 1
    # Attempt to load attributes
    if os.path.exists(path.with_suffix(".xml")):
        attrs = read_xml(path.with_suffix(".xml"))
        if "x_pixels" in attrs.keys() and "y_pixels" in attrs.keys():
            shape = (attrs["x_pixels"], attrs["y_pixels"])
        if "z_offsets" in attrs.keys():
            z_max = len(attrs["z_offsets"])
        if "colors" in attrs.keys():
            color_order = np.array(attrs["colors"])
            n_colors = len(attrs["colors"])
    else:
        attrs = {}
    # Process z-slice selection
    if z_slices is not None:
        if z_max is None:
            raise ValueError("Cannot select z-slices without xml metadata file")
        if isinstance(z_slices, int):
            z_slices = [z_slices]
        z_slices = np.array(z_slices)
        attrs["z_offsets"] = np.array(attrs["z_offsets"])[z_slices].astype(float).tolist()
    # Process color selection
    if colors is not None:
        if color_order is None:
            raise ValueError("Cannot select colors without xml file or color_order specified")
        if isinstance(colors, int) or isinstance(colors, str):
            colors = [colors]
        attrs["colors"] = colors
        colors = np.array(colors).astype(color_order.dtype)
        if not np.all(np.isin(colors, color_order)):
            missing = np.setdiff1d(colors, color_order)
            raise ValueError(f"Color {missing} not found in image colors {color_order}")
        color_slices = np.array([np.where(color_order == c)[0][0] for c in colors])
        n_colors = len(colors)
    # Get frames
    if z_slices is not None or colors is not None:
        frames = np.arange(z_max * len(color_order)).reshape(z_max, len(color_order))
        if z_slices is not None:
            frames = np.take(frames, z_slices, axis=0)
        if colors is not None:
            frames = np.take(frames, color_slices, axis=1)
        frames = frames.flatten()
    # Load image
    if suffix == ".dax":
        img = read_dax(path, shape=shape, frames=frames, **plugin_args)
    else:
        img = ski.io.imread(path, plugin=plugin, **plugin_args)[frames]
    # Reshape image if necessary
    if n_colors > 1:
        img = np.reshape(img, (n_colors, img.shape[0] // n_colors, *img.shape[1:]), order="F")
    # Apply transpose and flip operations
    if attrs["transpose"]:
        img = img.swapaxes(-1, -2)
    if attrs["flip_horizontal"]:
        img = np.flip(img, axis=-1)
    if attrs["flip_vertical"]:
        img = np.flip(img, axis=-2)
    # Z-project image if necessary
    if z_project:
        img = img.max(axis=-3)
    return img.squeeze(), attrs


def read_color_usage(path: str | pathlib.Path) -> pd.DataFrame:
    """Read MERFISH color usage file.

    Parameters
    ----------
    path
        Path to color usage csv file.

    Returns
    -------
    channels
        a DataFrame with columns "series", "color", and "bit" for each channel.
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


def read_fov(
    path: str | pathlib.Path,
    fov: int | str,
    series: int | str | list = None,
    colors: int | str | list = None,
    channels: pd.DataFrame = None,
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    z_slices: list = None,
    z_project: bool = False,
    ref_series: int | str = None,
) -> tuple[np.ndarray, dict]:
    """Read FOV from MERFISH experiment.

    Parameters
    ----------
    path
        Path to image files.
    fov
        Field of view number.
    series
        List of series names to read.
    colors
        List of color indices to read.
    channels
        DataFrame with "series" and "color" columns specifying the channels to read.
    file_pattern
        Pattern for image files.
    z_slices
        List of z-slice indices to read.
    z_project
        If True, z-project the image.

    Returns
    -------
    img
        a numpy array, such that a gray image is (Y,X), a 3D gray image is (Z,Y,X), and 3D multi-channel image is (C,Z,Y,X).
    attrs
        a dictionary of image attributes.
    """
    imgs = []
    attrs = []
    path = pathlib.Path(path)
    # Load images given list of series
    if series is not None:
        if isinstance(series, int) or isinstance(series, str):
            series = [series]
        file_pattern = determine_fov_format(path, fov=fov, series=series[0], file_pattern=file_pattern)
        for s in series:
            img, attr = read_img(
                path / file_pattern.format(series=s, fov=fov), z_slices=z_slices, z_project=z_project, colors=colors
            )
            imgs.append(img)
            attrs.append(attr)
    # Load images given channels df
    elif channels is not None:
        if colors is not None:
            channels = channels.query("color in @colors")
        file_pattern = determine_fov_format(
            path, fov=fov, series=channels["series"].values[0], file_pattern=file_pattern
        )
        for s, s_channels in channels.groupby("series", sort=False):
            img, attr = read_img(
                path / file_pattern.format(series=s, fov=fov),
                colors=s_channels["color"].values,
                z_slices=z_slices,
                z_project=z_project,
            )
            imgs.append(img)
            attrs.append(attr)
    # Reshape images
    if len(imgs) > 1:
        if len(imgs[0].shape) == 2:
            imgs = np.stack(imgs, axis=0)
        else:
            imgs = np.concatenate(imgs, axis=0)
    else:
        imgs = imgs[0]
    # Update attributes
    if "colors" in attrs[0].keys():
        colors = [color for attr in attrs for color in attr["colors"]]
    else:
        colors = []
    if ref_series is not None:
        attrs = attrs[np.where(series == ref_series)[0][0]]
    else:
        attrs = attrs[0]
    attrs["colors"] = colors
    return imgs, attrs


def read_mosaic(
    path: str | pathlib.Path,
    series: str,
    colors: int | str | list = None,
    fovs: list = None,
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    z_slices: int | list = None,
    z_project: bool = False,
    downsample: int | bool = 4,
    filter: callable = None,
    filter_args: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read FOV images as a mosaic.

    Parameters
    ----------
    path
        Path to image files.
    series
        Name of a series to read
    colors
        List of color indices to read.
    fovs
        List of field of view numbers to read.
    file_pattern
        Naming pattern for image files.
    z_slices
        List of z-slice indices to read.
    z_project
        If True, z-project the image.
    downsample
        Factor to downsample the image by. If False, no downsampling is performed.
    filter
        Function to filter the images with.
    filter_args
        Arguments to pass to the filter function.

    Returns
    -------
    mosaic
        a numpy array image with all FOVs stitched together.
    bounds
        the total bounds in microns (x_min, y_min, x_max, y_max).
    """
    # Setup
    if filter_args is None:
        filter_args = {}
    if fovs is None:
        fovs = list_fovs(path, file_pattern=file_pattern)
    if isinstance(z_slices, int):
        z_slices = [z_slices]
    if colors is None and len(z_slices) > 1:
        raise ValueError("Must specify a single color when using multiple z slices")
    if downsample is False:
        downsample = 1
    if downsample is True:
        downsample = 4
    # Read images
    imgs = []
    positions = []
    for fov in fovs:
        img, attrs = read_fov(path, series=series, fov=fov, colors=colors, z_slices=z_slices, z_project=z_project)
        if downsample > 1:
            if len(img.shape) == 2:
                img = img[::downsample, ::downsample]
            elif len(img.shape) == 3:
                img = img[:, ::downsample, ::downsample]
        if filter is not None:
            img = filter(img, **filter_args)
        imgs.append(img)
        positions.append(attrs["stage_position"])
        microns_per_pixel = attrs["micron_per_pixel"] * downsample
    # Create mosaic
    mosaic, bounds = create_mosaic(imgs, positions, microns_per_pixel)
    return mosaic, bounds
