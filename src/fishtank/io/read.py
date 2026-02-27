import os
import pathlib
import re
from xml.etree import ElementTree as xml

import numpy as np
import pandas as pd
import skimage as ski
import tifffile as tiff
from tqdm.auto import tqdm

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
    frames = int(tree["acquisition"]["number_frames"])
    attrs["number_frames"] = frames
    z_offsets = []
    z_positions = []
    for z_offset in tree["focuslock"]["hardware_z_scan"]["z_offsets"].split(","):
        z_positions.append(float(z_offset))
        if float(z_offset) not in z_offsets:
            z_offsets.append(float(z_offset))
    attrs["z_offsets"] = z_offsets
    attrs["z_positions"] = z_positions
    shutters_str = tree["illumination"]["shutters"]
    if "shutter_" in shutters_str:
        colors_str = re.search(r"shutter_([\d_]+)_s", shutters_str).group(1)
        attrs["colors"] = list(map(int, colors_str.split("_")))
        attrs["frames_per_color"] = [frames // len(attrs["colors"]) for _ in attrs["colors"]]
    elif re.search(r"f\d+", shutters_str):
        color_matches = re.findall(r"(\d+)(?=f\d+)", shutters_str)
        attrs["colors"] = list(map(int, color_matches))
        frames_matches = re.findall(r"f(\d+)", shutters_str)
        attrs["frames_per_color"] = list(map(int, frames_matches))
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
        file_pattern = re.sub(r"\{fov:[^}]+\}", "{fov}", file_pattern)
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


def _reconstruct_sparse_frame_map(
    z_positions: list[float],
    color_order: np.ndarray,
    frames_per_color: list[int],
) -> tuple[list[float], np.ndarray, dict[tuple[int, int], int]]:
    """
    Interleaved acquisition reconstruction.

    Assumptions:
      - z_positions defines acquisition order; identical contiguous values form a z-run.
      - Within each z-run, frames are acquired in color_order, skipping colors whose
        remaining frame budget is 0.
      - frames_per_color gives TOTAL frames acquired for each color across the whole stack.
      - (color, z_index) appears at most once (one frame per color per z plane).
    """
    z_positions = list(map(float, z_positions))
    if len(frames_per_color) != len(color_order):
        raise ValueError(
            f"frames_per_color length ({len(frames_per_color)}) must match " f"number of colors ({len(color_order)})."
        )
    # Discard extra frames
    n_frames = np.sum(frames_per_color)
    z_positions = z_positions[:n_frames]
    if sum(frames_per_color) != n_frames:
        raise ValueError(
            f"Inconsistent metadata: sum(frames_per_color)={sum(frames_per_color)} " f"but len(z_positions)={n_frames}."
        )
    # Build z planes in acquisition order (by change points)
    z_planes: list[float] = []
    run_starts: list[int] = []
    last = object()
    for i, z in enumerate(z_positions):
        if z != last:
            z_planes.append(z)
            run_starts.append(i)
            last = z
    run_starts.append(n_frames)  # sentinel end

    remaining = [int(n) for n in frames_per_color]
    frame_colors = np.empty(n_frames, dtype=color_order.dtype)

    # Assign colors within each z-run, restarting scan each z plane
    for r in range(len(z_planes)):
        start = run_starts[r]
        end = run_starts[r + 1]
        run_len = end - start
        # build the sequence of colors that can still be acquired at this z plane
        available = [idx for idx, rem in enumerate(remaining) if rem > 0]
        if run_len > len(available):
            raise ValueError(
                f"Inconsistent metadata: z plane {r} requires {run_len} frames, "
                f"but only {len(available)} colors have remaining frames."
            )
        # Acquire in color_order, skipping exhausted colors
        write_i = start
        for ci in range(len(color_order)):
            if remaining[ci] <= 0:
                continue
            frame_colors[write_i] = color_order[ci]
            remaining[ci] -= 1
            write_i += 1
            if write_i == end:
                break
        if write_i != end:
            raise ValueError(
                f"Inconsistent metadata: did not fill z-run {r} " f"(filled {write_i-start} of {run_len})."
            )
    if any(rem != 0 for rem in remaining):
        raise ValueError("Inconsistent metadata: did not consume frames_per_color exactly.")
    # Map (color, z_index) -> frame_index
    z_index_of_plane = {z: idx for idx, z in enumerate(z_planes)}
    cz_to_frame: dict[tuple[int, int], int] = {}
    for i, (z, c) in enumerate(zip(z_positions, frame_colors, strict=False)):
        zi = z_index_of_plane[float(z)]
        key = (int(c), zi)
        if key in cz_to_frame:
            raise ValueError(f"Inconsistent metadata: multiple frames for color={int(c)} at z_index={zi}.")
        cz_to_frame[key] = i

    return z_planes, frame_colors, cz_to_frame


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
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")
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
    # Process colors selection (values from color_order)
    if colors is not None:
        if color_order is None:
            raise ValueError("Cannot select colors without xml file or color_order specified")
        if isinstance(colors, int) or isinstance(colors, str):
            colors = [colors]
        attrs["colors"] = colors
        req_colors = np.array(colors).astype(np.array(color_order).dtype)
        if not np.all(np.isin(req_colors, np.array(color_order))):
            missing = np.setdiff1d(req_colors, np.array(color_order))
            raise ValueError(f"Color {missing} not found in image colors {np.array(color_order)}")
        colors_arr = req_colors
        color_slices = np.array([np.where(color_order == c)[0][0] for c in colors_arr])  # problem
        n_colors = len(colors_arr)
    else:
        colors_arr = np.array(color_order)
    # Discard extra frames
    frames_per_color = np.array(attrs.get("frames_per_color", []))
    if frames_per_color.size > 0:
        attrs["z_positions"] = attrs["z_positions"][: np.sum(frames_per_color)]
        attrs["z_offsets"] = np.array(attrs["z_offsets"])[: np.max(frames_per_color)].tolist()
        z_max = len(attrs["z_offsets"])
    # Process z-slice selection
    if z_slices is not None:
        if z_max is None:
            raise ValueError("Cannot select z-slices without xml metadata file")
        if isinstance(z_slices, int):
            z_slices = [z_slices]
        z_slices = np.array(z_slices)
        attrs["z_offsets"] = np.array(attrs["z_offsets"])[z_slices].astype(float).tolist()
    is_sparse = frames_per_color.size > 0 and frames_per_color.min() != frames_per_color.max()
    if is_sparse:
        if colors is None and z_slices is None:
            raise ValueError(
                "This image has ragged (non-rectangular) color-by-z acquisition. "
                "You must specify either 'colors' or 'z_slices' to read it."
            )
        z_planes, _, cz_to_frame = _reconstruct_sparse_frame_map(
            z_positions=attrs["z_positions"],
            color_order=np.array(color_order),
            frames_per_color=frames_per_color,
        )
        z_max = len(z_planes)
        # If selecting frames, validate all requested (color,z) exist
        if (z_slices is not None) or (colors is not None):
            if z_slices is None:
                z_slices = np.arange(z_max, dtype=int)

            frames_list: list[int] = []
            missing_pairs: list[tuple[int, int]] = []
            for zi in z_slices.tolist():
                for c in colors_arr.tolist():
                    key = (int(c), int(zi))
                    if key not in cz_to_frame:
                        missing_pairs.append(key)
                    else:
                        frames_list.append(cz_to_frame[key])

            if missing_pairs:
                # Build a helpful message grouped by color
                by_color: dict[int, list[int]] = {}
                for c, zi in missing_pairs:
                    by_color.setdefault(c, []).append(zi)
                detail = ", ".join(
                    f"{c}: missing z_slices {sorted(set(zis))}"
                    for c, zis in sorted(by_color.items(), key=lambda x: x[0])
                )
                raise ValueError(
                    "Invalid (colors, z_slices) selection for this image. "
                    f"Requested some color/z combinations that do not exist: {detail}."
                )
            frames = np.array(frames_list, dtype=int)
            # Update attrs to reflect selection
            attrs["colors"] = colors_arr.tolist()
            attrs["z_positions_selected"] = [z_planes[i] for i in z_slices.tolist()]
            n_colors = len(colors_arr)
        # Load image (frames can be None, meaning full read)
        if suffix == ".dax":
            img = read_dax(path, shape=shape, frames=frames, **plugin_args)
        else:
            raw = ski.io.imread(path, plugin=plugin, **plugin_args)
            img = raw if frames is None else raw[frames]
    else:
        # Get frames
        if z_slices is not None or colors is not None:
            frames = np.arange(z_max * len(color_order)).reshape(z_max, len(color_order))
            if z_slices is not None:
                frames = np.take(frames, z_slices, axis=0)
            if colors is not None:
                frames = np.take(frames, color_slices, axis=1)
            frames = frames.flatten()
        else:
            frames = np.arange(z_max * len(color_order)) if z_max is not None else None
        # Load image
        if suffix == ".dax":
            img = read_dax(path, shape=shape, frames=frames, **plugin_args)
        elif suffix in [".tif", ".tiff"] and plugin is None:
            with tiff.TiffFile(path) as tif:
                pages = tif.pages
                if isinstance(frames, list | tuple | np.ndarray):
                    img = np.stack([pages[i].asarray() for i in frames])
                else:
                    img = pages[frames].asarray()
            img = np.squeeze(img)
        else:
            img = ski.io.imread(path, plugin=plugin, **plugin_args)[frames]
            img = np.squeeze(img)
    # Reshape image if necessary
    if n_colors > 1:
        img = np.reshape(img, (n_colors, img.shape[0] // n_colors, *img.shape[1:]), order="F")
    # Apply transpose and flip operations
    if suffix == ".dax" and attrs.get("transpose", False):
        img = img.swapaxes(-1, -2)
    if attrs.get("flip_horizontal", False):
        img = np.flip(img, axis=-1)
    if attrs.get("flip_vertical", False):
        img = np.flip(img, axis=-2)
    if suffix != ".dax" and attrs.get("transpose", False):  # transpose second for tif image
        img = img.swapaxes(-1, -2)
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
        if file_pattern != "{series}":
            file_pattern = determine_fov_format(path, fov=fov, series=series[0], file_pattern=file_pattern)
        for s in series:
            file = path / file_pattern.format(series=s, fov=fov).format(fov=fov)
            img, attr = read_img(file, z_slices=z_slices, z_project=z_project, colors=colors)
            imgs.append(img)
            attrs.append(attr)
    # Load images given channels df
    elif channels is not None:
        if colors is not None:
            channels = channels.query("color in @colors")
        if file_pattern != "{series}":
            file_pattern = determine_fov_format(
                path, fov=fov, series=channels["series"].values[0], file_pattern=file_pattern
            )
        for s, s_channels in channels.groupby("series", sort=False):
            file = path / file_pattern.format(series=s, fov=fov).format(fov=fov)
            img, attr = read_img(
                file,
                colors=s_channels["color"].values,
                z_slices=z_slices,
                z_project=z_project,
            )
            imgs.append(img)
            attrs.append(attr)
    # Reshape images
    if len(imgs) > 1 and len(imgs[0].shape) > 2:
        imgs = np.concatenate(imgs, axis=0)
    elif len(imgs) > 1 and len(imgs[0].shape) == 2:
        imgs = np.stack(imgs, axis=0)
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
    microns_per_pixel: float | None = None,
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
    for fov in tqdm(fovs, desc=f"Reading mosaic {series}", unit="fov"):
        img, attrs = read_fov(
            path,
            series=series,
            fov=fov,
            colors=colors,
            z_slices=z_slices,
            z_project=z_project,
            file_pattern=file_pattern,
        )
        if downsample > 1:
            if len(img.shape) == 2:
                img = img[::downsample, ::downsample]
            elif len(img.shape) == 3:
                img = img[:, ::downsample, ::downsample]
        if filter is not None:
            img = filter(img, **filter_args)
        imgs.append(img)
        positions.append(attrs["stage_position"])
        if (microns_per_pixel is None) and ("micron_per_pixel" in attrs.keys()):
            microns_per_pixel = attrs["micron_per_pixel"]
    # Create mosaic
    mosaic, bounds = create_mosaic(imgs, positions, microns_per_pixel * downsample)
    return mosaic, bounds
