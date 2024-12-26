import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def _colorize(X, rgb, norm, vmin, vmax):
    """Colorize an image."""
    if isinstance(vmin, str):
        vmin = np.percentile(X, float(vmin.replace("p", "")))
    if isinstance(vmax, str):
        vmax = np.percentile(X, float(vmax.replace("p", "")))
    if isinstance(norm, str):
        norm = {
            "linear": mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True),
            "log": mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True),
            "power": mcolors.PowerNorm(gamma=1, vmin=vmin, vmax=vmax, clip=True),
        }.get(norm.lower(), None)
    X = norm(X)
    return np.stack([X * rgb[i] for i in range(3)], axis=-1)


def imshow(
    X: np.ndarray,
    colors: list | str = None,
    norm: str = "linear",
    vmin: int | str = None,
    vmax: int | str = None,
    aspect: str | float | None = None,
    origin: str = "lower",
    extent: tuple | None = None,
    ax: plt.Axes = None,
    **kwargs,
) -> mpimg.AxesImage:
    """Display image with specified colors

    Parameters
    ----------
    X
        a (C,Y,X) image array representing a multi-channel image.
    colors
        a list of colors to use for each channel. Default is ["magenta", "green", "blue","yellow","cyan","red"].
    norm
        the normalization method used to scale scalar data to the [0, 1] range before mapping to colors using cmap.
        By default, a linear scaling is used, mapping the lowest value to 0 and the highest to 1.
    vmin
        the minimum value of the color scale. Can be a single value of a list of values, one for each channel.
        Strings with format "p05" are interpreted as percentiles.
    vmax
        the maximum value of the color scale. Can be a single value of a list of values, one for each channel.
        Strings with format "p95" are interpreted as percentiles.
    aspect
        the aspect ratio of the image. By default, this is set to "equal".
    origin
        place [0,0]  index of the array in the upper left or lower left corner of the Axes.
    extent
        [left, right, bottom, top] the bounding box in data coordinates that the image will fill.
        These values may be unitful and match the units of the Axes. The image is stretched individually along x and y to fill the box.
    ax
        a matplotlib Axes object to plot the image on. If None, the current Axes will be used.
    kwargs
        additional keyword arguments passed to the matplotlib imshow function.


    Returns
    -------
    ax
        the matplotlib Axes object used to plot the image.
    """
    # Default color mapping
    color_rgb = {
        "magenta": [1, 0, 1],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "yellow": [1, 1, 0],
        "cyan": [0, 1, 1],
        "red": [1, 0, 0],
    }
    # Handle single channel images
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    # Default colors for up to 6 channels
    n_channels = X.shape[0]
    if colors is None:
        colors = list(color_rgb.keys())[0:n_channels]
    # Validate the number of channels
    if n_channels != len(colors):
        raise ValueError(f"Expected {n_channels} colors, but got {len(colors)}")
    # Convert single values to lists
    if isinstance(vmin, str | int) or vmin is None:
        vmin = [vmin] * n_channels
    if isinstance(vmax, str | int) or vmax is None:
        vmax = [vmax] * n_channels
    if isinstance(colors, str):
        colors = [colors] * n_channels
    # Normalize each channel
    colorized = np.zeros((X.shape[-2], X.shape[-1], 3), np.float32)
    for i in range(n_channels):
        colorized += _colorize(X[i], color_rgb[colors[i]], norm=norm, vmin=vmin[i], vmax=vmax[i])
    colorized = np.clip(colorized, 0, 1)
    # Plot the image
    if ax is None:
        ax = plt.gca()
    ax.imshow(colorized, aspect=aspect, origin=origin, extent=extent, **kwargs)
    return ax
