from collections.abc import Callable

import numpy as np


def _circle_mask(x, y, radius, shape):
    """Create a circular mask."""
    y, x = np.ogrid[-y : shape[0] - y, -x : shape[1] - x]
    mask = x**2 + y**2 <= radius**2
    return mask


def _spot_intensity(img, x, y, z, radius, mask, agg=np.max):
    """Get intensity vector for a spot."""
    y_min, y_max = max(0, y - radius), min(img.shape[-2], y + radius + 1)
    x_min, x_max = max(0, x - radius), min(img.shape[-1], x + radius + 1)
    if y_max - y_min != mask.shape[0] or x_max - x_min != mask.shape[1]:
        mask = _circle_mask(x - x_min, y - y_min, radius, (y_max - y_min, x_max - x_min))
    if z is None:
        intensity = agg(img[:, y_min:y_max, x_min:x_max][:, mask], axis=-1)
    else:
        intensity = agg(img[:, z, y_min:y_max, x_min:x_max][:, mask], axis=-1)
    return intensity


def get_intensities(
    img: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None, radius: int = 5, agg: Callable = np.max
) -> np.ndarray:
    """Correct image illumination.

    Parameters
    ----------
    img
        A (Y,X), (C,Y,X) or (C,Z,Y,X) image.
    x
        A numpy array of x coordinates.
    y
        A numpy array of y coordinates.
    z
        A numpy array of z coordinates. If None, the image is assumed to be 2D.
    radius
        The radius of the spot in pixels.
    agg
        The aggregation function to use.

    Returns
    -------
    intensities
        a N x C numpy array of intensities where N is the number of spots and C is the number of channels.
    """
    # Setup
    if z is None:
        if img.ndim == 4:
            raise ValueError("z must be specified for images with 4 dimensions")
        if img.ndim == 2:
            img = img[np.newaxis, ...]
    else:
        if img.ndim == 2:
            raise ValueError("z must be None for images with 2 dimensions")
        elif img.ndim == 3:
            img = img[np.newaxis, ...]
    intensities = np.zeros((len(x), img.shape[0]), dtype=img.dtype)
    circle_mask = _circle_mask(radius, radius, radius, (2 * radius + 1, 2 * radius + 1))
    # Get intensities
    for i in range(len(x)):
        if z is None:
            intensities[i] = _spot_intensity(img, x[i], y[i], None, radius, circle_mask, agg)
        else:
            intensities[i] = _spot_intensity(img, x[i], y[i], z[i], radius, circle_mask, agg)
    return intensities.astype(img.dtype)
