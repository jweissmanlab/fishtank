import numpy as np
import skimage as ski


def unsharp_mask(
    img: np.ndarray,
    sigma: float | list = 1,
    truncate: float = 2,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Gaussian unsharp mask.

    Parameters
    ----------
    img
        A (C,Z,X,Y), (Z,X,Y), (C,X,Y), or (X,Y) image.
    sigma
        Standard deviation for Gaussian kernel. If a list, the standard deviation for each channel.
    channel_axis
        If None, channel axis is assumed to be the first axis if the image has more than 2 dimensions.
    truncate
        Truncate the filter at this many standard deviations.
    gpu
        Wjether to use the GPU.
    tilesize
        Size of tiles. If None, no tiling is used.

    Returns
    -------
    filtered
        a numpy array of the same shape and dtype as the input image.
    """
    if channel_axis is None:
        channel_axis = 0 if len(img.shape) > 2 else None
    blur = ski.filters.gaussian(img, sigma=sigma, channel_axis=channel_axis, truncate=truncate, preserve_range=True)
    filtered = np.clip(img - blur, 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return filtered
