from pathlib import Path

import numpy as np


def illumination(
    img: np.ndarray, colors: list | np.ndarray | int, corr_path: str | Path, transpose: bool = True, flip: bool = True
) -> np.ndarray:
    """Correct image illumination.

    Parameters
    ----------
    img
        A (X,Y), (C,X,Y) or (C,Z,X,Y) image.
    colors
        A list of channel colors. Must be the same length as the number of channels in the image.
    corr_path
        The path to the illumination correction file.
    transpose
        Whether to transpose the correction matrix.
    flip
        Whether to flip the correction matrix.

    Returns
    -------
    corrected
        a numpy array of the same shape and dtype as the input image.
    """
    # Setup
    dtype = img.dtype
    img = img.astype(np.uint32)
    corrected = np.zeros_like(img)
    unique_colors = np.unique(colors)
    shape = img.shape
    # Check inputs
    if isinstance(colors, int):
        colors = [colors]
    if len(colors) > 1 and len(colors) != shape[0]:
        raise ValueError("The length of colors must equal the number channels in the image.")
    # Correct illumination
    for color in unique_colors:
        color_corr = np.load(corr_path / f"illumination_correction_{color}_{shape[-2]}x{shape[-1]}.npy")
        if transpose:
            color_corr = color_corr.T
        if flip:
            color_corr = np.flip(color_corr, axis=-2)
        color_idx = np.where(colors == color)[0]
        if len(colors) > 1:
            corrected[color_idx, :] = img[color_idx, :] / color_corr
        else:
            corrected = img / color_corr
    return corrected.astype(dtype)
