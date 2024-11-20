import numpy as np
import pandas as pd


def color_normalization(intensities: pd.DataFrame | np.ndarray, colors: list | np.ndarray) -> np.ndarray:
    """Normalize channel intensities by color.

    Parameters
    ----------
    intensities
        A (N,C) array of intensities where N is the number of spots and C is the number of channels.
    colors
        A list of length C of colors.

    Returns
    -------
    intensities
        a (N,C) array of normalized intensities.
    """
    # Setup
    if isinstance(colors, pd.Series):
        colors = colors.values
    if isinstance(intensities, pd.DataFrame):
        intensities = intensities.values
    intensities = intensities.astype(float)
    # Get normalization factors
    unique_colors = list(set(colors))
    color_means = np.zeros(len(unique_colors))
    for i, color in enumerate(unique_colors):
        color_means[i] = np.mean(intensities[:, colors == color])
    color_means /= np.mean(color_means)
    # Normalize intensities
    for i, color in enumerate(unique_colors):
        intensities[:, colors == color] /= color_means[i]
    return intensities
