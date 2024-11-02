import pathlib

import numpy as np


def imread(fname: str | pathlib.Path) -> np.ndarray:
    """Generate a basic plot for an AnnData object.

    Parameters
    ----------
    fname
        img file name

    Returns
    -------
    img
        Image as numpy array, such that a gray-image is (Y,X), a 3D gray-image is (Z,Y,X), and multi-channel image is (C,Z,Y,X).
    """
    return 0