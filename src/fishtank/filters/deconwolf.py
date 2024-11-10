import configparser
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import skimage as ski


def _configure_deconwolf():
    """Configure Deconwolf."""
    config = configparser.ConfigParser()
    config_path = Path(__file__).resolve().parent.parent.parent.parent / ".config.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Please create a .config.ini file.")
    config.read(config_path)
    dw_path = Path(config.get("Paths", "dw_path"))
    if not dw_path.exists():
        raise FileNotFoundError(
            f"Deconwolf not found at {dw_path}. To use Deconwolf, please install it and set the path in the .config.ini file."
        )
    psf_path = Path(config.get("Paths", "dw_psf_path"))
    if not psf_path.exists():
        raise FileNotFoundError(
            f"Deconwolf PSF not found at {psf_path}. To use Deconwolf, please compute PSFs using `dw_bw`"
        )
    return dw_path, psf_path


def deconwolf(
    img: np.ndarray,
    colors: list | np.ndarray | int,
    z_step: float = 0.6,
    iter: int = 100,
    gpu: bool = False,
    tilesize: int = None,
) -> np.ndarray:
    """Deconwolf deconvolution.

    Parameters
    ----------
    img
        A (Z,X,Y) or (C,Z,X,Y) image.
    colors
        A list of channel colors. Must be the same length as the number of channels in the image.
    z_step
        Z step size in microns.
    iter
        Number of iterations.
    gpu
        Wjether to use the GPU.
    tilesize
        Size of tiles. If None, no tiling is used.

    Returns
    -------
    filtered
        a numpy array of the same shape and dtype as the input image.
    """
    # Setup
    dw_path, psf_path = _configure_deconwolf()
    has_channels = len(img.shape) == 4
    if not has_channels:
        img = np.expand_dims(img, axis=0)
    filtered = np.zeros_like(img)
    if isinstance(colors, int):
        colors = [colors]
    # Run deconwolf on each channel
    with tempfile.TemporaryDirectory() as tmp_dir:
        gpu = "--gpu" if gpu else ""
        tile = "--tilesize {tilesize}" if tilesize else ""
        for i, color in enumerate(colors):
            color_psf = psf_path / f"{color}_z{int(z_step*1000)}_psf.tiff"
            if not color_psf.exists():
                raise FileNotFoundError(f"PSF for color {color} and z_step {z_step} not found at {color_psf}.")
            ski.io.imsave(f"{tmp_dir}/img.tiff", img[i])
            command = f"{dw_path} --out {tmp_dir}/decon.tiff --iter {iter} {tile} {gpu} {tmp_dir}/img.tiff {color_psf}"
            subprocess.run(command, check=True, shell=True)
            filtered[i] = ski.io.imread(f"{tmp_dir}/decon.tiff")
    if not has_channels:
        filtered = filtered[0]
    return filtered.astype(img.dtype)
