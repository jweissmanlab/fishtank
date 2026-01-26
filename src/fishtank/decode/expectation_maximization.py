import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns


def _parallel_query(kd_tree, corrected, threads=None):
    """Parallelize the query step of the kd-tree."""
    if threads is None:
        threads = min(mp.cpu_count(), 16)
    query_chunk = partial(kd_tree.query)
    chunks = np.array_split(corrected, threads)
    with mp.Pool(threads) as pool:
        results = pool.map(query_chunk, chunks)
    dists = np.concatenate([res[0] for res in results])
    decode = np.concatenate([res[1] for res in results])
    return dists, decode


def expectation_maximization(
    intensities: pd.DataFrame | np.ndarray,
    codebook: pd.DataFrame,
    max_dist: float = 1.5,
    min_snr: float = 4,
    iter: int = 10,
    plot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode intensities using expectation maximization.

    Parameters
    ----------
    intensities
        A (N,C) array of intensities where N is the number of spots and C is the number of channels.
    codebook
        A (K,C) array of codebook vectors where K is the number of codewords.
    max_dist
        Maximum distance between intensity and codebook vectors.
    min_snr
        Minimum signal-to-noise for bits in the codebook.
    iter
        Number of iterations.
    plot
        Whether to plot heatmap of corrected intensities annotated with decoded values.

    Returns
    -------
    decoding
        a DataFrame of spots with decoded value, dist, intensity, and snr columns
    bit_performance
        a DataFrame with bit, intensity, and snr columns
    """
    # Setup
    logger = logging.getLogger("fix_overlaps")
    logger.setLevel(logging.INFO)
    index = None
    if isinstance(intensities, pd.DataFrame):
        index = intensities.index
        intensities = intensities.loc[:, codebook.columns].values
    kd_tree = sp.spatial.KDTree(codebook.values)
    # Initialization
    bit_intensity = np.percentile(intensities, 95, axis=0, keepdims=True)
    bit_intensity = bit_intensity / np.mean(bit_intensity)
    bit_snr = np.ones_like(bit_intensity) * 10
    spot_intensity = np.percentile(intensities / bit_intensity, 95, axis=1, keepdims=True)
    spot_intensity[spot_intensity == 0] = 1
    corrected = intensities / np.dot(spot_intensity, bit_intensity)
    # EM Loop
    for i in range(iter):
        # E step
        bit_snr[bit_snr < min_snr] = min_snr
        corrected = np.clip((corrected - 1 / bit_snr) / (1 - 1 / bit_snr), 0, 1)
        dists, decode = _parallel_query(kd_tree, corrected)
        logger.info(f"Iteration {i}: {np.mean(dists < max_dist) * 100:.2f}% of spots with distance < {max_dist}")
        # M step
        noise_mask = ~codebook.values[decode].astype(bool)
        decoded_mask = dists < max_dist
        bit_intensity = np.ma.mean(
            np.ma.masked_array(intensities[decoded_mask] / spot_intensity[decoded_mask], noise_mask[decoded_mask]),
            axis=0,
            keepdims=True,
        )
        bit_intensity = bit_intensity / np.mean(bit_intensity)
        spot_intensity = np.ma.mean(np.ma.masked_array(intensities / bit_intensity, noise_mask), axis=1, keepdims=True)
        corrected = intensities / np.dot(spot_intensity, bit_intensity)
        bit_snr = np.array(
            np.ma.mean(
                np.ma.masked_array(1 / corrected[decoded_mask], ~noise_mask[decoded_mask]), axis=0, keepdims=True
            )
        )
    # Format results
    spot_noise = np.ma.mean(np.ma.masked_array(intensities, ~noise_mask), axis=1)
    decoding = pd.DataFrame(
        {
            "value": codebook.index[decode],
            "dist": dists,
            "intensity": spot_intensity[:, 0],
            "snr": spot_intensity[:, 0] / spot_noise,
        },
        index=index,
    )
    bit_performance = pd.DataFrame({"bit": codebook.columns, "intensity": bit_intensity[0], "snr": bit_snr[0]})
    # Plot
    if plot:
        corrected = pd.DataFrame(
            corrected[decoding.dist < max_dist],
            index=decoding[decoding.dist < max_dist].value.values,
            columns=codebook.columns,
        )
        if len(corrected) > 1000:
            corrected = corrected.sample(1000)
        sns.heatmap(corrected.sort_index(), vmax=2)
    return decoding, bit_performance
