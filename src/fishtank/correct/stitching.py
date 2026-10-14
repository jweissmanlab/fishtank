"""Stitch a grid of FOVs from the overlaps between neighbouring FOVs.

Ported from MERlin's ``merlin.util.globalpositions`` (MIT License,
Copyright (c) 2019 Harvard University). The method:

1. register every 4-connected neighbour pair on its overlap band
   (phase correlation, optionally Hann-windowed);
2. reject failed registrations per direction, on measured - nominal;
3. fit one 2x2 affine ``A`` on edge displacements, ``m = A d`` (camera vs
   stage rotation and scale);
4. solve every FOV's position by least squares on the kept edges, with a
   weak ``A d`` prior on every grid edge.

Positions stay in the camera frame with one translation per FOV, so a
pixel ``(x, y)`` of FOV ``f`` is at ``position[f] + (x, y) * pixel_size``.
Images must be oriented so that stage +x/+y run along image +columns/+rows.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import shift as ndi_shift
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
from skimage.registration import phase_cross_correlation

DIRECTIONS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
OPPOSITE = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}


def overlap_bands(img: np.ndarray, overlap_px: int) -> dict:
    """Get the four overlap bands of a 2D image.

    Parameters
    ----------
    img
        A (Y,X) image.
    overlap_px
        Band width in pixels.

    Returns
    -------
    bands
        a dict from side (the direction of the neighbour touching it: "+x", "-x", "+y", "-y") to band.
    """
    return {
        "+x": img[:, -overlap_px:],
        "-x": img[:, :overlap_px],
        "+y": img[-overlap_px:, :],
        "-y": img[:overlap_px, :],
    }


def find_edges(positions: pd.DataFrame, tolerance: float = 0.25) -> pd.DataFrame:
    """Find the 4-connected edges of a FOV grid.

    A neighbour in direction "+x" is the closest FOV whose offset is mostly along +x, within
    ``1 + tolerance`` times the FOV's distance to its nearest neighbour.

    Parameters
    ----------
    positions
        A DataFrame indexed by FOV with columns "x" and "y" (microns).
    tolerance
        Allowed extra distance, as a fraction of the local step.

    Returns
    -------
    edges
        a DataFrame with one row per edge: "anchor" < "neighbor", "direction" (anchor to neighbour),
        and the nominal displacement "nominal_dx", "nominal_dy" (microns).
    """
    fovs = positions.index.to_numpy()
    xy = positions[["x", "y"]].to_numpy(dtype=float)
    local_step = KDTree(xy).query(xy, k=2)[0][:, 1]
    found = {}
    for k, fov in enumerate(fovs):
        d = xy - xy[k]
        dist = np.hypot(d[:, 0], d[:, 1])
        dist[k] = np.inf
        for direction, (ux, uy) in DIRECTIONS.items():
            if ux != 0:
                ok = (np.abs(d[:, 0]) >= np.abs(d[:, 1])) & (np.sign(d[:, 0]) == ux)
            else:
                ok = (np.abs(d[:, 1]) > np.abs(d[:, 0])) & (np.sign(d[:, 1]) == uy)
            ok &= np.isfinite(dist)
            if not ok.any():
                continue
            best = np.where(ok)[0][np.argmin(dist[ok])]
            if dist[best] > (1 + tolerance) * local_step[k]:
                continue
            a, b = (fov, fovs[best]) if fov < fovs[best] else (fovs[best], fov)
            found[(a, b)] = direction if a == fov else OPPOSITE[direction]
    edges = pd.DataFrame([(a, b, direction) for (a, b), direction in sorted(found.items())],
                         columns=["anchor", "neighbor", "direction"])
    nominal = positions.loc[edges.neighbor, ["x", "y"]].to_numpy() - positions.loc[edges.anchor, ["x", "y"]].to_numpy()
    edges["nominal_dx"], edges["nominal_dy"] = nominal[:, 0], nominal[:, 1]
    return edges


def _crop_offset_um(frame_shape: tuple, direction: str, overlap_px: int, pixel_size: float) -> np.ndarray:
    """Anchor to neighbour offset (microns) at which the two overlap bands show the same area."""
    ux, uy = DIRECTIONS[direction]
    h, w = frame_shape
    return np.array([ux * (w - overlap_px), uy * (h - overlap_px)], dtype=float) * pixel_size


def _hann(band: np.ndarray) -> np.ndarray:
    """Mean-subtracted band times a 2D Hann window, so the band edges do not vote for zero shift."""
    band = band.astype(float) - band.mean()
    return band * np.outer(np.hanning(band.shape[0]), np.hanning(band.shape[1]))


def _pearson_at_shift(a: np.ndarray, b: np.ndarray, shift: np.ndarray) -> float:
    """Pearson correlation of a and b after shifting b by shift (rows, cols), over the shared pixels."""
    b = ndi_shift(b.astype(float), shift, order=1, mode="constant", cval=np.nan)
    ok = np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return 0.0
    return float(np.corrcoef(a[ok].astype(float), b[ok])[0, 1])


def register_edges(
    edges: pd.DataFrame,
    load_bands,
    frame_shape: tuple,
    overlap_px: int,
    pixel_size: float,
    hann: bool = False,
    upsample_factor: int = 100,
) -> pd.DataFrame:
    """Measure each edge's displacement by phase correlation of its overlap bands.

    Parameters
    ----------
    edges
        Output of :func:`find_edges`.
    load_bands
        Function mapping a FOV to its :func:`overlap_bands`.
    frame_shape
        Full image shape (Y, X).
    overlap_px
        Band width used by :func:`overlap_bands`.
    pixel_size
        Microns per pixel.
    hann
        Apply a Hann window before correlating. Helps smooth images (e.g. DAPI max projections) whose
        band edges otherwise vote for zero shift; can hurt on narrow bands.
    upsample_factor
        Sub-pixel precision of the phase correlation (1 / pixels).

    Returns
    -------
    edges
        a copy with the measured displacement "dx", "dy" (microns) and the band "pearson" at that shift.
    """
    edges = edges.copy()
    dx, dy, pearson = [], [], []
    for anchor, neighbor, direction in edges[["anchor", "neighbor", "direction"]].itertuples(index=False):
        a = load_bands(anchor)[direction].astype(float)
        b = load_bands(neighbor)[OPPOSITE[direction]].astype(float)
        if hann:
            shift = phase_cross_correlation(_hann(a), _hann(b), upsample_factor=upsample_factor)[0]
        else:
            shift = phase_cross_correlation(a, b, upsample_factor=upsample_factor)[0]
        offset = _crop_offset_um(frame_shape, direction, overlap_px, pixel_size)
        dx.append(offset[0] + shift[1] * pixel_size)
        dy.append(offset[1] + shift[0] * pixel_size)
        pearson.append(_pearson_at_shift(a, b, shift))
    edges["dx"], edges["dy"], edges["pearson"] = dx, dy, pearson
    return edges


def filter_edges(edges: pd.DataFrame, mad_threshold: float = 5.0, min_threshold: float = 0.0) -> pd.Series:
    """Flag failed registrations, separately per direction.

    The residual (measured - nominal displacement) depends on direction (stage backlash), so each
    direction is judged on its own: an edge is rejected when its distance from that direction's median
    residual exceeds ``median + mad_threshold * 1.4826 * MAD`` of those distances, and ``min_threshold``.

    Parameters
    ----------
    edges
        Output of :func:`register_edges`.
    mad_threshold
        Threshold in robust standard deviations.
    min_threshold
        Lowest allowed threshold (microns), so locked whole-pixel shifts (MAD = 0) are not all rejected.

    Returns
    -------
    keep
        a boolean Series aligned with edges.
    """
    residual = edges[["dx", "dy"]].to_numpy() - edges[["nominal_dx", "nominal_dy"]].to_numpy()
    keep = pd.Series(True, index=edges.index)
    for _, idx in edges.groupby("direction").groups.items():
        r = residual[edges.index.get_indexer(idx)]
        deviation = np.hypot(*(r - np.median(r, axis=0)).T)
        median = np.median(deviation)
        threshold = max(median + mad_threshold * 1.4826 * np.median(np.abs(deviation - median)), min_threshold)
        keep[idx] = deviation <= threshold
    return keep


def fit_affine(edges: pd.DataFrame) -> np.ndarray:
    """Fit the 2x2 affine ``A`` with measured = A nominal displacement, by least squares.

    A direction no edge spans falls back to the identity.
    """
    d = edges[["nominal_dx", "nominal_dy"]].to_numpy(dtype=float)
    m = edges[["dx", "dy"]].to_numpy(dtype=float)
    ridge = 1e-6 * np.eye(2)
    return np.linalg.solve(d.T @ d + ridge, d.T @ m + ridge).T


def rotation_and_scale(affine: np.ndarray) -> tuple[float, np.ndarray]:
    """Split an affine into its rotation (degrees) and singular values (scale)."""
    u, s, vt = np.linalg.svd(affine)
    r = u @ vt
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))), s


def apply_affine(affine: np.ndarray, positions: pd.DataFrame, origin) -> pd.DataFrame:
    """Positions from the affine alone, fixed at the origin FOV: ``p0 + A (p - p0)``."""
    p0 = positions.loc[origin, ["x", "y"]].to_numpy(dtype=float)
    xy = p0 + (positions[["x", "y"]].to_numpy(dtype=float) - p0) @ affine.T
    return pd.DataFrame(xy, index=positions.index, columns=["x", "y"])


def edge_errors(edges: pd.DataFrame, positions: pd.DataFrame) -> np.ndarray:
    """Distance (microns) between each edge's measured displacement and the one the positions imply."""
    solved = positions.loc[edges.neighbor, ["x", "y"]].to_numpy() - positions.loc[edges.anchor, ["x", "y"]].to_numpy()
    return np.hypot(*(solved - edges[["dx", "dy"]].to_numpy()).T)


def solve_positions(
    edges: pd.DataFrame,
    positions: pd.DataFrame,
    prior_weight: float = 1e-3,
    grid_edges: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Solve every FOV's position from the measured edges, with a weak affine prior on every grid edge.

    Minimises ``sum_e |P_b - P_a - m_e|^2 + prior_weight * sum_ab |P_b - P_a - A d_ab|^2`` with one FOV per
    connected component pinned at its affine position. The prior makes FOVs without a kept edge follow
    the affine instead of the nominal positions.

    Parameters
    ----------
    edges
        Measured edges to use (already filtered).
    positions
        Nominal positions, a DataFrame indexed by FOV with columns "x" and "y" (microns).
    prior_weight
        Weight of each grid edge's prior relative to a measured edge; 0 disables it.
    grid_edges
        Edges for the prior (default: :func:`find_edges` of positions).

    Returns
    -------
    solved
        positions in the same format.
    affine
        the 2x2 displacement affine.
    info
        a dict with "n_edges", "n_components", "rms_error_um" (in-sample) and "rotation_deg", "scale".
    """
    fovs = positions.index.to_numpy()
    index = pd.Series(np.arange(len(fovs)), index=fovs)
    affine = fit_affine(edges)
    i = list(index[edges.anchor])
    j = list(index[edges.neighbor])
    d = edges[["nominal_dx", "nominal_dy"]].to_numpy(dtype=float)
    target = [edges[["dx", "dy"]].to_numpy(dtype=float) - d @ affine.T]
    weight = [np.ones(len(edges))]
    if prior_weight > 0:
        if grid_edges is None:
            grid_edges = find_edges(positions)
        i += list(index[grid_edges.anchor])
        j += list(index[grid_edges.neighbor])
        target.append(np.zeros((len(grid_edges), 2)))
        weight.append(np.full(len(grid_edges), prior_weight))
    target, weight = np.vstack(target), np.concatenate(weight)
    n_edges, n_fovs = len(i), len(fovs)
    incidence = sparse.csr_matrix(
        (np.r_[-np.ones(n_edges), np.ones(n_edges)], (np.r_[np.arange(n_edges), np.arange(n_edges)], np.r_[i, j])),
        shape=(n_edges, n_fovs),
    )
    n_components, labels = connected_components(
        sparse.csr_matrix((np.ones(n_edges), (i, j)), shape=(n_fovs, n_fovs)), directed=False
    )
    # Pin the FOV with the most measured edges in each component
    counts = np.bincount(np.r_[index[edges.anchor], index[edges.neighbor]].astype(int), minlength=n_fovs)
    pins = [int(np.where(labels == c)[0][np.argmax(counts[labels == c])]) for c in range(n_components)]
    origin = fovs[pins[int(np.argmax(np.bincount(labels)))]]
    affine_positions = apply_affine(affine, positions, origin)
    laplacian = (incidence.T @ sparse.diags(weight) @ incidence).tolil()
    for p in pins:
        laplacian[p, p] += 1e6
    laplacian = laplacian.tocsc()
    offsets = np.column_stack([spsolve(laplacian, incidence.T @ (weight * target[:, k])) for k in range(2)])
    solved = affine_positions + offsets
    errors = edge_errors(edges, solved)
    rotation, scale = rotation_and_scale(affine)
    info = {
        "n_edges": len(edges),
        "n_components": int(n_components),
        "rms_error_um": float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0,
        "rotation_deg": rotation,
        "scale": scale.tolist(),
    }
    return solved, affine, info


def cross_validate(
    edges: pd.DataFrame, positions: pd.DataFrame, n_folds: int = 5, seed: int = 0, prior_weight: float = 1e-3
) -> dict:
    """Held-out edge error: solve without one fold of edges and measure :func:`edge_errors` on it.

    Returns
    -------
    errors
        a dict of arrays (microns) over every held-out edge for the full solve ("final"), the affine alone
        ("affine_only") and the nominal positions ("nominal").
    """
    grid_edges = find_edges(positions)
    fold = np.random.default_rng(seed).integers(0, n_folds, len(edges))
    errors = {"final": [], "affine_only": [], "nominal": []}
    for k in range(n_folds):
        train, test = edges[fold != k], edges[fold == k]
        if len(test) == 0 or len(train) == 0:
            continue
        solved, affine, _ = solve_positions(train, positions, prior_weight, grid_edges)
        errors["final"].append(edge_errors(test, solved))
        errors["affine_only"].append(edge_errors(test, apply_affine(affine, positions, positions.index[0])))
        errors["nominal"].append(edge_errors(test, positions))
    return {name: np.concatenate(v) if v else np.zeros(0) for name, v in errors.items()}
