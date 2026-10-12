import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

import fishtank as ft

stitching = ft.correct.stitching

PIXEL = 0.5  # microns per pixel
FRAME = 400  # frame size in pixels
STEP = 340.0  # nominal step in pixels (60 px overlap, ~55 px after scaling)


def _grid(n=4, angle_deg=0.3, scale=1.01, backlash_px=2.0, seed=0):
    """Tiles cut from one random texture at known true positions."""
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.normal(size=(1700, 1700)), 2)
    texture[rng.integers(0, 1700, 1500), rng.integers(0, 1700, 1500)] += 30
    texture = gaussian_filter(texture, 1)
    theta = np.radians(angle_deg)
    affine = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    nominal, true, tiles = {}, {}, {}
    for col in range(n):
        for row in range(n):
            fov = col * n + row
            nominal_px = np.array([col * STEP, row * STEP])
            true_px = affine @ nominal_px + np.array([0.0, backlash_px * (col % 2)]) + 100
            true_px = np.round(true_px)
            x, y = int(true_px[0]), int(true_px[1])
            # Independent camera noise per tile: without it the smooth texture has almost no
            # high-frequency content, which phase correlation's whitening turns into false peaks
            tiles[fov] = texture[y : y + FRAME, x : x + FRAME] + rng.normal(scale=0.1, size=(FRAME, FRAME))
            nominal[fov] = nominal_px * PIXEL
            true[fov] = true_px * PIXEL
    nominal = pd.DataFrame.from_dict(nominal, orient="index", columns=["x", "y"])
    true = pd.DataFrame.from_dict(true, orient="index", columns=["x", "y"])
    return tiles, nominal, true


def _measure(tiles, nominal, overlap_px=80):
    bands = {fov: stitching.overlap_bands(img, overlap_px) for fov, img in tiles.items()}
    edges = stitching.find_edges(nominal)
    return stitching.register_edges(edges, bands.get, (FRAME, FRAME), overlap_px, PIXEL)


def _edge_error(positions, true):
    """Largest error (microns) of a grid edge's displacement, against the true one."""
    edges = stitching.find_edges(true)
    diff = (positions.loc[edges.neighbor].to_numpy() - positions.loc[edges.anchor].to_numpy()) - (
        true.loc[edges.neighbor].to_numpy() - true.loc[edges.anchor].to_numpy()
    )
    return float(np.hypot(*diff.T).max())


def test_find_edges():
    _, nominal, _ = _grid()
    edges = stitching.find_edges(nominal)
    assert len(edges) == 2 * 4 * 3  # 4x4 grid has 24 edges
    assert set(edges.direction) == {"+x", "+y"}  # anchor < neighbor, fovs numbered by column
    assert (edges.anchor < edges.neighbor).all()


def test_register_edges_recovers_displacements():
    tiles, nominal, true = _grid()
    edges = _measure(tiles, nominal)
    expected = true.loc[edges.neighbor].to_numpy() - true.loc[edges.anchor].to_numpy()
    error = np.hypot(*(edges[["dx", "dy"]].to_numpy() - expected).T)
    assert error.max() < 0.15  # 0.3 px, limited by the tile noise
    assert edges.pearson.min() > 0.7


def test_solve_positions():
    tiles, nominal, true = _grid()
    edges = _measure(tiles, nominal)
    solved, affine, info = stitching.solve_positions(edges, nominal)
    assert _edge_error(solved, true) < 0.15
    assert _edge_error(nominal, true) > 2  # nominal positions are far off
    # The affine of the true displacements (backlash adds a little rotation to the 0.3 deg)
    exact = edges.copy()
    exact[["dx", "dy"]] = true.loc[edges.neighbor].to_numpy() - true.loc[edges.anchor].to_numpy()
    rotation, scale = stitching.rotation_and_scale(stitching.fit_affine(exact))
    assert info["rotation_deg"] == pytest.approx(rotation, abs=0.02)
    assert info["scale"] == pytest.approx(scale, abs=0.001)
    assert np.mean(scale) == pytest.approx(1.01, abs=0.005)


def test_filter_edges_rejects_failed_registration():
    tiles, nominal, true = _grid()
    tiles[5] = np.random.default_rng(1).normal(size=(FRAME, FRAME))  # no shared content
    edges = _measure(tiles, nominal)
    keep = stitching.filter_edges(edges, min_threshold=PIXEL)
    bad = (edges.anchor == 5) | (edges.neighbor == 5)
    assert not keep[bad].any()
    assert keep[~bad].all()
    solved, _, _ = stitching.solve_positions(edges[keep], nominal)
    others = true.index != 5
    assert _edge_error(solved[others], true[others]) < 0.15


def test_cross_validate():
    tiles, nominal, _ = _grid()
    edges = _measure(tiles, nominal)
    errors = stitching.cross_validate(edges, nominal)
    assert np.median(errors["final"]) < np.median(errors["affine_only"]) < np.median(errors["nominal"])
    assert np.median(errors["final"]) < 0.15


if __name__ == "__main__":
    pytest.main(["-v", __file__])
