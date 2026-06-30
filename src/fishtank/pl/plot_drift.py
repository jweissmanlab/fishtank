import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _robust_max(values, pct=99, floor=10.0):
    """Outlier-resistant upper bound for an axis limit.

    Returns the ``pct``-th percentile of ``|values|`` rounded up to a clean
    1/2/5 x 10**k number, so a few extreme outliers do not set the scale.

    Parameters
    ----------
    values
        Array-like of values (signed); the magnitude is used.
    pct
        Percentile of ``|values|`` to use as the (pre-rounding) limit.
    floor
        Minimum returned value, used when ``values`` is empty or all zero.
    """
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return floor
    hi = float(np.percentile(v, pct))
    if hi <= 0:
        return floor
    mag = 10 ** math.floor(math.log10(hi))
    for m in (1, 2, 5, 10):
        if hi <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def plot_drift(
    drift: pd.DataFrame,
    out: str = None,
    title: str = None,
    dpi: int = 150,
    scatter_max: float | str = 100,
    dmax: float | str = 100,
    bin_size: float = 5,
    auto_pct: float = 99,
    cmap: str = "coolwarm",
    fontsize: float = 12,
) -> plt.Figure:
    """Plot per-round/per-fov registration drift as a three-panel QC figure.

    Parameters
    ----------
    drift
        DataFrame with columns ``fov``, ``series``, ``x_drift``, ``y_drift`` and
        optionally ``distance`` (computed from x/y if absent). One row per (fov, round),
        as written by ``detect-spots`` to ``channels_{fov}.csv``.
    out
        If given, save the figure to this path.
    title
        Optional figure suptitle.
    dpi
        Resolution for the saved figure.
    scatter_max
        Half-range (px) of the square, symmetric axes in panel 1; a number, or the
        string ``"auto"`` for an outlier-resistant limit (the ``auto_pct``-th percentile
        of ``|x_drift|`` and ``|y_drift|``).
    dmax
        Upper bound (px) of the panel-2 distance histogram; a number, or ``"auto"``.
    bin_size
        Histogram bin width (px) in panel 2.
    auto_pct
        Percentile used to compute the ``"auto"`` limits (outlier-resistant).
    cmap
        Diverging colormap for the panel-3 ``distance - median`` heatmap.
    fontsize
        Base font size (matplotlib default is 10); titles/labels scale relative to it.

    Returns
    -------
    fig
        Figure with three panels: (1) x vs y drift scatter colored by round on square
        symmetric axes; (2) histogram of drift distance with the median marked;
        (3) heatmap of ``drift distance - median`` with x = fov, y = round.
    """
    df = drift.copy()
    if "distance" not in df.columns:
        df["distance"] = np.hypot(df["x_drift"], df["y_drift"])
    rounds = list(dict.fromkeys(df["series"]))
    df["round"] = df["series"].map({s: i for i, s in enumerate(rounds)})
    n_round = len(rounds)
    median = float(df["distance"].median())

    # Resolve "auto" limits robustly (a few outliers must not set the scale).
    smax = (
        _robust_max(np.concatenate([df["x_drift"].to_numpy(), df["y_drift"].to_numpy()]), auto_pct)
        if str(scatter_max).lower() == "auto"
        else float(scatter_max)
    )
    dlim = _robust_max(df["distance"].to_numpy(), auto_pct) if str(dmax).lower() == "auto" else float(dmax)

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize + 1,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "figure.titlesize": fontsize + 3,
        }
    )
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    if title:
        fig.suptitle(title)

    # (1) x vs y drift scatter on square symmetric axes (+/- smax); points beyond the
    # limit are off-scale (counted in the title) so outliers do not dominate the view.
    n_off = int(((df["x_drift"].abs() > smax) | (df["y_drift"].abs() > smax)).sum())
    sc = ax[0].scatter(df["x_drift"], df["y_drift"], c=df["round"], cmap="viridis", s=14, alpha=0.7, edgecolors="none")
    ax[0].axhline(0, color="k", lw=0.6, ls=":")
    ax[0].axvline(0, color="k", lw=0.6, ls=":")
    ax[0].set_xlim(-smax, smax)
    ax[0].set_ylim(-smax, smax)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x drift (px)")
    ax[0].set_ylabel("y drift (px)")
    ax[0].set_title("(1) drift vectors" + (f"  ({n_off} outliers)" if n_off else ""))
    fig.colorbar(sc, ax=ax[0], label="round", fraction=0.046, pad=0.04)

    # (2) histogram of drift distance over [0, dlim], with the median marked; values
    # above dlim are out of range (counted in the title).
    bins = np.arange(0, dlim + bin_size, bin_size)
    in_range = df["distance"][(df["distance"] >= 0) & (df["distance"] <= dlim)]
    n_over = int((df["distance"] > dlim).sum())
    ax[1].hist(in_range, bins=bins, color="slategray", edgecolor="white", lw=0.4)
    ax[1].axvline(median, color="crimson", lw=1.6, ls="--", label=f"median = {median:.2f}")
    ax[1].set_xlim(0, dlim)
    ax[1].set_xlabel("drift distance (px)")
    ax[1].set_ylabel("count (fov x round)")
    ax[1].set_title("(2) drift distance" + (f"  ({n_over} outliers)" if n_over else ""))
    ax[1].legend(loc="best")

    # (3) heatmap of (distance - median): x = fov, y = round, diverging colormap centered
    # at the median, with symmetric outlier-resistant color limits.
    piv = df.pivot_table(index="round", columns="fov", values="distance", aggfunc="mean")
    piv = piv.reindex(sorted(piv.columns), axis=1).sort_index()
    dev = piv.to_numpy() - median
    vlim = _robust_max(dev, max(auto_pct, 98), floor=1.0)
    im = ax[2].imshow(dev, aspect="auto", origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax[2].set_xlabel("fov")
    ax[2].set_ylabel("round")
    ax[2].set_title(f"(3) drift distance - median ({median:.1f} px)")
    fovvals = piv.columns.to_numpy()
    xidx = np.linspace(0, len(fovvals) - 1, min(len(fovvals), 12)).astype(int)
    ax[2].set_xticks(xidx)
    ax[2].set_xticklabels(fovvals[xidx])
    ax[2].set_yticks(np.linspace(0, n_round - 1, min(n_round, 14)).astype(int))
    fig.colorbar(im, ax=ax[2], label="distance - median (px)", fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    if out is not None:
        fig.savefig(out, dpi=dpi)
    return fig
