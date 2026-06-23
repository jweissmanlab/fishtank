import argparse
import glob
from pathlib import Path

import pandas as pd

import fishtank as ft

from ._utils import parse_path


def _limit(value):
    """Parse a panel-limit argument: the string ``"auto"`` or a float."""
    return "auto" if str(value).lower() == "auto" else float(value)


def get_parser():
    """Get parser for plot-drift script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-i", "--input", type=parse_path, required=True,
        help="detect-spots output directory containing drift_*.csv files",
    )
    parser.add_argument(
        "-o", "--output", type=parse_path, default=None,
        help="Output figure path (relative or absolute). Default: <input>/drift_qc.png",
    )
    parser.add_argument("--pattern", type=str, default="drift_*.csv", help="Glob for per-FOV drift files")
    parser.add_argument("--title", type=str, default="detect-spots drift QC", help="Figure title")
    parser.add_argument(
        "--scatter_max", type=_limit, default=100.0,
        help="Panel 1 axis half-range (px), symmetric. A number or 'auto'.",
    )
    parser.add_argument(
        "--dmax", type=_limit, default=100.0,
        help="Panel 2 histogram upper bound (px). A number or 'auto'.",
    )
    parser.add_argument("--bin_size", type=float, default=5.0, help="Panel 2 histogram bin width (px)")
    parser.add_argument("--auto_pct", type=float, default=99.0, help="Percentile used for 'auto' limits")
    parser.add_argument("--cmap", type=str, default="coolwarm", help="Diverging colormap for the panel 3 heatmap")
    parser.add_argument("--fontsize", type=float, default=12.0, help="Base font size (matplotlib default is 10)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution")
    parser.set_defaults(func=plot_drift)
    return parser


def plot_drift(
    input: str | Path,
    output: str | Path = None,
    pattern: str = "drift_*.csv",
    title: str = "detect-spots drift QC",
    scatter_max: float | str = 100,
    dmax: float | str = 100,
    bin_size: float = 5,
    auto_pct: float = 99,
    cmap: str = "coolwarm",
    fontsize: float = 12,
    dpi: int = 150,
    **kwargs,
):
    """Plot per-round/per-fov registration drift from detect-spots output.

    fishtank plot-drift -i spots

    Parameters
    ----------
    input
        detect-spots output directory containing drift_{fov}.csv files.
    output
        Output figure path (relative or absolute). Defaults to <input>/drift_qc.png.
    pattern
        Glob pattern for the per-FOV drift files.
    title
        Figure title.
    scatter_max
        Panel 1 axis half-range (px), symmetric. A number or "auto" (outlier-resistant).
    dmax
        Panel 2 histogram upper bound (px). A number or "auto".
    bin_size
        Panel 2 histogram bin width (px).
    auto_pct
        Percentile used to compute the "auto" limits.
    cmap
        Diverging colormap for the panel 3 (distance - median) heatmap.
    fontsize
        Base font size (matplotlib default is 10).
    dpi
        Figure resolution.
    """
    if output is None:
        output = Path(input) / "drift_qc.png"
    files = sorted(glob.glob(str(Path(input) / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {input}")
    drift = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ft.pl.plot_drift(
        drift, out=output, title=title, dpi=dpi, scatter_max=scatter_max, dmax=dmax,
        bin_size=bin_size, auto_pct=auto_pct, cmap=cmap, fontsize=fontsize,
    )
    print(
        f"Plotted drift for {drift['fov'].nunique()} fovs x {drift['series'].nunique()} rounds -> {output}"
    )
