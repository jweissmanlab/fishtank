import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import fishtank as ft

from ._utils import parse_path


def get_parser():
    """Get parser for map_proseg script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-t", "--transcripts", type=parse_path, required=True, help="Transcripts input file path")
    parser.add_argument(
        "-p", "--proseg_transcripts", type=parse_path, required=True, help="ProSeg transcripts input file path"
    )
    parser.add_argument("-o", "--output", type=parse_path, default="proseg_counts.csv", help="Output file path")
    parser.add_argument("--min_jaccard", type=float, default=0.4, help="Minimum transcript overlap for matching cells")
    parser.add_argument(
        "--min_prob", type=float, default=0.5, help="Minimum ProSeg probability for a transcript to be assigned"
    )
    parser.add_argument("--cell_column", type=str, default="cell", help="Column containing cell ID in transcripts")
    parser.add_argument("--cell_missing", type=int, default=0, help="Cell ID for unassigned transcripts")
    parser.add_argument(
        "--barcode_column", type=str, default="barcode_id", help="Column containing barcdoe ID in transcripts"
    )
    parser.add_argument(
        "--x_column", type=str, default="global_x", help="Column containing x-coordinate in transcripts"
    )
    parser.add_argument(
        "--y_column", type=str, default="global_y", help="Column containing y-coordinate in transcripts"
    )
    parser.add_argument("--z_column", type=str, default="global_z", help="Column containing z-slice in transcripts")
    parser.add_argument(
        "--codebook", type=parse_path, default=None, help="Codebook for assigning barcodes IDs to genes"
    )
    parser.set_defaults(func=assign_proseg)
    return parser


def assign_proseg(
    transcripts: str | Path,
    proseg_transcripts: str | Path,
    output: str | Path = "proseg_counts.csv",
    min_jaccard: float = 0.4,
    min_prob: float = 0.5,
    cell_column: str = "cell",
    cell_missing: int = 0,
    barcode_column: str = "barcode_id",
    x_column: str = "global_x",
    y_column: str = "global_y",
    z_column: str = "global_z",
    codebook: str | Path = None,
):
    """Assign additional transcripts to cells using ProSeg.

    fishtank assign-proseg -t transcripts -p proseg_transcripts -o output

    Parameters
    ----------
    transcripts
        Transcripts input file path.
    proseg_transcripts
        ProSeg transcripts input file path.
    output
        Output file path.
    min_jaccard
        Minimum transcript overlap for matching cells.
    min_prob
        Minimum ProSeg probability for a transcript to be assigned.
    cell_column
        Column containing cell ID in transcripts.
    cell_missing
        Cell ID for unassigned transcripts.
    barcode_column
        Column containing barcdoe ID in transcripts.
    x_column
        Column containing x-coordinate in transcripts.
    y_column
        Column containing y-coordinate in transcripts.
    z_column
        Column containing z-slice in transcripts.
    codebook
        Codebook for assigning barcodes IDs to genes.
    """
    # Setup
    logger = logging.getLogger("assign_proseg")
    logger.info(f"fishtank version: {ft.__version__}")
    x = x_column
    y = y_column
    z = z_column
    # Read transcripts
    logger.info("Loading transcripts")
    transcripts = pd.read_csv(transcripts)
    transcripts = transcripts[[cell_column, x, y, z, barcode_column]].rename(columns={cell_column: "cell"})
    transcripts["cell"] = transcripts["cell"].replace(cell_missing, np.nan)
    transcripts = transcripts.round(2)
    # Read ProSeg transcripts
    logger.info("Loading ProSeg transcripts")
    proseg = pd.read_csv(proseg_transcripts)
    proseg = proseg.query("probability > @min_prob")[["observed_x", "observed_y", "observed_z", "assignment"]].rename(
        columns={"observed_x": x, "observed_y": y, "observed_z": z}
    )
    proseg = proseg.round(2)
    # Merge transcripts
    merged = transcripts.merge(proseg, on=[x, y, z], how="left")
    # Get cell mapping
    logger.info("Using transcript assignment to map cells")
    mapping = (
        merged.query("assignment.notna() & cell.notna()")
        .groupby(["cell", "assignment"])
        .size()
        .reset_index(name="count")
    )
    mapping["cell_count"] = mapping.groupby("cell")["count"].transform("sum")
    mapping["assignment_count"] = mapping.groupby("assignment")["count"].transform("sum")
    mapping["jaccard"] = mapping["count"] / (mapping["cell_count"] + mapping["assignment_count"] - mapping["count"])
    mapping = (
        mapping.sort_values("jaccard", ascending=False)
        .groupby("assignment")
        .first()
        .reset_index()
        .groupby("cell")
        .first()
        .reset_index()
    )
    mapping = mapping.query("jaccard >= @min_jaccard").copy()
    mapping.rename(columns={"cell": "mapped_cell"}, inplace=True)
    logger.info(f"{len(mapping)} cells out of {len(merged['cell'].unique())} aligned to ProSeg")
    # Assign transcripts
    logger.info("Using ProSeg to assign additional transcripts to cells")
    merged = merged.merge(mapping[["assignment", "mapped_cell"]], on="assignment", how="left")
    logger.info(f"{(1 - merged.cell.isna().mean()):.2%} of transcripts in cells before")
    merged["cell"] = merged["mapped_cell"].fillna(merged["cell"])
    logger.info(f"{(1 - merged.cell.isna().mean()):.2%} of transcripts in cells after")
    # Update barcodes
    if codebook is not None:
        logger.info("Assigning gene names to barcodes")
        codebook = pd.read_csv(codebook)
        barcode_names = codebook[codebook.columns[0]].to_dict()
        merged[barcode_column] = merged[barcode_column].map(barcode_names)
    # Get counts
    counts = merged.groupby("cell")[barcode_column].value_counts().unstack().fillna(0)
    counts = counts.astype(int)
    counts.index = counts.index.astype(int).values
    counts.columns.name = None
    counts = counts.loc[:, ~counts.columns.str.contains("Blank")]
    # Save counts
    logger.info(f"Saving counts to {output}")
    counts.to_csv(output)
    logger.info("Done")
