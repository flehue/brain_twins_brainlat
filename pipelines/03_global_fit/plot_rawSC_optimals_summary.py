from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "optimals_BOLD_corr_rawSC.csv"
DEFAULT_OUT_PATH = SCRIPT_DIR / "rawSC_optimals_summary.png"

DIAG_ORDER = ["CN", "AD", "FTD", "MCI"]
DIAG_COLORS = {
    "CN": "#4C78A8",
    "AD": "#F58518",
    "FTD": "#54A24B",
    "MCI": "#E45756",
}
DIAG_MARKERS = {
    "CN": "o",
    "AD": "s",
    "FTD": "^",
    "MCI": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rawSC/global-fit optimals summary.")
    parser.add_argument("--input-csv", default=str(DEFAULT_CSV_PATH), help="Input optimals CSV/TSV.")
    parser.add_argument("--output-png", default=str(DEFAULT_OUT_PATH), help="Output figure path.")
    return parser.parse_args()


def finite(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    return arr[np.isfinite(arr)]


def plot_distribution(ax, df: pd.DataFrame, column: str, title: str, xlabel: str) -> None:
    all_vals = finite(df[column])
    if all_vals.size == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        return

    bins = np.histogram_bin_edges(all_vals, bins=35)
    ax.hist(
        all_vals,
        bins=bins,
        density=True,
        histtype="step",
        color="black",
        linewidth=2.0,
        label="Global",
    )

    for diag in DIAG_ORDER:
        sub = df[df["Diagnosis"] == diag]
        vals = finite(sub[column])
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=DIAG_COLORS[diag],
            label=diag,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)


def plot_scatter(ax, df: pd.DataFrame) -> None:
    plot_df = df[["Diagnosis", "G_BOLD_omat", "target_BOLD_omat", "GoF_BOLD_omat"]].copy()
    plot_df = plot_df.dropna(subset=["G_BOLD_omat", "target_BOLD_omat", "GoF_BOLD_omat", "Diagnosis"])
    if plot_df.empty:
        ax.set_title("G vs target_BOLD_omat")
        ax.set_xlabel("G_BOLD_omat")
        ax.set_ylabel("target_BOLD_omat")
        return

    vmin = float(plot_df["GoF_BOLD_omat"].min())
    vmax = float(plot_df["GoF_BOLD_omat"].max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    for diag in DIAG_ORDER:
        sub = plot_df[plot_df["Diagnosis"] == diag]
        if sub.empty:
            continue
        sc = ax.scatter(
            sub["G_BOLD_omat"],
            sub["target_BOLD_omat"],
            c=sub["GoF_BOLD_omat"],
            cmap=cmap,
            norm=norm,
            marker=DIAG_MARKERS[diag],
            s=18,
            alpha=0.8,
            linewidths=0,
            label=diag,
        )

    ax.set_title("G vs target_BOLD_omat")
    ax.set_xlabel("G_BOLD_omat")
    ax.set_ylabel("target_BOLD_omat")
    ax.legend(title="Diagnosis", frameon=False, fontsize=8)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("GoF_BOLD_omat")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv, sep=None, engine="python")
    if "Diagnosis" not in df.columns and "diagnosis" in df.columns:
        df = df.rename(columns={"diagnosis": "Diagnosis"})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    plot_distribution(axes[0], df, "GoF_BOLD_omat", "GOF distribution", "GoF_BOLD_omat")
    plot_distribution(axes[1], df, "G_BOLD_omat", "Optimal G distribution", "G_BOLD_omat")
    plot_distribution(
        axes[2],
        df,
        "target_BOLD_omat",
        "Optimal target distribution",
        "target_BOLD_omat",
    )
    plot_scatter(axes[3], df)

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
