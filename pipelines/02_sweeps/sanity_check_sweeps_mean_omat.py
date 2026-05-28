#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/sweeps")
DIAGS = ("CN", "AD", "FTD")
OUT = ROOT / "sanity_check_sweeps_mean_omat.png"
GS = np.round(np.linspace(1, 2.5, 51), 3)
TARGETS = np.round(np.linspace(2, 4, 51), 3)


def load_matrix(csv_path: Path) -> np.ndarray:
    vals = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = round(float(row["G"]), 3)
            t = round(float(row["target"]), 3)
            nums = [float(v) for k, v in row.items() if k not in {"G", "target", "mean_omat"} and v != ""]
            vals[(g, t)].append(float(np.mean(nums)))
    mat = np.full((len(GS), len(TARGETS)), np.nan)
    gi = {g: i for i, g in enumerate(GS)}
    ti = {t: i for i, t in enumerate(TARGETS)}
    for (g, t), arr in vals.items():
        if g in gi and t in ti:
            mat[gi[g], ti[t]] = float(np.mean(arr))
    return mat


def main() -> None:
    mats = [load_matrix(ROOT / f"sweep_G_target_{d}-opti_SC_extend" / "BOLD_omats.csv") for d in DIAGS]
    finite = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
    vmin, vmax = float(np.min(finite)), float(np.max(finite))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, diag, mat in zip(axes, DIAGS, mats):
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=(TARGETS[0], TARGETS[-1], GS[0], GS[-1]),
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(diag)
        ax.set_xlabel("target")
        ax.set_ylabel("G")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.savefig(OUT, dpi=200)
    print(OUT)


if __name__ == "__main__":
    main()
