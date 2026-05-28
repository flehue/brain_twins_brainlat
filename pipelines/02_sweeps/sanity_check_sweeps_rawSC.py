#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/sweeps/raw")
DEFAULT_OUT = DEFAULT_ROOT / "sanity_check_sweeps_rawSC.png"
DIAGS = ("CN", "AD", "FTD")
GS = np.concatenate([np.round(np.arange(0.5, 1.0, 0.06), 3), np.round(np.linspace(1, 4, 51), 3)])
TARGETS = np.round(np.linspace(2, 6, 51), 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_matrix(root: Path, diag: str, metric: str) -> np.ndarray:
    chunk_dir = root / f"sweep_G_target_{diag}-opti_SC_extend" / "chunks"
    vals = defaultdict(list)
    for csv_path in sorted(chunk_dir.glob("chunk_*.csv")):
        if csv_path.stat().st_size == 0:
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or metric not in reader.fieldnames:
                continue
            for row in reader:
                g = round(float(row["G"]), 3)
                t = round(float(row["target"]), 3)
                v = row.get(metric, "")
                if v:
                    vals[(g, t)].append(float(v))

    mat = np.full((len(GS), len(TARGETS)), np.nan)
    gi = {g: i for i, g in enumerate(GS)}
    ti = {t: i for i, t in enumerate(TARGETS)}
    for (g, t), arr in vals.items():
        if g in gi and t in ti:
            arr = np.asarray(arr, dtype=float)
            mat[gi[g], ti[t]] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
    return mat


def main() -> None:
    args = parse_args()
    omat = {d: load_matrix(args.root, d, "mean_omat") for d in DIAGS}
    fc = {d: load_matrix(args.root, d, "mean_FC") for d in DIAGS}

    omat_vals = np.concatenate([m[np.isfinite(m)] for m in omat.values() if np.isfinite(m).any()])
    fc_vals = np.concatenate([m[np.isfinite(m)] for m in fc.values() if np.isfinite(m).any()])
    omat_vmin, omat_vmax = (float(np.min(omat_vals)), float(np.max(omat_vals))) if omat_vals.size else (0.0, 1.0)
    fc_vmin, fc_vmax = (float(np.min(fc_vals)), float(np.max(fc_vals))) if fc_vals.size else (0.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    rows = [("mean_omat", omat, "magma", omat_vmin, omat_vmax), ("mean_FC", fc, "viridis", fc_vmin, fc_vmax)]

    for r, (label, mats, cmap, vmin, vmax) in enumerate(rows):
        for c, diag in enumerate(DIAGS):
            ax = axes[r, c]
            im = ax.imshow(
                mats[diag],
                origin="lower",
                aspect="auto",
                extent=(TARGETS[0], TARGETS[-1], GS[0], GS[-1]),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(diag)
            ax.set_xlabel("target" if r == 1 else "")
            ax.set_ylabel("G" if c == 0 else "")
            if c == 0:
                ax.text(-0.28, 0.5, label, rotation=90, va="center", ha="center", transform=ax.transAxes)
        fig.colorbar(im, ax=axes[r, :], fraction=0.025, pad=0.02)

    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
