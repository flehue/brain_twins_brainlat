#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/sweeps/raw")
DIAGS = ("CN", "AD", "FTD")
FILE_RE = re.compile(r"G=(?P<G>[0-9.]+)_target=(?P<target>[0-9.]+)_seed=(?P<seed>\d+)_(?P<kind>BOLD|all)\.npz$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--diagnosis", choices=DIAGS, default="CN")
    return p.parse_args()


def build_grid() -> list[tuple[float, float, int]]:
    gs = np.linspace(1, 4, 51, endpoint=True)
    targets = np.linspace(2, 6, 51, endpoint=True)
    seeds = range(25)
    return list(itertools.product(gs, targets, seeds))


def split_grid(grid: list[tuple[float, float, int]], num_chunks: int) -> list[np.ndarray]:
    return list(np.array_split(np.arange(len(grid)), num_chunks))


def load_existing_keys(chunk_path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not chunk_path.exists() or chunk_path.stat().st_size == 0:
        return keys
    with chunk_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row.get("G")
            target = row.get("target")
            seed = row.get("seed")
            if g and target and seed:
                keys.add((g, target, seed))
    return keys


def main() -> None:
    args = parse_args()
    out_root = args.root / f"sweep_G_target_{args.diagnosis}-opti_SC_extend"
    chunk_dir = out_root / "chunks"
    all_root = out_root / "all_per_seed"

    grid = build_grid()
    chunks = split_grid(grid, 100)
    total = 0
    for chunk_idx, indices in enumerate(chunks):
        rows: list[dict[str, str]] = []
        for idx in indices:
            g, target, seed = grid[int(idx)]
            g_s = f"{g:.3f}"
            t_s = f"{target:.3f}"
            seed_s = str(seed)
            all_path = all_root / f"seed_{seed_s}" / f"G={g_s}_target={t_s}_seed={seed_s}_all.npz"
            if not all_path.exists():
                continue
            with np.load(all_path) as data:
                mean_rates = float(data["mean_rates"])
                mean_fc = float(np.asarray(data["FC_BOLD"]).mean())
                mean_omat = float(np.asarray(data["omat_BOLD"]).mean())
            rows.append(
                {
                    "G": g_s,
                    "target": t_s,
                    "seed": seed_s,
                    "time": "nan",
                    "mean_rates": f"{mean_rates:.3f}",
                    "mean_FC": f"{mean_fc:.3f}",
                    "mean_omat": f"{mean_omat}",
                }
            )
        chunk_path = chunk_dir / f"chunk_{chunk_idx:03d}.csv"
        with chunk_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["G", "target", "seed", "time", "mean_rates", "mean_FC", "mean_omat"],
            )
            writer.writeheader()
            writer.writerows(rows)
        total += len(rows)

    print(f"rewrote {total} rows for {args.diagnosis}")


if __name__ == "__main__":
    main()
