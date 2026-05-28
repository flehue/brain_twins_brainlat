#!/usr/bin/env python3
"""Chunked BOLD sweep for CN using a plain Python entrypoint.

This script intentionally does not read any SLURM environment variables.
The caller passes a chunk index and the total number of chunks on the command
line, and each process handles one disjoint slice of the full G x target x seed
grid serially on a single core.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path
from time import time as tm

import numpy as np
from scipy import signal

WORKSPACE = Path("/home/rherzog/Documents/Brainlat")
DMF_ROOT = WORKSPACE / "optimize_targets_ruben"
SUPPORT_ROOT = WORKSPACE / "repo_test" / "support_scripts"
sys.path[:0] = [
    str(DMF_ROOT / "no_inhibitory"),
    str(DMF_ROOT),
    str(SUPPORT_ROOT),
]

import DMF_ISP_numba as DMF  # noqa: E402
import BOLDModel as BD  # noqa: E402
import calculate_omat  # noqa: E402


DEFAULT_SC_PATH = Path(
    "/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/optimized_SC.npz"
)
DEFAULT_OUT_ROOT = Path(
    "/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/sweeps_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one chunk of the CN BOLD sweep.")
    parser.add_argument("--chunk-index", type=int, required=True, help="0-based chunk index.")
    parser.add_argument("--num-chunks", type=int, required=True, help="Total number of chunks.")
    parser.add_argument("--diagnosis", type=str, default="CN", help="Diagnosis label, default: CN.")
    parser.add_argument("--sc-path", type=Path, default=DEFAULT_SC_PATH, help="Path to optimized_SC.npz.")
    parser.add_argument("--sc-key", type=str, default="SC_CN", help="Key inside the SC archive.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Base output root for chunk files and per-seed artifacts.",
    )
    return parser.parse_args()


def load_sc(sc_path: Path, sc_key: str) -> np.ndarray:
    with np.load(sc_path) as sc_archive:
        if sc_key not in sc_archive.files:
            raise KeyError(f"SC key {sc_key!r} not found in {sc_path}. Available keys: {sc_archive.files}")
        return sc_archive[sc_key]


def build_grid() -> list[tuple[float, float, int]]:
    Gs = np.concatenate([np.round(np.arange(0.5, 1.0, 0.06), 3), np.linspace(1, 4, 51, endpoint=True)])
    targets = np.linspace(2, 6, 51, endpoint=True)
    seeds = range(25)
    return list(itertools.product(Gs, targets, seeds))


def split_grid(grid: list[tuple[float, float, int]], num_chunks: int) -> list[np.ndarray]:
    indices = np.arange(len(grid))
    return list(np.array_split(indices, num_chunks))


def run_iter(G: float, target: float, seed: int, output_folder: Path) -> tuple[float, float, float, float]:
    start = tm()
    DMF.seed = seed
    DMF.target = target
    DMF.G = G

    rates, _ = DMF.Sim()
    rates = rates[12000::10, :]

    dtt = 0.01
    bold_signals = BD.Sim(rates, DMF.nnodes, dtt)
    bold_signals = bold_signals[12000:, :][::100, :]

    a0, b0 = signal.bessel(3, 2 * 1 * np.array([0.01, 0.1]), btype="bandpass")
    bold_filt = signal.filtfilt(a0, b0, bold_signals, axis=0)

    (output_folder / "BOLD_per_seed" / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_folder / "BOLD_per_seed" / f"seed_{seed}" / f"G={G:.3f}_target={target:.3f}_seed={seed}_BOLD"),
        bold_filt.astype(np.float32),
    )

    fc_bold = np.corrcoef(bold_filt.T)
    omat_bold = calculate_omat.multi_fc(bold_filt)
    mean_fc, mean_omat = fc_bold.mean(), omat_bold.mean()

    converged_time = int(100 / 0.001 / 10)
    mean_rates = rates[-converged_time:, :].mean().mean()

    (output_folder / "all_per_seed" / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_folder / "all_per_seed" / f"seed_{seed}" / f"G={G:.3f}_target={target:.3f}_seed={seed}_all"),
        FC_BOLD=fc_bold.astype(np.float32),
        omat_BOLD=omat_bold.astype(np.float32),
        mean_rates=mean_rates.astype(np.float32),
    )

    return float(tm() - start), float(mean_rates), float(mean_fc), float(mean_omat)


def iter_outputs_exist(G: float, target: float, seed: int, output_folder: Path) -> bool:
    bold_path = output_folder / "BOLD_per_seed" / f"seed_{seed}" / f"G={G:.3f}_target={target:.3f}_seed={seed}_BOLD.npz"
    all_path = output_folder / "all_per_seed" / f"seed_{seed}" / f"G={G:.3f}_target={target:.3f}_seed={seed}_all.npz"
    return bold_path.exists() and all_path.exists()


def main() -> None:
    args = parse_args()
    if args.chunk_index < 0 or args.num_chunks <= 0:
        raise ValueError("chunk-index must be >= 0 and num-chunks must be > 0")
    if args.chunk_index >= args.num_chunks:
        raise ValueError("chunk-index must be smaller than num-chunks")

    diagnosis = args.diagnosis
    sc_key = args.sc_key or f"SC_{diagnosis}"
    output_folder = args.output_root / f"sweep_G_target_{diagnosis}-opti_SC_extend"
    chunk_dir = output_folder / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    sc = load_sc(args.sc_path, sc_key)
    np.fill_diagonal(sc, 0)
    DMF.SC = 0.2 * sc.copy()
    DMF.nnodes = len(DMF.SC)
    DMF.update()

    DMF.tmax = 720000
    DMF.dt = 1
    DMF.sigma = 0.01
    DMF.tau_p = 1.5
    DMF.Jdecay = 400000
    DMF.model_1 = 1
    DMF.model_2 = 1

    grid = build_grid()
    chunks = split_grid(grid, args.num_chunks)
    my_indices = chunks[args.chunk_index]

    out_csv = chunk_dir / f"chunk_{args.chunk_index:03d}.csv"
    csv_exists = out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["G", "target", "seed", "time", "mean_rates", "mean_FC", "mean_omat"],
        )
        if not csv_exists:
            writer.writeheader()

        for idx in my_indices:
            G, target, seed = grid[int(idx)]
            if iter_outputs_exist(G, target, seed, output_folder):
                print(f"SKIP existing output G={G:.3f} target={target:.3f} seed={seed}", flush=True)
                continue
            time, mean_rates, mean_fc, mean_omat = run_iter(G, target, seed, output_folder)
            writer.writerow(
                {
                    "G": f"{G:.3f}",
                    "target": f"{target:.3f}",
                    "seed": seed,
                    "time": f"{time:.3f}",
                    "mean_rates": f"{mean_rates:.3f}",
                    "mean_FC": f"{mean_fc:.3f}",
                    "mean_omat": mean_omat,
                }
            )


if __name__ == "__main__":
    main()
