#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single DMF simulation using the CN structural connectivity stored in raw_SC.npz.

This keeps the model settings aligned with the sweep scripts and only swaps the SC
source. The script runs one parameter set across multiple seeds in parallel and
reports the mean excitatory firing rate.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import time as tm

import numpy as np

WORKSPACE = Path("/home/rherzog/Documents/Brainlat")
DMF_ROOT = WORKSPACE / "optimize_targets_ruben"
SUPPORT_ROOT = WORKSPACE / "repo_test" / "support_scripts"
sys.path[:0] = [
    str(DMF_ROOT / "no_inhibitory"),
    str(DMF_ROOT),
    str(SUPPORT_ROOT),
]

import DMF_ISP_numba as DMF  # noqa: E402


DEFAULT_SC_PATH = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/raw_SC.npz")
DEFAULT_OUT_ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/single_simulation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DMF simulation with CN SC.")
    parser.add_argument("--sc-path", type=Path, default=DEFAULT_SC_PATH, help="Path to raw_SC.npz.")
    parser.add_argument("--sc-key", type=str, default="SC_CN", help="Key inside the SC archive.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output folder.")
    parser.add_argument("--G", type=float, default=2.0, help="Global coupling.")
    parser.add_argument("--target", type=float, default=2.5, help="Target firing rate.")
    parser.add_argument("--tmax", type=int, default=720000, help="Simulation length in ms.")
    parser.add_argument("--dt", type=int, default=1, help="Integration step in ms.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds.")
    parser.add_argument("--cores", type=int, default=20, help="Parallel worker count.")
    parser.add_argument("--seed0", type=int, default=0, help="First seed value.")
    parser.add_argument("--cutfrom", type=int, default=12000, help="Initial points to discard.")
    parser.add_argument("--downsample-rates", type=int, default=10, help="Downsample factor for rates.")
    return parser.parse_args()


def load_sc(sc_path: Path, sc_key: str) -> np.ndarray:
    with np.load(sc_path) as sc_archive:
        if sc_key not in sc_archive.files:
            raise KeyError(f"SC key {sc_key!r} not found in {sc_path}. Available keys: {sc_archive.files}")
        return sc_archive[sc_key]


def configure_dmf(sc: np.ndarray, tmax: int, dt: int, G: float, target: float) -> None:
    np.fill_diagonal(sc, 0)
    DMF.SC = 0.2 * sc.copy()
    DMF.nnodes = len(DMF.SC)
    DMF.update()

    DMF.tmax = tmax
    DMF.dt = dt
    DMF.sigma = 0.01
    DMF.tau_p = 1.5
    DMF.Jdecay = 400000
    DMF.model_1 = 1
    DMF.model_2 = 1
    DMF.G = G
    DMF.target = target


def run_seed(seed: int, cutfrom: int, downsample_rates: int) -> tuple[int, float]:
    DMF.seed = seed
    rates, _ = DMF.Sim()
    rates = rates[cutfrom::downsample_rates, :]
    converged_time = int(100 / 0.001 / downsample_rates)
    mean_rates = rates[-converged_time:, :].mean().mean()
    return seed, float(mean_rates)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sc = load_sc(args.sc_path, args.sc_key)
    configure_dmf(sc, args.tmax, args.dt, args.G, args.target)

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    start = tm()

    results: dict[int, float] = {}
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.cores, mp_context=ctx) as executor:
        futures = {
            executor.submit(run_seed, seed, args.cutfrom, args.downsample_rates): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed, mean_rates = future.result()
            results[seed] = mean_rates
            print(f"seed={seed}\tmean_excitatory_rate={mean_rates:.6f}")

    ordered = np.array([results[seed] for seed in seeds], dtype=np.float64)
    summary_path = args.output_root / f"G={args.G:.3f}_target={args.target:.3f}_summary.npz"
    np.savez_compressed(
        summary_path,
        seeds=np.array(seeds, dtype=np.int32),
        mean_excitatory_rates=ordered.astype(np.float32),
        mean_excitatory_rate=ordered.mean().astype(np.float32),
        std_excitatory_rate=ordered.std().astype(np.float32),
    )

    elapsed = tm() - start
    print(f"overall_mean_excitatory_rate={ordered.mean():.6f}")
    print(f"overall_std_excitatory_rate={ordered.std():.6f}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
