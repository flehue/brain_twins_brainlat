"""
Find per-subject optimal G and target for the raw SC sweeps using PABLO omats.

The script compares each empirical subject O-matrix against the mean simulated
O-matrix for every (G, target) cell in the raw sweep grids.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input")
SWEEP_ROOT = INPUT_ROOT / "sweeps" / "raw"
EMP_OMATS = INPUT_ROOT / "PABLO_omats.npz"
EMP_DEMO = INPUT_ROOT / "PABLO_demo.csv"
TRIU = np.triu_indices(90, k=1)
CELL_RE = re.compile(
    r"G=(?P<G>[0-9]+(?:\.[0-9]+)?)_target=(?P<target>[0-9]+(?:\.[0-9]+)?)_seed=(?P<seed>\d+)_all\.npz$"
)

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize empirical PABLO omats against raw sweep mean omats."
    )
    parser.add_argument(
        "--diagnosis",
        default="ALL",
        choices=["ALL", "CN", "AD", "FTD", "MCI"],
        help="Which diagnosis to process. MCI uses the AD raw sweep.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=30,
        help="Parallel jobs for building simulated cell means.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(SCRIPT_DIR / "optimals_BOLD_corr_rawSC.csv"),
        help="Output TSV/CSV path for the optimals table.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional limit for a smoke test.",
    )
    parser.add_argument(
        "--max-target",
        type=float,
        default=None,
        help="Optional upper bound on target/rho used to restrict the sweep grid.",
    )
    return parser.parse_args()


def get_sim_diag(diagnosis: str) -> str:
    return "AD" if diagnosis == "MCI" else diagnosis


def get_sim_folder(sim_diag: str) -> Path:
    return SWEEP_ROOT / f"sweep_G_target_{sim_diag}-opti_SC_extend"


def read_expected_cells(sim_folder: Path, max_target: float | None = None) -> list[tuple[float, float]]:
    chunk_files = sorted((sim_folder / "chunks").glob("chunk_*.csv"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk CSVs found in {sim_folder / 'chunks'}")

    grid = pd.concat((pd.read_csv(path, usecols=["G", "target"]) for path in chunk_files), ignore_index=True)
    if max_target is not None:
        grid = grid[grid["target"] <= max_target].copy()
    grid = grid.drop_duplicates(["G", "target"]).sort_values(["G", "target"]).reset_index(drop=True)
    cells = [(float(g), float(t)) for g, t in grid[["G", "target"]].itertuples(index=False, name=None)]
    if not cells:
        raise ValueError(f"No G/target cells found in {sim_folder}")
    return cells


def build_sim_cache(sim_diag: str, n_jobs: int, max_target: float | None = None) -> dict[str, np.ndarray]:
    sim_folder = get_sim_folder(sim_diag)
    expected_cells = read_expected_cells(sim_folder, max_target=max_target)
    expected_set = set(expected_cells)

    cell_groups: dict[tuple[float, float], list[Path]] = defaultdict(list)
    for path in (sim_folder / "all_per_seed").glob("seed_*/G=*_target=*_seed=*_all.npz"):
        match = CELL_RE.search(path.name)
        if not match:
            continue
        cell = (float(match.group("G")), float(match.group("target")))
        if cell in expected_set:
            cell_groups[cell].append(path)

    missing = [cell for cell in expected_cells if cell not in cell_groups]
    if missing:
        raise FileNotFoundError(
            f"Missing simulated files for {sim_diag}: {len(missing)} cells, for example {missing[:5]}"
        )

    def mean_flat(cell: tuple[float, float]) -> tuple[float, float, np.ndarray]:
        files = cell_groups[cell]
        mats = []
        for path in files:
            with np.load(path) as z:
                mat = z["omat_BOLD"].astype(np.float32, copy=False)
            mats.append(np.where(np.isfinite(mat), mat, np.nan))
        stack = np.stack(mats, axis=0)
        mean_mat = np.nanmean(stack, axis=0)
        flat = mean_mat[TRIU].astype(np.float32, copy=False)
        finite = np.isfinite(flat)
        if finite.any():
            fill = float(np.nanmean(flat[finite]))
            flat = np.where(finite, flat, fill).astype(np.float32, copy=False)
        else:
            flat = np.zeros_like(flat, dtype=np.float32)
        return cell[0], cell[1], flat

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(mean_flat)(cell) for cell in expected_cells
    )

    gs = np.array([r[0] for r in results], dtype=np.float32)
    targets = np.array([r[1] for r in results], dtype=np.float32)
    sim_flat = np.stack([r[2] for r in results], axis=0).astype(np.float32, copy=False)
    sim_centered = sim_flat - sim_flat.mean(axis=1, keepdims=True)
    sim_norms = np.linalg.norm(sim_centered, axis=1)
    return {"G": gs, "target": targets, "flat": sim_flat, "centered": sim_centered, "norms": sim_norms}


def optimize_subjects(
    subjects: pd.DataFrame,
    sim_cache: dict[str, np.ndarray],
    emp_omats: np.lib.npyio.NpzFile,
    n_jobs: int,
) -> pd.DataFrame:
    nmega = subjects["N_MEGA"].astype(int).to_numpy()
    diag = subjects["Diagnosis"].astype(str).to_numpy()
    sweep_diag = subjects["Sweep_Diagnosis"].astype(str).to_numpy()

    emp_flat = np.stack([np.asarray(emp_omats[str(int(n))], dtype=np.float32).reshape(-1) for n in nmega], axis=0)
    emp_centered = emp_flat - emp_flat.mean(axis=1, keepdims=True)
    emp_norms = np.linalg.norm(emp_centered, axis=1)
    good_emp = np.isfinite(emp_norms) & (emp_norms > 0)

    best_idx = np.full(emp_flat.shape[0], -1, dtype=int)
    best_gof = np.full(emp_flat.shape[0], np.nan, dtype=np.float32)

    if good_emp.any():
        sim_centered = sim_cache["centered"]
        sim_norms = sim_cache["norms"]
        with threadpool_limits(limits=n_jobs) if threadpool_limits is not None else contextlib.nullcontext():
            numerators = sim_centered @ emp_centered[good_emp].T
        denom = sim_norms[:, None] * emp_norms[good_emp][None, :]
        scores = np.divide(
            numerators,
            denom,
            out=np.full_like(numerators, -np.inf, dtype=np.float32),
            where=denom > 0,
        )
        scores[~np.isfinite(scores)] = -np.inf
        local_best = np.argmax(scores, axis=0)
        best_idx[good_emp] = local_best
        best_gof[good_emp] = scores[local_best, np.arange(scores.shape[1])]

    out = pd.DataFrame(
        {
            "N_MEGA": nmega,
            "Diagnosis": diag,
            "Sweep_Diagnosis": sweep_diag,
            "GoF_BOLD_omat": best_gof,
        }
    )
    g_out = np.full(emp_flat.shape[0], np.nan, dtype=np.float32)
    t_out = np.full(emp_flat.shape[0], np.nan, dtype=np.float32)
    good_out = best_idx >= 0
    if np.any(good_out):
        g_out[good_out] = sim_cache["G"][best_idx[good_out]]
        t_out[good_out] = sim_cache["target"][best_idx[good_out]]
    out["G_BOLD_omat"] = g_out
    out["target_BOLD_omat"] = t_out
    return out


def main() -> None:
    args = parse_args()
    demo = pd.read_csv(EMP_DEMO).drop_duplicates("N_MEGA")
    if args.diagnosis != "ALL":
        demo = demo[demo["Diagnosis"].astype(str) == args.diagnosis].copy()
    if demo.empty:
        raise ValueError("No empirical subjects selected.")

    demo["Sweep_Diagnosis"] = demo["Diagnosis"].astype("string").replace({"MCI": "AD"})
    demo = demo.sort_values(["Sweep_Diagnosis", "N_MEGA"]).reset_index(drop=True)
    if args.max_subjects is not None:
        demo = demo.iloc[: args.max_subjects].copy()

    all_results = []
    with np.load(EMP_OMATS, allow_pickle=True) as emp_omats:
        for sim_diag in [d for d in ["CN", "AD", "FTD"] if (demo["Sweep_Diagnosis"] == d).any()]:
            sim_cache = build_sim_cache(sim_diag, args.n_jobs, max_target=args.max_target)
            group = demo[demo["Sweep_Diagnosis"] == sim_diag].reset_index(drop=True)
            all_results.append(optimize_subjects(group, sim_cache, emp_omats, args.n_jobs))

        missing = demo[~demo["Sweep_Diagnosis"].isin(["CN", "AD", "FTD"])].copy()
        if not missing.empty:
            missing_out = missing[["N_MEGA", "Diagnosis", "Sweep_Diagnosis"]].copy()
            missing_out["GoF_BOLD_omat"] = np.nan
            missing_out["G_BOLD_omat"] = np.nan
            missing_out["target_BOLD_omat"] = np.nan
            all_results.append(missing_out)

    result = pd.concat(all_results, ignore_index=True)
    result.to_csv(args.output_csv, index=False, sep="\t")


if __name__ == "__main__":
    main()
