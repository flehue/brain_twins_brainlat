"""Find the best raw-SC BOLD_omat match for each global vector group."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
INPUT_ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input")
SWEEP_ROOT = INPUT_ROOT / "sweeps" / "raw"
EMPIRICAL_GROUPS = INPUT_ROOT / "average_oinfo_matrices_by_group.csv"
TRIU = np.triu_indices(90, k=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match global vectors against raw-SC BOLD_omat grids."
    )
    parser.add_argument(
        "--input-csv",
        default=str(EMPIRICAL_GROUPS),
        help="Path to average_oinfo_matrices_by_group.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(SCRIPT_DIR / "optimals_BOLD_corr_by_group_rawSC.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_group_key(key: str) -> tuple[str, str, str, str]:
    dataset, diagnosis, sex, age = key.rsplit("_", 3)
    return dataset, diagnosis, sex, age


def sweep_diagnosis(diagnosis: str) -> str:
    return "AD" if diagnosis == "MCI" else diagnosis


def load_empirical_groups(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    df = pd.read_csv(path)
    link_cols = [col for col in df.columns if col.startswith("link_")]
    keys: list[str] = []
    vectors: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        dataset = str(row["Dataset"]).strip()
        diagnosis = str(row["Diagnosis"]).strip().upper()
        sex = str(row["Sex"]).strip()
        age = str(row["Age_Range"]).strip()
        key = f"{dataset}_{diagnosis}_{sex}_{age}"
        keys.append(key)
        vectors[key] = row[link_cols].to_numpy(dtype=np.float64)
    return keys, vectors


def load_sim_cache(diagnosis: str) -> dict[str, np.ndarray]:
    path = SWEEP_ROOT / f"sweep_G_target_{sweep_diagnosis(diagnosis)}-opti_SC_extend" / "BOLD_omats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw sweep grid: {path}")

    gs = []
    targets = []
    flats = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty BOLD_omats.csv: {path}")
        for row in reader:
            if not row:
                continue
            gs.append(float(row[0]))
            targets.append(float(row[1]))
            flats.append(np.asarray(row[3:], dtype=np.float64))

    sim_flat = np.stack(flats, axis=0)
    sim_centered = sim_flat - sim_flat.mean(axis=1, keepdims=True)
    sim_norms = np.linalg.norm(sim_centered, axis=1)
    return {
        "G": np.asarray(gs, dtype=np.float64),
        "target": np.asarray(targets, dtype=np.float64),
        "centered": sim_centered,
        "norms": sim_norms,
    }


def best_match(emp_flat: np.ndarray, sim_cache: dict[str, np.ndarray]) -> tuple[float, float, float]:
    emp_centered = emp_flat - emp_flat.mean()
    emp_norm = np.linalg.norm(emp_centered)
    if not np.isfinite(emp_norm) or emp_norm == 0:
        return np.nan, np.nan, np.nan

    numerators = sim_cache["centered"] @ emp_centered
    denom = sim_cache["norms"] * emp_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = numerators / denom
    corrs[~np.isfinite(corrs)] = -np.inf
    idx = int(np.argmax(corrs))
    if not np.isfinite(corrs[idx]):
        return np.nan, np.nan, np.nan
    return float(corrs[idx]), float(sim_cache["G"][idx]), float(sim_cache["target"][idx])


def main() -> None:
    args = parse_args()

    keys, empirical_groups = load_empirical_groups(Path(args.input_csv))
    sim_cache_by_diag: dict[str, dict[str, np.ndarray]] = {}
    rows = []

    for key in keys:
        dataset, diagnosis, sex, age = parse_group_key(key)
        diag_for_sweep = sweep_diagnosis(diagnosis)
        if diag_for_sweep not in sim_cache_by_diag:
            sim_cache_by_diag[diag_for_sweep] = load_sim_cache(diag_for_sweep)

        emp_flat = np.asarray(empirical_groups[key], dtype=np.float64).reshape(-1)
        gof, g_opt, t_opt = best_match(emp_flat, sim_cache_by_diag[diag_for_sweep])
        rows.append((dataset, diagnosis, sex, age, gof, g_opt, t_opt))

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    with Path(args.output_csv).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Dataset", "Diagnosis", "Sex", "Age_Range", "GoF_BOLD_omat", "G_BOLD_omat", "target_BOLD_omat"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
