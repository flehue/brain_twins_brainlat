from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/sweeps/raw")
DIAGS = ("CN", "AD", "FTD")
PAIR_COLS = [f"{i}_{j}" for i in range(90) for j in range(i + 1, 90)]
TRIU = np.triu_indices(90, k=1)
CELL_RE = re.compile(
    r"G=(?P<G>[0-9]+(?:\.[0-9]+)?)_target=(?P<target>[0-9]+(?:\.[0-9]+)?)_seed=(?P<seed>\d+)_all\.npz$"
)


def read_cells(diag: str) -> list[tuple[float, float]]:
    chunk_dir = ROOT / f"sweep_G_target_{diag}-opti_SC_extend" / "chunks"
    chunks = sorted(chunk_dir.glob("chunk_*.csv"))
    grid = pd.concat((pd.read_csv(path, usecols=["G", "target"]) for path in chunks), ignore_index=True)
    grid = grid.drop_duplicates(["G", "target"]).sort_values(["G", "target"]).reset_index(drop=True)
    return [(float(g), float(t)) for g, t in grid[["G", "target"]].itertuples(index=False, name=None)]


def build_cell_groups(diag: str, cells: list[tuple[float, float]]) -> dict[tuple[float, float], list[Path]]:
    sim_dir = ROOT / f"sweep_G_target_{diag}-opti_SC_extend" / "all_per_seed"
    allowed = set(cells)
    groups: dict[tuple[float, float], list[Path]] = defaultdict(list)
    for path in sim_dir.glob("seed_*/G=*_target=*_seed=*_all.npz"):
        match = CELL_RE.search(path.name)
        if not match:
            continue
        cell = (float(match.group("G")), float(match.group("target")))
        if cell in allowed:
            groups[cell].append(path)
    missing = [cell for cell in cells if cell not in groups]
    if missing:
        raise FileNotFoundError(f"{diag}: missing {len(missing)} cells, example {missing[:5]}")
    return groups


def build_row(cell: tuple[float, float], files: list[Path]) -> tuple[float, float, float, np.ndarray]:
    mats = []
    for path in files:
        with np.load(path) as z:
            mat = z["omat_BOLD"].astype(np.float32, copy=False)
        mats.append(np.where(np.isfinite(mat), mat, np.nan))
    mean_mat = np.nanmean(np.stack(mats, axis=0), axis=0)
    mean_mat = np.where(np.isfinite(mean_mat), mean_mat, np.nan)
    flat = mean_mat[TRIU].astype(np.float32, copy=False)
    finite = np.isfinite(flat)
    if finite.any():
        fill = float(np.nanmean(flat[finite]))
        flat = np.where(finite, flat, fill).astype(np.float32, copy=False)
    else:
        flat = np.zeros_like(flat, dtype=np.float32)
    mean_omat = float(np.nanmean(mean_mat)) if np.isfinite(mean_mat).any() else np.nan
    return cell[0], cell[1], mean_omat, flat


def write_csv(diag: str) -> None:
    cells = read_cells(diag)
    groups = build_cell_groups(diag, cells)
    rows = Parallel(n_jobs=30, prefer="threads")(
        delayed(build_row)(cell, groups[cell]) for cell in cells
    )

    data = {
        "G": [r[0] for r in rows],
        "target": [r[1] for r in rows],
        "mean_omat": [r[2] for r in rows],
    }
    flat = np.stack([r[3] for r in rows], axis=0)
    for idx, col in enumerate(PAIR_COLS):
        data[col] = flat[:, idx]

    out = ROOT / f"sweep_G_target_{diag}-opti_SC_extend" / "BOLD_omats.csv"
    pd.DataFrame(data, columns=["G", "target", "mean_omat", *PAIR_COLS]).to_csv(out, index=False)


def main() -> None:
    for diag in DIAGS:
        write_csv(diag)


if __name__ == "__main__":
    main()
