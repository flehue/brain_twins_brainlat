from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from calculate_BOLD_optimals_rawSC import build_sim_cache, optimize_subjects

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input")
RAW_OPT = SCRIPT_DIR / "optimals_BOLD_corr_rawSC.csv"
MODEL = SCRIPT_DIR.parents[1] / "data" / "derived" / "model_output_plus_exposome_data_v3.csv"
ALL_OMATS = INPUT_ROOT / "all_Omats.npz"


def main() -> None:
    raw = pd.read_csv(RAW_OPT, sep="\t")
    model = pd.read_csv(MODEL)
    missing = model[~model["N_MEGA"].astype(str).isin(raw["N_MEGA"].astype(str))].copy()
    if missing.empty:
        return

    missing["Sweep_Diagnosis"] = missing["Diagnosis"].astype("string").replace({"MCI": "AD"})
    missing = missing.sort_values(["Sweep_Diagnosis", "N_MEGA"]).reset_index(drop=True)

    out_rows = []
    with np.load(ALL_OMATS, allow_pickle=True) as emp_omats:
        for sim_diag in [d for d in ["CN", "AD", "FTD"] if (missing["Sweep_Diagnosis"] == d).any()]:
            sim_cache = build_sim_cache(sim_diag, n_jobs=30, max_target=None)
            group = missing[missing["Sweep_Diagnosis"] == sim_diag].reset_index(drop=True)
            out_rows.append(optimize_subjects(group, sim_cache, emp_omats, n_jobs=30))

    if out_rows:
        updated = pd.concat([raw, *out_rows], ignore_index=True)
        updated = updated.sort_values(["Diagnosis", "N_MEGA"], na_position="last").reset_index(drop=True)
        updated.to_csv(RAW_OPT, sep="\t", index=False)


if __name__ == "__main__":
    main()
