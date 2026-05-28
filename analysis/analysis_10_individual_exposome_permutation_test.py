#!/usr/bin/env python
# coding: utf-8
"""Permutation testing for the individual_exposome PCEV pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

import pcev_feature_effects as pfe
import analysis_09_individual_exposome as base_analysis

RESULTS_BASE = Path(os.environ.get("PCEV_RESULTS_DIR", REPO_ROOT / "analysis" / "results" / "pcev_results"))
RESULTS_DIR = RESULTS_BASE / "permutation_tests" / "individual_exposome_no_odq"

N_REPEATS = int(os.environ.get("N_REPEATS", 500))
N_PERM = int(os.environ.get("N_PERM", 500))
N_SPLITS = int(os.environ.get("N_SPLITS", 5))
SEED = int(os.environ.get("SEED", 2025))
N_JOBS = int(os.environ.get("N_JOBS", -1))
PERM_N_JOBS = int(os.environ.get("PERM_N_JOBS", 1))
PROGRESS = PERM_N_JOBS == 1

EXPOSURE_NAMES = list(base_analysis.SVI_VARS.values())


def _ts(message: str) -> None:
    print(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _best_combo_from_summary(summary_path: Path, all_combos: List[pfe.FeatureCombo]) -> pfe.FeatureCombo:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary = pd.read_csv(summary_path)
    if "combo_label" not in summary.columns or "h2_with_mean" not in summary.columns:
        raise ValueError(f"Unexpected summary format: {summary_path}")

    best_label = summary.loc[summary["h2_with_mean"].idxmax(), "combo_label"]
    try:
        return next(combo for combo in all_combos if combo.label == best_label)
    except StopIteration as exc:
        raise ValueError(f"Best combo '{best_label}' not found in feature combinations") from exc


def _prepare_exposure_df(df_raw: pd.DataFrame, all_neural_cols: Sequence[str], exposure_name: str) -> pd.DataFrame:
    required = list(all_neural_cols) + [
        base_analysis.ID_COL,
        exposure_name,
        base_analysis.AGE_COL,
        base_analysis.SEX_COL,
        base_analysis.DIAG_COL,
        base_analysis.COUNTRY_COL,
        base_analysis.GOF_COL,
    ]
    df = df_raw[required].dropna().reset_index(drop=True)
    df[base_analysis.SEX_COL] = df[base_analysis.SEX_COL].astype("category")
    df[base_analysis.DIAG_COL] = df[base_analysis.DIAG_COL].astype("category")
    df[base_analysis.COUNTRY_COL] = df[base_analysis.COUNTRY_COL].astype("category")
    return df


def _save_observed_h2(path: Path, observed_repeats: np.ndarray) -> None:
    pd.DataFrame({"repeat": np.arange(len(observed_repeats)), "h2_with": observed_repeats}).to_csv(path, index=False)


def _save_null_h2(path: Path, perm_means: np.ndarray) -> None:
    pd.DataFrame({"perm_id": np.arange(len(perm_means)), "h2_mean": perm_means}).to_csv(path, index=False)


def _write_summary(path: Path, row: Dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(path, index=False)


def _run_individual_exposome_analysis(
    df_raw: pd.DataFrame,
    all_neural_cols: Sequence[str],
    all_combos: List[pfe.FeatureCombo],
    exposure_name: str,
) -> Dict[str, object]:
    analysis_name = f"individual_exposome_{_safe_name(exposure_name)}"
    summary_path = base_analysis.RESULTS_DIR / _safe_name(exposure_name) / "all_combos_summary.csv"
    combo = _best_combo_from_summary(summary_path, all_combos)
    df = _prepare_exposure_df(df_raw, all_neural_cols, exposure_name)

    _ts(f"Running individual exposome permutation test for '{exposure_name}' on combo '{combo.label}'")
    results = pfe.run_repeated_cv_with_permutations(
        df,
        feature_combo=combo,
        x_cols=[exposure_name],
        id_col=base_analysis.ID_COL,
        n_repeats=N_REPEATS,
        n_splits=N_SPLITS,
        n_perm=N_PERM,
        seed=SEED,
        cv_n_jobs=N_JOBS,
        perm_n_jobs=PERM_N_JOBS,
        confounder_categorical=(base_analysis.SEX_COL, base_analysis.DIAG_COL, base_analysis.COUNTRY_COL),
        confounder_numeric=(base_analysis.AGE_COL, base_analysis.GOF_COL),
        progress=PROGRESS,
    )

    output_prefix = analysis_name
    _save_observed_h2(RESULTS_DIR / f"{output_prefix}_observed_h2.csv", results["observed_repeats"])
    _save_null_h2(RESULTS_DIR / f"{output_prefix}_null_h2.csv", results["perm_means"])
    summary = {
        "analysis": analysis_name,
        "exposure": exposure_name,
        "best_combo": combo.label,
        "observed_mean": float(results["observed_mean"]),
        "null_mean": float(np.nanmean(results["perm_means"])),
        "null_sd": float(np.nanstd(results["perm_means"], ddof=1)),
        "p_value": float(results["p_value_addone"]),
        "n_perm": N_PERM,
        "n_repeats": N_REPEATS,
        "n_splits": N_SPLITS,
        "n_subjects": int(len(df)),
        "dropped": int(results["dropped_permutations"]),
    }
    _write_summary(RESULTS_DIR / f"{output_prefix}_permtest_summary.csv", summary)
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _ts("Starting individual_exposome permutation test script")
    _ts(f"N_REPEATS={N_REPEATS}, N_PERM={N_PERM}, N_SPLITS={N_SPLITS}, N_JOBS={N_JOBS}, PERM_N_JOBS={PERM_N_JOBS}")

    df_raw, _, _ = base_analysis.load_base_data()
    feature_map = pfe.make_feature_map(df_raw, prefixes=base_analysis.FEATURE_PREFIXES)
    all_neural_cols = sorted({col for cols in feature_map.values() for col in cols})
    all_combos = pfe.build_combinations(feature_map)

    combined_summary: List[Dict[str, object]] = []
    for exposure_name in EXPOSURE_NAMES:
        combined_summary.append(_run_individual_exposome_analysis(df_raw, all_neural_cols, all_combos, exposure_name))

    summary_df = pd.DataFrame(combined_summary)
    all_summary_path = RESULTS_DIR / "permtest_all_summary.csv"
    summary_df.to_csv(all_summary_path, index=False)
    _ts(f"Wrote combined summary to {all_summary_path}")
    _ts("Completed individual_exposome permutation test script")


if __name__ == "__main__":
    main()
