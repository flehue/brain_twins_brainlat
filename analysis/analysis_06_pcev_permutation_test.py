#!/usr/bin/env python
# coding: utf-8
"""Permutation testing for the PCEV pipeline.

This script runs definitive permutation tests for age, sex, diagnosis, and all
exposome domains. It selects the best feature combo for each analysis from the
existing `all_combos_summary.csv` results, runs the permutation nulls, and
writes clean outputs to `analysis/results/pcev_results/permutation_tests/`.
"""

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
import pcev_diagnosis as pdg
import analysis_03_pcev_phenotype_expotype as base_analysis

ID_COL = "N_MEGA"
AGE_COL = "Age"
SEX_COL = "Sex"
DIAG_COL = "Diagnosis"
COUNTRY_COL = "Country"
GOF_COL = "gof_corr"
ODQ_COL = "ODQ_fMRI"

RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "pcev_results" / "permutation_tests"
PCEV_RESULTS_BASE = REPO_ROOT / "analysis" / "results" / "pcev_results"
EXPOSOME_RESULTS_BASE = PCEV_RESULTS_BASE / "exposome_odq_only_no_scanner"

N_REPEATS = int(os.environ.get("N_REPEATS", 500))
N_PERM = int(os.environ.get("N_PERM", 500))
N_SPLITS = int(os.environ.get("N_SPLITS", 5))
SEED = int(os.environ.get("SEED", 2025))
N_JOBS = int(os.environ.get("N_JOBS", -1))
PERM_N_JOBS = int(os.environ.get("PERM_N_JOBS", 1))
PROGRESS = PERM_N_JOBS == 1

SEX_MAP = {"Male": 0.0, "Female": 1.0}


def _ts(message: str) -> None:
    print(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _load_base_data() -> tuple[pd.DataFrame, List[str], List[pfe.FeatureCombo]]:
    _ts("Loading main dataset")
    df_raw = pd.read_csv(REPO_ROOT / "data" / "derived" / "model_output_plus_exposome_data_v3.csv", low_memory=False)
    df_raw = df_raw[~df_raw[COUNTRY_COL].astype(str).str.strip().str.lower().eq("new zeland")]

    feature_map = pfe.make_feature_map(df_raw, prefixes=pfe.FEATURE_PREFIXES)
    all_neural_cols = sorted({col for cols in feature_map.values() for col in cols})
    all_combos = pfe.build_combinations(feature_map)

    _ts(f"Loaded {df_raw.shape[0]:,} rows and {len(all_neural_cols)} neural features")
    return df_raw, all_neural_cols, all_combos


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


def _prepare_age_df(df_raw: pd.DataFrame, all_neural_cols: Sequence[str]) -> pd.DataFrame:
    required = list(all_neural_cols) + [ID_COL, AGE_COL, SEX_COL, DIAG_COL, COUNTRY_COL, GOF_COL, ODQ_COL]
    df = df_raw[required].dropna().reset_index(drop=True)
    df[SEX_COL] = df[SEX_COL].astype("category")
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")
    return df


def _prepare_sex_df(df_raw: pd.DataFrame, all_neural_cols: Sequence[str]) -> pd.DataFrame:
    df = df_raw.copy()
    df["Sex_numeric"] = df[SEX_COL].map(SEX_MAP)
    valid_sex = df[SEX_COL].isin(SEX_MAP)
    if not valid_sex.all():
        _ts(f"Dropping {int((~valid_sex).sum())} rows with invalid sex values")
    df = df.loc[valid_sex]
    required = [ID_COL, SEX_COL, "Sex_numeric", DIAG_COL, COUNTRY_COL, GOF_COL, AGE_COL, ODQ_COL] + list(all_neural_cols)
    df = df[required].dropna().reset_index(drop=True)
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")
    return df


def _prepare_diagnosis_df(df_raw: pd.DataFrame, all_neural_cols: Sequence[str]) -> pd.DataFrame:
    required = [ID_COL, DIAG_COL, SEX_COL, COUNTRY_COL, AGE_COL, GOF_COL, ODQ_COL] + list(all_neural_cols)
    df = df_raw[required].dropna().reset_index(drop=True)
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[SEX_COL] = df[SEX_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")
    return df


def _prepare_exposome_df(df_raw: pd.DataFrame, all_neural_cols: Sequence[str], exposome_cols: Sequence[str]) -> pd.DataFrame:
    required = list(all_neural_cols) + [ID_COL, SEX_COL, DIAG_COL, COUNTRY_COL, AGE_COL, GOF_COL, ODQ_COL] + list(exposome_cols)
    df = df_raw[required].dropna().reset_index(drop=True)
    df[SEX_COL] = df[SEX_COL].astype("category")
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")
    return df


def _save_observed_h2(path: Path, observed_repeats: np.ndarray) -> None:
    pd.DataFrame({"repeat": np.arange(len(observed_repeats)), "h2_with": observed_repeats}).to_csv(path, index=False)


def _save_null_h2(path: Path, perm_means: np.ndarray) -> None:
    pd.DataFrame({"perm_id": np.arange(len(perm_means)), "h2_mean": perm_means}).to_csv(path, index=False)


def _write_summary(path: Path, row: Dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(path, index=False)


def _run_age_analysis(df_raw: pd.DataFrame, all_neural_cols: Sequence[str], all_combos: List[pfe.FeatureCombo]) -> Dict[str, object]:
    analysis_name = "age"
    summary_path = PCEV_RESULTS_BASE / f"{analysis_name}_odq_only_no_scanner" / "all_combos_summary.csv"
    combo = _best_combo_from_summary(summary_path, all_combos)
    df = _prepare_age_df(df_raw, all_neural_cols)

    _ts(f"Running age permutation test on combo '{combo.label}'")
    results = pfe.run_repeated_cv_with_permutations(
        df,
        feature_combo=combo,
        x_cols=[AGE_COL],
        id_col=ID_COL,
        n_repeats=N_REPEATS,
        n_splits=N_SPLITS,
        n_perm=N_PERM,
        seed=SEED,
        cv_n_jobs=N_JOBS,
        perm_n_jobs=PERM_N_JOBS,
        confounder_categorical=(SEX_COL, COUNTRY_COL, DIAG_COL),
        confounder_numeric=(GOF_COL, ODQ_COL),
        progress=PROGRESS,
    )

    output_prefix = analysis_name
    _save_observed_h2(RESULTS_DIR / f"{output_prefix}_observed_h2.csv", results["observed_repeats"])
    _save_null_h2(RESULTS_DIR / f"{output_prefix}_null_h2.csv", results["perm_means"])
    summary = {
        "analysis": analysis_name,
        "best_combo": combo.label,
        "observed_mean": float(results["observed_mean"]),
        "null_mean": float(np.nanmean(results["perm_means"])),
        "null_sd": float(np.nanstd(results["perm_means"], ddof=1)),
        "p_value": float(results["p_value_addone"]),
        "n_perm": N_PERM,
        "dropped": int(results["dropped_permutations"]),
    }
    _write_summary(RESULTS_DIR / f"{output_prefix}_permtest_summary.csv", summary)
    return summary


def _run_sex_analysis(df_raw: pd.DataFrame, all_neural_cols: Sequence[str], all_combos: List[pfe.FeatureCombo]) -> Dict[str, object]:
    analysis_name = "sex"
    summary_path = PCEV_RESULTS_BASE / f"{analysis_name}_odq_only_no_scanner" / "all_combos_summary.csv"
    combo = _best_combo_from_summary(summary_path, all_combos)
    df = _prepare_sex_df(df_raw, all_neural_cols)

    # Strata for permutation must be Diagnosis-only (not Diagnosis×Sex), because
    # Sex_numeric is perfectly correlated with the sex dimension of Diag×Sex strata —
    # shuffling within those strata would leave Sex_numeric unchanged every time.
    diag_codes, _ = pd.factorize(df[DIAG_COL])
    perm_strata = diag_codes

    _ts(f"Running sex permutation test on combo '{combo.label}'")
    results = pfe.run_repeated_cv_with_permutations(
        df,
        feature_combo=combo,
        x_cols=["Sex_numeric"],
        id_col=ID_COL,
        n_repeats=N_REPEATS,
        n_splits=N_SPLITS,
        n_perm=N_PERM,
        seed=SEED,
        strat_labels=perm_strata,
        cv_n_jobs=N_JOBS,
        perm_n_jobs=PERM_N_JOBS,
        confounder_categorical=(DIAG_COL, COUNTRY_COL),
        confounder_numeric=(GOF_COL, AGE_COL, ODQ_COL),
        progress=PROGRESS,
    )

    output_prefix = analysis_name
    _save_observed_h2(RESULTS_DIR / f"{output_prefix}_observed_h2.csv", results["observed_repeats"])
    _save_null_h2(RESULTS_DIR / f"{output_prefix}_null_h2.csv", results["perm_means"])
    summary = {
        "analysis": analysis_name,
        "best_combo": combo.label,
        "observed_mean": float(results["observed_mean"]),
        "null_mean": float(np.nanmean(results["perm_means"])),
        "null_sd": float(np.nanstd(results["perm_means"], ddof=1)),
        "p_value": float(results["p_value_addone"]),
        "n_perm": N_PERM,
        "dropped": int(results["dropped_permutations"]),
    }
    _write_summary(RESULTS_DIR / f"{output_prefix}_permtest_summary.csv", summary)
    return summary


def _run_diagnosis_analysis(df_raw: pd.DataFrame, all_neural_cols: Sequence[str], all_combos: List[pfe.FeatureCombo]) -> Dict[str, object]:
    analysis_name = "diagnosis"
    summary_path = PCEV_RESULTS_BASE / f"{analysis_name}_odq_only_no_scanner" / "all_combos_summary.csv"
    combo = _best_combo_from_summary(summary_path, all_combos)
    df = _prepare_diagnosis_df(df_raw, all_neural_cols)

    _ts(f"Running diagnosis permutation test on combo '{combo.label}'")
    results = pdg.run_repeated_cv_with_permutations(
        df,
        feature_combo=combo,
        diag_col=DIAG_COL,
        id_col=ID_COL,
        n_repeats=N_REPEATS,
        n_splits=N_SPLITS,
        n_perm=N_PERM,
        seed=SEED,
        n_jobs=N_JOBS,
        perm_n_jobs=PERM_N_JOBS,
        progress=PROGRESS,
    )

    output_prefix = analysis_name
    _save_observed_h2(RESULTS_DIR / f"{output_prefix}_observed_h2.csv", results["observed_h2_repeats"])
    _save_null_h2(RESULTS_DIR / f"{output_prefix}_null_h2.csv", np.nanmean(results["perm_h2_repeats"], axis=1) if results["perm_h2_repeats"].size else np.array([]))
    summary = {
        "analysis": analysis_name,
        "best_combo": combo.label,
        "observed_mean": float(results["observed_h2_mean"]),
        "null_mean": float(np.nanmean(np.nanmean(results["perm_h2_repeats"], axis=1))) if results["perm_h2_repeats"].size else np.nan,
        "null_sd": float(np.nanstd(np.nanmean(results["perm_h2_repeats"], axis=1), ddof=1)) if results["perm_h2_repeats"].size else np.nan,
        "p_value": float(results["p_value_h2"]),
        "observed_epsilon_mean": float(results["observed_epsilon_mean"]),
        "observed_eta_mean": float(results["observed_eta_mean"]),
        "p_value_epsilon": float(results["p_value_epsilon"]),
        "p_value_eta": float(results["p_value_eta"]),
        "n_perm": N_PERM,
        "dropped": int(results["dropped_permutations"]),
    }
    _write_summary(RESULTS_DIR / f"{output_prefix}_permtest_summary.csv", summary)
    return summary


def _exposome_dir_for_group(group_name: str) -> Path:
    safe_name = _safe_name(group_name).replace("__", "_")
    candidate = EXPOSOME_RESULTS_BASE / safe_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Exposome results folder not found for '{group_name}' at {candidate}")


def _run_exposome_analysis(
    df_raw: pd.DataFrame,
    all_neural_cols: Sequence[str],
    all_combos: List[pfe.FeatureCombo],
    group_name: str,
    exposome_cols: Sequence[str],
) -> Dict[str, object]:
    analysis_name = f"exposome_{_safe_name(group_name)}"
    summary_path = _exposome_dir_for_group(group_name) / "all_combos_summary.csv"
    combo = _best_combo_from_summary(summary_path, all_combos)
    df = _prepare_exposome_df(df_raw, all_neural_cols, exposome_cols)

    _ts(f"Running exposome permutation test for '{group_name}' on combo '{combo.label}'")
    results = pfe.run_repeated_cv_with_permutations(
        df,
        feature_combo=combo,
        x_cols=list(exposome_cols),
        id_col=ID_COL,
        n_repeats=N_REPEATS,
        n_splits=N_SPLITS,
        n_perm=N_PERM,
        seed=SEED,
        cv_n_jobs=N_JOBS,
        perm_n_jobs=PERM_N_JOBS,
        confounder_categorical=(SEX_COL, COUNTRY_COL, DIAG_COL),
        confounder_numeric=(GOF_COL, ODQ_COL),
        progress=PROGRESS,
    )

    output_prefix = analysis_name
    _save_observed_h2(RESULTS_DIR / f"{output_prefix}_observed_h2.csv", results["observed_repeats"])
    _save_null_h2(RESULTS_DIR / f"{output_prefix}_null_h2.csv", results["perm_means"])
    summary = {
        "analysis": analysis_name,
        "best_combo": combo.label,
        "observed_mean": float(results["observed_mean"]),
        "null_mean": float(np.nanmean(results["perm_means"])),
        "null_sd": float(np.nanstd(results["perm_means"], ddof=1)),
        "p_value": float(results["p_value_addone"]),
        "n_perm": N_PERM,
        "dropped": int(results["dropped_permutations"]),
    }
    _write_summary(RESULTS_DIR / f"{output_prefix}_permtest_summary.csv", summary)
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _ts("Starting definitive permutation test script")
    _ts(f"N_REPEATS={N_REPEATS}, N_PERM={N_PERM}, N_SPLITS={N_SPLITS}, N_JOBS={N_JOBS}, PERM_N_JOBS={PERM_N_JOBS}")

    # Optionally restrict which top-level analyses to run (comma-separated: age,sex,diagnosis,exposome)
    _analyses_env = os.environ.get("ANALYSES", "").strip().lower()
    run_only: set[str] = set(_analyses_env.split(",")) - {""} if _analyses_env else set()

    df_raw, all_neural_cols, all_combos = _load_base_data()

    combined_summary: List[Dict[str, object]] = []
    if not run_only or "age" in run_only:
        combined_summary.append(_run_age_analysis(df_raw, all_neural_cols, all_combos))
    if not run_only or "sex" in run_only:
        combined_summary.append(_run_sex_analysis(df_raw, all_neural_cols, all_combos))
    if not run_only or "diagnosis" in run_only:
        combined_summary.append(_run_diagnosis_analysis(df_raw, all_neural_cols, all_combos))

    exposome_groups = base_analysis.EXPOSOME_GROUPS
    if not run_only or "exposome" in run_only:
        for group_name, vars_map in exposome_groups.items():
            combined_summary.append(_run_exposome_analysis(df_raw, all_neural_cols, all_combos, group_name, list(vars_map.values())))

    summary_df = pd.DataFrame(combined_summary)
    all_summary_path = RESULTS_DIR / "permtest_all_summary.csv"
    if run_only and all_summary_path.exists():
        existing = pd.read_csv(all_summary_path)
        existing = existing[~existing["analysis"].isin(summary_df["analysis"])]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(all_summary_path, index=False)
    _ts(f"Wrote combined summary to {RESULTS_DIR / 'permtest_all_summary.csv'}")
    _ts("Completed definitive permutation test script")


if __name__ == "__main__":
    main()
