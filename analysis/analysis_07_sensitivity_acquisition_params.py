#!/usr/bin/env python
# coding: utf-8

# Sensitivity analysis: acquisition parameters vs O-information and model fit.
# Tests whether TR, TE, scan duration, N volumes, and ODQ predict:
#   (1) empirical O-information mean  — full sample
#   (2) empirical O-information variance — full sample
#   (3) model GOF (gof_corr)           — full sample
#   (4) delta GOF (personalized - global) — personalized-model subsample
# LME: outcome_z ~ TR_z + TE_z + duration_z + nvol_z + ODQ_z
#      random intercept for Country, scanner nested within Country as VC.

from __future__ import annotations

import io as _io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT        = Path(__file__).resolve().parents[1]
DATA_PATH        = REPO_ROOT / "data" / "derived" / "model_output_plus_exposome_data_v3.csv"
METADATA_PATH    = REPO_ROOT / "data" / "derived" / "metadata.csv"
PERS_VEC_PATH    = REPO_ROOT / "data" / "derived" / "personalized_vectors.npz"
GLOBAL_VEC_PATH  = REPO_ROOT / "data" / "derived" / "global_vectors.npz"
OMATS_PATH       = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/all_Omats.npz")
SCANNER_PATH     = Path("/home/rherzog/Documents/Brainlat/exposome_tables/Table fMRI Scanning protocols.xlsx")
TABLE_DIR        = REPO_ROOT / "analysis" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PREDICTORS_Z = ["TR_z", "TE_z", "duration_z", "nvol_z", "ODQ_z"]

# ---------------------------------------------------------------------------
# Helpers (reused from analysis_01)
# ---------------------------------------------------------------------------

def load_vectors_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a_c, axis=1) * np.linalg.norm(b_c, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.sum(a_c * b_c, axis=1) / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


# ---------------------------------------------------------------------------
# LME helpers (reused from analysis_05)
# ---------------------------------------------------------------------------

def fit_mixedlm_with_retry(formula, data, groups, vc_formula):
    model = sm.MixedLM.from_formula(
        formula, data=data, groups=groups,
        re_formula="1", vc_formula=vc_formula,
    )
    for method, maxiter in [("lbfgs", 500), ("powell", 1000)]:
        try:
            fit = model.fit(reml=False, method=method, maxiter=maxiter)
            if fit.converged:
                return fit
        except Exception:
            pass
    return None


def get_iccs(fit):
    var_country = fit.cov_re.iloc[0, 0] if not fit.cov_re.empty else 0.0
    var_scanner = 0.0
    if hasattr(fit, "vcomp") and hasattr(fit.model, "exog_vc"):
        for name, val in zip(fit.model.exog_vc.names, fit.vcomp):
            if "scanner" in name:
                var_scanner = val
                break
    resid = fit.scale
    total = var_country + var_scanner + resid
    icc_country = var_country / total if total > 0 else 0.0
    icc_scanner = var_scanner / total if total > 0 else 0.0
    return icc_country, icc_scanner


# ---------------------------------------------------------------------------
# Load and merge acquisition parameters
# ---------------------------------------------------------------------------

print("Loading main data...")
main_df = pd.read_csv(DATA_PATH, low_memory=False)
main_df["N_MEGA"] = main_df["N_MEGA"].astype(str).str.strip()
main_df = main_df[~main_df["Country"].str.strip().str.lower().eq("new zeland")]

print("Loading scanner table...")
scanner_df = pd.read_excel(SCANNER_PATH, header=1)
scanner_df = scanner_df.rename(columns={
    "TR (ms)":       "TR_ms",
    "TE (ms)":       "TE_ms",
    "N° vol":        "N_vol",
    "Duration (min)":"Duration_min",
    "scanner":       "resonador",
})
scanner_df["resonador"] = scanner_df["resonador"].astype(float)
# 4 scanners have missing Duration; impute as TR_ms * N_vol / 60000
scanner_df["Duration_min"] = scanner_df["Duration_min"].fillna(
    scanner_df["TR_ms"] * scanner_df["N_vol"] / 60000
)

main_df = main_df.merge(
    scanner_df[["resonador", "TR_ms", "TE_ms", "N_vol", "Duration_min"]],
    on="resonador", how="left",
)

# ---------------------------------------------------------------------------
# Compute O-information mean and variance per subject
# ---------------------------------------------------------------------------

print("Loading O-mats...")
omats = np.load(OMATS_PATH)
omat_records = []
for key in omats.files:
    arr = omats[key]
    omat_records.append({
        "N_MEGA":    str(key).strip(),
        "omat_mean": float(np.mean(arr)),
        "omat_var":  float(np.var(arr, ddof=0)),
    })
omat_df = pd.DataFrame(omat_records)

# ---------------------------------------------------------------------------
# Build full-sample DataFrame
# ---------------------------------------------------------------------------

full_df = main_df.merge(omat_df, on="N_MEGA", how="inner")
req_cols = ["gof_corr", "omat_mean", "omat_var", "TR_ms", "TE_ms", "N_vol", "Duration_min", "ODQ_fMRI", "Country", "resonador"]
full_df = full_df.dropna(subset=req_cols).copy()

full_df["TR_z"]       = zscore(full_df["TR_ms"])
full_df["TE_z"]       = zscore(full_df["TE_ms"])
full_df["nvol_z"]     = zscore(full_df["N_vol"])
full_df["duration_z"] = zscore(full_df["Duration_min"])
full_df["ODQ_z"]      = zscore(full_df["ODQ_fMRI"])
full_df["scanner_country"] = full_df["Country"].astype(str) + "::" + full_df["resonador"].astype(str)

print(f"Full sample N = {len(full_df)}")

# ---------------------------------------------------------------------------
# Build delta-GOF DataFrame (personalized model subsample)
# ---------------------------------------------------------------------------

print("Computing delta GOF...")
metadata = pd.read_csv(METADATA_PATH)
metadata["N_MEGA"] = metadata["N_MEGA"].astype(str).str.strip()

pers_vecs   = load_vectors_npz(PERS_VEC_PATH)
global_vecs = load_vectors_npz(GLOBAL_VEC_PATH)

# load empirical O-mats as lookup (same source used in analysis_01 via concat.npy;
# here we use all_Omats.npz which stores the same upper-triangle vectors)
empirical_lookup = {str(k).strip(): omats[k] for k in omats.files}

valid_mask = (
    metadata["N_MEGA"].isin(empirical_lookup) &
    metadata["N_MEGA"].isin(pers_vecs) &
    metadata["group_id"].astype(str).isin(global_vecs)
)
metadata = metadata[valid_mask].reset_index(drop=True)

empirical_matrix    = np.stack([empirical_lookup[n]          for n in metadata["N_MEGA"]])
personalized_matrix = np.stack([pers_vecs[n]                 for n in metadata["N_MEGA"]])
global_matrix       = np.stack([global_vecs[str(g)]          for g in metadata["group_id"]])

pers_corr   = rowwise_correlation(empirical_matrix, personalized_matrix)
global_corr = rowwise_correlation(empirical_matrix, global_matrix)

delta_df = metadata.assign(
    gof_personalized=pers_corr,
    gof_global=global_corr,
    delta_gof=pers_corr - global_corr,
)

# merge acquisition params from full_df
delta_df = delta_df.merge(
    full_df[["N_MEGA", "TR_z", "TE_z", "nvol_z", "duration_z", "ODQ_z", "Country", "scanner_country"]],
    on="N_MEGA", how="inner",
)
delta_df = delta_df.dropna(subset=["delta_gof"] + PREDICTORS_Z).copy()
print(f"Delta-GOF sample N = {len(delta_df)}")

# ---------------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------------

formula = "outcome_z ~ " + " + ".join(PREDICTORS_Z)
vc_formula = {"scanner": "0 + C(scanner_country)"}

_buf = _io.StringIO()

def _p(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_buf)

_p("=" * 72)
_p("SENSITIVITY ANALYSIS: Acquisition parameters vs O-information & model fit")
_p(f"Formula: {formula}")
_p("Random effects: intercept | Country; scanner nested within Country (VC)")
_p("=" * 72)

outcomes_full = [
    ("omat_mean",  "O-information mean",     full_df),
    ("omat_var",   "O-information variance",  full_df),
    ("gof_corr",   "Model GOF",               full_df),
]
outcomes_delta = [
    ("delta_gof",  "Delta GOF (pers - global)", delta_df),
]

fe_full_rows  = []
fe_delta_rows = []
diag_rows     = []

for outcome_col, outcome_label, data in outcomes_full + outcomes_delta:
    _p(f"\n--- Outcome: {outcome_label} (N={len(data)}) ---")
    fit_data = data.copy()
    fit_data["outcome_z"] = zscore(fit_data[outcome_col])

    fit = fit_mixedlm_with_retry(formula, fit_data, fit_data["Country"], vc_formula)

    if fit is None:
        _p(f"  FAILED TO CONVERGE")
        continue

    icc_c, icc_s = get_iccs(fit)
    diag_rows.append({
        "Outcome":     outcome_label,
        "Sample":      "delta_GOF" if outcome_col == "delta_gof" else "full",
        "N":           len(fit_data),
        "LogLik":      fit.llf,
        "AIC":         fit.aic,
        "BIC":         fit.bic,
        "ICC_country": icc_c,
        "ICC_scanner": icc_s,
        "Converged":   fit.converged,
    })

    for term in PREDICTORS_Z:
        if term not in fit.fe_params:
            continue
        ci = fit.conf_int().loc[term]
        row = {
            "Outcome":   outcome_label,
            "Sample":    "delta_GOF" if outcome_col == "delta_gof" else "full",
            "Term":      term,
            "Estimate":  fit.fe_params[term],
            "SE":        fit.bse[term],
            "z":         fit.tvalues[term],
            "p":         fit.pvalues[term],
            "CI_low":    ci[0],
            "CI_high":   ci[1],
        }
        if outcome_col == "delta_gof":
            fe_delta_rows.append(row)
        else:
            fe_full_rows.append(row)

    _p(fit.summary().tables[1])

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

fe_full_df  = pd.DataFrame(fe_full_rows)
fe_delta_df = pd.DataFrame(fe_delta_rows)
diag_df     = pd.DataFrame(diag_rows)

fe_full_csv  = TABLE_DIR / "sensitivity_acquisition_full_sample_fixed_effects.csv"
fe_delta_csv = TABLE_DIR / "sensitivity_acquisition_delta_gof_fixed_effects.csv"
diag_csv     = TABLE_DIR / "sensitivity_acquisition_diagnostics.csv"
summary_txt  = TABLE_DIR / "sensitivity_acquisition_summary.txt"

fe_full_df.to_csv(fe_full_csv,   index=False)
fe_delta_df.to_csv(fe_delta_csv, index=False)
diag_df.to_csv(diag_csv,         index=False)
summary_txt.write_text(_buf.getvalue())

print(f"\nOutputs saved to {TABLE_DIR}:")
for f in [fe_full_csv, fe_delta_csv, diag_csv, summary_txt]:
    print(f"  {f.name}")
