#!/usr/bin/env python
# coding: utf-8

"""Individual exposome PCEV + LME pipeline."""

from __future__ import annotations

import io as _io
import os
import re
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

import sys

sys.path.append(".")
import pcev_feature_effects as pfe

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = Path(os.environ.get("PCEV_RESULTS_DIR", REPO_ROOT / "analysis" / "results" / "pcev_results"))
RESULTS_DIR = RESULTS_BASE / "individual_exposome_no_odq"
TABLE_DIR = REPO_ROOT / "analysis" / "tables_for_paper"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CURATED_PATH = REPO_ROOT / "data" / "derived" / "individual_exposome_curated.csv"

ID_COL = "record_id"
RECORD_ID_COL = "record_id"
SEX_COL = "Sex"
DIAG_COL = "Diagnosis"
COUNTRY_COL = "Country"
SITE_COL = "site"
GOF_COL = "gof_corr"
AGE_COL = "Age"

N_REPEATS = int(os.environ.get("N_REPEATS", 500))
N_SPLITS = int(os.environ.get("N_SPLITS", 5))
SEED = int(os.environ.get("SEED", 2025))
N_JOBS = int(os.environ.get("N_JOBS", -1))
MAX_COMBOS_ENV = os.environ.get("MAX_COMBOS", "").strip()
MAX_COMBOS = int(MAX_COMBOS_ENV) if MAX_COMBOS_ENV else None

SEX_MAP = {
    1: "Female",
    2: "Male",
    1.0: "Female",
    2.0: "Male",
    "1": "Female",
    "2": "Male",
    "Female": "Female",
    "Male": "Male",
}

FEATURE_PREFIXES = ("ent_E", "ent_I", "rate_E", "rate_I", "EI_ent", "EI_rate")

SVI_VARS = OrderedDict(
    {
        "SVI_access_healthcare": "SVI_access_healthcare",
        "SVI_assets": "SVI_assets",
        "SVI_childhood_experiences": "SVI_childhood_experiences",
        "SVI_education": "SVI_education",
        "SVI_financial_status": "SVI_financial_status",
        "SVI_occupation": "SVI_occupation",
        "SVI_relations": "SVI_relations",
        "SVI_traumatic_experiences": "SVI_traumatic_experiences",
    }
)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _safe_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")


def _ts(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - mean) / std


def _normalize_sex_series(series: pd.Series) -> pd.Series:
    mapped = series.map(SEX_MAP)
    if mapped.notna().any():
        return mapped
    text = series.astype("string").str.strip()
    return text.where(text.isin(["Female", "Male"]), pd.NA)


def _prefix_indices(df: pd.DataFrame, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    idxs = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            idxs.append(int(match.group(1)))
    return sorted(set(idxs))


def _make_repeated_infra(df: pd.DataFrame):
    strata = pfe.make_strata(df, diag_col=DIAG_COL, sex_col=SEX_COL)
    splits = pfe._make_repeated_splits(strata, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=SEED)
    conf_matrix, _ = pfe._fit_confounders_matrix(
        df,
        categorical=(SEX_COL, DIAG_COL, COUNTRY_COL),
        numeric=(AGE_COL, GOF_COL),
    )
    return strata, splits, conf_matrix


# =============================================================================
# DATA LOADING
# =============================================================================

def load_base_data():
    _ts("Loading curated individual exposome table")
    df = pd.read_csv(CURATED_PATH, low_memory=False)
    df[RECORD_ID_COL] = df[RECORD_ID_COL].astype("string").str.strip()

    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df[SEX_COL] = _normalize_sex_series(df[SEX_COL])
    df[DIAG_COL] = df[DIAG_COL].astype("string").str.strip()
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("string").str.strip()
    df[SITE_COL] = df[SITE_COL].astype("string").str.strip()
    df[GOF_COL] = pd.to_numeric(df[GOF_COL], errors="coerce")

    feature_map = pfe.make_feature_map(df, prefixes=FEATURE_PREFIXES)
    all_neural_cols = sorted({col for cols in feature_map.values() for col in cols})
    all_combos = pfe.build_combinations(feature_map)

    if MAX_COMBOS is not None and MAX_COMBOS < len(all_combos):
        full_combo = all_combos[-1]
        selected = all_combos[: max(1, MAX_COMBOS - 1)]
        if full_combo not in selected:
            selected.append(full_combo)
        all_combos = selected
        _ts(f"Created {len(all_combos)} feature combinations (MAX_COMBOS={MAX_COMBOS}, all_features retained)")
    else:
        _ts(f"Created {len(all_combos)} feature combinations")

    _ts(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns from curated table")
    return df, all_combos, all_neural_cols


# =============================================================================
# PCEV
# =============================================================================

def _run_single_exposure_pcev(
    df_base: pd.DataFrame,
    all_combos: list[pfe.FeatureCombo],
    all_neural_cols: list[str],
    exposure_name: str,
):
    results_dir = RESULTS_DIR / _safe_name(exposure_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    _ts(f"Running PCEV for {exposure_name}")
    required_cols = all_neural_cols + [ID_COL, exposure_name, AGE_COL, SEX_COL, DIAG_COL, COUNTRY_COL, GOF_COL]
    missing = [col for col in required_cols if col not in df_base.columns]
    if missing:
        raise ValueError(f"Missing columns for {exposure_name}: {missing}")

    df = df_base[required_cols].dropna().reset_index(drop=True)
    df[SEX_COL] = df[SEX_COL].astype("category")
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")

    conf_cat = (SEX_COL, DIAG_COL, COUNTRY_COL)
    conf_num = (AGE_COL, GOF_COL)
    _, splits, conf_matrix = _make_repeated_infra(df)

    all_summaries = []
    for idx, combo in enumerate(all_combos, 1):
        _ts(f"[{exposure_name}] combo {idx}/{len(all_combos)}: {combo.label}")
        start = time.time()

        fold_df, score_df = pfe._run_repeated_cv_generic(
            df,
            feature_combo=combo,
            x_cols=[exposure_name],
            id_col=ID_COL,
            covariate_group=None,
            diag_col=DIAG_COL,
            sex_col=SEX_COL,
            n_repeats=N_REPEATS,
            n_splits=N_SPLITS,
            seed=SEED,
            n_jobs=N_JOBS,
            pcev_kwargs={},
            confounder_categorical=conf_cat,
            confounder_numeric=conf_num,
            splits=splits,
            conf_matrix=conf_matrix,
            legacy_scalar_alias=True,
            context=None,
            joblib_verbose=10,
        )
        del fold_df

        h2_per_repeat = (
            score_df.groupby("repeat", group_keys=False)
            .apply(lambda g: pfe._compute_h2_from_scores(g))
            .values
        )

        combo_safe = combo.label.replace("+", "_")
        pd.DataFrame({"repeat": range(N_REPEATS), "h2_with": h2_per_repeat}).to_csv(
            results_dir / f"{combo_safe}_h2_per_repeat.csv",
            index=False,
        )

        avg_scores = score_df.groupby("subject_id")["score"].mean().reset_index()
        avg_scores.columns = ["subject_id", "score_with"]
        avg_scores.to_csv(results_dir / f"{combo_safe}_subject_scores.csv", index=False)

        all_summaries.append(
            pd.DataFrame(
                {
                    "exposure_variable": [exposure_name],
                    "combo_label": [combo.label],
                    "combo_key": [combo.key],
                    "n_features": [len(combo.columns)],
                    "n_subjects": [df.shape[0]],
                    "h2_with_mean": [float(np.nanmean(h2_per_repeat))],
                    "h2_with_std": [float(np.nanstd(h2_per_repeat))],
                    "n_repeats": [N_REPEATS],
                    "n_splits": [N_SPLITS],
                    "seed": [SEED],
                }
            )
        )

        _ts(f"[{exposure_name}] {combo.label} completed in {time.time() - start:.1f}s")

    pd.concat(all_summaries, ignore_index=True).to_csv(results_dir / "all_combos_summary.csv", index=False)
    return results_dir


def run_pcev_stage(df_base: pd.DataFrame, all_combos: list[pfe.FeatureCombo], all_neural_cols: list[str]) -> dict[str, Path]:
    exposure_dirs: dict[str, Path] = {}
    for exposure_name in SVI_VARS.values():
        exposure_dirs[exposure_name] = _run_single_exposure_pcev(df_base, all_combos, all_neural_cols, exposure_name)
    return exposure_dirs


# =============================================================================
# H2 COMPARISON
# =============================================================================

def paired_sign_flip_test(data_all, data_best, n_perm=10000, statistic="mean"):
    diff = np.array(data_all) - np.array(data_best)
    obs_stat = np.mean(diff) if statistic == "mean" else np.median(diff)
    signs = np.random.choice([-1, 1], size=(n_perm, len(diff)))
    perm_stats = np.mean(signs * diff, axis=1) if statistic == "mean" else np.median(signs * diff, axis=1)
    p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    return obs_stat, p_val


def bootstrap_ci(data, n_boot=10000, statistic="mean"):
    data = np.array(data)
    indices = np.random.randint(0, len(data), size=(n_boot, len(data)))
    samples = data[indices]
    boot_stats = np.mean(samples, axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return np.percentile(boot_stats, 2.5), np.percentile(boot_stats, 97.5)


def build_h2_comparison(exposure_dirs: dict[str, Path]) -> pd.DataFrame:
    np.random.seed(42)
    rows = []

    _ts("Running h² best-vs-all comparison")
    for exposure_name in SVI_VARS.values():
        expo_dir = exposure_dirs[exposure_name]
        summary_df = pd.read_csv(expo_dir / "all_combos_summary.csv")
        non_all = summary_df[summary_df["combo_label"] != "all_features"].copy()

        best_row = summary_df[summary_df["combo_label"] == "all_features"].iloc[0] if non_all.empty else non_all.sort_values("h2_with_mean", ascending=False).iloc[0]
        best_combo = best_row["combo_label"]
        best_safe = best_combo.replace("+", "_")

        h2_best_df = pd.read_csv(expo_dir / f"{best_safe}_h2_per_repeat.csv")
        h2_all_df = pd.read_csv(expo_dir / "all_features_h2_per_repeat.csv")
        if len(h2_best_df) != len(h2_all_df):
            _ts(f"Warning: repeat length mismatch for {exposure_name}")
            continue

        x = h2_best_df["h2_with"].values
        y = h2_all_df["h2_with"].values
        obs_stat, pval = paired_sign_flip_test(y, x, n_perm=10000, statistic="mean")
        ci_lower, ci_upper = bootstrap_ci(y - x, n_boot=10000, statistic="mean")

        rows.append(
            {
                "Exposome": exposure_name,
                "Best_Combo": best_combo,
                "N_Repeats": len(x),
                "Best_Mean": np.mean(x),
                "Best_SD": np.std(x, ddof=1),
                "All_Mean": np.mean(y),
                "All_SD": np.std(y, ddof=1),
                "Test_Paired_Type": "Sign-flip Permutation",
                "Delta_Obs": obs_stat,
                "Delta_CI_Low": ci_lower,
                "Delta_CI_High": ci_upper,
                "P_Value_Paired": pval,
            }
        )

    h2_table = pd.DataFrame(rows)
    h2_csv = TABLE_DIR / "individual_exposome_h2_comparison.csv"
    h2_table.to_csv(h2_csv, index=False)
    _ts(f"Saved h² comparison to {h2_csv}")
    return h2_table


# =============================================================================
# LME
# =============================================================================

def fit_mixedlm_with_retry(formula, data, groups, vc_formula):
    model = sm.MixedLM.from_formula(formula, data=data, groups=groups, re_formula="1", vc_formula=vc_formula)
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
    var_site = 0.0
    if hasattr(fit, "vcomp") and hasattr(fit.model, "exog_vc"):
        for name, val in zip(fit.model.exog_vc.names, fit.vcomp):
            if "site" in str(name).lower():
                var_site = val
                break
    resid = fit.scale
    total = var_country + var_site + resid
    return (var_country / total if total > 0 else 0.0, var_site / total if total > 0 else 0.0)


def run_lme_stage(df_base: pd.DataFrame, exposure_dirs: dict[str, Path]) -> pd.DataFrame:
    _ts("Running LME + LRT stage")
    results_list = []

    for exposure_name in SVI_VARS.values():
        print(f"Processing: {exposure_name}", flush=True)
        expo_dir = exposure_dirs[exposure_name]
        if not expo_dir.exists():
            continue

        try:
            summary_df = pd.read_csv(expo_dir / "all_combos_summary.csv")
            non_all = summary_df[summary_df["combo_label"] != "all_features"].copy()
            best_combo_row = non_all.sort_values("h2_with_mean", ascending=False).iloc[0] if not non_all.empty else summary_df[summary_df["combo_label"] == "all_features"].iloc[0]
            h2_best_combo = best_combo_row["h2_with_mean"]
            all_feat_row = summary_df[summary_df["combo_label"] == "all_features"].iloc[0]
            h2_all_feat = all_feat_row["h2_with_mean"]

            if h2_best_combo >= h2_all_feat:
                winning_label = best_combo_row["combo_label"]
                winning_safe = winning_label.replace("+", "_")
                model_type = "Best Combo"
            else:
                winning_label = "all_features"
                winning_safe = "all_features"
                model_type = "All Features"

            scores = pd.read_csv(expo_dir / f"{winning_safe}_subject_scores.csv")
            scores["subject_id"] = scores["subject_id"].astype(str).str.strip()
            scores["score_with"] = pd.to_numeric(scores["score_with"], errors="coerce")
            scores_avg = scores.groupby("subject_id", as_index=False)["score_with"].mean()
            merged = df_base.merge(scores_avg, left_on=ID_COL, right_on="subject_id", how="inner")
        except Exception as exc:
            print(f"  Skipping {exposure_name}: {exc}", flush=True)
            continue

        cols = ["score_with", exposure_name, AGE_COL, SEX_COL, DIAG_COL, COUNTRY_COL, SITE_COL]
        data = merged[cols].dropna().copy()
        if data.empty or data[exposure_name].std(ddof=0) == 0 or len(data) < 10:
            continue

        data["pcev_z"] = zscore(data["score_with"])
        data["exposure_z"] = zscore(data[exposure_name])
        data["Age_z"] = zscore(data[AGE_COL])
        data["Sex_male"] = (data[SEX_COL].astype(str).str.lower() == "male").astype(float)
        data["site_country"] = data[COUNTRY_COL].astype(str) + "::" + data[SITE_COL].astype(str)

        dx_dummies = pd.get_dummies(data[DIAG_COL].astype(str), prefix="Dx")
        drop_col = "Dx_CN" if "Dx_CN" in dx_dummies.columns else sorted(dx_dummies.columns)[0]
        dx_dummies = dx_dummies.drop(columns=[drop_col])
        data = pd.concat([data, dx_dummies], axis=1)

        covariates = ["Age_z", "Sex_male"] + list(dx_dummies.columns)
        formula_base = f"pcev_z ~ {' + '.join(covariates)}"
        formula_full = f"pcev_z ~ exposure_z + {' + '.join(covariates)}"
        vc_formula = {"site": "0 + C(site_country)"}

        try:
            fit_base = fit_mixedlm_with_retry(formula_base, data, data[COUNTRY_COL], vc_formula)
            fit_full = fit_mixedlm_with_retry(formula_full, data, data[COUNTRY_COL], vc_formula)
            if fit_base is None or fit_full is None:
                continue

            beta = fit_full.fe_params["exposure_z"]
            ci_low, ci_high = fit_full.conf_int().loc["exposure_z"].tolist()
            delta_ll = fit_full.llf - fit_base.llf
            lrt_stat = max(0, 2 * delta_ll)
            p_lrt = chi2.sf(lrt_stat, df=1)
            delta_aic = fit_full.aic - fit_base.aic
            icc_country_base, icc_site_base = get_iccs(fit_base)
            icc_country_full, icc_site_full = get_iccs(fit_full)

            results_list.append(
                {
                    "Exposome": exposure_name,
                    "Variable": exposure_name,
                    "Model_Used": model_type,
                    "N": len(data),
                    "Beta": beta,
                    "Beta_CI_Low": ci_low,
                    "Beta_CI_High": ci_high,
                    "LL_Base": fit_base.llf,
                    "LL_Full": fit_full.llf,
                    "Delta_LL": delta_ll,
                    "LRT_Stat": lrt_stat,
                    "P_LRT": p_lrt,
                    "AIC_Base": fit_base.aic,
                    "AIC_Full": fit_full.aic,
                    "Delta_AIC": delta_aic,
                    "ICC_country_base": icc_country_base,
                    "ICC_site_base": icc_site_base,
                    "ICC_country_full": icc_country_full,
                    "ICC_site_full": icc_site_full,
                    "Delta_ICC_country": icc_country_full - icc_country_base,
                    "Delta_ICC_site": icc_site_full - icc_site_base,
                }
            )
        except Exception as exc:
            print(f"  Error fitting {exposure_name}: {exc}", flush=True)

    lrt_df = pd.DataFrame(results_list)
    if not lrt_df.empty:
        lrt_df["FDR_q_lrt"] = np.nan
        pvals = lrt_df["P_LRT"].dropna()
        if not pvals.empty:
            _, qvals, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.001)
            lrt_df.loc[pvals.index, "FDR_q_lrt"] = qvals

    lrt_csv = TABLE_DIR / "individual_exposome_lme_lrt.csv"
    lrt_df.to_csv(lrt_csv, index=False)
    _ts(f"Saved LME/LRT results to {lrt_csv}")

    _buf = _io.StringIO()

    def _p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=_buf)

    _p("\n" + "=" * 72)
    _p("ANALYSIS: LME + LRT per individual SVI variable")
    _p("=" * 72)
    if not lrt_df.empty:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)
        _p(
            lrt_df[
                [
                    "Exposome",
                    "Variable",
                    "Model_Used",
                    "N",
                    "Beta",
                    "Beta_CI_Low",
                    "Beta_CI_High",
                    "LRT_Stat",
                    "P_LRT",
                    "FDR_q_lrt",
                    "Delta_AIC",
                ]
            ].to_string(index=False)
        )
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_colwidth")

    summary_txt = TABLE_DIR / "individual_exposome_lme_numeric_summary.txt"
    summary_txt.write_text(_buf.getvalue())
    _ts(f"Saved numeric summary to {summary_txt}")

    return lrt_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_base, all_combos, all_neural_cols = load_base_data()
    exposure_dirs = run_pcev_stage(df_base, all_combos, all_neural_cols)
    build_h2_comparison(exposure_dirs)
    run_lme_stage(df_base, exposure_dirs)
    _ts("ALL ANALYSES COMPLETE")


if __name__ == "__main__":
    main()
