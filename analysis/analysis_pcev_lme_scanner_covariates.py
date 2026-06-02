#!/usr/bin/env python
# coding: utf-8

"""
Scanner-parameter sensitivity for PCEV + downstream LME (reviewer comment 1).

Re-runs the EXISTING PCEV/LME pipeline (same seeds, same CV, same confounders)
for ONLY the best combo of each phenotype and expotype, adding N volumes and scan
duration as covariates in BOTH stages, then quantifies how much these scanner
variables move the PCEV metric distributions and the downstream LME coefficients.

Best combo = the winner of the reference comparison between the best subset
and all_features in the original all_combos_summary.csv. If all_features has
the higher mean, that is the best combo and must be used.

Sections (run top-to-bottom):
  1. PCEV best-combo re-run with N_vol + Duration_min as numeric confounders.
     Structure copied from analysis_03_pcev_phenotype_expotype.py.
  2. LME re-run with Nvol_z + Dur_z as fixed-effect covariates.
     Phenotype unified-ODQ model from analysis_04; expotype per-variable LRT
     from analysis_05 (only the LRT part, Analysis 2; the Bayes-factor h2
     comparison from analysis_05 Analysis 1 is not reproduced here).
  3. Comparison: with vs without scanner covariates (PCEV metric distributions
     and LME coefficients), with sample-aware paired/unpaired statistics.

Acquisition parameters (N_vol, Duration_min) are NOT in the main CSV; they are
merged from the Excel scanner table on `resonador`, exactly as in analysis_07.

Outputs:
  PCEV scores:  analysis/results/pcev_results/{age,sex,diagnosis}_with_scanner_params/
                analysis/results/pcev_results/exposome_with_scanner_params/<Group>/
  LME tables:   analysis/tables_for_paper/lme_phenotypes_unified_scanner_covariates.csv
                analysis/tables_for_paper/expotype_lme_lrt_scanner_covariates.csv
    Comparisons:  analysis/tables_for_paper/scanner_covariates_pcev_metric_comparison.csv
                                analysis/tables_for_paper/scanner_covariates_pcev_metric_comparison_phenotype.csv
                                analysis/tables_for_paper/scanner_covariates_pcev_metric_comparison_exposome.csv
                                analysis/tables_for_paper/scanner_covariates_lme_coefficient_comparison.csv
                                analysis/tables_for_paper/scanner_covariates_lme_coefficient_comparison_phenotype.csv
                                analysis/tables_for_paper/scanner_covariates_lme_coefficient_comparison_exposome.csv
                                analysis/tables_for_paper/scanner_covariates_summary.txt
                                analysis/tables_for_paper/scanner_covariates_exposome_summary.txt
"""

from __future__ import annotations

import io as _io
import os
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

sys.path.append(".")
import pcev_feature_effects as pfe
import pcev_diagnosis as pdg

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION (copied unchanged from analysis_03 so seeds/CV match)
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "derived" / "model_output_plus_exposome_data_v3.csv"
RESULTS_BASE = Path(os.environ.get("PCEV_RESULTS_DIR", REPO_ROOT / "analysis" / "results" / "pcev_results"))
SCANNER_PATH = Path("/home/rherzog/Documents/Brainlat/exposome_tables/Table fMRI Scanning protocols.xlsx")
TABLE_DIR = Path(os.environ.get("TABLE_DIR", REPO_ROOT / "analysis" / "tables_for_paper"))
TABLE_DIR.mkdir(parents=True, exist_ok=True)
SCANNER_RESULTS_ROOT = RESULTS_BASE / "scanner_covariates"

# New (with-scanner) result dirs.
NEW_DIRS = {
    "age":       RESULTS_BASE / "age_with_scanner_params",
    "sex":       RESULTS_BASE / "sex_with_scanner_params",
    "diagnosis": RESULTS_BASE / "diagnosis_with_scanner_params",
}
NEW_EXPO_BASE = RESULTS_BASE / "exposome_with_scanner_params"

# Column names
ID_COL = "N_MEGA"
SEX_COL = "Sex"
DIAG_COL = "Diagnosis"
COUNTRY_COL = "Country"
GOF_COL = "gof_corr"
AGE_COL = "Age"
ODQ_COL = "ODQ_fMRI"
NVOL_COL = "N_vol"
DUR_COL = "Duration_min"
SCANNER_COVS = (NVOL_COL, DUR_COL)

# CV parameters (default repeats kept at 100 to match the existing no-scanner outputs)
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0").strip().lower() in {"1", "true", "yes", "on"}
SMOKE_N_SUBJECTS = int(os.environ.get("SMOKE_N_SUBJECTS", 200))
N_REPEATS = int(os.environ.get("N_REPEATS", 2 if SMOKE_TEST else 100))
N_SPLITS  = int(os.environ.get("N_SPLITS",  2 if SMOKE_TEST else 5))
SEED      = int(os.environ.get("SEED",      2025))
N_JOBS    = int(os.environ.get("N_JOBS",    (os.cpu_count() or 1) if SMOKE_TEST else 40))

# Sex mapping
SEX_MAP = {"Male": 0.0, "Female": 1.0}

# =============================================================================
# EXPOSOME GROUPS (copied unchanged from analysis_03)
# =============================================================================

EXPOSOME_GROUPS = OrderedDict({
    "Air Pollution": OrderedDict({
        "PM2.5": "PM2.5_interpolated",
        "Nitrogen oxides (NOx)": "Nitrogen oxide (NOx)_interpolated",
        "Sulfur dioxide (SO2)": "Sulphur dioxide (SO₂) emissions_interpolated",
        "Carbon monoxide (CO)": "Carbon monoxide (CO) emissions_interpolated",
        "Black carbon (BC)": "Black carbon (BC) emissions_interpolated",
        "Ammoniac Nitrogen (NH3)": "Ammonia (NH₃) emissions_interpolated",
        "Non-methane volatile organic compounds (NMVOC)": "Non-methane volatile organic compounds (NMVOC) emissions_interpolated",
    }),
    "Green space access": OrderedDict({
        "Urban green area (%) 1990": "Average share of green area in city/urban area 1990 (%)_interpolated",
        "Urban green area (%) 2000": "Average share of green area in city/ urban area 2000 (%)_interpolated",
        "Urban green area (%) 2010": "Average share of green area in city/ urban area 2010 (%)_interpolated",
        "Urban green area (%) 2020": "Average share of green area in city/ urban area 2020 (%)_interpolated",
        "Green area per capita (m2/person) 1990": "Green area per capita 1990 (m2/person)_interpolated",
        "Green area per capita (m2/person) 2000": "Green area per capita 2000 (m2/person)_interpolated",
        "Green area per capita (m2/person) 2010": "Green area per capita 2010 (m2/person)_interpolated",
        "Green area per capita (m2/person) 2020": "Green area per capita 2020 (m2/person)_interpolated",
    }),
    "Temperature": OrderedDict({
        "Mean temperature": "mean_temp_areaw_o_interpolated",
        "Mean temperature pop-weighted": "mean_temp_o_interpolated",
        "Mean temperature anomalies": "mean_anomalies_areaw_o_interpolated",
        "Mean temperature anomalies pop-weighted": "mean_anomalies_o_interpolated",
        "Deviation of temperature anomalies": "sd_lr_o_interpolated",
        "Max temperature": "maxgtemp_o_interpolated",
    }),
    "Precipitation-droughts": OrderedDict({
        "Mean precipitation": "mean_prec2_areaw_o_interpolated",
        "Mean precipitation pop-weighted": "mean_prec2_o_interpolated",
        "Palmer drought severity index": "scpdsi_aw_o_interpolated",
        "Palmer drought severity index pop-weighted": "scpdsi_o_interpolated",
    }),
    "Soil and water quality": OrderedDict({
        "Poisoning mortality rate": "Poisoning_mortality_rate_interpolated",
        "Basic drinking water access": "Pop_basic_drinking-water(%)_interpolated",
        "Safely-managed drinking water access": "Pop_safely_drinking-water(%)_interpolated",
        "Agriculture employment rate": "agri_emp_o_interpolated",
    }),
    "Climate disasters": OrderedDict({
        "Number of disaster events": "climatedisaster_count_o_interpolated",
        "Population affected": "climatedisaster_naffected_o_interpolated",
    }),
    "Disease-related mortality": OrderedDict({
        "Non-communicable death rate (%)": "deaths_notrans_interpolated",
        "Communicable death rate (%)": "deaths_trans_interpolated",
    }),
    "Socioeconomic": OrderedDict({
        "Gini index": "GINI_interpolated",
        "Human capital index": "HCI_interpolated",
        "Human development index (HDI)": "HDI_interpolated",
        "Inequality-Adjusted HDI": "IHDI_interpolated",
        "Multidimensional poverty measures": "MPM_interpolated",
        "Energy demand": "Electricity_demand_interpolated",
        "Gender inequality index": "GII_interpolated",
        "Unemployment rate": "unemp_interpolated",
        "GDP per capita": "GDP_interpolated",
    }),
    "Democracy": OrderedDict({
        "Credible elections": "cred_elect_est_interpolated",
        "Inclusive suffrage": "inclu_suff_est_interpolated",
        "Free parties": "free_parties_est_interpolated",
        "Elected government": "elected_gov_est_interpolated",
        "Effective parliament": "effect_parl_est_interpolated",
        "Local democracy": "local_dem_est_interpolated",
        "Access to justice": "access_just_est_interpolated",
        "Civil liberties": "civil_lib_est_interpolated",
        "Freedom of expression (Civil liberties)": "free_express_est_interpolated",
        "Freedom of the press (Civil liberties)": "free_press_est_interpolated",
        "Freedom of association and assembly (Civil liberties)": "free_assoc_assem_est_interpolated",
        "Freedom of religion (Civil liberties)": "free_relig_est_interpolated",
        "Freedom of movement (Civil liberties)": "free_move_est_interpolated",
        "Political equality": "pol_equal_est_interpolated",
        "Social group equality (Political equality)": "soc_grp_equal_est_interpolated",
        "Economic equality (Political equality)": "econ_equal_est_interpolated",
        "Gender equality (Political equality)": "gender_equal_est_interpolated",
        "Judicial independence": "jud_ind_est_interpolated",
        "Absence of Corruption": "abs_corrupt_est_interpolated",
        "Predictability enforcement": "predict_enf_est_interpolated",
        "Personal integrity and security": "pers_integ_sec_est_interpolated",
        "Civil society": "civil_soc_est_interpolated",
        "Civic engagement": "civic_engage_est_interpolated",
        "Electoral participation": "elect_part_est_interpolated",
        "Direct democracy": "direct_dem_est_interpolated",
    }),
    "Democracy-Summary": OrderedDict({
        "Political representation": "representation_est_interpolated",
        "Rights": "rights_est_interpolated",
        "Basic welfare": "basic_welf_est_interpolated",
        "Rule of law": "rule_law_est_interpolated",
        "Participation": "participation_est_interpolated",
    }),
    "Migration": OrderedDict({
        "Inbound migration rate": "migration_interpolated",
    }),
})

# analysis_05 excludes Democracy-Summary from the LME/expotype stage.
EXCLUDED_EXPOSOMES = {"Democracy-Summary"}

PHENOTYPE_BEST_TABLE = REPO_ROOT / "analysis" / "tables" / "pcev_phenotypes_best_vs_all.csv"
EXPOSOME_BEST_TABLE = REPO_ROOT / "analysis" / "tables" / "pcev_expotypes_best_vs_all.csv"


def safe_name(name: str) -> str:
    return name.replace(" ", "_")


# =============================================================================
# SHARED HELPERS
# =============================================================================

def load_scanner_table() -> pd.DataFrame:
    """Load + clean the Excel scanner table (merge block from analysis_07)."""
    scanner_df = pd.read_excel(SCANNER_PATH, header=1)
    scanner_df = scanner_df.rename(columns={
        "TR (ms)":        "TR_ms",
        "TE (ms)":        "TE_ms",
        "N° vol":         "N_vol",
        "Duration (min)": "Duration_min",
        "scanner":        "resonador",
    })
    scanner_df["resonador"] = scanner_df["resonador"].astype(float)
    # 4 scanners have missing Duration; impute as TR_ms * N_vol / 60000
    scanner_df["Duration_min"] = scanner_df["Duration_min"].fillna(
        scanner_df["TR_ms"] * scanner_df["N_vol"] / 60000
    )
    return scanner_df[["resonador", "N_vol", "Duration_min"]]


def _summary_value(value) -> float:
    return float(str(value).split()[0])


def _metric_display_name(metric_col: str) -> str:
    mapping = {
        "h2_with": "h2",
        "cohens_d_with": "cohens_d",
        "epsilon_with": "eta2",
    }
    return mapping.get(metric_col, metric_col)


def _load_selected_combo_map(table_path: Path, group_col: str, metric_value: str = "h²") -> dict:
    table = pd.read_csv(table_path)
    selected = {}
    for group_name in table[group_col].dropna().unique():
        row_df = table[(table[group_col] == group_name) & (table["Metric"] == metric_value)]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        best_subset = str(row["Best Subset"])
        delta_all_minus_best = float(row["Δ (All - Best)"])
        selected[group_name] = "all_features" if delta_all_minus_best > 0 else best_subset.replace("+", "_")
    return selected


PHENOTYPE_BEST_COMBOS = _load_selected_combo_map(PHENOTYPE_BEST_TABLE, "Phenotype")
EXPO_BEST_COMBOS = _load_selected_combo_map(EXPOSOME_BEST_TABLE, "Exposome")


def best_combo_for_axis(axis: str) -> str:
    axis_name = {
        "age": "Age",
        "sex": "Sex",
        "diagnosis": "Diagnosis",
    }.get(axis)
    if axis_name is None:
        raise KeyError(f"Unknown phenotype axis: {axis}")
    return PHENOTYPE_BEST_COMBOS[axis_name]


def best_combo_for_exposome(expo_name: str) -> str:
    if expo_name not in EXPO_BEST_COMBOS:
        raise KeyError(f"Unknown exposome group: {expo_name}")
    return EXPO_BEST_COMBOS[expo_name]


def combo_labels_for_comparison(best_label: str) -> list[str]:
    return [best_label]


def select_combo(all_combos, label):
    for combo in all_combos:
        if combo.label == label or combo.label.replace("+", "_") == label:
            return combo
    raise ValueError(f"Combo with label '{label}' not found among built combinations.")


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean, std = values.mean(), values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - mean) / std


# =============================================================================
# SECTION 1: PCEV BEST-COMBO RE-RUN WITH SCANNER COVARIATES
# =============================================================================

def load_base_data():
    print("\n[SECTION 1] LOADING DATA + SCANNER PARAMS", flush=True)
    print("-" * 80, flush=True)

    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns", flush=True)

    scanner_df = load_scanner_table()
    df_raw = df_raw.merge(scanner_df, on="resonador", how="left")
    n_nvol = df_raw[NVOL_COL].notna().sum()
    n_dur = df_raw[DUR_COL].notna().sum()
    print(f"  Merged scanner params: N_vol non-null={n_nvol:,}, "
          f"Duration_min non-null={n_dur:,}", flush=True)

    feature_map = pfe.make_feature_map(df_raw, prefixes=pfe.FEATURE_PREFIXES)
    all_combos = pfe.build_combinations(feature_map)
    all_neural_cols = sorted({col for cols in feature_map.values() for col in cols})

    return df_raw, all_combos, all_neural_cols


def sample_subject_subset(df: pd.DataFrame, n_subjects: int) -> pd.DataFrame:
    subject_ids = pd.Index(df[ID_COL].astype(str).dropna().unique())
    if len(subject_ids) <= n_subjects:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(subject_ids.to_numpy(), size=n_subjects, replace=False)
    subset = df[df[ID_COL].astype(str).isin(chosen)].copy()
    return subset.reset_index(drop=True)


def _prepare_pfe_infra(df, categorical_conf, numeric_conf):
    strata = pfe.make_strata(df, diag_col=DIAG_COL, sex_col=SEX_COL)
    splits = pfe._make_repeated_splits(
        strata, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=SEED
    )
    conf_matrix, _ = pfe._fit_confounders_matrix(
        df, categorical=categorical_conf, numeric=numeric_conf
    )
    return splits, conf_matrix


def _run_generic_combo(df, combo, x_cols, conf_cat, conf_num, splits, conf_matrix):
    """Wrapper around pfe._run_repeated_cv_generic (age/sex/exposome path)."""
    fold_df, score_df = pfe._run_repeated_cv_generic(
        df,
        feature_combo=combo,
        x_cols=x_cols,
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
    return score_df


def _save_scores_and_summary(results_dir, combo, score_df, df, extra_metrics=None,
                             extra_summary=None):
    """Write {combo}_(h2|metrics)_per_repeat.csv, subject_scores, all_combos_summary."""
    results_dir.mkdir(parents=True, exist_ok=True)
    combo_safe = combo.label.replace("+", "_")

    h2_per_repeat = (
        score_df.groupby("repeat", group_keys=False)
        .apply(pfe._compute_h2_from_scores)
        .values
    )

    if extra_metrics is None:
        metrics_df = pd.DataFrame({"repeat": range(N_REPEATS), "h2_with": h2_per_repeat})
        metrics_df.to_csv(results_dir / f"{combo_safe}_h2_per_repeat.csv", index=False)
    else:
        cols = {"repeat": range(N_REPEATS), "h2_with": h2_per_repeat}
        cols.update(extra_metrics(h2_per_repeat))
        metrics_df = pd.DataFrame(cols)
        metrics_df.to_csv(results_dir / f"{combo_safe}_metrics_per_repeat.csv", index=False)

    avg_scores = score_df.groupby("subject_id")["score"].mean().reset_index()
    avg_scores.columns = ["subject_id", "score_with"]
    avg_scores.to_csv(results_dir / f"{combo_safe}_subject_scores.csv", index=False)

    summary = {
        "combo_label": combo.label,
        "combo_key": combo.key,
        "n_features": len(combo.columns),
        "n_subjects": df.shape[0],
        "h2_with_mean": float(np.nanmean(h2_per_repeat)),
        "h2_with_std": float(np.nanstd(h2_per_repeat)),
        "n_repeats": N_REPEATS,
        "n_splits": N_SPLITS,
        "seed": SEED,
    }
    if extra_summary is not None:
        summary.update(extra_summary)
    summary_path = results_dir / "all_combos_summary.csv"
    summary_df = pd.DataFrame([summary])
    if summary_path.exists():
        summary_df = pd.concat([pd.read_csv(summary_path), summary_df], ignore_index=True)
        summary_df = summary_df.drop_duplicates(subset=["combo_key", "combo_label"], keep="last")
    summary_df.to_csv(summary_path, index=False)
    return metrics_df, avg_scores, summary


def _compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan


def _run_scanner_pcev_combo(df_base, all_combos, combo_label, required_cols,
                            conf_cat, conf_num, x_cols, results_dir, *, extra_metrics=None,
                            extra_summary=None):
    combo = select_combo(all_combos, combo_label)
    df = df_base[required_cols].dropna().reset_index(drop=True)
    for col in conf_cat:
        df[col] = df[col].astype("category")
    splits, conf_matrix = _prepare_pfe_infra(df, conf_cat, conf_num)
    t0 = time.time()
    score_df = _run_generic_combo(df, combo, x_cols, conf_cat, conf_num, splits, conf_matrix)
    metrics_df, _, _ = _save_scores_and_summary(results_dir, combo, score_df, df,
                                                extra_metrics=extra_metrics,
                                                extra_summary=extra_summary)
    print(f"  Combo: {combo_label} | N={df.shape[0]} | done in {time.time()-t0:.1f}s", flush=True)
    return combo, metrics_df


def run_age(df_base, all_combos, all_neural_cols, *, conf_num=None, results_dir=None):
    print("\n" + "=" * 80 + "\nAGE PCEV: WITH SCANNER PARAMS\n" + "=" * 80, flush=True)
    best_label = best_combo_for_axis("age")
    required_cols = (all_neural_cols
                     + [ID_COL, AGE_COL, SEX_COL, DIAG_COL, COUNTRY_COL, GOF_COL, ODQ_COL]
                     + list(SCANNER_COVS))
    conf_cat = (SEX_COL, COUNTRY_COL, DIAG_COL)
    conf_num = conf_num or (GOF_COL, ODQ_COL, NVOL_COL, DUR_COL)
    results_dir = results_dir or NEW_DIRS["age"]
    outputs = {}
    for label in combo_labels_for_comparison(best_label):
        _, metrics_df = _run_scanner_pcev_combo(
            df_base, all_combos, label, required_cols,
            conf_cat, conf_num, [AGE_COL], results_dir,
        )
        outputs[label] = metrics_df
    return outputs


def run_sex(df_base, all_combos, all_neural_cols, *, conf_num=None, results_dir=None):
    print("\n" + "=" * 80 + "\nSEX PCEV: WITH SCANNER PARAMS\n" + "=" * 80, flush=True)
    best_label = best_combo_for_axis("sex")
    df_base = df_base.copy()
    df_base["Sex_numeric"] = df_base[SEX_COL].map(SEX_MAP)
    df_base = df_base.loc[df_base[SEX_COL].isin(SEX_MAP)].copy()
    required_cols = ([ID_COL, SEX_COL, "Sex_numeric", DIAG_COL, COUNTRY_COL,
                      GOF_COL, AGE_COL, ODQ_COL] + list(SCANNER_COVS) + all_neural_cols)
    conf_cat = (DIAG_COL, COUNTRY_COL)
    conf_num = conf_num or (GOF_COL, AGE_COL, ODQ_COL, NVOL_COL, DUR_COL)
    results_dir = results_dir or NEW_DIRS["sex"]

    outputs = {}
    for label in combo_labels_for_comparison(best_label):
        combo = select_combo(all_combos, label)
        df = df_base[required_cols].dropna().reset_index(drop=True)
        df[DIAG_COL] = df[DIAG_COL].astype("category")
        df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")

        splits, conf_matrix = _prepare_pfe_infra(df, conf_cat, conf_num)
        score_df = _run_generic_combo(df, combo, ["Sex_numeric"], conf_cat, conf_num,
                                      splits, conf_matrix)

        arr = []
        for repeat in range(N_REPEATS):
            rs = score_df[score_df["repeat"] == repeat].merge(
                df[[ID_COL, SEX_COL]], left_on="subject_id", right_on=ID_COL, how="left")
            male = rs[rs[SEX_COL] == "Male"]["score"]
            female = rs[rs[SEX_COL] == "Female"]["score"]
            arr.append(_compute_cohens_d(female, male) if len(male) and len(female) else np.nan)
        arr = np.array(arr)

        metrics_df, _, _ = _save_scores_and_summary(
            results_dir, combo,
            score_df,
            df,
            extra_metrics=lambda _h2, arr=arr: {"cohens_d_with": arr},
            extra_summary={"cohens_d_with_mean": float(np.nanmean(arr)),
                           "cohens_d_with_std": float(np.nanstd(arr))},
        )
        outputs[label] = metrics_df
        print(f"  Combo: {label} | N={df.shape[0]} | done", flush=True)
    return outputs


def run_diagnosis(df_base, all_combos, all_neural_cols, *, conf_num=None, results_dir=None):
    print("\n" + "=" * 80 + "\nDIAGNOSIS PCEV: WITH SCANNER PARAMS\n" + "=" * 80, flush=True)
    best_label = best_combo_for_axis("diagnosis")
    required_cols = ([ID_COL, DIAG_COL, SEX_COL, COUNTRY_COL, AGE_COL, GOF_COL, ODQ_COL]
                     + list(SCANNER_COVS) + all_neural_cols)
    conf_cat = (SEX_COL, COUNTRY_COL)
    conf_num = conf_num or (AGE_COL, GOF_COL, ODQ_COL, NVOL_COL, DUR_COL)
    results_dir = results_dir or NEW_DIRS["diagnosis"]
    outputs = {}
    for label in combo_labels_for_comparison(best_label):
        combo = select_combo(all_combos, label)
        df = df_base[required_cols].dropna().reset_index(drop=True)
        df[DIAG_COL] = df[DIAG_COL].astype("category")
        df[SEX_COL] = df[SEX_COL].astype("category")
        df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")
        conf_matrix, _ = pfe._fit_confounders_matrix(df, categorical=conf_cat, numeric=conf_num)
        strata = pdg.make_strata(df, diag_col=DIAG_COL, sex_col=SEX_COL)
        splits = pdg._make_repeated_splits(strata, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=SEED)
        X_onehot, diag_codes, diag_labels, _ = pdg._prep_target(df, DIAG_COL)
        subject_ids = df[ID_COL].to_numpy()
        t0 = time.time()
        context = pdg.DiagnosisCvContext(
            Y=df[list(combo.columns)].to_numpy(dtype=float), X=X_onehot,
            diag_codes=diag_codes, diag_labels=diag_labels, subject_ids=subject_ids,
            conf_matrix=conf_matrix, splits=splits, perm_strata=strata, n_repeats=N_REPEATS,
        )
        fold_df, score_df = pdg._run_repeated_cv_from_context(
            context, combo=combo, n_jobs=N_JOBS, pcev_kwargs={}, joblib_verbose=10)
        h2_per_repeat = fold_df.groupby("repeat")["h2_test"].mean().values
        score_merged = score_df.merge(df[[ID_COL, DIAG_COL]], left_on="subject_id",
                                      right_on=ID_COL, how="left")
        epsilon = np.array([
            pdg._effect_size_metrics(
                score_merged[score_merged["repeat"] == repeat]["score"].values,
                score_merged[score_merged["repeat"] == repeat][DIAG_COL].values,
            )["epsilon_sq"]
            for repeat in sorted(score_df["repeat"].unique())
        ])
        results_dir.mkdir(parents=True, exist_ok=True)
        combo_safe = combo.label.replace("+", "_")
        pd.DataFrame({"repeat": range(N_REPEATS), "h2_with": h2_per_repeat,
                      "epsilon_with": epsilon}).to_csv(
            results_dir / f"{combo_safe}_metrics_per_repeat.csv", index=False)
        score_df.groupby("subject_id")["score"].mean().reset_index().rename(
            columns={"score": "score_with"}
        ).to_csv(results_dir / f"{combo_safe}_subject_scores.csv", index=False)
        pd.DataFrame([{
            "combo_label": combo.label, "combo_key": combo.key,
            "n_features": len(combo.columns), "n_subjects": df.shape[0],
            "h2_with_mean": float(np.nanmean(h2_per_repeat)),
            "h2_with_std": float(np.nanstd(h2_per_repeat)),
            "epsilon_with_mean": float(np.nanmean(epsilon)),
            "epsilon_with_std": float(np.nanstd(epsilon)),
            "n_repeats": N_REPEATS, "n_splits": N_SPLITS, "seed": SEED,
        }]).to_csv(results_dir / "all_combos_summary.csv", index=False)
        outputs[label] = pd.DataFrame({"repeat": range(N_REPEATS), "h2_with": h2_per_repeat, "epsilon_with": epsilon})
        print(f"  Combo: {label} | N={df.shape[0]} | done in {time.time()-t0:.1f}s", flush=True)
    return outputs


def run_exposome(df_base, all_combos, all_neural_cols, *, conf_num=None, results_base=None):
    print("\n" + "=" * 80 + "\nEXPOSOME PCEV: WITH SCANNER PARAMS\n" + "=" * 80, flush=True)
    conf_num = conf_num or (AGE_COL, GOF_COL, ODQ_COL, NVOL_COL, DUR_COL)
    results_base = results_base or NEW_EXPO_BASE
    outputs = {}
    for group_name, factors_map in EXPOSOME_GROUPS.items():
        if group_name in EXCLUDED_EXPOSOMES:
            continue
        best_label = best_combo_for_exposome(group_name)
        for label in combo_labels_for_comparison(best_label):
            combo = select_combo(all_combos, label)
            print(f"\n[EXPO] {group_name} | combo: {label}", flush=True)

            selected_factors = list(factors_map.values())
            required_cols = (all_neural_cols + selected_factors
                             + [ID_COL, SEX_COL, DIAG_COL, COUNTRY_COL, AGE_COL, GOF_COL, ODQ_COL]
                             + list(SCANNER_COVS))
            df = df_base[required_cols].dropna().reset_index(drop=True)
            df[SEX_COL] = df[SEX_COL].astype("category")
            df[DIAG_COL] = df[DIAG_COL].astype("category")
            df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")

            conf_cat = (SEX_COL, COUNTRY_COL, DIAG_COL)
            splits, conf_matrix = _prepare_pfe_infra(df, conf_cat, conf_num)

            t0 = time.time()
            score_df = _run_generic_combo(df, combo, selected_factors, conf_cat, conf_num,
                                          splits, conf_matrix)
            results_dir = results_base / safe_name(group_name)
            metrics_df, _, _ = _save_scores_and_summary(
                results_dir, combo, score_df, df,
                extra_summary={"exposome_group": group_name,
                               "n_exposome_factors": len(selected_factors)},
            )
            outputs[group_name] = metrics_df
            print(f"  N={df.shape[0]} | done in {time.time()-t0:.1f}s", flush=True)
    return outputs


def run_pcev_smoke_end_to_end(df_base, all_combos, all_neural_cols):
    results_root = SCANNER_RESULTS_ROOT
    baseline_dirs = {
        "age": results_root / "age_odq_comparison_no_scanner",
        "sex": results_root / "sex_odq_comparison_no_scanner",
        "diagnosis": results_root / "diagnosis_odq_comparison_no_scanner",
    }
    baseline_expo_base = results_root / "exposome_odq_comparison_no_scanner"
    scanner_dirs = {
        "age": results_root / "age_with_scanner_params",
        "sex": results_root / "sex_with_scanner_params",
        "diagnosis": results_root / "diagnosis_with_scanner_params",
    }
    scanner_expo_base = results_root / "exposome_with_scanner_params"

    no_scanner_age = (GOF_COL, ODQ_COL)
    no_scanner_sex = (GOF_COL, AGE_COL, ODQ_COL)
    no_scanner_diag = (AGE_COL, GOF_COL, ODQ_COL)
    no_scanner_expo = (AGE_COL, GOF_COL, ODQ_COL)

    baseline_pcev = {
        "age": run_age(df_base, all_combos, all_neural_cols, conf_num=no_scanner_age, results_dir=baseline_dirs["age"]),
        "sex": run_sex(df_base, all_combos, all_neural_cols, conf_num=no_scanner_sex, results_dir=baseline_dirs["sex"]),
        "diagnosis": run_diagnosis(df_base, all_combos, all_neural_cols, conf_num=no_scanner_diag, results_dir=baseline_dirs["diagnosis"]),
        "exposome": run_exposome(df_base, all_combos, all_neural_cols, conf_num=no_scanner_expo, results_base=baseline_expo_base),
    }
    scanner_pcev = {
        "age": run_age(df_base, all_combos, all_neural_cols, conf_num=(GOF_COL, ODQ_COL, NVOL_COL, DUR_COL), results_dir=scanner_dirs["age"]),
        "sex": run_sex(df_base, all_combos, all_neural_cols, conf_num=(GOF_COL, AGE_COL, ODQ_COL, NVOL_COL, DUR_COL), results_dir=scanner_dirs["sex"]),
        "diagnosis": run_diagnosis(df_base, all_combos, all_neural_cols, conf_num=(AGE_COL, GOF_COL, ODQ_COL, NVOL_COL, DUR_COL), results_dir=scanner_dirs["diagnosis"]),
        "exposome": run_exposome(df_base, all_combos, all_neural_cols, conf_num=(AGE_COL, GOF_COL, ODQ_COL, NVOL_COL, DUR_COL), results_base=scanner_expo_base),
    }
    return baseline_pcev, scanner_pcev


def load_pcev_outputs_from_results_root(results_root: Path):
    """Load the best-combo PCEV outputs previously written under the results root."""
    baseline_pcev = {}
    scanner_pcev = {}

    for axis in ("age", "sex", "diagnosis"):
        best_label = best_combo_for_axis(axis)
        combo_key = best_label.replace("+", "_")
        baseline_dir = results_root / f"{axis}_odq_comparison_no_scanner"
        scanner_dir = results_root / f"{axis}_with_scanner_params"
        metric_file = "h2_per_repeat.csv" if axis == "age" else "metrics_per_repeat.csv"
        metric_cols = {
            "age": "h2_with",
            "sex": "cohens_d_with",
            "diagnosis": "epsilon_with",
        }
        metric_col = metric_cols[axis]

        baseline_path = baseline_dir / f"{combo_key}_{metric_file}"
        scanner_path = scanner_dir / f"{combo_key}_{metric_file}"
        baseline_pcev[axis] = {best_label: pd.read_csv(baseline_path)}
        scanner_pcev[axis] = {best_label: pd.read_csv(scanner_path)}
        if metric_col not in baseline_pcev[axis][best_label].columns or metric_col not in scanner_pcev[axis][best_label].columns:
            raise KeyError(f"Missing '{metric_col}' in {baseline_path} or {scanner_path}")

    baseline_pcev["exposome"] = {}
    scanner_pcev["exposome"] = {}
    for expo_name in (e for e in EXPOSOME_GROUPS if e not in EXCLUDED_EXPOSOMES):
        best_label = best_combo_for_exposome(expo_name)
        combo_key = best_label.replace("+", "_")
        baseline_dir = results_root / "exposome_odq_comparison_no_scanner" / safe_name(expo_name)
        scanner_dir = results_root / "exposome_with_scanner_params" / safe_name(expo_name)
        baseline_path = baseline_dir / f"{combo_key}_h2_per_repeat.csv"
        scanner_path = scanner_dir / f"{combo_key}_h2_per_repeat.csv"
        baseline_pcev["exposome"][expo_name] = pd.read_csv(baseline_path)
        scanner_pcev["exposome"][expo_name] = pd.read_csv(scanner_path)

    return baseline_pcev, scanner_pcev


# =============================================================================
# SECTION 2: LME RE-RUN WITH SCANNER COVARIATES
# =============================================================================

# --- LME helpers (verbatim from analysis_04/05) -----------------------------

def fit_mixed_model_with_retry(data, formula, *, reml=False, group_col="country",
                               re_formula="1", vc_formula=None):
    model = smf.mixedlm(formula, data=data, groups=data[group_col],
                        re_formula=re_formula, vc_formula=vc_formula)
    last_fit = None
    for method, maxiter in [("lbfgs", 500), ("powell", 1000)]:
        try:
            fit = model.fit(reml=reml, method=method, maxiter=maxiter)
            last_fit = fit
            if getattr(fit, "converged", True):
                return fit
        except Exception:
            pass
    return last_fit


def nested_variance_components(result) -> dict:
    var_country = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
    var_scanner = 0.0
    if hasattr(result, "vcomp") and hasattr(result.model, "exog_vc"):
        for name, val in zip(result.model.exog_vc.names, result.vcomp):
            if "scanner" in str(name).lower():
                var_scanner = float(val)
                break
    resid = float(result.scale)
    total = var_country + var_scanner + resid
    return {
        "var_country": var_country, "var_scanner": var_scanner, "var_resid": resid,
        "icc_country": var_country / total if total > 0 else np.nan,
        "icc_scanner": var_scanner / total if total > 0 else np.nan,
    }


SEX_NORMALISE_MAP = {"M": "M", "MALE": "M", "F": "F", "FEMALE": "F"}

COVARIATE_RENAME = {
    "N_MEGA": "subject_id", "Age": "age", "Sex": "sex", "Diagnosis": "diagnosis",
    "Country": "country", "gof_corr": "gof_corr", "ODQ_fMRI": "ODQ_fMRI",
    "resonador": "resonador",
}

# Phenotype unified formula extended with scanner covariates.
UNIFIED_FORMULA = ("score_z ~ Age_z + Sex_male + ODQ_z + GOF_z "
                   "+ Dx_AD + Dx_FTD + Dx_MCI + Nvol_z + Dur_z")
UNIFIED_FORMULA_NO_SCANNER = "score_z ~ Age_z + Sex_male + ODQ_z + GOF_z + Dx_AD + Dx_FTD + Dx_MCI"
UNIFIED_TERM_ORDER_NO_SCANNER = {
    "age": ["Age_z", "Sex_male", "ODQ_z", "GOF_z", "Dx_AD", "Dx_FTD", "Dx_MCI"],
    "sex": ["Sex_male", "Age_z", "ODQ_z", "GOF_z", "Dx_AD", "Dx_FTD", "Dx_MCI"],
    "diagnosis": ["Dx_AD", "Dx_FTD", "Dx_MCI", "Age_z", "Sex_male", "ODQ_z", "GOF_z"],
}
UNIFIED_TERM_ORDER = {
    "age": ["Age_z", "Sex_male", "ODQ_z", "GOF_z", "Dx_AD", "Dx_FTD", "Dx_MCI", "Nvol_z", "Dur_z"],
    "sex": ["Sex_male", "Age_z", "ODQ_z", "GOF_z", "Dx_AD", "Dx_FTD", "Dx_MCI", "Nvol_z", "Dur_z"],
    "diagnosis": ["Dx_AD", "Dx_FTD", "Dx_MCI", "Age_z", "Sex_male", "ODQ_z", "GOF_z", "Nvol_z", "Dur_z"],
}


def build_covariates_df(scanner_df):
    cov = (pd.read_csv(DATA_PATH, usecols=list(COVARIATE_RENAME.keys()),
                       dtype={"N_MEGA": str}, low_memory=False)
           .rename(columns=COVARIATE_RENAME)
           .drop_duplicates("subject_id").reset_index(drop=True))
    cov = cov[~cov["country"].str.strip().str.lower().eq("new zeland")]
    cov = cov.merge(scanner_df, on="resonador", how="left")
    return cov


def load_new_scores(axis_dir, combo_label):
    """Load subject scores for a combo from a with-scanner result directory."""
    file_key = combo_label.replace("+", "_")
    score_path = axis_dir / f"{file_key}_subject_scores.csv"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing subject scores for combo '{combo_label}': {score_path}")
    df = (pd.read_csv(score_path)
          .rename(columns={"score_with": "score"}))
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df.dropna(subset=["score"]), file_key


def build_unified_odq_dataset(scores_df, covariates_df) -> pd.DataFrame:
    subject_scores = (scores_df.groupby("subject_id", as_index=False)
                      .agg(mean_score=("score", "mean")))
    merged = subject_scores.merge(covariates_df, on="subject_id", how="left")
    merged = merged.dropna(subset=["age", "sex", "diagnosis", "country", "gof_corr",
                                   "ODQ_fMRI", "resonador", NVOL_COL, DUR_COL])
    merged["sex"] = merged["sex"].astype(str).str.strip().str.upper().map(SEX_NORMALISE_MAP)
    merged = merged[merged["sex"].isin(["M", "F"])]
    merged["diagnosis"] = merged["diagnosis"].astype(str).str.strip().str.upper()
    merged = merged[merged["diagnosis"].isin(["CN", "AD", "FTD", "MCI"])].copy()

    merged["score_z"] = zscore(merged["mean_score"])
    merged["Age_z"] = zscore(merged["age"])
    merged["ODQ_z"] = zscore(pd.to_numeric(merged["ODQ_fMRI"], errors="coerce"))
    merged["GOF_z"] = zscore(merged["gof_corr"])
    merged["Nvol_z"] = zscore(merged[NVOL_COL])
    merged["Dur_z"] = zscore(merged[DUR_COL])
    merged["Sex_male"] = (merged["sex"] == "M").astype(float)

    dx = pd.get_dummies(merged["diagnosis"], prefix="Dx")
    for col in ["Dx_AD", "Dx_FTD", "Dx_MCI"]:
        if col not in dx.columns:
            dx[col] = 0.0
    merged = pd.concat([merged, dx[["Dx_AD", "Dx_FTD", "Dx_MCI"]].astype(float)], axis=1)

    merged["scanner_country"] = (merged["country"].astype(str).str.strip() + "::" +
                                 merged["resonador"].astype(str).str.strip())
    merged["country"] = merged["country"].astype("category")
    merged["scanner_country"] = merged["scanner_country"].astype("category")
    return merged


def run_lme_phenotypes(scanner_df, scores_root: Path | None = None):
    print("\n" + "=" * 80 + "\n[SECTION 2a] LME PHENOTYPES WITH SCANNER COVARIATES\n" + "=" * 80,
          flush=True)
    scores_root = scores_root or SCANNER_RESULTS_ROOT
    covariates_df = build_covariates_df(scanner_df)
    rows_without_scanner = []
    rows_with_scanner = []
    for axis in ("age", "sex", "diagnosis"):
        combo_label = best_combo_for_axis(axis)
        axis_dir = scores_root / f"{axis}_with_scanner_params"
        scores_df, combo_key = load_new_scores(axis_dir, combo_label)
        ds = build_unified_odq_dataset(scores_df, covariates_df)
        fit_no = fit_mixed_model_with_retry(
            ds, UNIFIED_FORMULA_NO_SCANNER,
            reml=False, group_col="country", vc_formula={"scanner": "0 + C(scanner_country)"})
        fit_yes = fit_mixed_model_with_retry(
            ds, UNIFIED_FORMULA, reml=False, group_col="country",
            vc_formula={"scanner": "0 + C(scanner_country)"})
        if fit_no is None or fit_yes is None:
            continue

        var_info_no = nested_variance_components(fit_no)
        var_info_yes = nested_variance_components(fit_yes)
        ci_no = fit_no.conf_int()
        ci_yes = fit_yes.conf_int()
        for term in UNIFIED_TERM_ORDER_NO_SCANNER[axis]:
            term_ci_no = ci_no.loc[term]
            term_ci_yes = ci_yes.loc[term]
            rows_without_scanner.append({
                "phenotype": axis, "combo": combo_key, "n": int(ds.shape[0]),
                "n_country": int(ds["country"].nunique()),
                "n_scanner": int(ds["scanner_country"].nunique()),
                "var_country": var_info_no["var_country"], "var_scanner": var_info_no["var_scanner"],
                "var_resid": var_info_no["var_resid"], "icc_country": var_info_no["icc_country"],
                "icc_scanner": var_info_no["icc_scanner"], "term": term,
                "beta": float(fit_no.fe_params[term]), "se": float(fit_no.bse_fe[term]),
                "z": float(fit_no.fe_params[term] / fit_no.bse_fe[term]),
                "p": float(fit_no.pvalues[term]),
                "ci_low": float(term_ci_no.iloc[0]), "ci_high": float(term_ci_no.iloc[1]),
            })
            rows_with_scanner.append({
                "phenotype": axis, "combo": combo_key, "n": int(ds.shape[0]),
                "n_country": int(ds["country"].nunique()),
                "n_scanner": int(ds["scanner_country"].nunique()),
                "var_country": var_info_yes["var_country"], "var_scanner": var_info_yes["var_scanner"],
                "var_resid": var_info_yes["var_resid"], "icc_country": var_info_yes["icc_country"],
                "icc_scanner": var_info_yes["icc_scanner"], "term": term,
                "beta": float(fit_yes.fe_params[term]), "se": float(fit_yes.bse_fe[term]),
                "z": float(fit_yes.fe_params[term] / fit_yes.bse_fe[term]),
                "p": float(fit_yes.pvalues[term]),
                "ci_low": float(term_ci_yes.iloc[0]), "ci_high": float(term_ci_yes.iloc[1]),
            })
        for term in ["Nvol_z", "Dur_z"]:
            term_ci_yes = ci_yes.loc[term]
            rows_without_scanner.append({
                "phenotype": axis, "combo": combo_key, "n": int(ds.shape[0]),
                "n_country": int(ds["country"].nunique()),
                "n_scanner": int(ds["scanner_country"].nunique()),
                "var_country": var_info_no["var_country"], "var_scanner": var_info_no["var_scanner"],
                "var_resid": var_info_no["var_resid"], "icc_country": var_info_no["icc_country"],
                "icc_scanner": var_info_no["icc_scanner"], "term": term,
                "beta": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            })
            rows_with_scanner.append({
                "phenotype": axis, "combo": combo_key, "n": int(ds.shape[0]),
                "n_country": int(ds["country"].nunique()),
                "n_scanner": int(ds["scanner_country"].nunique()),
                "var_country": var_info_yes["var_country"], "var_scanner": var_info_yes["var_scanner"],
                "var_resid": var_info_yes["var_resid"], "icc_country": var_info_yes["icc_country"],
                "icc_scanner": var_info_yes["icc_scanner"], "term": term,
                "beta": float(fit_yes.fe_params[term]), "se": float(fit_yes.bse_fe[term]),
                "z": float(fit_yes.fe_params[term] / fit_yes.bse_fe[term]),
                "p": float(fit_yes.pvalues[term]),
                "ci_low": float(term_ci_yes.iloc[0]), "ci_high": float(term_ci_yes.iloc[1]),
            })
            rows_with_scanner.append({
                "phenotype": axis, "combo": combo_key, "n": int(ds.shape[0]),
                "n_country": int(ds["country"].nunique()),
                "n_scanner": int(ds["scanner_country"].nunique()),
                "var_country": var_info_yes["var_country"], "var_scanner": var_info_yes["var_scanner"],
                "var_resid": var_info_yes["var_resid"], "icc_country": var_info_yes["icc_country"],
                "icc_scanner": var_info_yes["icc_scanner"], "term": term,
                "beta": float(fit_yes.fe_params[term]), "se": float(fit_yes.bse_fe[term]),
                "z": float(fit_yes.fe_params[term] / fit_yes.bse_fe[term]),
                "p": float(fit_yes.pvalues[term]),
                "ci_low": float(term_ci_yes.iloc[0]), "ci_high": float(term_ci_yes.iloc[1]),
            })
        print(f"  {axis}: N={ds.shape[0]} combo={combo_key}", flush=True)
    out_without = pd.DataFrame(rows_without_scanner)
    out_with = pd.DataFrame(rows_with_scanner)
    out_with.to_csv(TABLE_DIR / "lme_phenotypes_unified_scanner_covariates.csv", index=False)
    print(f"  Saved lme_phenotypes_unified_scanner_covariates.csv ({len(out_with)} rows)", flush=True)
    return out_without, out_with


def run_lme_expotypes(scanner_df, scores_root: Path | None = None):
    print("\n" + "=" * 80 + "\n[SECTION 2b] LME EXPOTYPES WITH SCANNER COVARIATES\n" + "=" * 80,
          flush=True)
    scores_root = scores_root or SCANNER_RESULTS_ROOT
    main_df = pd.read_csv(DATA_PATH, low_memory=False)
    main_df[ID_COL] = main_df[ID_COL].astype(str).str.strip()
    main_df = main_df[~main_df["Country"].str.strip().str.lower().eq("new zeland")]
    main_df = main_df.merge(scanner_df, on="resonador", how="left")

    exposomes = [e for e in EXPOSOME_GROUPS if e not in EXCLUDED_EXPOSOMES]
    results_without_scanner = []
    results_with_scanner = []
    for expo_name in exposomes:
        expo_dir = scores_root / "exposome_with_scanner_params" / safe_name(expo_name)
        combo_label = best_combo_for_exposome(expo_name)
        scores_df, _ = load_new_scores(expo_dir, combo_label)
        scores_avg = scores_df.groupby("subject_id", as_index=False)["score"].mean()
        scores_avg = scores_avg.rename(columns={"score": "score_with"})
        merged = main_df.merge(scores_avg, left_on=ID_COL, right_on="subject_id", how="inner")

        for var_label, var_col in EXPOSOME_GROUPS[expo_name].items():
            cols = ["score_with", var_col, "Age", "Sex", "Diagnosis", "Country",
                    "ODQ_fMRI", "resonador", NVOL_COL, DUR_COL]
            data = merged[cols].dropna().copy()
            if data[var_col].std(ddof=0) == 0 or len(data) < 10:
                continue

            data["pcev_z"] = zscore(data["score_with"])
            data["exposure_z"] = zscore(data[var_col])
            data["Age_z"] = zscore(data["Age"])
            data["ODQ_z"] = zscore(data["ODQ_fMRI"])
            data["Nvol_z"] = zscore(data[NVOL_COL])
            data["Dur_z"] = zscore(data[DUR_COL])
            data["Sex_male"] = (data["Sex"].str.lower() == "male").astype(float)
            data["scanner_country"] = data["Country"].astype(str) + "::" + data["resonador"].astype(str)

            dx = pd.get_dummies(data["Diagnosis"].astype(str), prefix="Dx")
            drop_col = "Dx_CN" if "Dx_CN" in dx.columns else sorted(dx.columns)[0]
            dx = dx.drop(columns=[drop_col])
            data = pd.concat([data, dx], axis=1)

            covariates_no_scanner = ["Age_z", "Sex_male", "ODQ_z"] + list(dx.columns)
            covariates_with_scanner = ["Age_z", "Sex_male", "ODQ_z", "Nvol_z", "Dur_z"] + list(dx.columns)
            formula_base_no = f"pcev_z ~ {' + '.join(covariates_no_scanner)}"
            formula_full_no = f"pcev_z ~ exposure_z + {' + '.join(covariates_no_scanner)}"
            formula_base_sc = f"pcev_z ~ {' + '.join(covariates_with_scanner)}"
            formula_full_sc = f"pcev_z ~ exposure_z + {' + '.join(covariates_with_scanner)}"
            vc_formula = {"scanner": "0 + C(scanner_country)"}

            fit_base_no = fit_mixed_model_with_retry(
                data, formula_base_no, group_col="Country", vc_formula=vc_formula)
            fit_full_no = fit_mixed_model_with_retry(
                data, formula_full_no, group_col="Country", vc_formula=vc_formula)
            fit_base_sc = fit_mixed_model_with_retry(
                data, formula_base_sc, group_col="Country", vc_formula=vc_formula)
            fit_full_sc = fit_mixed_model_with_retry(
                data, formula_full_sc, group_col="Country", vc_formula=vc_formula)
            if fit_base_no is None or fit_full_no is None or fit_base_sc is None or fit_full_sc is None:
                continue

            beta_no = fit_full_no.fe_params["exposure_z"]
            ci_no = fit_full_no.conf_int().loc["exposure_z"]
            delta_ll_no = fit_full_no.llf - fit_base_no.llf
            lrt_stat_no = max(0, 2 * delta_ll_no)
            results_without_scanner.append({
                "Exposome": expo_name, "Variable": var_label, "N": len(data),
                "Beta": beta_no, "Beta_CI_Low": ci_no.iloc[0], "Beta_CI_High": ci_no.iloc[1],
                "LL_Base": fit_base_no.llf, "LL_Full": fit_full_no.llf, "Delta_LL": delta_ll_no,
                "LRT_Stat": lrt_stat_no, "P_LRT": chi2.sf(lrt_stat_no, df=1),
                "AIC_Base": fit_base_no.aic, "AIC_Full": fit_full_no.aic,
                "Delta_AIC": fit_full_no.aic - fit_base_no.aic,
            })

            beta_sc = fit_full_sc.fe_params["exposure_z"]
            ci_sc = fit_full_sc.conf_int().loc["exposure_z"]
            delta_ll_sc = fit_full_sc.llf - fit_base_sc.llf
            lrt_stat_sc = max(0, 2 * delta_ll_sc)
            results_with_scanner.append({
                "Exposome": expo_name, "Variable": var_label, "N": len(data),
                "Beta": beta_sc, "Beta_CI_Low": ci_sc.iloc[0], "Beta_CI_High": ci_sc.iloc[1],
                "LL_Base": fit_base_sc.llf, "LL_Full": fit_full_sc.llf, "Delta_LL": delta_ll_sc,
                "LRT_Stat": lrt_stat_sc, "P_LRT": chi2.sf(lrt_stat_sc, df=1),
                "AIC_Base": fit_base_sc.aic, "AIC_Full": fit_full_sc.aic,
                "Delta_AIC": fit_full_sc.aic - fit_base_sc.aic,
            })
        print(f"  {expo_name}: {len(EXPOSOME_GROUPS[expo_name])} vars processed", flush=True)

    lrt_df_without = pd.DataFrame(results_without_scanner)
    if not lrt_df_without.empty:
        from statsmodels.stats.multitest import multipletests
        lrt_df_without["FDR_q_lrt"] = np.nan
        for expo in lrt_df_without["Exposome"].unique():
            idx = lrt_df_without["Exposome"] == expo
            pvals = lrt_df_without.loc[idx, "P_LRT"].dropna()
            if not pvals.empty:
                _, q, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.001)
                lrt_df_without.loc[pvals.index, "FDR_q_lrt"] = q

    lrt_df_with = pd.DataFrame(results_with_scanner)
    if not lrt_df_with.empty:
        from statsmodels.stats.multitest import multipletests
        lrt_df_with["FDR_q_lrt"] = np.nan
        for expo in lrt_df_with["Exposome"].unique():
            idx = lrt_df_with["Exposome"] == expo
            pvals = lrt_df_with.loc[idx, "P_LRT"].dropna()
            if not pvals.empty:
                _, q, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.001)
                lrt_df_with.loc[pvals.index, "FDR_q_lrt"] = q

    lrt_df_with.to_csv(TABLE_DIR / "expotype_lme_lrt_scanner_covariates.csv", index=False)
    print(f"  Saved expotype_lme_lrt_scanner_covariates.csv ({len(lrt_df_with)} rows)", flush=True)
    return lrt_df_without, lrt_df_with


# =============================================================================
# SECTION 3: COMPARISON WITH vs WITHOUT SCANNER COVARIATES
# =============================================================================

def bootstrap_ci(data, n_boot=10000, statistic="mean"):
    data = np.asarray(data)
    idx = np.random.randint(0, len(data), size=(n_boot, len(data)))
    samples = data[idx]
    boot = np.mean(samples, axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def paired_sign_flip_test(a, b, n_perm=10000):
    diff = np.asarray(a) - np.asarray(b)
    obs = np.mean(diff)
    signs = np.random.choice([-1, 1], size=(n_perm, len(diff)))
    perm = np.mean(signs * diff, axis=1)
    return obs, np.mean(np.abs(perm) >= np.abs(obs))


def paired_cohens_d(without, with_):
    diff = np.asarray(with_) - np.asarray(without)
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else np.nan


def _compare_metric_frames(level, metric_col, baseline_df, scanner_df, combo_label):
    a = baseline_df.sort_values("repeat")[metric_col].to_numpy()
    b = scanner_df.sort_values("repeat")[metric_col].to_numpy()
    paired = len(a) == len(b)
    ci_a_low, ci_a_high = bootstrap_ci(a)
    ci_b_low, ci_b_high = bootstrap_ci(b)
    if paired:
        d = paired_cohens_d(a, b)
        _, p = paired_sign_flip_test(b, a)
        delta_ci_low, delta_ci_high = bootstrap_ci(b - a)
        mean_diff = float(np.mean(b - a))
    else:
        d = (np.mean(b) - np.mean(a)) / np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        p, delta_ci_low, delta_ci_high = np.nan, np.nan, np.nan
        mean_diff = float(np.mean(b) - np.mean(a))
    return {
        "level": level, "metric": metric_col, "metric_label": _metric_display_name(metric_col),
        "combo_without": combo_label, "combo_with": combo_label,
        "n_repeats_without": len(a), "n_repeats_with": len(b),
        "mean_without": float(np.mean(a)), "sd_without": float(np.std(a, ddof=1)),
        "ci_low_without": float(ci_a_low), "ci_high_without": float(ci_a_high),
        "mean_with": float(np.mean(b)), "sd_with": float(np.std(b, ddof=1)),
        "ci_low_with": float(ci_b_low), "ci_high_with": float(ci_b_high),
        "mean_diff": mean_diff, "delta_ci_low": delta_ci_low, "delta_ci_high": delta_ci_high,
        "cohens_d_paired": d, "p_value": p, "paired": paired,
    }


def run_pcev_metric_comparison_from_data(baseline_pcev, scanner_pcev):
    print("\n" + "=" * 80 + "\n[SECTION 3a] PCEV METRIC COMPARISON\n" + "=" * 80, flush=True)
    rows = []
    for axis in ("age", "sex", "diagnosis"):
        best_label = best_combo_for_axis(axis)
        rows.append(_compare_metric_frames(axis, "h2_with", baseline_pcev[axis][best_label], scanner_pcev[axis][best_label], best_label))
        if axis == "sex":
            rows.append(_compare_metric_frames(axis, "cohens_d_with", baseline_pcev[axis][best_label], scanner_pcev[axis][best_label], best_label))
        if axis == "diagnosis":
            rows.append(_compare_metric_frames(axis, "epsilon_with", baseline_pcev[axis][best_label], scanner_pcev[axis][best_label], best_label))
    for expo_name in (e for e in EXPOSOME_GROUPS if e not in EXCLUDED_EXPOSOMES):
        best_label = best_combo_for_exposome(expo_name)
        rows.append(_compare_metric_frames(f"exposome:{expo_name}", "h2_with", baseline_pcev["exposome"][expo_name], scanner_pcev["exposome"][expo_name], best_label))
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "scanner_covariates_pcev_metric_comparison.csv", index=False)
    phenotype_out = out[out["level"].isin(["age", "sex", "diagnosis"])].copy()
    exposome_out = out[out["level"].str.startswith("exposome:", na=False)].copy()
    phenotype_out.to_csv(TABLE_DIR / "scanner_covariates_pcev_metric_comparison_phenotype.csv", index=False)
    exposome_out.to_csv(TABLE_DIR / "scanner_covariates_pcev_metric_comparison_exposome.csv", index=False)
    print(f"  Saved scanner_covariates_pcev_metric_comparison.csv ({len(out)} rows)", flush=True)
    return out


def run_lme_coefficient_comparison_from_data(pheno_without, pheno_with, expo_without, expo_with):
    print("\n" + "=" * 80 + "\n[SECTION 3b] LME COEFFICIENT COMPARISON\n" + "=" * 80, flush=True)
    rows = []
    for axis in pheno_with["phenotype"].unique():
        ow = (pheno_without[pheno_without["phenotype"] == axis]
              .drop_duplicates(subset=["term"], keep="first")
              .set_index("term"))
        nw = (pheno_with[pheno_with["phenotype"] == axis]
              .drop_duplicates(subset=["term"], keep="first")
              .set_index("term"))
        for term in nw.index:
            w = nw.loc[term]
            o = ow.loc[term] if term in ow.index else None
            rows.append(_coef_row("phenotype", axis, term, o, w))

    for key in expo_with.set_index(["Exposome", "Variable"]).index:
        w = expo_with.set_index(["Exposome", "Variable"]).loc[key]
        if key in expo_without.set_index(["Exposome", "Variable"]).index:
            o = expo_without.set_index(["Exposome", "Variable"]).loc[key]
        else:
            o = None
        rows.append(_coef_row_expo(key, o, w))

    out = pd.DataFrame(rows)
    phenotype_out = out[out["level"] == "phenotype"].copy()
    exposome_out = out[out["level"] == "exposome"].copy()
    phenotype_out.to_csv(TABLE_DIR / "scanner_covariates_lme_coefficient_comparison_phenotype.csv", index=False)
    exposome_out.to_csv(TABLE_DIR / "scanner_covariates_lme_coefficient_comparison_exposome.csv", index=False)
    out.to_csv(TABLE_DIR / "scanner_covariates_lme_coefficient_comparison.csv", index=False)
    print(f"  Saved scanner_covariates_lme_coefficient_comparison.csv ({len(out)} rows)", flush=True)
    return out


def _ci_overlap(lo1, hi1, lo2, hi2):
    def _scalar(value):
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            return value.iloc[0]
        return value

    lo1, hi1, lo2, hi2 = map(_scalar, (lo1, hi1, lo2, hi2))
    if any(pd.isna(x) for x in (lo1, hi1, lo2, hi2)):
        return np.nan
    return not (hi1 < lo2 or hi2 < lo1)


def _coef_row(level, key, term, o, w):
    def _scalar(value):
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            return value.iloc[0]
        return value

    bw, lw, hw = map(_scalar, (w["beta"], w["ci_low"], w["ci_high"]))
    if o is None:
        return {"level": level, "key": key, "term": term,
                "beta_without": np.nan, "ci_low_without": np.nan, "ci_high_without": np.nan,
                "beta_with": bw, "ci_low_with": lw, "ci_high_with": hw,
                "beta_delta": np.nan, "ci_overlap": np.nan,
                "sign_changed": np.nan, "sig_changed_p05": np.nan, "new_only": True}
    bo, lo, ho = map(_scalar, (o["beta"], o["ci_low"], o["ci_high"]))
    return {"level": level, "key": key, "term": term,
            "beta_without": bo, "ci_low_without": lo, "ci_high_without": ho,
            "beta_with": bw, "ci_low_with": lw, "ci_high_with": hw,
            "beta_delta": bw - bo, "ci_overlap": _ci_overlap(lo, ho, lw, hw),
            "sign_changed": bool(np.sign(bo) != np.sign(bw)),
            "sig_changed_p05": bool((o["p"] < 0.05) != (w["p"] < 0.05)),
            "new_only": False}


def _coef_row_expo(key, o, w):
    expo, var = key
    def _scalar(value):
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            return value.iloc[0]
        return value

    bw, lw, hw = map(_scalar, (w["Beta"], w["Beta_CI_Low"], w["Beta_CI_High"]))
    if o is None:
        return {"level": "exposome", "key": f"{expo}|{var}", "term": "exposure_z",
                "beta_without": np.nan, "ci_low_without": np.nan, "ci_high_without": np.nan,
                "beta_with": bw, "ci_low_with": lw, "ci_high_with": hw,
                "beta_delta": np.nan, "ci_overlap": np.nan,
                "sign_changed": np.nan, "sig_changed_p05": np.nan, "new_only": True}
    bo, lo, ho = map(_scalar, (o["Beta"], o["Beta_CI_Low"], o["Beta_CI_High"]))
    return {"level": "exposome", "key": f"{expo}|{var}", "term": "exposure_z",
            "beta_without": bo, "ci_low_without": lo, "ci_high_without": ho,
            "beta_with": bw, "ci_low_with": lw, "ci_high_with": hw,
            "beta_delta": bw - bo, "ci_overlap": _ci_overlap(lo, ho, lw, hw),
            "sign_changed": bool(np.sign(bo) != np.sign(bw)),
            "sig_changed_p05": bool((o["P_LRT"] < 0.05) != (w["P_LRT"] < 0.05)),
            "new_only": False}


def write_summary(metric_cmp, coef_cmp):
    buf = _io.StringIO()
    def p(line=""):
        print(line)
        buf.write(line + "\n")

    p("=" * 72)
    p("SCANNER-COVARIATE SENSITIVITY: WITH vs WITHOUT (interpretation)")
    p("=" * 72)

    if not metric_cmp.empty:
        ds = metric_cmp["cohens_d_paired"].abs()
        p(f"\nPCEV metric distributions (paired Cohen's d, with vs without):")
        p(f"  |d| range: {ds.min():.4f} – {ds.max():.4f}  (median {ds.median():.4f})")
        p(f"  comparisons with |d| > 0.2: {(ds > 0.2).sum()} / {len(ds)}")
        p(f"  phenotype rows: {(metric_cmp['level'].isin(['age', 'sex', 'diagnosis'])).sum()}")
        p(f"  exposome rows:   {(metric_cmp['level'].str.startswith('exposome:', na=False)).sum()}")

    if not coef_cmp.empty:
        shared = coef_cmp[coef_cmp["new_only"] == False]
        if not shared.empty:
            md = shared["beta_delta"].abs()
            p(f"\nLME coefficients (shared terms, with vs without):")
            p(f"  max |beta shift|: {md.max():.4f}")
            p(f"  terms changing sign:        {int(shared['sign_changed'].sum())} / {len(shared)}")
            p(f"  terms crossing p<.05:       {int(shared['sig_changed_p05'].sum())} / {len(shared)}")
            n_ov = shared["ci_overlap"].sum()
            p(f"  terms with overlapping CIs: {int(n_ov)} / {len(shared)}")
            p(f"  phenotype rows: {(coef_cmp['level'] == 'phenotype').sum()}")
            p(f"  exposome rows:   {(coef_cmp['level'] == 'exposome').sum()}")

    p("\nInterpretation: small Cohen's d on the PCEV metric distributions and "
      "stable LME signs/significance indicate the downstream results are robust "
      "to N-volumes and scan-duration heterogeneity.")

    (TABLE_DIR / "scanner_covariates_summary.txt").write_text(buf.getvalue())
    print("\n  Saved scanner_covariates_summary.txt", flush=True)


def write_exposome_summary(expo_with: pd.DataFrame):
    buf = _io.StringIO()

    def p(line=""):
        print(line)
        buf.write(line + "\n")

    p("=" * 72)
    p("SCANNER-COVARIATE EXPOSOME SUMMARY")
    p("=" * 72)
    p(f"Rows: {len(expo_with)}")
    if expo_with.empty:
        p("No exposome rows were generated.")
    else:
        p("")
        for expo_name in expo_with["Exposome"].unique():
            block = expo_with[expo_with["Exposome"] == expo_name].copy()
            sig = int((block["P_LRT"] < 0.05).sum())
            fdr = int((block.get("FDR_q_lrt", pd.Series(index=block.index, dtype=float)) < 0.05).sum())
            top = block.loc[block["P_LRT"].idxmin()]
            p(f"{expo_name}:")
            p(f"  variables: {len(block)}")
            p(f"  p<0.05: {sig}")
            p(f"  FDR q<0.05: {fdr}")
            p(f"  top variable: {top['Variable']} | beta={top['Beta']:.4f} | p={top['P_LRT']:.4g} | q={top.get('FDR_q_lrt', np.nan):.4g}")

    (TABLE_DIR / "scanner_covariates_exposome_summary.txt").write_text(buf.getvalue())
    print("  Saved scanner_covariates_exposome_summary.txt", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    scanner_df = load_scanner_table()
    results_root = SCANNER_RESULTS_ROOT
    resume_from_lme = os.environ.get("RESUME_FROM_LME", "0").strip().lower() in {"1", "true", "yes", "on"}

    # Section 1: PCEV
    if resume_from_lme:
        baseline_pcev, scanner_pcev = load_pcev_outputs_from_results_root(results_root)
    else:
        df_base, all_combos, all_neural_cols = load_base_data()
        if SMOKE_TEST:
            df_base = sample_subject_subset(df_base, SMOKE_N_SUBJECTS)
            baseline_pcev, scanner_pcev = run_pcev_smoke_end_to_end(df_base, all_combos, all_neural_cols)
        else:
            baseline_pcev, scanner_pcev = run_pcev_smoke_end_to_end(df_base, all_combos, all_neural_cols)

    # Section 2: LME
    pheno_without, pheno_with = run_lme_phenotypes(scanner_df, scores_root=results_root)
    expo_without, expo_with = run_lme_expotypes(scanner_df, scores_root=results_root)
    write_exposome_summary(expo_with)

    # Section 3: comparison
    metric_cmp = run_pcev_metric_comparison_from_data(baseline_pcev, scanner_pcev)
    coef_cmp = run_lme_coefficient_comparison_from_data(pheno_without, pheno_with, expo_without, expo_with)
    write_summary(metric_cmp, coef_cmp)

    print("\n" + "=" * 80 + "\nALL SECTIONS COMPLETE\n" + "=" * 80, flush=True)


if __name__ == "__main__":
    main()
