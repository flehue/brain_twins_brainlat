#!/usr/bin/env python
# coding: utf-8

"""
Unified PCEV Analysis: Age, Sex, Diagnosis, Exposome (WITH ODQ, No Scanner)

Runs the same analyses as the individual ODQ-only scripts, in a single framework.
No baseline loading, no plots. Outputs per-combo metrics and subject scores, plus
combined summaries per analysis.
"""

import sys
import time
from pathlib import Path
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

sys.path.append(".")
import pcev_feature_effects as pfe
import pcev_diagnosis as pdg

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "derived" / "model_output_plus_exposome_data_v2.csv"
RESULTS_BASE = Path(os.environ.get("PCEV_RESULTS_DIR", REPO_ROOT / "analysis" / "results" / "pcev_results"))

# Column names
ID_COL = "N_MEGA"
SEX_COL = "Sex"
DIAG_COL = "Diagnosis"
COUNTRY_COL = "Country"
GOF_COL = "gof_corr"
AGE_COL = "Age"
ODQ_COL = "ODQ_fMRI"

# CV parameters
N_REPEATS = 100
N_SPLITS = 5
SEED = 2025
N_JOBS = -1

# Sex mapping
SEX_MAP = {"Male": 0.0, "Female": 1.0}

# =============================================================================
# EXPOSOME GROUPS
# =============================================================================

EXPOSOME_GROUPS = OrderedDict({
    "Air Pollution": OrderedDict({
        "PM2.5": "PM2.5_interpolated",
        "Nitrogen oxides (NOx)": "Nitrogen oxide (NOx)_interpolated",
        "Sulfur dioxide (SO2)": "Sulphur dioxide (SO2) emissions_interpolated",
        "Carbon monoxide (CO)": "Carbon monoxide (CO) emissions_interpolated",
        "Black carbon (BC)": "Black carbon (BC) emissions_interpolated",
        "Ammoniac Nitrogen (NH3)": "Ammonia (NH3) emissions_interpolated",
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

# =============================================================================
# SHARED DATA LOADING
# =============================================================================

def load_base_data():
    print("\n[GLOBAL] LOADING DATA", flush=True)
    print("-" * 80, flush=True)

    print("  Loading main dataset...", flush=True)
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns", flush=True)

    print("\n  Building neural feature combinations...", flush=True)
    feature_map = pfe.make_feature_map(df_raw, prefixes=pfe.FEATURE_PREFIXES)
    all_combos = pfe.build_combinations(feature_map)
    print(f"  Created {len(all_combos)} feature combinations", flush=True)

    all_neural_cols = sorted({col for cols in feature_map.values() for col in cols})

    return df_raw, all_combos, all_neural_cols


def _prepare_pfe_infra(df, categorical_conf, numeric_conf):
    strata = pfe.make_strata(df, diag_col=DIAG_COL, sex_col=SEX_COL)
    splits = pfe._make_repeated_splits(
        strata, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=SEED
    )
    conf_matrix, _ = pfe._fit_confounders_matrix(
        df, categorical=categorical_conf, numeric=numeric_conf
    )
    return splits, conf_matrix


# =============================================================================
# AGE ANALYSIS
# =============================================================================

def run_age(df_base, all_combos, all_neural_cols):
    analysis_name = "age"
    results_dir = RESULTS_BASE / f"{analysis_name}_odq_only_no_scanner"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80, flush=True)
    print("AGE PCEV: WITH ODQ (NO SCANNER)", flush=True)
    print("=" * 80, flush=True)

    required_cols = (
        all_neural_cols
        + [ID_COL, AGE_COL, SEX_COL, DIAG_COL, COUNTRY_COL, GOF_COL, ODQ_COL]
    )
    df = df_base[required_cols].dropna().reset_index(drop=True)
    df[SEX_COL] = df[SEX_COL].astype("category")
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")

    conf_cat = (SEX_COL, COUNTRY_COL, DIAG_COL)
    conf_num = (GOF_COL, ODQ_COL)
    splits, conf_matrix = _prepare_pfe_infra(df, conf_cat, conf_num)

    all_summaries = []
    for idx, combo in enumerate(all_combos, 1):
        print(
            f"\n[AGE COMBO {idx}/{len(all_combos)}] {combo.label} ({len(combo.columns)} features)",
            flush=True,
        )
        start = time.time()

        fold_df, score_df = pfe._run_repeated_cv_generic(
            df,
            feature_combo=combo,
            x_cols=[AGE_COL],
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
            .apply(pfe._compute_h2_from_scores, include_groups=False)
            .values
        )

        combo_safe = combo.label.replace("+", "_")
        h2_df = pd.DataFrame({"repeat": range(N_REPEATS), "h2_with": h2_per_repeat})
        h2_df.to_csv(results_dir / f"{combo_safe}_h2_per_repeat.csv", index=False)

        avg_scores = score_df.groupby("subject_id")["score"].mean().reset_index()
        avg_scores.columns = ["subject_id", "score_with"]
        avg_scores.to_csv(results_dir / f"{combo_safe}_subject_scores.csv", index=False)

        summary = pd.DataFrame({
            "combo_label": [combo.label],
            "combo_key": [combo.key],
            "n_features": [len(combo.columns)],
            "n_subjects": [df.shape[0]],
            "h2_with_mean": [h2_per_repeat.mean()],
            "h2_with_std": [h2_per_repeat.std()],
            "n_repeats": [N_REPEATS],
            "n_splits": [N_SPLITS],
            "seed": [SEED],
        })
        all_summaries.append(summary)

        elapsed = time.time() - start
        print(f"    Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(results_dir / "all_combos_summary.csv", index=False)


# =============================================================================
# SEX ANALYSIS
# =============================================================================

def _compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan


def run_sex(df_base, all_combos, all_neural_cols):
    analysis_name = "sex"
    results_dir = RESULTS_BASE / f"{analysis_name}_odq_only_no_scanner"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80, flush=True)
    print("SEX PCEV: WITH ODQ (NO SCANNER)", flush=True)
    print("=" * 80, flush=True)

    df_base = df_base.copy()
    df_base["Sex_numeric"] = df_base[SEX_COL].map(SEX_MAP)
    valid_sex = df_base[SEX_COL].isin(SEX_MAP)
    if not valid_sex.all():
        print(f"  Removing {(~valid_sex).sum()} rows with invalid sex values", flush=True)
    df_base = df_base.loc[valid_sex].copy()

    required_cols = [
        ID_COL,
        SEX_COL,
        "Sex_numeric",
        DIAG_COL,
        COUNTRY_COL,
        GOF_COL,
        AGE_COL,
        ODQ_COL,
    ] + all_neural_cols
    df = df_base[required_cols].dropna().reset_index(drop=True)
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")

    conf_cat = (DIAG_COL, COUNTRY_COL)
    conf_num = (GOF_COL, AGE_COL, ODQ_COL)
    splits, conf_matrix = _prepare_pfe_infra(df, conf_cat, conf_num)

    all_summaries = []
    for idx, combo in enumerate(all_combos, 1):
        print(
            f"\n[SEX COMBO {idx}/{len(all_combos)}] {combo.label} ({len(combo.columns)} features)",
            flush=True,
        )
        start = time.time()

        fold_df, score_df = pfe._run_repeated_cv_generic(
            df,
            feature_combo=combo,
            x_cols=["Sex_numeric"],
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
            .apply(pfe._compute_h2_from_scores, include_groups=False)
            .values
        )

        cohens_d_per_repeat = []
        for repeat in range(N_REPEATS):
            repeat_scores = score_df[score_df["repeat"] == repeat].merge(
                df[[ID_COL, SEX_COL]],
                left_on="subject_id",
                right_on=ID_COL,
                how="left",
            )
            male_scores = repeat_scores[repeat_scores[SEX_COL] == "Male"]["score"]
            female_scores = repeat_scores[repeat_scores[SEX_COL] == "Female"]["score"]
            if len(male_scores) > 0 and len(female_scores) > 0:
                cohens_d_per_repeat.append(_compute_cohens_d(female_scores, male_scores))
            else:
                cohens_d_per_repeat.append(np.nan)
        cohens_d_per_repeat = np.array(cohens_d_per_repeat)

        combo_safe = combo.label.replace("+", "_")
        metrics_df = pd.DataFrame({
            "repeat": range(N_REPEATS),
            "h2_with": h2_per_repeat,
            "cohens_d_with": cohens_d_per_repeat,
        })
        metrics_df.to_csv(results_dir / f"{combo_safe}_metrics_per_repeat.csv", index=False)

        avg_scores = score_df.groupby("subject_id")["score"].mean().reset_index()
        avg_scores.columns = ["subject_id", "score_with"]
        avg_scores.to_csv(results_dir / f"{combo_safe}_subject_scores.csv", index=False)

        summary = pd.DataFrame({
            "combo_label": [combo.label],
            "combo_key": [combo.key],
            "n_features": [len(combo.columns)],
            "n_subjects": [df.shape[0]],
            "h2_with_mean": [h2_per_repeat.mean()],
            "h2_with_std": [h2_per_repeat.std()],
            "cohens_d_with_mean": [cohens_d_per_repeat.mean()],
            "cohens_d_with_std": [cohens_d_per_repeat.std()],
            "n_repeats": [N_REPEATS],
            "n_splits": [N_SPLITS],
            "seed": [SEED],
        })
        all_summaries.append(summary)

        elapsed = time.time() - start
        print(f"    Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(results_dir / "all_combos_summary.csv", index=False)


# =============================================================================
# DIAGNOSIS ANALYSIS
# =============================================================================

def _make_confounder_transformer(categorical_conf, numeric_conf):
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_conf),
            ),
            ("numeric", "passthrough", list(numeric_conf)),
        ],
        remainder="drop",
    )
    return encoder


def _fit_confounders_matrix(df, categorical, numeric):
    confounder_cols = list(categorical) + list(numeric)
    transformer = _make_confounder_transformer(categorical, numeric)
    matrix = transformer.fit_transform(df[confounder_cols])
    return np.asarray(matrix, dtype=float), transformer


def _compute_epsilon_per_repeat(score_df: pd.DataFrame, df_full: pd.DataFrame):
    epsilon_list = []
    kw_stats_list = []

    score_df_merged = score_df.merge(
        df_full[[ID_COL, DIAG_COL]],
        left_on="subject_id",
        right_on=ID_COL,
        how="left",
    )

    for repeat in sorted(score_df["repeat"].unique()):
        repeat_scores = score_df_merged[score_df_merged["repeat"] == repeat]
        kw_metrics = pdg._effect_size_metrics(
            repeat_scores["score"].values, repeat_scores[DIAG_COL].values
        )
        epsilon_list.append(kw_metrics["epsilon_sq"])
        kw_stats_list.append(kw_metrics)

    return np.array(epsilon_list), kw_stats_list


def run_diagnosis(df_base, all_combos, all_neural_cols):
    analysis_name = "diagnosis"
    results_dir = RESULTS_BASE / f"{analysis_name}_odq_only_no_scanner"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80, flush=True)
    print("DIAGNOSIS PCEV: WITH ODQ (NO SCANNER)", flush=True)
    print("=" * 80, flush=True)

    required_cols = [
        ID_COL,
        DIAG_COL,
        SEX_COL,
        COUNTRY_COL,
        AGE_COL,
        GOF_COL,
        ODQ_COL,
    ] + all_neural_cols
    df = df_base[required_cols].dropna().reset_index(drop=True)
    df[DIAG_COL] = df[DIAG_COL].astype("category")
    df[SEX_COL] = df[SEX_COL].astype("category")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")

    conf_cat = (SEX_COL, COUNTRY_COL)
    conf_num = (AGE_COL, GOF_COL, ODQ_COL)
    conf_matrix, _ = _fit_confounders_matrix(df, conf_cat, conf_num)

    strata = pdg.make_strata(df, diag_col=DIAG_COL, sex_col=SEX_COL)
    splits = pdg._make_repeated_splits(
        strata, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=SEED
    )

    X_onehot, diag_codes, diag_labels, _ = pdg._prep_target(df, DIAG_COL)
    subject_ids = df[ID_COL].to_numpy()

    all_summaries = []
    for idx, combo in enumerate(all_combos, 1):
        print(
            f"\n[DIAG COMBO {idx}/{len(all_combos)}] {combo.label} ({len(combo.columns)} features)",
            flush=True,
        )
        start = time.time()

        Y = df[list(combo.columns)].to_numpy(dtype=float)
        context = pdg.DiagnosisCvContext(
            Y=Y,
            X=X_onehot,
            diag_codes=diag_codes,
            diag_labels=diag_labels,
            subject_ids=subject_ids,
            conf_matrix=conf_matrix,
            splits=splits,
            perm_strata=strata,
            n_repeats=N_REPEATS,
        )

        fold_df, score_df = pdg._run_repeated_cv_from_context(
            context,
            combo=combo,
            n_jobs=N_JOBS,
            pcev_kwargs={},
            joblib_verbose=10,
        )

        h2_per_repeat = fold_df.groupby("repeat")["h2_test"].mean().values
        del fold_df

        epsilon_per_repeat, _kw_stats = _compute_epsilon_per_repeat(score_df, df)

        combo_safe = combo.label.replace("+", "_")
        metrics_df = pd.DataFrame({
            "repeat": range(N_REPEATS),
            "h2_with": h2_per_repeat,
            "epsilon_with": epsilon_per_repeat,
        })
        metrics_df.to_csv(results_dir / f"{combo_safe}_metrics_per_repeat.csv", index=False)

        avg_scores = score_df.groupby("subject_id")["score"].mean().reset_index()
        avg_scores.columns = ["subject_id", "score_with"]
        avg_scores.to_csv(results_dir / f"{combo_safe}_subject_scores.csv", index=False)

        summary = pd.DataFrame({
            "combo_label": [combo.label],
            "combo_key": [combo.key],
            "n_features": [len(combo.columns)],
            "n_subjects": [df.shape[0]],
            "h2_with_mean": [np.nanmean(h2_per_repeat)],
            "h2_with_std": [np.nanstd(h2_per_repeat)],
            "epsilon_with_mean": [np.nanmean(epsilon_per_repeat)],
            "epsilon_with_std": [np.nanstd(epsilon_per_repeat)],
            "n_repeats": [N_REPEATS],
            "n_splits": [N_SPLITS],
            "seed": [SEED],
        })
        all_summaries.append(summary)

        elapsed = time.time() - start
        print(f"    Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(results_dir / "all_combos_summary.csv", index=False)


# =============================================================================
# EXPOSOME ANALYSIS
# =============================================================================

def run_exposome(df_base, all_combos, all_neural_cols):
    analysis_name = "exposome"
    print("\n" + "=" * 80, flush=True)
    print("EXPOSOME PCEV: WITH ODQ (NO SCANNER)", flush=True)
    print("=" * 80, flush=True)

    for group_name, factors_map in EXPOSOME_GROUPS.items():
        print("\n" + "-" * 80, flush=True)
        print(f"[EXPOSOME GROUP] {group_name}", flush=True)
        print("-" * 80, flush=True)

        selected_factors = list(factors_map.values())
        missing = [f for f in selected_factors if f not in df_base.columns]
        if missing:
            raise ValueError(f"Missing exposome factors in data for {group_name}: {missing}")

        required_cols = (
            all_neural_cols
            + selected_factors
            + [ID_COL, SEX_COL, DIAG_COL, COUNTRY_COL, AGE_COL, GOF_COL, ODQ_COL]
        )
        df = df_base[required_cols].dropna().reset_index(drop=True)
        df[SEX_COL] = df[SEX_COL].astype("category")
        df[DIAG_COL] = df[DIAG_COL].astype("category")
        df[COUNTRY_COL] = df[COUNTRY_COL].astype("category")

        conf_cat = (SEX_COL, COUNTRY_COL, DIAG_COL)
        conf_num = (AGE_COL, GOF_COL, ODQ_COL)
        splits, conf_matrix = _prepare_pfe_infra(df, conf_cat, conf_num)

        results_dir = (
            RESULTS_BASE
            / f"{analysis_name}_odq_only_no_scanner"
            / f"{group_name.replace(' ', '_')}"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        all_summaries = []
        for idx, combo in enumerate(all_combos, 1):
            print(
                f"\n[EXPO COMBO {idx}/{len(all_combos)}] {combo.label} ({len(combo.columns)} features)",
                flush=True,
            )
            start = time.time()

            fold_df, score_df = pfe._run_repeated_cv_generic(
                df,
                feature_combo=combo,
                x_cols=selected_factors,
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
                .apply(pfe._compute_h2_from_scores, include_groups=False)
                .values
            )

            combo_safe = combo.label.replace("+", "_")
            h2_df = pd.DataFrame({"repeat": range(N_REPEATS), "h2_with": h2_per_repeat})
            h2_df.to_csv(results_dir / f"{combo_safe}_h2_per_repeat.csv", index=False)

            avg_scores = score_df.groupby("subject_id")["score"].mean().reset_index()
            avg_scores.columns = ["subject_id", "score_with"]
            avg_scores.to_csv(results_dir / f"{combo_safe}_subject_scores.csv", index=False)

            summary = pd.DataFrame({
                "exposome_group": [group_name],
                "combo_label": [combo.label],
                "combo_key": [combo.key],
                "n_features": [len(combo.columns)],
                "n_exposome_factors": [len(selected_factors)],
                "n_subjects": [df.shape[0]],
                "h2_with_mean": [h2_per_repeat.mean()],
                "h2_with_std": [h2_per_repeat.std()],
                "n_repeats": [N_REPEATS],
                "n_splits": [N_SPLITS],
                "seed": [SEED],
            })
            all_summaries.append(summary)

            elapsed = time.time() - start
            print(f"    Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv(results_dir / "all_combos_summary.csv", index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_base, all_combos, all_neural_cols = load_base_data()

    run_age(df_base, all_combos, all_neural_cols)
    run_sex(df_base, all_combos, all_neural_cols)
    run_diagnosis(df_base, all_combos, all_neural_cols)
    run_exposome(df_base, all_combos, all_neural_cols)

    print("\n" + "=" * 80, flush=True)
    print("ALL ANALYSES COMPLETE", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
