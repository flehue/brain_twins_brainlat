#!/usr/bin/env python
# coding: utf-8

# LME analysis for age, sex, and diagnosis phenotypes using PCEV scores.
# Adapted from pcev/pcev_lme_analysis.ipynb; no plotting – outputs CSVs to TABLE_DIR.

from __future__ import annotations

import os
import warnings
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_PATH   = REPO_ROOT / 'data' / 'derived' / 'model_output_plus_exposome_data_v3.csv'
RESULTS_BASE = Path(os.environ.get('PCEV_RESULTS_DIR',
                    REPO_ROOT / 'analysis' / 'results' / 'pcev_results'))
TABLE_DIR   = REPO_ROOT / 'analysis' / 'tables_for_paper'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PHENOTYPE_BEST_TABLE = REPO_ROOT / 'analysis' / 'tables' / 'pcev_phenotypes_best_vs_all.csv'

AXIS_DIRS = {
    'age':       RESULTS_BASE / 'age_odq_only_no_scanner',
    'sex':       RESULTS_BASE / 'sex_odq_only_no_scanner',
    'diagnosis': RESULTS_BASE / 'diagnosis_odq_only_no_scanner',
}

COVARIATE_RENAME = {
    'N_MEGA':    'subject_id',
    'Age':       'age',
    'Sex':       'sex',
    'Diagnosis': 'diagnosis',
    'Country':   'country',
    'gof_corr':  'gof_corr',
    'ODQ_fMRI':  'ODQ_fMRI',
    'resonador': 'resonador',
}
COVARIATE_COLS = list(COVARIATE_RENAME.keys())
ALLOWED_AXES   = tuple(AXIS_DIRS.keys())

MIN_COUNTRY_SIZE = 1
SEX_NORMALISE_MAP = {'M': 'M', 'MALE': 'M', 'F': 'F', 'FEMALE': 'F'}

# ---------------------------------------------------------------------------
# 1. Discover best combos per axis from the analysis_03 summary table
# ---------------------------------------------------------------------------

PHENOTYPE_BEST_SUMMARY = pd.read_csv(PHENOTYPE_BEST_TABLE)


def selected_combo_key_for_axis(axis: str) -> str:
    phenotype_name = {
        'age': 'Age',
        'sex': 'Sex',
        'diagnosis': 'Diagnosis',
    }[axis]
    row_df = PHENOTYPE_BEST_SUMMARY[
        (PHENOTYPE_BEST_SUMMARY['Phenotype'] == phenotype_name)
        & (PHENOTYPE_BEST_SUMMARY['Metric'] == 'h²')
    ]
    if row_df.empty:
        raise ValueError(f'No h² best-subset row found for axis={axis}')
    row = row_df.iloc[0]
    delta_all_minus_best = float(row['Δ (All - Best)'])
    if delta_all_minus_best > 0:
        return 'all_features'
    return str(row['Best Subset']).replace('+', '_')


ANALYSIS_COMBOS: Dict[str, Dict[str, str]] = {}
for axis in AXIS_DIRS:
    ANALYSIS_COMBOS[axis] = {
        'all_features': 'all_features',
        'best_combo':   selected_combo_key_for_axis(axis),
    }

print('Analysis combos:')
for axis, mapping in ANALYSIS_COMBOS.items():
    print(f'  {axis}: {mapping}')

# ---------------------------------------------------------------------------
# 2. Load subject scores
# ---------------------------------------------------------------------------

score_frames = []
for axis, axis_dir in AXIS_DIRS.items():
    for combo_label, file_key in ANALYSIS_COMBOS[axis].items():
        t0 = perf_counter()
        score_file = axis_dir / f'{file_key}_subject_scores.csv'
        df = pd.read_csv(score_file).rename(columns={'score_with': 'score'})
        df['subject_id'] = df['subject_id'].astype(str).str.strip()
        df['score']      = pd.to_numeric(df['score'], errors='coerce')
        df['combo_key']   = file_key
        df['combo_label'] = combo_label
        df['axis']        = axis
        df = df.dropna(subset=['score'])
        score_frames.append(df)
        print(f'[{axis}|{combo_label}] {len(df):,} rows in {perf_counter()-t0:.2f}s')

RAW_SCORES_DF = pd.concat(score_frames, ignore_index=True)
print(f'\nTotal score rows: {RAW_SCORES_DF.shape[0]:,}')
print(RAW_SCORES_DF.groupby(['axis', 'combo_label'])
      .agg(n_subjects=('subject_id', 'nunique')).to_string())

# ---------------------------------------------------------------------------
# 3. Load covariates
# ---------------------------------------------------------------------------

COVARIATES_DF = (
    pd.read_csv(
        DATA_PATH,
        usecols=COVARIATE_COLS,
        dtype={'N_MEGA': str},
        low_memory=False,
    )
    .rename(columns=COVARIATE_RENAME)
    .drop_duplicates('subject_id')
    .reset_index(drop=True)
)
COVARIATES_DF = COVARIATES_DF[~COVARIATES_DF['country'].str.strip().str.lower().eq('new zeland')]
print(f'\nCovariates shape: {COVARIATES_DF.shape}')

# ---------------------------------------------------------------------------
# 4. Utility functions (identical logic to pcev_lme_analysis.ipynb)
# ---------------------------------------------------------------------------

def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean, std = values.mean(), values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - mean) / std


def build_subject_dataset(axis: str, combo_key: str) -> pd.DataFrame:
    subset = RAW_SCORES_DF.loc[
        (RAW_SCORES_DF['axis'] == axis) & (RAW_SCORES_DF['combo_key'] == combo_key)
    ].copy()

    # Aggregate per subject (handles both per-repeat and pre-averaged files)
    subject_scores = (
        subset.groupby('subject_id', as_index=False)
        .agg(mean_score=('score', 'mean'), repeats_seen=('score', 'count'))
    )

    merged = subject_scores.merge(COVARIATES_DF, on='subject_id', how='left')
    merged = merged.dropna(subset=['age', 'sex', 'diagnosis', 'country', 'gof_corr'])

    merged['sex'] = (
        merged['sex'].astype(str).str.strip().str.upper().map(SEX_NORMALISE_MAP)
    )
    merged = merged[merged['sex'].isin(['M', 'F'])]
    merged['diagnosis'] = merged['diagnosis'].astype(str).str.strip().str.upper()
    merged = merged.dropna(subset=['diagnosis'])

    merged['score_z']   = zscore(merged['mean_score'])
    merged['age_z']     = zscore(merged['age'])
    merged['gof_corr_z'] = zscore(merged['gof_corr'])

    country_counts  = merged['country'].value_counts()
    valid_countries = country_counts.index[country_counts >= MIN_COUNTRY_SIZE]
    filtered = merged.loc[merged['country'].isin(valid_countries)].copy()

    if filtered.empty:
        raise ValueError(
            f'No data after country filter (min {MIN_COUNTRY_SIZE}) '
            f'for axis={axis}, combo={combo_key}'
        )

    filtered['sex'] = pd.Categorical(filtered['sex'], categories=['M', 'F'], ordered=False)
    diag_categories = ['CN'] + [c for c in filtered['diagnosis'].unique() if c != 'CN']
    filtered['diagnosis'] = pd.Categorical(
        filtered['diagnosis'], categories=diag_categories, ordered=False
    )
    filtered['country']   = filtered['country'].astype('category')
    filtered['axis']      = axis
    filtered['combo_key'] = combo_key

    filtered.attrs['country_filter'] = {
        'min_country_size':   MIN_COUNTRY_SIZE,
        'countries_before':   int(country_counts.size),
        'countries_after':    int(len(valid_countries)),
        'countries_dropped':  sorted(set(country_counts.index) - set(valid_countries)),
    }
    return filtered


BASELINE_FORMULAS = {
    'age':       'score_z ~ age_z + sex + diagnosis + gof_corr_z',
    'sex':       'score_z ~ sex + age_z + diagnosis + gof_corr_z',
    'diagnosis': 'score_z ~ diagnosis + age_z + sex + gof_corr_z',
}

MODERATION_FORMULAS = {
    'age':       'score_z ~ age_z * sex + age_z * diagnosis + gof_corr_z',
    'sex':       'score_z ~ sex * age_z + sex * diagnosis + gof_corr_z',
    'diagnosis': 'score_z ~ diagnosis * age_z + diagnosis * sex + gof_corr_z',
}


def fit_mixed_model(
    data: pd.DataFrame,
    formula: str,
    *,
    reml: bool = False,
    group_col: str = 'country',
    re_formula: str | None = '1',
):
    model = smf.mixedlm(
        formula, data=data, groups=data[group_col], re_formula=re_formula
    )
    return model.fit(reml=reml)


def compare_models(baseline_result, moderation_result) -> dict:
    lr_stat = 2 * (moderation_result.llf - baseline_result.llf)
    df_diff = len(moderation_result.params) - len(baseline_result.params)
    return {
        'lr_statistic':    lr_stat,
        'df':              df_diff,
        'p_value':         chi2.sf(lr_stat, df_diff),
        'aic_baseline':    baseline_result.aic,
        'aic_moderation':  moderation_result.aic,
        'bic_baseline':    baseline_result.bic,
        'bic_moderation':  moderation_result.bic,
    }


def tidy_fixed_effects(result) -> pd.DataFrame:
    return pd.DataFrame({
        'term':      result.fe_params.index,
        'estimate':  result.fe_params.values,
        'std_error': result.bse_fe.values,
        'z_value':   result.fe_params.values / result.bse_fe.values,
        'p_value':   result.pvalues[result.fe_params.index],
    }).reset_index(drop=True)


def model_diagnostics(result) -> dict:
    re_var    = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else np.nan
    resid_var = float(result.scale)
    total     = re_var + resid_var
    icc       = re_var / total if total > 0 else np.nan
    retvals   = getattr(result, 'mle_retvals', {})
    converged = bool(retvals.get('converged', getattr(result, 'converged', True)))
    return {
        'log_likelihood':       float(result.llf),
        'aic':                  float(result.aic),
        'bic':                  float(result.bic),
        'random_intercept_var': re_var,
        'residual_var':         resid_var,
        'icc':                  icc,
        'converged':            converged,
    }


def model_based_effect_sizes(result) -> pd.DataFrame:
    nobs    = int(getattr(result, 'nobs', getattr(result.model, 'nobs', 0)))
    nparams = len(result.params)
    df_resid = max(1, nobs - nparams)
    rows = []
    for term in result.fe_params.index:
        coef = float(result.fe_params[term])
        se   = float(result.bse_fe[term])
        t    = coef / se if se > 0 else np.nan
        r_partial = t / np.sqrt(t**2 + df_resid) if not np.isnan(t) else np.nan
        d_from_r  = (
            (2 * r_partial) / np.sqrt(1 - r_partial**2)
            if (not np.isnan(r_partial) and abs(r_partial) < 1) else np.nan
        )
        rows.append({'term': term, 'coef': coef, 'std_err': se,
                     't': t, 'r_partial': r_partial, 'd_from_r': d_from_r})
    return pd.DataFrame(rows)


def fit_mixed_model_with_retry(
    data: pd.DataFrame,
    formula: str,
    *,
    reml: bool = False,
    group_col: str = 'country',
    re_formula: str | None = '1',
    vc_formula=None,
):
    model = smf.mixedlm(
        formula, data=data, groups=data[group_col], re_formula=re_formula,
        vc_formula=vc_formula,
    )
    last_fit = None
    last_err = None
    for method, maxiter in [('lbfgs', 500), ('powell', 1000)]:
        try:
            fit = model.fit(reml=reml, method=method, maxiter=maxiter)
            last_fit = fit
            if getattr(fit, 'converged', True):
                return fit
        except Exception as err:
            last_err = err
    if last_fit is not None:
        return last_fit
    if last_err is not None:
        raise last_err
    raise RuntimeError(f'Unable to fit model: {formula}')


def nested_variance_components(result) -> dict:
    var_country = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
    var_scanner = 0.0
    if hasattr(result, 'vcomp') and hasattr(result.model, 'exog_vc'):
        for name, val in zip(result.model.exog_vc.names, result.vcomp):
            if 'scanner' in str(name).lower():
                var_scanner = float(val)
                break
    resid = float(result.scale)
    total = var_country + var_scanner + resid
    return {
        'var_country':  var_country,
        'var_scanner':   var_scanner,
        'var_resid':     resid,
        'icc_country':   var_country / total if total > 0 else np.nan,
        'icc_scanner':   var_scanner / total if total > 0 else np.nan,
    }


def build_unified_odq_dataset(axis: str, combo_key: str) -> pd.DataFrame:
    subset = RAW_SCORES_DF.loc[
        (RAW_SCORES_DF['axis'] == axis) & (RAW_SCORES_DF['combo_key'] == combo_key)
    ].copy()

    subject_scores = (
        subset.groupby('subject_id', as_index=False)
        .agg(mean_score=('score', 'mean'), repeats_seen=('score', 'count'))
    )

    merged = subject_scores.merge(COVARIATES_DF, on='subject_id', how='left')
    merged = merged.dropna(subset=['age', 'sex', 'diagnosis', 'country', 'gof_corr',
                                   'ODQ_fMRI', 'resonador'])

    merged['sex'] = (
        merged['sex'].astype(str).str.strip().str.upper().map(SEX_NORMALISE_MAP)
    )
    merged = merged[merged['sex'].isin(['M', 'F'])]
    merged['diagnosis'] = merged['diagnosis'].astype(str).str.strip().str.upper()
    merged = merged[merged['diagnosis'].isin(['CN', 'AD', 'FTD', 'MCI'])].copy()

    merged['score_z'] = zscore(merged['mean_score'])
    merged['Age_z']   = zscore(merged['age'])
    merged['ODQ_z']   = zscore(pd.to_numeric(merged['ODQ_fMRI'], errors='coerce'))
    merged['GOF_z']   = zscore(merged['gof_corr'])
    merged['Sex_male'] = (merged['sex'] == 'M').astype(float)

    dx_dummies = pd.get_dummies(merged['diagnosis'], prefix='Dx')
    for col in ['Dx_AD', 'Dx_FTD', 'Dx_MCI']:
        if col not in dx_dummies.columns:
            dx_dummies[col] = 0.0
    merged = pd.concat(
        [merged, dx_dummies[['Dx_AD', 'Dx_FTD', 'Dx_MCI']].astype(float)],
        axis=1,
    )

    merged['scanner_country'] = (
        merged['country'].astype(str).str.strip() + '::' +
        merged['resonador'].astype(str).str.strip()
    )
    merged['country'] = merged['country'].astype('category')
    merged['scanner_country'] = merged['scanner_country'].astype('category')
    merged.attrs['nested_meta'] = {
        'n_country': int(merged['country'].nunique()),
        'n_scanner': int(merged['scanner_country'].nunique()),
    }
    return merged


# ---------------------------------------------------------------------------
# 5. Run LME for all axes and combos
# ---------------------------------------------------------------------------

LME_RESULTS: Dict[str, Dict[str, dict]] = {}

for axis in ALLOWED_AXES:
    LME_RESULTS[axis] = {}
    for combo_label, combo_key in ANALYSIS_COMBOS[axis].items():
        print(f'\n{"="*60}')
        print(f'{axis.upper()} | {combo_label}')
        print('='*60)

        ds   = build_subject_dataset(axis, combo_key)
        meta = ds.attrs.get('country_filter', {})
        print(f'N subjects: {ds.shape[0]} | Countries: {meta.get("countries_after", 0)}')

        baseline_model   = fit_mixed_model(ds, BASELINE_FORMULAS[axis])
        moderation_model = fit_mixed_model(ds, MODERATION_FORMULAS[axis])
        comparison       = compare_models(baseline_model, moderation_model)

        tidy_base = tidy_fixed_effects(baseline_model)
        tidy_mod  = tidy_fixed_effects(moderation_model)
        diag_base = model_diagnostics(baseline_model)
        diag_mod  = model_diagnostics(moderation_model)
        es_base   = model_based_effect_sizes(baseline_model)
        es_mod    = model_based_effect_sizes(moderation_model)

        LME_RESULTS[axis][combo_label] = {
            'dataset':                    ds,
            'baseline_model':             baseline_model,
            'moderation_model':           moderation_model,
            'fixed_effects_baseline':     tidy_base,
            'fixed_effects_moderation':   tidy_mod,
            'diagnostics_baseline':       diag_base,
            'diagnostics_moderation':     diag_mod,
            'model_comparison':           comparison,
            'effect_sizes_baseline':      es_base,
            'effect_sizes_moderation':    es_mod,
            'country_meta':               meta,
        }

        sig = ('***' if comparison['p_value'] < 0.001 else
               '**'  if comparison['p_value'] < 0.01  else
               '*'   if comparison['p_value'] < 0.05  else 'ns')
        print(f'LR χ²({comparison["df"]}) = {comparison["lr_statistic"]:.2f}, '
              f'p = {comparison["p_value"]:.4f} {sig}  |  '
              f'ΔAIC = {comparison["aic_moderation"] - comparison["aic_baseline"]:+.1f}')
        print(f'ICC (moderation): {diag_mod["icc"]:.4f} | '
              f'Converged: {diag_mod["converged"]}')

# ---------------------------------------------------------------------------
# 6. Consolidated output tables
# ---------------------------------------------------------------------------

fixed_effects_baseline_rows  = []
fixed_effects_moderation_rows = []
model_comparison_rows        = []
diagnostics_rows             = []
effect_sizes_rows            = []

for axis in ALLOWED_AXES:
    for combo_label, payload in LME_RESULTS[axis].items():
        fe_base = payload['fixed_effects_baseline'].copy()
        fe_base.insert(0, 'model',       'baseline')
        fe_base.insert(0, 'combo_label', combo_label)
        fe_base.insert(0, 'axis',        axis)
        fixed_effects_baseline_rows.append(fe_base)

        fe_mod = payload['fixed_effects_moderation'].copy()
        fe_mod.insert(0, 'model',       'moderation')
        fe_mod.insert(0, 'combo_label', combo_label)
        fe_mod.insert(0, 'axis',        axis)
        fixed_effects_moderation_rows.append(fe_mod)

        comp = payload['model_comparison'].copy()
        comp.update({'axis': axis, 'combo_label': combo_label})
        model_comparison_rows.append(comp)

        for model_tag, diag_dict in [('baseline',   payload['diagnostics_baseline']),
                                      ('moderation', payload['diagnostics_moderation'])]:
            row = diag_dict.copy()
            row.update({'axis': axis, 'combo_label': combo_label, 'model': model_tag})
            diagnostics_rows.append(row)

        es_mod = payload['effect_sizes_moderation'].copy()
        es_mod.insert(0, 'combo_label', combo_label)
        es_mod.insert(0, 'axis',        axis)
        effect_sizes_rows.append(es_mod)

fixed_effects_all  = pd.concat(
    fixed_effects_baseline_rows + fixed_effects_moderation_rows, ignore_index=True
)
model_comparisons  = pd.DataFrame(model_comparison_rows)[
    ['axis', 'combo_label', 'lr_statistic', 'df', 'p_value',
     'aic_baseline', 'aic_moderation', 'bic_baseline', 'bic_moderation']
]
diagnostics_all    = pd.DataFrame(diagnostics_rows)[
    ['axis', 'combo_label', 'model', 'log_likelihood', 'aic', 'bic',
     'random_intercept_var', 'residual_var', 'icc', 'converged']
]
effect_sizes_all   = pd.concat(effect_sizes_rows, ignore_index=True)

key_patterns = ['sex\\[T\\.F\\]', 'diagnosis\\[T\\.(AD|MCI|FTD)\\]', ':', 'age_z']
effect_sizes_key = effect_sizes_all[
    effect_sizes_all['term'].str.contains('|'.join(key_patterns), regex=True)
]

# Save
model_comparisons.to_csv( TABLE_DIR / 'lme_phenotypes_model_comparisons.csv',   index=False)
diagnostics_all.to_csv(   TABLE_DIR / 'lme_phenotypes_diagnostics.csv',          index=False)
fixed_effects_all.to_csv( TABLE_DIR / 'lme_phenotypes_fixed_effects.csv',        index=False)
effect_sizes_all.to_csv(  TABLE_DIR / 'lme_phenotypes_effect_sizes_all.csv',     index=False)
effect_sizes_key.to_csv(  TABLE_DIR / 'lme_phenotypes_effect_sizes_key.csv',     index=False)

print(f'\n{"="*60}')
print(f'All tables saved to: {TABLE_DIR}')
print(f'  lme_phenotypes_model_comparisons.csv  ({len(model_comparisons)} rows)')
print(f'  lme_phenotypes_diagnostics.csv         ({len(diagnostics_all)} rows)')
print(f'  lme_phenotypes_fixed_effects.csv       ({len(fixed_effects_all)} rows)')
print(f'  lme_phenotypes_effect_sizes_all.csv    ({len(effect_sizes_all)} rows)')
print(f'  lme_phenotypes_effect_sizes_key.csv    ({len(effect_sizes_key)} rows)')

# ---------------------------------------------------------------------------
# 6b. Unified ODQ nested table
# ---------------------------------------------------------------------------

UNIFIED_FORMULA = 'score_z ~ Age_z + Sex_male + ODQ_z + GOF_z + Dx_AD + Dx_FTD + Dx_MCI'
UNIFIED_TERM_ORDER = {
    'age': ['Age_z', 'Sex_male', 'ODQ_z', 'GOF_z', 'Dx_AD', 'Dx_FTD', 'Dx_MCI'],
    'sex': ['Sex_male', 'Age_z', 'ODQ_z', 'GOF_z', 'Dx_AD', 'Dx_FTD', 'Dx_MCI'],
    'diagnosis': ['Dx_AD', 'Dx_FTD', 'Dx_MCI', 'Age_z', 'Sex_male', 'ODQ_z', 'GOF_z'],
}
UNIFIED_WHICH = {
    'all_features': 'all_features',
    'best_combo':    'best',
}

unified_rows = []
for axis in ALLOWED_AXES:
    for combo_label, combo_key in ANALYSIS_COMBOS[axis].items():
        ds = build_unified_odq_dataset(axis, combo_key)
        meta = ds.attrs.get('nested_meta', {})
        fit = fit_mixed_model_with_retry(
            ds,
            UNIFIED_FORMULA,
            reml=False,
            group_col='country',
            vc_formula={'scanner': '0 + C(scanner_country)'},
        )
        var_info = nested_variance_components(fit)
        ci = fit.conf_int()

        for term in UNIFIED_TERM_ORDER[axis]:
            term_ci = ci.loc[term]
            unified_rows.append({
                'phenotype': axis,
                'combo': combo_key,
                'which': UNIFIED_WHICH[combo_label],
                'n': int(ds.shape[0]),
                'n_country': int(meta.get('n_country', ds['country'].nunique())),
                'n_scanner': int(meta.get('n_scanner', ds['scanner_country'].nunique())),
                'var_country': var_info['var_country'],
                'var_scanner': var_info['var_scanner'],
                'var_resid': var_info['var_resid'],
                'icc_country': var_info['icc_country'],
                'icc_scanner': var_info['icc_scanner'],
                'param': term,
                'term': term,
                'beta': float(fit.fe_params[term]),
                'se': float(fit.bse_fe[term]),
                'z': float(fit.fe_params[term] / fit.bse_fe[term]),
                'p': float(fit.pvalues[term]),
                'ci_low': float(term_ci.iloc[0]),
                'ci_high': float(term_ci.iloc[1]),
            })

unified_odq_nested = pd.DataFrame(unified_rows)[[
    'phenotype', 'combo', 'which', 'n', 'n_country', 'n_scanner',
    'var_country', 'var_scanner', 'var_resid', 'icc_country', 'icc_scanner',
    'param', 'term', 'beta', 'se', 'z', 'p', 'ci_low', 'ci_high',
]]
unified_csv = TABLE_DIR / 'lme_phenotypes_unified_odq_nested.csv'
unified_odq_nested.to_csv(unified_csv, index=False)
print(f'  lme_phenotypes_unified_odq_nested.csv ({len(unified_odq_nested)} rows)')

# ---------------------------------------------------------------------------
# 7. Literal numeric summary  (printed to stdout and saved to TABLE_DIR)
# ---------------------------------------------------------------------------

import io as _io

_summary_buf = _io.StringIO()

def _p(line=''):
    print(line)
    _summary_buf.write(line + '\n')

_p('\n' + '='*80)
_p('PCEV LME ANALYSIS: LITERAL NUMERIC SUMMARY')
_p('='*80)

_p('\n1. SAMPLE SIZES')
_p('-'*40)
for axis in ALLOWED_AXES:
    for combo_label, payload in LME_RESULTS[axis].items():
        ds   = payload['dataset']
        meta = payload['country_meta']
        _p(f'{axis.upper():10s} | {combo_label:15s} | N={ds.shape[0]:5d} subjects | '
           f'Countries={meta.get("countries_after", 0):2d} | '
           f'Median repeats={ds["repeats_seen"].median():.0f}')

_p('\n2. MODEL COMPARISONS (Baseline vs Moderation)')
_p('-'*40)
for _, row in model_comparisons.iterrows():
    sig = ('***' if row['p_value'] < 0.001 else
           '**'  if row['p_value'] < 0.01  else
           '*'   if row['p_value'] < 0.05  else 'ns')
    delta_aic = row['aic_moderation'] - row['aic_baseline']
    delta_bic = row['bic_moderation'] - row['bic_baseline']
    _p(f'{row["axis"].upper():10s} | {row["combo_label"]:15s} | '
       f'LR χ²({row["df"]:.0f})={row["lr_statistic"]:6.2f}, '
       f'p={row["p_value"]:.4f} {sig:3s} | '
       f'ΔAIC={delta_aic:+6.1f} | ΔBIC={delta_bic:+6.1f}')
    _p('           → ' + ('Prefer MODERATION model' if row['p_value'] < 0.05
                          else 'Prefer BASELINE model'))

_p('\n3. VARIANCE COMPONENTS & ICC')
_p('-'*40)
for _, row in diagnostics_all.iterrows():
    _p(f'{row["axis"].upper():10s} | {row["combo_label"]:15s} | {row["model"]:10s} | '
       f'τ²={row["random_intercept_var"]:7.4f} | '
       f'σ²={row["residual_var"]:7.4f} | '
       f'ICC={row["icc"]:6.4f} | '
       f'Converged={"YES" if row["converged"] else "NO"}')

_p('\n4. KEY EFFECT SIZES (Moderation Model)')
_p('-'*40)
for axis in ALLOWED_AXES:
    _p(f'\n{axis.upper()} AXIS:')
    for combo_label, payload in LME_RESULTS[axis].items():
        _p(f'\n  {combo_label}:')
        es = payload['effect_sizes_moderation']
        if axis == 'age':
            key_main  = ['sex[T.F]', 'diagnosis[T.AD]', 'diagnosis[T.MCI]', 'diagnosis[T.FTD]']
            key_inter = ['age_z:sex[T.F]', 'age_z:diagnosis[T.AD]',
                         'age_z:diagnosis[T.MCI]', 'age_z:diagnosis[T.FTD]']
        elif axis == 'sex':
            key_main  = ['sex[T.F]', 'diagnosis[T.AD]', 'diagnosis[T.MCI]', 'diagnosis[T.FTD]']
            key_inter = ['sex[T.F]:age_z', 'sex[T.F]:diagnosis[T.AD]',
                         'sex[T.F]:diagnosis[T.MCI]', 'sex[T.F]:diagnosis[T.FTD]']
        else:
            key_main  = ['diagnosis[T.AD]', 'diagnosis[T.MCI]', 'diagnosis[T.FTD]', 'sex[T.F]']
            key_inter = ['diagnosis[T.AD]:age_z', 'diagnosis[T.MCI]:age_z',
                         'diagnosis[T.FTD]:age_z', 'diagnosis[T.AD]:sex[T.F]',
                         'diagnosis[T.MCI]:sex[T.F]', 'diagnosis[T.FTD]:sex[T.F]']
        _p('    Main effects:')
        for term in key_main:
            r = es[es['term'] == term]
            if not r.empty:
                r = r.iloc[0]
                _p(f'      {term:30s}: coef={r["coef"]:+7.4f}, '
                   f't={r["t"]:+7.3f}, r={r["r_partial"]:+6.3f}, '
                   f'd={r["d_from_r"]:+7.4f}')
        _p('    Interactions:')
        for term in key_inter:
            r = es[es['term'] == term]
            if not r.empty:
                r = r.iloc[0]
                _p(f'      {term:30s}: coef={r["coef"]:+7.4f}, '
                   f't={r["t"]:+7.3f}, r={r["r_partial"]:+6.3f}, '
                   f'd={r["d_from_r"]:+7.4f}')

_p('\n' + '='*80)
_p('END OF SUMMARY')
_p('='*80)

summary_txt = TABLE_DIR / 'lme_phenotypes_numeric_summary.txt'
summary_txt.write_text(_summary_buf.getvalue(), encoding='utf-8')
print(f'  lme_phenotypes_numeric_summary.txt    (written)')
