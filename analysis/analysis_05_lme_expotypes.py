#!/usr/bin/env python
# coding: utf-8

# LME analysis for exposome domains using PCEV scores.
# Analysis 1: Bayes factor h² comparison (best combo vs all features).
# Analysis 2: LRT for each exposome variable.
# No figures — all outputs (CSVs + txt summary) saved to TABLE_DIR.

from __future__ import annotations

import io as _io
import warnings
from collections import OrderedDict
from pathlib import Path
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).resolve().parents[1]
DATA_PATH    = REPO_ROOT / 'data' / 'derived' / 'model_output_plus_exposome_data_v3.csv'
RESULTS_BASE = Path(os.environ.get(
    'PCEV_RESULTS_DIR',
    REPO_ROOT / 'analysis' / 'results' / 'pcev_results'
)) / 'exposome_odq_only_no_scanner'
TABLE_DIR    = REPO_ROOT / 'analysis' / 'tables_for_paper'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
EXPOSOME_BEST_TABLE = REPO_ROOT / 'analysis' / 'tables' / 'pcev_expotypes_best_vs_all.csv'

ID_COL = 'N_MEGA'

# ---------------------------------------------------------------------------
# Exposome definitions (unchanged)
# ---------------------------------------------------------------------------

EXPOSOME_GROUPS = OrderedDict({
    'Air Pollution': OrderedDict({
        'PM2.5': 'PM2.5_interpolated',
        'Nitrogen oxides (NOx)': 'Nitrogen oxide (NOx)_interpolated',
        'Sulfur dioxide (SO₂)': 'Sulphur dioxide (SO₂) emissions_interpolated',
        'Carbon monoxide (CO)': 'Carbon monoxide (CO) emissions_interpolated',
        'Black carbon (BC)': 'Black carbon (BC) emissions_interpolated',
        'Ammoniac Nitrogen (NH3)': 'Ammonia (NH₃) emissions_interpolated',
        'Non-methane volatile organic compounds (NMVOC)': 'Non-methane volatile organic compounds (NMVOC) emissions_interpolated',
    }),
    'Green space access': OrderedDict({
        'Urban green area (%) 1990': 'Average share of green area in city/urban area 1990 (%)_interpolated',
        'Urban green area (%) 2000': 'Average share of green area in city/ urban area 2000 (%)_interpolated',
        'Urban green area (%) 2010': 'Average share of green area in city/ urban area 2010 (%)_interpolated',
        'Urban green area (%) 2020': 'Average share of green area in city/ urban area 2020 (%)_interpolated',
        'Green area per capita (m²/person) 1990': 'Green area per capita 1990 (m2/person)_interpolated',
        'Green area per capita (m²/person) 2000': 'Green area per capita 2000 (m2/person)_interpolated',
        'Green area per capita (m²/person) 2010': 'Green area per capita 2010 (m2/person)_interpolated',
        'Green area per capita (m²/person) 2020': 'Green area per capita 2020 (m2/person)_interpolated',
    }),
    'Temperature': OrderedDict({
        'Mean temperature': 'mean_temp_areaw_o_interpolated',
        'Mean temperature pop-weighted': 'mean_temp_o_interpolated',
        'Mean temperature anomalies': 'mean_anomalies_areaw_o_interpolated',
        'Mean temperature anomalies pop-weighted': 'mean_anomalies_o_interpolated',
        'Deviation of temperature anomalies': 'sd_lr_o_interpolated',
        'Max temperature': 'maxgtemp_o_interpolated',
    }),
    'Precipitation-droughts': OrderedDict({
        'Mean precipitation': 'mean_prec2_areaw_o_interpolated',
        'Mean precipitation pop-weighted': 'mean_prec2_o_interpolated',
        'Palmer drought severity index': 'scpdsi_aw_o_interpolated',
        'Palmer drought severity index pop-weighted': 'scpdsi_o_interpolated',
    }),
    'Soil and water quality': OrderedDict({
        'Poisoning mortality rate': 'Poisoning_mortality_rate_interpolated',
        'Basic drinking water access': 'Pop_basic_drinking-water(%)_interpolated',
        'Safely-managed drinking water access': 'Pop_safely_drinking-water(%)_interpolated',
        'Agriculture employment rate': 'agri_emp_o_interpolated',
    }),
    'Climate disasters': OrderedDict({
        'Number of disaster events': 'climatedisaster_count_o_interpolated',
        'Population affected': 'climatedisaster_naffected_o_interpolated',
    }),
    'Disease-related mortality': OrderedDict({
        'Non-communicable death rate (%)': 'deaths_notrans_interpolated',
        'Communicable death rate (%)': 'deaths_trans_interpolated',
    }),
    'Socioeconomic': OrderedDict({
        'Gini index': 'GINI_interpolated',
        'Human capital index': 'HCI_interpolated',
        'Human development index (HDI)': 'HDI_interpolated',
        'Inequality-Adjusted HDI': 'IHDI_interpolated',
        'Multidimensional poverty measures': 'MPM_interpolated',
        'Energy demand': 'Electricity_demand_interpolated',
        'Gender inequality index': 'GII_interpolated',
        'Unemployment rate': 'unemp_interpolated',
        'GDP per capita': 'GDP_interpolated',
    }),
    'Democracy': OrderedDict({
        'Credible elections': 'cred_elect_est_interpolated',
        'Inclusive suffrage': 'inclu_suff_est_interpolated',
        'Free parties': 'free_parties_est_interpolated',
        'Elected government': 'elected_gov_est_interpolated',
        'Effective parliament': 'effect_parl_est_interpolated',
        'Local democracy': 'local_dem_est_interpolated',
        'Access to justice': 'access_just_est_interpolated',
        'Civil liberties': 'civil_lib_est_interpolated',
        'Freedom of expression (Civil liberties)': 'free_express_est_interpolated',
        'Freedom of the press (Civil liberties)': 'free_press_est_interpolated',
        'Freedom of association and assembly (Civil liberties)': 'free_assoc_assem_est_interpolated',
        'Freedom of religion (Civil liberties)': 'free_relig_est_interpolated',
        'Freedom of movement (Civil liberties)': 'free_move_est_interpolated',
        'Political equality': 'pol_equal_est_interpolated',
        'Social group equality (Political equality)': 'soc_grp_equal_est_interpolated',
        'Economic equality (Political equality)': 'econ_equal_est_interpolated',
        'Gender equality (Political equality)': 'gender_equal_est_interpolated',
        'Judicial independence': 'jud_ind_est_interpolated',
        'Absence of Corruption': 'abs_corrupt_est_interpolated',
        'Predictability enforcement': 'predict_enf_est_interpolated',
        'Personal integrity and security': 'pers_integ_sec_est_interpolated',
        'Civil society': 'civil_soc_est_interpolated',
        'Civic engagement': 'civic_engage_est_interpolated',
        'Electoral participation': 'elect_part_est_interpolated',
        'Direct democracy': 'direct_dem_est_interpolated',
    }),
    'Democracy-Summary': OrderedDict({
        'Political representation': 'representation_est_interpolated',
        'Rights': 'rights_est_interpolated',
        'Basic welfare': 'basic_welf_est_interpolated',
        'Rule of law': 'rule_law_est_interpolated',
        'Participation': 'participation_est_interpolated',
    }),
    'Migration': OrderedDict({
        'Inbound migration rate': 'migration_interpolated',
    }),
})

excluded_exposomes = {'Democracy-Summary'}
exposomes_to_process = [e for e in EXPOSOME_GROUPS.keys() if e not in excluded_exposomes]


def safe_name(name: str) -> str:
    return name.replace(' ', '_')


EXPOSOME_BEST_SUMMARY = pd.read_csv(EXPOSOME_BEST_TABLE)


def selected_combo_key_for_expo(expo_name: str) -> str:
    row_df = EXPOSOME_BEST_SUMMARY[
        (EXPOSOME_BEST_SUMMARY['Exposome'] == expo_name)
        & (EXPOSOME_BEST_SUMMARY['Metric'] == 'h²')
    ]
    if row_df.empty:
        raise ValueError(f'No h² best-subset row found for exposome={expo_name}')
    row = row_df.iloc[0]
    delta_all_minus_best = float(row['Δ (All - Best)'])
    if delta_all_minus_best > 0:
        return 'all_features'
    return str(row['Best Subset']).replace('+', '_')


# ---------------------------------------------------------------------------
# Load main data
# ---------------------------------------------------------------------------

main_df = pd.read_csv(DATA_PATH, low_memory=False)
main_df[ID_COL] = main_df[ID_COL].astype(str).str.strip()
main_df = main_df[~main_df['Country'].str.strip().str.lower().eq('new zeland')]
print(f'main_df shape after NZ exclusion: {main_df.shape}')

# ---------------------------------------------------------------------------
# Numeric summary helpers
# ---------------------------------------------------------------------------

_buf = _io.StringIO()

def _p(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_buf)


# ---------------------------------------------------------------------------
# Analysis 1: h² comparison — Bayes factor (best combo vs all features)
# ---------------------------------------------------------------------------

def paired_sign_flip_test(data_all, data_best, n_perm=10000, statistic='mean'):
    diff = np.array(data_all) - np.array(data_best)
    obs_stat = np.mean(diff) if statistic == 'mean' else np.median(diff)
    signs = np.random.choice([-1, 1], size=(n_perm, len(diff)))
    perm_stats = np.mean(signs * diff, axis=1) if statistic == 'mean' else np.median(signs * diff, axis=1)
    p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    return obs_stat, p_val


def bootstrap_ci(data, n_boot=10000, statistic='mean'):
    data = np.array(data)
    indices = np.random.randint(0, len(data), size=(n_boot, len(data)))
    samples = data[indices]
    boot_stats = np.mean(samples, axis=1) if statistic == 'mean' else np.median(samples, axis=1)
    return np.percentile(boot_stats, 2.5), np.percentile(boot_stats, 97.5)


def compute_bayes_factor(samples, prior_scale=0.1):
    """Savage-Dickey BF10: H0 mu=0 vs H1 mu~HalfNormal(0, prior_scale)."""
    prior_at_0 = stats.halfnorm.pdf(0, scale=prior_scale)
    reflected = np.concatenate([samples, -samples])
    try:
        kde = stats.gaussian_kde(reflected)
        posterior_at_0 = kde(0)[0] * 2
    except Exception:
        posterior_at_0 = np.inf if np.all(samples == 0) else 0
    if posterior_at_0 == 0:
        return np.inf
    if np.isinf(posterior_at_0):
        return 0.0
    return prior_at_0 / posterior_at_0


np.random.seed(42)
h2_results = []

_p('\n' + '=' * 72)
_p('ANALYSIS 1: h² COMPARISON (Best combo vs All features)')
_p('Bayes Factor (BF10): Savage-Dickey ratio testing H0: h²=0 vs H1: h²>0')
_p('  BF10 > 3   → substantial evidence h² > 0 (effect exists)')
_p('  BF10 < 1/3 → substantial evidence h² ≈ 0 (no effect)')
_p('  Prior: HalfNormal(0, scale=0.1) on h²')
_p('=' * 72)

for expo_name in exposomes_to_process:
    expo_dir = RESULTS_BASE / safe_name(expo_name)
    if not expo_dir.exists():
        print(f'Skipping {expo_name}: directory not found')
        continue

    best_safe = selected_combo_key_for_expo(expo_name)

    h2_best_df = pd.read_csv(expo_dir / f'{best_safe}_h2_per_repeat.csv')
    h2_all_df  = pd.read_csv(expo_dir / 'all_features_h2_per_repeat.csv')

    if len(h2_best_df) != len(h2_all_df):
        print(f'Warning: length mismatch for {expo_name}')
        continue

    x = h2_best_df['h2_with'].values
    y = h2_all_df['h2_with'].values

    obs_stat, pval = paired_sign_flip_test(y, x, n_perm=10000, statistic='mean')
    ci_lower, ci_upper = bootstrap_ci(y - x, n_boot=10000, statistic='mean')
    bf10_best = compute_bayes_factor(x, prior_scale=0.1)
    bf10_all  = compute_bayes_factor(y, prior_scale=0.1)

    h2_results.append({
        'Exposome':          expo_name,
        'Best_Combo':        best_safe,
        'N_Repeats':         len(x),
        'Best_Mean':         np.mean(x),
        'Best_SD':           np.std(x, ddof=1),
        'BF10_Best':         bf10_best,
        'All_Mean':          np.mean(y),
        'All_SD':            np.std(y, ddof=1),
        'BF10_All':          bf10_all,
        'Test_Paired_Type':  'Sign-flip Permutation',
        'Delta_Obs':         obs_stat,
        'Delta_CI_Low':      ci_lower,
        'Delta_CI_High':     ci_upper,
        'P_Value_Paired':    pval,
    })

h2_table = pd.DataFrame(h2_results)

_p('\nh² Comparison (Best vs All Features):')
_p(h2_table[['Exposome', 'Best_Mean', 'BF10_Best', 'All_Mean', 'BF10_All',
             'Delta_Obs', 'Delta_CI_Low', 'Delta_CI_High', 'P_Value_Paired']].to_string(index=False))

h2_csv = TABLE_DIR / 'expotype_h2_comparison_bayes.csv'
h2_table.to_csv(h2_csv, index=False)
print(f'Saved h² comparison to {h2_csv}')

# ---------------------------------------------------------------------------
# Analysis 2: LME + LRT
# ---------------------------------------------------------------------------

def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def fit_mixedlm_with_retry(formula, data, groups, vc_formula):
    model = sm.MixedLM.from_formula(
        formula, data=data, groups=groups,
        re_formula='1', vc_formula=vc_formula,
    )
    for method, maxiter in [('lbfgs', 500), ('powell', 1000)]:
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
    if hasattr(fit, 'vcomp') and hasattr(fit.model, 'exog_vc'):
        for name, val in zip(fit.model.exog_vc.names, fit.vcomp):
            if 'scanner' in name:
                var_scanner = val
                break
    resid = fit.scale
    total = var_country + var_scanner + resid
    icc_country = var_country / total if total > 0 else 0.0
    icc_scanner = var_scanner / total if total > 0 else 0.0
    return icc_country, icc_scanner


_p('\n' + '=' * 72)
_p('ANALYSIS 2: LME + LRT per exposome variable')
_p('=' * 72)

results_list = []

for expo_name in exposomes_to_process:
    print(f'Processing: {expo_name}')
    expo_dir = RESULTS_BASE / safe_name(expo_name)
    if not expo_dir.exists():
        continue

    try:
        winning_safe  = selected_combo_key_for_expo(expo_name)
        model_type    = 'Best Combo'

        scores = pd.read_csv(expo_dir / f'{winning_safe}_subject_scores.csv')
        scores['subject_id'] = scores['subject_id'].astype(str).str.strip()
        scores_avg = scores.groupby('subject_id', as_index=False)['score_with'].mean()
        merged = main_df.merge(scores_avg, left_on=ID_COL, right_on='subject_id', how='inner')

    except Exception as e:
        print(f'  Skipping {expo_name}: {e}')
        continue

    for var_label, var_col in EXPOSOME_GROUPS[expo_name].items():
        cols = ['score_with', var_col, 'Age', 'Sex', 'Diagnosis', 'Country', 'ODQ_fMRI', 'resonador']
        data = merged[cols].dropna().copy()

        if data[var_col].std(ddof=0) == 0 or len(data) < 10:
            continue

        data['pcev_z']          = zscore(data['score_with'])
        data['exposure_z']      = zscore(data[var_col])
        data['Age_z']           = zscore(data['Age'])
        data['ODQ_z']           = zscore(data['ODQ_fMRI'])
        data['Sex_male']        = (data['Sex'].str.lower() == 'male').astype(float)
        data['scanner_country'] = data['Country'].astype(str) + '::' + data['resonador'].astype(str)

        dx_dummies = pd.get_dummies(data['Diagnosis'].astype(str), prefix='Dx')
        drop_col   = 'Dx_CN' if 'Dx_CN' in dx_dummies.columns else sorted(dx_dummies.columns)[0]
        dx_dummies = dx_dummies.drop(columns=[drop_col])
        data = pd.concat([data, dx_dummies], axis=1)

        covariates    = ['Age_z', 'Sex_male', 'ODQ_z'] + list(dx_dummies.columns)
        formula_base  = f"pcev_z ~ {' + '.join(covariates)}"
        formula_full  = f"pcev_z ~ exposure_z + {' + '.join(covariates)}"
        vc_formula    = {'scanner': '0 + C(scanner_country)'}

        try:
            fit_base = fit_mixedlm_with_retry(formula_base, data, data['Country'], vc_formula)
            fit_full = fit_mixedlm_with_retry(formula_full, data, data['Country'], vc_formula)
            if fit_base is None or fit_full is None:
                continue

            beta     = fit_full.fe_params['exposure_z']
            ci       = fit_full.conf_int().loc['exposure_z']
            ci_low, ci_high = ci[0], ci[1]

            delta_ll  = fit_full.llf - fit_base.llf
            lrt_stat  = max(0, 2 * delta_ll)
            p_lrt     = chi2.sf(lrt_stat, df=1)
            delta_aic = fit_full.aic - fit_base.aic

            icc_country_base, icc_scanner_base = get_iccs(fit_base)
            icc_country_full, icc_scanner_full = get_iccs(fit_full)

            results_list.append({
                'Exposome':          expo_name,
                'Variable':          var_label,
                'Model_Used':        model_type,
                'N':                 len(data),
                'Beta':              beta,
                'Beta_CI_Low':       ci_low,
                'Beta_CI_High':      ci_high,
                'LL_Base':           fit_base.llf,
                'LL_Full':           fit_full.llf,
                'Delta_LL':          delta_ll,
                'LRT_Stat':          lrt_stat,
                'P_LRT':             p_lrt,
                'AIC_Base':          fit_base.aic,
                'AIC_Full':          fit_full.aic,
                'Delta_AIC':         delta_aic,
                'ICC_country_base':  icc_country_base,
                'ICC_scanner_base':  icc_scanner_base,
                'ICC_country_full':  icc_country_full,
                'ICC_scanner_full':  icc_scanner_full,
                'Delta_ICC_country': icc_country_full - icc_country_base,
                'Delta_ICC_scanner': icc_scanner_full - icc_scanner_base,
            })

        except Exception as e:
            print(f'  Error fitting {var_label}: {e}')

lrt_df = pd.DataFrame(results_list)

if not lrt_df.empty:
    lrt_df['FDR_q_lrt'] = np.nan
    for expo in lrt_df['Exposome'].unique():
        idx   = lrt_df['Exposome'] == expo
        pvals = lrt_df.loc[idx, 'P_LRT'].dropna()
        if not pvals.empty:
            _, qvals, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.001)
            lrt_df.loc[pvals.index, 'FDR_q_lrt'] = qvals

lrt_csv = TABLE_DIR / 'expotype_lme_lrt.csv'
lrt_df.to_csv(lrt_csv, index=False)
print(f'Saved LRT results to {lrt_csv}')

_p('\nLRT Results (all rows):')
if not lrt_df.empty:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    _p(lrt_df[['Exposome', 'Variable', 'Beta', 'Beta_CI_Low', 'Beta_CI_High',
               'LRT_Stat', 'P_LRT', 'FDR_q_lrt', 'Delta_AIC']].to_string(index=False))
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_colwidth')

# ---------------------------------------------------------------------------
# Save numeric summary to txt
# ---------------------------------------------------------------------------

summary_txt = TABLE_DIR / 'expotype_lme_numeric_summary.txt'
summary_txt.write_text(_buf.getvalue())
print(f'Saved numeric summary to {summary_txt}')

print('\nScript completed successfully.')
print(f'Outputs in {TABLE_DIR}:')
for f in sorted(TABLE_DIR.glob('expotype_*.csv')):
    print(f'  {f.name}  ({pd.read_csv(f).shape[0]} rows)')
print(f'  expotype_lme_numeric_summary.txt')
