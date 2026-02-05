#!/usr/bin/env python
# coding: utf-8

import warnings
from collections import OrderedDict
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import Normalize
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'data' / 'derived' / 'model_output_plus_exposome_data_v2.csv'
RESULTS_BASE = Path(os.environ.get('PCEV_RESULTS_DIR', REPO_ROOT / 'analysis' / 'results' / 'pcev_results')) / 'exposome_odq_only_no_scanner'
FIG_DIR = REPO_ROOT / 'analysis' / 'figures' / 'analysis_05_lme_expotypes'
FIG_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = 'N_MEGA'

# We will validate required columns later and, if needed, replace DATA_PATH with ALT_PATH contents.

# Exposome definitions
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

vip_feature_groups = ['rate_E', 'rate_I', 'EI_rate', 'ent_E', 'ent_I', 'EI_ent']

# Plot aesthetics
VIOLIN_COLORS = ['#3498db', '#c7c7c7']
VIP_CMAP = 'RdBu_r'
VIP_NORM = Normalize(vmin=-0.35, vmax=0.35)


def safe_name(name: str) -> str:
    return name.replace(' ', '_')


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def feature_to_region(feature: str) -> str:
    parts = feature.split('_')
    try:
        idx = int(parts[-1])
    except ValueError:
        return 'ROI?'
    return roi_regions[idx] if idx < len(roi_regions) else f'ROI {idx}'


def compute_vip(score_df: pd.DataFrame, feature_cols_by_group: dict, main_df: pd.DataFrame, label: str) -> pd.DataFrame:
    score_df = score_df[['subject_id', 'score_with']].dropna().copy()
    merged = score_df.merge(
        main_df[[ID_COL] + all_feature_cols],
        left_on='subject_id', right_on=ID_COL, how='inner'
    )
    if merged.empty:
        raise ValueError(f'No overlap between {label} scores and main data')
    results = []
    for group, cols in feature_cols_by_group.items():
        corrs = merged[cols].apply(lambda col: col.corr(merged['score_with']))
        results.append(pd.DataFrame({
            'feature_group': group,
            'feature': cols,
            'vip': corrs.values,
            'exposome': label,
        }))
    return pd.concat(results, ignore_index=True)


def top_vip_ei(vip_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    vip = vip_df[vip_df['feature_group'] == 'EI_rate'].copy()
    vip['abs_vip'] = vip['vip'].abs()
    vip = vip.sort_values('abs_vip', ascending=False).head(top_n)
    vip['region'] = vip['feature'].apply(feature_to_region)
    vip['label_short'] = vip['region']
    return vip


# --- DATA LOADING (no checks, as requested) ---
main_df = pd.read_csv(DATA_PATH, low_memory=False)
main_df[ID_COL] = main_df[ID_COL].astype(str).str.strip()

# ROI labels for VIP bar plots
roi_df = pd.read_csv(REPO_ROOT / 'data' / 'derived' / 'ROI_MNI_V4.csv')
roi_regions = roi_df['label'].tolist()

feature_cols_by_group = {g: [c for c in main_df.columns if c.startswith(f"{g}_")] for g in vip_feature_groups}
all_feature_cols = sorted({c for cols in feature_cols_by_group.values() for c in cols})

# --- ANALYSIS 1: h² Comparison (Best vs All Features) ---

def paired_sign_flip_test(data_all, data_best, n_perm=10000, statistic="mean"):
    diff = np.array(data_all) - np.array(data_best)
    if statistic == "mean":
        obs_stat = np.mean(diff)
    elif statistic == "median":
        obs_stat = np.median(diff)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    signs = np.random.choice([-1, 1], size=(n_perm, len(diff)))
    if statistic == "mean":
        perm_stats = np.mean(signs * diff, axis=1)
    else:
        perm_stats = np.median(signs * diff, axis=1)
    p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    return obs_stat, p_val


def bootstrap_ci(data, n_boot=10000, statistic="mean"):
    data = np.array(data)
    indices = np.random.randint(0, len(data), size=(n_boot, len(data)))
    samples = data[indices]
    if statistic == "mean":
        boot_stats = np.mean(samples, axis=1)
    else:
        boot_stats = np.median(samples, axis=1)
    ci_lower = np.percentile(boot_stats, 2.5)
    ci_upper = np.percentile(boot_stats, 97.5)
    return ci_lower, ci_upper


def compute_bayes_factor(samples, prior_scale=0.1):
    prior_at_0 = stats.halfnorm.pdf(0, scale=prior_scale)
    reflected_data = np.concatenate([samples, -samples])
    try:
        kde = stats.gaussian_kde(reflected_data)
        posterior_at_0 = kde(0)[0] * 2
    except Exception:
        posterior_at_0 = np.inf if np.all(samples == 0) else 0
    if posterior_at_0 == 0:
        return np.inf
    if np.isinf(posterior_at_0):
        return 0.0
    return prior_at_0 / posterior_at_0


h2_results = []
exposomes_to_process = [e for e in EXPOSOME_GROUPS.keys() if e not in excluded_exposomes]

for expo_name in exposomes_to_process:
    expo_dir = RESULTS_BASE / safe_name(expo_name)
    if not expo_dir.exists():
        print(f"Skipping {expo_name}: Directory not found")
        continue

    summary_df = pd.read_csv(expo_dir / 'all_combos_summary.csv')
    non_all = summary_df[summary_df['combo_label'] != 'all_features'].copy()
    best_row = non_all.sort_values('h2_with_mean', ascending=False).iloc[0]
    best_combo = best_row['combo_label']
    best_safe = best_combo.replace('+', '_')

    h2_best_df = pd.read_csv(expo_dir / f'{best_safe}_h2_per_repeat.csv')
    h2_all_df = pd.read_csv(expo_dir / 'all_features_h2_per_repeat.csv')

    if len(h2_best_df) != len(h2_all_df):
        print(f"Warning: Length mismatch for {expo_name}")
        continue

    x = h2_best_df['h2_with'].values
    y = h2_all_df['h2_with'].values

    obs_stat, pval = paired_sign_flip_test(y, x, n_perm=10000, statistic="mean")
    diffs = y - x
    ci_lower, ci_upper = bootstrap_ci(diffs, n_boot=10000, statistic="mean")

    bf10_best = compute_bayes_factor(x, prior_scale=0.1)
    bf10_all = compute_bayes_factor(y, prior_scale=0.1)

    h2_results.append({
        'Exposome': expo_name,
        'Best Combo': best_combo,
        'N_Repeats': len(x),
        'Best_Mean': np.mean(x),
        'Best_SD': np.std(x, ddof=1),
        'BF10_Best': bf10_best,
        'All_Mean': np.mean(y),
        'All_SD': np.std(y, ddof=1),
        'BF10_All': bf10_all,
        'Test_Paired_Type': 'Sign-flip Permutation',
        'Delta_Obs': obs_stat,
        'Delta_CI_Low': ci_lower,
        'Delta_CI_High': ci_upper,
        'P_Value_Paired': pval,
    })

h2_table = pd.DataFrame(h2_results)

print("h² Comparison Results (Best vs All Features)")
print(h2_table[['Exposome', 'Best_Mean', 'All_Mean', 'Delta_Obs', 'Delta_CI_Low', 'Delta_CI_High', 'P_Value_Paired']].to_string(index=False))

h2_out = FIG_DIR / 'h2_comparison_stats.csv'
h2_table.to_csv(h2_out, index=False)
print(f"Saved h² comparison to {h2_out}")

# --- LOAD PCEV RESULTS FOR FIGURE 1 (VIP + h²) ---

bundles = {}

for expo_name in exposomes_to_process:
    expo_dir = RESULTS_BASE / safe_name(expo_name)
    if not expo_dir.exists():
        continue

    summary_df = pd.read_csv(expo_dir / 'all_combos_summary.csv')
    non_all = summary_df[summary_df['combo_label'] != 'all_features'].copy()
    best_row = non_all.sort_values('h2_with_mean', ascending=False).iloc[0]
    best_combo = best_row['combo_label']
    best_safe = best_combo.replace('+', '_')

    h2_best = pd.read_csv(expo_dir / f'{best_safe}_h2_per_repeat.csv')
    h2_all = pd.read_csv(expo_dir / 'all_features_h2_per_repeat.csv')

    scores_best = pd.read_csv(expo_dir / f'{best_safe}_subject_scores.csv')
    scores_best['subject_id'] = scores_best['subject_id'].astype(str).str.strip()
    scores_best_avg = scores_best.groupby('subject_id', as_index=False)['score_with'].mean()

    vip_df = compute_vip(scores_best_avg, feature_cols_by_group, main_df, label=expo_name)

    bundles[expo_name] = {
        'dir': expo_dir,
        'best_combo': best_combo,
        'h2_best': h2_best,
        'h2_all': h2_all,
        'scores_avg': scores_best_avg,
        'vip_df': vip_df,
        'n_subjects': scores_best_avg.shape[0],
        'h2_mean': h2_best['h2_with'].mean(),
    }

# Ranking and violin data
exposome_order = [e for e, _ in sorted([(e, b['h2_mean']) for e, b in bundles.items()], key=lambda t: t[1], reverse=True)]
top_exposomes = exposome_order[:5]

all_violin_df = []
for expo in exposome_order:
    bund = bundles[expo]
    df = pd.DataFrame({
        'h2': pd.concat([bund['h2_best']['h2_with'], bund['h2_all']['h2_with']]),
        'condition': ['best'] * len(bund['h2_best']) + ['all_features'] * len(bund['h2_all']),
    })
    df['exposome'] = expo
    all_violin_df.append(df)
all_violin_df = pd.concat(all_violin_df, ignore_index=True)

# Print top 10 VIP values for top exposomes
print("\nTop 10 VIP (EI_rate) per top exposome:")
top_10_vip_per_exposome = []
for expo in top_exposomes:
    vip_df_expo = bundles[expo]['vip_df']
    vip_ei = vip_df_expo[vip_df_expo['feature_group'] == 'EI_rate'].copy()
    vip_ei['abs_vip'] = vip_ei['vip'].abs()
    vip_top10 = vip_ei.sort_values('abs_vip', ascending=False).head(10)
    vip_top10['region'] = vip_top10['feature'].apply(feature_to_region)
    vip_top10 = vip_top10[['exposome', 'region', 'vip']]
    top_10_vip_per_exposome.append(vip_top10)

if top_10_vip_per_exposome:
    top_10_vip_df = pd.concat(top_10_vip_per_exposome, ignore_index=True)
    print(top_10_vip_df.to_string(index=False))

# Save all top 10 VIPs per exposome
all_top_10_vip = []
for expo in bundles.keys():
    vip_df_expo = bundles[expo]['vip_df']
    vip_ei = vip_df_expo[vip_df_expo['feature_group'] == 'EI_rate'].copy()
    vip_ei['abs_vip'] = vip_ei['vip'].abs()
    vip_top10 = vip_ei.sort_values('abs_vip', ascending=False).head(10)
    vip_top10['region'] = vip_top10['feature'].apply(feature_to_region)
    vip_top10 = vip_top10[['exposome', 'region', 'vip']]
    all_top_10_vip.append(vip_top10)

if all_top_10_vip:
    all_top_10_vip_df = pd.concat(all_top_10_vip, ignore_index=True)
    vip_out = FIG_DIR / 'top10_vip_per_exposome.csv'
    all_top_10_vip_df.to_csv(vip_out, index=False)
    print(f"Saved top 10 VIP per exposome to {vip_out}")

# --- FIGURE 1: Exposome Overview (h² violins + VIP bars) ---

fig = plt.figure(figsize=(24, 12))
gs = gridspec.GridSpec(2, 6, figure=fig, width_ratios=[2.2, 1, 1, 1, 1, 1], wspace=0.72, hspace=0.55)

ax_all = fig.add_subplot(gs[0, 0])
sns.violinplot(
    data=all_violin_df, y='exposome', x='h2', hue='condition',
    split=True, palette=VIOLIN_COLORS, inner=None,
    orient='h', density_norm='count'
)
for violin in ax_all.collections:
    violin.set_edgecolor(None)
    violin.set_linewidth(0)
ax_all.set_ylabel('')
ax_all.set_xlabel('h²', fontsize=12)
ax_all.tick_params(axis='y', labelsize=18)
ax_all.tick_params(axis='x', labelsize=18)
ax_all.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
max_h2 = all_violin_df['h2'].max()
ax_all.set_xlim(0, max_h2 + 0.02)
ax_all.xaxis.grid(True, alpha=0.3)
ax_all.yaxis.grid(False)

handles = [
    plt.Line2D([0], [0], color=VIOLIN_COLORS[0], lw=4, label='best combo'),
    plt.Line2D([0], [0], color=VIOLIN_COLORS[1], lw=4, label='all_features')
]
ax_all.legend(handles=handles, loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.15))

fig.add_subplot(gs[1, 0]).axis('off')

for idx, expo in enumerate(top_exposomes):
    col = idx + 1
    bundle = bundles[expo]
    ax_vip = fig.add_subplot(gs[0, col])
    vip_ei = top_vip_ei(bundle['vip_df'], top_n=10)
    colors = [cm.get_cmap(VIP_CMAP)(VIP_NORM(v)) for v in vip_ei['vip']]
    ax_vip.barh(vip_ei['label_short'], vip_ei['vip'].abs(), color=colors, alpha=0.9)
    ax_vip.invert_yaxis()
    ax_vip.set_xlabel('|VIP|', fontsize=11)
    ax_vip.set_ylabel('Region' if col == 1 else '')
    ax_vip.tick_params(axis='y', labelsize=8)
    ax_vip.tick_params(axis='x', labelsize=9)
    ax_vip.set_title(expo, fontsize=12)
    ax_vip.xaxis.grid(True, alpha=0.3)
    ax_vip.set_xlim(0, 0.4)

cbar_ax = fig.add_axes([0.35, 0.04, 0.3, 0.025])
sm_vip = cm.ScalarMappable(cmap=VIP_CMAP, norm=VIP_NORM)
sm_vip.set_array([])
cbar = fig.colorbar(sm_vip, cax=cbar_ax, orientation='horizontal')
cbar.set_label('VIP (EI_rate, signed)', fontsize=10)
cbar.ax.tick_params(labelsize=9)
cbar.set_ticks([-0.3, -0.15, 0.0, 0.15, 0.3])

plt.tight_layout()

out_png = FIG_DIR / 'pcev_exposome_odq_overview.png'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'pcev_exposome_odq_overview.pdf', dpi=300, bbox_inches='tight')
print(f'Single overview figure saved to {out_png}')
plt.close(fig)

# --- ANALYSIS 2: LME + LRT (Wald CI + LRT/AIC) ---

def fit_mixedlm_with_retry(formula, data, groups, vc_formula, start_params=None):
    model = sm.MixedLM.from_formula(
        formula,
        data=data,
        groups=groups,
        re_formula='1',
        vc_formula=vc_formula
    )
    try:
        fit = model.fit(reml=False, method='lbfgs', maxiter=500, start_params=start_params)
        if fit.converged:
            return fit
    except Exception:
        pass
    try:
        fit = model.fit(reml=False, method='powell', maxiter=1000, start_params=start_params)
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
    if total > 0:
        icc_country = var_country / total
        icc_scanner = var_scanner / total
    else:
        icc_country = 0.0
        icc_scanner = 0.0
    return icc_country, icc_scanner


results_list = []
print("Running LME Analysis: Wald CI + LRT/AIC Model Comparison")

for expo_name in exposomes_to_process:
    print(f"\nProcessing Domain: {expo_name}")
    expo_dir = RESULTS_BASE / safe_name(expo_name)

    try:
        summary_df = pd.read_csv(expo_dir / 'all_combos_summary.csv')
        non_all = summary_df[summary_df['combo_label'] != 'all_features'].copy()
        best_combo_row = non_all.sort_values('h2_with_mean', ascending=False).iloc[0]
        h2_best_combo = best_combo_row['h2_with_mean']

        all_feat_row = summary_df[summary_df['combo_label'] == 'all_features'].iloc[0]
        h2_all_feat = all_feat_row['h2_with_mean']

        if h2_best_combo >= h2_all_feat:
            winning_model = best_combo_row['combo_label']
            winning_safe = winning_model.replace('+', '_')
            model_type = 'Best Combo'
        else:
            winning_model = 'all_features'
            winning_safe = 'all_features'
            model_type = 'All Features'

        scores = pd.read_csv(expo_dir / f'{winning_safe}_subject_scores.csv')
        scores['subject_id'] = scores['subject_id'].astype(str).str.strip()
        scores_avg = scores.groupby('subject_id', as_index=False)['score_with'].mean()

        merged = main_df.merge(scores_avg, left_on=ID_COL, right_on='subject_id', how='inner')

    except Exception as e:
        print(f"  Skipping {expo_name} due to data loading error: {e}")
        continue

    for var_label, var_col in EXPOSOME_GROUPS[expo_name].items():
        cols = ['score_with', var_col, 'Age', 'Sex', 'Diagnosis', 'Country', 'ODQ_fMRI', 'resonador']
        data = merged[cols].dropna().copy()

        if data[var_col].std(ddof=0) == 0 or len(data) < 10:
            continue

        data['pcev_z'] = zscore(data['score_with'])
        data['exposure_z'] = zscore(data[var_col])
        data['Age_z'] = zscore(data['Age'])
        data['ODQ_z'] = zscore(data['ODQ_fMRI'])
        data['Sex_male'] = (data['Sex'].str.lower() == 'male').astype(float)
        data['scanner_country'] = data['Country'].astype(str) + '::' + data['resonador'].astype(str)

        dx_dummies = pd.get_dummies(data['Diagnosis'].astype(str), prefix='Dx')
        if 'Dx_CN' in dx_dummies.columns:
            dx_dummies = dx_dummies.drop(columns=['Dx_CN'])
        else:
            drop_col = sorted(dx_dummies.columns)[0]
            dx_dummies = dx_dummies.drop(columns=[drop_col])
        data = pd.concat([data, dx_dummies], axis=1)

        covariates = ['Age_z', 'Sex_male', 'ODQ_z'] + list(dx_dummies.columns)
        formula_base = f"pcev_z ~ {' + '.join(covariates)}"
        formula_full = f"pcev_z ~ exposure_z + {' + '.join(covariates)}"

        groups_col = 'Country'
        vc_formula = {'scanner': '0 + C(scanner_country)'}

        try:
            fit_base = fit_mixedlm_with_retry(formula_base, data, data[groups_col], vc_formula)
            fit_full = fit_mixedlm_with_retry(formula_full, data, data[groups_col], vc_formula)

            if fit_base is None or fit_full is None:
                continue

            beta = fit_full.fe_params['exposure_z']
            ci = fit_full.conf_int().loc['exposure_z']
            ci_low, ci_high = ci[0], ci[1]

            ll_base = fit_base.llf
            ll_full = fit_full.llf
            aic_base = fit_base.aic
            aic_full = fit_full.aic

            delta_ll = ll_full - ll_base
            delta_aic = aic_full - aic_base

            lrt_stat = max(0, 2 * delta_ll)
            p_lrt = chi2.sf(lrt_stat, df=1)

            icc_country_base, icc_scanner_base = get_iccs(fit_base)
            icc_country_full, icc_scanner_full = get_iccs(fit_full)

            results_list.append({
                'Exposome': expo_name,
                'Variable': var_label,
                'Model_Used': model_type,
                'N': len(data),
                'Beta': beta,
                'Beta_CI_Low': ci_low,
                'Beta_CI_High': ci_high,
                'LL_Base': ll_base,
                'LL_Full': ll_full,
                'Delta_LL': delta_ll,
                'LRT_Stat': lrt_stat,
                'P_LRT': p_lrt,
                'AIC_Base': aic_base,
                'AIC_Full': aic_full,
                'Delta_AIC': delta_aic,
                'ICC_country_base': icc_country_base,
                'ICC_scanner_base': icc_scanner_base,
                'ICC_country_full': icc_country_full,
                'ICC_scanner_full': icc_scanner_full,
                'Delta_ICC_country': icc_country_full - icc_country_base,
                'Delta_ICC_scanner': icc_scanner_full - icc_scanner_base,
            })

        except Exception as e:
            print(f"  Error fitting {var_label}: {e}")

final_df = pd.DataFrame(results_list)

if not final_df.empty:
    final_df['FDR_q_lrt'] = np.nan
    for expo in final_df['Exposome'].unique():
        idx = final_df['Exposome'] == expo
        pvals = final_df.loc[idx, 'P_LRT'].dropna()
        if not pvals.empty:
            _, qvals, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.001)
            final_df.loc[pvals.index, 'FDR_q_lrt'] = qvals

out_path = FIG_DIR / 'exposome_lme_associations_wald_lrt_aic.csv'
final_df.to_csv(out_path, index=False)
print(f"Saved results to {out_path}")

if final_df.empty:
    print("\nNo LME results to display (final_df is empty).")
else:
    print("\nLME results (head):")
    print(final_df[['Exposome', 'Variable', 'Beta', 'Beta_CI_Low', 'Beta_CI_High', 'P_LRT', 'FDR_q_lrt', 'Delta_AIC']].head(10).to_string(index=False))

# --- FIGURE 2: LME Beta Barplot ---

_tab20b = sns.color_palette('tab20b', 20)
_remove_idx = [0, 1, 12, 13, 14, 15]
_custom_palette = [c for i, c in enumerate(_tab20b) if i not in _remove_idx][:10]


def plot_lme_beta_barplot(lme_df: pd.DataFrame, fname: Path, title: str = 'LME Associations: Beta'):
    if lme_df.empty:
        print('No LME results to plot.')
        return None

    plot_df = lme_df.copy().dropna(subset=['Beta'])

    exposome_h2 = {row['Exposome']: row['All_Mean'] for _, row in h2_table.iterrows()} if 'h2_table' in globals() else {}
    for expo in plot_df['Exposome'].unique():
        exposome_h2.setdefault(expo, 0)

    exposome_domains = sorted(plot_df['Exposome'].unique(), key=lambda e: exposome_h2.get(e, 0), reverse=True)
    exposome2color = {e: _custom_palette[i % len(_custom_palette)] for i, e in enumerate(exposome_domains)}

    variable_order = []
    for expo in exposome_domains:
        sub = plot_df[plot_df['Exposome'] == expo]
        sorted_vars = sub.sort_values('Beta', ascending=False)['Variable'].tolist()
        variable_order.extend([(expo, v) for v in sorted_vars])

    plot_df['var_label'] = plot_df['Exposome'] + ': ' + plot_df['Variable']
    order_labels = [f'{expo}: {var}' for expo, var in variable_order]
    plot_df['var_label'] = pd.Categorical(plot_df['var_label'], categories=order_labels, ordered=True)
    plot_df = plot_df.sort_values('var_label')

    lower_err = plot_df['Beta'] - plot_df['Beta_CI_Low']
    upper_err = plot_df['Beta_CI_High'] - plot_df['Beta']

    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(plot_df)), 6))
    bar_colors = [exposome2color[expo] for expo, _ in variable_order]

    ax.bar(
        x=np.arange(len(plot_df)),
        height=plot_df['Beta'],
        yerr=[lower_err, upper_err],
        color=bar_colors,
        edgecolor=None,
        capsize=4,
        alpha=1,
        linewidth=0
    )

    sig_mask = plot_df['FDR_q_lrt'] < 0.001
    for i, (is_sig, y, up_err) in enumerate(zip(sig_mask, plot_df['Beta'], upper_err)):
        if is_sig:
            y_star = y + (up_err if up_err > 0 else 0)
            ax.text(i, y_star, '*', color='black', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df['Variable'], rotation=90, ha='left', fontsize=14)
    ax.set_xlim(-0.5, len(plot_df) - 0.5)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    handles = [
        plt.Line2D([0], [0], color=exposome2color[expo], lw=6, label=expo)
        for expo in exposome_domains
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=16)

    ax.axhline(0, color='k', lw=0.8, alpha=0.7)
    ax.set_ylabel('LME Beta', fontsize=18)
    ax.set_title(title)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(fname.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print('Saved LME beta barplot to', fname)
    plt.close(fig)


fname = FIG_DIR / 'exposome_lme_beta_barplot.png'
plot_lme_beta_barplot(final_df, fname, title='LME Associations (Beta)')

# --- FIGURE 3: LRT Barplot ---

_tab20b = sns.color_palette('tab20b', 20)
_remove_idx = [0, 1, 12, 13, 14, 15]
_custom_palette = [c for i, c in enumerate(_tab20b) if i not in _remove_idx][:10]


def plot_lrt_barplot(lme_df: pd.DataFrame, fname: Path, title: str = 'LME Associations: LRT Statistics'):
    if lme_df.empty:
        print('No LME results to plot.')
        return None

    plot_df = lme_df.copy().dropna(subset=['LRT_Stat'])

    exposome_h2_max = {}
    if 'h2_table' in globals():
        for _, row in h2_table.iterrows():
            exposome_h2_max[row['Exposome']] = max(row.get('Best_Mean', 0), row.get('All_Mean', 0))

    if not exposome_h2_max:
        exposome_domains = plot_df.groupby('Exposome')['LRT_Stat'].mean().sort_values(ascending=False).index.tolist()
    else:
        unique_domains = plot_df['Exposome'].unique()
        exposome_domains = sorted(unique_domains, key=lambda d: exposome_h2_max.get(d, -1), reverse=True)

    exposome2color = {e: _custom_palette[i % len(_custom_palette)] for i, e in enumerate(exposome_domains)}

    variable_order = []
    for expo in exposome_domains:
        sub = plot_df[plot_df['Exposome'] == expo]
        sorted_vars = sub.sort_values('LRT_Stat', ascending=False)['Variable'].tolist()
        variable_order.extend([(expo, v) for v in sorted_vars])

    plot_df['var_label'] = plot_df['Exposome'] + ': ' + plot_df['Variable']
    order_labels = [f'{expo}: {var}' for expo, var in variable_order]
    plot_df['var_label'] = pd.Categorical(plot_df['var_label'], categories=order_labels, ordered=True)
    plot_df = plot_df.sort_values('var_label')

    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(plot_df)), 6))
    bar_colors = [exposome2color[expo] for expo, _ in variable_order]

    ax.bar(
        x=np.arange(len(plot_df)),
        height=plot_df['LRT_Stat'],
        color=bar_colors,
        edgecolor=None,
        alpha=1,
        linewidth=0
    )

    l1 = ax.axhline(6.6, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='p < 0.01')
    l2 = ax.axhline(10.8, color='gray', linestyle='-.', linewidth=1.5, alpha=0.8, label='p < 0.001')
    l3 = ax.axhline(15.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, label='p < 0.0001')

    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df['Variable'], rotation=90, ha='left', fontsize=14)
    ax.set_xlim(-0.5, len(plot_df) - 0.5)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    domain_handles = [
        plt.Line2D([0], [0], color=exposome2color[expo], lw=6, label=expo)
        for expo in exposome_domains
    ]
    threshold_handles = [l1, l2, l3]
    ax.legend(handles=domain_handles + threshold_handles, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)

    ax.set_ylabel('LRT Statistic', fontsize=18)
    ax.set_title(title)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(fname.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print('Saved LRT barplot to', fname)
    plt.close(fig)


fname = FIG_DIR / 'exposome_lrt_barplot.png'
plot_lrt_barplot(final_df, fname, title='LME Associations: LRT Statistics')

print("\nScript completed successfully.")
