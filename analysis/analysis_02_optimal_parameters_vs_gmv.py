#!/usr/bin/env python
# coding: utf-8

# Auto-converted from Analysis_02_optimal_parameters_vs_GMV.ipynb

from IPython.display import display
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'data' / 'derived' / 'model_output_plus_exposome_data_v2.csv'
ROI_PATH = REPO_ROOT / 'data' / 'derived' / 'ROI_MNI_V4.csv'


# # Integrated Structural-Functional Analysis
#
# This notebook integrates regional structural data (Gray Matter Volume, GMV) with optimal brain stimulation parameters to quantify their relationship.
#
# **Objectives:**
# 1. **Structural-Functional Correlation**: Calculate per-subject Spearman correlations between regional optimal parameters and regional GMV.
# 2. **Visualization**: Replicate the structural-functional relationship figures (CDF, group-wise violins, and scatter plots vs. GOF).
# 3. **Hierarchical LME Modeling**: Perform model selection for random effects and fixed-effect interactions to explain Model Fit (GOF).
#
# **Note:** Variance partitioning analysis has been excluded from this pipeline.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, bootstrap, spearmanr, pearsonr, gaussian_kde
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings

# Configure display settings
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Output directories
OUTPUT_DIR = REPO_ROOT / "analysis" / "tables"
FIGURES_DIR = REPO_ROOT / "analysis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting parameters
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ## 1. Load Data
#
# We use a consolidated dataset (`model_output_plus_exposome_data_v2.csv`) containing optimal parameters, exposome variables, and scanner metadata.

print("Loading data files...")

# 1. Load the single main dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

# 2. Load ROI labels (needed for ordering correlations)
roi_df = pd.read_csv(ROI_PATH)
roi_labels = roi_df['label'].tolist()

# 3. Filter out New Zealand (N=1)
print(f"Initial shape: {df.shape}")
df = df[df["Country"] != "New Zealand"].copy()
print(f"Final shape: {df.shape}")

# ## 2. Structural-Functional Correlation
#
# We calculate the Spearman correlation between Regional GMV (Gray Matter Volume) and the Optimal Parameter value for each subject across all regions.

print("Computing structural-functional correlations...")

# Identify GMV and Target columns
gmv_cols = [c for c in df.columns if c.startswith('G_')]
target_cols = [c for c in df.columns if c.startswith('target_')]

# Map ROI indices (0-89) to GMV columns
roi_to_gmv = {}
for i in range(min(len(roi_labels), 90)):
    roi_label = roi_labels[i]
    # Default naming: G_<Label>
    gmv_col = f"G_{roi_label}"
    if gmv_col in gmv_cols:
        roi_to_gmv[i] = gmv_col
    else:
        # Check alternative naming patterns if direct match fails
        alt_names = [
            roi_label.replace('Frontal_Sup_Medial', 'Frontal_Sup_Med'),
            roi_label.replace('Paracentral_Lobule', 'Paracentral_Lob'),
            roi_label.replace('Temporal_Pole_Mid', 'Temporal_Pol_Mid')
        ]
        for alt in alt_names:
            if f"G_{alt}" in gmv_cols:
                roi_to_gmv[i] = f"G_{alt}"
                break

# Compute correlations
correlation_results = []
for idx, row in df.iterrows():
    params = []
    gmv_vals = []
    
    for i in range(90):
        t_col = f"target_{i}"
        if i in roi_to_gmv:
            g_col = roi_to_gmv[i]
            
            if t_col in row and g_col in row:
                p = row[t_col]
                g = row[g_col]
                if not (pd.isna(p) or pd.isna(g)):
                    params.append(p)
                    gmv_vals.append(g)
    
    if len(params) >= 3: # Minimum regions for a stable correlation
        corr, p_val = spearmanr(params, gmv_vals)
        correlation_results.append({
            'N_MEGA': row['N_MEGA'],
            'gmv_param_corr': corr,
            'gmv_param_p': p_val
        })
    else:
        correlation_results.append({
            'N_MEGA': row['N_MEGA'],
            'gmv_param_corr': np.nan,
            'gmv_param_p': np.nan
        })

# Merge results back
corr_df = pd.DataFrame(correlation_results)
merged_df = df.merge(corr_df, on='N_MEGA', how='outer')

# Filter for analysis
merged_df = merged_df.dropna(subset=['gmv_param_corr', 'gof_corr']).copy()

print(f"Calculated correlations for {len(merged_df)} subjects.")
display(merged_df[['N_MEGA', 'Diagnosis', 'gmv_param_corr', 'gof_corr']].head())

# ## 3. Visualization
#
# Replication of the key structural-functional relationship figure.

print("Generating structural-functional relationship visualization...")

def calculate_f2(r_squared):
    """Calculate Cohen's f2 from R-squared."""
    return r_squared / (1 - r_squared) if r_squared < 1 else np.inf

fig = plt.figure(figsize=(25, 10))
palette = sns.color_palette("Set2", 5)
diagnosis_order = ['CN', 'MCI', 'AD', 'FTD']

# Grid layout
outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0], wspace=0.15, width_ratios=[1, 4])

# Panel A: CDF of correlations
ax1 = fig.add_subplot(top_gs[0, 0])
sorted_corr = np.sort(merged_df['gmv_param_corr'])
cdf = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
ax1.plot(sorted_corr, cdf, color='gray', linewidth=3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y * 100)))
ax1.set_ylabel('Percentage of Sample', fontsize=20)
ax1.axvline(merged_df['gmv_param_corr'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {merged_df["gmv_param_corr"].mean():.3f}')
ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
ax1.set_xlabel('Spearman Correlation (GMV vs. Params)', fontsize=20)
ax1.set_title('Full Sample CDF', fontsize=20, fontweight='bold')
ax1.legend(fontsize=18, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=18)

# Panel B: Violin plot by diagnosis
ax2 = fig.add_subplot(top_gs[0, 1])
sns.violinplot(data=merged_df, x='Diagnosis', y='gmv_param_corr', order=diagnosis_order, 
               palette=palette[1:], inner=None, ax=ax2, cut=0, linewidth=0, alpha=0.3)

for i, diag in enumerate(diagnosis_order):
    y_points = merged_df[merged_df['Diagnosis'] == diag]['gmv_param_corr']
    x_jitter = np.random.normal(i + 0.18, 0.04, size=len(y_points))
    ax2.scatter(x_jitter, y_points, color=palette[i+1], alpha=0.4, s=20, edgecolor='none')

ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
ax2.set_xlabel('Diagnostic Group', fontsize=20)
ax2.set_ylabel('Spearman Correlation', fontsize=20)
ax2.set_title('Correlation by Diagnosis', fontsize=20, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', labelsize=18)

# Panel C: Correlation vs GOF Scatter Plots
bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[1], wspace=0.2)
groups = [('All', 'All Subjects')] + [(d, d) for d in diagnosis_order]

for i, (group, title) in enumerate(groups):
    ax = fig.add_subplot(bottom_gs[0, i])
    data_sub = merged_df if group == 'All' else merged_df[merged_df['Diagnosis'] == group]
    
    if len(data_sub) > 1:
        x, y = data_sub['gmv_param_corr'], data_sub['gof_corr']
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)(xy)
        ax.scatter(x, y, c=kde, cmap='gray_r', s=50, alpha=0.8, edgecolors='none')
        
        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(x), p(np.sort(x)), color='black', linestyle='--', linewidth=2)
        
        # Stats
        rho, p_val = spearmanr(x, y)
        f2 = calculate_f2(rho**2)
        ax.legend([f'ρ = {rho:.2f}\n$f^2$ = {f2:.2f}'], loc='lower right', fontsize=16, frameon=True)
        
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlim(-0.4, 0.8)
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=18)
    if i == 0: ax.set_ylabel('Goodness of Fit (GOF)', fontsize=20)
    else: ax.tick_params(labelleft=False)

fig.text(0.5, 0.02, 'Gray Matter vs Optimal Parameters Correlation', ha='center', fontsize=20, fontweight='bold')
plt.savefig(FIGURES_DIR / "publication_structural_functional_relationship.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "publication_structural_functional_relationship.pdf", bbox_inches='tight')
plt.show()

# ## 4. Linear Mixed-Effects Model Selection
#
# We use LME to determine the best model explaining Goodness of Fit (GOF).
#
# 1. **Random Effects Selection**: Finding the optimal grouping structure (Country vs. Scanner nested in Country).
# 2. **Fixed Effects Selection**: Identifying significant interactions.

print("                  EXHAUSTIVE BACKWARD MODEL SELECTION (AIC)                     ")

# 1. Helper: R2 Metrics
def get_lme_metrics(model):
    """Marginal/Conditional R2 (Nakagawa-style) for statsmodels MixedLM."""
    fe_params = model.fe_params
    fe_fitted = np.asarray(model.model.exog @ fe_params.values)
    var_f = float(np.var(fe_fitted, ddof=1))

    var_r = 0.0
    if hasattr(model, 'cov_re') and model.cov_re is not None:
        var_r += float(np.sum(np.asarray(model.cov_re)))
    if hasattr(model, 'vcomp') and model.vcomp is not None:
        var_r += float(np.sum(np.asarray(model.vcomp)))

    var_e = float(model.scale)
    denom = var_f + var_r + var_e
    r2_m = var_f / denom if denom > 0 else np.nan
    r2_c = (var_f + var_r) / denom if denom > 0 else np.nan
    return r2_m, r2_c

# 2. Setup Data (LME DataFrame Preparation - duplicating logic to ensure self-containedness or reusing existing)
print("Starting LME model selection process...")

# Prepare data for LME
required_cols = ['gof_corr', 'Age', 'Sex', 'Diagnosis', 'gmv_param_corr', 'Country', 'resonador', 'ODQ_fMRI']
lme_df = merged_df.dropna(subset=required_cols).copy()

# Clean / standardize strings
lme_df['Country'] = lme_df['Country'].astype(str).str.strip()
lme_df['Diagnosis'] = lme_df['Diagnosis'].astype(str).str.strip()
lme_df['Sex'] = lme_df['Sex'].astype(str).str.strip()
lme_df['resonador'] = lme_df['resonador'].astype(str).str.strip()
lme_df['ODQ_fMRI'] = lme_df['ODQ_fMRI'].astype(str).str.strip()

# Z-score target and continuous predictors for stability
lme_df['gof_z'] = (lme_df['gof_corr'] - lme_df['gof_corr'].mean()) / lme_df['gof_corr'].std(ddof=1)
lme_df['Age_z'] = (lme_df['Age'] - lme_df['Age'].mean()) / lme_df['Age'].std(ddof=1)
lme_df['gmv_corr_z'] = (lme_df['gmv_param_corr'] - lme_df['gmv_param_corr'].mean()) / lme_df['gmv_param_corr'].std(ddof=1)
lme_df['ODQ_fMRI_z'] = (lme_df['ODQ_fMRI'].astype(float) - lme_df['ODQ_fMRI'].astype(float).mean()) / lme_df['ODQ_fMRI'].astype(float).std(ddof=1)

# Encode categorical
lme_df['Sex_bin'] = lme_df['Sex'].eq('Male').astype(int)
lme_df['Country_Scanner'] = lme_df['Country'] + "__" + lme_df['resonador']

def fit_robust_lme(formula, *, reml: bool, vc_formula=None):
    """Fit MixedLM with multiple fallbacks for stability."""
    model = smf.mixedlm(
        formula,
        lme_df,
        groups=lme_df['Country'],
        vc_formula=vc_formula,
        re_formula="1",
    )
    for method in (('powell',), ('lbfgs',), ('nm',), ('bfgs',)):
        try:
            return model.fit(method=list(method), reml=reml, disp=False)
        except Exception:
            continue
    return model.fit(reml=reml, disp=False)

# 3. Exhaustive Model Selection Logic

print("Comparing two random-effects branches: (1|Country) vs (1|Country) + (1|Country/Scanner)")

# Variables allowed in interactions
interaction_vars = ['Age_z', 'Sex_bin', 'gmv_corr_z', 'Diagnosis']
# Variables NOT allowed in interactions (covariates only)
covariate_vars = ['ODQ_fMRI_z']

# Map variable names to their formula string representations
term_map = {
    'Age_z': 'Age_z',
    'Sex_bin': 'Sex_bin',
    'gmv_corr_z': 'gmv_corr_z',
    'ODQ_fMRI_z': 'ODQ_fMRI_z',
    'Diagnosis': "C(Diagnosis, Treatment(reference='CN'))"
}

def build_formula(main_keys, interaction_keys):
    rhs_parts = []
    # Add main effects
    for k in main_keys:
        rhs_parts.append(term_map[k])
    # Add interactions
    for k1, k2 in interaction_keys:
        rhs_parts.append(f"{term_map[k1]}:{term_map[k2]}")
    
    if not rhs_parts:
        return "gof_z ~ 1"
    return "gof_z ~ " + " + ".join(rhs_parts)

def get_iccs(model, branch_name):
    """Calculate ICCs for Country and Scanner (if applicable)."""
    var_resid = model.scale
    # Random intercept for Country (grouping variable)
    var_country = model.cov_re.iloc[0,0]
    
    if branch_name == "Country Only":
        total_var = var_country + var_resid
        icc_country = var_country / total_var
        icc_scanner = np.nan
    else:
        # For Country + Scanner, extract scanner variance from vcomp
        # We assume the first component in vcomp corresponds to Scanner
        if hasattr(model, 'vcomp') and len(model.vcomp) > 0:
            var_scanner = model.vcomp[0]
        else:
            var_scanner = 0.0
            
        total_var = var_country + var_scanner + var_resid
        icc_country = var_country / total_var
        icc_scanner = var_scanner / total_var
        
    return icc_country, icc_scanner

def run_backward_selection(vc_structure, branch_name):
    print(f"\n--- Starting Branch: {branch_name} ---")
    
    # Initialize current model state
    current_main_effects = set(interaction_vars + covariate_vars)
    current_interactions = set()
    
    # Generate all possible pairwise interactions for the "Full Model"
    for i in range(len(interaction_vars)):
        for j in range(i + 1, len(interaction_vars)):
            v1, v2 = interaction_vars[i], interaction_vars[j]
            term_key = tuple(sorted((v1, v2)))
            current_interactions.add(term_key)

    # Initial Fit
    initial_formula = build_formula(current_main_effects, current_interactions)
    try:
        current_model = fit_robust_lme(initial_formula, reml=False, vc_formula=vc_structure)
        current_aic = current_model.aic
        r2m, r2c = get_lme_metrics(current_model)
        icc_c, icc_s = get_iccs(current_model, branch_name)
    except Exception as e:
        print(f"Initial fit failed: {e}")
        return None, None, None

    step = 0
    history = [{
        'step': 0, 
        'action': 'Initial Full Model', 
        'aic': current_aic, 
        'R2_marginal': r2m,
        'R2_conditional': r2c,
        'ICC_Country': icc_c,
        'ICC_Scanner': icc_s,
        'removed': None
    }]

    while True:
        step += 1
        # Identify candidates for removal
        candidates = []
        for pair in current_interactions:
            candidates.append(('interaction', pair))
        
        for var in current_main_effects:
            is_involved = False
            for k1, k2 in current_interactions:
                if var == k1 or var == k2:
                    is_involved = True
                    break
            if not is_involved:
                candidates.append(('main', var))
                
        if not candidates:
            break
            
        best_candidate = None
        best_new_aic = current_aic
        best_metrics = (None, None)
        best_iccs = (None, None)
        
        for c_type, c_key in candidates:
            temp_main = current_main_effects.copy()
            temp_inter = current_interactions.copy()
            
            if c_type == 'interaction':
                temp_inter.remove(c_key)
            else:
                temp_main.remove(c_key)
                
            formula = build_formula(temp_main, temp_inter)
            try:
                m = fit_robust_lme(formula, reml=False, vc_formula=vc_structure)
                aic = m.aic
                if aic < best_new_aic:
                    best_new_aic = aic
                    best_candidate = (c_type, c_key)
                    best_metrics = get_lme_metrics(m)
                    best_iccs = get_iccs(m, branch_name)
            except Exception:
                continue

        if best_candidate:
            c_type, c_key = best_candidate
            
            if c_type == 'interaction':
                current_interactions.remove(c_key)
                removed_str = f"{c_key[0]}:{c_key[1]}"
            else:
                current_main_effects.remove(c_key)
                removed_str = c_key
                
            current_aic = best_new_aic
            history.append({
                'step': step, 
                'action': 'Removed Term', 
                'aic': current_aic, 
                'R2_marginal': best_metrics[0],
                'R2_conditional': best_metrics[1],
                'ICC_Country': best_iccs[0],
                'ICC_Scanner': best_iccs[1],
                'removed': removed_str
            })
            print(f"Step {step}: Removed {removed_str} (AIC: {current_aic:.2f})")
        else:
            break
            
    final_formula = build_formula(current_main_effects, current_interactions)
    return history, final_formula, vc_structure

# Run both branches
configs = [
    ("Country Only", None),
    ("Country + Scanner", {"Scanner": "0 + C(Country_Scanner)"})
]

results = {}
for label, vc in configs:
    hist, form, vc_used = run_backward_selection(vc, label)
    results[label] = {'history': hist, 'formula': form, 'vc': vc_used}

# Display Results
for label, res in results.items():
    print(f"\n=== Results for Branch: {label} ===")
    if res['history']:
        df_hist = pd.DataFrame(res['history'])
        display(df_hist)
        print(f"Final Formula: {res['formula']}")
    else:
        print("Selection failed.")

# Select the best overall model (comparing final AICs)
best_label = None
best_aic = np.inf

for label, res in results.items():
    if res['history']:
        final_aic = res['history'][-1]['aic']
        if final_aic < best_aic:
            best_aic = final_aic
            best_label = label

print(f"\nOverall Best Model comes from branch: {best_label} (AIC={best_aic:.2f})")
if best_label:
    final_formula_aic = results[best_label]['formula']
    vc_structure = results[best_label]['vc']

    # Fit final model with REML for interpretation
    print("\nFitting final optimal model with REML...")
    final_model_optimal = fit_robust_lme(final_formula_aic, reml=True, vc_formula=vc_structure)
    print(final_model_optimal.summary())

    # Report Final ICCs
    icc_c, icc_s = get_iccs(final_model_optimal, best_label)
    print(f"\nFinal Model ICC (Country): {icc_c:.4f}")
    if best_label == "Country + Scanner":
        print(f"Final Model ICC (Scanner within Country): {icc_s:.4f}")

    # Save results
    optimal_summary_path = OUTPUT_DIR / "lme_optimal_model_summary.txt"
    with open(optimal_summary_path, "w") as f:
        f.write(final_model_optimal.summary().as_text())
        f.write(f"\n\nICC (Country): {icc_c:.4f}")
        if best_label == "Country + Scanner":
            f.write(f"\nICC (Scanner within Country): {icc_s:.4f}")
    print(f"\nSaved summary to: {optimal_summary_path}")
