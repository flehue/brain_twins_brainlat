#!/usr/bin/env python
# coding: utf-8

# Final PCEV phenotype analysis (figures + summaries).
# Converted from Final_PCEV_Analysis_Phenotype.ipynb and adapted to repo paths.

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'data' / 'derived' / 'model_output_plus_exposome_data_v2.csv'
RESULTS_BASE = Path(os.environ.get('PCEV_RESULTS_DIR', REPO_ROOT / 'analysis' / 'results' / 'pcev_results'))
TABLES_DIR = REPO_ROOT / 'analysis' / 'tables_for_paper'
FIG_DIR = REPO_ROOT / 'analysis' / 'figures' / 'analysis_04_lme_phenotypes'
BRAINPLOT_DIR = Path(os.environ.get('BRAINPLOT_DIR', REPO_ROOT / 'analysis' / 'brainplot'))

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _combo_from_filename(path: Path, suffix: str) -> str:
    name = path.stem
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def build_consolidated_pcev_outputs() -> None:
    metrics_path = TABLES_DIR / 'pcev_consolidated_metrics_phenotype.csv'
    scores_path = TABLES_DIR / 'pcev_consolidated_scores_phenotype.csv'
    if metrics_path.exists() and scores_path.exists():
        return

    rows_metrics = []
    rows_scores = []

    age_dir = RESULTS_BASE / 'age_odq_only_no_scanner'
    for p in age_dir.glob('*_h2_per_repeat.csv'):
        combo = _combo_from_filename(p, '_h2_per_repeat')
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows_metrics.append({'phenotype': 'Age', 'combo': combo, 'repeat': r.get('repeat'), 'value': r.get('h2_with')})
    for p in age_dir.glob('*_subject_scores.csv'):
        combo = _combo_from_filename(p, '_subject_scores')
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows_scores.append({'phenotype': 'Age', 'combo': combo, 'subject_id': r.get('subject_id'), 'score': r.get('score_with')})

    sex_dir = RESULTS_BASE / 'sex_odq_only_no_scanner'
    for p in sex_dir.glob('*_metrics_per_repeat.csv'):
        combo = _combo_from_filename(p, '_metrics_per_repeat')
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows_metrics.append({'phenotype': 'Sex', 'combo': combo, 'repeat': r.get('repeat'), 'value': r.get('cohens_d_with')})
    for p in sex_dir.glob('*_subject_scores.csv'):
        combo = _combo_from_filename(p, '_subject_scores')
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows_scores.append({'phenotype': 'Sex', 'combo': combo, 'subject_id': r.get('subject_id'), 'score': r.get('score_with')})

    diag_dir = RESULTS_BASE / 'diagnosis_odq_only_no_scanner'
    for p in diag_dir.glob('*_metrics_per_repeat.csv'):
        combo = _combo_from_filename(p, '_metrics_per_repeat')
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows_metrics.append({'phenotype': 'Diagnosis', 'combo': combo, 'repeat': r.get('repeat'), 'value': r.get('epsilon_with')})
    for p in diag_dir.glob('*_subject_scores.csv'):
        combo = _combo_from_filename(p, '_subject_scores')
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows_scores.append({'phenotype': 'Diagnosis', 'combo': combo, 'subject_id': r.get('subject_id'), 'score': r.get('score_with')})

    pd.DataFrame(rows_metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(rows_scores).to_csv(scores_path, index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde
import sys
import warnings
import brainspace.plotting.base as bs_base
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
import nbformat
import nibabel as nb
import vtk

# --- CONFIGURATION ---
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

KDE_BANDWIDTH = None
TEXTBOX_X = 0.95
TEXTBOX_Y = 0.95
STAR_FONTSIZE = 30
STAR_COLOR = 'red'
VIOLIN_COLORS = ["#555757", '#555757']
FONT_TITLE = 18
FONT_LABEL = 18
FONT_TICK = 16
FIGSIZE = (24, 12)
WSPACE = 0.25
HSPACE = 0.4

FIG_DIR = REPO_ROOT / 'analysis' / 'figures' / 'analysis_04_lme_phenotypes'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- HELPER FUNCTIONS ---

def parse_combo_label(label):
    if label == 'all_features':
        return None
    single_features = ['EI_ent', 'EI_rate', 'rate_I', 'FC', 'FCD', 'metastability']
    if label in single_features:
        return [label]
    for feat1 in single_features:
        for feat2 in single_features:
            if label == f"{feat1}_{feat2}" or label == f"{feat2}_{feat1}":
                return sorted([feat1, feat2])
    parts = label.split('_')
    if len(parts) == 4:
        return sorted([f"{parts[0]}_{parts[1]}", f"{parts[2]}_{parts[3]}"])
    elif len(parts) == 3:
        if f"{parts[0]}_{parts[1]}" in single_features:
            return sorted([f"{parts[0]}_{parts[1]}", parts[2]])
        elif f"{parts[1]}_{parts[2]}" in single_features:
            return sorted([parts[0], f"{parts[1]}_{parts[2]}"])
    return [label]

def build_matrix(df, metric_col):
    combos = [c for c in df['combo'].unique() if c != 'all_features']
    combo_groups = {}
    for c in combos:
        groups = parse_combo_label(c)
        if groups:
            combo_groups[c] = groups
    all_groups = set()
    for groups in combo_groups.values():
        all_groups.update(groups)
    groups_sorted = sorted(all_groups)
    matrix = pd.DataFrame(np.nan, index=groups_sorted, columns=groups_sorted)
    for combo, groups in combo_groups.items():
        avg_val = df[df['combo'] == combo][metric_col].mean()
        if len(groups) == 1:
            g = groups[0]
            matrix.loc[g, g] = avg_val
        elif len(groups) == 2:
            g1, g2 = groups
            matrix.loc[g1, g2] = avg_val
            matrix.loc[g2, g1] = avg_val
    return matrix

def combo_to_filename(combo_label):
    return combo_label.replace('+', '_')

def create_heatmap(ax, matrix, title, max_idx=None, show_xticklabels=True):
    sns.heatmap(
        matrix,
        annot=False,
        cmap='gray_r',
        ax=ax,
        cbar=True,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_TICK+4)
    vmin, vmax = cbar.vmin, cbar.vmax
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    
    ax.set_title(title, fontweight='bold', fontsize=FONT_TITLE)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if show_xticklabels:
        xlabels = [label.get_text().replace('_', ' ') for label in ax.get_xticklabels()]
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=FONT_TICK+2)
    else:
        ax.set_xticklabels([])
    
    ylabels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=FONT_TICK+2)
    ax.tick_params(axis='both', which='both', length=0)
    
    if max_idx is not None:
        ax.text(
            max_idx[1] + 0.5,
            max_idx[0] + 0.5,
            '★',
            ha='center',
            va='center',
            fontsize=STAR_FONTSIZE,
            color=STAR_COLOR,
            fontweight='bold'
        )

# --- BRAIN PLOTTING HELPERS ---

# VTK Patch
def _patch_vtk_mro_for_vtkname():
    probe = vtk.vtkPolyData()
    for cls in probe.__class__.mro():
        if cls is object:
            continue
        if not hasattr(cls, "__vtkname__"):
            nm = cls.__name__
            if not nm.startswith("vtk"):
                nm = f"vtk{nm}"
            cls.__vtkname__ = nm
_patch_vtk_mro_for_vtkname()

def patched_to_numpy(self, transparent_bg=True, scale=None):
    wf = self._win2img(transparent_bg, scale)
    img = bs_base.get_output(wf)
    dims = None
    if hasattr(img, "GetDimensions"):
        try: dims = img.GetDimensions()
        except Exception: dims = None
    if dims is None and hasattr(img, "dimensions"):
        d = img.dimensions
        if isinstance(d, tuple): dims = d
        elif callable(d): dims = d()
        else: dims = d
    dims = tuple(dims)
    shape = dims[::-1][1:] + (-1,)
    scalars = img.PointData['ImageScalars']
    if hasattr(scalars, "to_array"): scalars = scalars.to_array()
    scalars = np.asarray(scalars)
    return scalars.reshape(shape)[::-1]
bs_base.Plotter.to_numpy = patched_to_numpy

def remove_excluded_regions(vec):
    mask = np.ones(90, dtype=bool)
    mask[70:78] = False
    return np.array(vec)[mask]

def plot_brain_map(mapp, name, vmin=-0.3, vmax=0.3, cmap='seismic', savefile=None):
    # Determine Labels Path relative to Notebook
    base_path = BRAINPLOT_DIR
    lh_labels_gii = nb.load(base_path / 'brainplot' / 'AAL.32k.L.label.gii')
    lh_labels = lh_labels_gii.darrays[0].data.astype(int)
    rh_labels_gii = nb.load(base_path / 'brainplot' / 'AAL.32k.R.label.gii')
    rh_labels = rh_labels_gii.darrays[0].data.astype(int)
    
    Ds_left = mapp[::2]
    Ds_right = mapp[1::2]
    
    lh_vertex_data = np.zeros_like(lh_labels, dtype=float)
    for i in range(41):
        lh_vertex_data[lh_labels == i+1] = Ds_left[i]
    rh_vertex_data = np.zeros_like(rh_labels, dtype=float)
    for i in range(41):
        rh_vertex_data[rh_labels == i+1] = Ds_right[i]
        
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, size=(1000, 800), layout='grid', zoom=1.5)
    p.add_layer({'left': lh_vertex_data, 'right': rh_vertex_data},
                cmap=cmap, cbar=False, color_range=(vmin, vmax))
    fig = p.build()
    fig.patch.set_alpha(0)
    if savefile:
        fig.savefig(savefile, dpi=200, transparent=True)
    return fig

def crop_image_auto(img, bg_threshold=10):
    mask = (img > bg_threshold).any(axis=2) & (img < 255-bg_threshold).any(axis=2)
    if not mask.any(): return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0: return img
    return img[rows[0]:rows[-1]+100, cols[0]:cols[-1]+500, :]

# --- DATA LOADING & PREPARATION ---

build_consolidated_pcev_outputs()

print("Loading data...")
main_df = pd.read_csv(DATA_PATH, low_memory=False)
metrics_df = pd.read_csv(TABLES_DIR / 'pcev_consolidated_metrics_phenotype.csv')
scores_df = pd.read_csv(TABLES_DIR / 'pcev_consolidated_scores_phenotype.csv')

# Prepare Phenotype Dataframes (recreate age_df, sex_df, diag_df)
age_df = metrics_df[metrics_df['phenotype'] == 'Age'].copy()
age_df['h2_with'] = age_df['value']  # Map 'value' back to 'h2_with'

sex_df = metrics_df[metrics_df['phenotype'] == 'Sex'].copy()
sex_df['cohens_d_with'] = sex_df['value'] # Map back

diag_df = metrics_df[metrics_df['phenotype'] == 'Diagnosis'].copy()
diag_df['epsilon_with'] = diag_df['value'] # Map back

# Build Matrices
print("Building matrices...")
age_matrix = build_matrix(age_df, 'h2_with')
sex_matrix = build_matrix(sex_df, 'cohens_d_with').abs()
diag_matrix = build_matrix(diag_df, 'epsilon_with')

# Find Best Combos
age_best_idx = np.unravel_index(np.nanargmax(age_matrix.values), age_matrix.shape)
sex_best_idx = np.unravel_index(np.nanargmax(sex_matrix.values), sex_matrix.shape)
diag_best_idx = np.unravel_index(np.nanargmax(diag_matrix.values), diag_matrix.shape)

age_best_combo = 'EI_ent+rate_I' # Verified match
sex_best_combo = 'rate_I'        # Verified match

# Logic from original for Diagnosis best combo
if diag_matrix.index[diag_best_idx[0]] == diag_matrix.columns[diag_best_idx[1]]:
    diag_best_combo = diag_matrix.index[diag_best_idx[0]]
else:
    diag_best_combo = f"{diag_matrix.index[diag_best_idx[0]]}+{diag_matrix.columns[diag_best_idx[1]]}"

print(f"Best Combos:\n Age={age_best_combo}\n Sex={sex_best_combo}\n Diag={diag_best_combo}")

# Prepare Specific Data for Plots (Best vs All)
# Age
age_best_h2 = age_df[age_df['combo'] == combo_to_filename(age_best_combo)].copy()
age_all_h2 = age_df[age_df['combo'] == 'all_features'].copy()

# Sex
sex_best_metrics = sex_df[sex_df['combo'] == combo_to_filename(sex_best_combo)].copy()
sex_all_metrics = sex_df[sex_df['combo'] == 'all_features'].copy()

# Diagnosis
diag_best_metrics = diag_df[diag_df['combo'] == combo_to_filename(diag_best_combo)].copy()
diag_all_metrics = diag_df[diag_df['combo'] == 'all_features'].copy()

# Prepare Scores (Subject Averages)
def prepare_scores_avg(pheno, combo, groupby_cols):
    subset = scores_df[(scores_df['phenotype'] == pheno) & (scores_df['combo'] == combo_to_filename(combo))].copy()
    subset['score_with'] = subset['score']
    # Merge with Main DF metadata
    # NOTE: Original used N_MEGA for main_df, subject_id for scores
    merged = subset.merge(main_df[['N_MEGA'] + groupby_cols], left_on='subject_id', right_on='N_MEGA', how='inner')
    merged = merged.drop(columns=['N_MEGA'])
    avg = merged.groupby(['subject_id'] + groupby_cols, as_index=False)['score_with'].mean()
    return avg

age_best_scores_avg = prepare_scores_avg('Age', age_best_combo, ['Age'])
sex_best_scores_avg = prepare_scores_avg('Sex', sex_best_combo, ['Sex'])
diag_all_scores_avg = prepare_scores_avg('Diagnosis', 'all_features', ['Diagnosis']) # Uses 'all_features'

# VIP Calculation Helpers
vip_feature_groups = ['rate_E', 'rate_I', 'EI_rate','ent_E', 'ent_I', 'EI_ent']
feature_cols_by_group = {
    g: [c for c in main_df.columns if c.startswith(f"{g}_")] for g in vip_feature_groups
}
all_feature_cols = sorted({c for cols in feature_cols_by_group.values() for c in cols})

def compute_vip(score_df, score_col, feature_cols_by_group, main_df, label):
    score_df = score_df[['subject_id', score_col]].dropna().copy()
    # Ensure we have N_MEGA in main_df for merging
    merged = score_df.merge(
        main_df[['N_MEGA'] + all_feature_cols],
        left_on='subject_id', right_on='N_MEGA', how='inner'
    )
    if merged.empty:
        raise ValueError(f"No overlap between {label} scores and main data")
    results = []
    for group, cols in feature_cols_by_group.items():
        corrs = merged[cols].apply(lambda col: col.corr(merged[score_col]))
        group_df = pd.DataFrame({
            'feature_group': group,
            'feature': cols,
            'vip': corrs.values,
            'phenotype': label
        })
        results.append(group_df)
    return pd.concat(results, ignore_index=True)

# Compute VIPs now to be ready for plotting
print("Computing VIP Scores...")
age_vip_df = compute_vip(age_best_scores_avg, 'score_with', feature_cols_by_group, main_df, 'Age')
sex_vip_df = compute_vip(sex_best_scores_avg, 'score_with', feature_cols_by_group, main_df, 'Sex')
diag_vip_df = compute_vip(diag_all_scores_avg, 'score_with', feature_cols_by_group, main_df, 'Diagnosis')
print("Data Preparation Complete.")

# --- FIGURE 1: PERFORMANCE & DISTRIBUTIONS ---
print("Generating Figure 1...")
fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(3, 3, figure=fig, wspace=WSPACE, hspace=HSPACE)

# ========== ROW 1: AGE ==========
ax_age_heat = fig.add_subplot(gs[0, 0])
create_heatmap(ax_age_heat, age_matrix, 'Age: h²', max_idx=age_best_idx, show_xticklabels=False)

# Age violin
ax_age_viol = fig.add_subplot(gs[0, 1])
age_viol_data = pd.DataFrame({
    'h2': pd.concat([age_best_h2['h2_with'], age_all_h2['h2_with']]),
    'combo': [age_best_combo]*len(age_best_h2) + ['all_features']*len(age_all_h2),
    'x': [0]*len(age_best_h2) + [0]*len(age_all_h2)
})
sns.violinplot(data=age_viol_data, x='x', y='h2', hue='combo', ax=ax_age_viol, 
               palette=VIOLIN_COLORS, inner='box', split=True, legend=False,
               linewidth=0, alpha=0.3, cut=0)
sns.stripplot(data=age_viol_data, x='x', y='h2', hue='combo', ax=ax_age_viol,
              palette=VIOLIN_COLORS, size=6, alpha=0.8, dodge=True, legend=False,
              linewidth=0, edgecolor='none')
age_best_label = age_best_combo.replace('_', ' ').replace('+', ' + ')
handles = [plt.Line2D([0], [0], color=VIOLIN_COLORS[0], lw=4, label=age_best_label),
           plt.Line2D([0], [0], color=VIOLIN_COLORS[1], lw=4, label='All Features')]
ax_age_viol.legend(handles=handles, loc='center left', fontsize=FONT_TICK+1)
ax_age_viol.set_title('Age: h² Comparison', fontsize=FONT_TITLE)
ax_age_viol.set_xlabel('')
ax_age_viol.set_ylabel('h²', fontsize=FONT_LABEL+2)
ax_age_viol.set_xticks([])
ymin, ymax = ax_age_viol.get_ylim()
yticks = np.linspace(ymin, ymax, 4)
ax_age_viol.set_yticks(yticks)
ax_age_viol.set_yticklabels([f'{y:.2f}' for y in yticks])
ax_age_viol.tick_params(labelsize=FONT_TICK+4)
ax_age_viol.yaxis.grid(True, alpha=0.3)

# Age PCEV distribution
ax_age_dist = fig.add_subplot(gs[0, 2])
xy = age_best_scores_avg[['Age', 'score_with']].dropna().values.T
if KDE_BANDWIDTH:
    kde = gaussian_kde(xy, bw_method=KDE_BANDWIDTH)
else:
    kde = gaussian_kde(xy)
density = kde(xy)
density_normalized = (density - density.min()) / (density.max() - density.min())
scatter = ax_age_dist.scatter(xy[0], xy[1], c=density_normalized, s=20, cmap='flare_r', alpha=0.6)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax_age_dist)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(scatter, cax=cax, alpha=1.0)
cbar.solids.set_alpha(1.0)
cbar.set_label('Density', fontsize=FONT_LABEL+3)
cbar.ax.tick_params(labelsize=FONT_TICK+3)
z = np.polyfit(xy[0], xy[1], 1)
p = np.poly1d(z)
x_line = np.linspace(xy[0].min(), xy[0].max(), 100)
ax_age_dist.plot(x_line, p(x_line), 'k-', linewidth=2, alpha=0.8)
ax_age_dist.set_xlabel('Age', fontsize=FONT_LABEL)
ax_age_dist.set_ylabel('PCEV Score', fontsize=FONT_LABEL)
ax_age_dist.set_title('Age: PCEV Distribution', fontsize=FONT_TITLE)
ax_age_dist.set_yticks([-5, 0, 5])
ax_age_dist.set_yticklabels(['-5.0', '0.0', '5.0'])
ax_age_dist.tick_params(labelsize=FONT_TICK+4)
ax_age_dist.yaxis.grid(True, alpha=0.3)
ax_age_dist.xaxis.grid(True, alpha=0.3)
age_h2_mean = age_best_h2['h2_with'].mean()
age_h2_std = age_best_h2['h2_with'].std()
age_best_combo_hr = age_best_combo.replace('_', ' ').replace('+', ' + ')
age_text = f"Best: {age_best_combo_hr}\nh² = {age_h2_mean:.3f}±{age_h2_std:.3f}"
ax_age_dist.text(0.05, 0.95, age_text, transform=ax_age_dist.transAxes,
                 fontsize=13, verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# ========== ROW 2: SEX ==========
ax_sex_heat = fig.add_subplot(gs[1, 0])
create_heatmap(ax_sex_heat, sex_matrix, 'Sex: |Cohen\'s d|', max_idx=sex_best_idx, show_xticklabels=False)

# Sex violin
ax_sex_viol = fig.add_subplot(gs[1, 1])
sex_viol_data = pd.DataFrame({
    'cohens_d': pd.concat([sex_best_metrics['cohens_d_with'].abs(), sex_all_metrics['cohens_d_with'].abs()]),
    'combo': [sex_best_combo]*len(sex_best_metrics) + ['all_features']*len(sex_all_metrics),
    'x': [0]*len(sex_best_metrics) + [0]*len(sex_all_metrics)
})
sns.violinplot(data=sex_viol_data, x='x', y='cohens_d', hue='combo', ax=ax_sex_viol,
               palette=VIOLIN_COLORS, inner='box', split=True, legend=False,
               linewidth=0, alpha=0.3, cut=0)
sns.stripplot(data=sex_viol_data, x='x', y='cohens_d', hue='combo', ax=ax_sex_viol,
              palette=VIOLIN_COLORS, size=6, alpha=0.8, dodge=True, legend=False,
              linewidth=0, edgecolor='none')
sex_best_label = sex_best_combo.replace('_', ' ')
handles = [plt.Line2D([0], [0], color=VIOLIN_COLORS[0], lw=4, label=sex_best_label),
           plt.Line2D([0], [0], color=VIOLIN_COLORS[1], lw=4, label='All Features')]
ax_sex_viol.legend(handles=handles, loc='center left', fontsize=FONT_TICK+1)
ax_sex_viol.set_title('Sex: |Cohen\'s d| Comparison', fontsize=FONT_TITLE)
ax_sex_viol.set_xlabel('')
ax_sex_viol.set_ylabel('|Cohen\'s d|', fontsize=FONT_LABEL+2)
ax_sex_viol.set_xticks([])
ymin, ymax = ax_sex_viol.get_ylim()
yticks = np.linspace(ymin, ymax, 4)
ax_sex_viol.set_yticks(yticks)
ax_sex_viol.set_yticklabels([f'{y:.2f}' for y in yticks])
ax_sex_viol.tick_params(labelsize=FONT_TICK+4)
ax_sex_viol.yaxis.grid(True, alpha=0.3)

# Sex PCEV distribution
ax_sex_dist = fig.add_subplot(gs[1, 2])
sns.violinplot(data=sex_best_scores_avg, x='Sex', y='score_with', hue='Sex', ax=ax_sex_dist,
               palette=['lightblue', 'lightcoral'], inner='box', legend=False,
               split=False, cut=0, linewidth=0)
for collection in ax_sex_dist.collections:
    if hasattr(collection, 'get_paths'):
        for path in collection.get_paths():
            vertices = path.vertices
            center_x = vertices[:, 0].mean()
            mask = vertices[:, 0] <= center_x
            vertices[~mask, 0] = center_x
for i, sex in enumerate(['Female', 'Male']):
    sex_data = sex_best_scores_avg[sex_best_scores_avg['Sex'] == sex]['score_with']
    x_pos = np.random.normal(i + 0.25, 0.05, size=len(sex_data))
    ax_sex_dist.scatter(x_pos, sex_data, c='black', s=20, alpha=0.4, 
                       edgecolors='none', linewidths=0, zorder=3)
ax_sex_dist.set_title('Sex: PCEV by Sex', fontsize=FONT_TITLE)
ax_sex_dist.set_xlabel('Sex', fontsize=FONT_LABEL)
ax_sex_dist.set_ylabel('PCEV Score', fontsize=FONT_LABEL)
ax_sex_dist.tick_params(labelsize=FONT_TICK+4)
ax_sex_dist.yaxis.grid(True, alpha=0.3)
sex_d_mean = sex_best_metrics['cohens_d_with'].mean()
sex_d_std = sex_best_metrics['cohens_d_with'].std()
sex_best_combo_hr = sex_best_combo.replace('_', ' ').replace('+', ' + ')
sex_text = f"Best: {sex_best_combo_hr}\nd = {sex_d_mean:.3f}±{sex_d_std:.3f}"
ax_sex_dist.text(0.05, 0.95, sex_text, transform=ax_sex_dist.transAxes,
                 fontsize=13, verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# ========== ROW 3: DIAGNOSIS ==========
ax_diag_heat = fig.add_subplot(gs[2, 0])
create_heatmap(ax_diag_heat, diag_matrix, 'Diagnosis: ε²', max_idx=diag_best_idx, show_xticklabels=True)

# Diagnosis violin
ax_diag_viol = fig.add_subplot(gs[2, 1])
diag_viol_data = pd.DataFrame({
    'epsilon': pd.concat([diag_best_metrics['epsilon_with'], diag_all_metrics['epsilon_with']]),
    'combo': [diag_best_combo]*len(diag_best_metrics) + ['all_features']*len(diag_all_metrics),
    'x': [0]*len(diag_best_metrics) + [0]*len(diag_all_metrics)
})
sns.violinplot(data=diag_viol_data, x='x', y='epsilon', hue='combo', ax=ax_diag_viol,
               palette=VIOLIN_COLORS, inner='box', split=True, legend=False,
               linewidth=0, alpha=0.3, cut=0)
sns.stripplot(data=diag_viol_data, x='x', y='epsilon', hue='combo', ax=ax_diag_viol,
              palette=VIOLIN_COLORS, size=6, alpha=0.8, dodge=True, legend=False,
              linewidth=0, edgecolor='none')
diag_best_label = diag_best_combo.replace('_', ' ').replace('+', ' + ')
handles = [plt.Line2D([0], [0], color=VIOLIN_COLORS[0], lw=4, label=diag_best_label),
           plt.Line2D([0], [0], color=VIOLIN_COLORS[1], lw=4, label='All Features')]
ax_diag_viol.legend(handles=handles, loc='center left', fontsize=FONT_TICK+1)
ax_diag_viol.set_title('Diagnosis: ε² Comparison', fontsize=FONT_TITLE)
ax_diag_viol.set_xlabel('')
ax_diag_viol.set_ylabel('ε²', fontsize=FONT_LABEL+2)
ax_diag_viol.set_xticks([])
ymin, ymax = ax_diag_viol.get_ylim()
yticks = np.linspace(ymin, ymax, 4)
ax_diag_viol.set_yticks(yticks)
ax_diag_viol.set_yticklabels([f'{y:.2f}' for y in yticks])
ax_diag_viol.tick_params(labelsize=FONT_TICK+4)
ax_diag_viol.yaxis.grid(True, alpha=0.3)

# Diagnosis PCEV distribution
ax_diag_dist = fig.add_subplot(gs[2, 2])
diag_order = ['CN', 'MCI', 'AD', 'FTD']
diag_all_scores_ordered = diag_all_scores_avg[diag_all_scores_avg['Diagnosis'].isin(diag_order)]
import matplotlib.colors as mcolors
set2_colors = plt.cm.Set2.colors[1:5]
sns.violinplot(data=diag_all_scores_ordered, x='Diagnosis', y='score_with', hue='Diagnosis',
               order=diag_order, ax=ax_diag_dist, palette=set2_colors, inner='box', 
               legend=False, cut=0, linewidth=0)
for collection in ax_diag_dist.collections:
    if hasattr(collection, 'get_paths'):
        for path in collection.get_paths():
            vertices = path.vertices
            center_x = vertices[:, 0].mean()
            mask = vertices[:, 0] <= center_x
            vertices[~mask, 0] = center_x
for i, diag in enumerate(diag_order):
    diag_data = diag_all_scores_ordered[diag_all_scores_ordered['Diagnosis'] == diag]['score_with']
    x_pos = np.random.normal(i + 0.25, 0.05, size=len(diag_data))
    ax_diag_dist.scatter(x_pos, diag_data, c='black', s=20, alpha=0.4, 
                        edgecolors='none', linewidths=0, zorder=3)
ax_diag_dist.set_title('Diagnosis: PCEV by Diagnosis', fontsize=FONT_TITLE)
ax_diag_dist.set_xlabel('Diagnosis', fontsize=FONT_LABEL)
ax_diag_dist.set_ylabel('PCEV Score', fontsize=FONT_LABEL)
ax_diag_dist.set_yticks([-10, 0, 10, 20])
ax_diag_dist.set_yticklabels(['-10', '0', '10', '20'])
ax_diag_dist.tick_params(labelsize=FONT_TICK+4)
ax_diag_dist.yaxis.grid(True, alpha=0.3)
diag_eps_mean = diag_all_metrics['epsilon_with'].mean()
diag_eps_std = diag_all_metrics['epsilon_with'].std()
diag_text = f"All Features\nε² = {diag_eps_mean:.3f}±{diag_eps_std:.3f}"
ax_diag_dist.text(0.05, 0.95, diag_text, transform=ax_diag_dist.transAxes,
                  fontsize=13, verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

fig.savefig(FIG_DIR / 'pcev_odq_phenotype_figure.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'pcev_odq_phenotype_figure.svg', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 1 Saved.")

# --- FIGURE 2: BRAIN MAPS (VIP SCORES) ---
print("Generating Figure 2...")

try:
    # Grid Plotting
    vip_feature_groups_grid = ['rate_E', 'EI_rate', 'ent_E', 'EI_ent']
    phenotypes = [
        ("Age", age_vip_df),
        ("Sex", sex_vip_df),
        ("Diagnosis", diag_vip_df),
    ]

    nrows = len(phenotypes)
    ncols = len(vip_feature_groups_grid)
    BRAIN_VIP_FIGSIZE = (18, 9)

    fig, axes = plt.subplots(nrows, ncols, figsize=BRAIN_VIP_FIGSIZE)
    # Adjust layout manually to remove spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    vip_vmin = -0.35
    vip_vmax = 0.35
    vip_cmap = 'RdBu_r'

    for row, (pheno_name, vip_df) in enumerate(phenotypes):
        for col, group in enumerate(vip_feature_groups_grid):
            ax = axes[row, col]
            # Extract VIP vector for this group
            # Sort by feature suffix to ensure correct brain mapping (1..90)
            vip_vec = (
                vip_df[vip_df['feature_group'] == group]
                .sort_values('feature', key=lambda x: x.str.extract(r'_(\d+)$').astype(int)[0])
                .vip.values
            )
            # Remove excluded regions (downsample from 90 to 82)
            vip_vec_82 = remove_excluded_regions(vip_vec)
            
            # Plot transparent brain map
            brain_fig = plot_brain_map(vip_vec_82, name=f"{pheno_name}{group}", 
                                    vmin=vip_vmin, vmax=vip_vmax, cmap=vip_cmap, savefile=None)
            
            # Convert to image for grid display
            brain_fig.canvas.draw()
            buf = brain_fig.canvas.buffer_rgba()
            img = np.asarray(buf).reshape(brain_fig.canvas.get_width_height()[::-1] + (4,))
            img = img[..., :3]  # drop alpha
            
            # Crop whitespace
            img_cropped = crop_image_auto(img)
            
            ax.imshow(img_cropped)
            plt.close(brain_fig) # Cleanup offscreen figure
            ax.axis('off')
            
            if row == 0:
                ax.set_title(group.replace('_', ' '), fontsize=15)
        
        # Add Phenotype Label on the left
        axes[row, 0].text(-0.05, 0.5, pheno_name, 
                        va='center', ha='center', fontsize=18, rotation=90, 
                        transform=axes[row, 0].transAxes, fontweight='bold')

    # Add Colorbar at bottom
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    cbar_ax = fig.add_axes([0.2, -0.04, 0.6, 0.04])
    norm = Normalize(vmin=vip_vmin, vmax=vip_vmax)
    sm = cm.ScalarMappable(cmap=vip_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('VIP', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(FIG_DIR / 'brain_vip_grid_E.pdf', dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(FIG_DIR / 'brain_vip_grid_E.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    print("Figure 2 Saved.")

except ImportError as e:
    print(f"Skipping Figure 2 due to missing library: {e}")
except Exception as e:
    print(f"Error generating Figure 2: {e}")
    import traceback
    traceback.print_exc()
