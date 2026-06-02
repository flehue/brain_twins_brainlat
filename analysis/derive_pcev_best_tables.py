#!/usr/bin/env python3
"""Produce tables that compare best subset vs all features for PCEV results.

This script is intended as an isolated replacement for the notebook cells that
write `pcev_phenotypes_best_vs_all.csv` and `pcev_expotypes_best_vs_all.csv`.
It reads analysis outputs from `analysis/results/pcev_results` and writes the
resulting tables to a new output directory, never overwriting existing files.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = Path(os.environ.get('PCEV_RESULTS_DIR', REPO_ROOT / 'analysis' / 'results' / 'pcev_results'))
AGE_DIR = RESULTS_BASE / 'age_odq_only_no_scanner'
SEX_DIR = RESULTS_BASE / 'sex_odq_only_no_scanner'
DIAG_DIR = RESULTS_BASE / 'diagnosis_odq_only_no_scanner'
EXPO_RESULTS_BASE = RESULTS_BASE / 'exposome_odq_only_no_scanner'

EXPOSOME_GROUPS = OrderedDict([
    ('Air Pollution', None),
    ('Green space access', None),
    ('Temperature', None),
    ('Precipitation-droughts', None),
    ('Soil and water quality', None),
    ('Climate disasters', None),
    ('Disease-related mortality', None),
    ('Socioeconomic', None),
    ('Democracy', None),
    ('Democracy-Summary', None),
    ('Migration', None),
])


def parse_combo_label(label: str) -> list[str] | None:
    if label == 'all_features':
        return None

    single_features = ['EI_ent', 'EI_rate', 'rate_I', 'rate_E', 'ent_E', 'ent_I']
    if label in single_features:
        return [label]

    for feat1 in single_features:
        for feat2 in single_features:
            if label == f"{feat1}_{feat2}" or label == f"{feat2}_{feat1}":
                return sorted([feat1, feat2])

    parts = label.split('_')
    if len(parts) == 4:
        return sorted([f"{parts[0]}_{parts[1]}", f"{parts[2]}_{parts[3]}"])
    if len(parts) == 3:
        if f"{parts[0]}_{parts[1]}" in single_features:
            return sorted([f"{parts[0]}_{parts[1]}", parts[2]])
        if f"{parts[1]}_{parts[2]}" in single_features:
            return sorted([parts[0], f"{parts[1]}_{parts[2]}"])

    return [label]


def build_matrix(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    combos = [c for c in df['combo'].unique() if c != 'all_features']
    combo_groups = {c: parse_combo_label(c) for c in combos}

    all_groups = sorted({group for groups in combo_groups.values() if groups for group in groups})
    matrix = pd.DataFrame(np.nan, index=all_groups, columns=all_groups)

    for combo, groups in combo_groups.items():
        if not groups:
            continue
        avg_val = df.loc[df['combo'] == combo, metric_col].mean()
        if len(groups) == 1:
            matrix.loc[groups[0], groups[0]] = avg_val
        elif len(groups) == 2:
            g1, g2 = groups
            matrix.loc[g1, g2] = avg_val
            matrix.loc[g2, g1] = avg_val

    return matrix


def find_best_combo(matrix: pd.DataFrame) -> str:
    stacked = matrix.stack(dropna=True)
    if stacked.empty:
        raise ValueError('No valid combos found in matrix')
    max_idx = stacked.idxmax()
    if max_idx[0] == max_idx[1]:
        return max_idx[0]
    return f"{max_idx[0]}+{max_idx[1]}"


def combo_to_filename(combo_label: str) -> str:
    return combo_label.replace('+', '_')


def paired_sign_flip_test(data_all: np.ndarray, data_best: np.ndarray, n_perm: int = 10000, statistic: str = 'mean') -> tuple[float, float]:
    diff = np.asarray(data_all) - np.asarray(data_best)
    if statistic == 'mean':
        obs_stat = np.mean(diff)
    elif statistic == 'median':
        obs_stat = np.median(diff)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    signs = np.random.choice([-1, 1], size=(n_perm, len(diff)))
    if statistic == 'mean':
        perm_stats = np.mean(signs * diff, axis=1)
    else:
        perm_stats = np.median(signs * diff, axis=1)

    p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    return float(obs_stat), float(p_val)


def bootstrap_ci(data: np.ndarray, n_boot: int = 10000, statistic: str = 'mean') -> tuple[float, float]:
    data = np.asarray(data)
    indices = np.random.randint(0, len(data), size=(n_boot, len(data)))
    samples = data[indices]
    if statistic == 'mean':
        boot_stats = np.mean(samples, axis=1)
    elif statistic == 'median':
        boot_stats = np.median(samples, axis=1)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def get_simple_stats(data: np.ndarray) -> tuple[float, float, int]:
    data = np.asarray(data)
    return float(np.mean(data)), float(np.std(data, ddof=1)), int(len(data))


def safe_name(name: str) -> str:
    return name.replace(' ', '_')


def load_metric_files(directory: Path, pattern: str, metric_col: str) -> pd.DataFrame:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files found in {directory} matching {pattern}')

    if '_h2_per_repeat' in pattern:
        suffix = '_h2_per_repeat'
    elif '_metrics_per_repeat' in pattern:
        suffix = '_metrics_per_repeat'
    else:
        suffix = pattern.replace('*', '')

    pieces = []
    for file_path in files:
        combo_name = file_path.stem
        if combo_name.endswith(suffix):
            combo_name = combo_name[: -len(suffix)]
        df = pd.read_csv(file_path)
        df['combo'] = combo_name
        pieces.append(df)

    return pd.concat(pieces, ignore_index=True)


def phenotype_best_table(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    age_df = load_metric_files(AGE_DIR, '*_h2_per_repeat.csv', 'h2_with')
    sex_df = load_metric_files(SEX_DIR, '*_metrics_per_repeat.csv', 'cohens_d_with')
    diag_df = load_metric_files(DIAG_DIR, '*_metrics_per_repeat.csv', 'epsilon_with')

    age_best_combo = find_best_combo(build_matrix(age_df, 'h2_with'))
    sex_best_combo = find_best_combo(build_matrix(sex_df, 'cohens_d_with').abs())
    diag_best_combo = find_best_combo(build_matrix(diag_df, 'epsilon_with'))

    age_best_file = combo_to_filename(age_best_combo)
    sex_best_file = combo_to_filename(sex_best_combo)
    diag_best_file = combo_to_filename(diag_best_combo)

    age_best_h2 = pd.read_csv(AGE_DIR / f'{age_best_file}_h2_per_repeat.csv')
    age_all_h2 = pd.read_csv(AGE_DIR / 'all_features_h2_per_repeat.csv')

    sex_best_metrics = pd.read_csv(SEX_DIR / f'{sex_best_file}_metrics_per_repeat.csv')
    sex_all_metrics = pd.read_csv(SEX_DIR / 'all_features_metrics_per_repeat.csv')

    diag_best_metrics = pd.read_csv(DIAG_DIR / f'{diag_best_file}_metrics_per_repeat.csv')
    diag_all_metrics = pd.read_csv(DIAG_DIR / 'all_features_metrics_per_repeat.csv')

    results_list = []
    analysis_targets = [
        {
            'Phenotype': 'Age',
            'Metric': 'h²',
            'Best_Data': age_best_h2['h2_with'].values,
            'All_Data': age_all_h2['h2_with'].values,
            'Best_Label': age_best_combo,
        },
        {
            'Phenotype': 'Sex',
            'Metric': "|Cohen's d|",
            'Best_Data': np.abs(sex_best_metrics['cohens_d_with'].values),
            'All_Data': np.abs(sex_all_metrics['cohens_d_with'].values),
            'Best_Label': sex_best_combo,
        },
        {
            'Phenotype': 'Sex',
            'Metric': 'h²',
            'Best_Data': sex_best_metrics['h2_with'].values,
            'All_Data': sex_all_metrics['h2_with'].values,
            'Best_Label': sex_best_combo,
        },
        {
            'Phenotype': 'Diagnosis',
            'Metric': 'ε²',
            'Best_Data': diag_best_metrics['epsilon_with'].values,
            'All_Data': diag_all_metrics['epsilon_with'].values,
            'Best_Label': diag_best_combo,
        },
        {
            'Phenotype': 'Diagnosis',
            'Metric': 'h²',
            'Best_Data': diag_best_metrics['h2_with'].values,
            'All_Data': diag_all_metrics['h2_with'].values,
            'Best_Label': diag_best_combo,
        },
    ]

    for target in analysis_targets:
        best_mean, best_std, n = get_simple_stats(target['Best_Data'])
        all_mean, all_std, _ = get_simple_stats(target['All_Data'])

        obs_stat, pval = paired_sign_flip_test(target['All_Data'], target['Best_Data'])
        diffs = np.asarray(target['All_Data']) - np.asarray(target['Best_Data'])
        ci_lower, ci_upper = bootstrap_ci(diffs)

        results_list.append({
            'Phenotype': target['Phenotype'],
            'Metric': target['Metric'],
            'Best Subset': target['Best_Label'],
            'Best Mean (SD)': f"{best_mean:.6f} ({best_std:.6f})",
            'All Features Mean (SD)': f"{all_mean:.6f} ({all_std:.6f})",
            'N': n,
            'Test': 'Paired sign-flip permutation (repeat-level)',
            'Δ (All - Best)': f"{obs_stat:.6f}",
            'Δ 95% CI': f"[{ci_lower:.6f}, {ci_upper:.6f}]",
            'P-value': '< 0.001' if pval < 0.001 else f"{pval:.4f}",
        })

    phenotype_path = output_dir / 'pcev_phenotypes_best_vs_all.csv'
    pd.DataFrame(results_list).to_csv(phenotype_path, index=False)
    return phenotype_path


def exposome_best_table(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    exposome_names = [name for name in EXPOSOME_GROUPS if name != 'Democracy-Summary']

    results_list = []

    for expo_name in exposome_names:
        expo_dir = EXPO_RESULTS_BASE / safe_name(expo_name)
        if not expo_dir.exists():
            raise FileNotFoundError(f'Missing exposome directory: {expo_dir}')

        summary_df = pd.read_csv(expo_dir / 'all_combos_summary.csv')
        non_all = summary_df[summary_df['combo_label'] != 'all_features'].copy()
        if non_all.empty:
            raise ValueError(f'No non-all_features combos found for {expo_name}')

        best_row = non_all.sort_values('h2_with_mean', ascending=False).iloc[0]
        best_combo = best_row['combo_label']
        best_safe = combo_to_filename(best_combo)

        h2_best = pd.read_csv(expo_dir / f'{best_safe}_h2_per_repeat.csv')
        h2_all = pd.read_csv(expo_dir / 'all_features_h2_per_repeat.csv')

        best_data = h2_best['h2_with'].values
        all_data = h2_all['h2_with'].values

        best_mean, best_std, n = get_simple_stats(best_data)
        all_mean, all_std, _ = get_simple_stats(all_data)

        obs_stat, pval = paired_sign_flip_test(all_data, best_data)
        diffs = np.asarray(all_data) - np.asarray(best_data)
        ci_lower, ci_upper = bootstrap_ci(diffs)

        results_list.append({
            'Exposome': expo_name,
            'Metric': 'h²',
            'Best Subset': best_combo,
            'Best Mean (SD)': f"{best_mean:.6f} ({best_std:.6f})",
            'All Features Mean (SD)': f"{all_mean:.6f} ({all_std:.6f})",
            'N': n,
            'Test': 'Paired sign-flip permutation (repeat-level)',
            'Δ (All - Best)': f"{obs_stat:.6f}",
            'Δ 95% CI': f"[{ci_lower:.6f}, {ci_upper:.6f}]",
            'P-value': '< 0.001' if pval < 0.001 else f"{pval:.4f}",
        })

    exposome_path = output_dir / 'pcev_expotypes_best_vs_all.csv'
    pd.DataFrame(results_list).to_csv(exposome_path, index=False)
    return exposome_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Derive PCEV best_vs_all tables.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directory where derived tables will be written.')
    parser.add_argument('--results-dir', type=Path, default=RESULTS_BASE,
                        help='Root directory for PCEV result folders.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite the output directory if it already exists.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = REPO_ROOT / 'analysis' / f'tmp_pcev_best_tables_{timestamp}'

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f'Output directory {output_dir} already exists and is not empty. Use --force to overwrite.')

    if output_dir.exists() and args.force:
        pass
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    global RESULTS_BASE, AGE_DIR, SEX_DIR, DIAG_DIR, EXPO_RESULTS_BASE
    RESULTS_BASE = args.results_dir
    AGE_DIR = RESULTS_BASE / 'age_odq_only_no_scanner'
    SEX_DIR = RESULTS_BASE / 'sex_odq_only_no_scanner'
    DIAG_DIR = RESULTS_BASE / 'diagnosis_odq_only_no_scanner'
    EXPO_RESULTS_BASE = RESULTS_BASE / 'exposome_odq_only_no_scanner'

    print('Using results dir:', RESULTS_BASE)
    print('Writing derived tables to:', output_dir)

    phenotype_path = phenotype_best_table(output_dir)
    exposome_path = exposome_best_table(output_dir)

    print('Created phenotype table:', phenotype_path)
    print('Created exposome table:', exposome_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
