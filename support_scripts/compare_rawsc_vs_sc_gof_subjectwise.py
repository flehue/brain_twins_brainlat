"""Subject-wise GOF comparison for rawSC vs optimized SC.

This version matches the analysis pattern used in
analysis/analysis_01_group_vs_individualized_models.py:

- empirical OMATs are taken from data/derived/Omats_FCs/concat.npy
- each subject is matched to its group-level simulated vector by metadata.csv
- GOF is the row-wise Pearson correlation between empirical and simulated vectors

That subject-wise weighting is what reproduces the figure-level values like
the global GOF ~ 0.550 in analysis_01.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, rankdata, shapiro, ttest_rel, wilcoxon


GROUP_COLS = ["Dataset", "Diagnosis", "Sex", "Age_Range"]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    metadata_path: Path
    empirical_concat_path: Path
    rawsc_vectors_path: Path
    sc_vectors_path: Path
    output_dir: Path
    per_subject_csv: Path
    summary_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rawSC and optimized-SC GOF against empirical subject-level OMATs.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root (default: parent of this script).",
    )
    return parser.parse_args()


def resolve_paths(repo_root: Path) -> Paths:
    output_dir = repo_root / "analysis" / "results"
    return Paths(
        repo_root=repo_root,
        metadata_path=repo_root / "data" / "derived" / "metadata.csv",
        empirical_concat_path=repo_root / "data" / "derived" / "Omats_FCs" / "concat.npy",
        rawsc_vectors_path=repo_root / "data" / "derived" / "global_vectors_rawSC.npz",
        sc_vectors_path=repo_root / "data" / "derived" / "global_vectors.npz",
        output_dir=output_dir,
        per_subject_csv=output_dir / "rawsc_vs_sc_gof_subjectwise.csv",
        summary_csv=output_dir / "rawsc_vs_sc_gof_subjectwise_summary.csv",
    )


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast row-wise Pearson correlation."""
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.sum(a_centered * b_centered, axis=1) / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr


def load_empirical_lookup(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing empirical concat.npy at {path}")
    lookup = np.load(path, allow_pickle=True).item()
    return {str(k): np.asarray(v, dtype=np.float64) for k, v in lookup.items()}


def load_vectors_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing vectors archive at {path}")
    data = np.load(path, allow_pickle=True)
    return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


def get_bootstrap_ci(data: np.ndarray, statistic_func, n_resamples=1000, confidence_level=0.95, bounds=None):
    if len(data) < 2:
        return np.nan, np.nan
    try:
        res = bootstrap(
            (data,),
            statistic_func,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="percentile",
            random_state=42,
        )
        low, high = res.confidence_interval.low, res.confidence_interval.high
        if bounds:
            low = max(bounds[0], low)
            high = min(bounds[1], high)
        return low, high
    except Exception:
        return np.nan, np.nan


def cohens_d_func(diff: np.ndarray) -> float:
    if len(diff) < 2:
        return np.nan
    return float(np.mean(diff) / np.std(diff, ddof=1))


def rank_biserial_func(diff: np.ndarray) -> float:
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return 0.0
    return float((w_pos - w_neg) / total)


def analyze_comparison(name: str, global_gof: np.ndarray, rawsc_gof: np.ndarray) -> dict[str, float | str]:
    diff = rawsc_gof - global_gof
    n = len(diff)
    if n >= 3:
        _, shapiro_p = shapiro(diff)
    else:
        shapiro_p = 1.0

    if shapiro_p >= 0.01 and n >= 3:
        test_type = "Paired t-test"
        stat, p_val = ttest_rel(rawsc_gof, global_gof)
        eff_size_type = "Cohen's d"
        eff_size = cohens_d_func(diff)
        es_ci_low, es_ci_high = get_bootstrap_ci(diff, cohens_d_func)
        statistic_name = "t"
        df = n - 1
    else:
        test_type = "Wilcoxon"
        try:
            stat, p_val = wilcoxon(rawsc_gof, global_gof)
        except ValueError:
            stat, p_val = np.nan, np.nan
        eff_size_type = "Rank-biserial"
        eff_size = rank_biserial_func(diff)
        es_ci_low, es_ci_high = get_bootstrap_ci(diff, rank_biserial_func, bounds=(-1, 1))
        statistic_name = "W"
        df = np.nan

    return {
        "Scope": name,
        "N": n,
        "Global_Mean": float(np.mean(global_gof)),
        "Global_SD": float(np.std(global_gof, ddof=1)),
        "Global_CI_Low": float(get_bootstrap_ci(global_gof, np.mean, bounds=(0, 1))[0]),
        "Global_CI_High": float(get_bootstrap_ci(global_gof, np.mean, bounds=(0, 1))[1]),
        "RawSC_Mean": float(np.mean(rawsc_gof)),
        "RawSC_SD": float(np.std(rawsc_gof, ddof=1)),
        "RawSC_CI_Low": float(get_bootstrap_ci(rawsc_gof, np.mean, bounds=(0, 1))[0]),
        "RawSC_CI_High": float(get_bootstrap_ci(rawsc_gof, np.mean, bounds=(0, 1))[1]),
        "Delta_Mean": float(np.mean(diff)),
        "Delta_SD": float(np.std(diff, ddof=1)),
        "Delta_CI_Low": float(get_bootstrap_ci(diff, np.mean, bounds=(-1, 1))[0]),
        "Delta_CI_High": float(get_bootstrap_ci(diff, np.mean, bounds=(-1, 1))[1]),
        "Shapiro_p": float(shapiro_p),
        "Test_Type": test_type,
        "Statistic_Name": statistic_name,
        "Statistic_Value": float(stat) if np.isfinite(stat) else np.nan,
        "DF": df,
        "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        "Effect_Size_Type": eff_size_type,
        "Effect_Size": float(eff_size) if np.isfinite(eff_size) else np.nan,
        "ES_CI_Low": float(es_ci_low) if np.isfinite(es_ci_low) else np.nan,
        "ES_CI_High": float(es_ci_high) if np.isfinite(es_ci_high) else np.nan,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    paths = resolve_paths(repo_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(paths.metadata_path)
    if metadata.empty:
        raise ValueError(f"Empty metadata at {paths.metadata_path}")

    empirical_lookup = load_empirical_lookup(paths.empirical_concat_path)
    rawsc_vectors = load_vectors_npz(paths.rawsc_vectors_path)
    sc_vectors = load_vectors_npz(paths.sc_vectors_path)

    nmega_ids = metadata["N_MEGA"].astype(str).tolist()
    group_ids = metadata["group_id"].astype(str).tolist()

    missing_empirical = [n for n in nmega_ids if n not in empirical_lookup]
    missing_rawsc = [g for g in group_ids if g not in rawsc_vectors]
    missing_sc = [g for g in group_ids if g not in sc_vectors]
    if missing_empirical:
        raise KeyError(f"Missing empirical vectors for {len(missing_empirical)} subjects, e.g. {missing_empirical[:5]}")
    if missing_rawsc:
        raise KeyError(f"Missing rawSC vectors for {len(missing_rawsc)} groups, e.g. {missing_rawsc[:5]}")
    if missing_sc:
        raise KeyError(f"Missing optimized-SC vectors for {len(missing_sc)} groups, e.g. {missing_sc[:5]}")

    empirical_matrix = np.stack([empirical_lookup[nmega] for nmega in nmega_ids])
    rawsc_matrix = np.stack([rawsc_vectors[group] for group in group_ids])
    sc_matrix = np.stack([sc_vectors[group] for group in group_ids])

    if empirical_matrix.shape[1] != rawsc_matrix.shape[1] or empirical_matrix.shape[1] != sc_matrix.shape[1]:
        raise ValueError(
            "Vector length mismatch among empirical, rawSC, and optimized-SC matrices: "
            f"empirical={empirical_matrix.shape}, rawSC={rawsc_matrix.shape}, sc={sc_matrix.shape}"
        )

    rawsc_gof = rowwise_correlation(empirical_matrix, rawsc_matrix)
    sc_gof = rowwise_correlation(empirical_matrix, sc_matrix)
    delta = rawsc_gof - sc_gof

    comparison_df = metadata.loc[:, GROUP_COLS + ["N_MEGA", "group_id"]].copy()
    comparison_df["gof_empirical_vs_rawSC"] = rawsc_gof
    comparison_df["gof_empirical_vs_sc_optimized"] = sc_gof
    comparison_df["delta_rawSC_minus_sc"] = delta
    comparison_df["better_model"] = np.where(delta > 0, "rawSC", np.where(delta < 0, "SC_optimized", "tie"))
    comparison_df.to_csv(paths.per_subject_csv, index=False)

    summary_rows = [analyze_comparison("Overall", sc_gof, rawsc_gof)]
    for dataset, df_group in comparison_df.groupby("Dataset", sort=True):
        idx = df_group.index.to_numpy()
        summary_rows.append(analyze_comparison(str(dataset), sc_gof[idx], rawsc_gof[idx]))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.summary_csv, index=False)

    print(f"Saved subject-level results to {paths.per_subject_csv}")
    print(f"Saved summary results to {paths.summary_csv}")
    print("\nOverall means:")
    print(f"  rawSC mean = {rawsc_gof.mean():.6f}, SD = {rawsc_gof.std(ddof=1):.6f}")
    rawsc_ci = get_bootstrap_ci(rawsc_gof, np.mean, bounds=(0, 1))
    print(f"  rawSC 95% CI = [{rawsc_ci[0]:.6f}, {rawsc_ci[1]:.6f}]")
    print(f"  SC optimized mean = {sc_gof.mean():.6f}, SD = {sc_gof.std(ddof=1):.6f}")
    sc_ci = get_bootstrap_ci(sc_gof, np.mean, bounds=(0, 1))
    print(f"  SC optimized 95% CI = [{sc_ci[0]:.6f}, {sc_ci[1]:.6f}]")
    print(f"  delta mean = {delta.mean():.6f}")
    print(f"  rawSC better count = {int((delta > 0).sum())}")
    print(f"  SC better count = {int((delta < 0).sum())}")


if __name__ == "__main__":
    main()
