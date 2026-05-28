"""Compare rawSC vs optimized-SC OMAT goodness of fit against empirical group OMATs.

This script mirrors the lightweight vectorized logic used in
analysis/analysis_01_group_vs_individualized_models.py, but applies it to:

- data/derived/global_vectors_rawSC.npz
- data/derived/global_vectors.npz
- /data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/average_oinfo_matrices_by_group.csv

It produces:
- a per-group table with empirical-vs-simulated GOF for both models
- a summary table with paired statistics for the rawSC vs optimized-SC delta
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


GROUP_COLS = ["Dataset", "Diagnosis", "Sex", "Age_Range"]
TRIU_SIZE = 90 * 89 // 2


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    empirical_csv: Path
    rawsc_vectors: Path
    sc_vectors: Path
    output_dir: Path
    per_group_csv: Path
    summary_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rawSC and optimized-SC OMAT goodness of fit against empirical group OMATs."
    )
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
        empirical_csv=Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/average_oinfo_matrices_by_group.csv"),
        rawsc_vectors=repo_root / "data" / "derived" / "global_vectors_rawSC.npz",
        sc_vectors=repo_root / "data" / "derived" / "global_vectors.npz",
        output_dir=output_dir,
        per_group_csv=output_dir / "rawsc_vs_sc_gof_by_group.csv",
        summary_csv=output_dir / "rawsc_vs_sc_gof_summary.csv",
    )


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast row-wise Pearson correlation for two equally shaped matrices."""
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.sum(a_centered * b_centered, axis=1) / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr


def load_empirical_groups(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing empirical CSV: {path}")
    df = pd.read_csv(path)
    link_cols = [col for col in df.columns if col.startswith("link_")]
    if len(link_cols) != TRIU_SIZE:
        raise ValueError(f"Expected {TRIU_SIZE} link columns, found {len(link_cols)} in {path}")
    return df


def load_vectors_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing vectors archive: {path}")
    data = np.load(path, allow_pickle=True)
    return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


def group_key_from_row(row: pd.Series) -> str:
    dataset = str(row["Dataset"]).strip()
    diagnosis = str(row["Diagnosis"]).strip().upper()
    sex = str(row["Sex"]).strip()
    age = str(row["Age_Range"]).strip()
    return f"{dataset}_{diagnosis}_{sex}_{age}"


def paired_summary(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, float | str]:
    diff = a - b
    n = int(diff.size)
    if n == 0:
        return {
            "Model_Comparison": name,
            "N": 0,
            "Mean_Delta": np.nan,
            "Median_Delta": np.nan,
            "SD_Delta": np.nan,
            "Shapiro_p": np.nan,
            "Test_Type": "n/a",
            "Statistic": np.nan,
            "p_value": np.nan,
        }

    if n >= 3:
        _, shapiro_p = shapiro(diff)
    else:
        shapiro_p = 1.0

    if shapiro_p >= 0.01 and n >= 3:
        test_type = "Paired t-test"
        stat, p_val = ttest_rel(a, b)
    else:
        test_type = "Wilcoxon"
        try:
            stat, p_val = wilcoxon(a, b)
        except ValueError:
            stat, p_val = np.nan, np.nan

    return {
        "Model_Comparison": name,
        "N": n,
        "Mean_Delta": float(np.mean(diff)),
        "Median_Delta": float(np.median(diff)),
        "SD_Delta": float(np.std(diff, ddof=1)) if n >= 2 else np.nan,
        "Shapiro_p": float(shapiro_p),
        "Test_Type": test_type,
        "Statistic": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    paths = resolve_paths(repo_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    empirical_df = load_empirical_groups(paths.empirical_csv)
    link_cols = [col for col in empirical_df.columns if col.startswith("link_")]
    rawsc_vectors = load_vectors_npz(paths.rawsc_vectors)
    sc_vectors = load_vectors_npz(paths.sc_vectors)

    required_groups = [group_key_from_row(row) for _, row in empirical_df.iterrows()]
    missing_rawsc = [g for g in required_groups if g not in rawsc_vectors]
    missing_sc = [g for g in required_groups if g not in sc_vectors]
    if missing_rawsc:
        raise KeyError(f"Missing {len(missing_rawsc)} groups in rawSC archive, e.g. {missing_rawsc[:5]}")
    if missing_sc:
        raise KeyError(f"Missing {len(missing_sc)} groups in SC archive, e.g. {missing_sc[:5]}")

    empirical_matrix = empirical_df[link_cols].to_numpy(dtype=np.float64, copy=True)
    rawsc_matrix = np.stack([rawsc_vectors[g] for g in required_groups])
    sc_matrix = np.stack([sc_vectors[g] for g in required_groups])

    if empirical_matrix.shape != rawsc_matrix.shape or empirical_matrix.shape != sc_matrix.shape:
        raise ValueError(
            "Shape mismatch among empirical, rawSC, and optimized-SC matrices: "
            f"empirical={empirical_matrix.shape}, rawSC={rawsc_matrix.shape}, SC={sc_matrix.shape}"
        )

    gof_rawsc = rowwise_correlation(empirical_matrix, rawsc_matrix)
    gof_sc = rowwise_correlation(empirical_matrix, sc_matrix)
    delta = gof_rawsc - gof_sc

    out_df = empirical_df.loc[:, GROUP_COLS].copy()
    out_df["group_id"] = required_groups
    out_df["gof_empirical_vs_rawSC"] = gof_rawsc
    out_df["gof_empirical_vs_sc_optimized"] = gof_sc
    out_df["delta_rawSC_minus_sc"] = delta
    out_df["better_model"] = np.where(delta > 0, "rawSC", np.where(delta < 0, "SC_optimized", "tie"))
    out_df.to_csv(paths.per_group_csv, index=False)

    summary_rows = [
        {
            "Scope": "Overall",
            **paired_summary("rawSC - SC_optimized", gof_rawsc, gof_sc),
            "Mean_rawSC": float(np.mean(gof_rawsc)),
            "Mean_SC_optimized": float(np.mean(gof_sc)),
        }
    ]

    for dataset in sorted(empirical_df["Dataset"].astype(str).unique()):
        mask = empirical_df["Dataset"].astype(str) == dataset
        summary_rows.append(
            {
                "Scope": dataset,
                **paired_summary(
                    f"{dataset}: rawSC - SC_optimized",
                    gof_rawsc[mask.to_numpy()],
                    gof_sc[mask.to_numpy()],
                ),
                "Mean_rawSC": float(np.mean(gof_rawsc[mask.to_numpy()])),
                "Mean_SC_optimized": float(np.mean(gof_sc[mask.to_numpy()])),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.summary_csv, index=False)

    print(f"Saved per-group results to {paths.per_group_csv}")
    print(f"Saved summary results to {paths.summary_csv}")
    print("\nTop 5 groups where rawSC improves the most:")
    top = out_df.sort_values("delta_rawSC_minus_sc", ascending=False).head(5)
    print(top[["group_id", "gof_empirical_vs_rawSC", "gof_empirical_vs_sc_optimized", "delta_rawSC_minus_sc"]].to_string(index=False))
    print("\nTop 5 groups where optimized SC improves the most:")
    bottom = out_df.sort_values("delta_rawSC_minus_sc", ascending=True).head(5)
    print(bottom[["group_id", "gof_empirical_vs_rawSC", "gof_empirical_vs_sc_optimized", "delta_rawSC_minus_sc"]].to_string(index=False))


if __name__ == "__main__":
    main()
