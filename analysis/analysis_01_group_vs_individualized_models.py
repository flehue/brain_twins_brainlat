"""Global vs personalized OMAT goodness-of-fit comparison.

Uses precomputed inputs stored in data/derived:
- concat.npy
- global_vectors.npz
- personalized_vectors.npz
- metadata.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, rankdata, bootstrap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


plt.rcParams["text.usetex"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


GROUP_COLS = ["Dataset", "Diagnosis", "Sex", "Age_Range"]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    figures_dir: Path
    tables_dir: Path
    concat_path: Path
    metadata_path: Path
    personalized_vectors_path: Path
    global_vectors_path: Path


def resolve_paths(repo_root: Path) -> Paths:
    data_dir = repo_root / "data" / "derived"
    return Paths(
        repo_root=repo_root,
        data_dir=data_dir,
        figures_dir=repo_root / "analysis" / "figures",
        tables_dir=repo_root / "analysis" / "tables",
        concat_path=data_dir / "concat.npy",
        metadata_path=data_dir / "metadata.csv",
        personalized_vectors_path=data_dir / "personalized_vectors.npz",
        global_vectors_path=data_dir / "global_vectors.npz",
    )


def load_empirical_lookup(concat_path: Path) -> dict[str, np.ndarray]:
    if not concat_path.exists():
        raise FileNotFoundError(f"Missing concat.npy at {concat_path}.")
    lookup = np.load(concat_path, allow_pickle=True).item()
    return {str(k): np.asarray(v, dtype=np.float64) for k, v in lookup.items()}


def load_vectors_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    data = np.load(path, allow_pickle=True)
    return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.sum(a_centered * b_centered, axis=1) / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr


def get_bootstrap_ci(data: np.ndarray, statistic_func, n_resamples=1000, confidence_level=0.95, bounds=None):
    if len(data) < 2:
        return np.nan, np.nan
    try:
        res = bootstrap(
            (data,), statistic_func, confidence_level=confidence_level,
            n_resamples=n_resamples, method="percentile", random_state=42
        )
        low, high = res.confidence_interval.low, res.confidence_interval.high
        if bounds:
            low = max(bounds[0], low)
            high = min(bounds[1], high)
        return low, high
    except Exception as exc:
        print(f"Bootstrap failed: {exc}")
        return np.nan, np.nan


def cohens_d_func(diff: np.ndarray) -> float:
    if len(diff) < 2:
        return np.nan
    return np.mean(diff) / np.std(diff, ddof=1)


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
    return (w_pos - w_neg) / total


def analyze_group(group_name: str, df_group: pd.DataFrame) -> dict[str, float | str]:
    global_gof = df_group["gof_empirical_vs_global"].values
    pers_gof = df_group["gof_empirical_vs_personalized"].values
    diff = pers_gof - global_gof
    n = len(diff)

    desc_stats: dict[str, float | str] = {
        "Dataset": group_name,
        "N": n,
        "Global_Mean": np.mean(global_gof),
        "Global_Median": np.median(global_gof),
        "Global_SD": np.std(global_gof, ddof=1),
        "Pers_Mean": np.mean(pers_gof),
        "Pers_Median": np.median(pers_gof),
        "Pers_SD": np.std(pers_gof, ddof=1),
        "Delta_Mean": np.mean(diff),
        "Delta_Median": np.median(diff),
        "Delta_SD": np.std(diff, ddof=1),
    }

    desc_stats["Global_CI_Low"], desc_stats["Global_CI_High"] = get_bootstrap_ci(global_gof, np.mean, bounds=(0, 1))
    desc_stats["Pers_CI_Low"], desc_stats["Pers_CI_High"] = get_bootstrap_ci(pers_gof, np.mean, bounds=(0, 1))
    desc_stats["Delta_CI_Low"], desc_stats["Delta_CI_High"] = get_bootstrap_ci(diff, np.mean, bounds=(-1, 1))

    if n >= 3:
        _, shapiro_p = shapiro(diff)
    else:
        shapiro_p = 1.0

    desc_stats["Shapiro_p"] = shapiro_p

    if shapiro_p >= 0.01 and n >= 3:
        test_type = "Paired t-test"
        stat, p_val = ttest_rel(pers_gof, global_gof)
        eff_size_type = "Cohen's d"
        eff_size = cohens_d_func(diff)
        es_ci_low, es_ci_high = get_bootstrap_ci(diff, cohens_d_func)
        statistic_name = "t"
        df = n - 1
    else:
        test_type = "Wilcoxon"
        try:
            stat, p_val = wilcoxon(pers_gof, global_gof)
        except ValueError:
            stat, p_val = np.nan, np.nan
        eff_size_type = "Rank-biserial"
        eff_size = rank_biserial_func(diff)
        es_ci_low, es_ci_high = get_bootstrap_ci(diff, rank_biserial_func, bounds=(-1, 1))
        statistic_name = "W"
        df = np.nan

    desc_stats.update({
        "Test_Type": test_type,
        "Statistic_Name": statistic_name,
        "Statistic_Value": stat,
        "DF": df,
        "p_value": p_val,
        "Effect_Size_Type": eff_size_type,
        "Effect_Size": eff_size,
        "ES_CI_Low": es_ci_low,
        "ES_CI_High": es_ci_high,
    })

    return desc_stats


def build_results_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    results_list = [analyze_group("Whole Dataset", comparison_df)]
    for dataset_name, group_df in comparison_df.groupby("Dataset"):
        results_list.append(analyze_group(dataset_name, group_df))

    results_df = pd.DataFrame(results_list)
    cols = [
        "Dataset", "N", "Shapiro_p", "Test_Type", "Statistic_Name", "Statistic_Value", "DF", "p_value",
        "Effect_Size_Type", "Effect_Size", "ES_CI_Low", "ES_CI_High",
        "Global_Mean", "Global_SD", "Global_CI_Low", "Global_CI_High",
        "Pers_Mean", "Pers_SD", "Pers_CI_Low", "Pers_CI_High",
        "Delta_Mean", "Delta_SD", "Delta_CI_Low", "Delta_CI_High",
    ]
    return results_df[cols]


def plot_last_figure(comparison_df: pd.DataFrame, all_groups: list[str], figures_dir: Path) -> None:
    dataset_colors = {
        "ADNI-PET": "#66c2a5",
        "ALFA": "#fc8d62",
        "ARWIBO": "#8da0cb",
        "CamCan": "#e78ac3",
        "FRONTIERS": "#a6d854",
        "GERO": "#ffd92f",
        "HCP": "#e5c494",
        "SRPBS": "#b3b3b3",
    }
    sex_colors = {"Female": "#d62728", "Male": "#1f77b4"}
    separation = 0.08
    tick_size = 16

    def plot_split_violin(ax, data_left, data_right, position, color, edge_left, edge_right, width=0.7, linewidth=3):
        for data, pos_offset, edge_color in (
            (data_left, -separation / 2, edge_left),
            (data_right, separation / 2, edge_right),
        ):
            if len(data) == 0:
                continue
            parts = ax.violinplot(
                [data],
                positions=[position + pos_offset],
                widths=width,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(edge_color)
                pc.set_linewidth(linewidth)
                pc.set_alpha(0.7)
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                vertices = pc.get_paths()[0].vertices
                vertices[:, 0] = np.clip(
                    vertices[:, 0],
                    -np.inf if pos_offset < 0 else m,
                    m if pos_offset < 0 else np.inf,
                )

    fig = plt.figure(figsize=(24, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 4], wspace=0.05, figure=fig)

    ax0 = fig.add_subplot(gs[0])
    datasets = sorted(comparison_df["Dataset"].unique())

    for i, dataset in enumerate(datasets):
        dataset_data = comparison_df[comparison_df["Dataset"] == dataset]
        color = dataset_colors.get(dataset, "#999999")
        plot_split_violin(
            ax0,
            dataset_data["gof_empirical_vs_global"].values,
            dataset_data["gof_empirical_vs_personalized"].values,
            i,
            color,
            "#d3d3d3",
            "#000000",
        )

    ax0.set_ylabel("Goodness of Fit (GOF)", fontsize=18)
    ax0.set_xticks(range(len(datasets)))
    ax0.set_xticklabels(datasets, rotation=45, ha="right")
    ax0.set_xlim(-0.5, len(datasets) - 0.5)
    ax0.set_ylim(0.2, 0.9)
    ax0.tick_params(axis="both", labelsize=tick_size)
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="#d3d3d3", linewidth=3, alpha=0.7, label="Global"),
            Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="#000000", linewidth=3, alpha=0.7, label="Personalized"),
        ],
        loc="lower right",
        fontsize=13,
    )

    ax1 = fig.add_subplot(gs[1])

    def get_group_color(group):
        prefix_map = {
            "ADNI": "ADNI-PET",
            "ALFA": "ALFA",
            "ARWIBO": "ARWIBO",
            "CamCan": "CamCan",
            "FRONTIERS": "FRONTIERS",
            "GERO": "GERO",
            "HCP": "HCP",
            "SRPBS": "SRPBS",
        }
        prefix = group.split("-")[0]
        return dataset_colors.get(prefix_map.get(prefix, ""), "#999999")

    rng = np.random.default_rng(42)

    for i, group in enumerate(all_groups):
        group_data = comparison_df[comparison_df["group_label"] == group]
        color = get_group_color(group)

        female_data = group_data[group_data["Sex"] == "Female"]["improvement"].values
        male_data = group_data[group_data["Sex"] == "Male"]["improvement"].values

        plot_split_violin(ax1, female_data, male_data, i, color, sex_colors["Female"], sex_colors["Male"])

        for sex, data, x_range in (
            ("Female", female_data, (-0.25, -0.05)),
            ("Male", male_data, (0.05, 0.25)),
        ):
            if len(data) == 0:
                continue
            x_jitter = i + rng.uniform(*x_range, size=len(data))
            ax1.scatter(x_jitter, data, color=sex_colors[sex], alpha=0.5, s=40, edgecolors="none")

    formatted_labels = [
        f"{parts[0]}-$\\mathbf{{{parts[1]}}}$" if "-" in (parts := group.rsplit("-", 1)) else group
        for group in all_groups
    ]

    ax1.set_ylabel("$\\Delta$ GOF (Personalized - Global)", fontsize=18)
    ax1.set_xticks(range(len(all_groups)))
    ax1.set_xticklabels(formatted_labels, rotation=45, ha="right")
    ax1.set_xlim(-0.5, len(all_groups) - 0.5)
    ax1.set_ylim(comparison_df["improvement"].min() - 0.02, comparison_df["improvement"].max() + 0.02)
    ax1.tick_params(axis="both", labelsize=tick_size)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.axhline(0, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=sex_colors[sex], markersize=10, alpha=0.7, label=sex)
            for sex in ["Female", "Male"]
        ],
        loc="upper right",
        fontsize=12,
        framealpha=0.9,
    )

    figures_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(figures_dir / "global_vs_pers_gof_comparison_by_condition_split_violin.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "global_vs_pers_gof_comparison_by_condition_split_violin.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(repo_root)

    paths.figures_dir.mkdir(exist_ok=True, parents=True)
    paths.tables_dir.mkdir(exist_ok=True, parents=True)

    empirical_lookup = load_empirical_lookup(paths.concat_path)
    personalized_vectors = load_vectors_npz(paths.personalized_vectors_path)
    global_vectors = load_vectors_npz(paths.global_vectors_path)

    metadata = pd.read_csv(paths.metadata_path)
    if metadata.empty:
        raise ValueError("metadata.csv is empty. Run prepare_minimal_gof_inputs.py first.")

    missing_empirical = [n for n in metadata["N_MEGA"].astype(str) if str(n) not in empirical_lookup]
    if missing_empirical:
        raise KeyError(f"Missing empirical vectors for {len(missing_empirical)} subjects.")

    missing_personal = [n for n in metadata["N_MEGA"].astype(str) if str(n) not in personalized_vectors]
    if missing_personal:
        raise KeyError(f"Missing personalized vectors for {len(missing_personal)} subjects.")

    missing_global = [g for g in metadata["group_id"].unique() if str(g) not in global_vectors]
    if missing_global:
        raise KeyError(f"Missing global vectors for {len(missing_global)} groups.")

    empirical_matrix = np.stack([empirical_lookup[str(nmega)] for nmega in metadata["N_MEGA"]])
    personalized_matrix = np.stack([personalized_vectors[str(nmega)] for nmega in metadata["N_MEGA"]])
    global_matrix = np.stack([global_vectors[str(group_id)] for group_id in metadata["group_id"]])

    if empirical_matrix.shape[1] != personalized_matrix.shape[1] or empirical_matrix.shape[1] != global_matrix.shape[1]:
        raise ValueError("Vector length mismatch among empirical, personalized, and global vectors.")

    personalized_corr = rowwise_correlation(empirical_matrix, personalized_matrix)
    global_corr = rowwise_correlation(empirical_matrix, global_matrix)

    comparison_df = metadata.assign(
        gof_empirical_vs_personalized=personalized_corr,
        gof_empirical_vs_global=global_corr,
    )
    comparison_df["improvement"] = (
        comparison_df["gof_empirical_vs_personalized"] - comparison_df["gof_empirical_vs_global"]
    )

    comparison_df["group_label"] = comparison_df.apply(
        lambda row: f"{row['Dataset']}-{row['Diagnosis']}" + (f" {row['Age_Range']}" if row["Diagnosis"] == "CN" else ""),
        axis=1,
    )

    all_groups = sorted(comparison_df["group_label"].unique())

    plot_last_figure(comparison_df, all_groups, paths.figures_dir)

    results_df = build_results_table(comparison_df)
    results_df.to_csv(paths.tables_dir / "global_vs_pers_gof_comparison_stats.csv", index=False)

    no_improve_subjects = comparison_df[comparison_df["improvement"] < 0][
        [
            "N_MEGA",
            "Dataset",
            "Diagnosis",
            "Sex",
            "Age_Range",
            "gof_empirical_vs_global",
            "gof_empirical_vs_personalized",
            "improvement",
        ]
    ]
    no_improve_subjects.to_csv(paths.tables_dir / "subjects_no_improvement.csv", index=False)

    avg_delta = no_improve_subjects["improvement"].mean()
    ci_low, ci_high = get_bootstrap_ci(no_improve_subjects["improvement"].values, np.mean, bounds=(-1, 1))
    summary_df = pd.DataFrame([
        {
            "Group": "No improvement (delta < 0)",
            "N": len(no_improve_subjects),
            "Mean_Delta": avg_delta,
            "CI_Low": ci_low,
            "CI_High": ci_high,
        }
    ])
    summary_df.to_csv(paths.tables_dir / "no_improvement_summary.csv", index=False)

    print("Saved figure to", paths.figures_dir)
    print("Saved tables to", paths.tables_dir)


if __name__ == "__main__":
    main()
