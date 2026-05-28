"""Global vs personalized OMAT goodness-of-fit comparison for raw_SC.

This is a raw_SC ablation of the main group-vs-personalized analysis.

Main differences versus the original analysis:
- personalized GOF values are read from data/derived/rawSC_extracted_features_merged.csv
- group GOF values are computed from data/derived/global_vectors_rawSC.npz
- all output files carry a raw_SC suffix
- an extra scatter figure compares subjectwise delta GOF for raw_SC vs SC-optimized models
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import bootstrap, linregress, rankdata, shapiro, spearmanr, ttest_rel, wilcoxon


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
    results_dir: Path
    concat_path: Path
    metadata_path: Path
    rawsc_features_path: Path
    rawsc_global_vectors_path: Path
    optimized_personal_vectors_path: Path
    optimized_global_vectors_path: Path


def resolve_paths(repo_root: Path) -> Paths:
    data_dir = repo_root / "data" / "derived"
    return Paths(
        repo_root=repo_root,
        data_dir=data_dir,
        figures_dir=repo_root / "analysis" / "figures",
        tables_dir=repo_root / "analysis" / "tables",
        results_dir=repo_root / "analysis" / "results",
        concat_path=data_dir / "Omats_FCs" / "concat.npy",
        metadata_path=data_dir / "metadata.csv",
        rawsc_features_path=data_dir / "rawSC_extracted_features_merged.csv",
        rawsc_global_vectors_path=data_dir / "global_vectors_rawSC.npz",
        optimized_personal_vectors_path=data_dir / "personalized_vectors.npz",
        optimized_global_vectors_path=data_dir / "global_vectors.npz",
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


def get_bootstrap_ci(data: np.ndarray, statistic_func, n_resamples: int = 1000, confidence_level: float = 0.95, bounds=None):
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
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
    except Exception as exc:
        print(f"Bootstrap failed: {exc}")
        return np.nan, np.nan


def cohens_d_func(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)


def rank_biserial_func(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
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


def dataset_group_label(row: pd.Series) -> str:
    return f"{row['Dataset']}-{row['Diagnosis']}" + (f" {row['Age_Range']}" if row["Diagnosis"] == "CN" else "")


def build_rawsc_comparison(
    metadata: pd.DataFrame,
    empirical_matrix: np.ndarray,
    rawsc_scores: pd.DataFrame,
    rawsc_global_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    if "N_MEGA" not in rawsc_scores.columns or "gof_corr" not in rawsc_scores.columns:
        raise KeyError("rawSC_extracted_features_merged.csv must contain N_MEGA and gof_corr columns.")

    rawsc_scores = rawsc_scores[["N_MEGA", "gof_corr", "opt_it"]].rename(columns={"gof_corr": "gof_corr_rawSC"})
    df = metadata.merge(rawsc_scores, on="N_MEGA", how="left", validate="one_to_one")
    personal = pd.to_numeric(df["gof_corr_rawSC"], errors="coerce").to_numpy(dtype=np.float64)
    global_group_ids = df["group_id"].astype(str).tolist()
    missing_global = [g for g in sorted(set(global_group_ids)) if str(g) not in rawsc_global_vectors]
    if missing_global:
        raise KeyError(f"Missing rawSC global vectors for {len(missing_global)} groups: {missing_global[:10]}")
    global_matrix = np.stack([rawsc_global_vectors[str(group_id)] for group_id in global_group_ids])
    global_gof = rowwise_correlation(empirical_matrix, global_matrix)

    df = df.assign(
        gof_empirical_vs_personalized_rawSC=personal,
        gof_empirical_vs_global_rawSC=global_gof,
    )
    df["improvement_rawSC"] = df["gof_empirical_vs_personalized_rawSC"] - df["gof_empirical_vs_global_rawSC"]
    df["group_label"] = df.apply(dataset_group_label, axis=1)
    return df


def build_optimized_comparison(
    metadata: pd.DataFrame,
    empirical_matrix: np.ndarray,
    personalized_vectors: dict[str, np.ndarray],
    global_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    missing_personal = [n for n in metadata["N_MEGA"].astype(str) if str(n) not in personalized_vectors]
    if missing_personal:
        raise KeyError(f"Missing personalized vectors for {len(missing_personal)} subjects.")
    missing_global = [g for g in sorted(set(metadata["group_id"].astype(str))) if str(g) not in global_vectors]
    if missing_global:
        raise KeyError(f"Missing global vectors for {len(missing_global)} groups.")

    personalized_matrix = np.stack([personalized_vectors[str(nmega)] for nmega in metadata["N_MEGA"]])
    global_matrix = np.stack([global_vectors[str(group_id)] for group_id in metadata["group_id"]])
    personalized_gof = rowwise_correlation(empirical_matrix, personalized_matrix)
    global_gof = rowwise_correlation(empirical_matrix, global_matrix)

    df = metadata.assign(
        gof_empirical_vs_personalized=personalized_gof,
        gof_empirical_vs_global=global_gof,
    )
    df["improvement"] = df["gof_empirical_vs_personalized"] - df["gof_empirical_vs_global"]
    df["group_label"] = df.apply(dataset_group_label, axis=1)
    return df


def clean_pairs(df: pd.DataFrame, global_col: str, personal_col: str) -> pd.DataFrame:
    return df.loc[np.isfinite(df[global_col]) & np.isfinite(df[personal_col])].copy()


def analyze_group(group_name: str, df_group: pd.DataFrame, global_col: str, personal_col: str) -> dict[str, float | str]:
    subset = clean_pairs(df_group, global_col, personal_col)
    global_gof = subset[global_col].to_numpy(dtype=np.float64)
    pers_gof = subset[personal_col].to_numpy(dtype=np.float64)
    diff = pers_gof - global_gof
    n = len(diff)

    desc_stats: dict[str, float | str] = {
        "Dataset": group_name,
        "N": n,
        "Global_Mean": np.nan,
        "Global_Median": np.nan,
        "Global_SD": np.nan,
        "Pers_Mean": np.nan,
        "Pers_Median": np.nan,
        "Pers_SD": np.nan,
        "Delta_Mean": np.nan,
        "Delta_Median": np.nan,
        "Delta_SD": np.nan,
    }

    if n == 0:
        desc_stats.update(
            {
                "Global_CI_Low": np.nan,
                "Global_CI_High": np.nan,
                "Pers_CI_Low": np.nan,
                "Pers_CI_High": np.nan,
                "Delta_CI_Low": np.nan,
                "Delta_CI_High": np.nan,
                "Shapiro_p": np.nan,
                "Test_Type": "NA",
                "Statistic_Name": "NA",
                "Statistic_Value": np.nan,
                "DF": np.nan,
                "p_value": np.nan,
                "Effect_Size_Type": "NA",
                "Effect_Size": np.nan,
                "ES_CI_Low": np.nan,
                "ES_CI_High": np.nan,
            }
        )
        return desc_stats

    desc_stats.update(
        {
            "Global_Mean": float(np.mean(global_gof)),
            "Global_Median": float(np.median(global_gof)),
            "Global_SD": float(np.std(global_gof, ddof=1)) if n > 1 else np.nan,
            "Pers_Mean": float(np.mean(pers_gof)),
            "Pers_Median": float(np.median(pers_gof)),
            "Pers_SD": float(np.std(pers_gof, ddof=1)) if n > 1 else np.nan,
            "Delta_Mean": float(np.mean(diff)),
            "Delta_Median": float(np.median(diff)),
            "Delta_SD": float(np.std(diff, ddof=1)) if n > 1 else np.nan,
        }
    )

    desc_stats["Global_CI_Low"], desc_stats["Global_CI_High"] = get_bootstrap_ci(global_gof, np.mean, bounds=(0, 1))
    desc_stats["Pers_CI_Low"], desc_stats["Pers_CI_High"] = get_bootstrap_ci(pers_gof, np.mean, bounds=(0, 1))
    desc_stats["Delta_CI_Low"], desc_stats["Delta_CI_High"] = get_bootstrap_ci(diff, np.mean, bounds=(-1, 1))

    if n >= 3:
        _, shapiro_p = shapiro(diff)
    else:
        shapiro_p = 1.0

    desc_stats["Shapiro_p"] = float(shapiro_p)

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

    desc_stats.update(
        {
            "Test_Type": test_type,
            "Statistic_Name": statistic_name,
            "Statistic_Value": float(stat) if np.isfinite(stat) else np.nan,
            "DF": df,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
            "Effect_Size_Type": eff_size_type,
            "Effect_Size": eff_size,
            "ES_CI_Low": es_ci_low,
            "ES_CI_High": es_ci_high,
        }
    )
    return desc_stats


def build_results_table(comparison_df: pd.DataFrame, global_col: str, personal_col: str) -> pd.DataFrame:
    results_list = [analyze_group("Whole Dataset", comparison_df, global_col, personal_col)]
    for dataset_name, group_df in comparison_df.groupby("Dataset"):
        results_list.append(analyze_group(dataset_name, group_df, global_col, personal_col))

    results_df = pd.DataFrame(results_list)
    cols = [
        "Dataset",
        "N",
        "Shapiro_p",
        "Test_Type",
        "Statistic_Name",
        "Statistic_Value",
        "DF",
        "p_value",
        "Effect_Size_Type",
        "Effect_Size",
        "ES_CI_Low",
        "ES_CI_High",
        "Global_Mean",
        "Global_SD",
        "Global_CI_Low",
        "Global_CI_High",
        "Pers_Mean",
        "Pers_SD",
        "Pers_CI_Low",
        "Pers_CI_High",
        "Delta_Mean",
        "Delta_SD",
        "Delta_CI_Low",
        "Delta_CI_High",
    ]
    return results_df[cols]


def plot_group_violin(comparison_df: pd.DataFrame, global_col: str, personal_col: str, delta_col: str, title_suffix: str, output_prefix: str, figures_dir: Path) -> None:
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
    clean_df = clean_pairs(comparison_df, global_col, personal_col)

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
    datasets = sorted(clean_df["Dataset"].dropna().unique())
    for i, dataset in enumerate(datasets):
        dataset_data = clean_df[clean_df["Dataset"] == dataset]
        color = dataset_colors.get(dataset, "#999999")
        plot_split_violin(
            ax0,
            dataset_data[global_col].values,
            dataset_data[personal_col].values,
            i,
            color,
            "#d3d3d3",
            "#000000",
        )

    ax0.set_ylabel("Goodness of Fit (GOF)", fontsize=18)
    ax0.set_xticks(range(len(datasets)))
    ax0.set_xticklabels(datasets, rotation=45, ha="right")
    ax0.set_xlim(-0.5, len(datasets) - 0.5)
    ax0.set_ylim(0.0, 0.9)
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
    ax0.set_title(f"GOF distribution by dataset {title_suffix}", fontsize=16)

    ax1 = fig.add_subplot(gs[1])
    all_groups = sorted(clean_df["group_label"].dropna().unique())

    rng = np.random.default_rng(42)
    for i, group in enumerate(all_groups):
        group_data = clean_df[clean_df["group_label"] == group]
        color = dataset_colors.get(group.split("-")[0], "#999999")
        female_data = group_data[group_data["Sex"] == "Female"][delta_col].values
        male_data = group_data[group_data["Sex"] == "Male"][delta_col].values

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
    ax1.set_ylim(clean_df[delta_col].min() - 0.02, clean_df[delta_col].max() + 0.02)
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
    ax1.set_title(f"Subjectwise improvement by group {title_suffix}", fontsize=16)

    figures_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(figures_dir / f"{output_prefix}_split_violin_raw_SC.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{output_prefix}_split_violin_raw_SC.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta_scatter(delta_df: pd.DataFrame, figures_dir: Path) -> None:
    diag_markers = {"CN": "o", "MCI": "s", "AD": "^", "FTD": "D"}
    diag_colors = {"CN": "#1f77b4", "MCI": "#ff7f0e", "AD": "#d62728", "FTD": "#2ca02c"}
    clean = delta_df.loc[
        np.isfinite(delta_df["delta_sc_optimized"]) & np.isfinite(delta_df["delta_rawSC"])
    ].copy()

    x = clean["delta_sc_optimized"].to_numpy(dtype=np.float64)
    y = clean["delta_rawSC"].to_numpy(dtype=np.float64)
    rho, p_val = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    for diagnosis, group_df in clean.groupby("Diagnosis"):
        ax.scatter(
            group_df["delta_sc_optimized"],
            group_df["delta_rawSC"],
            label=diagnosis,
            s=28,
            alpha=0.65,
            marker=diag_markers.get(diagnosis, "o"),
            color=diag_colors.get(diagnosis, "#999999"),
            edgecolors="none",
        )

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5, alpha=0.8)
    if len(clean) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, intercept + slope * xs, color="#444444", linewidth=2.0, alpha=0.8)

    ax.set_xlabel("Delta GOF SC optimized")
    ax.set_ylabel("Delta GOF raw_SC")
    ax.set_title("Subjectwise delta GOF: raw_SC vs SC optimized")
    ax.text(
        0.04,
        0.96,
        f"N = {len(clean)}\nSpearman $\\rho$ = {rho:.3f}\np = {p_val:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, title="Diagnosis", loc="lower right")
    figures_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(figures_dir / "rawSC_delta_vs_sc_optimized_scatter_raw_SC.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "rawSC_delta_vs_sc_optimized_scatter_raw_SC.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(repo_root)

    paths.figures_dir.mkdir(exist_ok=True, parents=True)
    paths.tables_dir.mkdir(exist_ok=True, parents=True)
    paths.results_dir.mkdir(exist_ok=True, parents=True)

    metadata = pd.read_csv(paths.metadata_path)
    if metadata.empty:
        raise ValueError("metadata.csv is empty.")

    empirical_lookup = load_empirical_lookup(paths.concat_path)
    rawsc_global_vectors = load_vectors_npz(paths.rawsc_global_vectors_path)
    rawsc_scores = pd.read_csv(paths.rawsc_features_path, sep="\t", usecols=["N_MEGA", "gof_corr", "opt_it"])

    missing_empirical = [n for n in metadata["N_MEGA"].astype(str) if str(n) not in empirical_lookup]
    if missing_empirical:
        raise KeyError(f"Missing empirical vectors for {len(missing_empirical)} subjects.")

    empirical_matrix = np.stack([empirical_lookup[str(nmega)] for nmega in metadata["N_MEGA"]])
    rawsc_comparison = build_rawsc_comparison(metadata, empirical_matrix, rawsc_scores, rawsc_global_vectors)

    rawsc_group_col = "gof_empirical_vs_global_rawSC"
    rawsc_personal_col = "gof_empirical_vs_personalized_rawSC"
    rawsc_delta_col = "improvement_rawSC"

    plot_group_violin(
        rawsc_comparison,
        rawsc_group_col,
        rawsc_personal_col,
        rawsc_delta_col,
        "(raw_SC)",
        "global_vs_pers_gof_comparison_by_condition",
        paths.figures_dir,
    )

    results_df = build_results_table(rawsc_comparison, rawsc_group_col, rawsc_personal_col)
    results_df.to_csv(paths.tables_dir / "global_vs_pers_gof_comparison_stats_raw_SC.csv", index=False)

    no_improve_subjects = rawsc_comparison[rawsc_comparison[rawsc_delta_col] < 0][
        [
            "N_MEGA",
            "Dataset",
            "Diagnosis",
            "Sex",
            "Age_Range",
            rawsc_group_col,
            rawsc_personal_col,
            rawsc_delta_col,
        ]
    ].copy()
    no_improve_subjects.to_csv(paths.tables_dir / "subjects_no_improvement_raw_SC.csv", index=False)

    avg_delta = no_improve_subjects[rawsc_delta_col].mean()
    ci_low, ci_high = get_bootstrap_ci(no_improve_subjects[rawsc_delta_col].values, np.mean, bounds=(-1, 1))
    summary_df = pd.DataFrame(
        [
            {
                "Group": "No improvement (delta < 0)",
                "N": len(no_improve_subjects),
                "Mean_Delta": avg_delta,
                "CI_Low": ci_low,
                "CI_High": ci_high,
            }
        ]
    )
    summary_df.to_csv(paths.tables_dir / "no_improvement_summary_raw_SC.csv", index=False)

    # Delta comparison versus the SC-optimized analysis.
    optimized_personalized_vectors = load_vectors_npz(paths.optimized_personal_vectors_path)
    optimized_global_vectors = load_vectors_npz(paths.optimized_global_vectors_path)
    optimized_comparison = build_optimized_comparison(metadata, empirical_matrix, optimized_personalized_vectors, optimized_global_vectors)

    delta_compare = rawsc_comparison[[
        "N_MEGA",
        "Dataset",
        "Diagnosis",
        "Sex",
        "Age_Range",
        "group_id",
        "improvement_rawSC",
    ]].merge(
        optimized_comparison[[
            "N_MEGA",
            "improvement",
        ]].rename(columns={"improvement": "improvement_sc_optimized"}),
        on="N_MEGA",
        how="inner",
        validate="one_to_one",
    )
    delta_compare["delta_rawSC"] = delta_compare["improvement_rawSC"]
    delta_compare["delta_sc_optimized"] = delta_compare["improvement_sc_optimized"]
    delta_compare["delta_rawSC_minus_sc_optimized"] = delta_compare["delta_rawSC"] - delta_compare["delta_sc_optimized"]
    delta_compare.to_csv(paths.results_dir / "rawSC_vs_sc_optimized_delta_subjectwise_raw_SC.csv", index=False)

    delta_summary = build_results_table(
        delta_compare.rename(
            columns={
                "delta_sc_optimized": "global_model",
                "delta_rawSC": "personalized_model",
            }
        ),
        "global_model",
        "personalized_model",
    )
    delta_summary.to_csv(paths.results_dir / "rawSC_vs_sc_optimized_delta_summary_raw_SC.csv", index=False)

    plot_delta_scatter(delta_compare, paths.figures_dir)

    print("Saved raw_SC comparison figure to", paths.figures_dir)
    print("Saved raw_SC tables to", paths.tables_dir)
    print("Saved delta comparison outputs to", paths.results_dir)


if __name__ == "__main__":
    main()
