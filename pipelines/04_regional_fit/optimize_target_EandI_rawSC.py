"""Optimize DMF targets for the raw-SC analysis.

Self-contained Omat-only version. One subject per run, selected by the
1-based row index in metadata.csv.
"""
from __future__ import annotations

import argparse
import gc
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import time as tm
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress

import DMF_ISP_numba_EandI as DMF
import BOLDModel as BD
import calculate_omat

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
METADATA_PATH = REPO_ROOT / "data" / "derived" / "metadata.csv"
OPTIMALS_PATH = SCRIPT_DIR.parent / "03_global_fit" / "optimals_BOLD_corr_rawSC.csv"
RAW_SC_PATH = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/raw_SC.npz")
EMP_OMATS_PATH = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/input/all_Omats.npz")
OUTPUT_BASE = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/optimize_target_from_omat_epsilon0.04/rawSC")

NNODES = 90
EPSILON = 0.04
NSEEDS = 20
ITERS = 200
PATIENCE = 3

CUT_FROM = 12_000
RATE_DOWNSAMPLE = 10
BOLD_DOWNSAMPLE = 100
CONVERGED_TIME = int(100 / 0.001 / RATE_DOWNSAMPLE)

DMF.nnodes = NNODES
DMF.tmax = 720_000
DMF.dt = 1
DMF.sigma = 0.01
DMF.tau_p = 1.5
DMF.Jdecay = 400_000
DMF.model_1 = 1
DMF.model_2 = 1

TRIU_INDICES = np.triu_indices(NNODES, k=1)
BESSEL_A, BESSEL_B = signal.bessel(3, 2 * 1 * np.array([0.01, 0.1]), btype="bandpass")


@dataclass(frozen=True)
class SubjectParams:
    row_index: int
    nmega: str
    g_value: float
    target_value: float
    diagnosis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize DMF targets for a single raw-SC subject")
    parser.add_argument("row_index", type=int, help="1-based row index in metadata.csv")
    parser.add_argument("--max-iters", type=int, default=ITERS, help=f"Maximum iterations (default {ITERS})")
    parser.add_argument("--patience", type=int, default=PATIENCE, help=f"Stop after this many non-improving iterations (default {PATIENCE})")
    return parser.parse_args()


def resolve_subject(row_index_1_based: int) -> SubjectParams:
    metadata = pd.read_csv(METADATA_PATH)
    idx = row_index_1_based - 1
    if not 0 <= idx < len(metadata):
        raise IndexError(f"Row index {row_index_1_based} is out of bounds for {len(metadata)} subjects.")

    row = metadata.iloc[idx]
    nmega = str(row["N_MEGA"])
    diagnosis = str(row["Diagnosis"]).strip().upper()

    optimals = pd.read_csv(OPTIMALS_PATH, sep="\t")
    match = optimals[optimals["N_MEGA"].astype(str) == nmega]
    if match.empty:
        raise KeyError(f"N_MEGA {nmega} not found in {OPTIMALS_PATH.name}.")

    optimal_row = match.iloc[0]
    return SubjectParams(
        row_index=idx,
        nmega=nmega,
        g_value=float(optimal_row["G_BOLD_omat"]),
        target_value=float(optimal_row["target_BOLD_omat"]),
        diagnosis=diagnosis,
    )


def load_subject_data(params: SubjectParams) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(RAW_SC_PATH) as sc_file:
        sc_key = "SC_AD" if params.diagnosis == "MCI" else f"SC_{params.diagnosis}"
        if sc_key not in sc_file:
            raise KeyError(f"SC key '{sc_key}' not found. Available: {', '.join(sc_file.files)}")
        structural_connectivity = sc_file[sc_key]

    with np.load(EMP_OMATS_PATH, allow_pickle=True) as omat_file:
        if params.nmega not in omat_file:
            sample = ", ".join(list(omat_file.keys())[:10])
            raise KeyError(f"N_MEGA {params.nmega} not found in all_Omats.npz. Sample keys: {sample}")
        empirical_omats = omat_file[params.nmega]

    return structural_connectivity, empirical_omats


def reconstruct_symm(flattened: np.ndarray, diag_value: float = 0.0) -> np.ndarray:
    out = np.zeros((NNODES, NNODES), dtype=np.float64)
    out[TRIU_INDICES] = flattened
    out = out + out.T
    np.fill_diagonal(out, diag_value)
    return out


def entropies_per_channel(rates: np.ndarray) -> np.ndarray:
    from scipy.stats import gamma as gamma_dist

    entropies = np.zeros(rates.shape[1], dtype=np.float64)
    for roi in range(rates.shape[1]):
        values = rates[:, roi]
        values = values[np.isfinite(values)]
        if values.size < 2:
            entropies[roi] = np.nan
            continue
        try:
            a, loc, scale = gamma_dist.fit(data=values, floc=0)
            entropies[roi] = gamma_dist.entropy(a, loc, scale)
        except ValueError:
            entropies[roi] = np.nan
    return entropies


def generate_matrices(target_vector: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    DMF.target = target_vector
    DMF.seed = seed

    rates_e, rates_i, _ = DMF.Sim()
    rates_e_bold = rates_e[CUT_FROM::RATE_DOWNSAMPLE, :]
    rates_i = rates_i[CUT_FROM::RATE_DOWNSAMPLE, :]

    rates_e = rates_e[-CONVERGED_TIME:, :]
    rates_i = rates_i[-CONVERGED_TIME:, :]

    mean_rates_e = rates_e.mean(axis=0)
    mean_rates_i = rates_i.mean(axis=0)
    entropies_e = entropies_per_channel(rates_e)
    entropies_i = entropies_per_channel(rates_i)

    bold = BD.Sim(rates_e_bold, NNODES, 0.01)
    bold = bold[CUT_FROM:, :][::BOLD_DOWNSAMPLE, :]
    bold = signal.filtfilt(BESSEL_A, BESSEL_B, bold, axis=0).astype(np.float32)

    return calculate_omat.multi_fc(bold), mean_rates_e, mean_rates_i, entropies_e, entropies_i, bold


def safe_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return None
    x_fit = x[finite]
    y_fit = y[finite]
    if np.unique(x_fit).size < 2:
        return None
    slope, intercept, r, _, _ = linregress(x_fit, y_fit)
    return float(slope), float(intercept), float(r)


def ensure_output_structure(output_root: Path) -> Dict[str, Path]:
    folder = output_root
    plots = folder / "plots"
    bold = folder / "BOLD_per_seed"
    output = folder / "output_per_nmega"
    for path in (plots, bold, output):
        path.mkdir(parents=True, exist_ok=True)

    master = folder / "master_file_rawSC.csv"
    if not master.exists():
        roi_cols = "\t".join(f"ROI_{n}" for n in range(NNODES))
        master.write_text(
            "N_MEGA\tG_base\ttarget_base\ttime_taken\tmean_entropy_E\tmean_entropy_I\tmean_omat_emp\t"
            + roi_cols + "\n"
        )
    return {"plots": plots, "bold": bold, "output": output, "master": master}


def plot_results(plot_path: Path, params: SubjectParams, corrs: np.ndarray, all_targets: np.ndarray,
                 all_mean_rates_e: np.ndarray, all_mean_rates_i: np.ndarray,
                 all_entropies_e: np.ndarray, all_entropies_i: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    opt_it = int(np.argmax(corrs))
    x = all_targets[:, opt_it]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Target optimisation according to OMAT node strength\nN_MEGA = {params.nmega}", fontweight="bold")

    axes[0, 0].set_title("GoF")
    axes[0, 0].plot(corrs, label="corr to omat")
    axes[0, 0].axvline(opt_it, color="red", linestyle="--")
    axes[0, 0].legend()

    for ax, y, title, ylabel in [
        (axes[0, 1], all_mean_rates_e[:, opt_it], "Targets vs mean rate (E)", "Mean rate (E)"),
        (axes[1, 0], all_mean_rates_i[:, opt_it], "Targets vs mean rate (I)", "Mean rate (I)"),
    ]:
        ax.set_title(title)
        ax.scatter(x, y, s=10)
        fit = safe_linear_fit(x, y)
        if fit is not None:
            slope, intercept, r = fit
            ax.plot(x, intercept + slope * x, label=f"r = {r:.4f}")
        ax.set_xlabel("Targets at optimal iteration")
        ax.set_ylabel(ylabel)
        if fit is not None:
            ax.legend()

    axes[1, 1].set_title("Targets vs entropy")
    fit_e = safe_linear_fit(x, all_entropies_e[:, opt_it])
    fit_i = safe_linear_fit(x, all_entropies_i[:, opt_it])
    axes[1, 1].scatter(x, all_entropies_e[:, opt_it], s=10, label=f"E, r = {fit_e[2]:.4f}" if fit_e is not None else "E, no fit")
    axes[1, 1].scatter(x, all_entropies_i[:, opt_it], s=10, label=f"I, r = {fit_i[2]:.4f}" if fit_i is not None else "I, no fit")
    if fit_e is not None:
        slope_e, intercept_e, r_e = fit_e
        axes[1, 1].plot(x, intercept_e + slope_e * x, color="C0")
    if fit_i is not None:
        slope_i, intercept_i, r_i = fit_i
        axes[1, 1].plot(x, intercept_i + slope_i * x, color="C1")
    axes[1, 1].set_xlabel("Targets at optimal iteration")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(plot_path / f"{params.nmega}")
    plt.close(fig)


def write_master_line(master_path: Path, params: SubjectParams, base_target: float, elapsed: float,
                      mean_entropy_e: float, mean_entropy_i: float, mean_omat_emp: float,
                      optimal_targets: np.ndarray) -> None:
    with master_path.open("a") as handle:
        handle.write(
            f"{params.nmega}\t{params.g_value}\t{base_target}\t{elapsed}\t{mean_entropy_e}\t"
            f"{mean_entropy_i}\t{mean_omat_emp}\t" + "\t".join(map(str, optimal_targets)) + "\n"
        )


def optimise_subject(params: SubjectParams, max_iters: int, patience: int) -> None:
    structural_connectivity, empirical_omats_flat = load_subject_data(params)
    output = ensure_output_structure(OUTPUT_BASE)

    empirical_omat = reconstruct_symm(empirical_omats_flat, diag_value=0.0)
    empirical_strengths = empirical_omat.sum(axis=1)
    mean_omat_emp = empirical_omat.mean()

    DMF.G = params.g_value
    DMF.SC = 0.2 * structural_connectivity
    target_vector = np.full(NNODES, params.target_value, dtype=np.float64)

    all_targets = np.zeros((NNODES, max_iters), dtype=np.float64)
    all_mean_rates_e = np.zeros((NNODES, max_iters), dtype=np.float64)
    all_mean_rates_i = np.zeros((NNODES, max_iters), dtype=np.float64)
    all_entropies_e = np.zeros((NNODES, max_iters), dtype=np.float64)
    all_entropies_i = np.zeros((NNODES, max_iters), dtype=np.float64)
    corrs = np.zeros(max_iters, dtype=np.float64)

    best_corr = -np.inf
    no_improve = 0
    bold_output = output["bold"] / f"BOLD_optimal_nmega{params.nmega}.npz"
    if bold_output.exists():
        with np.load(bold_output, allow_pickle=True) as checkpoint:
            if "target_vector" in checkpoint:
                target_vector = np.asarray(checkpoint["target_vector"], dtype=np.float64)
                print(f"N_MEGA={params.nmega} resuming from checkpoint", flush=True)

    start = tm()
    for iteration in range(max_iters):
        all_targets[:, iteration] = target_vector
        sim_omats = []
        agg_mean_e = np.zeros(NNODES, dtype=np.float64)
        agg_mean_i = np.zeros(NNODES, dtype=np.float64)
        agg_entropy_e = np.zeros(NNODES, dtype=np.float64)
        agg_entropy_i = np.zeros(NNODES, dtype=np.float64)
        bold_to_save: Dict[str, np.ndarray] = {"it": iteration}
        iter_start = tm()

        for seed in range(NSEEDS):
            sim_omat, mean_e, mean_i, ent_e, ent_i, bold = generate_matrices(target_vector, seed)
            sim_omats.append(np.asarray(sim_omat, dtype=np.float64))
            agg_mean_e += mean_e
            agg_mean_i += mean_i
            agg_entropy_e += ent_e
            agg_entropy_i += ent_i
            bold_to_save[f"seed{seed}_array"] = bold
            print(f"N_MEGA={params.nmega} iter={iteration + 1}/{max_iters} seed={seed} done", flush=True)

        agg_omat = np.nanmean(np.stack(sim_omats, axis=0), axis=0)
        agg_mean_e /= NSEEDS
        agg_mean_i /= NSEEDS
        agg_entropy_e /= NSEEDS
        agg_entropy_i /= NSEEDS

        corr = np.corrcoef(agg_omat[TRIU_INDICES], empirical_omat[TRIU_INDICES])[0, 1]
        corrs[iteration] = corr
        print(
            f"N_MEGA={params.nmega} iter={iteration + 1}/{max_iters} corr={corr:.6f} seeds_time={tm() - iter_start:.2f}s",
            flush=True,
        )
        all_mean_rates_e[:, iteration] = agg_mean_e
        all_mean_rates_i[:, iteration] = agg_mean_i
        all_entropies_e[:, iteration] = agg_entropy_e
        all_entropies_i[:, iteration] = agg_entropy_i

        if corr > best_corr:
            np.savez_compressed(bold_output, target_vector=target_vector, **bold_to_save)
            best_corr = corr
            no_improve = 0
        else:
            no_improve += 1

        target_vector += EPSILON * (empirical_strengths - agg_omat.sum(axis=1))
        np.maximum(target_vector, 0, out=target_vector)
        if no_improve >= patience:
            break

    elapsed = tm() - start
    n_done = iteration + 1
    all_targets = all_targets[:, :n_done]
    all_mean_rates_e = all_mean_rates_e[:, :n_done]
    all_mean_rates_i = all_mean_rates_i[:, :n_done]
    all_entropies_e = all_entropies_e[:, :n_done]
    all_entropies_i = all_entropies_i[:, :n_done]
    corrs = corrs[:n_done]

    opt_it = int(np.argmax(corrs))
    optimal_targets = all_targets[:, opt_it]

    np.savez_compressed(
        output["output"] / f"nmega{params.nmega}_output.npz",
        all_targets=all_targets,
        optimal_targets=optimal_targets,
        all_mean_rates_E=all_mean_rates_e,
        all_entropies_E=all_entropies_e,
        all_mean_rates_I=all_mean_rates_i,
        all_entropies_I=all_entropies_i,
        corrs=corrs,
    )
    plot_results(output["plots"], params, corrs, all_targets, all_mean_rates_e, all_mean_rates_i, all_entropies_e, all_entropies_i)
    write_master_line(output["master"], params, params.target_value, elapsed,
                      all_entropies_e[:, opt_it].mean(), all_entropies_i[:, opt_it].mean(),
                      mean_omat_emp, optimal_targets)
    gc.collect()


def main() -> None:
    args = parse_args()
    params = resolve_subject(args.row_index)
    optimise_subject(params, max_iters=args.max_iters, patience=args.patience)


if __name__ == "__main__":
    main()
