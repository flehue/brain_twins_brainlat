"""Optimize DMF target parameters using unmatched empirical data.

This script mirrors the behaviour of `optimize_target_EandI.py` while
reading inputs from the unmatched datasets and processing a single
subject per execution. The only runtime argument is the (1-based) row
index in `optimals_BOLD_corr_unmatched.csv` that selects the subject.
"""
from __future__ import annotations

import argparse
import gc
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import time as tm
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress
from skimage.metrics import structural_similarity as ssim

import DMF_ISP_numba_EandI as DMF
import BOLDModel as BD
import calculate_omat

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OPTIMALS_PATH = PROJECT_ROOT / "optimals" / "optimals_BOLD_corr_unmatched.csv"
EMP_OMATS_PATH = PROJECT_ROOT / "unmatched_omats.npz"
EMP_FCS_PATH = PROJECT_ROOT / "unmatched_FCs.npz"
SC_PATH = SCRIPT_DIR / "input" / "optimized_SC.npz"
OUTPUT_BASE = PROJECT_ROOT / "output"

NNODES = 90
DMF.nnodes = NNODES

DMF.tmax = 720_000  # ms
DMF.dt = 1
DMF.sigma = 0.01
DMF.tau_p = 1.5
DMF.Jdecay = 400_000
DMF.model_1 = 1
DMF.model_2 = 1

CUT_FROM = 12_000
RATE_DOWNSAMPLE = 10
BOLD_DOWNSAMPLE = 100
CONVERGED_TIME = int(100 / 0.001 / RATE_DOWNSAMPLE)
EPSILON = 0.04
NSEEDS = 20
ITERS = 100

TRIU_INDICES = np.triu_indices(NNODES, k=1)
BESSEL_A, BESSEL_B = signal.bessel(3, 2 * 1 * np.array([0.01, 0.1]), btype="bandpass")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data containers and helpers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SubjectParams:
    row_index: int
    nmega: str
    g_value: float
    target_value: float
    diagnosis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize DMF targets for a single unmatched subject",
    )
    parser.add_argument(
        "row_index",
        type=int,
        help="1-based row index in optimals_BOLD_corr_unmatched.csv",
    )
    return parser.parse_args()


def resolve_subject(row_index_1_based: int) -> SubjectParams:
    dataframe = pd.read_csv(OPTIMALS_PATH)
    if dataframe.empty:
        raise ValueError("The optimals table is empty.")

    zero_based_index = row_index_1_based - 1
    if not 0 <= zero_based_index < len(dataframe):
        raise IndexError(
            f"Row index {row_index_1_based} is out of bounds for the table of size {len(dataframe)}.")

    row = dataframe.iloc[zero_based_index]

    diagnosis_column = "Diagnosis" if "Diagnosis" in row else "diagnosis"
    diagnosis = str(row[diagnosis_column]).strip().upper()

    return SubjectParams(
        row_index=zero_based_index,
        nmega=str(row["N_MEGA"]),
        g_value=float(row["G_BOLD_omat"]),
        target_value=float(row["target_BOLD_omat"]),
        diagnosis=diagnosis,
    )


def load_subject_data(params: SubjectParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SC_PATH) as sc_file:
        if params.diagnosis == "MCI":
            sc_key = "SC_AD"
        else:
            sc_key = f"SC_{params.diagnosis}"
        if sc_key not in sc_file:
            available = ", ".join(sc_file.files)
            raise KeyError(f"SC key '{sc_key}' not found. Available keys: {available}")
        structural_connectivity = sc_file[sc_key]

    with np.load(EMP_OMATS_PATH, allow_pickle=True) as omat_file:
        if params.nmega not in omat_file:
            available = ", ".join(list(omat_file.keys())[:10]) + ("..." if len(omat_file.keys()) > 10 else "")
            raise KeyError(
                f"N_MEGA {params.nmega} not found in unmatched_omats.npz. Sample keys: {available}")
        empirical_omats = omat_file[params.nmega]

    with np.load(EMP_FCS_PATH, allow_pickle=True) as fc_file:
        if params.nmega not in fc_file:
            available = ", ".join(list(fc_file.keys())[:10]) + ("..." if len(fc_file.keys()) > 10 else "")
            raise KeyError(
                f"N_MEGA {params.nmega} not found in unmatched_FCs.npz. Sample keys: {available}")
        empirical_fcs = fc_file[params.nmega]

    return structural_connectivity, empirical_omats, empirical_fcs


def reconstruct_symm(flattened: np.ndarray, N: int = NNODES, k: int = 1, diag_value: float = 1.0) -> np.ndarray:
    triu_idx = np.triu_indices(N, k=k)
    matrix = np.zeros((N, N), dtype=flattened.dtype)
    matrix[triu_idx] = flattened
    matrix = matrix + matrix.T
    if k == 1:
        np.fill_diagonal(matrix, diag_value)
    return matrix


def get_range(array1: np.ndarray, array2: np.ndarray) -> float:
    return max(array1.max(), array2.max()) - min(array1.min(), array2.min())


def entropies_per_channel(rates: np.ndarray, t_as_col: bool = False) -> np.ndarray:
    from scipy.stats import gamma as gamma_dist

    if t_as_col:
        rates = rates.T

    entropies = np.zeros(rates.shape[1], dtype=np.float64)
    for roi in range(rates.shape[1]):
        a, loc, scale = gamma_dist.fit(data=rates[:, roi], floc=0)
        entropies[roi] = gamma_dist.entropy(a, loc, scale)
    return entropies


def generate_matrices(target_vector: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    DMF.target = target_vector
    DMF.seed = seed

    rates_e, rates_i, _ = DMF.Sim()
    rates_e = rates_e[CUT_FROM::RATE_DOWNSAMPLE, :]
    rates_i = rates_i[CUT_FROM::RATE_DOWNSAMPLE, :]

    rates_considered_e = rates_e[-CONVERGED_TIME:, :]
    rates_considered_i = rates_i[-CONVERGED_TIME:, :]

    entropies_e = entropies_per_channel(rates_considered_e)
    mean_rates_e = rates_considered_e.mean(axis=0)
    entropies_i = entropies_per_channel(rates_considered_i)
    mean_rates_i = rates_considered_i.mean(axis=0)

    bold_dt = 0.01
    bold_signals = BD.Sim(rates_e, NNODES, bold_dt)
    bold_signals = bold_signals[CUT_FROM:, :][::BOLD_DOWNSAMPLE, :]

    bold_filtered = signal.filtfilt(BESSEL_A, BESSEL_B, bold_signals, axis=0).astype(np.float32)

    simulated_fc_bold = np.corrcoef(bold_filtered.T)
    simulated_omat_bold = calculate_omat.multi_fc(bold_filtered)

    return (
        simulated_fc_bold,
        simulated_omat_bold,
        mean_rates_e,
        mean_rates_i,
        entropies_e,
        entropies_i,
        bold_filtered,
    )


def ensure_output_structure(output_root: Path) -> Dict[str, Path]:
    folder = output_root / f"optimize_target_from_omat_epsilon{EPSILON}"
    folder_plots = folder / "plots"
    folder_bold = folder / "BOLD_per_seed"
    folder_output = folder / "output_per_nmega"

    for directory in (folder_plots, folder_bold, folder_output):
        directory.mkdir(parents=True, exist_ok=True)

    masterfile_path = folder / "master_file.csv"
    if not masterfile_path.exists():
        target_cols = "\t".join([f"ROI_{n}" for n in range(NNODES)])
        header = (
            "N_MEGA\tG_base\ttarget_base\ttime_taken\tmean_entropy_E\tmean_entropy_I\t"
            "mean_FC_emp\tmean_omat_emp\t" + target_cols + "\n"
        )
        masterfile_path.write_text(header)

    return {
        "folder": folder,
        "plots": folder_plots,
        "bold": folder_bold,
        "output": folder_output,
        "master": masterfile_path,
    }


def plot_results(plot_path: Path, params: SubjectParams, corrs: np.ndarray, corrs_fc: np.ndarray,
                 ssims: np.ndarray, all_targets: np.ndarray, all_mean_rates_e: np.ndarray,
                 all_mean_rates_i: np.ndarray, all_entropies_e: np.ndarray,
                 all_entropies_i: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Target optimisation according to OMAT node strength\nN_MEGA = {params.nmega}",
        fontweight="bold",
    )
    axes = axes.flatten()

    axes[0].set_title("Goodness of fit")
    axes[0].plot(corrs, label="corr to omat")
    axes[0].plot(corrs_fc, label="corr to FC")
    axes[0].plot(ssims, label="ssim to omat")
    optimal_iteration = int(np.argmax(corrs))
    axes[0].axvline(optimal_iteration, color="red", linestyle="--")
    axes[0].legend()

    axes[1].set_title("Targets vs mean rate (E)")
    x_e = all_targets[:, optimal_iteration]
    y_e = all_mean_rates_e[:, optimal_iteration]
    slope_e, intercept_e, r_e, _, _ = linregress(x_e, y_e)
    axes[1].scatter(x_e, y_e, s=10)
    axes[1].plot(x_e, intercept_e + slope_e * x_e, label=f"r = {r_e:.4f}")
    axes[1].set_xlabel("Targets at optimal iteration")
    axes[1].set_ylabel("Mean rate (E)")
    axes[1].legend()

    axes[2].set_title("Targets vs mean rate (I)")
    x_i = all_targets[:, optimal_iteration]
    y_i = all_mean_rates_i[:, optimal_iteration]
    slope_i, intercept_i, r_i, _, _ = linregress(x_i, y_i)
    axes[2].scatter(x_i, y_i, s=10)
    axes[2].plot(x_i, intercept_i + slope_i * x_i, label=f"r = {r_i:.4f}")
    axes[2].set_xlabel("Targets at optimal iteration")
    axes[2].set_ylabel("Mean rate (I)")
    axes[2].legend()

    axes[3].set_title("Targets vs entropy")
    y_entropy_e = all_entropies_e[:, optimal_iteration]
    ent_slope_e, ent_intercept_e, ent_r_e, _, _ = linregress(x_e, y_entropy_e)
    axes[3].scatter(x_e, y_entropy_e, s=10, label=f"E, r = {ent_r_e:.4f}")
    axes[3].plot(x_e, ent_intercept_e + ent_slope_e * x_e, color="C0")

    y_entropy_i = all_entropies_i[:, optimal_iteration]
    ent_slope_i, ent_intercept_i, ent_r_i, _, _ = linregress(x_e, y_entropy_i)
    axes[3].scatter(x_e, y_entropy_i, s=10, label=f"I, r = {ent_r_i:.4f}")
    axes[3].plot(x_e, ent_intercept_i + ent_slope_i * x_e, color="C1")
    axes[3].set_xlabel("Targets at optimal iteration")
    axes[3].set_ylabel("Entropy")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(plot_path / f"{params.nmega}")
    plt.close(fig)


def write_master_line(master_path: Path, params: SubjectParams, base_target: float,
                      time_taken: float, mean_entropy_e: float, mean_entropy_i: float,
                      mean_fc: float, mean_omat: float, optimal_targets: np.ndarray) -> None:
    target_values = "\t".join(f"{val}" for val in optimal_targets)
    with master_path.open("a") as handle:
        handle.write(
            f"{params.nmega}\t{params.g_value}\t{base_target}\t{time_taken}\t"
            f"{mean_entropy_e}\t{mean_entropy_i}\t{mean_fc}\t{mean_omat}\t{target_values}\n"
        )


def optimise_subject(params: SubjectParams) -> None:
    structural_connectivity, empirical_omats_flat, empirical_fcs_flat = load_subject_data(params)

    output_dirs = ensure_output_structure(OUTPUT_BASE)

    empirical_fc_bold = reconstruct_symm(empirical_fcs_flat, diag_value=1.0).astype(np.float64)
    empirical_omat_bold = reconstruct_symm(empirical_omats_flat, diag_value=0.0).astype(np.float64)

    mean_fc_emp = empirical_fc_bold.mean()
    mean_omat_emp = empirical_omat_bold.mean()
    empirical_strengths = np.sum(empirical_omat_bold, axis=1, dtype=np.float64).astype(np.float64)

    DMF.G = params.g_value
    DMF.SC = 0.2 * structural_connectivity

    target_vector = np.full(NNODES, params.target_value, dtype=np.float64)

    all_targets = np.zeros((NNODES, ITERS), dtype=np.float64)
    all_mean_rates_e = np.zeros((NNODES, ITERS), dtype=np.float64)
    all_entropies_e = np.zeros((NNODES, ITERS), dtype=np.float64)
    all_mean_rates_i = np.zeros((NNODES, ITERS), dtype=np.float64)
    all_entropies_i = np.zeros((NNODES, ITERS), dtype=np.float64)
    ssims = np.zeros(ITERS, dtype=np.float64)
    corrs = np.zeros(ITERS, dtype=np.float64)
    corrs_fc = np.zeros(ITERS, dtype=np.float64)

    best_corr = -np.inf
    bold_output_path = output_dirs["bold"] / f"BOLD_optimal_nmega{params.nmega}.npz"

    start_time = tm()

    for iteration in range(ITERS):
        all_targets[:, iteration] = target_vector

        aggregated_omat = np.zeros((NNODES, NNODES), dtype=np.float64)
        aggregated_mean_e = np.zeros(NNODES, dtype=np.float64)
        aggregated_entropy_e = np.zeros(NNODES, dtype=np.float64)
        aggregated_mean_i = np.zeros(NNODES, dtype=np.float64)
        aggregated_entropy_i = np.zeros(NNODES, dtype=np.float64)
        bold_to_save: Dict[str, np.ndarray] = {"it": iteration}
        last_simulated_fc = None

        for seed in range(NSEEDS):
            logger.info("Running N_MEGA %s, iteration %d/%d, seed %d", params.nmega, iteration + 1, ITERS, seed)
            (
                sim_fc,
                sim_omat,
                mean_rates_e,
                mean_rates_i,
                entropies_e,
                entropies_i,
                bold_filtered,
            ) = generate_matrices(target_vector, seed)

            sim_omat = np.asarray(sim_omat, dtype=np.float64)
            mean_rates_e = np.asarray(mean_rates_e, dtype=np.float64)
            entropies_e = np.asarray(entropies_e, dtype=np.float64)
            mean_rates_i = np.asarray(mean_rates_i, dtype=np.float64)
            entropies_i = np.asarray(entropies_i, dtype=np.float64)
            aggregated_omat += sim_omat
            aggregated_mean_e += mean_rates_e
            aggregated_entropy_e += entropies_e
            aggregated_mean_i += mean_rates_i
            aggregated_entropy_i += entropies_i
            bold_to_save[f"seed{seed}_array"] = np.asarray(bold_filtered, dtype=np.float32)
            last_simulated_fc = np.asarray(sim_fc, dtype=np.float64)

        aggregated_omat /= NSEEDS
        aggregated_mean_e /= NSEEDS
        aggregated_entropy_e /= NSEEDS
        aggregated_mean_i /= NSEEDS
        aggregated_entropy_i /= NSEEDS

        sim_strengths = np.sum(aggregated_omat, axis=1, dtype=np.float64).astype(np.float64)
        target_vector += EPSILON * (empirical_strengths - sim_strengths)
        np.maximum(target_vector, 0, out=target_vector)

        iteration_corr = np.corrcoef(aggregated_omat[TRIU_INDICES], empirical_omat_bold[TRIU_INDICES])[0, 1]
        corrs[iteration] = iteration_corr
        ssims[iteration] = ssim(
            aggregated_omat,
            empirical_omat_bold,
            data_range=get_range(aggregated_omat, empirical_omat_bold),
        )

        if last_simulated_fc is not None:
            corrs_fc[iteration] = np.corrcoef(last_simulated_fc[TRIU_INDICES], empirical_fc_bold[TRIU_INDICES])[0, 1]

        all_mean_rates_e[:, iteration] = aggregated_mean_e
        all_entropies_e[:, iteration] = aggregated_entropy_e
        all_mean_rates_i[:, iteration] = aggregated_mean_i
        all_entropies_i[:, iteration] = aggregated_entropy_i

        if iteration_corr > best_corr:
            np.savez_compressed(bold_output_path, **bold_to_save)
            best_corr = iteration_corr

    elapsed = tm() - start_time

    optimal_iteration = int(np.argmax(corrs))
    optimal_targets = all_targets[:, optimal_iteration]
    mean_entropy_e = all_entropies_e[:, optimal_iteration].mean()
    mean_entropy_i = all_entropies_i[:, optimal_iteration].mean()

    output_filename = output_dirs["output"] / f"nmega{params.nmega}_output.npz"
    np.savez_compressed(
        output_filename,
        all_targets=all_targets,
        optimal_targets=optimal_targets,
        all_mean_rates_E=all_mean_rates_e,
        all_entropies_E=all_entropies_e,
        all_mean_rates_I=all_mean_rates_i,
        all_entropies_I=all_entropies_i,
        ssims=ssims,
        corrs=corrs,
        corrs_FC=corrs_fc,
    )

    plot_results(
        output_dirs["plots"],
        params,
        corrs,
        corrs_fc,
        ssims,
        all_targets,
        all_mean_rates_e,
        all_mean_rates_i,
        all_entropies_e,
        all_entropies_i,
    )

    write_master_line(
        output_dirs["master"],
        params,
        params.target_value,
        elapsed,
        mean_entropy_e,
        mean_entropy_i,
        mean_fc_emp,
        mean_omat_emp,
        optimal_targets,
    )

    gc.collect()

    import matplotlib.pyplot as plt

    if last_simulated_fc is not None:
        plt.figure(100)
        plt.clf()
        plt.imshow(last_simulated_fc, cmap="jet")
        plt.colorbar()
        plt.show()


def main() -> None:
    args = parse_args()
    params = resolve_subject(args.row_index)
    optimise_subject(params)


if __name__ == "__main__":
    main()
