#!/usr/bin/env python
# coding: utf-8

# Convergence diagnostics for personalized model optimization.
# Reviewer-requested empirical diagnostics:
#   (1) Number of iterations to GOF maximum (iter_to_max)
#   (2) Regional parameter change from iteration 0 to iter_to_max (delta_per_roi)
#   (3) GOF stability in a 10-iteration window centered on iter_to_max (gof_around_max_std)
#   (4) Percentage of subjects reaching stopping criterion (iter_to_max == 99)

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH      = REPO_ROOT / "data" / "derived" / "model_output_plus_exposome_data_v3.csv"
ROI_LABEL_PATH = REPO_ROOT / "data" / "derived" / "ROI_MNI_V4.csv"
OPT_DIR        = Path("/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/optimization/"
                      "optimize_target_from_omat_epsilon0.04/output_per_nmega")
TABLE_DIR      = REPO_ROOT / "analysis" / "tables"
FIGURES_DIR    = REPO_ROOT / "analysis" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["text.usetex"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

N_ROI  = 90
N_ITER = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_opt_npz(nmega: str) -> dict[str, np.ndarray] | None:
    path = OPT_DIR / f"nmega{nmega}_output.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


# ---------------------------------------------------------------------------
# ROI labels (first 90 rows: cortex + subcortex)
# ---------------------------------------------------------------------------

roi_df     = pd.read_csv(ROI_LABEL_PATH)
roi_labels = roi_df[roi_df["structure"].isin(["cortex", "subcortex"])]["label"].values  # shape (90,)

# ---------------------------------------------------------------------------
# Load main data and iterate over subjects
# ---------------------------------------------------------------------------

main_df = pd.read_csv(DATA_PATH, low_memory=False)
main_df["N_MEGA"] = main_df["N_MEGA"].astype(str).str.strip()
nmega_list = [str(n) for n in main_df["N_MEGA"].unique()]

print(f"Subjects in main data: {len(nmega_list)}")

records      = []
delta_matrix = []   # (N_subjects, N_ROI)
missing      = 0

for nmega in nmega_list:
    d = load_opt_npz(nmega)
    if d is None:
        missing += 1
        continue

    corrs       = d["corrs"]        # (100,)
    all_targets = d["all_targets"]  # (90, 100)

    iter_to_max = int(np.argmax(corrs))

    delta_per_roi = np.abs(all_targets[:, iter_to_max] - all_targets[:, 0])  # (90,)

    lo = max(0, iter_to_max - 5)
    hi = min(N_ITER, iter_to_max + 5)
    gof_around_max_std = float(np.std(corrs[lo:hi]))

    reached_100 = iter_to_max == (N_ITER - 1)

    records.append({
        "N_MEGA":             nmega,
        "iter_to_max":        iter_to_max,
        "gof_at_max":         float(corrs[iter_to_max]),
        "gof_around_max_std": gof_around_max_std,
        "reached_100":        int(reached_100),
    })
    delta_matrix.append(delta_per_roi)

print(f"Subjects with optimization file: {len(records)} / {len(nmega_list)}  (missing: {missing})")

diag_df      = pd.DataFrame(records)
delta_matrix = np.stack(delta_matrix)   # (N_subjects, 90)

pct_reached_100 = 100.0 * diag_df["reached_100"].mean()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

summary_rows = []
for col in ["iter_to_max", "gof_at_max", "gof_around_max_std"]:
    summary_rows.append({
        "metric": col,
        "mean":   diag_df[col].mean(),
        "sd":     diag_df[col].std(ddof=1),
        "median": diag_df[col].median(),
        "min":    diag_df[col].min(),
        "max":    diag_df[col].max(),
    })
summary_rows.append({
    "metric": "pct_reached_100",
    "mean":   pct_reached_100,
    "sd":     np.nan,
    "median": np.nan,
    "min":    np.nan,
    "max":    np.nan,
})

summary_df = pd.DataFrame(summary_rows)

diag_df.to_csv(TABLE_DIR / "convergence_diagnostics_per_subject.csv", index=False)
summary_df.to_csv(TABLE_DIR / "convergence_diagnostics_summary.csv", index=False)

print(summary_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------

n_reached = int(diag_df["reached_100"].sum())
n_not     = len(diag_df) - n_reached

fig = plt.figure(figsize=(18, 10))
gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.52, wspace=0.35)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[1, :])

# --- panel a ---
ax_a.hist(diag_df["iter_to_max"], bins=20, color="steelblue", edgecolor="white")
ax_a.set_xlabel("iteration of maximum GOF")
ax_a.set_ylabel("subjects")
ax_a.text(
    -0.12, 1.06,
    f"a. iteration to maximum  "
    f"(median={diag_df['iter_to_max'].median():.0f}, "
    f"mean\u00b1SD={diag_df['iter_to_max'].mean():.1f}\u00b1{diag_df['iter_to_max'].std(ddof=1):.1f})",
    transform=ax_a.transAxes, fontsize=8, va="bottom", ha="left",
)

# --- panel b ---
ax_b.hist(diag_df["gof_around_max_std"], bins=20, color="darkorange", edgecolor="white")
ax_b.set_xlabel("GOF SD (10-iter window around maximum)")
ax_b.set_ylabel("subjects")
ax_b.text(
    -0.12, 1.06,
    f"b. GOF stability around maximum  "
    f"(median={diag_df['gof_around_max_std'].median():.4f}, "
    f"mean={diag_df['gof_around_max_std'].mean():.4f})",
    transform=ax_b.transAxes, fontsize=8, va="bottom", ha="left",
)

# --- panel c ---
ax_c.bar(["< 100", "= 100"], [n_not, n_reached], color=["#4C9BE8", "#E84C4C"], edgecolor="white")
ax_c.set_xlabel("iteration of maximum GOF")
ax_c.set_ylabel("subjects")
ax_c.text(
    -0.12, 1.06,
    f"c. subjects reaching stopping criterion  "
    f"({pct_reached_100:.1f}%, n={n_reached}/{len(diag_df)})",
    transform=ax_c.transAxes, fontsize=8, va="bottom", ha="left",
)

# --- panel d ---
parts = ax_d.violinplot(
    [delta_matrix[:, r] for r in range(N_ROI)],
    positions=range(N_ROI),
    showmedians=True,
    widths=0.7,
)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.6)
for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
    parts[part].set_color("navy")
    parts[part].set_linewidth(0.8)

ax_d.set_xticks(range(N_ROI))
ax_d.set_xticklabels(roi_labels, rotation=90, fontsize=9.5)
ax_d.set_xlabel("region")
ax_d.set_ylabel("|$\Delta$ target| (Hz)")
ax_d.text(
    -0.03, 1.03,
    "d. regional parameter change from iteration 0 to GOF maximum",
    transform=ax_d.transAxes, fontsize=8, va="bottom", ha="left",
)

fig.savefig(FIGURES_DIR / "convergence_diagnostics.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "convergence_diagnostics.pdf", bbox_inches="tight")
plt.close(fig)

print("\nOutputs saved:")
print(f"  {TABLE_DIR}/convergence_diagnostics_per_subject.csv")
print(f"  {TABLE_DIR}/convergence_diagnostics_summary.csv")
print(f"  {FIGURES_DIR}/convergence_diagnostics.png/.pdf")
