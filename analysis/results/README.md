# `analysis/results/` — regenerable analysis outputs

This directory holds outputs produced by the analysis scripts. **Its contents are not
version-controlled** (they are listed in `.gitignore`) because they are fully regenerable
and can be large. Only this `README.md` is tracked, so the folder exists in a fresh clone.

Run the scripts from the repository root with `PYTHONPATH` set (see the top-level `README.md`).

## What produces what

| Subfolder / file | Produced by | Notes |
|---|---|---|
| `pcev_demo_{weights,scores}.npy` | Demo in top-level `README.md` (`analysis/pcev.py`) | Quick smoke test of the PCEV core. |
| `pcev_results/age_odq_only_no_scanner/` | `analysis/analysis_03_pcev_phenotype_expotype.py` | Age phenotype PCEV. |
| `pcev_results/sex_odq_only_no_scanner/` | `analysis/analysis_03_pcev_phenotype_expotype.py` | Sex phenotype PCEV. |
| `pcev_results/diagnosis_odq_only_no_scanner/` | `analysis/analysis_03_pcev_phenotype_expotype.py` | Diagnosis PCEV. |
| `pcev_results/exposome_odq_only_no_scanner/` | `analysis/analysis_03_pcev_phenotype_expotype.py` | Exposome-domain PCEV (Air Pollution, Green space, etc.). |
| `pcev_results/individual_exposome_no_odq/` | `analysis/analysis_09_individual_exposome.py` | Individual-level (SVI) exposome PCEV. |
| `pcev_results/permutation_tests/` | `analysis/analysis_06_pcev_permutation_test.py` and `analysis/analysis_10_individual_exposome_permutation_test.py` | Null distributions / permutation p-values. |

Each PCEV subfolder typically contains `all_combos_summary.csv` plus, for the best
feature combination, `{combo}_h2_per_repeat.csv` and `{combo}_subject_scores.csv`.

## Controlling output location and run size

`analysis_03` (and the permutation-test scripts) honor environment variables, e.g.:

```bash
# Quick, reduced run (writes elsewhere so it does not clobber a full run):
N_REPEATS=2 N_SPLITS=2 MAX_COMBOS=1 SEED=2025 N_JOBS=1 \
PCEV_RESULTS_DIR=/tmp/pcev_smoke \
python analysis/analysis_03_pcev_phenotype_expotype.py
```

For publication-scale runs use the defaults (`N_REPEATS=500`, all combinations); these are
intended for an HPC/SLURM environment — see the `run_*.sh` / `submit_*.sh` wrappers in `analysis/`.
