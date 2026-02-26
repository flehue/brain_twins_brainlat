# brain_twins_brainlat

This repository supports the BrainLat biophysical digital twins project, which builds individualized whole-brain models to link neural dynamics with phenotypes (age, sex, diagnosis) and expotypes (environmental and social exposures).

## Contents

1. [System Requirements](#1-system-requirements)
2. [Installation Guide](#2-installation-guide)
3. [Demo](#3-demo)
4. [Instructions for use](#4-instructions-for-use)

## 1. System Requirements

### Software Requirements

#### OS Requirements

The codebase is Python-based and has been tested on:

- Ubuntu 22.04.5 LTS (x86_64)
- Linux kernel `6.8.0-100-generic`

It should also run on recent Linux and macOS versions with a compatible Python scientific stack.

#### Python Dependencies

Tested analysis environment:

- Python `3.13.2`
- numpy `2.4.2`
- scipy `1.17.0`
- pandas `3.0.0`
- matplotlib `3.10.8`
- seaborn `0.13.2`
- scikit-learn `1.8.0`
- statsmodels `0.14.6`
- joblib `1.5.3`

Additional dependencies required by simulation/optimization pipelines:

- numba `>=0.60`
- scikit-image `>=0.24`
- psutil `>=5.9`

### Versions Tested

- The demo in Section 3 has been tested on Ubuntu 22.04.5 + Python 3.13.2 with the package versions listed above.
- End-to-end sweeps and large-scale optimization scripts are intended for HPC/cluster execution (SLURM is referenced in pipeline notes).

### Non-standard Hardware Requirements

- No non-standard hardware is required for the demo.
- For full cohort-scale sweeps, a multi-core HPC environment is strongly recommended (typical: many CPU cores and high RAM).

## 2. Installation Guide

### Instructions

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install \
  numpy==2.4.2 scipy==1.17.0 pandas==3.0.0 \
  matplotlib==3.10.8 seaborn==0.13.2 \
  scikit-learn==1.8.0 statsmodels==0.14.6 joblib==1.5.3 \
  numba scikit-image psutil
```

For scripts that import local modules (`model/`, `support_scripts/`, `analysis/`), set:

```bash
export PYTHONPATH="$PWD/model:$PWD/support_scripts:$PWD/analysis:$PYTHONPATH"
```

If `numba` installation fails on your Python version, use Python 3.11 and reinstall.

### Typical Install Time

- On a normal desktop computer: approximately `5-15 minutes` (depends on network and whether wheels are prebuilt).

## 3. Demo

This demo runs PCEV on included repository data (`data/derived/model_output_plus_exposome_data_v2.csv`).

### Instructions to Run on Data

From the repository root:

```bash
mkdir -p analysis/results

./.venv/bin/python - <<'PY'
import pandas as pd
import numpy as np

src = 'data/derived/model_output_plus_exposome_data_v2.csv'
df = pd.read_csv(src, usecols=['ent_E_0','ent_E_1','ent_E_2','Age','gof_corr','Sex'])
df = df.dropna()
df = df[df['Sex'].isin(['Male', 'Female'])].copy()
df['Sex_num'] = df['Sex'].map({'Male': 0.0, 'Female': 1.0})
df = df.head(1200)

np.save('/tmp/pcev_demo_Y.npy', df[['ent_E_0','ent_E_1','ent_E_2']].to_numpy(float))
np.save('/tmp/pcev_demo_X.npy', df[['Age']].to_numpy(float))
np.save('/tmp/pcev_demo_C.npy', df[['gof_corr','Sex_num']].to_numpy(float))
print(f'demo rows: {len(df)}')
PY

/usr/bin/time -f 'elapsed=%E' ./.venv/bin/python analysis/pcev.py \
  --Y /tmp/pcev_demo_Y.npy \
  --X /tmp/pcev_demo_X.npy \
  --C /tmp/pcev_demo_C.npy \
  --inference approx \
  --save_prefix analysis/results/pcev_demo
```

### Expected Output

Terminal output prints a JSON summary with fields like:

- `n`, `p`, `q`
- `h2`
- `lambda_max`
- `p_analytic`

Output files:

- `analysis/results/pcev_demo_weights.npy`
- `analysis/results/pcev_demo_scores.npy`

### Expected Run Time

- On a normal desktop computer: typically `~1-10 seconds` for the demo above.

## 4. Instructions for use

### Project Overview

This repository supports the BrainLat biophysical digital twins project, which builds individualized whole-brain models to link neural dynamics with phenotypes (age, sex, diagnosis) and expotypes (environmental and social exposures). A large multinational cohort (n=6,986 across 16 countries; HC, MCI, AD, FTLD) is used to fit dynamic mean-field (DMF) models with inhibitory plasticity to empirical high-order functional connectivity. From the fitted models we derive mechanistic regional signatures (excitatory/inhibitory activity, entropy, and E/I balance) and quantify phenotype/expotype associations with multivariate PCEV analyses.

### Repository Structure

- `pipelines/` end-to-end processing pipelines
- `model/` DMF model implementations and utilities
- `data/` original and derived datasets used across pipelines and analyses
- `analysis/` downstream analyses, including PCEV phenotype/expotype mapping
- `support_scripts/` helper scripts
- `notebooks_tutorials/` notebooks and tutorials
- `docs/` documentation materials

### Pipeline Progression (See `tree.txt`)

1. `pipelines/00_omat_FC_from_BOLD` computes functional connectivity inputs and O-information matrices from BOLD time series.
2. `pipelines/01_optimize_SC` optimizes structural connectivity parameters.
3. `pipelines/02_sweeps` runs parameter sweeps for multiple targets.
4. `pipelines/03_global_fit` performs global model fitting.
5. `pipelines/04_regional_fit` performs regional fitting to obtain individualized parameters.

### Analysis (Phenotype + Expotype)

The `analysis/` folder contains the unified PCEV pipeline that links model-derived neural features to phenotypes and exposome variables while controlling for confounders:

- `analysis/analysis_03_pcev_phenotype_expotype.py` (age, sex, diagnosis, exposome)

Required PCEV utilities:

- `analysis/pcev_feature_effects.py`
- `analysis/pcev_diagnosis.py`
- `analysis/pcev_sklearn.py`
- `analysis/pcev.py`

### Analysis Map (Index)

1. Group-based vs individualized models  
   Location: `analysis/analysis_01_group_vs_individualized_models.py`
2. Parameter-GMV correlations  
   Location: `analysis/analysis_02_optimal_parameters_vs_gmv.py`
3. PCEV phenotype/expotype mapping  
   Location: `analysis/analysis_03_pcev_phenotype_expotype.py`
4. LME analysis for phenotypes  
   Location: `analysis/analysis_04_lme_phenotypes.py`
5. LME analysis for expotypes  
   Location: `analysis/analysis_05_lme_expotypes.py`

### Data

The `data/` folder contains:

- `data/original/` raw inputs (BOLD and SC)
- `data/derived/` processed data, sweeps, and model outputs used by analyses

Additional notes from repository documentation:

- `pipelines/00_omat_FC_from_BOLD/process_functional.py` generates functional derivatives (e.g., FC/Omat outputs under `data/derived/`).
- Sweeps are stored under `data/derived/sweeps/`.
- `data/derived/model_output_plus_exposome_data_v2.csv` is the merged analysis-ready table used by phenotype/expotype analyses.
- Some large artifacts (e.g., `concat.npy`) may be unavailable in lightweight clones; coordinate with the repository manager if needed.

### How to Run the Software on Your Data

1. Place/organize your raw inputs following the existing layout in `data/original/` (BOLD time series and structural connectivity inputs).
2. Export local module paths:

```bash
export PYTHONPATH="$PWD/model:$PWD/support_scripts:$PWD/analysis:$PYTHONPATH"
```

3. Run preprocessing to derive FC and O-information matrices:

```bash
python pipelines/00_omat_FC_from_BOLD/process_functional.py
```

4. Optimize structural connectivity:

```bash
python pipelines/01_optimize_SC/optimize_SC.py
```

5. Run cohort/group sweeps (typically via SLURM for full datasets), then global fit:

```bash
python pipelines/03_global_fit/calculate_BOLD_optimals_cluster.py
```

6. Run regional optimization to obtain individualized features:

```bash
python pipelines/04_regional_fit/optimize_target_EandI.py
# or
python pipelines/04_regional_fit/optimize_target_EandI_unmatched.py --help
```

7. Assemble your final analysis table (same schema as `data/derived/model_output_plus_exposome_data_v2.csv`) and run:

```bash
python analysis/analysis_03_pcev_phenotype_expotype.py
```

8. Review outputs under `analysis/results/`.
