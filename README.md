# brain_twins_brainlat

**Overview**
This repository supports the BrainLat biophysical digital twins project, which builds individualized whole-brain models to link neural dynamics with phenotypes (age, sex, diagnosis) and expotypes (environmental and social exposures). A large multinational cohort (n=6,986 across 16 countries; HC, MCI, AD, FTLD) is used to fit dynamic mean-field (DMF) models with inhibitory plasticity to empirical high-order functional connectivity. From the fitted models we derive mechanistic regional signatures (excitatory/inhibitory activity, entropy, and E/I balance) and quantify phenotype/expotype associations with multivariate PCEV analyses.

**Repository Structure**
- `pipelines/` end-to-end processing pipelines
- `model/` DMF model implementations and utilities
- `data/` original and derived datasets used across pipelines and analyses
- `analysis/` downstream analyses, including PCEV phenotype/expotype mapping
- `support_scripts/` helper scripts
- `notebooks_tutorials/` notebooks and tutorials
- `docs/` documentation materials

**Pipeline Progression (See `tree.txt`)**
1. `pipelines/00_omat_FC_from_BOLD` computes functional connectivity inputs and O-information matrices from BOLD time series.
2. `pipelines/01_optimize_SC` optimizes structural connectivity parameters.
3. `pipelines/02_sweeps` runs parameter sweeps for multiple targets.
4. `pipelines/03_global_fit` performs global model fitting.
5. `pipelines/04_regional_fit` performs regional fitting to obtain individualized parameters.

**Analysis (Phenotype + Expotype)**
The `analysis/` folder contains the unified PCEV pipeline that links model-derived neural features to phenotypes and exposome variables while controlling for confounders:
- `analysis/pcev_analysis_phenotype_expotype.py` (age, sex, diagnosis, exposome)

Required PCEV utilities:
- `analysis/pcev_feature_effects.py`
- `analysis/pcev_diagnosis.py`
- `analysis/pcev_sklearn.py`
- `analysis/pcev.py`

**Analysis Map (Index)**
This section lists the main analyses and their locations in the repository.

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

**Data**
The `data/` folder contains:
- `data/original/` raw inputs (BOLD and SC)
- `data/derived/` processed data, sweeps, and model outputs used by analyses
