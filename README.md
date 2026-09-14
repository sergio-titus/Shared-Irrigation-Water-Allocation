# Shared Irrigation-Water Allocation: Lightweight Rules vs PPO

Reproducibility repository for the manuscript **“Comparing Reinforcement Learning and Lightweight Strategies for Shared Irrigation-Water Allocation Using AquaCrop-OSPy.”**

**Authors:** Koffi Titus Sergio Aglin, Anthony K. Muchiri, and Celestin Nkundineza.

This repository contains the simulation code, processed weather data, frozen protocol manifests, controller configurations, held-out evaluation outputs, statistical-analysis code, diagnostic analyses, publication figures, and the ten frozen final PPO checkpoints used in the study.

## Study design

The study compares three controllers for allocating a limited shared daily irrigation supply among four staggered maize fields:

- **Equal** — divides available shared water equally among requesting fields.
- **Priority** — ranks requesting fields by depletion ratio and allocates water by priority.
- **PPO** — a pooled multi-climate Proximal Policy Optimization controller with a climate one-hot indicator.

The three climates are **Tunis**, **Niamey**, and **Cotonou**. Controller development uses **1984–2018** (35 years per climate; 105 climate-years). The frozen held-out evaluation uses **2019–2025** (7 years per climate; 21 climate-years) at seasonal water availability levels of **100%, 60%, and 40%**. The complete 1984–2025 record is used only for descriptive climate characterization.

Core irrigation constraints are a 40 mm shared daily capacity, 25 mm maximum irrigation per field per day, and a depletion trigger of 0.40. Four maize fields are planted on May 1, May 8, May 15, and May 22.

## Repository layout

```text
.
├── scripts/                              
│   ├── multiclimate_optimization_core.py 
│   ├── audit_multiclimate_weather.py
│   ├── freeze_experimental_protocol.py
│   ├── generate_development_reference.py
│   ├── select_ppo_training_horizon.py
│   ├── train_final_ppo_ensemble.py
│   ├── evaluate_all_controllers.py
│   ├── run_statistical_analysis.py
│   └── ...                              
```


## Installation

Python **3.11** is recommended. Create a fresh environment and install:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The saved PPO checkpoints record Python 3.11.16, Stable-Baselines3 2.9.0, PyTorch 2.14.0+cpu, NumPy 2.4.6, Gymnasium 1.3.0, and Cloudpickle 3.1.2. The exact AquaCrop-OSPy package version was not embedded in the PPO checkpoint metadata; see `ENVIRONMENT.md`.

## Reproducing the main experiment

Run commands from the repository root. The scripts contain integrity checks designed to preserve the temporal split and frozen protocol.

### 1. Weather and protocol

The processed weather files used in the paper are already included. To audit/regenerate the weather acquisition products, run:

```bash
python scripts/audit_multiclimate_weather.py
```

Then freeze/recreate the development protocol and fully irrigated development references:

```bash
python scripts/freeze_experimental_protocol.py
python scripts/generate_development_reference.py
python scripts/validate_multiclimate_environment.py
python scripts/freeze_ppo_training_configuration.py
```

### 2. PPO horizon selection and final training

```bash
python scripts/select_ppo_training_horizon.py
python scripts/train_final_ppo_ensemble.py
```

The selected horizon is 102,400 timesteps. Final seeds are `1103, 2207, 3319, 4421, 5527, 6637, 7741, 8849, 9967, 11071`. All ten are retained; no final-seed model selection is performed.

The repository already contains the ten final trained checkpoints, so users interested only in reproducing final evaluation can skip the expensive training stages.

### 3. Frozen held-out evaluation

```bash
python scripts/generate_final_test_reference.py
python scripts/freeze_final_evaluation_protocol.py
python scripts/evaluate_all_controllers.py
python scripts/run_statistical_analysis.py
python scripts/run_controller_diagnostics.py
```

`53J` evaluates Equal, Priority, and the frozen ten-seed PPO ensemble on Tunis, Niamey, and Cotonou for 2019–2025 at 100%, 60%, and 40% seasonal availability. PPO seed outcomes are averaged within each climate-year before controller-level inference.

### 4. Publication figures

```bash
python scripts/generate_publication_figures.py
```

### 5. Diagnostic analyses reported in the manuscript

Projection diagnostic:

```bash
python scripts/analyze_ppo_action_projection.py
python scripts/analyze_ppo_projection_mechanisms.py
```

The large step-level CSV files produced by these scripts are intentionally not versioned because they are deterministic/regenerable intermediates; the compact summary outputs used in the manuscript are included.

Offline perfect-foresight numerical reference under 40% availability:

```bash
python scripts/assess_exhaustive_optimization_feasibility.py
python scripts/run_perfect_foresight_smoke_test.py
python scripts/analyze_perfect_foresight_parameterization.py
python scripts/run_perfect_foresight_final_panel.py
python scripts/analyze_tunis2022_parameterization_sensitivity.py
```

These analyses are **post-hoc diagnostics**. The perfect-foresight calculation is a restricted, non-deployable numerical reference and is not claimed to be a mathematical upper bound or globally optimal controller.

Robust descriptive water-productivity summary:

```bash
python scripts/summarize_water_productivity_robustly.py
```

The original frozen Friedman/Wilcoxon-Holm inferential analyses remain in `53K`; the WP script changes descriptive presentation to median [Q1, Q3] because of structural zeros and strong skew.
`

## Reproducibility notes

The final-test script explicitly certifies that controller choices were frozen before the 2019–2025 test was opened. The final evaluation retains all ten PPO seeds and does not perform statistical testing until the subsequent analysis stage. Horizon selection is performed only on the development period and uses agronomic validation metrics rather than PPO training reward.

Some numerical outputs may show very small platform-dependent floating-point differences when rerun with different operating systems, BLAS libraries, PyTorch builds, or AquaCrop-OSPy versions. The committed CSV/JSON files are the outputs used for the manuscript.

## Data source

Weather inputs were obtained from NASA POWER: https://power.larc.nasa.gov/data-access-viewer and converted to the AquaCrop input structure by the project scripts. Raw API responses and processed weather tables used by the simulations are retained under data/weather/` for traceability.

## Citation

If you use this repository, please cite the associated paper. A machine-readable citation template is provided in `CITATION.cff`. Update the DOI and repository release information after publication/archiving.

## License

Code in this repository is released under the MIT License. Third-party software and external data remain subject to their respective licenses and terms of use.

## Public file naming

The public repository uses descriptive filenames based on each script or output's scientific function (for example, `train_final_ppo_ensemble.py`, `evaluate_all_controllers.py`, and `run_statistical_analysis.py`). Internal development milestone numbers used during the research workflow have been removed from public filenames to make the repository easier to navigate and cite.
