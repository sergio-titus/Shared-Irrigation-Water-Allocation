# ============================================================
# CONTROLLER DIAGNOSTICS
# FINAL MULTI-CLIMATE DIAGNOSTIC ANALYSIS
#
# Analysis of already-frozen 53J final-test results only.
#
# NO AquaCrop simulation
# NO PPO inference
# NO PPO training
# NO controller modification
# NO model/seed selection
# NO new hypothesis tests
#
# Experimental unit for controller results remains climate-year.
#
# Diagnostic definitions inherited from 53J:
# - Budget utilization:
#       total allocated irrigation / seasonal budget * 100
# - Budget binding:
#       seasonal budget exhausted during episode
# - Request days:
#       days with positive aggregate irrigation request
# - Competition days:
#       days aggregate request exceeded the 40 mm/day
#       shared daily capacity
# - Irrigation days:
#       days with positive allocated irrigation
# - Decision latency:
#       controller decision computation only
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

EVAL_DIR = ROOT / "final_evaluation"
STATS_DIR = ROOT / "final_statistics"

OUTPUT_DIR = ROOT / "final_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = EVAL_DIR / "controller_evaluation_final_controller_panel.csv"
BINDING_FILE = EVAL_DIR / "controller_evaluation_binding_water_summary.csv"
CLIMATE_FILE = EVAL_DIR / "controller_evaluation_final_climate_summary.csv"
SEED_FILE = EVAL_DIR / "controller_evaluation_ppo_seed_robustness_40pct.csv"
INTEGRITY_53J = EVAL_DIR / "controller_evaluation_final_evaluation_integrity.json"

STATS_INTEGRITY = STATS_DIR / "statistical_analysis_statistical_integrity.json"

OUT_CLIMATE = OUTPUT_DIR / "controller_diagnostics_climate_controller_diagnostics.csv"
OUT_BINDING = OUTPUT_DIR / "controller_diagnostics_water_binding_diagnostics.csv"
OUT_LATENCY = OUTPUT_DIR / "controller_diagnostics_decision_latency_summary.csv"
OUT_SEEDS = OUTPUT_DIR / "controller_diagnostics_ppo_seed_robustness_summary.csv"
OUT_40 = OUTPUT_DIR / "controller_diagnostics_40pct_climate_summary.csv"
OUT_INTEGRITY = OUTPUT_DIR / "controller_diagnostics_diagnostic_integrity.json"


# ============================================================
# 2. LOAD
# ============================================================

required_files = [
    PANEL_FILE,
    BINDING_FILE,
    CLIMATE_FILE,
    SEED_FILE,
    INTEGRITY_53J,
    STATS_INTEGRITY,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen file missing: {path}"
        )

panel = pd.read_csv(PANEL_FILE)
binding = pd.read_csv(BINDING_FILE)
climate_summary = pd.read_csv(CLIMATE_FILE)
seed_df = pd.read_csv(SEED_FILE)

with open(INTEGRITY_53J, "r", encoding="utf-8") as f:
    integrity_53j = json.load(f)

with open(STATS_INTEGRITY, "r", encoding="utf-8") as f:
    integrity_53k = json.load(f)


# ============================================================
# 3. BASIC INTEGRITY
# ============================================================

if len(panel) != 189:
    raise RuntimeError(
        f"Expected 189 controller-panel rows; got {len(panel)}."
    )

if len(climate_summary) != 27:
    raise RuntimeError(
        f"Expected 27 climate-summary rows; got "
        f"{len(climate_summary)}."
    )

if len(binding) != 27:
    raise RuntimeError(
        f"Expected 27 binding-summary rows; got {len(binding)}."
    )

if len(seed_df) != 10:
    raise RuntimeError(
        f"Expected 10 retained PPO seeds; got {len(seed_df)}."
    )

controllers = set(panel["Controller"].unique())

expected_controllers = {
    "Equal",
    "Priority",
    "Optimized PPO",
}

if controllers != expected_controllers:
    raise RuntimeError(
        f"Unexpected controllers: {sorted(controllers)}"
    )

climates = set(panel["Climate"].unique())

expected_climates = {
    "Tunis",
    "Niamey",
    "Cotonou",
}

if climates != expected_climates:
    raise RuntimeError(
        f"Unexpected climates: {sorted(climates)}"
    )

if integrity_53k.get("controller_tuning_after_test") is not False:
    raise RuntimeError(
        "53K integrity does not certify frozen controllers."
    )


# ============================================================
# 4. CLIMATE × SCARCITY × CONTROLLER DIAGNOSTIC TABLE
#
# Use the already-generated 53J climate summary.
# Add irrigation-days and latency from the common panel.
# ============================================================

panel_diag = (
    panel
    .groupby(
        [
            "Climate",
            "Scarcity_pct",
            "Controller",
        ],
        as_index=False,
    )
    .agg(
        Irrigation_days_mean=(
            "Irrigation_days",
            "mean",
        ),
        Irrigation_days_SD=(
            "Irrigation_days",
            "std",
        ),
        Decision_latency_mean_ms=(
            "Decision_latency_mean_ms",
            "mean",
        ),
        Decision_latency_SD_ms=(
            "Decision_latency_mean_ms",
            "std",
        ),
        Total_irrigation_mean_mm=(
            "Total_irrigation_mm",
            "mean",
        ),
        Seasonal_budget_mean_mm=(
            "Seasonal_budget_mm",
            "mean",
        ),
        Cumulative_request_mean_mm=(
            "Cumulative_request_mm",
            "mean",
        ),
    )
)

diagnostic = climate_summary.merge(
    panel_diag,
    on=[
        "Climate",
        "Scarcity_pct",
        "Controller",
    ],
    how="left",
    validate="one_to_one",
)

diagnostic = diagnostic.sort_values(
    [
        "Scarcity_pct",
        "Climate",
        "Controller",
    ],
    ascending=[
        False,
        True,
        True,
    ],
).reset_index(drop=True)

diagnostic.to_csv(
    OUT_CLIMATE,
    index=False,
)


# ============================================================
# 5. WATER-BINDING TABLE
# ============================================================

binding_out = binding.copy()

binding_out["Water_regime"] = np.where(
    binding_out["Fraction_budget_binding"] > 0,
    "Binding observed",
    "No budget exhaustion",
)

binding_out = binding_out.sort_values(
    [
        "Scarcity_pct",
        "Climate",
        "Controller",
    ],
    ascending=[
        False,
        True,
        True,
    ],
).reset_index(drop=True)

binding_out.to_csv(
    OUT_BINDING,
    index=False,
)


# ============================================================
# 6. DECISION LATENCY
#
# Descriptive only.
# ============================================================

latency = (
    panel
    .groupby(
        "Controller",
        as_index=False,
    )
    .agg(
        N_cases=(
            "Decision_latency_mean_ms",
            "size",
        ),
        Mean_latency_ms=(
            "Decision_latency_mean_ms",
            "mean",
        ),
        SD_latency_ms=(
            "Decision_latency_mean_ms",
            "std",
        ),
        Median_latency_ms=(
            "Decision_latency_mean_ms",
            "median",
        ),
        Min_latency_ms=(
            "Decision_latency_mean_ms",
            "min",
        ),
        Max_latency_ms=(
            "Decision_latency_mean_ms",
            "max",
        ),
    )
)

latency.to_csv(
    OUT_LATENCY,
    index=False,
)


# ============================================================
# 7. PPO SEED ROBUSTNESS AT 40%
#
# Descriptive only.
# ============================================================

seed_metrics = [
    "Mean_yield_retention_pct",
    "Worst_field_retention_mean_pct",
    "Jain_mean",
    "Water_productivity_mean",
    "Budget_utilization_mean_pct",
]

seed_rows = []

for metric in seed_metrics:

    values = seed_df[metric].to_numpy(
        dtype=float
    )

    seed_rows.append(
        {
            "Metric": metric,
            "N_seeds": len(values),
            "Mean": float(np.mean(values)),
            "SD": float(np.std(values, ddof=1)),
            "Min": float(np.min(values)),
            "Max": float(np.max(values)),
            "Range": float(
                np.max(values)
                - np.min(values)
            ),
        }
    )

seed_summary = pd.DataFrame(seed_rows)

seed_summary.to_csv(
    OUT_SEEDS,
    index=False,
)


# ============================================================
# 8. 40% CLIMATE-SPECIFIC TABLE
#
# Main diagnostic table for severe water limitation.
# ============================================================

summary_40 = diagnostic[
    np.isclose(
        diagnostic["Scarcity_pct"],
        40.0,
    )
].copy()

cols_40 = [
    "Climate",
    "Controller",
    "Yield_retention_mean_pct",
    "Worst_field_retention_mean_pct",
    "Jain_mean",
    "Water_productivity_mean",
    "Budget_utilization_mean_pct",
    "Binding_fraction",
    "Request_days_mean",
    "Competition_days_mean",
    "Irrigation_days_mean",
    "Total_irrigation_mean_mm",
    "Decision_latency_mean_ms",
]

summary_40 = summary_40[
    cols_40
]

summary_40.to_csv(
    OUT_40,
    index=False,
)


# ============================================================
# 9. ADDITIONAL DESCRIPTIVE CHECKS
# ============================================================

# Number of climate-year cases in which the seasonal budget
# actually exhausted, using the frozen controller panel.
binding_case_summary = (
    panel
    .groupby(
        [
            "Climate",
            "Scarcity_pct",
            "Controller",
        ],
        as_index=False,
    )
    .agg(
        N_cases=("Year", "size"),
        Mean_binding_fraction=(
            "Budget_binding_fraction_across_seeds",
            "mean",
        ),
        Mean_budget_utilization_pct=(
            "Budget_utilization_pct",
            "mean",
        ),
    )
)


# ============================================================
# 10. INTEGRITY RECORD
# ============================================================

integrity = {
    "milestone": "53L",
    "status": "COMPLETE",
    "analysis_type":
        "descriptive diagnostics of frozen 53J results",
    "aquacrop_simulation_performed": False,
    "ppo_inference_performed": False,
    "ppo_training_performed": False,
    "controller_modified": False,
    "model_selection_performed": False,
    "seed_selection_performed": False,
    "new_hypothesis_tests_performed": False,
    "controller_final_test_already_open": True,
    "controller_panel_rows": int(len(panel)),
    "climate_summary_rows": int(len(climate_summary)),
    "binding_summary_rows": int(len(binding)),
    "ppo_final_seeds": int(len(seed_df)),
    "competition_days_definition":
        "days aggregate irrigation request exceeded "
        "the 40 mm/day shared daily capacity",
    "request_days_definition":
        "days with positive aggregate irrigation request",
    "irrigation_days_definition":
        "days with positive allocated irrigation",
    "budget_binding_definition":
        "seasonal irrigation budget exhausted during episode",
    "decision_latency_definition":
        "controller decision computation only",
}

with open(
    OUT_INTEGRITY,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        integrity,
        f,
        indent=2,
    )


# ============================================================
# 11. TERMINAL OUTPUT
# ============================================================

print()
print("=" * 78)
print("CONTROLLER DIAGNOSTICS COMPLETE")
print("=" * 78)

print()
print("40% SCARCITY — CLIMATE-SPECIFIC DIAGNOSTICS")
print("=" * 78)

print(
    summary_40.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("=" * 78)
print("WATER-BINDING DIAGNOSTICS — ALL CONDITIONS")
print("=" * 78)

binding_display = [
    "Climate",
    "Scarcity_pct",
    "Controller",
    "Mean_budget_utilization_pct",
    "Mean_request_days",
    "Mean_competition_days",
    "Mean_irrigation_days",
    "Fraction_budget_binding",
]

print(
    binding_out[
        binding_display
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("=" * 78)
print("DECISION LATENCY — ALL 63 CASES PER CONTROLLER")
print("=" * 78)

print(
    latency.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("=" * 78)
print("PPO SEED ROBUSTNESS — 40% SCARCITY")
print("=" * 78)

print(
    seed_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("=" * 78)
print("DIAGNOSTIC INTEGRITY")
print("=" * 78)

print("AquaCrop simulation       : NO")
print("PPO inference             : NO")
print("PPO training              : NO")
print("Controller modification   : NO")
print("Model/seed selection      : NO")
print("New hypothesis tests      : NO")
print("Frozen 53J results only   : YES")

print()
print("Saved:")
print(f"  {OUT_CLIMATE}")
print(f"  {OUT_BINDING}")
print(f"  {OUT_LATENCY}")
print(f"  {OUT_SEEDS}")
print(f"  {OUT_40}")
print(f"  {OUT_INTEGRITY}")

print()
print(
    "NEXT: freeze the complete multi-climate result set "
    "and prepare the manuscript tables, figures, Results, "
    "Discussion, and Conclusion."
)