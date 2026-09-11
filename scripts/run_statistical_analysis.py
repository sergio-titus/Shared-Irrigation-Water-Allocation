# ============================================================
# STATISTICAL ANALYSIS
# PREDECLARED FINAL STATISTICAL ANALYSIS
#
# Uses the frozen 53J controller panel:
#   - experimental unit = climate-year
#   - n = 21 paired climate-years per scarcity level
#   - PPO already averaged across all 10 frozen seeds
#
# For each primary metric x scarcity family:
#   1. Friedman test across Equal, Priority, Optimized PPO
#   2. Three paired two-sided Wilcoxon signed-rank tests
#   3. Holm correction across the three pairwise tests
#   4. Paired rank-biserial correlation
#
# NO controller tuning.
# NO PPO training.
# NO seed/model selection.
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata


ROOT = Path("results") / "multiclimate_extension"
EVAL_DIR = ROOT / "final_evaluation"

PANEL_FILE = EVAL_DIR / "controller_evaluation_final_controller_panel.csv"
EVAL_INTEGRITY_FILE = EVAL_DIR / "controller_evaluation_final_evaluation_integrity.json"
MANIFEST_FILE = ROOT / "final_evaluation_protocol" / "final_evaluation_protocol_final_evaluation_manifest_frozen.json"

OUTPUT_DIR = ROOT / "final_statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRIEDMAN_FILE = OUTPUT_DIR / "statistical_analysis_friedman_results.csv"
PAIRWISE_FILE = OUTPUT_DIR / "statistical_analysis_pairwise_wilcoxon_holm.csv"
DESCRIPTIVE_FILE = OUTPUT_DIR / "statistical_analysis_descriptive_summary.csv"
CLIMATE_DESCRIPTIVE_FILE = OUTPUT_DIR / "statistical_analysis_climate_descriptive_summary.csv"
INTEGRITY_FILE = OUTPUT_DIR / "statistical_analysis_statistical_integrity.json"


if not PANEL_FILE.exists():
    raise FileNotFoundError(f"Missing controller panel: {PANEL_FILE}")
if not EVAL_INTEGRITY_FILE.exists():
    raise FileNotFoundError(f"Missing 53J integrity file: {EVAL_INTEGRITY_FILE}")
if not MANIFEST_FILE.exists():
    raise FileNotFoundError(f"Missing frozen manifest: {MANIFEST_FILE}")

panel = pd.read_csv(PANEL_FILE)

with open(EVAL_INTEGRITY_FILE, "r", encoding="utf-8") as f:
    eval_integrity = json.load(f)

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)


if eval_integrity.get("controller_final_test_opened") is not True:
    raise RuntimeError("53J final test is not certified open.")
if eval_integrity.get("water_accounting_pass") is not True:
    raise RuntimeError("53J water accounting did not pass.")
if eval_integrity.get("ppo_training_performed") is not False:
    raise RuntimeError("Unexpected PPO training during final evaluation.")
if eval_integrity.get("ppo_model_selection") is not False:
    raise RuntimeError("Unexpected PPO model selection.")
if eval_integrity.get("ppo_seed_selection") is not False:
    raise RuntimeError("Unexpected PPO seed selection.")
if eval_integrity.get("primary_experimental_unit") != "climate-year":
    raise RuntimeError("Experimental unit no longer matches frozen protocol.")
if eval_integrity.get("ppo_seeds_averaged_before_inference") is not True:
    raise RuntimeError("PPO seeds were not averaged before inference.")


CONTROLLERS = ["Equal", "Priority", "Optimized PPO"]
SCARCITY_LEVELS = [100.0, 60.0, 40.0]

PRIMARY_METRICS = {
    "Yield retention": "Yield_retention_pct",
    "Worst-field retention": "Worst_field_retention_pct",
    "Jain fairness": "Jain",
    "Water productivity": "Water_productivity",
}

PAIR_DEFINITIONS = [
    ("Equal", "Priority"),
    ("Equal", "Optimized PPO"),
    ("Priority", "Optimized PPO"),
]

required_columns = {
    "Controller",
    "Climate",
    "Year",
    "Scarcity_pct",
    *PRIMARY_METRICS.values(),
}

missing_columns = required_columns - set(panel.columns)
if missing_columns:
    raise RuntimeError(
        "53J controller panel missing columns: "
        f"{sorted(missing_columns)}"
    )

if len(panel) != 189:
    raise RuntimeError(f"Expected 189 controller-panel rows, got {len(panel)}.")


def paired_rank_biserial(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    nonzero = np.abs(diff) > 1e-15
    diff = diff[nonzero]

    if len(diff) == 0:
        return 0.0

    ranks = rankdata(np.abs(diff), method="average")
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    denom = w_plus + w_minus

    if denom <= 1e-15:
        return 0.0

    return float((w_plus - w_minus) / denom)


def safe_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n_nonzero = int(np.sum(np.abs(diff) > 1e-15))

    if n_nonzero == 0:
        return {
            "Statistic": 0.0,
            "P_raw": 1.0,
            "N_nonzero": 0,
        }

    result = wilcoxon(
        a,
        b,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )

    return {
        "Statistic": float(result.statistic),
        "P_raw": float(result.pvalue),
        "N_nonzero": n_nonzero,
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank_index, original_index in enumerate(order):
        multiplier = m - rank_index
        adjusted = min(
            1.0,
            multiplier * p_values[original_index],
        )
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank_index] = running_max

    adjusted = np.empty(m, dtype=float)

    for rank_index, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank_index]

    return adjusted


# ============================================================
# DESCRIPTIVE SUMMARY
# ============================================================

descriptive_rows = []

for scarcity in SCARCITY_LEVELS:
    d_s = panel[
        np.isclose(panel["Scarcity_pct"], scarcity)
    ].copy()

    for controller in CONTROLLERS:
        d = d_s[
            d_s["Controller"] == controller
        ].copy()

        if len(d) != 21:
            raise RuntimeError(
                f"{scarcity}% {controller}: expected 21 "
                f"climate-years, got {len(d)}."
            )

        descriptive_rows.append(
            {
                "Scarcity_pct": scarcity,
                "Controller": controller,
                "N_climate_years": int(len(d)),
                "Yield_retention_mean_pct":
                    float(d["Yield_retention_pct"].mean()),
                "Yield_retention_SD_pct":
                    float(d["Yield_retention_pct"].std(ddof=1)),
                "Worst_field_retention_mean_pct":
                    float(d["Worst_field_retention_pct"].mean()),
                "Worst_field_retention_SD_pct":
                    float(d["Worst_field_retention_pct"].std(ddof=1)),
                "Jain_mean":
                    float(d["Jain"].mean()),
                "Jain_SD":
                    float(d["Jain"].std(ddof=1)),
                "Water_productivity_mean":
                    float(d["Water_productivity"].mean()),
                "Water_productivity_SD":
                    float(d["Water_productivity"].std(ddof=1)),
            }
        )

descriptive_df = pd.DataFrame(descriptive_rows)
descriptive_df.to_csv(DESCRIPTIVE_FILE, index=False)


# Climate-specific summaries are descriptive only, per frozen 53I protocol.
climate_descriptive = (
    panel
    .groupby(
        ["Climate", "Scarcity_pct", "Controller"],
        as_index=False,
    )
    .agg(
        N_years=("Year", "size"),
        Yield_retention_mean_pct=("Yield_retention_pct", "mean"),
        Yield_retention_SD_pct=("Yield_retention_pct", "std"),
        Worst_field_retention_mean_pct=("Worst_field_retention_pct", "mean"),
        Worst_field_retention_SD_pct=("Worst_field_retention_pct", "std"),
        Jain_mean=("Jain", "mean"),
        Jain_SD=("Jain", "std"),
        Water_productivity_mean=("Water_productivity", "mean"),
        Water_productivity_SD=("Water_productivity", "std"),
    )
)

climate_descriptive.to_csv(
    CLIMATE_DESCRIPTIVE_FILE,
    index=False,
)


# ============================================================
# FRIEDMAN + PAIRWISE WILCOXON/HOLM
# ============================================================

friedman_rows = []
pairwise_rows = []

for scarcity in SCARCITY_LEVELS:
    d_s = panel[
        np.isclose(panel["Scarcity_pct"], scarcity)
    ].copy()

    for metric_name, metric_col in PRIMARY_METRICS.items():
        wide = (
            d_s
            .pivot(
                index=["Climate", "Year"],
                columns="Controller",
                values=metric_col,
            )
            .sort_index()
        )

        if len(wide) != 21:
            raise RuntimeError(
                f"{scarcity}% {metric_name}: expected 21 paired "
                f"climate-years, got {len(wide)}."
            )

        wide = wide[CONTROLLERS]

        if wide.isna().any().any():
            raise RuntimeError(
                f"Missing paired data for {scarcity}% {metric_name}."
            )

        equal = wide["Equal"].to_numpy(dtype=float)
        priority = wide["Priority"].to_numpy(dtype=float)
        ppo = wide["Optimized PPO"].to_numpy(dtype=float)

        friedman = friedmanchisquare(
            equal,
            priority,
            ppo,
        )

        friedman_rows.append(
            {
                "Scarcity_pct": scarcity,
                "Metric": metric_name,
                "N_climate_years": 21,
                "Friedman_statistic":
                    float(friedman.statistic),
                "P_value":
                    float(friedman.pvalue),
                "Significant_alpha_0_05":
                    bool(friedman.pvalue < 0.05),
            }
        )

        family_rows = []

        for method_a, method_b in PAIR_DEFINITIONS:
            a = wide[method_a].to_numpy(dtype=float)
            b = wide[method_b].to_numpy(dtype=float)

            wilcox = safe_wilcoxon(a, b)
            difference = a - b

            family_rows.append(
                {
                    "Scarcity_pct": scarcity,
                    "Metric": metric_name,
                    "Method_A": method_a,
                    "Method_B": method_b,
                    "N_pairs": 21,
                    "N_nonzero_differences":
                        int(wilcox["N_nonzero"]),
                    "Mean_A":
                        float(np.mean(a)),
                    "Mean_B":
                        float(np.mean(b)),
                    "Mean_difference_A_minus_B":
                        float(np.mean(difference)),
                    "Median_difference_A_minus_B":
                        float(np.median(difference)),
                    "Wilcoxon_statistic":
                        float(wilcox["Statistic"]),
                    "P_raw":
                        float(wilcox["P_raw"]),
                    "Rank_biserial_A_minus_B":
                        float(
                            paired_rank_biserial(a, b)
                        ),
                }
            )

        adjusted_ps = holm_adjust(
            [row["P_raw"] for row in family_rows]
        )

        for row, p_holm in zip(
            family_rows,
            adjusted_ps,
        ):
            row["P_Holm"] = float(p_holm)
            row["Significant_after_Holm_0_05"] = bool(
                p_holm < 0.05
            )
            pairwise_rows.append(row)


friedman_df = pd.DataFrame(friedman_rows)
pairwise_df = pd.DataFrame(pairwise_rows)

friedman_df.to_csv(FRIEDMAN_FILE, index=False)
pairwise_df.to_csv(PAIRWISE_FILE, index=False)


if len(friedman_df) != 12:
    raise RuntimeError(
        f"Expected 12 Friedman tests, got {len(friedman_df)}."
    )

if len(pairwise_df) != 36:
    raise RuntimeError(
        f"Expected 36 pairwise tests, got {len(pairwise_df)}."
    )


integrity = {
    "milestone": "53K",
    "status": "COMPLETE",
    "controller_final_test_already_open_before_analysis": True,
    "controller_tuning_after_test": False,
    "ppo_training": False,
    "ppo_model_selection": False,
    "ppo_seed_selection": False,
    "primary_experimental_unit": "climate-year",
    "paired_climate_years_per_scarcity": 21,
    "ppo_seed_pseudoreplication": False,
    "ppo_seeds_averaged_before_inference": True,
    "primary_metrics": list(PRIMARY_METRICS.keys()),
    "scarcity_levels_pct": SCARCITY_LEVELS,
    "friedman_tests": int(len(friedman_df)),
    "pairwise_tests": int(len(pairwise_df)),
    "pairwise_test": "two-sided paired Wilcoxon signed-rank",
    "multiple_testing_correction":
        "Holm across the three controller pairs within each metric x scarcity family",
    "effect_size":
        "paired rank-biserial correlation; positive means Method A > Method B",
    "alpha": 0.05,
    "climate_specific_inference_added": False,
    "climate_specific_results": "descriptive only",
}

with open(
    INTEGRITY_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(integrity, f, indent=2)


print()
print("#" * 78)
print("STATISTICAL ANALYSIS COMPLETE")
print("#" * 78)

print()
print("FINAL DESCRIPTIVE RESULTS")
print(
    "(mean ± SD across 21 paired final-test climate-years; "
    "PPO seeds averaged within climate-year first)"
)
print()

display_cols = [
    "Scarcity_pct",
    "Controller",
    "Yield_retention_mean_pct",
    "Yield_retention_SD_pct",
    "Worst_field_retention_mean_pct",
    "Worst_field_retention_SD_pct",
    "Jain_mean",
    "Jain_SD",
    "Water_productivity_mean",
    "Water_productivity_SD",
]

print(
    descriptive_df[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("=" * 78)
print("FRIEDMAN TESTS")
print("=" * 78)

print(
    friedman_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

print()
print("=" * 78)
print("PAIRWISE WILCOXON + HOLM")
print("=" * 78)

pair_display_cols = [
    "Scarcity_pct",
    "Metric",
    "Method_A",
    "Method_B",
    "Mean_difference_A_minus_B",
    "N_nonzero_differences",
    "P_raw",
    "P_Holm",
    "Rank_biserial_A_minus_B",
    "Significant_after_Holm_0_05",
]

print(
    pairwise_df[
        pair_display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

print()
print("=" * 78)
print("40% SCARCITY — PRIMARY PAIRWISE RESULTS")
print("=" * 78)

pair_40 = pairwise_df[
    np.isclose(
        pairwise_df["Scarcity_pct"],
        40.0,
    )
].copy()

print(
    pair_40[
        pair_display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

print()
print("=" * 78)
print("STATISTICAL INTEGRITY")
print("=" * 78)

print("Experimental unit           : climate-year")
print("Paired observations/scarcity: 21")
print("PPO seeds averaged first    : YES")
print("Seed pseudoreplication      : NO")
print("Friedman tests              : 12")
print("Pairwise Wilcoxon tests     : 36")
print(
    "Holm family                 : 3 controller pairs "
    "within metric x scarcity"
)
print(
    "Climate-specific inference  : NO "
    "(descriptive only)"
)
print("Controller tuning after test: NO")

print()
print("Saved:")
print(f"  {DESCRIPTIVE_FILE}")
print(f"  {CLIMATE_DESCRIPTIVE_FILE}")
print(f"  {FRIEDMAN_FILE}")
print(f"  {PAIRWISE_FILE}")
print(f"  {INTEGRITY_FILE}")

print()
print(
    "NEXT: interpret the final multi-climate results, "
    "including climate-specific behavior and binding-water "
    "diagnostics, then update the manuscript without any "
    "further controller tuning."
)
