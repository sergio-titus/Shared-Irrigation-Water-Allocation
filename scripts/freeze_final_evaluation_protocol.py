# ============================================================
# FINAL EVALUATION PROTOCOL FREEZE
# FREEZE COMMON FINAL-EVALUATION MANIFEST
#
# NO CONTROLLER EVALUATION.
# NO PPO INFERENCE.
# NO EQUAL/PRIORITY RUNS.
#
# This milestone freezes the exact common final-test panel
# before any controller performance is observed.
# ============================================================

from pathlib import Path
import hashlib
import json
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

TRAINING_SPEC_FILE = (
    ROOT
    / "training_protocol"
    / "multiclimate_ppo_training_spec_frozen.json"
)

HORIZON_SELECTION_FILE = (
    ROOT
    / "horizon_reconfirmation"
    / "ppo_horizon_selection_horizon_selection.json"
)

FINAL_TRAINING_SUMMARY_FILE = (
    ROOT
    / "final_training"
    / "ppo_final_training_final_training_summary.csv"
)

FINAL_TRAINING_INTEGRITY_FILE = (
    ROOT
    / "final_training"
    / "ppo_final_training_final_training_integrity.json"
)

FINAL_REFERENCE_FILE = (
    ROOT
    / "final_test_reference"
    / "final_test_reference_by_climate_year.csv"
)

FINAL_REFERENCE_INTEGRITY_FILE = (
    ROOT
    / "final_test_reference"
    / "final_test_reference_final_test_reference_integrity.json"
)

OUTPUT_DIR = (
    ROOT
    / "final_evaluation_protocol"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "final_evaluation_protocol_final_evaluation_manifest_frozen.json"
)

CASE_FILE = (
    OUTPUT_DIR
    / "final_evaluation_protocol_final_evaluation_cases.csv"
)

SEED_FILE = (
    OUTPUT_DIR
    / "final_evaluation_protocol_final_ppo_seed_manifest.csv"
)


# ============================================================
# 2. HELPERS
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# 3. REQUIRED FILE CHECKS
# ============================================================

required_files = [
    TRAINING_SPEC_FILE,
    HORIZON_SELECTION_FILE,
    FINAL_TRAINING_SUMMARY_FILE,
    FINAL_TRAINING_INTEGRITY_FILE,
    FINAL_REFERENCE_FILE,
    FINAL_REFERENCE_INTEGRITY_FILE,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


# ============================================================
# 4. FROZEN FINAL-EVALUATION DESIGN
# ============================================================

CLIMATES = [
    "Tunis",
    "Niamey",
    "Cotonou",
]

FINAL_TEST_YEARS = list(
    range(
        2019,
        2026,
    )
)

SCARCITY_LEVELS = [
    1.00,
    0.60,
    0.40,
]

CONTROLLERS = [
    "Equal",
    "Priority",
    "Optimized PPO",
]

FINAL_PPO_SEEDS = [
    1103,
    2207,
    3319,
    4421,
    5527,
    6637,
    7741,
    8849,
    9967,
    11071,
]

SELECTED_HORIZON = 102400


# ============================================================
# 5. VERIFY FINAL PPO TRAINING
# ============================================================

training_df = pd.read_csv(
    FINAL_TRAINING_SUMMARY_FILE
)

training_df[
    "Seed"
] = training_df[
    "Seed"
].astype(int)

if len(training_df) != 10:

    raise RuntimeError(
        "Expected exactly 10 final PPO models."
    )

if set(
    training_df[
        "Seed"
    ].tolist()
) != set(
    FINAL_PPO_SEEDS
):

    raise RuntimeError(
        "Final PPO seed set does not match "
        "the frozen manifest."
    )

if not (
    training_df[
        "Selected_horizon"
    ].astype(int)
    == SELECTED_HORIZON
).all():

    raise RuntimeError(
        "Unexpected PPO horizon."
    )

if not (
    training_df[
        "Actual_timesteps"
    ].astype(int)
    == SELECTED_HORIZON
).all():

    raise RuntimeError(
        "Unexpected PPO actual timestep count."
    )

training_integrity = load_json(
    FINAL_TRAINING_INTEGRITY_FILE
)

if training_integrity.get(
    "all_final_seeds_retained"
) is not True:

    raise RuntimeError(
        "Final training integrity does not certify "
        "retention of all PPO seeds."
    )

if training_integrity.get(
    "seed_selection"
) is not False:

    raise RuntimeError(
        "Seed selection unexpectedly reported."
    )

if training_integrity.get(
    "model_selection_among_final_seeds"
) is not False:

    raise RuntimeError(
        "Model selection unexpectedly reported."
    )


# ============================================================
# 6. VERIFY FINAL REFERENCE PANEL
# ============================================================

reference_df = pd.read_csv(
    FINAL_REFERENCE_FILE
)

required_reference_columns = {
    "Climate",
    "Year",
    "Total reference yield (tonne/ha)",
    "Total reference irrigation (mm)",
}

missing_reference_columns = (
    required_reference_columns
    - set(
        reference_df.columns
    )
)

if missing_reference_columns:

    raise RuntimeError(
        "Final reference library missing columns: "
        f"{sorted(missing_reference_columns)}"
    )

reference_df[
    "Year"
] = reference_df[
    "Year"
].astype(int)

if len(reference_df) != 21:

    raise RuntimeError(
        f"Expected 21 climate-year reference rows, "
        f"got {len(reference_df)}."
    )

expected_pairs = {
    (
        climate,
        year,
    )
    for climate in CLIMATES
    for year in FINAL_TEST_YEARS
}

actual_pairs = set(
    zip(
        reference_df[
            "Climate"
        ],
        reference_df[
            "Year"
        ],
    )
)

if actual_pairs != expected_pairs:

    raise RuntimeError(
        "Final reference climate-year panel "
        "does not match the frozen 21-case panel."
    )

reference_integrity = load_json(
    FINAL_REFERENCE_INTEGRITY_FILE
)

if reference_integrity.get(
    "controller_performance_seen"
) is not False:

    raise RuntimeError(
        "Controller performance was unexpectedly "
        "reported during reference generation."
    )

if reference_integrity.get(
    "controller_final_test_opened"
) is not False:

    raise RuntimeError(
        "Controller final test was unexpectedly opened "
        "before manifest freeze."
    )


# ============================================================
# 7. BUILD EXACT FINAL-EVALUATION CASE TABLE
# ============================================================

rows = []

case_id = 0

for scarcity in SCARCITY_LEVELS:

    for climate in CLIMATES:

        for year in FINAL_TEST_YEARS:

            case_id += 1

            ref_row = reference_df[
                (
                    reference_df[
                        "Climate"
                    ]
                    == climate
                )
                &
                (
                    reference_df[
                        "Year"
                    ]
                    == year
                )
            ]

            if len(ref_row) != 1:

                raise RuntimeError(
                    f"Expected one reference row for "
                    f"{climate} {year}."
                )

            ref_row = ref_row.iloc[0]

            reference_irrigation = float(
                ref_row[
                    "Total reference irrigation (mm)"
                ]
            )

            reference_yield = float(
                ref_row[
                    "Total reference yield (tonne/ha)"
                ]
            )

            seasonal_budget = (
                scarcity
                * reference_irrigation
            )

            rows.append(
                {
                    "Case_ID":
                        int(
                            case_id
                        ),

                    "Climate":
                        climate,

                    "Year":
                        int(
                            year
                        ),

                    "Scarcity_fraction":
                        float(
                            scarcity
                        ),

                    "Scarcity_pct":
                        float(
                            scarcity
                            * 100.0
                        ),

                    "Reference_yield_t_ha":
                        reference_yield,

                    "Reference_irrigation_mm":
                        reference_irrigation,

                    "Seasonal_budget_mm":
                        float(
                            seasonal_budget
                        ),
                }
            )


case_df = pd.DataFrame(
    rows
)


if len(case_df) != 63:

    raise RuntimeError(
        f"Expected 63 unique final-evaluation cases, "
        f"got {len(case_df)}."
    )


if case_df.duplicated(
    subset=[
        "Climate",
        "Year",
        "Scarcity_fraction",
    ]
).any():

    raise RuntimeError(
        "Duplicate final-evaluation case."
    )


# ============================================================
# 8. FREEZE METRICS AND ANALYSIS RULES
# ============================================================

PRIMARY_METRICS = [
    "Yield retention",
    "Worst-field retention",
    "Jain fairness",
    "Water productivity",
]

SUPPORTING_METRICS = [
    "Budget utilization",
    "Decision latency",
    "Request days",
    "Competition days",
    "Irrigation days",
]

ANALYSIS_RULES = {

    "primary_experimental_unit":
        "climate-year",

    "ppo_seed_handling":
        (
            "For each climate-year and scarcity level, "
            "average the results of all 10 frozen PPO seeds "
            "before controller-level inferential statistics."
        ),

    "ppo_seed_pseudoreplication":
        False,

    "all_ppo_seeds_retained":
        True,

    "controller_comparisons": [
        "Equal vs Priority",
        "Equal vs Optimized PPO",
        "Priority vs Optimized PPO",
    ],

    "overall_panel_analysis":
        (
            "Compare controllers on the 21 paired climate-years "
            "at each scarcity level."
        ),

    "climate_specific_analysis":
        (
            "Also summarize results separately for Tunis, "
            "Niamey, and Cotonou."
        ),

    "overall_inferential_test":
        (
            "Friedman test across the three controllers "
            "within each primary metric and scarcity level."
        ),

    "pairwise_test":
        (
            "Two-sided paired Wilcoxon signed-rank tests "
            "for the three controller pairs."
        ),

    "multiple_testing_correction":
        (
            "Holm correction across the three controller-pair "
            "comparisons within each metric x scarcity family."
        ),

    "effect_size":
        "paired rank-biserial correlation",

    "alpha":
        0.05,

    "binding_water_diagnostics":
        (
            "Report request days, competition days, budget "
            "utilization, and the fraction of climate-years "
            "where the seasonal water constraint is actually binding."
        ),

    "no_controller_selection_after_test":
        True,
}


# ============================================================
# 9. PPO SEED MANIFEST
# ============================================================

seed_df = pd.DataFrame(
    {
        "Seed":
            FINAL_PPO_SEEDS,

        "Retained":
            [
                True
                for _
                in FINAL_PPO_SEEDS
            ],

        "Eligible_for_final_test":
            [
                True
                for _
                in FINAL_PPO_SEEDS
            ],

        "Seed_selection":
            [
                False
                for _
                in FINAL_PPO_SEEDS
            ],
    }
)


# ============================================================
# 10. BUILD MANIFEST
# ============================================================

manifest = {

    "milestone":
        "53I",

    "status":
        "FINAL_EVALUATION_MANIFEST_FROZEN",

    "climates":
        CLIMATES,

    "final_test_years":
        FINAL_TEST_YEARS,

    "scarcity_levels":
        SCARCITY_LEVELS,

    "controllers":
        CONTROLLERS,

    "number_of_climate_years":
        21,

    "number_of_scarcity_levels":
        3,

    "number_of_common_cases":
        63,

    "final_ppo_seeds":
        FINAL_PPO_SEEDS,

    "number_of_final_ppo_seeds":
        10,

    "selected_ppo_horizon":
        SELECTED_HORIZON,

    "primary_metrics":
        PRIMARY_METRICS,

    "supporting_metrics":
        SUPPORTING_METRICS,

    "analysis_rules":
        ANALYSIS_RULES,

    "controller_test_status_before_execution": {
        "ppo_inference_performed":
            False,

        "equal_evaluated":
            False,

        "priority_evaluated":
            False,

        "controller_performance_seen":
            False,

        "controller_final_test_opened":
            False,
    },

    "reference_status": {
        "final_test_weather_accessed":
            True,

        "reason":
            (
                "Deterministic fully irrigated reference "
                "library generated after final PPO training "
                "was frozen."
            ),
    },

    "source_hashes": {
        "training_spec_sha256":
            sha256_file(
                TRAINING_SPEC_FILE
            ),

        "horizon_selection_sha256":
            sha256_file(
                HORIZON_SELECTION_FILE
            ),

        "final_training_summary_sha256":
            sha256_file(
                FINAL_TRAINING_SUMMARY_FILE
            ),

        "final_training_integrity_sha256":
            sha256_file(
                FINAL_TRAINING_INTEGRITY_FILE
            ),

        "final_reference_sha256":
            sha256_file(
                FINAL_REFERENCE_FILE
            ),

        "final_reference_integrity_sha256":
            sha256_file(
                FINAL_REFERENCE_INTEGRITY_FILE
            ),
    },
}


# ============================================================
# 11. SAVE
# ============================================================

case_df.to_csv(
    CASE_FILE,
    index=False,
)

seed_df.to_csv(
    SEED_FILE,
    index=False,
)

with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# 12. TERMINAL REPORT
# ============================================================

print()
print("#" * 78)
print("FINAL EVALUATION PROTOCOL FREEZE")
print(
    "COMMON FINAL-EVALUATION MANIFEST FROZEN"
)
print("#" * 78)

print()
print("=" * 78)
print("FINAL TEST PANEL")
print("=" * 78)

print(
    f"Climates                  : "
    f"{CLIMATES}"
)

print(
    f"Years                     : "
    f"{FINAL_TEST_YEARS}"
)

print(
    f"Scarcity levels           : "
    f"{[int(x * 100) for x in SCARCITY_LEVELS]}%"
)

print(
    f"Climate-years             : 21"
)

print(
    f"Common climate-year-scarcity cases: "
    f"{len(case_df)}"
)

print(
    f"Controllers               : "
    f"{CONTROLLERS}"
)

print(
    f"PPO final seeds           : 10"
)

print(
    f"PPO horizon               : "
    f"{SELECTED_HORIZON:,}"
)


print()
print("=" * 78)
print("ANALYSIS FREEZE")
print("=" * 78)

print(
    "Primary experimental unit : climate-year"
)

print(
    "PPO seeds averaged first   : YES"
)

print(
    "Seed pseudoreplication      : NO"
)

print(
    "Primary metrics             : "
    "Yield retention, worst-field retention, "
    "Jain fairness, water productivity"
)

print(
    "Overall panel               : 21 paired climate-years"
)

print(
    "Climate-specific summaries  : YES"
)

print(
    "Friedman test               : YES"
)

print(
    "Pairwise Wilcoxon            : two-sided"
)

print(
    "Holm correction             : YES"
)

print(
    "Effect size                 : paired rank-biserial"
)

print(
    "Binding-water diagnostics   : YES"
)


print()
print("=" * 78)
print("PRE-EVALUATION INTEGRITY")
print("=" * 78)

print(
    "Final PPO models frozen      : YES"
)

print(
    "Final reference frozen       : YES"
)

print(
    "2019-2025 weather accessed   : YES "
    "(reference generation only)"
)

print(
    "PPO inference performed      : NO"
)

print(
    "Equal evaluated              : NO"
)

print(
    "Priority evaluated           : NO"
)

print(
    "Controller performance seen  : NO"
)

print(
    "Controller final test opened : NO"
)


print()
print("Saved:")

print(
    f"  {MANIFEST_FILE}"
)

print(
    f"  {CASE_FILE}"
)

print(
    f"  {SEED_FILE}"
)


print()
print("=" * 78)
print("FINAL EVALUATION PROTOCOL FREEZE COMPLETE")
print("=" * 78)

print()
print(
    "NEXT: execute the common final test for "
    "Equal, Priority, and all 10 frozen PPO seeds "
    "on the 63 frozen climate-year-scarcity cases."
)
