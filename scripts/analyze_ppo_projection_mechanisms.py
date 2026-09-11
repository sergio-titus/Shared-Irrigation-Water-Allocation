# ============================================================
# PPO PROJECTION-MECHANISM ANALYSIS
# PPO PROJECTION-MECHANISM DECOMPOSITION
#
# PURPOSE
# -------
# Use the already generated Milestone-55A step-level diagnostic
# to distinguish:
#
#   1. daily-capacity projection;
#   2. seasonal-budget projection while some water remains;
#   3. complete projection after the seasonal budget has
#      effectively reached zero.
#
# NO AQUACROP SIMULATION
# NO PPO INFERENCE
# NO PPO TRAINING
# NO CONTROLLER MODIFICATION
# NO MODEL/SEED SELECTION
# NO HYPOTHESIS TESTING
#
# This is a descriptive post-hoc mechanism diagnostic only.
# ============================================================

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = (
    Path("results")
    / "multiclimate_extension"
)

INPUT_DIR = (
    ROOT
    / "projection_diagnostic"
)

STEP_FILE = (
    INPUT_DIR
    / "ppo_projection_step_level.csv"
)

OUTPUT_DIR = INPUT_DIR

MECHANISM_STEP_FILE = (
    OUTPUT_DIR
    / "55B_PPO_projection_mechanism_step_level.csv"
)

CLIMATE_SCARCITY_FILE = (
    OUTPUT_DIR
    / "ppo_projection_mechanism_PPO_projection_mechanism_climate_scarcity.csv"
)

PROJECTED_ONLY_FILE = (
    OUTPUT_DIR
    / "ppo_projection_mechanism_PPO_projected_steps_by_mechanism.csv"
)

TUNIS40_FILE = (
    OUTPUT_DIR
    / "ppo_projection_mechanism_Tunis40_projection_mechanism.csv"
)

INTEGRITY_FILE = (
    OUTPUT_DIR
    / "ppo_projection_mechanism_PPO_projection_mechanism_integrity.json"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

EPS = 1e-9


# ============================================================
# 3. LOAD 55A
# ============================================================

start_time = time.perf_counter()

print()
print("=" * 78)
print("PPO PROJECTION-MECHANISM ANALYSIS")
print("PPO PROJECTION-MECHANISM DECOMPOSITION")
print("=" * 78)

if not STEP_FILE.exists():

    raise FileNotFoundError(
        f"Missing Milestone-55A file: "
        f"{STEP_FILE}"
    )


df = pd.read_csv(
    STEP_FILE
)


print()
print(
    f"Loaded step-level rows: "
    f"{len(df):,}"
)


# ============================================================
# 4. REQUIRED-COLUMN CHECK
# ============================================================

required_columns = [
    "Climate",
    "Year",
    "Scarcity_fraction",
    "Scarcity_pct",
    "PPO_seed",
    "Step",
    "Request_day",
    "Competition_day",
    "Remaining_budget_before_mm",
    "Available_today_mm",
    "Total_request_mm",
    "Desired_total_mm",
    "Executed_total_mm",
    "Projected",
    "Projection_reason",
    "Projection_scale_factor",
    "Projection_absolute_L1_mm",
    "Projection_relative_L1",
    "Projection_relative_L1_pct",
    "Desired_water_removed_pct",
]


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    raise RuntimeError(
        "Missing required 55A columns: "
        f"{missing}"
    )


# ============================================================
# 5. NORMALIZE BOOLEAN COLUMNS
# ============================================================

def to_bool(series):

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


df[
    "Projected"
] = to_bool(
    df[
        "Projected"
    ]
)


df[
    "Request_day"
] = to_bool(
    df[
        "Request_day"
    ]
)


df[
    "Competition_day"
] = to_bool(
    df[
        "Competition_day"
    ]
)


# ============================================================
# 6. DEFINE BUDGET STATE BEFORE EACH DECISION
# ============================================================

df[
    "Budget_zero_before"
] = (
    df[
        "Remaining_budget_before_mm"
    ]
    <= EPS
)


df[
    "Budget_positive_before"
] = (
    df[
        "Remaining_budget_before_mm"
    ]
    > EPS
)


# A PPO "attempt" after exhaustion means:
# there is a positive field request and PPO's desired total
# is also positive, despite zero seasonal water remaining.

df[
    "Positive_PPO_desire"
] = (
    df[
        "Desired_total_mm"
    ]
    > EPS
)


df[
    "Post_exhaustion_PPO_attempt"
] = (
    df[
        "Budget_zero_before"
    ]
    &
    df[
        "Request_day"
    ]
    &
    df[
        "Positive_PPO_desire"
    ]
)


# ============================================================
# 7. MECHANISM CLASSIFICATION
# ============================================================

def classify_mechanism(row):

    projected = bool(
        row[
            "Projected"
        ]
    )

    budget_zero = bool(
        row[
            "Budget_zero_before"
        ]
    )

    reason = str(
        row[
            "Projection_reason"
        ]
    )

    if not projected:

        return "no_projection"

    if budget_zero:

        return "post_budget_exhaustion_zeroing"

    if reason == "daily_capacity":

        return "daily_capacity_rescaling"

    if reason == "seasonal_budget":

        return "seasonal_budget_rescaling_positive_budget"

    return "other_boundary_projection"


df[
    "Projection_mechanism"
] = df.apply(
    classify_mechanism,
    axis=1,
)


# ============================================================
# 8. BASIC CONSISTENCY CHECKS
# ============================================================

post_exhaustion = df[
    df[
        "Projection_mechanism"
    ]
    ==
    "post_budget_exhaustion_zeroing"
]


if len(post_exhaustion) > 0:

    max_executed_after_zero = float(
        post_exhaustion[
            "Executed_total_mm"
        ].max()
    )

else:

    max_executed_after_zero = 0.0


if max_executed_after_zero > 1e-8:

    raise RuntimeError(
        "A post-budget-exhaustion step executed "
        "positive irrigation unexpectedly."
    )


projected_df = df[
    df[
        "Projected"
    ]
].copy()


classified_projected = df[
    df[
        "Projection_mechanism"
    ]
    !=
    "no_projection"
]


if len(projected_df) != len(
    classified_projected
):

    raise RuntimeError(
        "Projected-step mechanism classification "
        "is incomplete."
    )


# ============================================================
# 9. SAVE AUGMENTED STEP DATA
# ============================================================

df.to_csv(
    MECHANISM_STEP_FILE,
    index=False,
)


# ============================================================
# 10. PROJECTED-STEP MECHANISM COUNTS
# ============================================================

mechanism_counts = (
    projected_df
    .groupby(
        [
            "Climate",
            "Scarcity_pct",
            "Projection_mechanism",
        ],
        as_index=False,
    )
    .agg(
        Projected_steps=(
            "Step",
            "size",
        ),

        Desired_total_mm=(
            "Desired_total_mm",
            "sum",
        ),

        Executed_total_mm=(
            "Executed_total_mm",
            "sum",
        ),

        Removed_mm=(
            "Projection_absolute_L1_mm",
            "sum",
        ),

        Median_relative_projection_pct=(
            "Projection_relative_L1_pct",
            "median",
        ),

        Mean_relative_projection_pct=(
            "Projection_relative_L1_pct",
            "mean",
        ),
    )
)


# Total projected steps within each climate × scarcity
projected_totals = (
    mechanism_counts
    .groupby(
        [
            "Climate",
            "Scarcity_pct",
        ],
        as_index=False,
    )[
        "Projected_steps"
    ]
    .sum()
    .rename(
        columns={
            "Projected_steps":
                "All_projected_steps"
        }
    )
)


mechanism_counts = (
    mechanism_counts
    .merge(
        projected_totals,
        on=[
            "Climate",
            "Scarcity_pct",
        ],
        how="left",
        validate="many_to_one",
    )
)


mechanism_counts[
    "Share_of_projected_steps_pct"
] = (
    100.0
    *
    mechanism_counts[
        "Projected_steps"
    ]
    /
    mechanism_counts[
        "All_projected_steps"
    ]
)


mechanism_counts.to_csv(
    PROJECTED_ONLY_FILE,
    index=False,
)


# ============================================================
# 11. CLIMATE × SCARCITY SUMMARY
# ============================================================

def summarize_case(group):

    n_steps = len(
        group
    )

    request = group[
        "Request_day"
    ]

    competition = group[
        "Competition_day"
    ]

    projected = group[
        "Projected"
    ]

    budget_zero = group[
        "Budget_zero_before"
    ]

    post_attempt = group[
        "Post_exhaustion_PPO_attempt"
    ]

    daily_projection = (
        group[
            "Projection_mechanism"
        ]
        ==
        "daily_capacity_rescaling"
    )

    positive_budget_projection = (
        group[
            "Projection_mechanism"
        ]
        ==
        "seasonal_budget_rescaling_positive_budget"
    )

    zeroing = (
        group[
            "Projection_mechanism"
        ]
        ==
        "post_budget_exhaustion_zeroing"
    )

    projected_before_exhaustion = (
        projected
        &
        (~budget_zero)
    )

    projected_before_df = group[
        projected_before_exhaustion
    ]

    projected_all_df = group[
        projected
    ]


    total_projected = int(
        projected.sum()
    )

    total_zeroing = int(
        zeroing.sum()
    )


    return pd.Series(
        {
            "N_steps":
                int(
                    n_steps
                ),

            "Request_steps":
                int(
                    request.sum()
                ),

            "Competition_steps":
                int(
                    competition.sum()
                ),

            "Projected_steps":
                total_projected,

            "Projection_frequency_all_steps_pct":
                (
                    100.0
                    * projected.mean()
                ),

            "Projection_frequency_request_steps_pct":
                (
                    100.0
                    * projected[
                        request
                    ].mean()
                    if request.sum() > 0
                    else 0.0
                ),

            "Daily_capacity_projection_steps":
                int(
                    daily_projection.sum()
                ),

            "Positive_budget_seasonal_projection_steps":
                int(
                    positive_budget_projection.sum()
                ),

            "Post_budget_exhaustion_zeroing_steps":
                total_zeroing,

            "Post_exhaustion_share_of_projected_steps_pct":
                (
                    100.0
                    * total_zeroing
                    / total_projected
                    if total_projected > 0
                    else 0.0
                ),

            "Zero_budget_steps":
                int(
                    budget_zero.sum()
                ),

            "Post_exhaustion_PPO_attempt_steps":
                int(
                    post_attempt.sum()
                ),

            "Post_exhaustion_attempt_frequency_pct":
                (
                    100.0
                    * post_attempt[
                        budget_zero
                    ].mean()
                    if budget_zero.sum() > 0
                    else 0.0
                ),

            "Median_projection_all_projected_pct":
                (
                    float(
                        projected_all_df[
                            "Projection_relative_L1_pct"
                        ].median()
                    )
                    if len(
                        projected_all_df
                    ) > 0
                    else 0.0
                ),

            "Median_projection_before_exhaustion_pct":
                (
                    float(
                        projected_before_df[
                            "Projection_relative_L1_pct"
                        ].median()
                    )
                    if len(
                        projected_before_df
                    ) > 0
                    else 0.0
                ),

            "Q1_projection_before_exhaustion_pct":
                (
                    float(
                        projected_before_df[
                            "Projection_relative_L1_pct"
                        ].quantile(
                            0.25
                        )
                    )
                    if len(
                        projected_before_df
                    ) > 0
                    else 0.0
                ),

            "Q3_projection_before_exhaustion_pct":
                (
                    float(
                        projected_before_df[
                            "Projection_relative_L1_pct"
                        ].quantile(
                            0.75
                        )
                    )
                    if len(
                        projected_before_df
                    ) > 0
                    else 0.0
                ),

            "Total_removed_all_projection_mm":
                float(
                    projected_all_df[
                        "Projection_absolute_L1_mm"
                    ].sum()
                ),

            "Removed_before_exhaustion_mm":
                float(
                    projected_before_df[
                        "Projection_absolute_L1_mm"
                    ].sum()
                ),

            "Removed_post_exhaustion_mm":
                float(
                    group.loc[
                        zeroing,
                        "Projection_absolute_L1_mm",
                    ].sum()
                ),

            "Desired_post_exhaustion_mm":
                float(
                    group.loc[
                        post_attempt,
                        "Desired_total_mm",
                    ].sum()
                ),
        }
    )


climate_scarcity = (
    df
    .groupby(
        [
            "Climate",
            "Scarcity_pct",
        ],
        sort=True,
    )
    .apply(
        summarize_case
    )
    .reset_index()
)


climate_scarcity.to_csv(
    CLIMATE_SCARCITY_FILE,
    index=False,
)


# ============================================================
# 12. TUNIS 40% DETAILED SUMMARY BY SEED
# ============================================================

tunis40 = df[
    (
        df[
            "Climate"
        ]
        ==
        "Tunis"
    )
    &
    (
        np.isclose(
            df[
                "Scarcity_pct"
            ],
            40.0,
        )
    )
].copy()


if len(tunis40) == 0:

    raise RuntimeError(
        "No Tunis 40% rows found."
    )


tunis40_seed = (
    tunis40
    .groupby(
        "PPO_seed",
        sort=True,
    )
    .apply(
        summarize_case
    )
    .reset_index()
)


tunis40_seed.to_csv(
    TUNIS40_FILE,
    index=False,
)


# ============================================================
# 13. OVERALL MECHANISM COUNTS
# ============================================================

overall_mechanisms = (
    projected_df[
        "Projection_mechanism"
    ]
    .value_counts()
)


total_projected_steps = int(
    len(
        projected_df
    )
)


post_exhaustion_steps = int(
    (
        projected_df[
            "Projection_mechanism"
        ]
        ==
        "post_budget_exhaustion_zeroing"
    ).sum()
)


daily_capacity_steps = int(
    (
        projected_df[
            "Projection_mechanism"
        ]
        ==
        "daily_capacity_rescaling"
    ).sum()
)


positive_budget_seasonal_steps = int(
    (
        projected_df[
            "Projection_mechanism"
        ]
        ==
        "seasonal_budget_rescaling_positive_budget"
    ).sum()
)


# ============================================================
# 14. TUNIS-40 AGGREGATE VALUES
# ============================================================

t40_projected = tunis40[
    tunis40[
        "Projected"
    ]
]


t40_zeroing = tunis40[
    tunis40[
        "Projection_mechanism"
    ]
    ==
    "post_budget_exhaustion_zeroing"
]


t40_before = tunis40[
    tunis40[
        "Projected"
    ]
    &
    (
        ~tunis40[
            "Budget_zero_before"
        ]
    )
]


t40_total_projected = len(
    t40_projected
)


t40_zeroing_count = len(
    t40_zeroing
)


t40_zeroing_share = (
    100.0
    * t40_zeroing_count
    / t40_total_projected
    if t40_total_projected > 0
    else 0.0
)


t40_before_median = (
    float(
        t40_before[
            "Projection_relative_L1_pct"
        ].median()
    )
    if len(
        t40_before
    ) > 0
    else 0.0
)


t40_before_q1 = (
    float(
        t40_before[
            "Projection_relative_L1_pct"
        ].quantile(
            0.25
        )
    )
    if len(
        t40_before
    ) > 0
    else 0.0
)


t40_before_q3 = (
    float(
        t40_before[
            "Projection_relative_L1_pct"
        ].quantile(
            0.75
        )
    )
    if len(
        t40_before
    ) > 0
    else 0.0
)


# ============================================================
# 15. INTEGRITY RECORD
# ============================================================

runtime_seconds = (
    time.perf_counter()
    - start_time
)


integrity = {

    "milestone":
        "55B",

    "status":
        "COMPLETE",

    "input":
        str(
            STEP_FILE
        ),

    "input_rows":
        int(
            len(
                df
            )
        ),

    "analysis_type":
        (
            "Post-hoc descriptive decomposition "
            "of Milestone-55A projection events"
        ),

    "simulation_performed":
        False,

    "ppo_inference_performed":
        False,

    "ppo_training_performed":
        False,

    "controller_modified":
        False,

    "model_selection":
        False,

    "seed_selection":
        False,

    "hypothesis_tests_performed":
        False,

    "total_projected_steps":
        total_projected_steps,

    "daily_capacity_projection_steps":
        daily_capacity_steps,

    "positive_budget_seasonal_projection_steps":
        positive_budget_seasonal_steps,

    "post_budget_exhaustion_zeroing_steps":
        post_exhaustion_steps,

    "tunis40_total_projected_steps":
        int(
            t40_total_projected
        ),

    "tunis40_post_exhaustion_zeroing_steps":
        int(
            t40_zeroing_count
        ),

    "tunis40_post_exhaustion_share_pct":
        float(
            t40_zeroing_share
        ),

    "tunis40_pre_exhaustion_projection_median_pct":
        float(
            t40_before_median
        ),

    "tunis40_pre_exhaustion_projection_q1_pct":
        float(
            t40_before_q1
        ),

    "tunis40_pre_exhaustion_projection_q3_pct":
        float(
            t40_before_q3
        ),

    "maximum_executed_water_after_zero_budget_mm":
        float(
            max_executed_after_zero
        ),

    "runtime_seconds":
        float(
            runtime_seconds
        ),
}


with open(
    INTEGRITY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        integrity,
        f,
        indent=2,
    )


# ============================================================
# 16. TERMINAL REPORT
# ============================================================

print()
print("=" * 78)
print("OVERALL PROJECTED-STEP MECHANISMS")
print("=" * 78)

print(
    overall_mechanisms.to_string()
)

print()
print(
    f"Total projected steps                 : "
    f"{total_projected_steps:,}"
)

print(
    f"Daily-capacity rescaling              : "
    f"{daily_capacity_steps:,}"
)

print(
    f"Seasonal rescaling, positive budget   : "
    f"{positive_budget_seasonal_steps:,}"
)

print(
    f"Post-budget-exhaustion zeroing        : "
    f"{post_exhaustion_steps:,}"
)


print()
print("=" * 78)
print("CLIMATE × SCARCITY DECOMPOSITION")
print("=" * 78)

display_columns = [
    "Climate",
    "Scarcity_pct",
    "Projected_steps",
    "Daily_capacity_projection_steps",
    "Positive_budget_seasonal_projection_steps",
    "Post_budget_exhaustion_zeroing_steps",
    "Post_exhaustion_share_of_projected_steps_pct",
    "Median_projection_all_projected_pct",
    "Median_projection_before_exhaustion_pct",
]


print(
    climate_scarcity[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


print()
print("=" * 78)
print("TUNIS 40% — KEY MECHANISM")
print("=" * 78)

print(
    f"Projected steps                       : "
    f"{t40_total_projected:,}"
)

print(
    f"Post-exhaustion zeroing steps         : "
    f"{t40_zeroing_count:,}"
)

print(
    f"Share due to post-exhaustion zeroing  : "
    f"{t40_zeroing_share:.3f}%"
)

print(
    "Projection severity BEFORE exhaustion:"
)

print(
    f"  median                              : "
    f"{t40_before_median:.3f}%"
)

print(
    f"  IQR                                 : "
    f"[{t40_before_q1:.3f}, "
    f"{t40_before_q3:.3f}]%"
)


print()
print("=" * 78)
print("TUNIS 40% BY PPO SEED")
print("=" * 78)

seed_display = [
    "PPO_seed",
    "Projected_steps",
    "Daily_capacity_projection_steps",
    "Positive_budget_seasonal_projection_steps",
    "Post_budget_exhaustion_zeroing_steps",
    "Post_exhaustion_share_of_projected_steps_pct",
    "Median_projection_before_exhaustion_pct",
    "Removed_before_exhaustion_mm",
    "Removed_post_exhaustion_mm",
]


print(
    tunis40_seed[
        seed_display
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


print()
print("=" * 78)
print("INTEGRITY")
print("=" * 78)

print(
    "AquaCrop simulation performed : NO"
)

print(
    "PPO inference performed       : NO"
)

print(
    "PPO training performed        : NO"
)

print(
    "Controller modified           : NO"
)

print(
    "Model/seed selection          : NO"
)

print(
    "Hypothesis tests              : NO"
)

print(
    "55A data only                 : YES"
)

print(
    f"Maximum irrigation after zero "
    f"budget: "
    f"{max_executed_after_zero:.3e} mm"
)

print(
    f"Runtime                       : "
    f"{runtime_seconds:.2f} s"
)


print()
print("Saved:")

for path in [
    MECHANISM_STEP_FILE,
    CLIMATE_SCARCITY_FILE,
    PROJECTED_ONLY_FILE,
    TUNIS40_FILE,
    INTEGRITY_FILE,
]:

    print(
        f"  {path}"
    )


print()
print("=" * 78)
print("PPO PROJECTION-MECHANISM ANALYSIS COMPLETE")
print("=" * 78)