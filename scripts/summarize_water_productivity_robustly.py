# ============================================================
# ROBUST WATER-PRODUCTIVITY SUMMARY
# ROBUST WATER-PRODUCTIVITY REPORTING
#
# PURPOSE
# -------
# Reanalyse WATER PRODUCTIVITY (WP) descriptively using the
# already frozen final-test controller panel.
#
# Motivation:
# WP can be strongly right-skewed because very small positive
# irrigation denominators can produce large ratios, while
# zero-irrigation episodes are represented as WP = 0.
#
# Therefore:
#   - retain the frozen data;
#   - DO NOT rerun simulations;
#   - DO NOT retrain PPO;
#   - DO NOT add new hypothesis tests;
#   - report WP primarily as median [Q1, Q3].
#
# Other agronomic metrics remain mean ± SD.
# ============================================================


from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = (
    Path("results")
    / "multiclimate_extension"
    / "final_evaluation"
    / "controller_evaluation_final_controller_panel.csv"
)


OUTPUT_DIR = (
    Path("results")
    / "multiclimate_extension"
    / "water_productivity_robust_summary"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OVERALL_FILE = (
    OUTPUT_DIR
    / "water_productivity_WP_controller_scarcity_summary.csv"
)


CLIMATE_FILE = (
    OUTPUT_DIR
    / "water_productivity_WP_climate_controller_scarcity_summary.csv"
)


COMPARISON_FILE = (
    OUTPUT_DIR
    / "water_productivity_WP_mean_vs_median_comparison.csv"
)


MANUSCRIPT_FILE = (
    OUTPUT_DIR
    / "water_productivity_WP_manuscript_table.csv"
)


SUMMARY_JSON = (
    OUTPUT_DIR
    / "water_productivity_WP_summary.json"
)


# Numerical threshold only for identifying exact/effectively
# zero WP values.
ZERO_TOL = 1e-12


# ============================================================
# 2. LOAD FROZEN PANEL
# ============================================================

start_time = time.perf_counter()


print()

print("=" * 78)

print("ROBUST WATER-PRODUCTIVITY SUMMARY")

print("ROBUST WATER-PRODUCTIVITY DESCRIPTIVE SUMMARY")

print("=" * 78)


if not INPUT_FILE.is_file():

    raise FileNotFoundError(
        f"Frozen final panel not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print()

print(
    f"Input file                : {INPUT_FILE}"
)

print(
    f"Rows loaded               : {len(df):,}"
)

print(
    f"Columns                   : {len(df.columns)}"
)


# ============================================================
# 3. ROBUST COLUMN IDENTIFICATION
# ============================================================

def find_column(
    dataframe,
    candidates,
    label,
):

    exact_map = {
        str(col).lower():
            col
        for col in dataframe.columns
    }


    # Exact case-insensitive match first.
    for candidate in candidates:

        key = candidate.lower()

        if key in exact_map:

            return exact_map[key]


    # Normalized match second.
    def normalize(text):

        return (
            str(text)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("%", "")
        )


    normalized_map = {
        normalize(col):
            col
        for col in dataframe.columns
    }


    for candidate in candidates:

        key = normalize(candidate)

        if key in normalized_map:

            return normalized_map[key]


    print()

    print(
        f"Could not identify {label} column."
    )

    print(
        "Available columns:"
    )

    for col in dataframe.columns:

        print(
            f"  - {col}"
        )


    raise KeyError(
        f"Missing required column: {label}"
    )


controller_col = find_column(
    df,
    [
        "Controller",
        "controller",
        "Method",
        "method",
    ],
    "controller",
)


climate_col = find_column(
    df,
    [
        "Climate",
        "climate",
        "Site",
        "site",
    ],
    "climate",
)


year_col = find_column(
    df,
    [
        "Year",
        "year",
    ],
    "year",
)


scarcity_col = find_column(
    df,
    [
        "Scarcity",
        "scarcity",
        "Scarcity_fraction",
        "scarcity_fraction",
        "Availability",
        "availability",
        "Water_availability",
        "water_availability",
    ],
    "scarcity",
)


wp_col = find_column(
    df,
    [
        "Water_productivity",
        "water_productivity",
        "WaterProductivity",
        "WP",
        "wp",
    ],
    "water productivity",
)


print()

print(
    "Detected columns:"
)

print(
    f"  Controller              : {controller_col}"
)

print(
    f"  Climate                 : {climate_col}"
)

print(
    f"  Year                    : {year_col}"
)

print(
    f"  Scarcity                : {scarcity_col}"
)

print(
    f"  Water productivity      : {wp_col}"
)


# ============================================================
# 4. CLEAN / STANDARDIZE
# ============================================================

work = df[
    [
        controller_col,
        climate_col,
        year_col,
        scarcity_col,
        wp_col,
    ]
].copy()


work.columns = [
    "Controller",
    "Climate",
    "Year",
    "Scarcity_raw",
    "Water_productivity",
]


work[
    "Water_productivity"
] = pd.to_numeric(
    work[
        "Water_productivity"
    ],
    errors="raise",
)


work[
    "Year"
] = pd.to_numeric(
    work[
        "Year"
    ],
    errors="raise",
).astype(int)


# ------------------------------------------------------------
# Normalize scarcity to percentages:
#
# supports either:
#   0.4 / 0.6 / 1.0
#
# or:
#   40 / 60 / 100
# ------------------------------------------------------------

scarcity_numeric = pd.to_numeric(
    work[
        "Scarcity_raw"
    ],
    errors="coerce",
)


if scarcity_numeric.isna().any():

    # Try strings such as "40%" or "100%".
    scarcity_numeric = pd.to_numeric(
        work[
            "Scarcity_raw"
        ]
        .astype(str)
        .str.replace(
            "%",
            "",
            regex=False,
        )
        .str.strip(),
        errors="raise",
    )


if float(
    scarcity_numeric.max()
) <= 1.01:

    scarcity_pct = (
        scarcity_numeric
        *
        100.0
    )

else:

    scarcity_pct = (
        scarcity_numeric
    )


work[
    "Scarcity_pct"
] = (
    scarcity_pct
    .round(6)
)


# Standardize controller names lightly.

controller_map = {
    "equal":
        "Equal",

    "priority":
        "Priority",

    "ppo":
        "PPO",

    "optimized ppo":
        "PPO",
}


work[
    "Controller"
] = (
    work[
        "Controller"
    ]
    .astype(str)
    .str.strip()
    .apply(
        lambda value:
            controller_map.get(
                value.lower(),
                value,
            )
    )
)


work[
    "Climate"
] = (
    work[
        "Climate"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# 5. INTEGRITY CHECKS
# ============================================================

if not np.isfinite(
    work[
        "Water_productivity"
    ].to_numpy(
        dtype=float
    )
).all():

    raise RuntimeError(
        "Non-finite water-productivity values detected."
    )


if (
    work[
        "Water_productivity"
    ]
    <
    -ZERO_TOL
).any():

    raise RuntimeError(
        "Negative water-productivity values detected."
    )


controllers = sorted(
    work[
        "Controller"
    ]
    .unique()
    .tolist()
)


climates = sorted(
    work[
        "Climate"
    ]
    .unique()
    .tolist()
)


scarcity_levels = sorted(
    work[
        "Scarcity_pct"
    ]
    .unique()
    .tolist()
)


print()

print(
    f"Controllers detected      : {controllers}"
)

print(
    f"Climates detected         : {climates}"
)

print(
    f"Scarcity levels detected  : {scarcity_levels}"
)


expected_controllers = {
    "Equal",
    "Priority",
    "PPO",
}


missing_controllers = (
    expected_controllers
    -
    set(controllers)
)


if missing_controllers:

    raise RuntimeError(
        "Expected controllers missing: "
        f"{sorted(missing_controllers)}"
    )


# ============================================================
# 6. SUMMARY FUNCTION
# ============================================================

def summarize_wp(
    group,
):

    values = (
        pd.to_numeric(
            group[
                "Water_productivity"
            ],
            errors="raise",
        )
        .to_numpy(
            dtype=np.float64
        )
    )


    n = int(
        len(values)
    )


    q1 = float(
        np.quantile(
            values,
            0.25,
        )
    )


    median = float(
        np.quantile(
            values,
            0.50,
        )
    )


    q3 = float(
        np.quantile(
            values,
            0.75,
        )
    )


    iqr = (
        q3
        -
        q1
    )


    zero_count = int(
        np.sum(
            np.abs(values)
            <=
            ZERO_TOL
        )
    )


    zero_pct = (
        100.0
        *
        zero_count
        /
        n
        if n > 0
        else np.nan
    )


    mean = float(
        np.mean(values)
    )


    sd = (
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        if n > 1
        else np.nan
    )


    return pd.Series(
        {

            "N":
                n,

            "WP_mean":
                mean,

            "WP_SD":
                sd,

            "WP_median":
                median,

            "WP_Q1":
                q1,

            "WP_Q3":
                q3,

            "WP_IQR":
                iqr,

            "WP_min":
                float(
                    np.min(values)
                ),

            "WP_max":
                float(
                    np.max(values)
                ),

            "WP_zero_count":
                zero_count,

            "WP_zero_pct":
                zero_pct,

            "Mean_minus_median":
                mean
                -
                median,

            "Mean_to_median_ratio":
                (
                    mean
                    /
                    median
                    if abs(median) > ZERO_TOL
                    else np.nan
                ),
        }
    )


# ============================================================
# 7. CONTROLLER × SCARCITY SUMMARY
# ============================================================

overall = (
    work
    .groupby(
        [
            "Scarcity_pct",
            "Controller",
        ],
        sort=True,
        observed=True,
    )
    .apply(
        summarize_wp,
        include_groups=False,
    )
    .reset_index()
)


# Stable controller ordering.

controller_order = {
    "Equal":
        0,

    "Priority":
        1,

    "PPO":
        2,
}


overall[
    "_controller_order"
] = (
    overall[
        "Controller"
    ]
    .map(
        controller_order
    )
    .fillna(99)
)


overall = (
    overall
    .sort_values(
        [
            "Scarcity_pct",
            "_controller_order",
        ]
    )
    .drop(
        columns=[
            "_controller_order"
        ]
    )
    .reset_index(
        drop=True
    )
)


overall.to_csv(
    OVERALL_FILE,
    index=False,
)


# ============================================================
# 8. CLIMATE × CONTROLLER × SCARCITY SUMMARY
# ============================================================

climate_summary = (
    work
    .groupby(
        [
            "Scarcity_pct",
            "Climate",
            "Controller",
        ],
        sort=True,
        observed=True,
    )
    .apply(
        summarize_wp,
        include_groups=False,
    )
    .reset_index()
)


climate_summary[
    "_controller_order"
] = (
    climate_summary[
        "Controller"
    ]
    .map(
        controller_order
    )
    .fillna(99)
)


climate_summary = (
    climate_summary
    .sort_values(
        [
            "Scarcity_pct",
            "Climate",
            "_controller_order",
        ]
    )
    .drop(
        columns=[
            "_controller_order"
        ]
    )
    .reset_index(
        drop=True
    )
)


climate_summary.to_csv(
    CLIMATE_FILE,
    index=False,
)


# ============================================================
# 9. OLD MEAN ± SD VS ROBUST MEDIAN [IQR]
# ============================================================

comparison = overall[
    [
        "Scarcity_pct",
        "Controller",
        "N",
        "WP_mean",
        "WP_SD",
        "WP_median",
        "WP_Q1",
        "WP_Q3",
        "WP_IQR",
        "WP_zero_count",
        "WP_zero_pct",
        "WP_min",
        "WP_max",
        "Mean_minus_median",
        "Mean_to_median_ratio",
    ]
].copy()


comparison[
    "Old_mean_SD"
] = comparison.apply(
    lambda row:
        (
            f"{row['WP_mean']:.6f} "
            f"± {row['WP_SD']:.6f}"
        ),
    axis=1,
)


comparison[
    "Robust_median_IQR"
] = comparison.apply(
    lambda row:
        (
            f"{row['WP_median']:.6f} "
            f"[{row['WP_Q1']:.6f}, "
            f"{row['WP_Q3']:.6f}]"
        ),
    axis=1,
)


comparison.to_csv(
    COMPARISON_FILE,
    index=False,
)


# ============================================================
# 10. MANUSCRIPT-READY TABLE
# ============================================================

manuscript = comparison[
    [
        "Scarcity_pct",
        "Controller",
        "N",
        "WP_median",
        "WP_Q1",
        "WP_Q3",
        "WP_zero_count",
        "WP_zero_pct",
    ]
].copy()


manuscript[
    "Water_productivity_median_IQR"
] = manuscript.apply(
    lambda row:
        (
            f"{row['WP_median']:.4f} "
            f"[{row['WP_Q1']:.4f}, "
            f"{row['WP_Q3']:.4f}]"
        ),
    axis=1,
)


manuscript[
    "Zero_WP"
] = manuscript.apply(
    lambda row:
        (
            f"{int(row['WP_zero_count'])}"
            f"/{int(row['N'])} "
            f"({row['WP_zero_pct']:.1f}%)"
        ),
    axis=1,
)


manuscript = manuscript[
    [
        "Scarcity_pct",
        "Controller",
        "Water_productivity_median_IQR",
        "Zero_WP",
    ]
]


manuscript.to_csv(
    MANUSCRIPT_FILE,
    index=False,
)


# ============================================================
# 11. TERMINAL REPORT — OVERALL
# ============================================================

print()

print("=" * 78)

print(
    "WP MEDIAN [Q1, Q3] — ALL CLIMATES"
)

print("=" * 78)


for scarcity in scarcity_levels:

    subset = overall[
        np.isclose(
            overall[
                "Scarcity_pct"
            ],
            scarcity,
        )
    ]


    print()

    print(
        f"{scarcity:.0f}% WATER AVAILABILITY"
    )


    for controller in [
        "Equal",
        "Priority",
        "PPO",
    ]:

        row = subset[
            subset[
                "Controller"
            ]
            ==
            controller
        ]


        if row.empty:

            continue


        row = row.iloc[0]


        print(
            f"{controller:10s} "
            f"median = "
            f"{row['WP_median']:.6f} "
            f"[{row['WP_Q1']:.6f}, "
            f"{row['WP_Q3']:.6f}] "
            f"| mean = "
            f"{row['WP_mean']:.6f} "
            f"± {row['WP_SD']:.6f} "
            f"| zeros = "
            f"{int(row['WP_zero_count'])}/"
            f"{int(row['N'])} "
            f"({row['WP_zero_pct']:.1f}%)"
        )


# ============================================================
# 12. CLIMATE-SPECIFIC REPORT
# ============================================================

print()

print("=" * 78)

print(
    "WP MEDIAN [Q1, Q3] — CLIMATE × SCARCITY"
)

print("=" * 78)


for scarcity in scarcity_levels:

    print()

    print(
        f"{scarcity:.0f}% WATER AVAILABILITY"
    )


    for climate in climates:

        print()

        print(
            f"  {climate}"
        )


        subset = climate_summary[
            (
                np.isclose(
                    climate_summary[
                        "Scarcity_pct"
                    ],
                    scarcity,
                )
            )
            &
            (
                climate_summary[
                    "Climate"
                ]
                ==
                climate
            )
        ]


        for controller in [
            "Equal",
            "Priority",
            "PPO",
        ]:

            row = subset[
                subset[
                    "Controller"
                ]
                ==
                controller
            ]


            if row.empty:

                continue


            row = row.iloc[0]


            print(
                f"    {controller:10s} "
                f"{row['WP_median']:.6f} "
                f"[{row['WP_Q1']:.6f}, "
                f"{row['WP_Q3']:.6f}] "
                f"| zeros "
                f"{int(row['WP_zero_count'])}/"
                f"{int(row['N'])}"
            )


# ============================================================
# 13. IDENTIFY STRONGEST MEAN-MEDIAN DISCREPANCIES
# ============================================================

ranked = comparison.copy()


ranked[
    "Absolute_mean_median_difference"
] = np.abs(
    ranked[
        "Mean_minus_median"
    ]
)


ranked = ranked.sort_values(
    "Absolute_mean_median_difference",
    ascending=False,
).reset_index(
    drop=True
)


print()

print("=" * 78)

print(
    "LARGEST MEAN–MEDIAN WP DISCREPANCIES"
)

print("=" * 78)


print(
    ranked[
        [
            "Scarcity_pct",
            "Controller",
            "WP_mean",
            "WP_median",
            "WP_Q1",
            "WP_Q3",
            "WP_max",
            "Mean_minus_median",
        ]
    ]
    .head(9)
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 14. SCIENTIFIC INTERPRETATION FLAGS
# ============================================================

ppo_rows = overall[
    overall[
        "Controller"
    ]
    ==
    "PPO"
].copy()


ppo_skew_flag = bool(
    (
        np.abs(
            ppo_rows[
                "WP_mean"
            ]
            -
            ppo_rows[
                "WP_median"
            ]
        )
        >
        0.05
    ).any()
)


any_zero_wp = bool(
    (
        overall[
            "WP_zero_count"
        ]
        >
        0
    ).any()
)


print()

print("=" * 78)

print(
    "DIAGNOSTIC INTERPRETATION"
)

print("=" * 78)


print(
    f"Any zero-WP episodes      : "
    f"{'YES' if any_zero_wp else 'NO'}"
)

print(
    f"Large PPO mean/median gap : "
    f"{'YES' if ppo_skew_flag else 'NO'}"
)


if ppo_skew_flag:

    interpretation = (
        "Water productivity is materially skewed for at least "
        "one PPO condition. Median [IQR] is therefore more "
        "representative than mean ± SD for descriptive reporting."
    )

else:

    interpretation = (
        "Median [IQR] remains preferable for WP because the "
        "metric contains structural zeros and can be sensitive "
        "to very small irrigation denominators, although the "
        "observed mean–median differences are limited."
    )


print()

print(
    interpretation
)


# ============================================================
# 15. SCIENTIFIC STATUS
# ============================================================

print()

print("=" * 78)

print(
    "SCIENTIFIC STATUS"
)

print("=" * 78)


print(
    "New AquaCrop simulations    : NO"
)

print(
    "PPO inference rerun          : NO"
)

print(
    "PPO retrained                : NO"
)

print(
    "Controller outputs modified  : NO"
)

print(
    "Original hypothesis tests    : UNCHANGED"
)

print(
    "New hypothesis tests         : NO"
)

print(
    "Frozen panel modified        : NO"
)

print(
    "Descriptive summary changed  : WP ONLY"
)

print(
    "Primary WP summary           : MEDIAN [Q1, Q3]"
)


# ============================================================
# 16. SAVE JSON
# ============================================================

runtime = (
    time.perf_counter()
    -
    start_time
)


json_summary = {

    "milestone":
        "57",

    "input_file":
        str(
            INPUT_FILE
        ),

    "rows":
        int(
            len(work)
        ),

    "controllers":
        controllers,

    "climates":
        climates,

    "scarcity_levels_pct":
        [
            float(x)
            for x in scarcity_levels
        ],

    "water_productivity_reporting": {

        "primary_summary":
            "median [Q1, Q3]",

        "reason":
            (
                "WP may be right-skewed because small positive "
                "irrigation denominators produce large ratios, "
                "while zero-irrigation episodes are represented "
                "as WP=0."
            ),

        "zero_tolerance":
            ZERO_TOL,
    },

    "overall_records":
        overall.to_dict(
            orient="records"
        ),

    "interpretation":
        interpretation,

    "scientific_status": {

        "new_simulations":
            False,

        "ppo_inference_rerun":
            False,

        "ppo_retrained":
            False,

        "controller_outputs_modified":
            False,

        "original_hypothesis_tests_changed":
            False,

        "new_hypothesis_tests":
            False,

        "frozen_panel_modified":
            False,
    },

    "runtime_seconds":
        float(
            runtime
        ),
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        json_summary,
        file,
        indent=2,
    )


# ============================================================
# 17. OUTPUT FILES
# ============================================================

print()

print(
    f"Runtime                    : "
    f"{runtime:.2f} s"
)

print()

print(
    "Saved:"
)

print(
    f"  {OVERALL_FILE}"
)

print(
    f"  {CLIMATE_FILE}"
)

print(
    f"  {COMPARISON_FILE}"
)

print(
    f"  {MANUSCRIPT_FILE}"
)

print(
    f"  {SUMMARY_JSON}"
)

print()

print("=" * 78)

print(
    "ROBUST WATER-PRODUCTIVITY SUMMARY COMPLETE"
)

print("=" * 78)