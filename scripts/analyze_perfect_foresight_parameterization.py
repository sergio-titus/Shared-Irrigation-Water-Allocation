# ============================================================
# PERFECT-FORESIGHT PARAMETERIZATION SENSITIVITY
# OFFLINE PERFECT-FORESIGHT PARAMETERIZATION SENSITIVITY
#
# PURPOSE
# -------
# Determine whether the offline perfect-foresight numerical reference is
# materially limited by its four-block temporal
# parameterization.
#
# CASE
# ----
# Tunis, 2019, 40% water availability
#
# TESTED HERE
# -----------
#   2 blocks x 4 fields =  8 parameters
#   6 blocks x 4 fields = 24 parameters
#   8 blocks x 4 fields = 32 parameters
#
# EXISTING FOUR-BLOCK PERFECT-FORESIGHT RESULT
# ------------------------
#   4 blocks x 4 fields = 16 parameters
#   Best yield retention = 90.526681%
#
# IMPORTANT
# ---------
# This is:
#   - post-hoc;
#   - offline;
#   - perfect-foresight;
#   - a numerical sensitivity diagnostic.
#
# This is NOT:
#   - a globally certified optimum;
#   - a mathematical upper bound;
#   - part of the original confirmatory controller tests;
#   - PPO training/tuning.
#
# OBJECTIVE
# ---------
# Maximize total final yield retention using the exact frozen
# AquaCrop final-test environment.
# ============================================================


from pathlib import Path
import importlib.util
import json
import time
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution


# ============================================================
# 1. CONFIGURATION
# ============================================================

FINAL_SCRIPT = Path(
    "evaluate_all_controllers.py"
)

CLIMATE = "Tunis"
YEAR = 2019
SCARCITY = 0.40
ENV_SEED = 0

N_FIELDS = 4

# ------------------------------------------------------------
# New parameterizations to test.
#
# Do NOT rerun four blocks here. We already have the much
# stronger three-restart four-block result for that configuration.
# ------------------------------------------------------------

BLOCK_CONFIGURATIONS = [
    2,
    6,
    8,
]

# ------------------------------------------------------------
# Existing four-block result
# ------------------------------------------------------------

EXISTING_4_BLOCK_YIELD_PCT = 90.526681

EXISTING_4_BLOCK_WORST_PCT = 88.125764

EXISTING_4_BLOCK_JAIN = 0.999628

EXISTING_4_BLOCK_BUDGET_UTIL_PCT = 100.0


# ------------------------------------------------------------
# Same differential-evolution settings as the smoke test
# ------------------------------------------------------------

DE_MAXITER = 8
DE_POPSIZE = 5
DE_TOL = 1e-3
DE_POLISH = False

# One common seed for the sensitivity experiment.
OPTIMIZER_SEED = 5610

EPS = 1e-10


OUTPUT_DIR = (
    Path("results")
    / "multiclimate_extension"
    / "offline_perfect_foresight"
    / "perfect_foresight_parameterization_sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULT_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_parameterization_parameterization_results.csv"
)

WEIGHT_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_parameterization_best_weights.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_parameterization_parameterization_summary.json"
)


# ============================================================
# 2. IMPORT FROZEN FINAL EVALUATOR
# ============================================================

print()

print("=" * 78)

print(
    "PERFECT-FORESIGHT PARAMETERIZATION SENSITIVITY"
)

print(
    "OFFLINE PERFECT-FORESIGHT PARAMETERIZATION SENSITIVITY"
)

print("=" * 78)


if not FINAL_SCRIPT.exists():

    raise FileNotFoundError(
        f"Cannot find {FINAL_SCRIPT}"
    )


spec = (
    importlib.util
    .spec_from_file_location(
        "final53j_56c",
        FINAL_SCRIPT,
    )
)


if (
    spec is None
    or spec.loader is None
):

    raise RuntimeError(
        "Could not import frozen 53J evaluator."
    )


final53j = (
    importlib.util
    .module_from_spec(
        spec
    )
)


spec.loader.exec_module(
    final53j
)


FIELD_NAMES = list(
    final53j.FIELD_NAMES
)


if len(FIELD_NAMES) != N_FIELDS:

    raise RuntimeError(
        f"Expected {N_FIELDS} fields, "
        f"found {len(FIELD_NAMES)}."
    )


DAILY_CAPACITY = float(
    final53j.DAILY_SYSTEM_CAPACITY
)


print()

print(
    f"Imported frozen evaluator : "
    f"{FINAL_SCRIPT}"
)


print(
    f"Case                      : "
    f"{CLIMATE} {YEAR}, "
    f"{100 * SCARCITY:.0f}%"
)


print(
    f"Daily shared capacity     : "
    f"{DAILY_CAPACITY:.3f} mm"
)


print(
    f"Optimizer seed            : "
    f"{OPTIMIZER_SEED}"
)


print(
    f"DE maxiter                : "
    f"{DE_MAXITER}"
)


print(
    f"DE popsize                : "
    f"{DE_POPSIZE}"
)


# ============================================================
# 3. HELPERS
# ============================================================

def allocation_vector(
    info,
):

    allocations = info[
        "allocations"
    ]


    if isinstance(
        allocations,
        dict,
    ):

        return np.asarray(
            [
                allocations[
                    name
                ]
                for name
                in FIELD_NAMES
            ],
            dtype=np.float64,
        )


    return np.asarray(
        allocations,
        dtype=np.float64,
    )


# ============================================================
# 4. WEIGHTED WATER-FILLING
# ============================================================

def weighted_water_fill(
    requests_dict,
    available_today,
    weights,
):

    requests = np.asarray(
        [
            float(
                requests_dict[
                    name
                ]
            )
            for name
            in FIELD_NAMES
        ],
        dtype=np.float64,
    )


    weights = np.asarray(
        weights,
        dtype=np.float64,
    )


    weights = np.maximum(
        weights,
        EPS,
    )


    allocations = np.zeros(
        N_FIELDS,
        dtype=np.float64,
    )


    remaining = float(
        max(
            0.0,
            available_today,
        )
    )


    active = [
        i
        for i
        in range(
            N_FIELDS
        )
        if requests[i] > EPS
    ]


    while (
        active
        and
        remaining > EPS
    ):

        active_weights = np.asarray(
            [
                weights[i]
                for i
                in active
            ],
            dtype=np.float64,
        )


        weight_sum = float(
            active_weights.sum()
        )


        if weight_sum <= EPS:

            active_weights = np.ones(
                len(active),
                dtype=np.float64,
            )

            weight_sum = float(
                len(active)
            )


        proposed = (
            remaining
            *
            active_weights
            /
            weight_sum
        )


        amount_used = 0.0

        next_active = []


        for (
            local_index,
            field_index,
        ) in enumerate(
            active
        ):

            unmet = (
                requests[
                    field_index
                ]
                -
                allocations[
                    field_index
                ]
            )


            amount = min(
                float(
                    proposed[
                        local_index
                    ]
                ),
                float(
                    unmet
                ),
            )


            allocations[
                field_index
            ] += amount


            amount_used += amount


            if (
                requests[
                    field_index
                ]
                -
                allocations[
                    field_index
                ]
                >
                EPS
            ):

                next_active.append(
                    field_index
                )


        remaining -= amount_used


        if amount_used <= EPS:

            break


        active = next_active


    allocations = np.minimum(
        allocations,
        requests,
    )


    total = float(
        allocations.sum()
    )


    if (
        total
        >
        available_today
        +
        1e-7
    ):

        raise RuntimeError(
            "Weighted allocation exceeded "
            "available daily water."
        )


    return {
        FIELD_NAMES[i]:
            float(
                allocations[i]
            )
        for i
        in range(
            N_FIELDS
        )
    }


# ============================================================
# 5. PARAMETER NORMALIZATION
# ============================================================

def normalize_parameters(
    parameters,
    n_blocks,
):

    matrix = np.asarray(
        parameters,
        dtype=np.float64,
    ).reshape(
        n_blocks,
        N_FIELDS,
    )


    matrix = np.maximum(
        matrix,
        1e-6,
    )


    for block in range(
        n_blocks
    ):

        block_mean = float(
            matrix[
                block
            ].mean()
        )


        if block_mean > EPS:

            matrix[
                block
            ] /= block_mean


    return matrix


# ============================================================
# 6. TEMPORAL BLOCK
# ============================================================

def get_block_index(
    step_index,
    n_blocks,
    total_steps=153,
):

    fraction = (
        float(
            step_index
        )
        /
        float(
            total_steps
        )
    )


    block = int(
        fraction
        *
        n_blocks
    )


    return int(
        np.clip(
            block,
            0,
            n_blocks - 1,
        )
    )


# ============================================================
# 7. RUN ONE PARAMETERIZED EPISODE
# ============================================================

evaluation_counters = {}


def run_episode(
    parameters,
    n_blocks,
):

    evaluation_counters[
        n_blocks
    ] = (
        evaluation_counters.get(
            n_blocks,
            0,
        )
        +
        1
    )


    weights = (
        normalize_parameters(
            parameters,
            n_blocks,
        )
    )


    env = (
        final53j.FinalTestEnv(
            climate=CLIMATE,
            year=YEAR,
            scarcity_fraction=SCARCITY,
            seed=ENV_SEED,
        )
    )


    obs, _ = env.reset(
        seed=ENV_SEED
    )


    terminated = False
    truncated = False

    step_index = 0
    final_info = None


    while not (
        terminated
        or truncated
    ):

        requests = (
            final53j.current_requests(
                env
            )
        )


        available_today = float(
            min(
                DAILY_CAPACITY,
                env.remaining_budget,
            )
        )


        block = (
            get_block_index(
                step_index=step_index,
                n_blocks=n_blocks,
            )
        )


        allocations = (
            weighted_water_fill(
                requests_dict=requests,
                available_today=(
                    available_today
                ),
                weights=weights[
                    block
                ],
            )
        )


        action = (
            final53j
            .allocation_to_action(
                requests=requests,
                allocations=allocations,
            )
        )


        desired = np.asarray(
            [
                allocations[
                    name
                ]
                for name
                in FIELD_NAMES
            ],
            dtype=np.float64,
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        executed = (
            allocation_vector(
                info
            )
        )


        error = float(
            np.max(
                np.abs(
                    desired
                    -
                    executed
                )
            )
        )


        if error > 1e-5:

            raise RuntimeError(
                "Unexpected environment projection "
                f"in {n_blocks}-block optimization: "
                f"{error:.6e} mm"
            )


        final_info = info

        step_index += 1


    if truncated:

        raise RuntimeError(
            "Unexpected episode truncation."
        )


    if final_info is None:

        raise RuntimeError(
            "No final info returned."
        )


    final_results = (
        final_info.get(
            "final_results"
        )
    )


    if final_results is None:

        raise RuntimeError(
            "Missing final_results."
        )


    return {

        "yield_retention":
            float(
                final_results[
                    "total_yield_retention"
                ]
            ),

        "yield_retention_pct":
            100.0
            *
            float(
                final_results[
                    "total_yield_retention"
                ]
            ),

        "worst_retention_pct":
            100.0
            *
            float(
                final_results[
                    "worst_retention"
                ]
            ),

        "jain":
            float(
                final_results[
                    "jain"
                ]
            ),

        "total_yield":
            float(
                final_results[
                    "total_yield"
                ]
            ),

        "total_irrigation":
            float(
                final_results[
                    "total_irrigation"
                ]
            ),

        "budget_utilization_pct":
            (
                float(
                    env.total_allocated
                )
                /
                float(
                    env.seasonal_budget
                )
                *
                100.0
                if
                env.seasonal_budget
                >
                EPS
                else
                0.0
            ),

        "weights":
            weights,
    }


# ============================================================
# 8. BASELINE VALUES
# ============================================================

print()

print("=" * 78)

print(
    "FROZEN BASELINES"
)

print("=" * 78)


equal_result = (
    final53j.run_episode(
        climate=CLIMATE,
        year=YEAR,
        scarcity_fraction=SCARCITY,
        controller="Equal",
    )
)


priority_result = (
    final53j.run_episode(
        climate=CLIMATE,
        year=YEAR,
        scarcity_fraction=SCARCITY,
        controller="Priority",
    )
)


EQUAL_YIELD = float(
    equal_result[
        "Yield_retention_pct"
    ]
)


PRIORITY_YIELD = float(
    priority_result[
        "Yield_retention_pct"
    ]
)


print(
    f"Equal yield retention     : "
    f"{EQUAL_YIELD:.6f}%"
)


print(
    f"Priority yield retention  : "
    f"{PRIORITY_YIELD:.6f}%"
)


print(
    f"Existing 4-block reference: "
    f"{EXISTING_4_BLOCK_YIELD_PCT:.6f}%"
)


# ============================================================
# 9. VALIDATE EACH PARAMETERIZATION WITH EQUAL WEIGHTS
# ============================================================

print()

print("=" * 78)

print(
    "PARAMETERIZATION VALIDATION"
)

print("=" * 78)


for n_blocks in (
    BLOCK_CONFIGURATIONS
):

    equal_parameters = np.ones(
        n_blocks
        *
        N_FIELDS,
        dtype=np.float64,
    )


    validation = (
        run_episode(
            parameters=(
                equal_parameters
            ),
            n_blocks=n_blocks,
        )
    )


    difference = (
        validation[
            "yield_retention_pct"
        ]
        -
        EQUAL_YIELD
    )


    passed = bool(
        abs(
            difference
        )
        <
        1e-5
    )


    print(
        f"{n_blocks:2d} blocks | "
        f"parameterized Equal="
        f"{validation['yield_retention_pct']:.6f}% | "
        f"difference={difference:+.8f} pp | "
        f"{'PASS' if passed else 'FAIL'}"
    )


    if not passed:

        raise RuntimeError(
            f"{n_blocks}-block Equal validation failed."
        )


# ============================================================
# 10. OPTIMIZATION
# ============================================================

print()

print("=" * 78)

print(
    "PARAMETERIZATION SENSITIVITY OPTIMIZATION"
)

print("=" * 78)


all_results = []

all_weights = []


# ------------------------------------------------------------
# Insert existing 4-block result first.
# ------------------------------------------------------------

all_results.append(
    {
        "Blocks":
            4,

        "Parameters":
            16,

        "Source":
            "existing_perfect_foresight_best",

        "Optimizer_seed":
            5603,

        "Yield_retention_pct":
            EXISTING_4_BLOCK_YIELD_PCT,

        "Worst_field_retention_pct":
            EXISTING_4_BLOCK_WORST_PCT,

        "Jain":
            EXISTING_4_BLOCK_JAIN,

        "Budget_utilization_pct":
            EXISTING_4_BLOCK_BUDGET_UTIL_PCT,

        "Gain_vs_Equal_pp":
            (
                EXISTING_4_BLOCK_YIELD_PCT
                -
                EQUAL_YIELD
            ),

        "Gain_vs_Priority_pp":
            (
                EXISTING_4_BLOCK_YIELD_PCT
                -
                PRIORITY_YIELD
            ),

        "Function_evaluations":
            np.nan,

        "Runtime_minutes":
            np.nan,
    }
)


for n_blocks in (
    BLOCK_CONFIGURATIONS
):

    n_parameters = (
        n_blocks
        *
        N_FIELDS
    )


    print()

    print("-" * 78)

    print(
        f"{n_blocks} BLOCKS × "
        f"{N_FIELDS} FIELDS = "
        f"{n_parameters} PARAMETERS"
    )

    print("-" * 78)


    best_seen = {
        "yield_pct":
            -np.inf,
    }


    def objective(
        parameters,
    ):

        result = (
            run_episode(
                parameters=parameters,
                n_blocks=n_blocks,
            )
        )


        yield_pct = float(
            result[
                "yield_retention_pct"
            ]
        )


        if (
            yield_pct
            >
            best_seen[
                "yield_pct"
            ]
        ):

            best_seen[
                "yield_pct"
            ] = yield_pct


            print(
                f"  evaluation "
                f"{evaluation_counters[n_blocks]:5d} | "
                f"new best = "
                f"{yield_pct:.4f}%"
            )


        return -float(
            result[
                "yield_retention"
            ]
        )


    bounds = [
        (
            0.05,
            5.0,
        )
        for _
        in range(
            n_parameters
        )
    ]


    evaluations_before = (
        evaluation_counters.get(
            n_blocks,
            0,
        )
    )


    start = (
        time.perf_counter()
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        optimization = (
            differential_evolution(
                func=objective,
                bounds=bounds,
                strategy="best1bin",
                maxiter=DE_MAXITER,
                popsize=DE_POPSIZE,
                tol=DE_TOL,
                mutation=(
                    0.5,
                    1.0,
                ),
                recombination=0.7,
                seed=OPTIMIZER_SEED,
                polish=DE_POLISH,
                updating="immediate",
                workers=1,
                disp=False,
            )
        )


    runtime = (
        time.perf_counter()
        -
        start
    )


    evaluations_after = (
        evaluation_counters[
            n_blocks
        ]
    )


    n_evaluations = (
        evaluations_after
        -
        evaluations_before
    )


    best = (
        run_episode(
            parameters=(
                optimization.x
            ),
            n_blocks=n_blocks,
        )
    )


    normalized_weights = (
        normalize_parameters(
            parameters=(
                optimization.x
            ),
            n_blocks=n_blocks,
        )
    )


    gain_equal = (
        best[
            "yield_retention_pct"
        ]
        -
        EQUAL_YIELD
    )


    gain_priority = (
        best[
            "yield_retention_pct"
        ]
        -
        PRIORITY_YIELD
    )


    print()

    print(
        f"Best yield retention      : "
        f"{best['yield_retention_pct']:.6f}%"
    )


    print(
        f"Gain vs Equal             : "
        f"{gain_equal:+.6f} pp"
    )


    print(
        f"Gain vs Priority          : "
        f"{gain_priority:+.6f} pp"
    )


    print(
        f"Worst-field retention     : "
        f"{best['worst_retention_pct']:.6f}%"
    )


    print(
        f"Jain fairness             : "
        f"{best['jain']:.6f}"
    )


    print(
        f"Budget utilization        : "
        f"{best['budget_utilization_pct']:.3f}%"
    )


    print(
        f"Function evaluations      : "
        f"{optimization.nfev}"
    )


    print(
        f"Runtime                   : "
        f"{runtime / 60.0:.2f} min"
    )


    all_results.append(
        {
            "Blocks":
                int(
                    n_blocks
                ),

            "Parameters":
                int(
                    n_parameters
                ),

            "Source":
                "parameterization_sensitivity",

            "Optimizer_seed":
                int(
                    OPTIMIZER_SEED
                ),

            "Yield_retention_pct":
                float(
                    best[
                        "yield_retention_pct"
                    ]
                ),

            "Worst_field_retention_pct":
                float(
                    best[
                        "worst_retention_pct"
                    ]
                ),

            "Jain":
                float(
                    best[
                        "jain"
                    ]
                ),

            "Budget_utilization_pct":
                float(
                    best[
                        "budget_utilization_pct"
                    ]
                ),

            "Gain_vs_Equal_pp":
                float(
                    gain_equal
                ),

            "Gain_vs_Priority_pp":
                float(
                    gain_priority
                ),

            "Function_evaluations":
                int(
                    optimization.nfev
                ),

            "Runtime_minutes":
                float(
                    runtime
                    /
                    60.0
                ),
        }
    )


    for block in range(
        n_blocks
    ):

        for field_index, field in enumerate(
            FIELD_NAMES
        ):

            all_weights.append(
                {
                    "Blocks":
                        int(
                            n_blocks
                        ),

                    "Block":
                        int(
                            block + 1
                        ),

                    "Field":
                        str(
                            field
                        ),

                    "Normalized_weight":
                        float(
                            normalized_weights[
                                block,
                                field_index,
                            ]
                        ),
                }
            )


# ============================================================
# 11. FINAL TABLE
# ============================================================

results_df = (
    pd.DataFrame(
        all_results
    )
    .sort_values(
        "Blocks"
    )
    .reset_index(
        drop=True
    )
)


weights_df = pd.DataFrame(
    all_weights
)


results_df.to_csv(
    RESULT_FILE,
    index=False,
)


weights_df.to_csv(
    WEIGHT_FILE,
    index=False,
)


print()

print("=" * 78)

print(
    "FINAL PARAMETERIZATION COMPARISON"
)

print("=" * 78)


display_columns = [
    "Blocks",
    "Parameters",
    "Yield_retention_pct",
    "Gain_vs_Equal_pp",
    "Gain_vs_Priority_pp",
    "Worst_field_retention_pct",
    "Jain",
    "Budget_utilization_pct",
]


print(
    results_df[
        display_columns
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 12. SENSITIVITY DIAGNOSTICS
# ============================================================

yield_by_blocks = {
    int(
        row[
            "Blocks"
        ]
    ):
        float(
            row[
                "Yield_retention_pct"
            ]
        )
    for _, row
    in results_df.iterrows()
}


best_blocks = int(
    results_df.loc[
        results_df[
            "Yield_retention_pct"
        ].idxmax(),
        "Blocks",
    ]
)


best_yield = float(
    results_df[
        "Yield_retention_pct"
    ].max()
)


four_block_yield = float(
    yield_by_blocks[
        4
    ]
)


best_gain_over_4 = (
    best_yield
    -
    four_block_yield
)


# Difference between 6- and 8-block solutions is useful
# for judging high-flexibility stabilization.

if (
    6 in yield_by_blocks
    and
    8 in yield_by_blocks
):

    high_flex_difference = abs(
        yield_by_blocks[
            8
        ]
        -
        yield_by_blocks[
            6
        ]
    )

else:

    high_flex_difference = None


print()

print("=" * 78)

print(
    "SENSITIVITY DIAGNOSTICS"
)

print("=" * 78)


print(
    f"Best parameterization     : "
    f"{best_blocks} blocks"
)


print(
    f"Best numerical yield      : "
    f"{best_yield:.6f}%"
)


print(
    f"4-block numerical yield   : "
    f"{four_block_yield:.6f}%"
)


print(
    f"Best gain over 4 blocks   : "
    f"{best_gain_over_4:+.6f} pp"
)


if (
    high_flex_difference
    is not None
):

    print(
        f"|8-block - 6-block|       : "
        f"{high_flex_difference:.6f} pp"
    )


# ============================================================
# 13. INTERPRETATION FLAGS
#
# These thresholds are descriptive diagnostics only.
# They are NOT statistical tests.
# ============================================================

near_plateau_025 = bool(
    best_gain_over_4
    <=
    0.25
)


near_plateau_050 = bool(
    best_gain_over_4
    <=
    0.50
)


print()

print(
    "Descriptive plateau checks:"
)


print(
    f"  Best gain over 4 blocks "
    f"<= 0.25 pp : "
    f"{'YES' if near_plateau_025 else 'NO'}"
)


print(
    f"  Best gain over 4 blocks "
    f"<= 0.50 pp : "
    f"{'YES' if near_plateau_050 else 'NO'}"
)


# ============================================================
# 14. SAVE SUMMARY
# ============================================================

summary = {

    "milestone":
        "56C",

    "case": {
        "climate":
            CLIMATE,

        "year":
            int(
                YEAR
            ),

        "scarcity_fraction":
            float(
                SCARCITY
            ),
    },

    "objective":
        "maximize total_yield_retention",

    "tested_blocks":
        [
            2,
            4,
            6,
            8,
        ],

    "newly_optimized_blocks":
        BLOCK_CONFIGURATIONS,

    "four_block_source":
        "existing four-block best of three restarts",

    "optimizer_seed_for_new_runs":
        int(
            OPTIMIZER_SEED
        ),

    "results":
        results_df.to_dict(
            orient="records"
        ),

    "diagnostics": {
        "best_blocks":
            int(
                best_blocks
            ),

        "best_yield_retention_pct":
            float(
                best_yield
            ),

        "best_gain_over_4_blocks_pp":
            float(
                best_gain_over_4
            ),

        "six_vs_eight_difference_pp":
            (
                float(
                    high_flex_difference
                )
                if
                high_flex_difference
                is not None
                else
                None
            ),

        "best_gain_over_4_le_0_25_pp":
            bool(
                near_plateau_025
            ),

        "best_gain_over_4_le_0_50_pp":
            bool(
                near_plateau_050
            ),
    },

    "scientific_status": {
        "global_optimality_guarantee":
            False,

        "mathematical_upper_bound":
            False,

        "offline_perfect_foresight_reference":
            True,

        "parameterization_sensitivity":
            True,

        "post_hoc":
            True,

        "included_in_original_hypothesis_tests":
            False,
    },

    "integrity": {
        "ppo_training_performed":
            False,

        "ppo_inference_performed":
            False,

        "original_controllers_modified":
            False,

        "original_final_results_modified":
            False,

        "new_hypothesis_tests":
            False,
    },
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# 15. FINAL INTEGRITY REPORT
# ============================================================

print()

print("=" * 78)

print(
    "SCIENTIFIC STATUS"
)

print("=" * 78)


print(
    "Global-optimality guarantee : NO"
)

print(
    "Mathematical upper bound    : NO"
)

print(
    "Perfect-foresight reference : YES"
)

print(
    "Parameter sensitivity       : YES"
)

print(
    "Post-hoc diagnostic         : YES"
)

print(
    "Original hypothesis tests   : UNCHANGED"
)

print(
    "PPO retrained               : NO"
)

print(
    "Frozen controllers modified : NO"
)


print()

print(
    "Saved:"
)


print(
    f"  {RESULT_FILE}"
)


print(
    f"  {WEIGHT_FILE}"
)


print(
    f"  {SUMMARY_FILE}"
)


print()

print("=" * 78)

print(
    "PERFECT-FORESIGHT PARAMETERIZATION SENSITIVITY COMPLETE"
)

print("=" * 78)