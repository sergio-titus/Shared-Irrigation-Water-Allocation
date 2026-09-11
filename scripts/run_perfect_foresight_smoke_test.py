# ============================================================
# PERFECT-FORESIGHT SMOKE TEST
# OFFLINE PERFECT-FORESIGHT OPTIMIZATION — SMOKE TEST
#
# CASE
# ----
# Tunis, 2019, 40% water availability
#
# PURPOSE
# -------
# Test whether a low-dimensional, non-deployable offline
# optimization reference can be:
#
#   1. evaluated through the exact frozen AquaCrop environment;
#   2. optimized reproducibly;
#   3. compared against Equal and Priority;
#   4. executed at acceptable computational cost.
#
# IMPORTANT
# ---------
# This is NOT:
#   - a globally certified optimum;
#   - a mathematical upper bound;
#   - a new controller in the original confirmatory comparison;
#   - PPO training or tuning.
#
# The optimizer receives the realized seasonal outcome through
# repeated full-season AquaCrop simulations. It is therefore a
# post-hoc OFFLINE PERFECT-FORESIGHT NUMERICAL REFERENCE.
#
# PARAMETERIZATION
# ----------------
# Four temporal blocks × four fields = 16 continuous weights.
#
# Within each block:
#   - current AquaCrop requests are observed;
#   - available daily water is known;
#   - water is distributed using weighted water-filling;
#   - no field may exceed its request;
#   - daily capacity and seasonal budget remain unchanged.
#
# OBJECTIVE
# ---------
# Maximize the exact final-test total yield retention:
#
#       total_yield / total_reference_yield
#
# from the frozen AquaCrop evaluator.
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

N_BLOCKS = 4
N_FIELDS = 4
N_PARAMETERS = N_BLOCKS * N_FIELDS

# ------------------------------------------------------------
# Smoke-test optimization budget
#
# 16 parameters.
#
# popsize=5 gives approximately 80 population members.
# maxiter=8 gives a modest exploratory budget suitable for
# checking feasibility/runtime before a larger optimization.
# ------------------------------------------------------------

DE_MAXITER = 8
DE_POPSIZE = 5
DE_TOL = 1e-3
DE_POLISH = False

OPTIMIZER_SEEDS = [
    5601,
    5602,
    5603,
]

EPS = 1e-10


OUTPUT_DIR = (
    Path("results")
    / "multiclimate_extension"
    / "offline_perfect_foresight"
    / "perfect_foresight_smoke_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RUN_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_smoke_test_optimizer_runs.csv"
)

BEST_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_smoke_test_best_solution.json"
)

TRACE_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_smoke_test_best_episode_trace.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_smoke_test_smoke_test_summary.json"
)


# ============================================================
# 2. IMPORT EXACT FROZEN FINAL EVALUATOR
# ============================================================

print()
print("=" * 78)
print("PERFECT-FORESIGHT SMOKE TEST")
print("OFFLINE PERFECT-FORESIGHT OPTIMIZATION — SMOKE TEST")
print("=" * 78)


if not FINAL_SCRIPT.exists():

    raise FileNotFoundError(
        f"Cannot find {FINAL_SCRIPT}"
    )


spec = importlib.util.spec_from_file_location(
    "final53j_56b",
    FINAL_SCRIPT,
)


if spec is None or spec.loader is None:

    raise RuntimeError(
        "Could not import frozen 53J evaluator."
    )


final53j = importlib.util.module_from_spec(
    spec
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


print(
    f"Imported frozen evaluator: {FINAL_SCRIPT}"
)

print(
    f"Case                     : "
    f"{CLIMATE} {YEAR}, "
    f"{100 * SCARCITY:.0f}%"
)

print(
    f"Decision parameterization: "
    f"{N_BLOCKS} blocks × {N_FIELDS} fields "
    f"= {N_PARAMETERS} parameters"
)


# ============================================================
# 3. HELPER — ALLOCATION VECTOR
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
                allocations[name]
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
#
# Weights define RELATIVE priority.
#
# Allocation proceeds continuously:
#
#     share_i ∝ weight_i
#
# among active requesting fields.
#
# If a field reaches its request, unused water is redistributed
# among remaining active fields.
#
# The allocation is therefore always feasible BEFORE env.step.
# ============================================================

def weighted_water_fill(
    requests_dict,
    available_today,
    weights,
):

    requests = np.asarray(
        [
            float(
                requests_dict[name]
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
        for i in range(N_FIELDS)
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
                for i in active
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


        for local_index, field_index in enumerate(
            active
        ):

            unmet = (
                requests[field_index]
                -
                allocations[field_index]
            )


            amount = min(
                float(
                    proposed[local_index]
                ),
                float(
                    unmet
                ),
            )


            allocations[field_index] += amount

            amount_used += amount


            if (
                requests[field_index]
                -
                allocations[field_index]
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
            "Weighted water-filling exceeded "
            "available daily water."
        )


    return {
        FIELD_NAMES[i]:
            float(
                allocations[i]
            )
        for i in range(
            N_FIELDS
        )
    }


# ============================================================
# 5. NORMALIZE PARAMETERS
#
# Absolute scale of weights does not matter.
# Only relative weights within each block matter.
#
# Normalize each block to mean = 1.
# ============================================================

def normalize_parameters(
    parameters,
):

    matrix = np.asarray(
        parameters,
        dtype=np.float64,
    ).reshape(
        N_BLOCKS,
        N_FIELDS,
    )


    matrix = np.maximum(
        matrix,
        1e-6,
    )


    for b in range(
        N_BLOCKS
    ):

        mean_value = float(
            matrix[b].mean()
        )


        if mean_value > EPS:

            matrix[b] /= mean_value


    return matrix


# ============================================================
# 6. DETERMINE TEMPORAL BLOCK
#
# The 153-step simulation horizon is divided into four
# approximately equal temporal blocks.
#
# The block is based only on simulation position.
# ============================================================

def get_block_index(
    step_index,
    total_steps=153,
):

    fraction = (
        float(step_index)
        /
        float(total_steps)
    )


    block = int(
        fraction
        *
        N_BLOCKS
    )


    return int(
        np.clip(
            block,
            0,
            N_BLOCKS - 1,
        )
    )


# ============================================================
# 7. RUN ONE OFFLINE-PARAMETERIZED AQUACROP EPISODE
# ============================================================

evaluation_counter = {
    "count": 0
}


def run_parameterized_episode(
    parameters,
    return_trace=False,
):

    evaluation_counter[
        "count"
    ] += 1


    weight_matrix = (
        normalize_parameters(
            parameters
        )
    )


    env = final53j.FinalTestEnv(
        climate=CLIMATE,
        year=YEAR,
        scarcity_fraction=SCARCITY,
        seed=ENV_SEED,
    )


    obs, _ = env.reset(
        seed=ENV_SEED
    )


    terminated = False
    truncated = False

    trace = []

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


        request_vector = np.asarray(
            [
                float(
                    requests[name]
                )
                for name
                in FIELD_NAMES
            ],
            dtype=np.float64,
        )


        available_today = float(
            min(
                DAILY_CAPACITY,
                env.remaining_budget,
            )
        )


        block = get_block_index(
            step_index
        )


        weights = (
            weight_matrix[
                block
            ]
        )


        allocations = (
            weighted_water_fill(
                requests_dict=requests,
                available_today=available_today,
                weights=weights,
            )
        )


        action = (
            final53j
            .allocation_to_action(
                requests=requests,
                allocations=allocations,
            )
        )


        remaining_before = float(
            env.remaining_budget
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


        executed = allocation_vector(
            info
        )


        desired = np.asarray(
            [
                allocations[name]
                for name
                in FIELD_NAMES
            ],
            dtype=np.float64,
        )


        reproduction_error = float(
            np.max(
                np.abs(
                    executed
                    -
                    desired
                )
            )
        )


        if reproduction_error > 1e-5:

            raise RuntimeError(
                "Offline allocation was unexpectedly "
                "modified by environment projection. "
                f"Error={reproduction_error:.6e}"
            )


        if return_trace:

            row = {
                "Step":
                    int(
                        step_index + 1
                    ),

                "Block":
                    int(
                        block + 1
                    ),

                "Remaining_budget_before_mm":
                    remaining_before,

                "Available_today_mm":
                    available_today,

                "Total_request_mm":
                    float(
                        request_vector.sum()
                    ),

                "Total_allocation_mm":
                    float(
                        executed.sum()
                    ),

                "Allocation_reproduction_error_mm":
                    reproduction_error,
            }


            for i, name in enumerate(
                FIELD_NAMES
            ):

                row[
                    f"{name}_request_mm"
                ] = float(
                    request_vector[i]
                )

                row[
                    f"{name}_weight"
                ] = float(
                    weights[i]
                )

                row[
                    f"{name}_allocation_mm"
                ] = float(
                    executed[i]
                )


            trace.append(
                row
            )


        final_info = info

        step_index += 1


    if truncated:

        raise RuntimeError(
            "Unexpected episode truncation."
        )


    if final_info is None:

        raise RuntimeError(
            "No final environment info."
        )


    final_results = final_info.get(
        "final_results"
    )


    if final_results is None:

        raise RuntimeError(
            "Missing final_results."
        )


    yield_retention = float(
        final_results[
            "total_yield_retention"
        ]
    )


    mean_field_retention = float(
        final_results[
            "mean_retention"
        ]
    )


    worst_retention = float(
        final_results[
            "worst_retention"
        ]
    )


    jain = float(
        final_results[
            "jain"
        ]
    )


    total_yield = float(
        final_results[
            "total_yield"
        ]
    )


    total_irrigation = float(
        final_results[
            "total_irrigation"
        ]
    )


    result = {
        "yield_retention":
            yield_retention,

        "yield_retention_pct":
            100.0
            *
            yield_retention,

        "mean_field_retention":
            mean_field_retention,

        "worst_retention":
            worst_retention,

        "worst_retention_pct":
            100.0
            *
            worst_retention,

        "jain":
            jain,

        "total_yield":
            total_yield,

        "total_irrigation":
            total_irrigation,

        "seasonal_budget":
            float(
                env.seasonal_budget
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
                if env.seasonal_budget
                >
                EPS
                else
                0.0
            ),

        "budget_exhausted":
            bool(
                env.budget_exhaustion_step
                is not None
            ),

        "budget_exhaustion_step":
            (
                int(
                    env.budget_exhaustion_step
                )
                if
                env.budget_exhaustion_step
                is not None
                else
                None
            ),

        "weights":
            weight_matrix.copy(),
    }


    if return_trace:

        result[
            "trace"
        ] = pd.DataFrame(
            trace
        )


    return result


# ============================================================
# 8. OBJECTIVE
#
# scipy minimizes, therefore:
#
#       objective = - yield retention
#
# No fairness or water-productivity term is included.
# This benchmark answers only:
#
# "How high can final aggregate yield retention be made by
# this offline parameterized numerical search?"
# ============================================================

best_seen = {
    "objective":
        np.inf,

    "yield_retention":
        -np.inf,

    "parameters":
        None,
}


def objective(
    parameters,
):

    result = (
        run_parameterized_episode(
            parameters,
            return_trace=False,
        )
    )


    value = -float(
        result[
            "yield_retention"
        ]
    )


    if (
        value
        <
        best_seen[
            "objective"
        ]
    ):

        best_seen[
            "objective"
        ] = value

        best_seen[
            "yield_retention"
        ] = float(
            result[
                "yield_retention"
            ]
        )

        best_seen[
            "parameters"
        ] = np.asarray(
            parameters,
            dtype=np.float64,
        ).copy()


        print(
            f"  evaluation "
            f"{evaluation_counter['count']:5d} | "
            f"new best yield retention = "
            f"{100 * result['yield_retention']:.4f}%"
        )


    return value


# ============================================================
# 9. BASELINE CONTROLLERS
#
# These are the exact frozen 53J implementations.
# ============================================================

print()
print("=" * 78)
print("FROZEN BASELINES")
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


print(
    f"Equal yield retention     : "
    f"{equal_result['Yield_retention_pct']:.4f}%"
)


print(
    f"Priority yield retention  : "
    f"{priority_result['Yield_retention_pct']:.4f}%"
)


# ============================================================
# 10. EQUAL-WEIGHT PARAMETERIZATION CHECK
#
# [1,1,1,1] in every block should reproduce the Equal
# water-filling strategy numerically.
# ============================================================

print()
print("=" * 78)
print("PARAMETERIZATION VALIDATION")
print("=" * 78)


equal_parameters = np.ones(
    N_PARAMETERS,
    dtype=np.float64,
)


parameterized_equal = (
    run_parameterized_episode(
        equal_parameters,
        return_trace=False,
    )
)


equal_difference_pp = (
    parameterized_equal[
        "yield_retention_pct"
    ]
    -
    float(
        equal_result[
            "Yield_retention_pct"
        ]
    )
)


print(
    f"Frozen Equal              : "
    f"{equal_result['Yield_retention_pct']:.6f}%"
)


print(
    f"Parameterized Equal       : "
    f"{parameterized_equal['yield_retention_pct']:.6f}%"
)


print(
    f"Difference                : "
    f"{equal_difference_pp:.8f} percentage points"
)


parameterization_validation_pass = bool(
    abs(
        equal_difference_pp
    )
    <
    1e-5
)


print(
    f"Validation                : "
    f"{'PASS' if parameterization_validation_pass else 'FAIL'}"
)


if not parameterization_validation_pass:

    raise RuntimeError(
        "Parameterized Equal does not reproduce "
        "the frozen Equal controller."
    )


# ============================================================
# 11. OPTIMIZATION BOUNDS
#
# Each raw weight lies in [0.05, 5.0].
#
# The weights are normalized within each block before use.
# Therefore only relative allocation preferences matter.
# ============================================================

bounds = [
    (
        0.05,
        5.0,
    )
    for _ in range(
        N_PARAMETERS
    )
]


# ============================================================
# 12. RUN MULTIPLE DIFFERENTIAL-EVOLUTION RESTARTS
# ============================================================

print()
print("=" * 78)
print("OFFLINE OPTIMIZATION")
print("=" * 78)


run_records = []

global_best = None


optimization_start = (
    time.perf_counter()
)


for run_index, optimizer_seed in enumerate(
    OPTIMIZER_SEEDS,
    start=1,
):

    print()
    print("-" * 78)

    print(
        f"Optimizer run "
        f"{run_index}/{len(OPTIMIZER_SEEDS)} "
        f"| seed={optimizer_seed}"
    )

    print("-" * 78)


    evaluations_before = int(
        evaluation_counter[
            "count"
        ]
    )


    run_start = (
        time.perf_counter()
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        de_result = (
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
                seed=optimizer_seed,
                polish=DE_POLISH,
                updating="immediate",
                workers=1,
                disp=False,
            )
        )


    run_runtime = (
        time.perf_counter()
        -
        run_start
    )


    evaluations_after = int(
        evaluation_counter[
            "count"
        ]
    )


    run_evaluations = (
        evaluations_after
        -
        evaluations_before
    )


    candidate = (
        run_parameterized_episode(
            de_result.x,
            return_trace=False,
        )
    )


    normalized_weights = (
        normalize_parameters(
            de_result.x
        )
    )


    record = {
        "Run":
            int(
                run_index
            ),

        "Optimizer_seed":
            int(
                optimizer_seed
            ),

        "Success":
            bool(
                de_result.success
            ),

        "Message":
            str(
                de_result.message
            ),

        "Iterations":
            int(
                de_result.nit
            ),

        "Function_evaluations_reported":
            int(
                de_result.nfev
            ),

        "Episode_evaluations":
            int(
                run_evaluations
            ),

        "Runtime_minutes":
            float(
                run_runtime
                /
                60.0
            ),

        "Yield_retention_pct":
            float(
                candidate[
                    "yield_retention_pct"
                ]
            ),

        "Mean_field_retention_pct":
            float(
                100.0
                *
                candidate[
                    "mean_field_retention"
                ]
            ),

        "Worst_field_retention_pct":
            float(
                candidate[
                    "worst_retention_pct"
                ]
            ),

        "Jain":
            float(
                candidate[
                    "jain"
                ]
            ),

        "Total_yield_t_ha":
            float(
                candidate[
                    "total_yield"
                ]
            ),

        "Total_irrigation_mm":
            float(
                candidate[
                    "total_irrigation"
                ]
            ),

        "Budget_utilization_pct":
            float(
                candidate[
                    "budget_utilization_pct"
                ]
            ),
    }


    for b in range(
        N_BLOCKS
    ):

        for i, field in enumerate(
            FIELD_NAMES
        ):

            record[
                f"B{b + 1}_{field}_weight"
            ] = float(
                normalized_weights[
                    b,
                    i,
                ]
            )


    run_records.append(
        record
    )


    print(
        f"Yield retention          : "
        f"{candidate['yield_retention_pct']:.4f}%"
    )


    print(
        f"Worst-field retention    : "
        f"{candidate['worst_retention_pct']:.4f}%"
    )


    print(
        f"Jain fairness            : "
        f"{candidate['jain']:.6f}"
    )


    print(
        f"Budget utilization       : "
        f"{candidate['budget_utilization_pct']:.3f}%"
    )


    print(
        f"Function evaluations     : "
        f"{de_result.nfev}"
    )


    print(
        f"Runtime                  : "
        f"{run_runtime / 60.0:.2f} min"
    )


    if (
        global_best is None
        or
        candidate[
            "yield_retention"
        ]
        >
        global_best[
            "result"
        ][
            "yield_retention"
        ]
    ):

        global_best = {
            "optimizer_seed":
                int(
                    optimizer_seed
                ),

            "raw_parameters":
                np.asarray(
                    de_result.x,
                    dtype=np.float64,
                ).copy(),

            "normalized_weights":
                normalized_weights.copy(),

            "result":
                candidate,

            "scipy_result": {
                "success":
                    bool(
                        de_result.success
                    ),

                "message":
                    str(
                        de_result.message
                    ),

                "nit":
                    int(
                        de_result.nit
                    ),

                "nfev":
                    int(
                        de_result.nfev
                    ),
            },
        }


optimization_runtime = (
    time.perf_counter()
    -
    optimization_start
)


# ============================================================
# 13. SAVE RUN COMPARISON
# ============================================================

runs_df = pd.DataFrame(
    run_records
)


runs_df.to_csv(
    RUN_FILE,
    index=False,
)


# ============================================================
# 14. REPLAY BEST SOLUTION WITH TRACE
# ============================================================

if global_best is None:

    raise RuntimeError(
        "No optimizer result produced."
    )


best_replay = (
    run_parameterized_episode(
        global_best[
            "raw_parameters"
        ],
        return_trace=True,
    )
)


best_trace = (
    best_replay[
        "trace"
    ]
)


best_trace.to_csv(
    TRACE_FILE,
    index=False,
)


# ============================================================
# 15. STABILITY ACROSS OPTIMIZER RESTARTS
# ============================================================

run_yields = runs_df[
    "Yield_retention_pct"
].to_numpy(
    dtype=np.float64
)


run_mean = float(
    np.mean(
        run_yields
    )
)


run_sd = float(
    np.std(
        run_yields,
        ddof=1,
    )
) if len(run_yields) > 1 else 0.0


run_range = float(
    np.max(
        run_yields
    )
    -
    np.min(
        run_yields
    )
)


best_yield_pct = float(
    best_replay[
        "yield_retention_pct"
    ]
)


equal_yield_pct = float(
    equal_result[
        "Yield_retention_pct"
    ]
)


priority_yield_pct = float(
    priority_result[
        "Yield_retention_pct"
    ]
)


best_vs_equal_pp = (
    best_yield_pct
    -
    equal_yield_pct
)


best_vs_priority_pp = (
    best_yield_pct
    -
    priority_yield_pct
)


# ============================================================
# 16. PRINT BEST WEIGHTS
# ============================================================

print()
print("=" * 78)
print("BEST OFFLINE NUMERICAL REFERENCE")
print("=" * 78)


print(
    f"Best optimizer seed       : "
    f"{global_best['optimizer_seed']}"
)


print(
    f"Best yield retention      : "
    f"{best_yield_pct:.4f}%"
)


print(
    f"Equal                     : "
    f"{equal_yield_pct:.4f}%"
)


print(
    f"Priority                  : "
    f"{priority_yield_pct:.4f}%"
)


print(
    f"Gain vs Equal             : "
    f"{best_vs_equal_pp:+.4f} pp"
)


print(
    f"Gain vs Priority          : "
    f"{best_vs_priority_pp:+.4f} pp"
)


print(
    f"Worst-field retention     : "
    f"{best_replay['worst_retention_pct']:.4f}%"
)


print(
    f"Jain fairness             : "
    f"{best_replay['jain']:.6f}"
)


print(
    f"Budget utilization        : "
    f"{best_replay['budget_utilization_pct']:.3f}%"
)


print()
print(
    "Normalized block × field weights:"
)


weight_table = pd.DataFrame(
    global_best[
        "normalized_weights"
    ],
    columns=FIELD_NAMES,
    index=[
        f"Block_{i + 1}"
        for i in range(
            N_BLOCKS
        )
    ],
)


print(
    weight_table.round(
        4
    ).to_string()
)


# ============================================================
# 17. RESTART STABILITY
# ============================================================

print()
print("=" * 78)
print("OPTIMIZER RESTART STABILITY")
print("=" * 78)


print(
    runs_df[
        [
            "Run",
            "Optimizer_seed",
            "Yield_retention_pct",
            "Worst_field_retention_pct",
            "Jain",
            "Runtime_minutes",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print()
print(
    f"Yield across restarts     : "
    f"{run_mean:.4f} ± {run_sd:.4f}%"
)


print(
    f"Restart range             : "
    f"{run_range:.4f} percentage points"
)


# ============================================================
# 18. SAVE JSON SUMMARY
# ============================================================

best_json = {
    "label":
        (
            "offline perfect-foresight numerical "
            "optimization reference"
        ),

    "global_optimality_guarantee":
        False,

    "mathematical_upper_bound":
        False,

    "case": {
        "climate":
            CLIMATE,

        "year":
            YEAR,

        "scarcity_fraction":
            SCARCITY,
    },

    "parameterization": {
        "temporal_blocks":
            N_BLOCKS,

        "fields":
            N_FIELDS,

        "parameters":
            N_PARAMETERS,

        "allocation_rule":
            "weighted water-filling",
    },

    "objective":
        "maximize total_yield_retention",

    "optimizer":
        "SciPy differential_evolution",

    "optimizer_seed":
        int(
            global_best[
                "optimizer_seed"
            ]
        ),

    "raw_parameters":
        [
            float(x)
            for x in
            global_best[
                "raw_parameters"
            ]
        ],

    "normalized_weights":
        (
            global_best[
                "normalized_weights"
            ]
            .tolist()
        ),

    "best_result": {
        "yield_retention_pct":
            best_yield_pct,

        "mean_field_retention_pct":
            float(
                100.0
                *
                best_replay[
                    "mean_field_retention"
                ]
            ),

        "worst_field_retention_pct":
            float(
                best_replay[
                    "worst_retention_pct"
                ]
            ),

        "jain":
            float(
                best_replay[
                    "jain"
                ]
            ),

        "total_yield_t_ha":
            float(
                best_replay[
                    "total_yield"
                ]
            ),

        "total_irrigation_mm":
            float(
                best_replay[
                    "total_irrigation"
                ]
            ),

        "budget_utilization_pct":
            float(
                best_replay[
                    "budget_utilization_pct"
                ]
            ),
    },

    "comparisons": {
        "equal_yield_retention_pct":
            equal_yield_pct,

        "priority_yield_retention_pct":
            priority_yield_pct,

        "gain_vs_equal_pp":
            float(
                best_vs_equal_pp
            ),

        "gain_vs_priority_pp":
            float(
                best_vs_priority_pp
            ),
    },

    "restart_stability": {
        "mean_yield_retention_pct":
            run_mean,

        "sd_yield_retention_pct":
            run_sd,

        "range_pp":
            run_range,
    },

    "integrity": {
        "post_hoc_diagnostic":
            True,

        "included_in_original_hypothesis_tests":
            False,

        "ppo_training_performed":
            False,

        "ppo_modified":
            False,

        "equal_modified":
            False,

        "priority_modified":
            False,

        "frozen_test_results_modified":
            False,
    },
}


with open(
    BEST_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_json,
        f,
        indent=2,
    )


summary_json = {
    "milestone":
        "56B",

    "status":
        "complete",

    "case":
        f"{CLIMATE}-{YEAR}-{int(100*SCARCITY)}",

    "number_optimizer_restarts":
        len(
            OPTIMIZER_SEEDS
        ),

    "total_episode_evaluations":
        int(
            evaluation_counter[
                "count"
            ]
        ),

    "optimization_runtime_minutes":
        float(
            optimization_runtime
            /
            60.0
        ),

    "parameterization_validation_pass":
        bool(
            parameterization_validation_pass
        ),

    "best_yield_retention_pct":
        best_yield_pct,

    "equal_yield_retention_pct":
        equal_yield_pct,

    "priority_yield_retention_pct":
        priority_yield_pct,

    "gain_vs_equal_pp":
        float(
            best_vs_equal_pp
        ),

    "gain_vs_priority_pp":
        float(
            best_vs_priority_pp
        ),

    "restart_range_pp":
        run_range,

    "global_optimality_guarantee":
        False,

    "mathematical_upper_bound":
        False,
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_json,
        f,
        indent=2,
    )


# ============================================================
# 19. FINAL INTEGRITY REPORT
# ============================================================

print()
print("=" * 78)
print("SCIENTIFIC INTERPRETATION STATUS")
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
    "Post-hoc diagnostic         : YES"
)

print(
    "Original tests modified     : NO"
)

print(
    "PPO retrained               : NO"
)

print(
    "Frozen controllers changed  : NO"
)


print()
print(
    f"Total episode evaluations   : "
    f"{evaluation_counter['count']}"
)


print(
    f"Optimization runtime        : "
    f"{optimization_runtime / 60.0:.2f} min"
)


print()
print("Saved:")

print(
    f"  {RUN_FILE}"
)

print(
    f"  {BEST_FILE}"
)

print(
    f"  {TRACE_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)


print()
print("=" * 78)
print("PERFECT-FORESIGHT SMOKE TEST COMPLETE")
print("=" * 78)