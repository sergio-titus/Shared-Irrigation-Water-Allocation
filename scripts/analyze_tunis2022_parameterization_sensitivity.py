# ============================================================
# TUNIS 2022 PARAMETERIZATION SENSITIVITY
# TUNIS 2022 / 40%
# RICHER PERFECT-FORESIGHT PARAMETERIZATION
#
# CONTINUATION OF 56E
#
# Already completed 4-block independent restarts:
#
#   Seed 5681 -> retained from previous 56E run
#   Seed 5682 -> 67.013042%
#   Seed 5683 -> 67.287007%
#
# Reported 4-block summary:
#   Mean = 67.109709%
#   SD   = 0.153754%
#   Min  = 67.013042%
#   Max  = 67.287007%
#
# Frozen PPO:
#   68.139400%
#
# PURPOSE
# -------
# Test whether increasing temporal flexibility from
# 4 blocks to 6 and 8 blocks closes the remaining gap to PPO.
#
# IMPORTANT
# ---------
# - Does NOT rerun the expensive 4-block restarts.
# - Saves checkpoint immediately after each richer run.
# - Can be safely restarted.
# - No PPO training.
# - No PPO inference.
# - No new hypothesis tests.
# - No global-optimality claim.
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

YEAR = 2022

SCARCITY = 0.40

ENV_SEED = 0


N_FIELDS = 4

EPS = 1e-10


# ------------------------------------------------------------
# Frozen controller results from 56D
# ------------------------------------------------------------

EQUAL_REFERENCE = 65.0109

PRIORITY_REFERENCE = 65.0852

PPO_REFERENCE = 68.1394

PF_four_block_panel_REFERENCE = 67.1959


# ------------------------------------------------------------
# Completed 4-block 56E results
#
# We only need the summary for comparison.
# These runs will NOT be repeated.
# ------------------------------------------------------------

FOUR_BLOCK_MEAN = 67.109709

FOUR_BLOCK_SD = 0.153754

FOUR_BLOCK_MIN = 67.013042

FOUR_BLOCK_MAX = 67.287007

FOUR_BLOCK_RANGE = (
    FOUR_BLOCK_MAX
    -
    FOUR_BLOCK_MIN
)


# ------------------------------------------------------------
# Richer parameterizations
# ------------------------------------------------------------

CONFIGURATIONS = [
    {
        "blocks": 6,
        "seed": 5690,
    },
    {
        "blocks": 8,
        "seed": 5690,
    },
]


# ------------------------------------------------------------
# Same DE settings as 56B/56C/56D/56E
# ------------------------------------------------------------

DE_MAXITER = 8

DE_POPSIZE = 5

DE_TOL = 1e-3

DE_POLISH = False


# ============================================================
# 2. OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path("results")
    / "multiclimate_extension"
    / "offline_perfect_foresight"
    / "tunis2022_parameterization_sensitivity"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "tunis2022_parameterization_richer_checkpoint.csv"
)


WEIGHTS_FILE = (
    OUTPUT_DIR
    / "tunis2022_parameterization_richer_weights.csv"
)


FINAL_FILE = (
    OUTPUT_DIR
    / "tunis2022_parameterization_final_comparison.csv"
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "tunis2022_parameterization_summary.json"
)


# ============================================================
# 3. IMPORT EXACT FROZEN 53J
# ============================================================

script_start = time.perf_counter()


print()

print("=" * 78)

print(
    "TUNIS 2022 PARAMETERIZATION SENSITIVITY"
)

print(
    "TUNIS 2022 / 40% — RICHER PERFECT-FORESIGHT PARAMETERIZATION"
)

print("=" * 78)


if not FINAL_SCRIPT.is_file():

    raise FileNotFoundError(
        f"Cannot find {FINAL_SCRIPT}"
    )


spec = importlib.util.spec_from_file_location(
    "final53j_56eb",
    FINAL_SCRIPT,
)


if (
    spec is None
    or spec.loader is None
):

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


print()

print(
    f"Climate                   : {CLIMATE}"
)

print(
    f"Year                      : {YEAR}"
)

print(
    f"Scarcity                  : {100 * SCARCITY:.0f}%"
)

print(
    f"Equal                     : {EQUAL_REFERENCE:.4f}%"
)

print(
    f"Priority                  : {PRIORITY_REFERENCE:.4f}%"
)

print(
    f"Frozen PPO                : {PPO_REFERENCE:.4f}%"
)

print(
    f"56D PF                    : {PF_four_block_panel_REFERENCE:.4f}%"
)

print(
    f"Best completed 4-block PF : {FOUR_BLOCK_MAX:.6f}%"
)

print(
    f"4-block - PPO gap         : "
    f"{FOUR_BLOCK_MAX - PPO_REFERENCE:+.6f} pp"
)


# ============================================================
# 4. ALLOCATION HELPER
# ============================================================

def allocation_vector(info):

    allocations = info[
        "allocations"
    ]


    if isinstance(
        allocations,
        dict,
    ):

        return np.asarray(
            [
                allocations[field]
                for field in FIELD_NAMES
            ],
            dtype=np.float64,
        )


    arr = np.asarray(
        allocations,
        dtype=np.float64,
    )


    if arr.shape != (
        N_FIELDS,
    ):

        raise RuntimeError(
            f"Unexpected allocation shape: {arr.shape}"
        )


    return arr


# ============================================================
# 5. WEIGHTED WATER-FILLING
# ============================================================

def weighted_water_fill(
    requests_dict,
    available_today,
    weights,
):

    requests = np.asarray(
        [
            float(
                requests_dict[field]
            )
            for field in FIELD_NAMES
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
                active_weights.sum()
            )


        proposed = (
            remaining
            *
            active_weights
            /
            weight_sum
        )


        used = 0.0

        next_active = []


        for (
            local_index,
            field_index,
        ) in enumerate(active):

            unmet = (
                requests[field_index]
                -
                allocations[field_index]
            )


            amount = min(
                float(
                    proposed[local_index]
                ),
                float(unmet),
            )


            allocations[field_index] += amount

            used += amount


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


        remaining -= used


        if used <= EPS:

            break


        active = next_active


    allocations = np.minimum(
        allocations,
        requests,
    )


    if (
        float(
            allocations.sum()
        )
        >
        float(available_today)
        +
        1e-7
    ):

        raise RuntimeError(
            "Weighted allocation exceeded "
            "available water."
        )


    return {
        FIELD_NAMES[i]:
            float(
                allocations[i]
            )
        for i in range(N_FIELDS)
    }


# ============================================================
# 6. NORMALIZE PARAMETERS
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


    for block in range(n_blocks):

        mean_value = float(
            matrix[block].mean()
        )


        if mean_value > EPS:

            matrix[block] /= mean_value


    return matrix


# ============================================================
# 7. TEMPORAL BLOCK
# ============================================================

def get_block_index(
    step_index,
    n_blocks,
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
# 8. PARAMETERIZED PF EPISODE
# ============================================================

def run_pf_episode(
    parameters,
    n_blocks,
):

    weights = normalize_parameters(
        parameters,
        n_blocks,
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

    final_info = None

    step_index = 0

    max_allocation_error = 0.0


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


        block = get_block_index(
            step_index=step_index,
            n_blocks=n_blocks,
        )


        allocations = weighted_water_fill(
            requests_dict=requests,
            available_today=available_today,
            weights=weights[block],
        )


        action = (
            final53j.allocation_to_action(
                requests=requests,
                allocations=allocations,
            )
        )


        intended = np.asarray(
            [
                allocations[field]
                for field in FIELD_NAMES
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


        executed = allocation_vector(
            info
        )


        error = float(
            np.max(
                np.abs(
                    intended
                    -
                    executed
                )
            )
        )


        max_allocation_error = max(
            max_allocation_error,
            error,
        )


        if error > 1e-5:

            raise RuntimeError(
                "Allocation reproduction error: "
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
            "Missing final episode information."
        )


    final_results = final_info.get(
        "final_results"
    )


    if final_results is None:

        raise RuntimeError(
            "Missing final_results."
        )


    seasonal_budget = float(
        env.seasonal_budget
    )


    total_allocated = float(
        env.total_allocated
    )


    budget_utilization = (
        100.0
        *
        total_allocated
        /
        seasonal_budget
        if seasonal_budget > EPS
        else 0.0
    )


    return {

        "Yield_retention_pct":
            100.0
            *
            float(
                final_results[
                    "total_yield_retention"
                ]
            ),

        "Worst_field_retention_pct":
            100.0
            *
            float(
                final_results[
                    "worst_retention"
                ]
            ),

        "Jain":
            float(
                final_results[
                    "jain"
                ]
            ),

        "Water_productivity":
            float(
                final_results[
                    "water_productivity"
                ]
            ),

        "Total_irrigation_mm":
            float(
                final_results[
                    "total_irrigation"
                ]
            ),

        "Seasonal_budget_mm":
            seasonal_budget,

        "Budget_utilization_pct":
            float(
                budget_utilization
            ),

        "Max_allocation_error_mm":
            float(
                max_allocation_error
            ),

        "Weights":
            weights.copy(),
    }


# ============================================================
# 9. VALIDATE 6- AND 8-BLOCK PARAMETERIZATIONS
# ============================================================

print()

print("=" * 78)

print(
    "PARAMETERIZATION VALIDATION"
)

print("=" * 78)


# Get exact Equal again rather than relying only on rounded
# printed value.

equal_exact_result = final53j.run_episode(
    climate=CLIMATE,
    year=YEAR,
    scarcity_fraction=SCARCITY,
    controller="Equal",
)


equal_exact = float(
    equal_exact_result[
        "Yield_retention_pct"
    ]
)


priority_exact_result = final53j.run_episode(
    climate=CLIMATE,
    year=YEAR,
    scarcity_fraction=SCARCITY,
    controller="Priority",
)


priority_exact = float(
    priority_exact_result[
        "Yield_retention_pct"
    ]
)


print(
    f"Exact frozen Equal        : "
    f"{equal_exact:.8f}%"
)

print(
    f"Exact frozen Priority     : "
    f"{priority_exact:.8f}%"
)


for n_blocks in [
    6,
    8,
]:

    all_one = np.ones(
        n_blocks
        *
        N_FIELDS,
        dtype=np.float64,
    )


    validation = run_pf_episode(
        parameters=all_one,
        n_blocks=n_blocks,
    )


    difference = (
        validation[
            "Yield_retention_pct"
        ]
        -
        equal_exact
    )


    print()

    print(
        f"{n_blocks}-block Equal validation"
    )

    print(
        f"  Parameterized           : "
        f"{validation['Yield_retention_pct']:.8f}%"
    )

    print(
        f"  Frozen Equal            : "
        f"{equal_exact:.8f}%"
    )

    print(
        f"  Difference              : "
        f"{difference:+.10f} pp"
    )


    if abs(difference) > 1e-5:

        raise RuntimeError(
            f"{n_blocks}-block Equal validation failed."
        )


# ============================================================
# 10. LOAD CHECKPOINT
# ============================================================

if CHECKPOINT_FILE.is_file():

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )


    print()

    print(
        f"Existing checkpoint       : "
        f"{CHECKPOINT_FILE}"
    )

    print(
        f"Completed richer runs     : "
        f"{len(checkpoint)}"
    )

else:

    checkpoint = pd.DataFrame()


def already_completed(
    n_blocks,
    optimizer_seed,
):

    if checkpoint.empty:

        return False


    required = [
        "Blocks",
        "Optimizer_seed",
    ]


    if any(
        column not in checkpoint.columns
        for column in required
    ):

        return False


    mask = (
        pd.to_numeric(
            checkpoint["Blocks"],
            errors="coerce",
        )
        ==
        int(n_blocks)
    ) & (
        pd.to_numeric(
            checkpoint["Optimizer_seed"],
            errors="coerce",
        )
        ==
        int(optimizer_seed)
    )


    return bool(
        mask.any()
    )


# ============================================================
# 11. OPTIMIZATION FUNCTION
# ============================================================

def optimize_configuration(
    n_blocks,
    optimizer_seed,
):

    n_parameters = (
        n_blocks
        *
        N_FIELDS
    )


    bounds = [
        (
            0.05,
            5.0,
        )
        for _
        in range(n_parameters)
    ]


    eval_count = 0

    best_seen = -np.inf


    def objective(parameters):

        nonlocal eval_count
        nonlocal best_seen


        eval_count += 1


        result = run_pf_episode(
            parameters=parameters,
            n_blocks=n_blocks,
        )


        current_yield = float(
            result[
                "Yield_retention_pct"
            ]
        )


        if current_yield > best_seen:

            best_seen = current_yield


            print(
                f"  {n_blocks} blocks | "
                f"seed {optimizer_seed} | "
                f"eval {eval_count:4d} | "
                f"new best = "
                f"{current_yield:.6f}%"
            )


        return -current_yield


    runtime_start = time.perf_counter()


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        de_result = differential_evolution(
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


    runtime_seconds = (
        time.perf_counter()
        -
        runtime_start
    )


    candidate_result = run_pf_episode(
        parameters=de_result.x,
        n_blocks=n_blocks,
    )


    candidate_yield = float(
        candidate_result[
            "Yield_retention_pct"
        ]
    )


    # --------------------------------------------------------
    # Scientific safeguard:
    #
    # Equal weights are explicitly inside this policy family.
    # Therefore the retained numerical reference should never
    # be worse than the known all-one feasible candidate merely
    # because finite DE happened to return a poorer point.
    # --------------------------------------------------------

    if candidate_yield + 1e-10 < equal_exact:

        print()

        print(
            "WARNING:"
        )

        print(
            "DE candidate was below the feasible "
            "all-one Equal candidate."
        )

        print(
            "Retaining all-one candidate instead."
        )


        selected_parameters = np.ones(
            n_parameters,
            dtype=np.float64,
        )


        selected_result = run_pf_episode(
            parameters=selected_parameters,
            n_blocks=n_blocks,
        )


        selected_source = "Equal_feasible_candidate"

    else:

        selected_parameters = np.asarray(
            de_result.x,
            dtype=np.float64,
        )


        selected_result = candidate_result

        selected_source = "DE"


    normalized_weights = normalize_parameters(
        selected_parameters,
        n_blocks,
    )


    selected_yield = float(
        selected_result[
            "Yield_retention_pct"
        ]
    )


    row = {

        "Blocks":
            int(n_blocks),

        "Parameters":
            int(n_parameters),

        "Optimizer_seed":
            int(optimizer_seed),

        "Selected_source":
            selected_source,

        "Yield_retention_pct":
            selected_yield,

        "Gain_vs_Equal_pp":
            float(
                selected_yield
                -
                equal_exact
            ),

        "Gain_vs_Priority_pp":
            float(
                selected_yield
                -
                priority_exact
            ),

        "Gap_vs_PPO_pp":
            float(
                selected_yield
                -
                PPO_REFERENCE
            ),

        "Worst_field_retention_pct":
            float(
                selected_result[
                    "Worst_field_retention_pct"
                ]
            ),

        "Jain":
            float(
                selected_result[
                    "Jain"
                ]
            ),

        "Water_productivity":
            float(
                selected_result[
                    "Water_productivity"
                ]
            ),

        "Total_irrigation_mm":
            float(
                selected_result[
                    "Total_irrigation_mm"
                ]
            ),

        "Seasonal_budget_mm":
            float(
                selected_result[
                    "Seasonal_budget_mm"
                ]
            ),

        "Budget_utilization_pct":
            float(
                selected_result[
                    "Budget_utilization_pct"
                ]
            ),

        "Max_allocation_error_mm":
            float(
                selected_result[
                    "Max_allocation_error_mm"
                ]
            ),

        "DE_candidate_yield_pct":
            candidate_yield,

        "Optimizer_success":
            bool(
                de_result.success
            ),

        "Optimizer_message":
            str(
                de_result.message
            ),

        "Optimizer_iterations":
            int(
                de_result.nit
            ),

        "Optimizer_function_evaluations":
            int(
                de_result.nfev
            ),

        "Runtime_minutes":
            float(
                runtime_seconds
                /
                60.0
            ),
    }


    weight_rows = []


    for block in range(n_blocks):

        for (
            field_index,
            field_name,
        ) in enumerate(FIELD_NAMES):

            weight_rows.append(
                {
                    "Blocks":
                        int(n_blocks),

                    "Optimizer_seed":
                        int(optimizer_seed),

                    "Block":
                        int(
                            block
                            +
                            1
                        ),

                    "Field":
                        str(field_name),

                    "Normalized_weight":
                        float(
                            normalized_weights[
                                block,
                                field_index,
                            ]
                        ),
                }
            )


    return (
        row,
        weight_rows,
    )


# ============================================================
# 12. RUN 6-BLOCK AND 8-BLOCK ONLY
# ============================================================

weight_records = []


if WEIGHTS_FILE.is_file():

    try:

        existing_weights = pd.read_csv(
            WEIGHTS_FILE
        )


        weight_records = (
            existing_weights
            .to_dict(
                orient="records"
            )
        )

    except Exception:

        weight_records = []


for configuration in CONFIGURATIONS:

    n_blocks = int(
        configuration[
            "blocks"
        ]
    )


    optimizer_seed = int(
        configuration[
            "seed"
        ]
    )


    print()

    print("=" * 78)

    print(
        f"{n_blocks}-BLOCK OPTIMIZATION "
        f"| SEED {optimizer_seed}"
    )

    print("=" * 78)


    if already_completed(
        n_blocks,
        optimizer_seed,
    ):

        print(
            "Checkpoint status          : "
            "ALREADY COMPLETE — SKIPPED"
        )

        continue


    row, new_weight_rows = (
        optimize_configuration(
            n_blocks=n_blocks,
            optimizer_seed=optimizer_seed,
        )
    )


    new_df = pd.DataFrame(
        [
            row
        ]
    )


    if checkpoint.empty:

        checkpoint = new_df.copy()

    else:

        checkpoint = pd.concat(
            [
                checkpoint,
                new_df,
            ],
            ignore_index=True,
        )


    checkpoint = (
        checkpoint
        .drop_duplicates(
            subset=[
                "Blocks",
                "Optimizer_seed",
            ],
            keep="last",
        )
        .sort_values(
            [
                "Blocks",
                "Optimizer_seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # SAVE IMMEDIATELY AFTER THIS RUN
    # --------------------------------------------------------

    checkpoint.to_csv(
        CHECKPOINT_FILE,
        index=False,
    )


    # Remove any old weight rows for same configuration.

    weight_records = [
        item
        for item
        in weight_records
        if not (
            int(item["Blocks"])
            ==
            n_blocks
            and
            int(item["Optimizer_seed"])
            ==
            optimizer_seed
        )
    ]


    weight_records.extend(
        new_weight_rows
    )


    pd.DataFrame(
        weight_records
    ).to_csv(
        WEIGHTS_FILE,
        index=False,
    )


    print()

    print(
        "RUN COMPLETE"
    )

    print(
        f"Yield retention           : "
        f"{row['Yield_retention_pct']:.6f}%"
    )

    print(
        f"Gain vs Equal             : "
        f"{row['Gain_vs_Equal_pp']:+.6f} pp"
    )

    print(
        f"Gain vs Priority          : "
        f"{row['Gain_vs_Priority_pp']:+.6f} pp"
    )

    print(
        f"PF - PPO                  : "
        f"{row['Gap_vs_PPO_pp']:+.6f} pp"
    )

    print(
        f"Worst-field retention     : "
        f"{row['Worst_field_retention_pct']:.6f}%"
    )

    print(
        f"Jain                      : "
        f"{row['Jain']:.6f}"
    )

    print(
        f"Budget utilization        : "
        f"{row['Budget_utilization_pct']:.3f}%"
    )

    print(
        f"Selected source           : "
        f"{row['Selected_source']}"
    )

    print(
        f"Runtime                   : "
        f"{row['Runtime_minutes']:.2f} min"
    )

    print()

    print(
        f"Checkpoint saved          : "
        f"{CHECKPOINT_FILE}"
    )


# ============================================================
# 13. VERIFY BOTH RICHER RUNS COMPLETED
# ============================================================

expected = {
    (
        int(item["blocks"]),
        int(item["seed"]),
    )
    for item in CONFIGURATIONS
}


observed = {
    (
        int(row["Blocks"]),
        int(row["Optimizer_seed"]),
    )
    for _, row in checkpoint.iterrows()
}


missing = (
    expected
    -
    observed
)


if missing:

    print()

    print(
        "56E-B is incomplete."
    )

    print(
        "Missing:"
    )


    for item in sorted(missing):

        print(
            f"  blocks={item[0]}, "
            f"seed={item[1]}"
        )


    print()

    print(
        "Run the same script again. "
        "Completed configurations will be skipped."
    )


    raise RuntimeError(
        "Incomplete richer-parameterization diagnostic."
    )


# ============================================================
# 14. BUILD FINAL COMPARISON
# ============================================================

richer_df = (
    checkpoint[
        checkpoint.apply(
            lambda row:
                (
                    int(row["Blocks"]),
                    int(row["Optimizer_seed"]),
                )
                in expected,
            axis=1,
        )
    ]
    .copy()
    .sort_values(
        "Blocks"
    )
    .reset_index(
        drop=True
    )
)


comparison_rows = [

    {
        "Configuration":
            "Equal",

        "Blocks":
            np.nan,

        "Yield_retention_pct":
            equal_exact,

        "Gap_vs_PPO_pp":
            equal_exact
            -
            PPO_REFERENCE,

        "Status":
            "Frozen lightweight controller",
    },

    {
        "Configuration":
            "Priority",

        "Blocks":
            np.nan,

        "Yield_retention_pct":
            priority_exact,

        "Gap_vs_PPO_pp":
            priority_exact
            -
            PPO_REFERENCE,

        "Status":
            "Frozen lightweight controller",
    },

    {
        "Configuration":
            "PPO",

        "Blocks":
            np.nan,

        "Yield_retention_pct":
            PPO_REFERENCE,

        "Gap_vs_PPO_pp":
            0.0,

        "Status":
            "Frozen PPO climate-year mean",
    },

    {
        "Configuration":
            "PF 56D",

        "Blocks":
            4,

        "Yield_retention_pct":
            PF_four_block_panel_REFERENCE,

        "Gap_vs_PPO_pp":
            PF_four_block_panel_REFERENCE
            -
            PPO_REFERENCE,

        "Status":
            "Original 56D numerical reference",
    },

    {
        "Configuration":
            "PF 4-block best restart",

        "Blocks":
            4,

        "Yield_retention_pct":
            FOUR_BLOCK_MAX,

        "Gap_vs_PPO_pp":
            FOUR_BLOCK_MAX
            -
            PPO_REFERENCE,

        "Status":
            "Best of completed independent 56E restarts",
    },
]


for _, row in richer_df.iterrows():

    comparison_rows.append(
        {
            "Configuration":
                (
                    f"PF {int(row['Blocks'])}-block"
                ),

            "Blocks":
                int(
                    row[
                        "Blocks"
                    ]
                ),

            "Yield_retention_pct":
                float(
                    row[
                        "Yield_retention_pct"
                    ]
                ),

            "Gap_vs_PPO_pp":
                float(
                    row[
                        "Gap_vs_PPO_pp"
                    ]
                ),

            "Status":
                "56E-B richer parameterization",
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(
    FINAL_FILE,
    index=False,
)


# ============================================================
# 15. IDENTIFY BEST PF RESULT
# ============================================================

pf_candidates = [

    {
        "name":
            "PF 56D",

        "blocks":
            4,

        "yield":
            PF_four_block_panel_REFERENCE,
    },

    {
        "name":
            "PF 4-block best restart",

        "blocks":
            4,

        "yield":
            FOUR_BLOCK_MAX,
    },
]


for _, row in richer_df.iterrows():

    pf_candidates.append(
        {
            "name":
                (
                    f"PF {int(row['Blocks'])}-block"
                ),

            "blocks":
                int(
                    row[
                        "Blocks"
                    ]
                ),

            "yield":
                float(
                    row[
                        "Yield_retention_pct"
                    ]
                ),
        }
    )


best_pf = max(
    pf_candidates,
    key=lambda item:
        item[
            "yield"
        ],
)


best_pf_yield = float(
    best_pf[
        "yield"
    ]
)


best_pf_gap_ppo = (
    best_pf_yield
    -
    PPO_REFERENCE
)


# ============================================================
# 16. DIAGNOSTIC CLASSIFICATION
# ============================================================

# 0.05 pp is used only as a descriptive numerical tolerance,
# not as a statistical threshold.

if best_pf_yield >= (
    PPO_REFERENCE
    -
    0.05
):

    diagnosis = (
        "The richer offline numerical search reached the "
        "frozen PPO result within 0.05 percentage points. "
        "The original 56D deficit is therefore consistent "
        "with numerical optimization and/or parameterization "
        "sensitivity rather than evidence that PPO exceeds "
        "the attainable performance of the offline policy family."
    )


elif (
    best_pf[
        "blocks"
    ]
    >
    4
    and
    best_pf_yield
    >
    FOUR_BLOCK_MAX
    +
    0.25
):

    diagnosis = (
        "Increasing temporal parameterization materially "
        "improved the offline numerical reference, indicating "
        "that the original 4-block formulation was restrictive "
        "for Tunis 2022. The benchmark remains a numerical "
        "reference rather than a global optimum."
    )


else:

    diagnosis = (
        "Independent 4-block restarts and richer 6- and "
        "8-block parameterizations did not close the gap to "
        "frozen PPO. The result therefore indicates a "
        "limitation of the restricted weighted water-filling "
        "benchmark family for this climate-year, rather than "
        "a failure of the feasibility implementation. "
        "The benchmark must not be interpreted as a global "
        "optimum or mathematical upper bound."
    )


# ============================================================
# 17. FINAL REPORT
# ============================================================

print()

print("=" * 78)

print(
    "FINAL 56E-B COMPARISON"
)

print("=" * 78)


print(
    comparison_df[
        [
            "Configuration",
            "Yield_retention_pct",
            "Gap_vs_PPO_pp",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print()

print(
    "4-BLOCK RESTART ROBUSTNESS"
)

print(
    f"Mean                     : "
    f"{FOUR_BLOCK_MEAN:.6f}%"
)

print(
    f"SD                       : "
    f"{FOUR_BLOCK_SD:.6f}%"
)

print(
    f"Min                      : "
    f"{FOUR_BLOCK_MIN:.6f}%"
)

print(
    f"Max                      : "
    f"{FOUR_BLOCK_MAX:.6f}%"
)

print(
    f"Range                    : "
    f"{FOUR_BLOCK_RANGE:.6f} pp"
)


print()

print(
    "BEST OFFLINE PF RESULT"
)

print(
    f"Configuration            : "
    f"{best_pf['name']}"
)

print(
    f"Yield retention          : "
    f"{best_pf_yield:.6f}%"
)

print(
    f"Frozen PPO               : "
    f"{PPO_REFERENCE:.6f}%"
)

print(
    f"PF - PPO                 : "
    f"{best_pf_gap_ppo:+.6f} pp"
)


print()

print(
    "DIAGNOSTIC INTERPRETATION"
)

print(
    diagnosis
)


# ============================================================
# 18. INTEGRITY
# ============================================================

max_allocation_error = float(
    pd.to_numeric(
        richer_df[
            "Max_allocation_error_mm"
        ],
        errors="raise",
    ).max()
)


print()

print("=" * 78)

print(
    "SCIENTIFIC STATUS"
)

print("=" * 78)


print(
    "4-block expensive runs rerun : NO"
)

print(
    "PPO retrained                : NO"
)

print(
    "PPO inference rerun           : NO"
)

print(
    "Frozen controllers modified   : NO"
)

print(
    "Original hypothesis tests     : UNCHANGED"
)

print(
    "New hypothesis tests          : NO"
)

print(
    "Perfect foresight             : YES"
)

print(
    "Post-hoc diagnostic           : YES"
)

print(
    "Global-optimality guarantee   : NO"
)

print(
    "Mathematical upper bound      : NO"
)

print(
    f"Max allocation error          : "
    f"{max_allocation_error:.3e} mm"
)


# ============================================================
# 19. SAVE JSON
# ============================================================

total_runtime = (
    time.perf_counter()
    -
    script_start
)


json_summary = {

    "milestone":
        "56E-B",

    "case": {

        "climate":
            CLIMATE,

        "year":
            YEAR,

        "scarcity_fraction":
            SCARCITY,
    },

    "frozen_references": {

        "Equal":
            float(
                equal_exact
            ),

        "Priority":
            float(
                priority_exact
            ),

        "PPO":
            PPO_REFERENCE,

        "PF_four_block_panel":
            PF_four_block_panel_REFERENCE,
    },

    "four_block_restarts": {

        "rerun":
            False,

        "mean":
            FOUR_BLOCK_MEAN,

        "sd":
            FOUR_BLOCK_SD,

        "min":
            FOUR_BLOCK_MIN,

        "max":
            FOUR_BLOCK_MAX,

        "range_pp":
            FOUR_BLOCK_RANGE,
    },

    "richer_results":
        richer_df.to_dict(
            orient="records"
        ),

    "best_offline_pf": {

        "configuration":
            best_pf[
                "name"
            ],

        "blocks":
            int(
                best_pf[
                    "blocks"
                ]
            ),

        "yield_retention_pct":
            best_pf_yield,

        "gap_vs_PPO_pp":
            best_pf_gap_ppo,
    },

    "diagnosis":
        diagnosis,

    "scientific_status": {

        "perfect_foresight":
            True,

        "post_hoc":
            True,

        "global_optimality_guarantee":
            False,

        "mathematical_upper_bound":
            False,

        "ppo_retrained":
            False,

        "ppo_inference_rerun":
            False,

        "new_hypothesis_tests":
            False,
    },

    "runtime_minutes":
        float(
            total_runtime
            /
            60.0
        ),
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        json_summary,
        file,
        indent=2,
    )


# ============================================================
# 20. SAVED FILES
# ============================================================

print()

print(
    f"56E-B runtime             : "
    f"{total_runtime / 60.0:.2f} min"
)


print()

print(
    "Saved:"
)

print(
    f"  {CHECKPOINT_FILE}"
)

print(
    f"  {WEIGHTS_FILE}"
)

print(
    f"  {FINAL_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)


print()

print("=" * 78)

print(
    "TUNIS 2022 PARAMETERIZATION SENSITIVITY COMPLETE"
)

print("=" * 78)