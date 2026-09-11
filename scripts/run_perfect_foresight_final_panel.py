# ============================================================
# PERFECT-FORESIGHT FINAL PANEL
# OFFLINE PERFECT-FORESIGHT NUMERICAL REFERENCE
# ACROSS ALL 21 HELD-OUT CLIMATE-YEARS AT 40% SCARCITY
#
# PURPOSE
# -------
# Extend the validated 4-block × 4-field parameterization from
# Milestones 56B/56C to:
#
#     3 climates × 7 held-out years = 21 climate-years
#
# at the severe 40% seasonal-water availability condition.
#
# Equal/Priority:
#     Exact frozen 53J controllers.
#
# PPO:
#     NOT re-run. Read from frozen final controller panel and
#     averaged across retained PPO seeds within climate-year.
#
# Perfect foresight:
#     Post-hoc offline numerical optimization using the exact
#     frozen AquaCrop final-test environment.
#
# IMPORTANT
# ---------
# The PF reference is NOT:
#     - a globally certified optimum;
#     - a mathematical upper bound;
#     - part of the original Friedman/Wilcoxon tests.
#
# CHECKPOINTING
# -------------
# One row is saved immediately after every completed case.
# If execution stops, simply rerun the same script.
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


CLIMATES = [
    "Cotonou",
    "Niamey",
    "Tunis",
]


YEARS = list(
    range(
        2019,
        2026,
    )
)


SCARCITY = 0.40

ENV_SEED = 0


N_BLOCKS = 4

N_FIELDS = 4

N_PARAMETERS = (
    N_BLOCKS
    *
    N_FIELDS
)


DE_MAXITER = 8

DE_POPSIZE = 5

DE_TOL = 1e-3

DE_POLISH = False


# One common optimization seed for every climate-year.
OPTIMIZER_SEED = 5671


EPS = 1e-10


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

OUTPUT_DIR = (
    Path("results")
    / "multiclimate_extension"
    / "offline_perfect_foresight"
    / "perfect_foresight_40pct_panel"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_panel_case_checkpoint.csv"
)


FINAL_CASE_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_panel_40pct_climate_year_results.csv"
)


CLIMATE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_panel_40pct_climate_summary.csv"
)


OVERALL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_panel_40pct_overall_summary.csv"
)


WEIGHTS_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_panel_best_weights.csv"
)


SUMMARY_JSON_FILE = (
    OUTPUT_DIR
    / "perfect_foresight_panel_summary.json"
)


# ============================================================
# 3. IMPORT EXACT FROZEN 53J EVALUATOR
# ============================================================

script_start = time.perf_counter()


print()

print("=" * 78)

print(
    "PERFECT-FORESIGHT FINAL PANEL"
)

print(
    "PERFECT-FORESIGHT NUMERICAL REFERENCE — "
    "ALL 21 TEST CLIMATE-YEARS AT 40%"
)

print("=" * 78)


if not FINAL_SCRIPT.is_file():

    raise FileNotFoundError(
        f"Cannot find {FINAL_SCRIPT}"
    )


spec = (
    importlib.util
    .spec_from_file_location(
        "final53j_56d",
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


if len(
    FIELD_NAMES
) != N_FIELDS:

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
    f"Climates                  : "
    f"{', '.join(CLIMATES)}"
)


print(
    f"Held-out years            : "
    f"{YEARS[0]}–{YEARS[-1]}"
)


print(
    f"Scarcity                  : "
    f"{100 * SCARCITY:.0f}%"
)


print(
    f"Cases                     : "
    f"{len(CLIMATES) * len(YEARS)}"
)


print(
    f"PF parameters             : "
    f"{N_PARAMETERS}"
)


print(
    f"Optimizer seed            : "
    f"{OPTIMIZER_SEED}"
)


# ============================================================
# 4. FIND FROZEN 53J FINAL CONTROLLER PANEL
#
# IMPORTANT:
# Search only scientifically relevant project locations.
# Do NOT recursively scan the whole C:\Users\User directory.
# ============================================================

def find_final_panel():

    filename = (
        "controller_evaluation_final_controller_panel.csv"
    )


    preferred = [

        (
            Path("results")
            / "multiclimate_extension"
            / "final_evaluation"
            / filename
        ),

        (
            Path("results")
            / "multiclimate_extension"
            / filename
        ),

        (
            Path("results")
            / "multiclimate_extension"
            / "final_test"
            / filename
        ),

        (
            Path("results")
            / "multiclimate_extension"
            / "final"
            / filename
        ),

        (
            Path("results")
            / filename
        ),

        Path(
            filename
        ),
    ]


    for path in preferred:

        if path.is_file():

            return path


    # Safe fallback:
    # recursively search ONLY results/.

    root = Path(
        "results"
    )


    matches = []


    if root.is_dir():

        try:

            for candidate in root.rglob(
                filename
            ):

                if candidate.is_file():

                    matches.append(
                        candidate
                    )

        except (
            FileNotFoundError,
            PermissionError,
            OSError,
        ) as error:

            print(
                f"Warning while searching results/: "
                f"{error}"
            )


    # Remove duplicates.

    unique_matches = []

    seen = set()


    for path in matches:

        try:

            identifier = str(
                path.resolve()
            )

        except OSError:

            identifier = str(
                path
            )


        if identifier not in seen:

            seen.add(
                identifier
            )

            unique_matches.append(
                path
            )


    matches = unique_matches


    if len(
        matches
    ) == 1:

        return matches[0]


    if len(
        matches
    ) > 1:

        print()

        print(
            "Multiple final panels found:"
        )


        for index, path in enumerate(
            matches,
            start=1,
        ):

            print(
                f"  [{index}] {path}"
            )


        raise RuntimeError(
            "Multiple copies of "
            "controller_evaluation_final_controller_panel.csv "
            "were found. Set the exact frozen panel path."
        )


    raise FileNotFoundError(
        "Could not locate "
        "controller_evaluation_final_controller_panel.csv "
        "inside the project results directory."
    )


FINAL_PANEL_PATH = (
    find_final_panel()
)


# ============================================================
# 5. READ FROZEN FINAL PANEL
# ============================================================

final_panel = pd.read_csv(
    FINAL_PANEL_PATH
)


print()

print(
    f"Frozen controller panel   : "
    f"{FINAL_PANEL_PATH}"
)


print(
    f"Frozen panel rows          : "
    f"{len(final_panel)}"
)


# ============================================================
# 6. VALIDATE REQUIRED PANEL COLUMNS
# ============================================================

required_panel_columns = [
    "Controller",
    "Climate",
    "Year",
    "Yield_retention_pct",
]


missing_panel_columns = [
    column
    for column
    in required_panel_columns
    if column
    not in final_panel.columns
]


if missing_panel_columns:

    print()

    print(
        "Available panel columns:"
    )


    for column in final_panel.columns:

        print(
            f"  {column}"
        )


    raise RuntimeError(
        "Frozen final panel is missing required columns: "
        f"{missing_panel_columns}"
    )


if (
    "Scarcity_fraction"
    not in final_panel.columns
    and
    "Scarcity_pct"
    not in final_panel.columns
):

    raise RuntimeError(
        "Frozen final panel has neither "
        "'Scarcity_fraction' nor 'Scarcity_pct'."
    )


# ============================================================
# 7. SELECT THE 40% FINAL-TEST PANEL
# ============================================================

if (
    "Scarcity_fraction"
    in final_panel.columns
):

    scarcity_values = pd.to_numeric(
        final_panel[
            "Scarcity_fraction"
        ],
        errors="coerce",
    )


    scarcity_mask = np.isclose(
        scarcity_values,
        SCARCITY,
    )

else:

    scarcity_values = pd.to_numeric(
        final_panel[
            "Scarcity_pct"
        ],
        errors="coerce",
    )


    scarcity_mask = np.isclose(
        scarcity_values,
        100.0
        *
        SCARCITY,
    )


panel40 = final_panel.loc[
    scarcity_mask
].copy()


if panel40.empty:

    raise RuntimeError(
        "No 40% scarcity rows were found "
        "in the frozen final panel."
    )


panel40[
    "Year"
] = pd.to_numeric(
    panel40[
        "Year"
    ],
    errors="raise",
).astype(
    int
)


print(
    f"Frozen 40% panel rows      : "
    f"{len(panel40)}"
)


# ============================================================
# 8. DETECT PPO CONTROLLER LABEL
# ============================================================

controller_names = (
    panel40[
        "Controller"
    ]
    .astype(str)
    .unique()
    .tolist()
)


print()

print(
    "Controllers in frozen 40% panel:"
)


for controller_name in controller_names:

    print(
        f"  {controller_name}"
    )


ppo_candidates = [
    name
    for name
    in controller_names
    if "ppo"
    in name.lower()
]


if len(
    ppo_candidates
) != 1:

    raise RuntimeError(
        "Expected exactly one PPO controller label "
        f"in the frozen panel; found {ppo_candidates}"
    )


PPO_LABEL = (
    ppo_candidates[0]
)


print()

print(
    f"Frozen PPO label          : "
    f"{PPO_LABEL}"
)


# ============================================================
# 9. HELPER — ALLOCATION VECTOR
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


    array = np.asarray(
        allocations,
        dtype=np.float64,
    )


    if array.shape != (
        N_FIELDS,
    ):

        raise RuntimeError(
            f"Unexpected allocation shape: "
            f"{array.shape}"
        )


    return array


# ============================================================
# 10. WEIGHTED WATER-FILLING
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
        index
        for index
        in range(
            N_FIELDS
        )
        if requests[
            index
        ] > EPS
    ]


    while (
        active
        and
        remaining > EPS
    ):

        active_weights = np.asarray(
            [
                weights[
                    index
                ]
                for index
                in active
            ],
            dtype=np.float64,
        )


        weight_sum = float(
            active_weights.sum()
        )


        if weight_sum <= EPS:

            active_weights = np.ones(
                len(
                    active
                ),
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


            used += amount


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


        remaining -= used


        if used <= EPS:

            break


        active = next_active


    allocations = np.minimum(
        allocations,
        requests,
    )


    total_allocation = float(
        allocations.sum()
    )


    if (
        total_allocation
        >
        float(
            available_today
        )
        +
        1e-7
    ):

        raise RuntimeError(
            "Weighted water-filling exceeded "
            "available daily water."
        )


    return {
        FIELD_NAMES[
            index
        ]:
            float(
                allocations[
                    index
                ]
            )
        for index
        in range(
            N_FIELDS
        )
    }


# ============================================================
# 11. NORMALIZE PARAMETERS
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


    for block in range(
        N_BLOCKS
    ):

        mean_value = float(
            matrix[
                block
            ].mean()
        )


        if mean_value > EPS:

            matrix[
                block
            ] /= mean_value


    return matrix


# ============================================================
# 12. TEMPORAL BLOCK
# ============================================================

def get_block_index(
    step_index,
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
# 13. PERFECT-FORESIGHT EPISODE
# ============================================================

evaluation_counter = {}


def run_pf_episode(
    climate,
    year,
    parameters,
):

    key = (
        str(
            climate
        ),
        int(
            year
        ),
    )


    evaluation_counter[
        key
    ] = (
        evaluation_counter.get(
            key,
            0,
        )
        +
        1
    )


    weights = normalize_parameters(
        parameters
    )


    env = (
        final53j.FinalTestEnv(
            climate=climate,
            year=int(
                year
            ),
            scarcity_fraction=SCARCITY,
            seed=ENV_SEED,
        )
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
            step_index
        )


        allocations = (
            weighted_water_fill(
                requests_dict=requests,
                available_today=available_today,
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


        executed = allocation_vector(
            info
        )


        allocation_error = float(
            np.max(
                np.abs(
                    executed
                    -
                    desired
                )
            )
        )


        max_allocation_error = max(
            max_allocation_error,
            allocation_error,
        )


        if allocation_error > 1e-5:

            raise RuntimeError(
                "Unexpected environment modification "
                f"for {climate} {year}: "
                f"{allocation_error:.6e} mm"
            )


        final_info = info

        step_index += 1


    if truncated:

        raise RuntimeError(
            f"Unexpected truncation: "
            f"{climate} {year}"
        )


    if final_info is None:

        raise RuntimeError(
            f"Missing final info: "
            f"{climate} {year}"
        )


    final_results = (
        final_info.get(
            "final_results"
        )
    )


    if final_results is None:

        raise RuntimeError(
            f"Missing final_results: "
            f"{climate} {year}"
        )


    seasonal_budget = float(
        env.seasonal_budget
    )


    total_allocated = float(
        env.total_allocated
    )


    budget_utilization = (
        total_allocated
        /
        seasonal_budget
        *
        100.0
        if seasonal_budget > EPS
        else 0.0
    )


    return {

        "Yield_retention_pct":
            (
                100.0
                *
                float(
                    final_results[
                        "total_yield_retention"
                    ]
                )
            ),

        "Mean_field_retention_pct":
            (
                100.0
                *
                float(
                    final_results[
                        "mean_retention"
                    ]
                )
            ),

        "Worst_field_retention_pct":
            (
                100.0
                *
                float(
                    final_results[
                        "worst_retention"
                    ]
                )
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

        "Total_yield_t_ha":
            float(
                final_results[
                    "total_yield"
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

        "Budget_binding":
            bool(
                env.budget_exhaustion_step
                is not None
            ),

        "Budget_exhaustion_step":
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

        "Max_allocation_error_mm":
            float(
                max_allocation_error
            ),

        "Weights":
            weights.copy(),
    }


# ============================================================
# 14. EXTRACT FROZEN PPO RESULT FOR ONE CLIMATE-YEAR
# ============================================================

def frozen_ppo_case(
    climate,
    year,
):

    subset = panel40[
        (
            panel40[
                "Climate"
            ].astype(str)
            ==
            str(
                climate
            )
        )
        &
        (
            panel40[
                "Year"
            ].astype(int)
            ==
            int(
                year
            )
        )
        &
        (
            panel40[
                "Controller"
            ].astype(str)
            ==
            str(
                PPO_LABEL
            )
        )
    ].copy()


    if subset.empty:

        raise RuntimeError(
            f"No frozen PPO result for "
            f"{climate} {year}."
        )


    result = {

        "PPO_rows":
            int(
                len(
                    subset
                )
            ),

        "PPO_Yield_retention_pct":
            float(
                pd.to_numeric(
                    subset[
                        "Yield_retention_pct"
                    ],
                    errors="raise",
                ).mean()
            ),
    }


    optional_mapping = {

        "Worst_field_retention_pct":
            "PPO_Worst_field_retention_pct",

        "Jain":
            "PPO_Jain",

        "Water_productivity":
            "PPO_Water_productivity",

        "Total_irrigation_mm":
            "PPO_Total_irrigation_mm",

        "Budget_utilization_pct":
            "PPO_Budget_utilization_pct",
    }


    for (
        source_column,
        output_name,
    ) in optional_mapping.items():

        if source_column in subset.columns:

            result[
                output_name
            ] = float(
                pd.to_numeric(
                    subset[
                        source_column
                    ],
                    errors="coerce",
                ).mean()
            )

        else:

            result[
                output_name
            ] = np.nan


    return result


# ============================================================
# 15. EXACT FROZEN LIGHTWEIGHT CONTROLLERS
# ============================================================

def frozen_lightweight_case(
    climate,
    year,
):

    equal = (
        final53j.run_episode(
            climate=climate,
            year=int(
                year
            ),
            scarcity_fraction=SCARCITY,
            controller="Equal",
        )
    )


    priority = (
        final53j.run_episode(
            climate=climate,
            year=int(
                year
            ),
            scarcity_fraction=SCARCITY,
            controller="Priority",
        )
    )


    return (
        equal,
        priority,
    )


# ============================================================
# 16. LOAD CHECKPOINT
# ============================================================

if CHECKPOINT_FILE.is_file():

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )


    print()

    print(
        f"Existing checkpoint found : "
        f"{CHECKPOINT_FILE}"
    )


    print(
        f"Completed rows             : "
        f"{len(checkpoint)}"
    )

else:

    checkpoint = pd.DataFrame()


def case_completed(
    climate,
    year,
):

    if checkpoint.empty:

        return False


    if (
        "Climate"
        not in checkpoint.columns
        or
        "Year"
        not in checkpoint.columns
    ):

        return False


    mask = (
        checkpoint[
            "Climate"
        ].astype(str)
        ==
        str(
            climate
        )
    ) & (
        pd.to_numeric(
            checkpoint[
                "Year"
            ],
            errors="coerce",
        )
        ==
        int(
            year
        )
    )


    return bool(
        mask.any()
    )


# ============================================================
# 17. OPTIMIZATION BOUNDS
# ============================================================

bounds = [
    (
        0.05,
        5.0,
    )
    for _
    in range(
        N_PARAMETERS
    )
]


# ============================================================
# 18. LOOP THROUGH ALL 21 CLIMATE-YEARS
# ============================================================

total_cases = (
    len(
        CLIMATES
    )
    *
    len(
        YEARS
    )
)


case_number = 0


for climate in CLIMATES:

    for year in YEARS:

        case_number += 1


        print()

        print("=" * 78)

        print(
            f"CASE {case_number}/{total_cases}: "
            f"{climate} {year} | 40%"
        )

        print("=" * 78)


        if case_completed(
            climate,
            year,
        ):

            print(
                "Checkpoint status          : "
                "ALREADY COMPLETE — SKIPPED"
            )

            continue


        # ====================================================
        # Frozen lightweight controllers
        # ====================================================

        print(
            "Running frozen Equal/Priority..."
        )


        (
            equal_result,
            priority_result,
        ) = frozen_lightweight_case(
            climate=climate,
            year=year,
        )


        equal_yield = float(
            equal_result[
                "Yield_retention_pct"
            ]
        )


        priority_yield = float(
            priority_result[
                "Yield_retention_pct"
            ]
        )


        # ====================================================
        # Frozen PPO from panel
        # ====================================================

        ppo_result = frozen_ppo_case(
            climate=climate,
            year=year,
        )


        ppo_yield = float(
            ppo_result[
                "PPO_Yield_retention_pct"
            ]
        )


        print(
            f"Equal                     : "
            f"{equal_yield:.4f}%"
        )


        print(
            f"Priority                  : "
            f"{priority_yield:.4f}%"
        )


        print(
            f"Frozen PPO mean           : "
            f"{ppo_yield:.4f}% "
            f"(rows={ppo_result['PPO_rows']})"
        )


        # ====================================================
        # Validate equal-weight PF parameterization
        # ====================================================

        equal_parameters = np.ones(
            N_PARAMETERS,
            dtype=np.float64,
        )


        validation = run_pf_episode(
            climate=climate,
            year=year,
            parameters=equal_parameters,
        )


        validation_difference = (
            float(
                validation[
                    "Yield_retention_pct"
                ]
            )
            -
            equal_yield
        )


        validation_pass = bool(
            abs(
                validation_difference
            )
            <
            1e-5
        )


        print()

        print(
            "PF parameterization validation:"
        )


        print(
            f"  Parameterized Equal      : "
            f"{validation['Yield_retention_pct']:.6f}%"
        )


        print(
            f"  Frozen Equal             : "
            f"{equal_yield:.6f}%"
        )


        print(
            f"  Difference               : "
            f"{validation_difference:+.8f} pp"
        )


        print(
            f"  Status                   : "
            f"{'PASS' if validation_pass else 'FAIL'}"
        )


        if not validation_pass:

            raise RuntimeError(
                f"Equal validation failed for "
                f"{climate} {year}."
            )


        # ====================================================
        # Objective
        # ====================================================

        case_key = (
            str(
                climate
            ),
            int(
                year
            ),
        )


        best_seen = {
            "yield":
                -np.inf,
        }


        def objective(
            parameters,
        ):

            result = run_pf_episode(
                climate=climate,
                year=year,
                parameters=parameters,
            )


            current_yield = float(
                result[
                    "Yield_retention_pct"
                ]
            )


            if (
                current_yield
                >
                best_seen[
                    "yield"
                ]
            ):

                best_seen[
                    "yield"
                ] = current_yield


                print(
                    f"  eval "
                    f"{evaluation_counter[case_key]:4d} | "
                    f"new best = "
                    f"{current_yield:.4f}%"
                )


            return -float(
                result[
                    "Yield_retention_pct"
                ]
            )


        # ====================================================
        # Differential evolution
        # ====================================================

        evaluations_before = (
            evaluation_counter.get(
                case_key,
                0,
            )
        )


        optimization_start = (
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
                    seed=OPTIMIZER_SEED,
                    polish=DE_POLISH,
                    updating="immediate",
                    workers=1,
                    disp=False,
                )
            )


        optimization_runtime = (
            time.perf_counter()
            -
            optimization_start
        )


        evaluations_after = (
            evaluation_counter[
                case_key
            ]
        )


        measured_evaluations = (
            evaluations_after
            -
            evaluations_before
        )


        # ====================================================
        # Replay selected PF solution
        # ====================================================

        pf_result = run_pf_episode(
            climate=climate,
            year=year,
            parameters=de_result.x,
        )


        pf_yield = float(
            pf_result[
                "Yield_retention_pct"
            ]
        )


        normalized_weights = (
            normalize_parameters(
                de_result.x
            )
        )


        gap_equal = (
            pf_yield
            -
            equal_yield
        )


        gap_priority = (
            pf_yield
            -
            priority_yield
        )


        gap_ppo = (
            pf_yield
            -
            ppo_yield
        )


        print()

        print(
            "CASE RESULT"
        )


        print(
            f"PF numerical reference    : "
            f"{pf_yield:.6f}%"
        )


        print(
            f"Gap PF - Equal            : "
            f"{gap_equal:+.6f} pp"
        )


        print(
            f"Gap PF - Priority         : "
            f"{gap_priority:+.6f} pp"
        )


        print(
            f"Gap PF - PPO              : "
            f"{gap_ppo:+.6f} pp"
        )


        print(
            f"PF worst-field retention  : "
            f"{pf_result['Worst_field_retention_pct']:.6f}%"
        )


        print(
            f"PF Jain                   : "
            f"{pf_result['Jain']:.6f}"
        )


        print(
            f"PF budget utilization     : "
            f"{pf_result['Budget_utilization_pct']:.3f}%"
        )


        print(
            f"DE reported evaluations   : "
            f"{de_result.nfev}"
        )


        print(
            f"Measured evaluations      : "
            f"{measured_evaluations}"
        )


        print(
            f"Runtime                   : "
            f"{optimization_runtime / 60.0:.2f} min"
        )


        # ====================================================
        # Build checkpoint row
        # ====================================================

        row = {

            "Climate":
                str(
                    climate
                ),

            "Year":
                int(
                    year
                ),

            "Scarcity_pct":
                40.0,

            "PF_Yield_retention_pct":
                pf_yield,

            "Equal_Yield_retention_pct":
                equal_yield,

            "Priority_Yield_retention_pct":
                priority_yield,

            "PPO_Yield_retention_pct":
                ppo_yield,

            "PF_minus_Equal_pp":
                float(
                    gap_equal
                ),

            "PF_minus_Priority_pp":
                float(
                    gap_priority
                ),

            "PF_minus_PPO_pp":
                float(
                    gap_ppo
                ),

            "PF_Mean_field_retention_pct":
                float(
                    pf_result[
                        "Mean_field_retention_pct"
                    ]
                ),

            "PF_Worst_field_retention_pct":
                float(
                    pf_result[
                        "Worst_field_retention_pct"
                    ]
                ),

            "PF_Jain":
                float(
                    pf_result[
                        "Jain"
                    ]
                ),

            "PF_Water_productivity":
                float(
                    pf_result[
                        "Water_productivity"
                    ]
                ),

            "PF_Total_yield_t_ha":
                float(
                    pf_result[
                        "Total_yield_t_ha"
                    ]
                ),

            "PF_Total_irrigation_mm":
                float(
                    pf_result[
                        "Total_irrigation_mm"
                    ]
                ),

            "PF_Seasonal_budget_mm":
                float(
                    pf_result[
                        "Seasonal_budget_mm"
                    ]
                ),

            "PF_Budget_utilization_pct":
                float(
                    pf_result[
                        "Budget_utilization_pct"
                    ]
                ),

            "PF_Budget_binding":
                bool(
                    pf_result[
                        "Budget_binding"
                    ]
                ),

            "PF_Budget_exhaustion_step":
                (
                    float(
                        pf_result[
                            "Budget_exhaustion_step"
                        ]
                    )
                    if
                    pf_result[
                        "Budget_exhaustion_step"
                    ]
                    is not None
                    else
                    np.nan
                ),

            "PF_Max_allocation_error_mm":
                float(
                    pf_result[
                        "Max_allocation_error_mm"
                    ]
                ),

            "PPO_rows_averaged":
                int(
                    ppo_result[
                        "PPO_rows"
                    ]
                ),

            "PPO_Worst_field_retention_pct":
                float(
                    ppo_result[
                        "PPO_Worst_field_retention_pct"
                    ]
                ),

            "PPO_Jain":
                float(
                    ppo_result[
                        "PPO_Jain"
                    ]
                ),

            "PPO_Water_productivity":
                float(
                    ppo_result[
                        "PPO_Water_productivity"
                    ]
                ),

            "Equal_Worst_field_retention_pct":
                float(
                    equal_result[
                        "Worst_field_retention_pct"
                    ]
                ),

            "Equal_Jain":
                float(
                    equal_result[
                        "Jain"
                    ]
                ),

            "Priority_Worst_field_retention_pct":
                float(
                    priority_result[
                        "Worst_field_retention_pct"
                    ]
                ),

            "Priority_Jain":
                float(
                    priority_result[
                        "Jain"
                    ]
                ),

            "Optimizer_seed":
                int(
                    OPTIMIZER_SEED
                ),

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

            "Measured_episode_evaluations":
                int(
                    measured_evaluations
                ),

            "Runtime_minutes":
                float(
                    optimization_runtime
                    /
                    60.0
                ),

            "Equal_parameterization_validation_difference_pp":
                float(
                    validation_difference
                ),

            "Equal_parameterization_validation_pass":
                bool(
                    validation_pass
                ),
        }


        # Save normalized weights.

        for block in range(
            N_BLOCKS
        ):

            for (
                field_index,
                field_name,
            ) in enumerate(
                FIELD_NAMES
            ):

                row[
                    f"B{block + 1}_{field_name}_weight"
                ] = float(
                    normalized_weights[
                        block,
                        field_index,
                    ]
                )


        new_row_df = pd.DataFrame(
            [
                row
            ]
        )


        if checkpoint.empty:

            checkpoint = (
                new_row_df.copy()
            )

        else:

            checkpoint = pd.concat(
                [
                    checkpoint,
                    new_row_df,
                ],
                ignore_index=True,
            )


        checkpoint = (
            checkpoint
            .drop_duplicates(
                subset=[
                    "Climate",
                    "Year",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "Climate",
                    "Year",
                ]
            )
            .reset_index(
                drop=True
            )
        )


        checkpoint.to_csv(
            CHECKPOINT_FILE,
            index=False,
        )


        print()

        print(
            f"Checkpoint saved          : "
            f"{CHECKPOINT_FILE}"
        )


# ============================================================
# 19. VERIFY COMPLETE PANEL
# ============================================================

expected_cases = {
    (
        str(
            climate
        ),
        int(
            year
        ),
    )
    for climate
    in CLIMATES
    for year
    in YEARS
}


observed_cases = {
    (
        str(
            row[
                "Climate"
            ]
        ),
        int(
            row[
                "Year"
            ]
        ),
    )
    for _, row
    in checkpoint.iterrows()
}


missing_cases = (
    expected_cases
    -
    observed_cases
)


if missing_cases:

    print()

    print("=" * 78)

    print(
        "PERFECT-FORESIGHT FINAL PANEL INCOMPLETE"
    )

    print("=" * 78)


    print(
        "Missing climate-years:"
    )


    for climate, year in sorted(
        missing_cases
    ):

        print(
            f"  {climate} {year}"
        )


    raise RuntimeError(
        "56D is incomplete. "
        "Run the same script again; "
        "completed cases will be skipped."
    )


# ============================================================
# 20. KEEP EXACTLY THE 21 EXPECTED CASES
# ============================================================

mask_expected = checkpoint.apply(
    lambda row:
        (
            str(
                row[
                    "Climate"
                ]
            ),
            int(
                row[
                    "Year"
                ]
            ),
        )
        in expected_cases,
    axis=1,
)


case_df = (
    checkpoint.loc[
        mask_expected
    ]
    .copy()
    .sort_values(
        [
            "Climate",
            "Year",
        ]
    )
    .reset_index(
        drop=True
    )
)


if len(
    case_df
) != 21:

    raise RuntimeError(
        f"Expected exactly 21 final cases, "
        f"found {len(case_df)}."
    )


case_df.to_csv(
    FINAL_CASE_FILE,
    index=False,
)


# ============================================================
# 21. SAVE WEIGHTS IN LONG FORM
# ============================================================

weight_records = []


for _, row in case_df.iterrows():

    for block in range(
        N_BLOCKS
    ):

        for field in (
            FIELD_NAMES
        ):

            column = (
                f"B{block + 1}_"
                f"{field}_weight"
            )


            weight_records.append(
                {
                    "Climate":
                        str(
                            row[
                                "Climate"
                            ]
                        ),

                    "Year":
                        int(
                            row[
                                "Year"
                            ]
                        ),

                    "Block":
                        int(
                            block
                            +
                            1
                        ),

                    "Field":
                        str(
                            field
                        ),

                    "Normalized_weight":
                        float(
                            row[
                                column
                            ]
                        ),
                }
            )


weights_df = pd.DataFrame(
    weight_records
)


weights_df.to_csv(
    WEIGHTS_FILE,
    index=False,
)


# ============================================================
# 22. CLIMATE-LEVEL DESCRIPTIVE SUMMARY
# ============================================================

climate_records = []


summary_metrics = [

    "PF_Yield_retention_pct",

    "Equal_Yield_retention_pct",

    "Priority_Yield_retention_pct",

    "PPO_Yield_retention_pct",

    "PF_minus_Equal_pp",

    "PF_minus_Priority_pp",

    "PF_minus_PPO_pp",

    "PF_Worst_field_retention_pct",

    "PF_Jain",

    "PF_Budget_utilization_pct",
]


for climate in CLIMATES:

    subset = case_df[
        case_df[
            "Climate"
        ].astype(str)
        ==
        str(
            climate
        )
    ].copy()


    record = {

        "Climate":
            str(
                climate
            ),

        "N_years":
            int(
                len(
                    subset
                )
            ),
    }


    for metric in summary_metrics:

        values = pd.to_numeric(
            subset[
                metric
            ],
            errors="coerce",
        ).dropna()


        record[
            f"{metric}_mean"
        ] = float(
            values.mean()
        )


        record[
            f"{metric}_sd"
        ] = (
            float(
                values.std(
                    ddof=1
                )
            )
            if len(
                values
            )
            >
            1
            else
            0.0
        )


        record[
            f"{metric}_median"
        ] = float(
            values.median()
        )


        record[
            f"{metric}_min"
        ] = float(
            values.min()
        )


        record[
            f"{metric}_max"
        ] = float(
            values.max()
        )


    climate_records.append(
        record
    )


climate_summary = pd.DataFrame(
    climate_records
)


climate_summary.to_csv(
    CLIMATE_SUMMARY_FILE,
    index=False,
)


# ============================================================
# 23. OVERALL DESCRIPTIVE SUMMARY
# ============================================================

overall_records = []


for metric in summary_metrics:

    values = pd.to_numeric(
        case_df[
            metric
        ],
        errors="coerce",
    ).dropna()


    overall_records.append(
        {
            "Metric":
                metric,

            "N":
                int(
                    len(
                        values
                    )
                ),

            "Mean":
                float(
                    values.mean()
                ),

            "SD":
                float(
                    values.std(
                        ddof=1
                    )
                ),

            "Median":
                float(
                    values.median()
                ),

            "Q1":
                float(
                    values.quantile(
                        0.25
                    )
                ),

            "Q3":
                float(
                    values.quantile(
                        0.75
                    )
                ),

            "Min":
                float(
                    values.min()
                ),

            "Max":
                float(
                    values.max()
                ),
        }
    )


overall_summary = pd.DataFrame(
    overall_records
)


overall_summary.to_csv(
    OVERALL_SUMMARY_FILE,
    index=False,
)


# ============================================================
# 24. CONSISTENCY DIAGNOSTICS
# ============================================================

max_equal_validation_error = float(
    np.max(
        np.abs(
            pd.to_numeric(
                case_df[
                    "Equal_parameterization_validation_difference_pp"
                ],
                errors="raise",
            )
        )
    )
)


max_pf_allocation_error = float(
    pd.to_numeric(
        case_df[
            "PF_Max_allocation_error_mm"
        ],
        errors="raise",
    ).max()
)


ppo_row_counts = sorted(
    pd.to_numeric(
        case_df[
            "PPO_rows_averaged"
        ],
        errors="raise",
    )
    .astype(int)
    .unique()
    .tolist()
)


pf_below_equal = int(
    (
        case_df[
            "PF_minus_Equal_pp"
        ].astype(float)
        <
        -1e-6
    ).sum()
)


pf_below_priority = int(
    (
        case_df[
            "PF_minus_Priority_pp"
        ].astype(float)
        <
        -1e-6
    ).sum()
)


pf_below_ppo = int(
    (
        case_df[
            "PF_minus_PPO_pp"
        ].astype(float)
        <
        -1e-6
    ).sum()
)


# ============================================================
# 25. PRINT FINAL CLIMATE-YEAR PANEL
# ============================================================

print()

print("=" * 78)

print(
    "FINAL 40% CLIMATE-YEAR PANEL"
)

print("=" * 78)


display_columns = [

    "Climate",

    "Year",

    "PF_Yield_retention_pct",

    "Equal_Yield_retention_pct",

    "Priority_Yield_retention_pct",

    "PPO_Yield_retention_pct",

    "PF_minus_Equal_pp",

    "PF_minus_Priority_pp",

    "PF_minus_PPO_pp",
]


print(
    case_df[
        display_columns
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 26. PRINT CLIMATE-LEVEL SUMMARY
# ============================================================

print()

print("=" * 78)

print(
    "CLIMATE-LEVEL SUMMARY"
)

print("=" * 78)


for climate in CLIMATES:

    subset = case_df[
        case_df[
            "Climate"
        ].astype(str)
        ==
        str(
            climate
        )
    ]


    print()

    print(
        climate.upper()
    )


    for column, label in [

        (
            "PF_Yield_retention_pct",
            "PF yield",
        ),

        (
            "Equal_Yield_retention_pct",
            "Equal",
        ),

        (
            "Priority_Yield_retention_pct",
            "Priority",
        ),

        (
            "PPO_Yield_retention_pct",
            "PPO",
        ),
    ]:

        values = subset[
            column
        ].astype(
            float
        )


        print(
            f"  {label:24s}: "
            f"{values.mean():.4f}"
            f" ± "
            f"{values.std(ddof=1):.4f}%"
        )


    print(
        f"  {'PF - Equal gap':24s}: "
        f"{subset['PF_minus_Equal_pp'].mean():+.4f}"
        f" ± "
        f"{subset['PF_minus_Equal_pp'].std(ddof=1):.4f} pp"
    )


    print(
        f"  {'PF - Priority gap':24s}: "
        f"{subset['PF_minus_Priority_pp'].mean():+.4f}"
        f" ± "
        f"{subset['PF_minus_Priority_pp'].std(ddof=1):.4f} pp"
    )


    print(
        f"  {'PF - PPO gap':24s}: "
        f"{subset['PF_minus_PPO_pp'].mean():+.4f}"
        f" ± "
        f"{subset['PF_minus_PPO_pp'].std(ddof=1):.4f} pp"
    )


# ============================================================
# 27. PRINT OVERALL SUMMARY
# ============================================================

print()

print("=" * 78)

print(
    "OVERALL 21-CLIMATE-YEAR SUMMARY"
)

print("=" * 78)


for column, label in [

    (
        "PF_Yield_retention_pct",
        "PF numerical reference",
    ),

    (
        "Equal_Yield_retention_pct",
        "Equal",
    ),

    (
        "Priority_Yield_retention_pct",
        "Priority",
    ),

    (
        "PPO_Yield_retention_pct",
        "PPO",
    ),
]:

    values = case_df[
        column
    ].astype(
        float
    )


    print(
        f"{label:25s}: "
        f"{values.mean():.4f}"
        f" ± "
        f"{values.std(ddof=1):.4f}%"
    )


print()

print(
    f"PF - Equal mean gap       : "
    f"{case_df['PF_minus_Equal_pp'].mean():+.4f} pp"
)


print(
    f"PF - Priority mean gap    : "
    f"{case_df['PF_minus_Priority_pp'].mean():+.4f} pp"
)


print(
    f"PF - PPO mean gap         : "
    f"{case_df['PF_minus_PPO_pp'].mean():+.4f} pp"
)


print()

print(
    f"PF below Equal cases      : "
    f"{pf_below_equal}/21"
)


print(
    f"PF below Priority cases   : "
    f"{pf_below_priority}/21"
)


print(
    f"PF below PPO cases        : "
    f"{pf_below_ppo}/21"
)


# ============================================================
# 28. INTEGRITY AND VALIDATION
# ============================================================

print()

print("=" * 78)

print(
    "INTEGRITY AND VALIDATION"
)

print("=" * 78)


print(
    f"Completed cases            : "
    f"{len(case_df)}/21"
)


print(
    f"PPO rows/case encountered  : "
    f"{ppo_row_counts}"
)


print(
    f"Max Equal validation error : "
    f"{max_equal_validation_error:.3e} pp"
)


print(
    f"Max PF allocation error    : "
    f"{max_pf_allocation_error:.3e} mm"
)


print(
    "PPO training performed     : NO"
)


print(
    "PPO inference performed    : NO"
)


print(
    "Frozen PPO panel modified  : NO"
)


print(
    "Equal/Priority modified     : NO"
)


print(
    "Original hypothesis tests  : UNCHANGED"
)


print(
    "New hypothesis tests       : NO"
)


print(
    "Global-optimality guarantee: NO"
)


print(
    "Mathematical upper bound   : NO"
)


print(
    "Perfect-foresight reference: YES"
)


print(
    "Post-hoc numerical analysis: YES"
)


# ============================================================
# 29. SAVE JSON SUMMARY
# ============================================================

total_runtime = (
    time.perf_counter()
    -
    script_start
)


summary_json = {

    "milestone":
        "56D",

    "status":
        "complete",

    "scope": {

        "climates":
            CLIMATES,

        "years":
            YEARS,

        "scarcity_fraction":
            SCARCITY,

        "climate_years":
            int(
                len(
                    case_df
                )
            ),
    },

    "perfect_foresight": {

        "temporal_blocks":
            N_BLOCKS,

        "fields":
            N_FIELDS,

        "parameters":
            N_PARAMETERS,

        "objective":
            "maximize total_yield_retention",

        "optimizer":
            "SciPy differential_evolution",

        "optimizer_seed":
            OPTIMIZER_SEED,

        "maxiter":
            DE_MAXITER,

        "popsize":
            DE_POPSIZE,

        "global_optimality_guarantee":
            False,

        "mathematical_upper_bound":
            False,
    },

    "overall_yield_retention_pct": {

        "perfect_foresight_mean":
            float(
                case_df[
                    "PF_Yield_retention_pct"
                ].mean()
            ),

        "equal_mean":
            float(
                case_df[
                    "Equal_Yield_retention_pct"
                ].mean()
            ),

        "priority_mean":
            float(
                case_df[
                    "Priority_Yield_retention_pct"
                ].mean()
            ),

        "ppo_mean":
            float(
                case_df[
                    "PPO_Yield_retention_pct"
                ].mean()
            ),
    },

    "overall_mean_gaps_pp": {

        "PF_minus_Equal":
            float(
                case_df[
                    "PF_minus_Equal_pp"
                ].mean()
            ),

        "PF_minus_Priority":
            float(
                case_df[
                    "PF_minus_Priority_pp"
                ].mean()
            ),

        "PF_minus_PPO":
            float(
                case_df[
                    "PF_minus_PPO_pp"
                ].mean()
            ),
    },

    "diagnostics": {

        "PF_below_Equal_cases":
            int(
                pf_below_equal
            ),

        "PF_below_Priority_cases":
            int(
                pf_below_priority
            ),

        "PF_below_PPO_cases":
            int(
                pf_below_ppo
            ),

        "max_equal_validation_error_pp":
            float(
                max_equal_validation_error
            ),

        "max_pf_allocation_error_mm":
            float(
                max_pf_allocation_error
            ),

        "ppo_rows_per_case":
            ppo_row_counts,
    },

    "integrity": {

        "ppo_training_performed":
            False,

        "ppo_inference_performed":
            False,

        "controller_modified":
            False,

        "original_hypothesis_tests_modified":
            False,

        "new_hypothesis_tests":
            False,

        "post_hoc":
            True,
    },

    "runtime_minutes":
        float(
            total_runtime
            /
            60.0
        ),
}


with open(
    SUMMARY_JSON_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary_json,
        file,
        indent=2,
    )


# ============================================================
# 30. FILE REPORT
# ============================================================

print()

print(
    f"Total 56D runtime         : "
    f"{total_runtime / 60.0:.2f} min"
)


print()

print(
    "Saved:"
)


for path in [

    CHECKPOINT_FILE,

    FINAL_CASE_FILE,

    CLIMATE_SUMMARY_FILE,

    OVERALL_SUMMARY_FILE,

    WEIGHTS_FILE,

    SUMMARY_JSON_FILE,
]:

    print(
        f"  {path}"
    )


print()

print("=" * 78)

print(
    "PERFECT-FORESIGHT FINAL PANEL COMPLETE"
)

print("=" * 78)