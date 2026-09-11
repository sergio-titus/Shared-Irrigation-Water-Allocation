# ============================================================
# PPO ACTION-PROJECTION DIAGNOSTIC
# PPO ACTION-PROJECTION / FEASIBILITY-RESCALING DIAGNOSTIC
#
# PURPOSE
# -------
# Diagnose whether the frozen PPO policies are materially altered
# by the action-execution constraint layer.
#
# NO PPO TRAINING
# NO CONTROLLER MODIFICATION
# NO MODEL/SEED SELECTION
# NO NEW HYPOTHESIS TESTS
#
# The already frozen 10 PPO policies are replayed on the already
# opened 2019-2025 final-test panel.
#
# IMPORTANT
# ---------
# PPO action a_i is a fraction of current field request:
#
#     desired_i = a_i * request_i
#
# If sum(desired_i) exceeds the feasible water available today:
#
#     available_today =
#         min(daily capacity, remaining seasonal budget)
#
# the environment proportionally rescales:
#
#     executed_i =
#         desired_i * available_today / sum(desired_i)
#
# Therefore the diagnostic compares DESIRED IRRIGATION (mm)
# against EXECUTED IRRIGATION (mm), not raw action fractions
# against irrigation depths.
# ============================================================

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

# Import the already validated final-test implementation.
# This script must be in the same directory as the original
# 53J script.
import importlib.util


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

FINAL_MODELS_DIR = (
    ROOT
    / "final_training"
    / "models"
)

PROTOCOL_DIR = (
    ROOT
    / "final_evaluation_protocol"
)

CASE_FILE = (
    PROTOCOL_DIR
    / "final_evaluation_protocol_final_evaluation_cases.csv"
)

SEED_FILE = (
    PROTOCOL_DIR
    / "final_evaluation_protocol_final_ppo_seed_manifest.csv"
)

ORIGINAL_PANEL_FILE = (
    ROOT
    / "final_evaluation"
    / "controller_evaluation_final_controller_panel.csv"
)

OUTPUT_DIR = (
    ROOT
    / "projection_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEP_FILE = (
    OUTPUT_DIR
    / "ppo_projection_step_level.csv"
)

EPISODE_FILE = (
    OUTPUT_DIR
    / "ppo_projection_diagnostic_PPO_projection_episode_summary.csv"
)

CLIMATE_SCARCITY_FILE = (
    OUTPUT_DIR
    / "ppo_projection_diagnostic_PPO_projection_climate_scarcity_summary.csv"
)

SCARCITY_FILE = (
    OUTPUT_DIR
    / "ppo_projection_diagnostic_PPO_projection_scarcity_summary.csv"
)

CLIMATE_FILE = (
    OUTPUT_DIR
    / "ppo_projection_diagnostic_PPO_projection_climate_summary.csv"
)

SEED_FILE_OUT = (
    OUTPUT_DIR
    / "ppo_projection_diagnostic_PPO_projection_seed_summary.csv"
)

INTEGRITY_FILE = (
    OUTPUT_DIR
    / "ppo_projection_diagnostic_PPO_projection_integrity.json"
)


# ============================================================
# 2. LOCATE ORIGINAL 53J SCRIPT
# ============================================================

candidate_scripts = [
    Path("53J_common_multiclimate_final_controller_test.py"),
    Path("53J_final_controller_test.py"),
    Path("53J_final_evaluation.py"),
    Path("53J_common_final_test.py"),
]

script_path = None

for candidate in candidate_scripts:

    if candidate.exists():

        script_path = candidate
        break


if script_path is None:

    possible = sorted(
        Path(".").glob("53J*.py")
    )

    if len(possible) == 1:

        script_path = possible[0]

    elif len(possible) > 1:

        print()
        print("Found multiple 53J scripts:")

        for p in possible:
            print(f"  {p}")

        raise RuntimeError(
            "Multiple 53J scripts found. "
            "Keep the final 53J script in this folder "
            "and rename it to "
            "'53J_final_evaluation.py', then rerun."
        )

    else:

        raise FileNotFoundError(
            "Could not locate the original 53J Python script. "
            "Place this 55A script in the same folder as 53J."
        )


print()
print("=" * 78)
print("PPO ACTION-PROJECTION DIAGNOSTIC")
print("PPO ACTION-PROJECTION DIAGNOSTIC")
print("=" * 78)

print()
print(
    f"Using final-test implementation: "
    f"{script_path}"
)


# ============================================================
# 3. IMPORT 53J WITHOUT RUNNING main()
# ============================================================

spec = importlib.util.spec_from_file_location(
    "final53j",
    script_path,
)

final53j = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    final53j
)


# ============================================================
# 4. FROZEN CONSTANTS
# ============================================================

FIELD_NAMES = list(
    final53j.FIELD_NAMES
)

DAILY_SYSTEM_CAPACITY = float(
    final53j.DAILY_SYSTEM_CAPACITY
)

PPO_SEEDS = [
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

CLIMATES = [
    "Tunis",
    "Niamey",
    "Cotonou",
]

FINAL_TEST_YEARS = list(
    range(2019, 2026)
)

SCARCITY_LEVELS = [
    1.00,
    0.60,
    0.40,
]

EPS = 1e-12

PROJECTION_TOL = 1e-9


# ============================================================
# 5. REQUIRED FILE CHECKS
# ============================================================

required_files = [
    CASE_FILE,
    SEED_FILE,
    ORIGINAL_PANEL_FILE,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required frozen file: {path}"
        )


for seed in PPO_SEEDS:

    model_file = (
        FINAL_MODELS_DIR
        / f"multiclimate_ppo_seed_{seed}.zip"
    )

    if not model_file.exists():

        raise FileNotFoundError(
            f"Missing frozen PPO model: {model_file}"
        )


# ============================================================
# 6. VERIFY FROZEN PANEL
# ============================================================

cases = pd.read_csv(
    CASE_FILE
)

seed_manifest = pd.read_csv(
    SEED_FILE
)

original_panel = pd.read_csv(
    ORIGINAL_PANEL_FILE
)


if len(cases) != 63:

    raise RuntimeError(
        f"Expected 63 frozen final cases; "
        f"found {len(cases)}."
    )


expected_pairs = {
    (
        climate,
        year,
        scarcity,
    )
    for climate in CLIMATES
    for year in FINAL_TEST_YEARS
    for scarcity in SCARCITY_LEVELS
}


actual_pairs = {
    (
        str(row.Climate),
        int(row.Year),
        float(row.Scarcity_fraction),
    )
    for row
    in cases.itertuples()
}


if actual_pairs != expected_pairs:

    raise RuntimeError(
        "Frozen 53I case panel does not match "
        "the expected 3 x 7 x 3 design."
    )


manifest_seeds = set(
    seed_manifest[
        "Seed"
    ].astype(int)
)


if manifest_seeds != set(PPO_SEEDS):

    raise RuntimeError(
        "PPO seeds do not match the frozen manifest."
    )


if not (
    seed_manifest[
        "Retained"
    ].astype(bool)
).all():

    raise RuntimeError(
        "Not all frozen PPO seeds are retained."
    )


print()
print("Frozen panel verified:")
print(f"  climates       : {CLIMATES}")
print(f"  years          : {FINAL_TEST_YEARS}")
print("  scarcity       : 100%, 60%, 40%")
print(f"  PPO seeds      : {len(PPO_SEEDS)}")
print("  PPO training   : NO")
print("  model selection: NO")


# ============================================================
# 7. LOAD ALL FROZEN PPO MODELS
# ============================================================

print()
print("=" * 78)
print("LOADING 10 FROZEN PPO MODELS")
print("=" * 78)

models = {}

for seed in PPO_SEEDS:

    model_path = (
        FINAL_MODELS_DIR
        / f"multiclimate_ppo_seed_{seed}.zip"
    )

    models[
        seed
    ] = PPO.load(
        model_path
    )

    print(
        f"Loaded seed {seed}"
    )


# ============================================================
# 8. DIAGNOSTIC EPISODE
# ============================================================

def run_projection_diagnostic_episode(
    climate,
    year,
    scarcity_fraction,
    ppo_seed,
    model,
):

    env = final53j.FinalTestEnv(
        climate=climate,
        year=year,
        scarcity_fraction=scarcity_fraction,
        seed=0,
    )

    obs, _ = env.reset()

    terminated = False
    truncated = False

    step_rows = []

    while not (
        terminated
        or truncated
    ):

        # ----------------------------------------------------
        # Current requests BEFORE PPO decision
        # ----------------------------------------------------

        requests = (
            final53j.current_requests(
                env
            )
        )

        request_vector = np.asarray(
            [
                float(
                    requests[field]
                )
                for field
                in FIELD_NAMES
            ],
            dtype=np.float64,
        )

        total_request = float(
            request_vector.sum()
        )

        remaining_budget_before = float(
            env.remaining_budget
        )

        available_today = float(
            min(
                DAILY_SYSTEM_CAPACITY,
                remaining_budget_before,
            )
        )

        competition_day = bool(
            total_request
            >
            DAILY_SYSTEM_CAPACITY
            + PROJECTION_TOL
        )

        request_day = bool(
            total_request
            > PROJECTION_TOL
        )

        # ----------------------------------------------------
        # Frozen deterministic PPO action
        # ----------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        raw_action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        if raw_action.shape != (4,):

            raise RuntimeError(
                f"Unexpected PPO action shape: "
                f"{raw_action.shape}"
            )

        # Match env.step() exactly.
        clipped_action = np.clip(
            raw_action,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # PPO desired irrigation BEFORE feasibility rescaling
        # ----------------------------------------------------

        desired = (
            clipped_action
            * request_vector
        )

        desired_total = float(
            desired.sum()
        )

        # ----------------------------------------------------
        # Reproduce exact feasibility rule from env.step()
        # ----------------------------------------------------

        if (
            desired_total
            > available_today
            and desired_total
            > EPS
        ):

            scale_factor = float(
                available_today
                / desired_total
            )

            predicted_executed = (
                desired
                * scale_factor
            )

            projected = True

        else:

            scale_factor = 1.0

            predicted_executed = (
                desired.copy()
            )

            projected = False

        # ----------------------------------------------------
        # Projection magnitudes
        # ----------------------------------------------------

        delta = (
            desired
            - predicted_executed
        )

        abs_l1_mm = float(
            np.abs(
                delta
            ).sum()
        )

        relative_l1 = float(
            abs_l1_mm
            /
            (
                np.abs(
                    desired
                ).sum()
                + EPS
            )
        )

        removed_fraction = float(
            (
                desired_total
                -
                predicted_executed.sum()
            )
            /
            (
                desired_total
                + EPS
            )
        )

        # Cause of projection.
        daily_capacity_limiting = bool(
            DAILY_SYSTEM_CAPACITY
            <= remaining_budget_before
            + PROJECTION_TOL
            and
            desired_total
            >
            DAILY_SYSTEM_CAPACITY
            + PROJECTION_TOL
        )

        seasonal_budget_limiting = bool(
            remaining_budget_before
            <
            DAILY_SYSTEM_CAPACITY
            - PROJECTION_TOL
            and
            desired_total
            >
            remaining_budget_before
            + PROJECTION_TOL
        )

        if projected:

            if seasonal_budget_limiting:
                projection_reason = (
                    "seasonal_budget"
                )

            elif daily_capacity_limiting:
                projection_reason = (
                    "daily_capacity"
                )

            else:
                projection_reason = (
                    "combined_or_boundary"
                )

        else:

            projection_reason = "none"

        # ----------------------------------------------------
        # Execute ORIGINAL environment
        # ----------------------------------------------------

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            raw_action
        )

        # ----------------------------------------------------
        # Verify our diagnostic reproduces env.step allocation
        # ----------------------------------------------------

        executed = np.asarray(
            [
                float(
                    info[
                        "allocations"
                    ][
                        field
                    ]
                )
                for field
                in FIELD_NAMES
            ],
            dtype=np.float64,
        )

        reproduction_error = float(
            np.max(
                np.abs(
                    predicted_executed
                    - executed
                )
            )
        )

        if reproduction_error > 1e-8:

            raise RuntimeError(
                "Diagnostic failed to reproduce "
                "the environment allocation. "
                f"Error={reproduction_error:.12g}"
            )

        # ----------------------------------------------------
        # Save step
        # ----------------------------------------------------

        row = {
            "Climate":
                str(climate),

            "Year":
                int(year),

            "Scarcity_fraction":
                float(
                    scarcity_fraction
                ),

            "Scarcity_pct":
                float(
                    scarcity_fraction
                    * 100.0
                ),

            "PPO_seed":
                int(
                    ppo_seed
                ),

            "Step":
                int(
                    info[
                        "step"
                    ]
                ),

            "Request_day":
                request_day,

            "Competition_day":
                competition_day,

            "Remaining_budget_before_mm":
                remaining_budget_before,

            "Available_today_mm":
                available_today,

            "Total_request_mm":
                total_request,

            "Desired_total_mm":
                desired_total,

            "Executed_total_mm":
                float(
                    executed.sum()
                ),

            "Projected":
                bool(
                    projected
                ),

            "Projection_reason":
                projection_reason,

            "Projection_scale_factor":
                scale_factor,

            "Projection_absolute_L1_mm":
                abs_l1_mm,

            "Projection_relative_L1":
                relative_l1,

            "Projection_relative_L1_pct":
                relative_l1
                * 100.0,

            "Desired_water_removed_pct":
                removed_fraction
                * 100.0,

            "Allocation_reproduction_error_mm":
                reproduction_error,
        }

        for i, field in enumerate(
            FIELD_NAMES
        ):

            row[
                f"{field}_request_mm"
            ] = float(
                request_vector[i]
            )

            row[
                f"{field}_raw_action"
            ] = float(
                raw_action[i]
            )

            row[
                f"{field}_clipped_action"
            ] = float(
                clipped_action[i]
            )

            row[
                f"{field}_desired_mm"
            ] = float(
                desired[i]
            )

            row[
                f"{field}_executed_mm"
            ] = float(
                executed[i]
            )

            row[
                f"{field}_projection_delta_mm"
            ] = float(
                desired[i]
                - executed[i]
            )

        step_rows.append(
            row
        )

        obs = next_obs


    if truncated:

        raise RuntimeError(
            f"Unexpected truncation: "
            f"{climate} {year}, "
            f"{scarcity_fraction}, "
            f"seed={ppo_seed}"
        )


    step_df = pd.DataFrame(
        step_rows
    )


    # --------------------------------------------------------
    # Episode-level summary
    # --------------------------------------------------------

    projected_steps = (
        step_df[
            "Projected"
        ]
        .astype(bool)
    )

    request_steps = (
        step_df[
            "Request_day"
        ]
        .astype(bool)
    )

    competition_steps = (
        step_df[
            "Competition_day"
        ]
        .astype(bool)
    )


    n_steps = len(
        step_df
    )

    n_request_steps = int(
        request_steps.sum()
    )

    n_competition_steps = int(
        competition_steps.sum()
    )

    n_projected = int(
        projected_steps.sum()
    )


    projected_request_steps = int(
        (
            projected_steps
            &
            request_steps
        ).sum()
    )

    projected_competition_steps = int(
        (
            projected_steps
            &
            competition_steps
        ).sum()
    )


    projected_values = step_df.loc[
        projected_steps,
        "Projection_relative_L1_pct",
    ]


    projected_abs_values = step_df.loc[
        projected_steps,
        "Projection_absolute_L1_mm",
    ]


    episode_summary = {

        "Climate":
            str(climate),

        "Year":
            int(year),

        "Scarcity_fraction":
            float(
                scarcity_fraction
            ),

        "Scarcity_pct":
            float(
                scarcity_fraction
                * 100.0
            ),

        "PPO_seed":
            int(
                ppo_seed
            ),

        "Steps":
            int(
                n_steps
            ),

        "Request_steps":
            n_request_steps,

        "Competition_steps":
            n_competition_steps,

        "Projected_steps":
            n_projected,

        "Projection_frequency_all_steps_pct":
            (
                100.0
                * n_projected
                / n_steps
                if n_steps > 0
                else 0.0
            ),

        "Projection_frequency_request_steps_pct":
            (
                100.0
                * projected_request_steps
                / n_request_steps
                if n_request_steps > 0
                else 0.0
            ),

        "Projection_frequency_competition_steps_pct":
            (
                100.0
                * projected_competition_steps
                / n_competition_steps
                if n_competition_steps > 0
                else 0.0
            ),

        "Mean_projection_relative_pct_when_projected":
            (
                float(
                    projected_values.mean()
                )
                if len(
                    projected_values
                ) > 0
                else 0.0
            ),

        "Median_projection_relative_pct_when_projected":
            (
                float(
                    projected_values.median()
                )
                if len(
                    projected_values
                ) > 0
                else 0.0
            ),

        "Max_projection_relative_pct":
            float(
                step_df[
                    "Projection_relative_L1_pct"
                ].max()
            ),

        "Mean_projection_absolute_mm_when_projected":
            (
                float(
                    projected_abs_values.mean()
                )
                if len(
                    projected_abs_values
                ) > 0
                else 0.0
            ),

        "Total_desired_mm":
            float(
                step_df[
                    "Desired_total_mm"
                ].sum()
            ),

        "Total_executed_mm":
            float(
                step_df[
                    "Executed_total_mm"
                ].sum()
            ),

        "Total_removed_by_projection_mm":
            float(
                (
                    step_df[
                        "Desired_total_mm"
                    ]
                    -
                    step_df[
                        "Executed_total_mm"
                    ]
                ).sum()
            ),

        "Daily_capacity_projection_steps":
            int(
                (
                    step_df[
                        "Projection_reason"
                    ]
                    ==
                    "daily_capacity"
                ).sum()
            ),

        "Seasonal_budget_projection_steps":
            int(
                (
                    step_df[
                        "Projection_reason"
                    ]
                    ==
                    "seasonal_budget"
                ).sum()
            ),

        "Boundary_projection_steps":
            int(
                (
                    step_df[
                        "Projection_reason"
                    ]
                    ==
                    "combined_or_boundary"
                ).sum()
            ),

        "Max_allocation_reproduction_error_mm":
            float(
                step_df[
                    "Allocation_reproduction_error_mm"
                ].max()
            ),
    }


    env.close()

    return (
        step_df,
        episode_summary,
    )


# ============================================================
# 9. RUN FROZEN PPO REPLAY
# ============================================================

all_steps = []
episode_rows = []

start_time = time.perf_counter()

total_episodes = (
    len(CLIMATES)
    * len(FINAL_TEST_YEARS)
    * len(SCARCITY_LEVELS)
    * len(PPO_SEEDS)
)

episode_counter = 0


print()
print("=" * 78)
print("REPLAYING FROZEN PPO FINAL-TEST EPISODES")
print("=" * 78)

print(
    f"Expected PPO episodes: "
    f"{total_episodes}"
)


for scarcity in SCARCITY_LEVELS:

    print()
    print(
        f"Scarcity: "
        f"{int(scarcity * 100)}%"
    )

    for climate in CLIMATES:

        print(
            f"  Climate: {climate}"
        )

        for year in FINAL_TEST_YEARS:

            for seed in PPO_SEEDS:

                episode_counter += 1

                step_df, summary = (
                    run_projection_diagnostic_episode(
                        climate=climate,
                        year=year,
                        scarcity_fraction=scarcity,
                        ppo_seed=seed,
                        model=models[seed],
                    )
                )

                all_steps.append(
                    step_df
                )

                episode_rows.append(
                    summary
                )

            print(
                f"    {year}: "
                f"10/10 seeds complete"
            )


# ============================================================
# 10. COMBINE RESULTS
# ============================================================

step_df = pd.concat(
    all_steps,
    ignore_index=True,
)

episode_df = pd.DataFrame(
    episode_rows
)


if len(episode_df) != 630:

    raise RuntimeError(
        f"Expected 630 PPO episodes; "
        f"found {len(episode_df)}."
    )


if episode_df.duplicated(
    subset=[
        "Climate",
        "Year",
        "Scarcity_fraction",
        "PPO_seed",
    ]
).any():

    raise RuntimeError(
        "Duplicate PPO diagnostic episode."
    )


max_reproduction_error = float(
    step_df[
        "Allocation_reproduction_error_mm"
    ].max()
)


if max_reproduction_error > 1e-8:

    raise RuntimeError(
        "Projection diagnostic did not reproduce "
        "the original execution rule."
    )


# ============================================================
# 11. SAVE RAW DIAGNOSTIC DATA
# ============================================================

step_df.to_csv(
    STEP_FILE,
    index=False,
)

episode_df.to_csv(
    EPISODE_FILE,
    index=False,
)


# ============================================================
# 12. CLIMATE × SCARCITY SUMMARY
#
# IMPORTANT:
# Descriptive diagnostic only.
# No inferential tests.
# ============================================================

def summarize_group(
    df,
    grouping,
):

    summary = (
        df
        .groupby(
            grouping,
            as_index=False,
        )
        .agg(
            N_episodes=(
                "PPO_seed",
                "size",
            ),

            Projection_frequency_all_steps_mean_pct=(
                "Projection_frequency_all_steps_pct",
                "mean",
            ),

            Projection_frequency_request_steps_mean_pct=(
                "Projection_frequency_request_steps_pct",
                "mean",
            ),

            Projection_frequency_competition_steps_mean_pct=(
                "Projection_frequency_competition_steps_pct",
                "mean",
            ),

            Projection_relative_when_projected_mean_pct=(
                "Mean_projection_relative_pct_when_projected",
                "mean",
            ),

            Projection_relative_when_projected_median_pct=(
                "Median_projection_relative_pct_when_projected",
                "median",
            ),

            Projection_relative_max_pct=(
                "Max_projection_relative_pct",
                "max",
            ),

            Projection_absolute_when_projected_mean_mm=(
                "Mean_projection_absolute_mm_when_projected",
                "mean",
            ),

            Total_desired_mean_mm=(
                "Total_desired_mm",
                "mean",
            ),

            Total_executed_mean_mm=(
                "Total_executed_mm",
                "mean",
            ),

            Total_removed_by_projection_mean_mm=(
                "Total_removed_by_projection_mm",
                "mean",
            ),

            Daily_capacity_projection_steps_mean=(
                "Daily_capacity_projection_steps",
                "mean",
            ),

            Seasonal_budget_projection_steps_mean=(
                "Seasonal_budget_projection_steps",
                "mean",
            ),
        )
    )

    return summary


climate_scarcity_summary = (
    summarize_group(
        episode_df,
        [
            "Climate",
            "Scarcity_pct",
        ],
    )
)


scarcity_summary = (
    summarize_group(
        episode_df,
        [
            "Scarcity_pct",
        ],
    )
)


climate_summary = (
    summarize_group(
        episode_df,
        [
            "Climate",
        ],
    )
)


seed_summary = (
    summarize_group(
        episode_df,
        [
            "PPO_seed",
        ],
    )
)


climate_scarcity_summary.to_csv(
    CLIMATE_SCARCITY_FILE,
    index=False,
)

scarcity_summary.to_csv(
    SCARCITY_FILE,
    index=False,
)

climate_summary.to_csv(
    CLIMATE_FILE,
    index=False,
)

seed_summary.to_csv(
    SEED_FILE_OUT,
    index=False,
)


# ============================================================
# 13. EXTRA STEP-LEVEL DESCRIPTIVES
# ============================================================

projected_step_df = step_df[
    step_df[
        "Projected"
    ].astype(bool)
].copy()


overall_projection_frequency = (
    100.0
    * step_df[
        "Projected"
    ].astype(bool).mean()
)


request_only = step_df[
    step_df[
        "Request_day"
    ].astype(bool)
]


competition_only = step_df[
    step_df[
        "Competition_day"
    ].astype(bool)
]


projection_on_request_days = (
    100.0
    * request_only[
        "Projected"
    ].astype(bool).mean()
    if len(request_only) > 0
    else 0.0
)


projection_on_competition_days = (
    100.0
    * competition_only[
        "Projected"
    ].astype(bool).mean()
    if len(competition_only) > 0
    else 0.0
)


if len(projected_step_df) > 0:

    projection_relative_median = float(
        projected_step_df[
            "Projection_relative_L1_pct"
        ].median()
    )

    projection_relative_q1 = float(
        projected_step_df[
            "Projection_relative_L1_pct"
        ].quantile(
            0.25
        )
    )

    projection_relative_q3 = float(
        projected_step_df[
            "Projection_relative_L1_pct"
        ].quantile(
            0.75
        )
    )

else:

    projection_relative_median = 0.0
    projection_relative_q1 = 0.0
    projection_relative_q3 = 0.0


# ============================================================
# 14. INTEGRITY RECORD
# ============================================================

runtime_seconds = (
    time.perf_counter()
    - start_time
)


integrity = {

    "milestone":
        "55A",

    "status":
        "COMPLETE",

    "purpose":
        (
            "Post-hoc diagnostic of how often and how strongly "
            "the frozen PPO desired irrigation allocations are "
            "altered by the already-existing feasibility rescaling."
        ),

    "final_test_status":
        "already opened in milestone 53J",

    "climates":
        CLIMATES,

    "years":
        FINAL_TEST_YEARS,

    "scarcity_levels":
        SCARCITY_LEVELS,

    "ppo_seeds":
        PPO_SEEDS,

    "ppo_episodes":
        int(
            len(
                episode_df
            )
        ),

    "step_rows":
        int(
            len(
                step_df
            )
        ),

    "projection_definition":
        (
            "desired_i = clipped PPO action_i * current request_i; "
            "if desired total exceeds min(daily capacity, remaining "
            "seasonal budget), all desired allocations are "
            "proportionally rescaled."
        ),

    "relative_projection_metric":
        (
            "L1(desired - executed) / "
            "(L1(desired) + epsilon)"
        ),

    "overall_projection_frequency_all_steps_pct":
        float(
            overall_projection_frequency
        ),

    "projection_frequency_request_steps_pct":
        float(
            projection_on_request_days
        ),

    "projection_frequency_competition_steps_pct":
        float(
            projection_on_competition_days
        ),

    "projected_step_relative_change_median_pct":
        projection_relative_median,

    "projected_step_relative_change_q1_pct":
        projection_relative_q1,

    "projected_step_relative_change_q3_pct":
        projection_relative_q3,

    "maximum_allocation_reproduction_error_mm":
        max_reproduction_error,

    "ppo_training_performed":
        False,

    "ppo_parameters_modified":
        False,

    "reward_modified":
        False,

    "controller_modified":
        False,

    "model_selection":
        False,

    "seed_selection":
        False,

    "new_hypothesis_tests":
        False,

    "diagnostic_only":
        True,

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
# 15. TERMINAL REPORT
# ============================================================

print()
print("=" * 78)
print("OVERALL PPO PROJECTION DIAGNOSTIC")
print("=" * 78)

print(
    f"Step-level observations       : "
    f"{len(step_df):,}"
)

print(
    f"PPO episodes                  : "
    f"{len(episode_df):,}"
)

print(
    f"Projection frequency, all     : "
    f"{overall_projection_frequency:.3f}%"
)

print(
    f"Projection on request days    : "
    f"{projection_on_request_days:.3f}%"
)

print(
    f"Projection on competition days: "
    f"{projection_on_competition_days:.3f}%"
)

print(
    "Relative change when projected:"
)

print(
    f"  median                      : "
    f"{projection_relative_median:.3f}%"
)

print(
    f"  IQR                         : "
    f"[{projection_relative_q1:.3f}, "
    f"{projection_relative_q3:.3f}]%"
)

print(
    f"Max allocation reproduction error: "
    f"{max_reproduction_error:.3e} mm"
)


print()
print("=" * 78)
print("CLIMATE × SCARCITY SUMMARY")
print("=" * 78)

display_columns = [
    "Climate",
    "Scarcity_pct",
    "Projection_frequency_request_steps_mean_pct",
    "Projection_frequency_competition_steps_mean_pct",
    "Projection_relative_when_projected_median_pct",
    "Total_removed_by_projection_mean_mm",
]

print(
    climate_scarcity_summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


print()
print("=" * 78)
print("PROJECTION CAUSE")
print("=" * 78)

reason_counts = (
    step_df[
        "Projection_reason"
    ]
    .value_counts()
)

print(
    reason_counts.to_string()
)


print()
print("=" * 78)
print("INTEGRITY")
print("=" * 78)

print(
    "PPO training performed       : NO"
)

print(
    "PPO parameters modified      : NO"
)

print(
    "Reward modified              : NO"
)

print(
    "Controller modified          : NO"
)

print(
    "Model selection              : NO"
)

print(
    "Seed selection               : NO"
)

print(
    "New hypothesis tests         : NO"
)

print(
    "Frozen PPO policies replayed : YES"
)

print(
    "Original feasibility rule    : UNCHANGED"
)

print(
    "Diagnostic reproduction      : PASS"
)

print(
    f"Runtime                      : "
    f"{runtime_seconds / 60.0:.2f} min"
)


print()
print("Saved:")

for path in [
    STEP_FILE,
    EPISODE_FILE,
    CLIMATE_SCARCITY_FILE,
    SCARCITY_FILE,
    CLIMATE_FILE,
    SEED_FILE_OUT,
    INTEGRITY_FILE,
]:

    print(
        f"  {path}"
    )


print()
print("=" * 78)
print("PPO ACTION-PROJECTION DIAGNOSTIC COMPLETE")
print("=" * 78)

print()
print(
    "NEXT: interpret whether PPO's final-test deficit "
    "can plausibly be attributed to the feasibility "
    "projection/rescaling layer."
)