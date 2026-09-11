# ============================================================
# COMMON CONTROLLER EVALUATION
# COMMON MULTI-CLIMATE FINAL CONTROLLER TEST
#
# THIS MILESTONE OPENS THE CONTROLLER FINAL TEST.
#
# Evaluates, without any retraining or tuning:
#   1. Equal
#   2. Priority (depletion-ratio ranking)
#   3. Frozen 10-seed Optimized PPO ensemble
#
# Common frozen panel:
#   - Tunis, Niamey, Cotonou
#   - 2019-2025
#   - 100%, 60%, 40% seasonal budgets
#
# IMPORTANT
# ---------
# - All controller/configuration choices are already frozen.
# - PPO models are loaded only; no training occurs.
# - All 10 PPO seeds are retained.
# - PPO seed results are averaged within each climate-year
#   before controller-level inference in the NEXT milestone.
# - No statistical significance tests are run here.
# - After this script runs, controller final-test performance
#   has been observed. Do not tune any controller afterward.
# ============================================================

from pathlib import Path
import hashlib
import json
import time

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd

from aquacrop import (
    AquaCropModel,
    Soil,
    Crop,
    InitialWaterContent,
    IrrigationManagement,
)

from stable_baselines3 import PPO

import multiclimate_optimization_core as core


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

PROTOCOL_DIR = (
    ROOT
    / "final_evaluation_protocol"
)

MANIFEST_FILE = (
    PROTOCOL_DIR
    / "final_evaluation_protocol_final_evaluation_manifest_frozen.json"
)

CASE_FILE = (
    PROTOCOL_DIR
    / "final_evaluation_protocol_final_evaluation_cases.csv"
)

SEED_FILE = (
    PROTOCOL_DIR
    / "final_evaluation_protocol_final_ppo_seed_manifest.csv"
)

REFERENCE_FILE = (
    ROOT
    / "final_test_reference"
    / "final_test_reference_by_climate_year.csv"
)

REFERENCE_INTEGRITY_FILE = (
    ROOT
    / "final_test_reference"
    / "final_test_reference_final_test_reference_integrity.json"
)

FINAL_TRAINING_INTEGRITY_FILE = (
    ROOT
    / "final_training"
    / "ppo_final_training_final_training_integrity.json"
)

FINAL_MODELS_DIR = (
    ROOT
    / "final_training"
    / "models"
)

OUTPUT_DIR = (
    ROOT
    / "final_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_final_raw_results.csv"
)

PPO_ENSEMBLE_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_final_ppo_ensemble_by_case.csv"
)

CONTROLLER_PANEL_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_final_controller_panel.csv"
)

OVERALL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_final_overall_summary.csv"
)

CLIMATE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_final_climate_summary.csv"
)

BINDING_SUMMARY_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_binding_water_summary.csv"
)

PPO_SEED_ROBUSTNESS_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_ppo_seed_robustness_40pct.csv"
)

INTEGRITY_FILE = (
    OUTPUT_DIR
    / "controller_evaluation_final_evaluation_integrity.json"
)


# ============================================================
# 2. HELPERS
# ============================================================

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


# ============================================================
# 3. LOAD FROZEN FINAL-EVALUATION MANIFEST
# ============================================================

manifest = load_json(
    MANIFEST_FILE
)

cases = pd.read_csv(
    CASE_FILE
)

seed_manifest = pd.read_csv(
    SEED_FILE
)

reference_df = pd.read_csv(
    REFERENCE_FILE
)


CLIMATES = list(
    manifest[
        "climates"
    ]
)

FINAL_TEST_YEARS = [
    int(y)
    for y in manifest[
        "final_test_years"
    ]
]

SCARCITY_LEVELS = [
    float(x)
    for x in manifest[
        "scarcity_levels"
    ]
]

CONTROLLERS = list(
    manifest[
        "controllers"
    ]
)

PPO_SEEDS = [
    int(x)
    for x in manifest[
        "final_ppo_seeds"
    ]
]


# ============================================================
# 4. STRICT FROZEN-PROTOCOL CHECKS
# ============================================================

assert CLIMATES == [
    "Tunis",
    "Niamey",
    "Cotonou",
]

assert FINAL_TEST_YEARS == list(
    range(2019, 2026)
)

assert SCARCITY_LEVELS == [
    1.00,
    0.60,
    0.40,
]

assert CONTROLLERS == [
    "Equal",
    "Priority",
    "Optimized PPO",
]

assert PPO_SEEDS == [
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

if len(cases) != 63:

    raise RuntimeError(
        f"Expected 63 frozen final cases, got {len(cases)}."
    )

if cases.duplicated(
    subset=[
        "Climate",
        "Year",
        "Scarcity_fraction",
    ]
).any():

    raise RuntimeError(
        "Duplicate frozen final case."
    )

if len(seed_manifest) != 10:

    raise RuntimeError(
        "Expected 10 frozen PPO seeds."
    )

if not (
    seed_manifest[
        "Retained"
    ].astype(bool)
).all():

    raise RuntimeError(
        "One or more PPO seeds are not marked retained."
    )

if (
    seed_manifest[
        "Seed_selection"
    ].astype(bool)
).any():

    raise RuntimeError(
        "Seed selection is unexpectedly enabled."
    )

pre_status = manifest[
    "controller_test_status_before_execution"
]

if pre_status[
    "ppo_inference_performed"
] is not False:

    raise RuntimeError(
        "Manifest does not certify pre-test PPO status."
    )

if pre_status[
    "equal_evaluated"
] is not False:

    raise RuntimeError(
        "Manifest does not certify pre-test Equal status."
    )

if pre_status[
    "priority_evaluated"
] is not False:

    raise RuntimeError(
        "Manifest does not certify pre-test Priority status."
    )

if pre_status[
    "controller_final_test_opened"
] is not False:

    raise RuntimeError(
        "Controller final test was already marked open."
    )


# ============================================================
# 5. VERIFY FROZEN FINAL MODELS
# ============================================================

for seed in PPO_SEEDS:

    model_file = (
        FINAL_MODELS_DIR
        / f"multiclimate_ppo_seed_{seed}.zip"
    )

    if not model_file.exists():

        raise FileNotFoundError(
            f"Missing frozen PPO model: {model_file}"
        )


training_integrity = load_json(
    FINAL_TRAINING_INTEGRITY_FILE
)

if training_integrity.get(
    "all_final_seeds_retained"
) is not True:

    raise RuntimeError(
        "53G does not certify all final seeds retained."
    )

if training_integrity.get(
    "seed_selection"
) is not False:

    raise RuntimeError(
        "53G unexpectedly reports seed selection."
    )


# ============================================================
# 6. VERIFY FINAL REFERENCE LIBRARY
# ============================================================

reference_df[
    "Year"
] = reference_df[
    "Year"
].astype(int)

if len(reference_df) != 21:

    raise RuntimeError(
        f"Expected 21 final reference rows, got "
        f"{len(reference_df)}."
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
        "Reference library does not match frozen final panel."
    )


# ============================================================
# 7. FIELD / PHYSICAL CONSTANTS
# ============================================================

FIELD_CONFIGS = list(
    core.FIELD_CONFIGS
)

FIELD_NAMES = list(
    core.FIELD_NAMES
)

DAILY_SYSTEM_CAPACITY = float(
    core.DAILY_SYSTEM_CAPACITY
)

MAX_FIELD_IRRIGATION = float(
    core.MAX_FIELD_IRRIGATION
)

DEPLETION_TRIGGER = float(
    core.DEPLETION_TRIGGER
)

CLIMATE_TO_ONEHOT = dict(
    core.CLIMATE_TO_ONEHOT
)

WEATHER_BY_CLIMATE = dict(
    core.WEATHER_BY_CLIMATE
)


# ============================================================
# 8. FINAL-TEST REFERENCE LOOKUP
# ============================================================

def get_final_reference(
    climate,
    year,
    scarcity_fraction,
):

    climate = str(climate)
    year = int(year)
    scarcity_fraction = float(
        scarcity_fraction
    )

    if climate not in CLIMATES:

        raise RuntimeError(
            f"Unknown climate: {climate}"
        )

    if year not in FINAL_TEST_YEARS:

        raise RuntimeError(
            f"Year {year} is outside frozen final test."
        )

    if scarcity_fraction not in SCARCITY_LEVELS:

        raise RuntimeError(
            f"Unexpected scarcity: {scarcity_fraction}"
        )

    rows = reference_df[
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

    if len(rows) != 1:

        raise RuntimeError(
            f"Expected one reference row for "
            f"{climate} {year}."
        )

    row = rows.iloc[0]

    reference_yields = {
        field_name:
            float(
                row[
                    f"{field_name}_reference_yield"
                ]
            )
        for field_name
        in FIELD_NAMES
    }

    reference_irrigation = {
        field_name:
            float(
                row[
                    f"{field_name}_reference_irrigation"
                ]
            )
        for field_name
        in FIELD_NAMES
    }

    total_reference_irrigation = float(
        row[
            "Total reference irrigation (mm)"
        ]
    )

    total_reference_yield = float(
        sum(
            reference_yields.values()
        )
    )

    seasonal_budget = (
        scarcity_fraction
        * total_reference_irrigation
    )

    return {
        "climate":
            climate,

        "year":
            year,

        "reference_yields":
            reference_yields,

        "reference_irrigation":
            reference_irrigation,

        "total_reference_irrigation":
            total_reference_irrigation,

        "total_reference_yield":
            total_reference_yield,

        "seasonal_budget":
            float(
                seasonal_budget
            ),
    }


# ============================================================
# 9. FINAL-TEST AQUACROP FIELD
#
# This is deliberately the same field definition as the
# development core, except final-test years are now allowed.
# ============================================================

def create_final_test_field(
    climate,
    year,
    planting_date,
):

    climate = str(climate)
    year = int(year)

    if climate not in CLIMATES:

        raise RuntimeError(
            f"Unknown climate: {climate}"
        )

    if year not in FINAL_TEST_YEARS:

        raise RuntimeError(
            f"Non-final-test year passed to final evaluator: "
            f"{year}"
        )

    weather = (
        WEATHER_BY_CLIMATE[
            climate
        ]
        .copy()
    )

    soil = Soil(
        soil_type="SandyLoam"
    )

    crop = Crop(
        "Maize",
        planting_date=planting_date,
    )

    initial_water = (
        InitialWaterContent(
            value=[
                "FC"
            ]
        )
    )

    irrigation = (
        IrrigationManagement(
            irrigation_method=5,
            depth=0,
            MaxIrr=MAX_FIELD_IRRIGATION,
            MaxIrrSeason=10000,
        )
    )

    model = AquaCropModel(
        sim_start_time=(
            f"{year}/05/01"
        ),
        sim_end_time=(
            f"{year}/10/31"
        ),
        weather_df=weather,
        soil=soil,
        crop=crop,
        initial_water_content=initial_water,
        irrigation_management=irrigation,
    )

    model._initialize()

    return model


# ============================================================
# 10. FINAL-TEST ENVIRONMENT
#
# Inherits the already validated state, request, stress,
# final-metric and step implementations from the development
# core. Only constructor/reset are replaced to permit the
# already-frozen 2019-2025 final panel and arbitrary frozen
# scarcity levels.
# ============================================================

class FinalTestEnv(
    core.MultiClimateSharedWaterEnv
):

    def __init__(
        self,
        climate,
        year,
        scarcity_fraction,
        seed=0,
    ):

        gym.Env.__init__(
            self
        )

        self.current_climate = str(
            climate
        )

        self.current_year = int(
            year
        )

        self.scarcity_fraction = float(
            scarcity_fraction
        )

        if (
            self.current_climate
            not in CLIMATES
        ):

            raise RuntimeError(
                f"Unknown climate: "
                f"{self.current_climate}"
            )

        if (
            self.current_year
            not in FINAL_TEST_YEARS
        ):

            raise RuntimeError(
                "FinalTestEnv accepts only "
                "2019-2025."
            )

        if (
            self.scarcity_fraction
            not in SCARCITY_LEVELS
        ):

            raise RuntimeError(
                "FinalTestEnv received a scarcity "
                "outside the frozen manifest."
            )

        self.base_seed = int(
            seed
        )

        self.rng = np.random.default_rng(
            self.base_seed
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(34,),
            dtype=np.float32,
        )

        self.models = {}

        self.reference_yields = {}

        self.reference_irrigation = {}

        self.total_reference_irrigation = 0.0

        self.total_reference_yield = 0.0

        self.seasonal_budget = 0.0

        self.remaining_budget = 0.0

        self.total_allocated = 0.0

        self.cumulative_request = 0.0

        self.request_days = 0

        self.competition_days = 0

        self.irrigation_days = 0

        self.step_count = 0

        self.budget_exhaustion_step = None

        self.episode_finished = False

        self.episode_number = 0


    def reset(
        self,
        seed=None,
        options=None,
    ):

        gym.Env.reset(
            self,
            seed=seed,
        )

        if seed is not None:

            self.rng = (
                np.random.default_rng(
                    seed
                )
            )

        ref = get_final_reference(
            climate=self.current_climate,
            year=self.current_year,
            scarcity_fraction=(
                self.scarcity_fraction
            ),
        )

        self.reference_yields = (
            ref[
                "reference_yields"
            ]
        )

        self.reference_irrigation = (
            ref[
                "reference_irrigation"
            ]
        )

        self.total_reference_irrigation = float(
            ref[
                "total_reference_irrigation"
            ]
        )

        self.total_reference_yield = float(
            ref[
                "total_reference_yield"
            ]
        )

        self.seasonal_budget = float(
            ref[
                "seasonal_budget"
            ]
        )

        self.remaining_budget = float(
            self.seasonal_budget
        )

        self.models = {}

        for cfg in FIELD_CONFIGS:

            self.models[
                cfg[
                    "field"
                ]
            ] = (
                create_final_test_field(
                    climate=(
                        self.current_climate
                    ),
                    year=(
                        self.current_year
                    ),
                    planting_date=(
                        cfg[
                            "planting_date"
                        ]
                    ),
                )
            )

        self.total_allocated = 0.0

        self.cumulative_request = 0.0

        self.request_days = 0

        self.competition_days = 0

        self.irrigation_days = 0

        self.step_count = 0

        self.budget_exhaustion_step = None

        self.episode_finished = False

        self.episode_number += 1

        observation = (
            self._get_observation()
        )

        info = {
            "climate":
                self.current_climate,

            "year":
                self.current_year,

            "scarcity_fraction":
                self.scarcity_fraction,

            "seasonal_budget":
                self.seasonal_budget,

            "total_reference_irrigation":
                self.total_reference_irrigation,

            "total_reference_yield":
                self.total_reference_yield,
        }

        return (
            observation,
            info,
        )


# ============================================================
# 11. CURRENT REQUEST VECTOR
# ============================================================

def current_requests(
    env,
):

    return {
        field_name:
            float(
                env._get_field_request(
                    env.models[
                        field_name
                    ]
                )
            )
        for field_name
        in FIELD_NAMES
    }


# ============================================================
# 12. EQUAL CONTROLLER
#
# Equal water-filling among active requesters, with iterative
# redistribution when a field cannot consume its full share.
# ============================================================

def equal_allocate(
    requests,
    available_today,
):

    allocation = {
        field:
            0.0
        for field
        in FIELD_NAMES
    }

    active = [
        field
        for field
        in FIELD_NAMES
        if (
            requests[
                field
            ]
            > 1e-9
        )
    ]

    remaining = float(
        available_today
    )

    while (
        active
        and remaining > 1e-9
    ):

        share = (
            remaining
            / len(
                active
            )
        )

        next_active = []

        amount_used = 0.0

        for field in active:

            unmet = (
                requests[
                    field
                ]
                - allocation[
                    field
                ]
            )

            amount = min(
                share,
                unmet,
            )

            allocation[
                field
            ] += amount

            remaining -= amount

            amount_used += amount

            if (
                requests[
                    field
                ]
                - allocation[
                    field
                ]
                > 1e-9
            ):

                next_active.append(
                    field
                )

        if amount_used <= 1e-9:

            break

        active = next_active

    return allocation


# ============================================================
# 13. PRIORITY CONTROLLER
#
# Frozen simple priority:
#
#       P_i(t) = D_i(t)
#
# where D_i is current normalized root-zone depletion
# (depletion / TAW) for requesting fields.
#
# Water is filled in descending priority order.
# Stable FIELD_NAMES order resolves exact ties.
# ============================================================

def priority_allocate(
    env,
    requests,
    available_today,
):

    priorities = {}

    for field in FIELD_NAMES:

        if (
            requests[
                field
            ]
            <= 1e-9
        ):

            priorities[
                field
            ] = -np.inf

            continue

        model = env.models[
            field
        ]

        state = (
            model._init_cond
        )

        depletion = float(
            getattr(
                state,
                "depletion",
                0.0,
            )
        )

        taw = float(
            getattr(
                state,
                "taw",
                0.0,
            )
        )

        if taw > 1e-9:

            depletion_ratio = (
                depletion
                / taw
            )

        else:

            depletion_ratio = 0.0

        priorities[
            field
        ] = float(
            depletion_ratio
        )

    ranked_fields = sorted(
        FIELD_NAMES,
        key=lambda field:
            priorities[
                field
            ],
        reverse=True,
    )

    allocation = {
        field:
            0.0
        for field
        in FIELD_NAMES
    }

    remaining = float(
        available_today
    )

    for field in ranked_fields:

        if remaining <= 1e-9:

            break

        request = float(
            requests[
                field
            ]
        )

        if request <= 1e-9:

            continue

        amount = min(
            request,
            remaining,
        )

        allocation[
            field
        ] = float(
            amount
        )

        remaining -= amount

    return allocation


# ============================================================
# 14. CONVERT DESIRED ALLOCATIONS TO THE ENVIRONMENT'S
#     REQUEST-FRACTION ACTION PARAMETERIZATION
# ============================================================

def allocation_to_action(
    requests,
    allocations,
):

    action = np.zeros(
        4,
        dtype=np.float32,
    )

    for i, field in enumerate(
        FIELD_NAMES
    ):

        request = float(
            requests[
                field
            ]
        )

        allocation = float(
            allocations[
                field
            ]
        )

        if (
            allocation
            > request
            + 1e-8
        ):

            raise RuntimeError(
                f"Allocation exceeds request for "
                f"{field}."
            )

        if request > 1e-12:

            action[
                i
            ] = float(
                np.clip(
                    allocation
                    / request,
                    0.0,
                    1.0,
                )
            )

    return action


# ============================================================
# 15. ONE FINAL-TEST EPISODE
# ============================================================

def run_episode(
    climate,
    year,
    scarcity_fraction,
    controller,
    ppo_model=None,
    ppo_seed=None,
):

    env = FinalTestEnv(
        climate=climate,
        year=year,
        scarcity_fraction=(
            scarcity_fraction
        ),
        seed=0,
    )

    obs, _ = env.reset()

    terminated = False
    truncated = False

    decision_times_ms = []

    while not (
        terminated
        or truncated
    ):

        if controller == "Optimized PPO":

            if ppo_model is None:

                raise RuntimeError(
                    "PPO controller requires a model."
                )

            t0 = time.perf_counter()

            action, _ = (
                ppo_model.predict(
                    obs,
                    deterministic=True,
                )
            )

            t1 = time.perf_counter()

        elif controller == "Equal":

            requests = (
                current_requests(
                    env
                )
            )

            available_today = min(
                DAILY_SYSTEM_CAPACITY,
                env.remaining_budget,
            )

            t0 = time.perf_counter()

            allocations = (
                equal_allocate(
                    requests=(
                        requests
                    ),
                    available_today=(
                        available_today
                    ),
                )
            )

            action = (
                allocation_to_action(
                    requests=(
                        requests
                    ),
                    allocations=(
                        allocations
                    ),
                )
            )

            t1 = time.perf_counter()

        elif controller == "Priority":

            requests = (
                current_requests(
                    env
                )
            )

            available_today = min(
                DAILY_SYSTEM_CAPACITY,
                env.remaining_budget,
            )

            t0 = time.perf_counter()

            allocations = (
                priority_allocate(
                    env=env,
                    requests=(
                        requests
                    ),
                    available_today=(
                        available_today
                    ),
                )
            )

            action = (
                allocation_to_action(
                    requests=(
                        requests
                    ),
                    allocations=(
                        allocations
                    ),
                )
            )

            t1 = time.perf_counter()

        else:

            raise RuntimeError(
                f"Unknown controller: {controller}"
            )

        decision_times_ms.append(
            (
                t1
                - t0
            )
            * 1000.0
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

    if truncated:

        raise RuntimeError(
            f"Unexpected truncation: "
            f"{controller}, "
            f"{climate} {year}, "
            f"{scarcity_fraction}"
        )

    final_results = (
        info[
            "final_results"
        ]
    )

    if final_results is None:

        raise RuntimeError(
            "Missing final results."
        )

    allocated = float(
        env.total_allocated
    )

    aquacrop_irrigation = float(
        final_results[
            "total_irrigation"
        ]
    )

    accounting_difference = (
        allocated
        - aquacrop_irrigation
    )

    if abs(
        accounting_difference
    ) > 1e-6:

        raise RuntimeError(
            f"Water accounting failed: "
            f"{controller} | {climate} {year} | "
            f"{scarcity_fraction:.2f} | "
            f"difference={accounting_difference}"
        )

    seasonal_budget = float(
        env.seasonal_budget
    )

    budget_utilization_pct = (
        allocated
        / seasonal_budget
        * 100.0
        if seasonal_budget > 1e-12
        else 0.0
    )

    budget_binding = bool(
        env.budget_exhaustion_step
        is not None
    )

    result = {
        "Controller":
            controller,

        "PPO_seed":
            (
                int(
                    ppo_seed
                )
                if ppo_seed
                is not None
                else np.nan
            ),

        "Climate":
            str(
                climate
            ),

        "Year":
            int(
                year
            ),

        "Scarcity_fraction":
            float(
                scarcity_fraction
            ),

        "Scarcity_pct":
            float(
                scarcity_fraction
                * 100.0
            ),

        "Yield_retention":
            float(
                final_results[
                    "total_yield_retention"
                ]
            ),

        "Yield_retention_pct":
            float(
                final_results[
                    "total_yield_retention"
                ]
                * 100.0
            ),

        "Mean_field_retention":
            float(
                final_results[
                    "mean_retention"
                ]
            ),

        "Worst_field_retention":
            float(
                final_results[
                    "worst_retention"
                ]
            ),

        "Worst_field_retention_pct":
            float(
                final_results[
                    "worst_retention"
                ]
                * 100.0
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
            aquacrop_irrigation,

        "Reference_irrigation_mm":
            float(
                env.total_reference_irrigation
            ),

        "Seasonal_budget_mm":
            seasonal_budget,

        "Budget_utilization_pct":
            float(
                budget_utilization_pct
            ),

        "Budget_binding":
            budget_binding,

        "Budget_exhaustion_step":
            (
                int(
                    env.budget_exhaustion_step
                )
                if env.budget_exhaustion_step
                is not None
                else np.nan
            ),

        "Request_days":
            int(
                env.request_days
            ),

        "Competition_days":
            int(
                env.competition_days
            ),

        "Irrigation_days":
            int(
                env.irrigation_days
            ),

        "Cumulative_request_mm":
            float(
                env.cumulative_request
            ),

        "Steps":
            int(
                env.step_count
            ),

        "Decision_latency_mean_ms":
            float(
                np.mean(
                    decision_times_ms
                )
            ),

        "Accounting_difference_mm":
            float(
                accounting_difference
            ),
    }

    for field in FIELD_NAMES:

        result[
            f"{field}_yield_t_ha"
        ] = float(
            final_results[
                "yields"
            ][
                field
            ]
        )

        result[
            f"{field}_irrigation_mm"
        ] = float(
            final_results[
                "irrigations"
            ][
                field
            ]
        )

        result[
            f"{field}_retention"
        ] = float(
            final_results[
                "retentions"
            ][
                field
            ]
        )

        result[
            f"{field}_retention_pct"
        ] = float(
            final_results[
                "retentions"
            ][
                field
            ]
            * 100.0
        )

    env.close()

    return result


# ============================================================
# 16. AGGREGATION
# ============================================================

def build_ppo_ensemble(
    raw_df,
):

    ppo = raw_df[
        raw_df[
            "Controller"
        ]
        == "Optimized PPO"
    ].copy()

    expected_ppo_rows = (
        63
        * len(
            PPO_SEEDS
        )
    )

    if len(ppo) != expected_ppo_rows:

        raise RuntimeError(
            f"Expected {expected_ppo_rows} PPO raw rows, "
            f"got {len(ppo)}."
        )

    grouping = [
        "Climate",
        "Year",
        "Scarcity_fraction",
        "Scarcity_pct",
    ]

    numeric_columns = [
        col
        for col in ppo.columns
        if (
            col not in grouping
            and col not in [
                "Controller",
                "PPO_seed",
                "Budget_binding",
            ]
            and pd.api.types.is_numeric_dtype(
                ppo[
                    col
                ]
            )
        )
    ]

    ensemble = (
        ppo
        .groupby(
            grouping,
            as_index=False,
        )[
            numeric_columns
        ]
        .mean()
    )

    binding = (
        ppo
        .groupby(
            grouping,
            as_index=False,
        )[
            "Budget_binding"
        ]
        .mean()
        .rename(
            columns={
                "Budget_binding":
                    "Budget_binding_fraction_across_seeds"
            }
        )
    )

    ensemble = ensemble.merge(
        binding,
        on=grouping,
        how="left",
        validate="one_to_one",
    )

    ensemble.insert(
        0,
        "Controller",
        "Optimized PPO",
    )

    if len(ensemble) != 63:

        raise RuntimeError(
            f"Expected 63 PPO ensemble rows, "
            f"got {len(ensemble)}."
        )

    return ensemble


def build_controller_panel(
    raw_df,
    ppo_ensemble,
):

    baselines = raw_df[
        raw_df[
            "Controller"
        ].isin(
            [
                "Equal",
                "Priority",
            ]
        )
    ].copy()

    if len(baselines) != 126:

        raise RuntimeError(
            f"Expected 126 baseline rows, "
            f"got {len(baselines)}."
        )

    baseline_columns_to_drop = [
        "PPO_seed",
    ]

    baselines = baselines.drop(
        columns=[
            col
            for col
            in baseline_columns_to_drop
            if col
            in baselines.columns
        ]
    )

    if (
        "Budget_binding_fraction_across_seeds"
        not in baselines.columns
    ):

        baselines[
            "Budget_binding_fraction_across_seeds"
        ] = (
            baselines[
                "Budget_binding"
            ]
            .astype(float)
        )

    panel = pd.concat(
        [
            baselines,
            ppo_ensemble,
        ],
        ignore_index=True,
        sort=False,
    )

    if len(panel) != 189:

        raise RuntimeError(
            f"Expected 189 controller-panel rows, "
            f"got {len(panel)}."
        )

    counts = (
        panel
        .groupby(
            [
                "Controller",
                "Scarcity_pct",
            ]
        )
        .size()
    )

    if not (
        counts
        == 21
    ).all():

        raise RuntimeError(
            "Controller panel is not balanced "
            "at 21 climate-years per controller/scarcity."
        )

    return panel


# ============================================================
# 17. MAIN FINAL TEST
# ============================================================

def main():

    start_all = time.perf_counter()

    print()
    print("#" * 78)
    print("COMMON CONTROLLER EVALUATION")
    print(
        "COMMON MULTI-CLIMATE FINAL CONTROLLER TEST"
    )
    print("#" * 78)

    print()
    print(
        "WARNING: controller final-test performance "
        "will now be opened."
    )

    print()
    print(
        f"Climates              : {CLIMATES}"
    )

    print(
        f"Years                 : {FINAL_TEST_YEARS}"
    )

    print(
        "Scarcity levels       : "
        f"{[int(x * 100) for x in SCARCITY_LEVELS]}%"
    )

    print(
        f"Frozen common cases   : {len(cases)}"
    )

    print(
        f"Frozen PPO seeds      : {len(PPO_SEEDS)}"
    )

    print(
        "PPO training          : NO"
    )

    print(
        "Controller tuning     : NO"
    )


    # ========================================================
    # LOAD ALL PPO MODELS BEFORE EVALUATION
    # ========================================================

    print()
    print("=" * 78)
    print("LOADING FROZEN PPO MODELS")
    print("=" * 78)

    ppo_models = {}

    for seed in PPO_SEEDS:

        model_path = (
            FINAL_MODELS_DIR
            / f"multiclimate_ppo_seed_{seed}.zip"
        )

        ppo_models[
            seed
        ] = PPO.load(
            str(
                model_path
            ),
            device="auto",
        )

        print(
            f"Loaded seed {seed}"
        )


    # ========================================================
    # EXECUTE ALL FROZEN CASES
    # ========================================================

    rows = []

    sorted_cases = (
        cases
        .sort_values(
            [
                "Scarcity_fraction",
                "Climate",
                "Year",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    total_raw_episodes = (
        63
        * (
            2
            + len(
                PPO_SEEDS
            )
        )
    )

    episode_counter = 0

    print()
    print("=" * 78)
    print("EXECUTING FINAL TEST")
    print("=" * 78)

    for _, case in sorted_cases.iterrows():

        climate = str(
            case[
                "Climate"
            ]
        )

        year = int(
            case[
                "Year"
            ]
        )

        scarcity_fraction = float(
            case[
                "Scarcity_fraction"
            ]
        )

        scarcity_pct = int(
            round(
                scarcity_fraction
                * 100.0
            )
        )

        print()
        print(
            f"{climate:<8} {year} | "
            f"{scarcity_pct}%"
        )

        # ----------------------------------------
        # EQUAL
        # ----------------------------------------

        result = run_episode(
            climate=climate,
            year=year,
            scarcity_fraction=(
                scarcity_fraction
            ),
            controller="Equal",
        )

        rows.append(
            result
        )

        episode_counter += 1

        # ----------------------------------------
        # PRIORITY
        # ----------------------------------------

        result = run_episode(
            climate=climate,
            year=year,
            scarcity_fraction=(
                scarcity_fraction
            ),
            controller="Priority",
        )

        rows.append(
            result
        )

        episode_counter += 1

        # ----------------------------------------
        # PPO — ALL 10 FROZEN SEEDS
        # ----------------------------------------

        ppo_case_retentions = []

        for seed in PPO_SEEDS:

            result = run_episode(
                climate=climate,
                year=year,
                scarcity_fraction=(
                    scarcity_fraction
                ),
                controller=(
                    "Optimized PPO"
                ),
                ppo_model=(
                    ppo_models[
                        seed
                    ]
                ),
                ppo_seed=seed,
            )

            rows.append(
                result
            )

            ppo_case_retentions.append(
                result[
                    "Yield_retention_pct"
                ]
            )

            episode_counter += 1

        print(
            "  Complete | "
            f"PPO seed mean yield retention="
            f"{np.mean(ppo_case_retentions):.2f}%"
        )


    raw_df = pd.DataFrame(
        rows
    )


    # ========================================================
    # 18. RAW INTEGRITY
    # ========================================================

    if episode_counter != total_raw_episodes:

        raise RuntimeError(
            f"Expected {total_raw_episodes} final episodes, "
            f"executed {episode_counter}."
        )

    if len(raw_df) != total_raw_episodes:

        raise RuntimeError(
            f"Expected {total_raw_episodes} raw rows, "
            f"got {len(raw_df)}."
        )

    if (
        np.abs(
            raw_df[
                "Accounting_difference_mm"
            ].to_numpy(
                dtype=float
            )
        )
        > 1e-6
    ).any():

        raise RuntimeError(
            "Water accounting integrity failed."
        )

    raw_df.to_csv(
        RAW_FILE,
        index=False,
    )


    # ========================================================
    # 19. PPO ENSEMBLE + COMMON CONTROLLER PANEL
    # ========================================================

    ppo_ensemble = (
        build_ppo_ensemble(
            raw_df
        )
    )

    ppo_ensemble.to_csv(
        PPO_ENSEMBLE_FILE,
        index=False,
    )

    panel = (
        build_controller_panel(
            raw_df=raw_df,
            ppo_ensemble=(
                ppo_ensemble
            ),
        )
    )

    panel.to_csv(
        CONTROLLER_PANEL_FILE,
        index=False,
    )


    # ========================================================
    # 20. OVERALL DESCRIPTIVE SUMMARY
    #
    # 21 climate-year observations per method/scarcity.
    # ========================================================

    overall_summary = (
        panel
        .groupby(
            [
                "Scarcity_pct",
                "Controller",
            ],
            as_index=False,
        )
        .agg(
            N_climate_years=(
                "Year",
                "size",
            ),

            Yield_retention_mean_pct=(
                "Yield_retention_pct",
                "mean",
            ),

            Yield_retention_SD_pct=(
                "Yield_retention_pct",
                "std",
            ),

            Worst_field_retention_mean_pct=(
                "Worst_field_retention_pct",
                "mean",
            ),

            Worst_field_retention_SD_pct=(
                "Worst_field_retention_pct",
                "std",
            ),

            Jain_mean=(
                "Jain",
                "mean",
            ),

            Jain_SD=(
                "Jain",
                "std",
            ),

            Water_productivity_mean=(
                "Water_productivity",
                "mean",
            ),

            Water_productivity_SD=(
                "Water_productivity",
                "std",
            ),

            Budget_utilization_mean_pct=(
                "Budget_utilization_pct",
                "mean",
            ),

            Decision_latency_mean_ms=(
                "Decision_latency_mean_ms",
                "mean",
            ),

            Request_days_mean=(
                "Request_days",
                "mean",
            ),

            Competition_days_mean=(
                "Competition_days",
                "mean",
            ),

            Irrigation_days_mean=(
                "Irrigation_days",
                "mean",
            ),

            Binding_fraction=(
                "Budget_binding_fraction_across_seeds",
                "mean",
            ),
        )
    )

    overall_summary.to_csv(
        OVERALL_SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # 21. CLIMATE-SPECIFIC DESCRIPTIVE SUMMARY
    #
    # 7 years per climate/controller/scarcity.
    # ========================================================

    climate_summary = (
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
            N_years=(
                "Year",
                "size",
            ),

            Yield_retention_mean_pct=(
                "Yield_retention_pct",
                "mean",
            ),

            Yield_retention_SD_pct=(
                "Yield_retention_pct",
                "std",
            ),

            Worst_field_retention_mean_pct=(
                "Worst_field_retention_pct",
                "mean",
            ),

            Worst_field_retention_SD_pct=(
                "Worst_field_retention_pct",
                "std",
            ),

            Jain_mean=(
                "Jain",
                "mean",
            ),

            Water_productivity_mean=(
                "Water_productivity",
                "mean",
            ),

            Budget_utilization_mean_pct=(
                "Budget_utilization_pct",
                "mean",
            ),

            Request_days_mean=(
                "Request_days",
                "mean",
            ),

            Competition_days_mean=(
                "Competition_days",
                "mean",
            ),

            Binding_fraction=(
                "Budget_binding_fraction_across_seeds",
                "mean",
            ),
        )
    )

    climate_summary.to_csv(
        CLIMATE_SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # 22. BINDING-WATER DIAGNOSTICS
    # ========================================================

    binding_summary = (
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
            N_cases=(
                "Year",
                "size",
            ),

            Mean_budget_utilization_pct=(
                "Budget_utilization_pct",
                "mean",
            ),

            Mean_request_days=(
                "Request_days",
                "mean",
            ),

            Mean_competition_days=(
                "Competition_days",
                "mean",
            ),

            Mean_irrigation_days=(
                "Irrigation_days",
                "mean",
            ),

            Fraction_budget_binding=(
                "Budget_binding_fraction_across_seeds",
                "mean",
            ),
        )
    )

    binding_summary.to_csv(
        BINDING_SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # 23. PPO FINAL-SEED ROBUSTNESS AT 40%
    #
    # Supporting only. These seeds are NOT independent
    # climate replicates and are NOT used as n=10 inferential
    # observations.
    # ========================================================

    ppo_40 = raw_df[
        (
            raw_df[
                "Controller"
            ]
            == "Optimized PPO"
        )
        &
        (
            np.isclose(
                raw_df[
                    "Scarcity_fraction"
                ],
                0.40,
            )
        )
    ].copy()

    ppo_seed_robustness = (
        ppo_40
        .groupby(
            "PPO_seed",
            as_index=False,
        )
        .agg(
            N_climate_years=(
                "Year",
                "size",
            ),

            Mean_yield_retention_pct=(
                "Yield_retention_pct",
                "mean",
            ),

            Worst_field_retention_mean_pct=(
                "Worst_field_retention_pct",
                "mean",
            ),

            Jain_mean=(
                "Jain",
                "mean",
            ),

            Water_productivity_mean=(
                "Water_productivity",
                "mean",
            ),

            Budget_utilization_mean_pct=(
                "Budget_utilization_pct",
                "mean",
            ),
        )
    )

    ppo_seed_robustness[
        "PPO_seed"
    ] = (
        ppo_seed_robustness[
            "PPO_seed"
        ]
        .astype(int)
    )

    ppo_seed_robustness.to_csv(
        PPO_SEED_ROBUSTNESS_FILE,
        index=False,
    )


    # ========================================================
    # 24. FINAL INTEGRITY RECORD
    # ========================================================

    runtime_seconds = (
        time.perf_counter()
        - start_all
    )

    integrity = {
        "milestone":
            "53J",

        "status":
            "COMPLETE",

        "controller_final_test_opened":
            True,

        "controller_performance_seen":
            True,

        "climates":
            CLIMATES,

        "years":
            FINAL_TEST_YEARS,

        "scarcity_levels":
            SCARCITY_LEVELS,

        "controllers":
            CONTROLLERS,

        "common_cases":
            63,

        "equal_episodes":
            63,

        "priority_episodes":
            63,

        "ppo_seed_episodes":
            630,

        "total_raw_episodes":
            int(
                len(
                    raw_df
                )
            ),

        "ppo_ensemble_rows":
            int(
                len(
                    ppo_ensemble
                )
            ),

        "controller_panel_rows":
            int(
                len(
                    panel
                )
            ),

        "all_10_ppo_seeds_retained":
            True,

        "ppo_training_performed":
            False,

        "ppo_model_selection":
            False,

        "ppo_seed_selection":
            False,

        "controller_tuning_performed":
            False,

        "statistical_tests_performed":
            False,

        "primary_experimental_unit":
            "climate-year",

        "ppo_seeds_averaged_before_inference":
            True,

        "water_accounting_pass":
            True,

        "priority_definition":
            (
                "Descending normalized root-zone depletion "
                "(depletion/TAW) among requesting fields; "
                "stable field order resolves exact ties."
            ),

        "equal_definition":
            (
                "Equal water-filling among active requesters "
                "with iterative redistribution of unused share."
            ),

        "manifest_sha256":
            sha256_file(
                MANIFEST_FILE
            ),

        "case_manifest_sha256":
            sha256_file(
                CASE_FILE
            ),

        "reference_sha256":
            sha256_file(
                REFERENCE_FILE
            ),

        "raw_results_sha256":
            sha256_file(
                RAW_FILE
            ),

        "controller_panel_sha256":
            sha256_file(
                CONTROLLER_PANEL_FILE
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


    # ========================================================
    # 25. TERMINAL REPORT
    # ========================================================

    print()
    print("#" * 78)
    print("COMMON CONTROLLER EVALUATION COMPLETE")
    print("#" * 78)

    print()
    print(
        "FINAL THREE-CONTROLLER DESCRIPTIVE RESULTS"
    )

    print(
        "(mean ± SD across 21 final-test climate-years; "
        "PPO ensemble averaged across 10 seeds first)"
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
        "Budget_utilization_mean_pct",
        "Decision_latency_mean_ms",
        "Binding_fraction",
    ]

    print(
        overall_summary[
            display_cols
        ].to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.4f}"
            ),
        )
    )


    print()
    print("=" * 78)
    print("40% SCARCITY — CLIMATE-SPECIFIC RESULTS")
    print("=" * 78)

    climate_40 = (
        climate_summary[
            np.isclose(
                climate_summary[
                    "Scarcity_pct"
                ],
                40.0,
            )
        ]
        .copy()
    )

    climate_display_cols = [
        "Climate",
        "Controller",
        "Yield_retention_mean_pct",
        "Worst_field_retention_mean_pct",
        "Jain_mean",
        "Water_productivity_mean",
        "Budget_utilization_mean_pct",
        "Binding_fraction",
    ]

    print(
        climate_40[
            climate_display_cols
        ].to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.4f}"
            ),
        )
    )


    print()
    print("=" * 78)
    print("PPO SEED ROBUSTNESS AT 40%")
    print("=" * 78)

    print(
        ppo_seed_robustness.to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.4f}"
            ),
        )
    )


    print()
    print("=" * 78)
    print("FINAL-TEST INTEGRITY")
    print("=" * 78)

    print(
        "Frozen common cases        : 63"
    )

    print(
        "Equal episodes             : 63"
    )

    print(
        "Priority episodes          : 63"
    )

    print(
        "PPO seed episodes          : 630"
    )

    print(
        f"Total final episodes       : "
        f"{len(raw_df)}"
    )

    print(
        "PPO training               : NO"
    )

    print(
        "PPO model selection        : NO"
    )

    print(
        "PPO seed selection         : NO"
    )

    print(
        "All 10 PPO seeds retained  : YES"
    )

    print(
        "Water-accounting checks    : PASS"
    )

    print(
        "Statistical tests          : NOT YET"
    )

    print(
        "Controller final test OPEN : YES"
    )

    print(
        f"Runtime                    : "
        f"{runtime_seconds:.2f} s "
        f"({runtime_seconds / 60.0:.2f} min)"
    )


    print()
    print("Saved:")

    print(
        f"  {RAW_FILE}"
    )

    print(
        f"  {PPO_ENSEMBLE_FILE}"
    )

    print(
        f"  {CONTROLLER_PANEL_FILE}"
    )

    print(
        f"  {OVERALL_SUMMARY_FILE}"
    )

    print(
        f"  {CLIMATE_SUMMARY_FILE}"
    )

    print(
        f"  {BINDING_SUMMARY_FILE}"
    )

    print(
        f"  {PPO_SEED_ROBUSTNESS_FILE}"
    )

    print(
        f"  {INTEGRITY_FILE}"
    )


    print()
    print(
        "NEXT: perform the predeclared Friedman + "
        "paired two-sided Wilcoxon + Holm analysis "
        "using climate-year as the experimental unit. "
        "No controller tuning is permitted after this point."
    )


if __name__ == "__main__":

    main()
