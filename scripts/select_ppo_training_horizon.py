# ============================================================
# PPO TRAINING-HORIZON SELECTION
# MULTI-CLIMATE PPO HORIZON RECONFIRMATION
#
# DEVELOPMENT ONLY:
#   - 3 temporal folds
#   - 3 tuning seeds
#   - 4 frozen horizon checkpoints
#   - 3 climates
#   - 7 validation years per fold
#
# FINAL TEST 2019-2025 IS NEVER ACCESSED.
#
# IMPORTANT
# ---------
# - PPO algorithm is frozen from Milestone 53E.
# - Reward is frozen from Milestone 53E.
# - Horizon is selected using AGRONOMIC validation metrics,
#   not PPO training reward.
# - No Equal/Priority evaluation is performed here.
# ============================================================

from pathlib import Path
import json
import math
import time
import gc

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import multiclimate_optimization_core as core


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

SPEC_FILE = (
    ROOT
    / "training_protocol"
    / "multiclimate_ppo_training_spec_frozen.json"
)

OUTPUT_DIR = (
    ROOT
    / "horizon_reconfirmation"
)

MODELS_DIR = (
    OUTPUT_DIR
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_VALIDATION_FILE = (
    OUTPUT_DIR
    / "ppo_horizon_selection_validation_episode_results.csv"
)

REPLICATE_FILE = (
    OUTPUT_DIR
    / "ppo_horizon_selection_fold_seed_checkpoint_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "ppo_horizon_selection_horizon_summary.csv"
)

SELECTION_FILE = (
    OUTPUT_DIR
    / "ppo_horizon_selection_horizon_selection.json"
)

PROGRESS_FILE = (
    OUTPUT_DIR
    / "ppo_horizon_selection_completed_replicates.json"
)


# ============================================================
# 2. LOAD FROZEN TRAINING SPECIFICATION
# ============================================================

if not SPEC_FILE.exists():
    raise FileNotFoundError(
        f"Missing frozen specification: {SPEC_FILE}"
    )

with open(
    SPEC_FILE,
    "r",
    encoding="utf-8",
) as f:
    spec = json.load(f)


DEVELOPMENT_YEARS = [
    int(y)
    for y in spec["development_years"]
]

FINAL_TEST_YEARS = [
    int(y)
    for y in spec["protected_final_test_years"]
]

CLIMATES = list(
    spec["climates"]
)

TEMPORAL_FOLDS = list(
    spec["temporal_folds"]
)

TUNING_SEEDS = [
    int(x)
    for x in spec["tuning_seeds"]
]

HORIZON_CANDIDATES = [
    int(x)
    for x in spec["horizon_candidates"]
]

HP = dict(
    spec["ppo_hyperparameters"]
)

REWARD = dict(
    spec["reward_configuration"]
)

TRAINING_SCARCITY = float(
    spec["training_scarcity"]
)


# ============================================================
# 3. STRICT FROZEN-PROTOCOL CHECKS
# ============================================================

assert DEVELOPMENT_YEARS == list(
    range(1984, 2019)
)

assert FINAL_TEST_YEARS == list(
    range(2019, 2026)
)

assert CLIMATES == [
    "Tunis",
    "Niamey",
    "Cotonou",
]

assert TUNING_SEEDS == [
    101,
    202,
    303,
]

assert HORIZON_CANDIDATES == [
    51200,
    76800,
    102400,
    153600,
]

assert abs(
    TRAINING_SCARCITY - 0.40
) < 1e-12

if set(DEVELOPMENT_YEARS) & set(
    FINAL_TEST_YEARS
):
    raise RuntimeError(
        "Development/final-test overlap."
    )

if any(
    y >= 2019
    for y in DEVELOPMENT_YEARS
):
    raise RuntimeError(
        "Protected final-test year entered development."
    )


# ============================================================
# 4. FORCE FROZEN REWARD INTO CORE
# ============================================================

core.configure_reward(
    daily_stress_weight=(
        REWARD[
            "daily_stress_weight"
        ]
    ),
    terminal_yield_weight=(
        REWARD[
            "terminal_yield_weight"
        ]
    ),
    terminal_jain_weight=(
        REWARD[
            "terminal_jain_weight"
        ]
    ),
    terminal_wp_weight=(
        REWARD[
            "terminal_wp_weight"
        ]
    ),
    terminal_loss_weight=(
        REWARD[
            "terminal_loss_weight"
        ]
    ),
    terminal_worst_weight=(
        REWARD[
            "terminal_worst_weight"
        ]
    ),
)

# The core was built with 40% as the development scarcity.
if abs(
    float(core.SCARCITY_FRACTION)
    - TRAINING_SCARCITY
) > 1e-12:
    raise RuntimeError(
        "Core scarcity does not match frozen "
        "53E training scarcity."
    )


# ============================================================
# 5. PPO FACTORY
# ============================================================

def build_model(
    train_years,
    seed,
):
    """
    Build one fresh PPO model for a fold/seed replicate.
    """

    env = core.MultiClimateSharedWaterEnv(
        years=train_years,
        climates=CLIMATES,
        seed=seed,
    )

    env = Monitor(env)

    policy_kwargs = {
        "net_arch":
            list(
                HP[
                    "net_arch"
                ]
            )
    }

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(
            HP[
                "learning_rate"
            ]
        ),
        n_steps=int(
            HP[
                "n_steps"
            ]
        ),
        batch_size=int(
            HP[
                "batch_size"
            ]
        ),
        n_epochs=int(
            HP[
                "n_epochs"
            ]
        ),
        gamma=float(
            HP[
                "gamma"
            ]
        ),
        gae_lambda=float(
            HP[
                "gae_lambda"
            ]
        ),
        clip_range=float(
            HP[
                "clip_range"
            ]
        ),
        ent_coef=float(
            HP[
                "ent_coef"
            ]
        ),
        vf_coef=float(
            HP[
                "vf_coef"
            ]
        ),
        max_grad_norm=float(
            HP[
                "max_grad_norm"
            ]
        ),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=int(seed),
        device="auto",
    )

    return model, env


# ============================================================
# 6. DETERMINISTIC VALIDATION EPISODE
# ============================================================

def evaluate_one_episode(
    model,
    climate,
    year,
    validation_years,
):
    """
    Evaluate one fixed climate-year using deterministic PPO.
    """

    year = int(year)
    climate = str(climate)

    if year in FINAL_TEST_YEARS or year >= 2019:
        raise RuntimeError(
            f"FINAL TEST ACCESS BLOCKED in 53F: "
            f"{climate} {year}"
        )

    if year not in validation_years:
        raise RuntimeError(
            f"{year} is outside current validation years."
        )

    env = core.MultiClimateSharedWaterEnv(
        years=validation_years,
        climates=CLIMATES,
        seed=0,
    )

    obs, reset_info = env.reset(
        options={
            "climate": climate,
            "year": year,
        }
    )

    terminated = False
    truncated = False
    decision_times_ms = []

    while not (
        terminated
        or truncated
    ):

        t0 = time.perf_counter()

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        t1 = time.perf_counter()

        decision_times_ms.append(
            (t1 - t0) * 1000.0
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
            f"{climate} {year}"
        )

    final_results = info[
        "final_results"
    ]

    if final_results is None:
        raise RuntimeError(
            f"Missing final results: "
            f"{climate} {year}"
        )

    # ----------------------------------------
    # WATER ACCOUNTING
    # ----------------------------------------

    allocated = float(
        env.total_allocated
    )

    aquacrop_water = float(
        final_results[
            "total_irrigation"
        ]
    )

    accounting_difference = (
        allocated
        - aquacrop_water
    )

    if abs(
        accounting_difference
    ) > 1e-6:
        raise RuntimeError(
            f"Water accounting failed for "
            f"{climate} {year}: "
            f"{accounting_difference}"
        )

    seasonal_budget = float(
        env.seasonal_budget
    )

    budget_utilization = (
        allocated
        / seasonal_budget
        * 100.0
        if seasonal_budget > 1e-12
        else 0.0
    )

    mean_latency = (
        float(
            np.mean(
                decision_times_ms
            )
        )
        if decision_times_ms
        else 0.0
    )

    return {
        "Climate":
            climate,

        "Year":
            year,

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
            aquacrop_water,

        "Seasonal_budget_mm":
            seasonal_budget,

        "Budget_utilization_pct":
            float(
                budget_utilization
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

        "Steps":
            int(
                env.step_count
            ),

        "Accounting_difference_mm":
            float(
                accounting_difference
            ),

        "Decision_latency_mean_ms":
            mean_latency,
    }


# ============================================================
# 7. VALIDATE ONE CHECKPOINT
# ============================================================

def validate_checkpoint(
    model,
    fold_name,
    seed,
    checkpoint,
    validation_years,
):

    rows = []

    for climate in CLIMATES:

        for year in validation_years:

            result = evaluate_one_episode(
                model=model,
                climate=climate,
                year=year,
                validation_years=validation_years,
            )

            result[
                "Fold"
            ] = fold_name

            result[
                "Seed"
            ] = int(seed)

            result[
                "Checkpoint"
            ] = int(checkpoint)

            rows.append(
                result
            )

    df = pd.DataFrame(
        rows
    )

    expected = (
        len(CLIMATES)
        * len(validation_years)
    )

    if len(df) != expected:
        raise RuntimeError(
            f"{fold_name} seed {seed} checkpoint "
            f"{checkpoint}: expected {expected} "
            f"validation rows, got {len(df)}."
        )

    # Each climate gets exactly 7 validation years.
    counts = (
        df.groupby(
            "Climate"
        )
        .size()
    )

    if not (
        counts
        == len(validation_years)
    ).all():
        raise RuntimeError(
            "Unbalanced climate validation cases."
        )

    return df


# ============================================================
# 8. FOLD-SEED CHECKPOINT METRICS
# ============================================================

def aggregate_replicate_metrics(
    validation_df,
):

    climate_metrics = (
        validation_df
        .groupby(
            "Climate",
            as_index=False,
        )
        .agg(
            Mean_yield_retention=(
                "Yield_retention",
                "mean",
            ),

            Mean_worst_field_retention=(
                "Worst_field_retention",
                "mean",
            ),

            Mean_jain=(
                "Jain",
                "mean",
            ),

            Mean_water_productivity=(
                "Water_productivity",
                "mean",
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
        )
    )

    macro_yield_retention = float(
        climate_metrics[
            "Mean_yield_retention"
        ].mean()
    )

    minimum_climate_retention = float(
        climate_metrics[
            "Mean_yield_retention"
        ].min()
    )

    macro_worst_field_retention = float(
        climate_metrics[
            "Mean_worst_field_retention"
        ].mean()
    )

    macro_jain = float(
        climate_metrics[
            "Mean_jain"
        ].mean()
    )

    macro_wp = float(
        climate_metrics[
            "Mean_water_productivity"
        ].mean()
    )

    macro_budget_utilization = float(
        climate_metrics[
            "Mean_budget_utilization_pct"
        ].mean()
    )

    # Keep climate-specific retention for audit/tie-breaking.
    by_climate = {
        row[
            "Climate"
        ]:
        float(
            row[
                "Mean_yield_retention"
            ]
        )
        for _, row
        in climate_metrics.iterrows()
    }

    return {
        "Macro_yield_retention":
            macro_yield_retention,

        "Macro_yield_retention_pct":
            macro_yield_retention
            * 100.0,

        "Minimum_climate_retention":
            minimum_climate_retention,

        "Minimum_climate_retention_pct":
            minimum_climate_retention
            * 100.0,

        "Macro_worst_field_retention":
            macro_worst_field_retention,

        "Macro_worst_field_retention_pct":
            macro_worst_field_retention
            * 100.0,

        "Macro_jain":
            macro_jain,

        "Macro_water_productivity":
            macro_wp,

        "Macro_budget_utilization_pct":
            macro_budget_utilization,

        "Tunis_mean_retention_pct":
            by_climate[
                "Tunis"
            ]
            * 100.0,

        "Niamey_mean_retention_pct":
            by_climate[
                "Niamey"
            ]
            * 100.0,

        "Cotonou_mean_retention_pct":
            by_climate[
                "Cotonou"
            ]
            * 100.0,
    }


# ============================================================
# 9. INCREMENTAL SAVE HELPERS
# ============================================================

def read_csv_or_empty(
    path,
):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_progress(
    completed,
):
    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "completed_fold_seed_replicates":
                    sorted(
                        completed
                    ),

                "final_test_opened":
                    False,
            },
            f,
            indent=2,
        )


def load_completed():

    if not PROGRESS_FILE.exists():
        return set()

    with open(
        PROGRESS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return set(
        data.get(
            "completed_fold_seed_replicates",
            [],
        )
    )


# ============================================================
# 10. HORIZON SUMMARY
# ============================================================

def build_horizon_summary(
    replicate_df,
):

    rows = []

    expected_replicates = (
        len(TEMPORAL_FOLDS)
        * len(TUNING_SEEDS)
    )

    for checkpoint in HORIZON_CANDIDATES:

        d = replicate_df[
            replicate_df[
                "Checkpoint"
            ]
            == checkpoint
        ].copy()

        if len(d) != expected_replicates:
            raise RuntimeError(
                f"Checkpoint {checkpoint}: expected "
                f"{expected_replicates} fold-seed "
                f"replicates, got {len(d)}."
            )

        values = d[
            "Macro_yield_retention"
        ].to_numpy(
            dtype=float
        )

        mean_retention = float(
            np.mean(values)
        )

        sd_retention = float(
            np.std(
                values,
                ddof=1,
            )
        )

        se_retention = float(
            sd_retention
            / math.sqrt(
                len(values)
            )
        )

        rows.append(
            {
                "Checkpoint":
                    int(
                        checkpoint
                    ),

                "Replicates":
                    int(
                        len(d)
                    ),

                "Mean_macro_yield_retention":
                    mean_retention,

                "Mean_macro_yield_retention_pct":
                    mean_retention
                    * 100.0,

                "SD_macro_yield_retention":
                    sd_retention,

                "SD_macro_yield_retention_pct":
                    sd_retention
                    * 100.0,

                "SE_macro_yield_retention":
                    se_retention,

                "SE_macro_yield_retention_pct":
                    se_retention
                    * 100.0,

                "Mean_minimum_climate_retention":
                    float(
                        d[
                            "Minimum_climate_retention"
                        ].mean()
                    ),

                "Mean_minimum_climate_retention_pct":
                    float(
                        d[
                            "Minimum_climate_retention_pct"
                        ].mean()
                    ),

                "Mean_macro_worst_field_retention":
                    float(
                        d[
                            "Macro_worst_field_retention"
                        ].mean()
                    ),

                "Mean_macro_worst_field_retention_pct":
                    float(
                        d[
                            "Macro_worst_field_retention_pct"
                        ].mean()
                    ),

                "Mean_macro_jain":
                    float(
                        d[
                            "Macro_jain"
                        ].mean()
                    ),

                "Mean_macro_water_productivity":
                    float(
                        d[
                            "Macro_water_productivity"
                        ].mean()
                    ),

                "Mean_macro_budget_utilization_pct":
                    float(
                        d[
                            "Macro_budget_utilization_pct"
                        ].mean()
                    ),

                "Mean_Tunis_retention_pct":
                    float(
                        d[
                            "Tunis_mean_retention_pct"
                        ].mean()
                    ),

                "Mean_Niamey_retention_pct":
                    float(
                        d[
                            "Niamey_mean_retention_pct"
                        ].mean()
                    ),

                "Mean_Cotonou_retention_pct":
                    float(
                        d[
                            "Cotonou_mean_retention_pct"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. PREDECLARED ONE-SE SELECTION
# ============================================================

def select_horizon(
    summary_df,
):

    best_index = (
        summary_df[
            "Mean_macro_yield_retention"
        ]
        .idxmax()
    )

    best_row = (
        summary_df
        .loc[
            best_index
        ]
    )

    best_mean = float(
        best_row[
            "Mean_macro_yield_retention"
        ]
    )

    best_se = float(
        best_row[
            "SE_macro_yield_retention"
        ]
    )

    threshold = (
        best_mean
        - best_se
    )

    summary_df = (
        summary_df.copy()
    )

    summary_df[
        "One_SE_eligible"
    ] = (
        summary_df[
            "Mean_macro_yield_retention"
        ]
        >= threshold
        - 1e-15
    )

    eligible = (
        summary_df[
            summary_df[
                "One_SE_eligible"
            ]
        ]
        .copy()
    )

    if eligible.empty:
        raise RuntimeError(
            "No one-SE eligible horizon."
        )

    # Predeclared tie-break order:
    # 1) highest minimum-climate retention
    # 2) highest macro worst-field retention
    # 3) highest macro mean retention
    # 4) shortest horizon
    eligible = eligible.sort_values(
        by=[
            "Mean_minimum_climate_retention",
            "Mean_macro_worst_field_retention",
            "Mean_macro_yield_retention",
            "Checkpoint",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    )

    selected = (
        eligible.iloc[0]
    )

    return (
        summary_df,
        {
            "best_mean_checkpoint":
                int(
                    best_row[
                        "Checkpoint"
                    ]
                ),

            "best_mean_macro_retention":
                best_mean,

            "best_mean_macro_retention_pct":
                best_mean
                * 100.0,

            "best_mean_SE":
                best_se,

            "best_mean_SE_pct":
                best_se
                * 100.0,

            "one_SE_threshold":
                threshold,

            "one_SE_threshold_pct":
                threshold
                * 100.0,

            "eligible_horizons":
                [
                    int(x)
                    for x in eligible[
                        "Checkpoint"
                    ].tolist()
                ],

            "selected_horizon":
                int(
                    selected[
                        "Checkpoint"
                    ]
                ),

            "selection_rule":
                (
                    "One-SE eligibility on macro-climate "
                    "mean yield retention; among eligible: "
                    "highest mean minimum-climate retention, "
                    "then highest mean macro worst-field "
                    "retention, then highest macro mean "
                    "retention, then shortest horizon."
                ),

            "final_test_used":
                False,

            "ppo_reward_used_for_selection":
                False,
        }
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    start_all = time.perf_counter()

    print()
    print("#" * 78)
    print("PPO TRAINING-HORIZON SELECTION")
    print(
        "MULTI-CLIMATE PPO HORIZON "
        "RECONFIRMATION"
    )
    print("#" * 78)

    print()
    print("Development only       : 1984-2018")
    print("Protected final test   : 2019-2025")
    print(f"Climates               : {CLIMATES}")
    print(f"Tuning seeds           : {TUNING_SEEDS}")
    print(f"Horizon candidates     : {HORIZON_CANDIDATES}")
    print("Training scarcity      : 40%")
    print("Validation metric      : macro-climate yield retention")
    print("PPO reward selects     : NO")
    print("Equal/Priority used    : NO")
    print("Final test used        : NO")

    completed = load_completed()

    existing_raw = read_csv_or_empty(
        RAW_VALIDATION_FILE
    )

    existing_replicates = read_csv_or_empty(
        REPLICATE_FILE
    )

    all_new_raw = []
    all_new_replicates = []

    total_replicates = (
        len(TEMPORAL_FOLDS)
        * len(TUNING_SEEDS)
    )

    replicate_counter = 0

    # ========================================================
    # TRAIN FOLD-SEED REPLICATES
    # ========================================================

    for fold in TEMPORAL_FOLDS:

        fold_name = str(
            fold[
                "name"
            ]
        )

        train_years = [
            int(y)
            for y in fold[
                "train_years"
            ]
        ]

        validation_years = [
            int(y)
            for y in fold[
                "validation_years"
            ]
        ]

        # Strict leakage guard.
        if (
            set(train_years)
            & set(validation_years)
        ):
            raise RuntimeError(
                f"{fold_name}: train/validation overlap."
            )

        if (
            set(train_years)
            & set(FINAL_TEST_YEARS)
        ):
            raise RuntimeError(
                f"{fold_name}: final-test leakage."
            )

        if (
            set(validation_years)
            & set(FINAL_TEST_YEARS)
        ):
            raise RuntimeError(
                f"{fold_name}: final-test leakage."
            )

        for seed in TUNING_SEEDS:

            replicate_counter += 1

            replicate_key = (
                f"{fold_name}__seed_{seed}"
            )

            print()
            print("=" * 78)
            print(
                f"REPLICATE {replicate_counter}/"
                f"{total_replicates}: "
                f"{fold_name}, seed {seed}"
            )
            print("=" * 78)

            print(
                f"Train      : "
                f"{train_years[0]}-"
                f"{train_years[-1]}"
            )

            print(
                f"Validation : "
                f"{validation_years[0]}-"
                f"{validation_years[-1]}"
            )

            if replicate_key in completed:

                print(
                    "Status     : already complete; SKIPPED"
                )

                continue

            # Remove any stale rows from an interrupted previous
            # attempt of this same fold-seed replicate.
            if not existing_raw.empty:
                existing_raw = existing_raw[
                    ~(
                        (
                            existing_raw[
                                "Fold"
                            ]
                            == fold_name
                        )
                        &
                        (
                            existing_raw[
                                "Seed"
                            ].astype(int)
                            == int(seed)
                        )
                    )
                ].copy()

            if not existing_replicates.empty:
                existing_replicates = (
                    existing_replicates[
                        ~(
                            (
                                existing_replicates[
                                    "Fold"
                                ]
                                == fold_name
                            )
                            &
                            (
                                existing_replicates[
                                    "Seed"
                                ].astype(int)
                                == int(seed)
                            )
                        )
                    ]
                    .copy()
                )

            # Reproducibility.
            np.random.seed(
                seed
            )

            torch.manual_seed(
                seed
            )

            model, train_env = build_model(
                train_years=train_years,
                seed=seed,
            )

            previous_checkpoint = 0

            replicate_raw = []
            replicate_metrics = []

            replicate_start = (
                time.perf_counter()
            )

            for checkpoint in HORIZON_CANDIDATES:

                additional_steps = (
                    checkpoint
                    - previous_checkpoint
                )

                if additional_steps <= 0:
                    raise RuntimeError(
                        "Checkpoint order is invalid."
                    )

                print()
                print(
                    f"  Training to "
                    f"{checkpoint:,} timesteps "
                    f"(+{additional_steps:,})..."
                )

                train_start = (
                    time.perf_counter()
                )

                model.learn(
                    total_timesteps=(
                        additional_steps
                    ),
                    reset_num_timesteps=False,
                    progress_bar=False,
                )

                train_seconds = (
                    time.perf_counter()
                    - train_start
                )

                actual_steps = int(
                    model.num_timesteps
                )

                if actual_steps != checkpoint:
                    raise RuntimeError(
                        f"Expected {checkpoint} "
                        f"timesteps, got {actual_steps}."
                    )

                # Save audit checkpoint.
                model_path = (
                    MODELS_DIR
                    / (
                        f"{fold_name}"
                        f"_seed{seed}"
                        f"_step{checkpoint}"
                    )
                )

                model.save(
                    str(
                        model_path
                    )
                )

                # ----------------------------------------
                # VALIDATION
                # ----------------------------------------

                print(
                    f"  Validating {len(CLIMATES) * len(validation_years)} "
                    f"climate-years..."
                )

                eval_start = (
                    time.perf_counter()
                )

                val_df = (
                    validate_checkpoint(
                        model=model,
                        fold_name=fold_name,
                        seed=seed,
                        checkpoint=checkpoint,
                        validation_years=(
                            validation_years
                        ),
                    )
                )

                eval_seconds = (
                    time.perf_counter()
                    - eval_start
                )

                metrics = (
                    aggregate_replicate_metrics(
                        val_df
                    )
                )

                metrics.update(
                    {
                        "Fold":
                            fold_name,

                        "Seed":
                            int(
                                seed
                            ),

                        "Checkpoint":
                            int(
                                checkpoint
                            ),

                        "Train_start_year":
                            int(
                                train_years[0]
                            ),

                        "Train_end_year":
                            int(
                                train_years[-1]
                            ),

                        "Validation_start_year":
                            int(
                                validation_years[0]
                            ),

                        "Validation_end_year":
                            int(
                                validation_years[-1]
                            ),

                        "Training_segment_seconds":
                            float(
                                train_seconds
                            ),

                        "Validation_seconds":
                            float(
                                eval_seconds
                            ),
                    }
                )

                replicate_raw.append(
                    val_df
                )

                replicate_metrics.append(
                    metrics
                )

                print(
                    "  Macro yield retention : "
                    f"{metrics['Macro_yield_retention_pct']:.3f}%"
                )

                print(
                    "  Min-climate retention : "
                    f"{metrics['Minimum_climate_retention_pct']:.3f}%"
                )

                print(
                    "  Macro worst retention : "
                    f"{metrics['Macro_worst_field_retention_pct']:.3f}%"
                )

                print(
                    "  Climate retention     : "
                    f"Tunis={metrics['Tunis_mean_retention_pct']:.2f}% | "
                    f"Niamey={metrics['Niamey_mean_retention_pct']:.2f}% | "
                    f"Cotonou={metrics['Cotonou_mean_retention_pct']:.2f}%"
                )

                previous_checkpoint = (
                    checkpoint
                )

            # ==================================================
            # COMPLETE THIS REPLICATE ATOMICALLY
            # ==================================================

            replicate_raw_df = pd.concat(
                replicate_raw,
                ignore_index=True,
            )

            replicate_metrics_df = pd.DataFrame(
                replicate_metrics
            )

            all_new_raw.append(
                replicate_raw_df
            )

            all_new_replicates.append(
                replicate_metrics_df
            )

            # Save everything completed so far.
            raw_parts = [
                existing_raw
            ]

            replicate_parts = [
                existing_replicates
            ]

            if all_new_raw:
                raw_parts.extend(
                    all_new_raw
                )

            if all_new_replicates:
                replicate_parts.extend(
                    all_new_replicates
                )

            raw_to_save = pd.concat(
                [
                    x
                    for x in raw_parts
                    if not x.empty
                ],
                ignore_index=True,
            )

            replicate_to_save = pd.concat(
                [
                    x
                    for x in replicate_parts
                    if not x.empty
                ],
                ignore_index=True,
            )

            raw_to_save.to_csv(
                RAW_VALIDATION_FILE,
                index=False,
            )

            replicate_to_save.to_csv(
                REPLICATE_FILE,
                index=False,
            )

            completed.add(
                replicate_key
            )

            save_progress(
                completed
            )

            replicate_minutes = (
                time.perf_counter()
                - replicate_start
            ) / 60.0

            print()
            print(
                f"Replicate complete in "
                f"{replicate_minutes:.2f} min"
            )

            # Release memory before next model.
            train_env.close()
            del model
            del train_env

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # LOAD COMPLETE RESULTS
    # ========================================================

    raw_df = pd.read_csv(
        RAW_VALIDATION_FILE
    )

    replicate_df = pd.read_csv(
        REPLICATE_FILE
    )

    expected_raw_rows = (
        len(TEMPORAL_FOLDS)
        * len(TUNING_SEEDS)
        * len(HORIZON_CANDIDATES)
        * len(CLIMATES)
        * 7
    )

    expected_replicate_rows = (
        len(TEMPORAL_FOLDS)
        * len(TUNING_SEEDS)
        * len(HORIZON_CANDIDATES)
    )

    if len(raw_df) != expected_raw_rows:
        raise RuntimeError(
            f"Expected {expected_raw_rows} raw validation rows, "
            f"got {len(raw_df)}."
        )

    if len(replicate_df) != expected_replicate_rows:
        raise RuntimeError(
            f"Expected {expected_replicate_rows} replicate rows, "
            f"got {len(replicate_df)}."
        )

    if (
        raw_df[
            "Year"
        ].astype(int)
        .isin(
            FINAL_TEST_YEARS
        )
        .any()
    ):
        raise RuntimeError(
            "FINAL TEST LEAKAGE detected in saved validation."
        )

    # ========================================================
    # SUMMARY + PREDECLARED SELECTION
    # ========================================================

    summary_df = build_horizon_summary(
        replicate_df
    )

    (
        summary_df,
        selection,
    ) = select_horizon(
        summary_df
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    selection[
        "development_years"
    ] = DEVELOPMENT_YEARS

    selection[
        "protected_final_test_years"
    ] = FINAL_TEST_YEARS

    selection[
        "tuning_seeds"
    ] = TUNING_SEEDS

    selection[
        "horizon_candidates"
    ] = HORIZON_CANDIDATES

    selection[
        "total_training_replicates"
    ] = int(
        total_replicates
    )

    selection[
        "validation_climate_years_per_checkpoint_replicate"
    ] = 21

    selection[
        "raw_validation_rows"
    ] = int(
        len(raw_df)
    )

    selection[
        "final_test_opened"
    ] = False

    with open(
        SELECTION_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selection,
            f,
            indent=2,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    runtime_minutes = (
        time.perf_counter()
        - start_all
    ) / 60.0

    print()
    print("#" * 78)
    print("PPO TRAINING-HORIZON SELECTION COMPLETE")
    print("#" * 78)

    print()
    print("HORIZON SUMMARY")
    print()

    display_columns = [
        "Checkpoint",
        "Mean_macro_yield_retention_pct",
        "SD_macro_yield_retention_pct",
        "SE_macro_yield_retention_pct",
        "Mean_minimum_climate_retention_pct",
        "Mean_macro_worst_field_retention_pct",
        "Mean_Tunis_retention_pct",
        "Mean_Niamey_retention_pct",
        "Mean_Cotonou_retention_pct",
        "One_SE_eligible",
    ]

    print(
        summary_df[
            display_columns
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
    print("PREDECLARED ONE-SE SELECTION")
    print("=" * 78)

    print(
        "Best mean checkpoint       : "
        f"{selection['best_mean_checkpoint']:,}"
    )

    print(
        "Best mean retention        : "
        f"{selection['best_mean_macro_retention_pct']:.4f}%"
    )

    print(
        "Best-checkpoint SE         : "
        f"{selection['best_mean_SE_pct']:.4f} pp"
    )

    print(
        "One-SE threshold           : "
        f"{selection['one_SE_threshold_pct']:.4f}%"
    )

    print(
        "Eligible horizons          : "
        f"{selection['eligible_horizons']}"
    )

    print(
        "SELECTED HORIZON           : "
        f"{selection['selected_horizon']:,}"
    )

    print()
    print("=" * 78)
    print("INTEGRITY")
    print("=" * 78)

    print(
        f"Fold-seed replicates       : "
        f"{total_replicates}"
    )

    print(
        f"Checkpoint evaluations     : "
        f"{expected_replicate_rows}"
    )

    print(
        f"Validation episodes        : "
        f"{expected_raw_rows}"
    )

    print(
        "PPO reward used to select  : NO"
    )

    print(
        "Equal evaluated            : NO"
    )

    print(
        "Priority evaluated         : NO"
    )

    print(
        "2019-2025 accessed         : NO"
    )

    print(
        "Final test remains unopened: YES"
    )

    print(
        f"Total runtime              : "
        f"{runtime_minutes:.2f} min "
        f"({runtime_minutes / 60.0:.2f} h)"
    )

    print()
    print("Saved:")
    print(f"  {RAW_VALIDATION_FILE}")
    print(f"  {REPLICATE_FILE}")
    print(f"  {SUMMARY_FILE}")
    print(f"  {SELECTION_FILE}")
    print(f"  {PROGRESS_FILE}")
    print(f"  {MODELS_DIR}")

    print()
    print(
        "NEXT: freeze the selected horizon, then "
        "train all 10 final PPO seeds on the full "
        "1984-2018 multi-climate development set."
    )


if __name__ == "__main__":
    main()
