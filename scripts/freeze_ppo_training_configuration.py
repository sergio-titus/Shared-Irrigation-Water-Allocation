# ============================================================
# PPO TRAINING CONFIGURATION
# FREEZE MULTI-CLIMATE PPO TRAINING SPECIFICATION
#
# NO PPO TRAINING.
# NO BASELINE EVALUATION.
# NO 2019-2025 ACCESS.
#
# This file freezes:
#   - PPO algorithm hyperparameters
#   - reward coefficients
#   - tuning/final seeds
#   - horizon candidates
#   - climate sampling
#   - validation protocol
#   - horizon selection rule
# ============================================================

from pathlib import Path
import hashlib
import json
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

PROTOCOL_FILE = (
    ROOT
    / "protocol"
    / "multiclimate_protocol_frozen.json"
)

REFERENCE_FILE = (
    ROOT
    / "reference"
    / "development_reference_by_climate_year.csv"
)

CORE_FILE = Path(
    "multiclimate_optimization_core.py"
)

TRAINING_DIR = (
    ROOT / "training_protocol"
)

TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. REQUIRED FILES
# ============================================================

for path in [
    PROTOCOL_FILE,
    REFERENCE_FILE,
    CORE_FILE,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


# ============================================================
# 3. HASH FUNCTION
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


# ============================================================
# 4. LOAD FROZEN EXPERIMENTAL PROTOCOL
# ============================================================

with open(
    PROTOCOL_FILE,
    "r",
    encoding="utf-8",
) as f:

    protocol = json.load(f)


DEVELOPMENT_YEARS = [
    int(x)
    for x in protocol[
        "development_years"
    ]
]

FINAL_TEST_YEARS = [
    int(x)
    for x in protocol[
        "final_test_years"
    ]
]

CLIMATES = list(
    protocol[
        "climates"
    ].keys()
)


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


# ============================================================
# 5. TEMPORAL FOLDS
# ============================================================

TEMPORAL_FOLDS = [
    {
        "name": "Fold_1",
        "train_years":
            list(range(1984, 1998)),
        "validation_years":
            list(range(1998, 2005)),
    },

    {
        "name": "Fold_2",
        "train_years":
            list(range(1984, 2005)),
        "validation_years":
            list(range(2005, 2012)),
    },

    {
        "name": "Fold_3",
        "train_years":
            list(range(1984, 2012)),
        "validation_years":
            list(range(2012, 2019)),
    },
]


# ============================================================
# 6. FROZEN PPO ALGORITHM
#
# Retained from the already optimized Tunis experiment.
# No new algorithm HPO will be conducted.
# ============================================================

PPO_HYPERPARAMETERS = {

    "learning_rate":
        0.0002512430479863,

    "n_steps":
        1024,

    "batch_size":
        64,

    "n_epochs":
        10,

    "gamma":
        0.999,

    "gae_lambda":
        0.95,

    "clip_range":
        0.20,

    "ent_coef":
        0.005,

    "vf_coef":
        0.5,

    "max_grad_norm":
        0.5,

    "net_arch":
        [128, 128],
}


# ============================================================
# 7. FROZEN REWARD
#
# Retained from the final reward selected before the
# original final PPO training.
#
# This extension will NOT retune these coefficients.
# ============================================================

REWARD_CONFIGURATION = {

    "daily_stress_weight":
        0.0049575737779768,

    "terminal_yield_weight":
        1.0,

    "terminal_jain_weight":
        0.1489025972983975,

    "terminal_worst_weight":
        0.0,

    "terminal_wp_weight":
        0.1501306929871174,

    "terminal_loss_weight":
        0.5952617301954728,
}


# ============================================================
# 8. TRAINING SCARCITY
#
# PPO is developed only under the severe 40% condition.
#
# 60% and 100% remain cross-scarcity evaluations,
# not training conditions.
# ============================================================

TRAINING_SCARCITY = 0.40

FINAL_EVALUATION_SCARCITIES = [
    1.00,
    0.60,
    0.40,
]


# ============================================================
# 9. OBSERVATION / ACTION SPECIFICATION
# ============================================================

OBSERVATION_SPECIFICATION = {

    "dimension":
        34,

    "field_features":
        28,

    "global_water_time_features":
        3,

    "climate_identity_features":
        3,

    "climate_encoding": {
        "Tunis":
            [1.0, 0.0, 0.0],

        "Niamey":
            [0.0, 1.0, 0.0],

        "Cotonou":
            [0.0, 0.0, 1.0],
    },

    "future_weather_information":
        False,

    "weather_forecast_information":
        False,
}


ACTION_SPECIFICATION = {

    "dimension":
        4,

    "range":
        [0.0, 1.0],

    "interpretation":
        (
            "Each action scales the current "
            "request of one field."
        ),

    "constraint_projection":
        (
            "Proportional projection under "
            "daily shared capacity and remaining "
            "seasonal budget."
        ),
}


# ============================================================
# 10. SAMPLING
#
# Each training episode:
#
#   1. sample one climate uniformly;
#   2. sample one eligible training year uniformly.
#
# This ensures every climate has equal probability,
# independently of climatic water demand.
# ============================================================

TRAINING_SAMPLING = {

    "climate_sampling":
        "uniform",

    "year_sampling_within_climate":
        "uniform",

    "sampling_unit":
        "climate-year episode",

    "number_of_climates":
        3,

    "climate_probabilities": {
        "Tunis":
            1.0 / 3.0,

        "Niamey":
            1.0 / 3.0,

        "Cotonou":
            1.0 / 3.0,
    },
}


# ============================================================
# 11. TUNING SEEDS
#
# These seeds are used ONLY for development-stage
# horizon confirmation.
# ============================================================

TUNING_SEEDS = [
    101,
    202,
    303,
]


# ============================================================
# 12. FINAL ROBUSTNESS SEEDS
#
# Retain the same independently declared final seeds as
# the original optimized PPO analysis.
#
# No final seed may be discarded.
# ============================================================

FINAL_SEEDS = [
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


# ============================================================
# 13. HORIZON CANDIDATES
#
# Horizon is reconfirmed AFTER reward fixation.
#
# This explicitly avoids the sequencing limitation of the
# earlier experiment, where horizon selection preceded
# final reward optimization.
# ============================================================

HORIZON_CANDIDATES = [
    51200,
    76800,
    102400,
    153600,
]


N_STEPS = int(
    PPO_HYPERPARAMETERS[
        "n_steps"
    ]
)


for horizon in HORIZON_CANDIDATES:

    if horizon % N_STEPS != 0:

        raise RuntimeError(
            f"Horizon {horizon} is not "
            f"an exact multiple of "
            f"n_steps={N_STEPS}."
        )


# ============================================================
# 14. VALIDATION METRICS
# ============================================================

VALIDATION_METRICS = {

    "primary":
        (
            "macro-climate mean "
            "yield retention"
        ),

    "secondary_1":
        (
            "minimum climate-specific "
            "mean yield retention"
        ),

    "secondary_2":
        (
            "macro-climate mean "
            "worst-field retention"
        ),

    "supporting": [
        "Jain fairness",
        "water productivity",
        "budget utilization",
    ],

    "ppo_training_reward_used_for_selection":
        False,
}


# ============================================================
# 15. HORIZON SELECTION RULE
#
# Step 1:
#   For every checkpoint, fold and tuning seed,
#   evaluate all 21 validation climate-years.
#
# Step 2:
#   Within each fold-seed replicate:
#       - calculate mean retention within each climate;
#       - average the 3 climate means = macro retention.
#
# Step 3:
#   Across 3 folds x 3 seeds = 9 replicates,
#   calculate mean macro retention and SE.
#
# Step 4:
#   Find the checkpoint with highest mean.
#
# Step 5:
#   One-SE eligible:
#
#       candidate_mean >=
#       best_mean - best_SE
#
# Step 6:
#   Among eligible horizons select in this order:
#
#       a) highest minimum-climate retention
#       b) highest macro worst-field retention
#       c) highest macro mean retention
#       d) shortest horizon
#
# No final-test information may enter this decision.
# ============================================================

HORIZON_SELECTION = {

    "replicates":
        (
            "3 temporal folds x "
            "3 tuning seeds = 9"
        ),

    "validation_cases_per_replicate":
        21,

    "primary_statistic":
        (
            "macro-climate mean "
            "yield retention"
        ),

    "standard_error_unit":
        "fold-seed replicate",

    "one_se_rule":
        (
            "candidate mean >= "
            "best mean - best SE"
        ),

    "tie_break_order": [
        (
            "highest minimum-climate "
            "mean retention"
        ),
        (
            "highest macro-climate "
            "worst-field retention"
        ),
        (
            "highest macro-climate "
            "mean yield retention"
        ),
        "shortest horizon",
    ],

    "selection_uses_training_reward":
        False,

    "selection_uses_final_test":
        False,
}


# ============================================================
# 16. FINAL POLICY PROTOCOL
# ============================================================

FINAL_POLICY_PROTOCOL = {

    "after_horizon_selection":
        (
            "Retrain from scratch on all "
            "1984-2018 development years "
            "across all three climates."
        ),

    "number_of_final_seeds":
        len(FINAL_SEEDS),

    "retain_all_final_seeds":
        True,

    "seed_selection":
        False,

    "model_selection_among_final_seeds":
        False,

    "final_policy_summary":
        (
            "For each final climate-year, "
            "average predictions/results across "
            "all 10 frozen final PPO seeds "
            "before controller-level statistics."
        ),
}


# ============================================================
# 17. FINAL TEST RULES
# ============================================================

FINAL_TEST_RULES = {

    "years":
        FINAL_TEST_YEARS,

    "climates":
        CLIMATES,

    "climate_years":
        (
            len(FINAL_TEST_YEARS)
            * len(CLIMATES)
        ),

    "scarcity_levels":
        FINAL_EVALUATION_SCARCITIES,

    "final_test_must_remain_closed_until":
        (
            "PPO horizon, final training "
            "configuration, all final models, "
            "and analysis rules are frozen."
        ),

    "no_hyperparameter_tuning":
        True,

    "no_reward_tuning":
        True,

    "no_horizon_selection":
        True,

    "no_seed_selection":
        True,

    "no_baseline_tuning":
        True,
}


# ============================================================
# 18. SPECIAL HUMID-CLIMATE INTERPRETATION
# ============================================================

HUMID_CLIMATE_RULE = {

    "statement":
        (
            "A nominal scarcity scenario may be "
            "non-binding when rainfall prevents "
            "irrigation requests."
        ),

    "zero_irrigation_episode_is_error":
        False,

    "controller_or_trigger_modified_if_nonbinding":
        False,

    "report_later": [
        "request days",
        "competition days",
        "budget utilization",
        "fraction of binding climate-years",
    ],
}


# ============================================================
# 19. INTEGRITY CHECKS
# ============================================================

dev_set = set(
    DEVELOPMENT_YEARS
)

test_set = set(
    FINAL_TEST_YEARS
)


if dev_set & test_set:

    raise RuntimeError(
        "Development/test overlap."
    )


for fold in TEMPORAL_FOLDS:

    train = set(
        fold["train_years"]
    )

    validation = set(
        fold[
            "validation_years"
        ]
    )

    if train & validation:

        raise RuntimeError(
            f"{fold['name']}: "
            "training/validation overlap."
        )

    if not train.issubset(
        dev_set
    ):

        raise RuntimeError(
            f"{fold['name']}: "
            "training outside development."
        )

    if not validation.issubset(
        dev_set
    ):

        raise RuntimeError(
            f"{fold['name']}: "
            "validation outside development."
        )

    if train & test_set:

        raise RuntimeError(
            f"{fold['name']}: "
            "final-test leakage in training."
        )

    if validation & test_set:

        raise RuntimeError(
            f"{fold['name']}: "
            "final-test leakage in validation."
        )


# ============================================================
# 20. BUILD FROZEN SPECIFICATION
# ============================================================

training_spec = {

    "milestone":
        "53E",

    "status":
        "FROZEN_BEFORE_PPO_TRAINING",

    "study":
        (
            "Multi-climate PPO shared-water "
            "allocation extension"
        ),

    "development_years":
        DEVELOPMENT_YEARS,

    "protected_final_test_years":
        FINAL_TEST_YEARS,

    "climates":
        CLIMATES,

    "temporal_folds":
        TEMPORAL_FOLDS,

    "ppo_hyperparameters":
        PPO_HYPERPARAMETERS,

    "reward_configuration":
        REWARD_CONFIGURATION,

    "training_scarcity":
        TRAINING_SCARCITY,

    "final_evaluation_scarcities":
        FINAL_EVALUATION_SCARCITIES,

    "observation_specification":
        OBSERVATION_SPECIFICATION,

    "action_specification":
        ACTION_SPECIFICATION,

    "training_sampling":
        TRAINING_SAMPLING,

    "tuning_seeds":
        TUNING_SEEDS,

    "final_seeds":
        FINAL_SEEDS,

    "horizon_candidates":
        HORIZON_CANDIDATES,

    "validation_metrics":
        VALIDATION_METRICS,

    "horizon_selection":
        HORIZON_SELECTION,

    "final_policy_protocol":
        FINAL_POLICY_PROTOCOL,

    "final_test_rules":
        FINAL_TEST_RULES,

    "humid_climate_rule":
        HUMID_CLIMATE_RULE,

    "source_hashes": {

        "frozen_protocol_sha256":
            sha256_file(
                PROTOCOL_FILE
            ),

        "development_reference_sha256":
            sha256_file(
                REFERENCE_FILE
            ),

        "multiclimate_core_sha256":
            sha256_file(
                CORE_FILE
            ),
    },

    "integrity": {

        "ppo_training_performed":
            False,

        "equal_evaluated":
            False,

        "priority_evaluated":
            False,

        "final_test_opened":
            False,
    },
}


# ============================================================
# 21. SAVE JSON
# ============================================================

SPEC_FILE = (
    TRAINING_DIR
    / "multiclimate_ppo_training_spec_frozen.json"
)


with open(
    SPEC_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        training_spec,
        f,
        indent=2,
    )


# ============================================================
# 22. SAVE HORIZON MANIFEST
# ============================================================

horizon_df = pd.DataFrame(
    {
        "Horizon_timesteps":
            HORIZON_CANDIDATES,

        "N_steps":
            [
                N_STEPS
                for _
                in HORIZON_CANDIDATES
            ],

        "PPO_updates":
            [
                int(
                    horizon
                    / N_STEPS
                )
                for horizon
                in HORIZON_CANDIDATES
            ],
    }
)


HORIZON_FILE = (
    TRAINING_DIR
    / "horizon_candidates.csv"
)

horizon_df.to_csv(
    HORIZON_FILE,
    index=False,
)


# ============================================================
# 23. SAVE SEED MANIFEST
# ============================================================

seed_rows = []

for seed in TUNING_SEEDS:

    seed_rows.append(
        {
            "Seed":
                seed,

            "Purpose":
                "Horizon_confirmation",

            "Final_seed":
                False,
        }
    )


for seed in FINAL_SEEDS:

    seed_rows.append(
        {
            "Seed":
                seed,

            "Purpose":
                "Final_robustness",

            "Final_seed":
                True,
        }
    )


seed_df = pd.DataFrame(
    seed_rows
)


SEED_FILE = (
    TRAINING_DIR
    / "seed_manifest.csv"
)


seed_df.to_csv(
    SEED_FILE,
    index=False,
)


# ============================================================
# 24. TERMINAL REPORT
# ============================================================

print()
print("#" * 78)
print("PPO TRAINING CONFIGURATION")
print(
    "MULTI-CLIMATE PPO TRAINING "
    "SPECIFICATION FROZEN"
)
print("#" * 78)


print()
print("=" * 78)
print("ALGORITHM")
print("=" * 78)

for key, value in (
    PPO_HYPERPARAMETERS.items()
):

    print(
        f"{key:<20}: {value}"
    )


print()
print("=" * 78)
print("REWARD")
print("=" * 78)

for key, value in (
    REWARD_CONFIGURATION.items()
):

    print(
        f"{key:<28}: {value}"
    )


print()
print("=" * 78)
print("TRAINING DESIGN")
print("=" * 78)

print(
    "Training scarcity       : 40%"
)

print(
    "Climates                : "
    f"{CLIMATES}"
)

print(
    "Climate sampling        : uniform"
)

print(
    "Observation dimension   : 34"
)

print(
    "Tuning seeds            : "
    f"{TUNING_SEEDS}"
)

print(
    "Final seeds             : "
    f"{FINAL_SEEDS}"
)


print()
print("=" * 78)
print("HORIZON RECONFIRMATION")
print("=" * 78)

print(
    "Candidates              : "
    f"{HORIZON_CANDIDATES}"
)

print(
    "Temporal folds          : 3"
)

print(
    "Tuning seeds            : 3"
)

print(
    "Fold-seed replicates    : 9"
)

print(
    "Validation cases/model  : 21"
)

print(
    "Primary selection metric: "
    "macro-climate mean yield retention"
)

print(
    "One-SE rule             : YES"
)

print(
    "PPO reward selects model: NO"
)


print()
print("=" * 78)
print("FINAL-TEST INTEGRITY")
print("=" * 78)

print(
    "Protected years         : "
    "2019-2025"
)

print(
    "PPO training performed  : NO"
)

print(
    "Equal evaluated         : NO"
)

print(
    "Priority evaluated      : NO"
)

print(
    "Final test opened       : NO"
)


print()
print("Saved:")

print(
    f"  {SPEC_FILE}"
)

print(
    f"  {HORIZON_FILE}"
)

print(
    f"  {SEED_FILE}"
)


print()
print("=" * 78)
print("PPO TRAINING CONFIGURATION COMPLETE")
print("=" * 78)

print()
print(
    "NEXT: development-only PPO horizon "
    "reconfirmation across 3 climates, "
    "3 temporal folds and 3 tuning seeds."
)