# ============================================================
# EXPERIMENTAL PROTOCOL FREEZE
# FREEZE MULTI-CLIMATE EXPERIMENTAL PROTOCOL
#
# IMPORTANT:
# This script performs NO AquaCrop simulation,
# NO baseline evaluation,
# NO PPO training,
# and NO final-test evaluation.
#
# Its sole purpose is to permanently declare the
# development / validation / final-test partition
# BEFORE controller results are observed.
# ============================================================

from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

WEATHER_DIR = ROOT / "weather"
PROTOCOL_DIR = ROOT / "protocol"

PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)


COMMON_YEARS_FILE = (
    WEATHER_DIR / "common_complete_years.json"
)

ANNUAL_FILE = (
    WEATHER_DIR / "annual_climate_summary.csv"
)


# ============================================================
# FROZEN CLIMATES
# ============================================================

CLIMATES = {
    "Tunis": {
        "country": "Tunisia",
        "role": "Mediterranean",
    },
    "Niamey": {
        "country": "Niger",
        "role": "Hot semi-arid",
    },
    "Cotonou": {
        "country": "Benin",
        "role": "Humid tropical/coastal",
    },
}


# ============================================================
# FROZEN TEMPORAL PARTITION
# ============================================================

DEVELOPMENT_YEARS = list(range(1984, 2019))

FINAL_TEST_YEARS = list(range(2019, 2026))


TEMPORAL_FOLDS = {
    "Fold_1": {
        "train_years": list(range(1984, 1998)),
        "validation_years": list(range(1998, 2005)),
    },
    "Fold_2": {
        "train_years": list(range(1984, 2005)),
        "validation_years": list(range(2005, 2012)),
    },
    "Fold_3": {
        "train_years": list(range(1984, 2012)),
        "validation_years": list(range(2012, 2019)),
    },
}


# ============================================================
# FROZEN AGRONOMIC DESIGN
# ============================================================

AGRONOMIC_DESIGN = {
    "number_of_fields": 4,
    "crop": "Maize",
    "soil": "SandyLoam",

    "field_planting_dates": {
        "F1": "05/01",
        "F2": "05/08",
        "F3": "05/15",
        "F4": "05/22",
    },

    "simulation_window": {
        "start": "May 1",
        "end": "October 31",
    },

    "daily_system_capacity_mm": 40.0,
    "maximum_field_irrigation_mm_day": 25.0,
    "depletion_trigger": 0.40,
}


# ============================================================
# FROZEN CONTROLLERS
# ============================================================

CONTROLLERS = {
    "Equal": {
        "type": "deterministic_lightweight",
        "training_required": False,
        "description": (
            "Equal sharing among requesting fields with "
            "iterative redistribution of unused shares."
        ),
    },

    "Priority": {
        "type": "deterministic_lightweight",
        "training_required": False,
        "priority_signal": "normalized soil-water depletion",
        "description": (
            "Fields ranked by depletion; water allocated "
            "in descending priority subject to shared constraints."
        ),
    },

    "PPO": {
        "type": "reinforcement_learning",
        "training_required": True,
        "training_scope": (
            "single multi-climate policy trained across "
            "development climate-year combinations"
        ),
    },
}


# ============================================================
# SCARCITY CONDITIONS
# ============================================================

SCARCITY_LEVELS = [1.00, 0.60, 0.40]

PPO_PRIMARY_TRAINING_SCARCITY = 0.40


# ============================================================
# VALIDATE WEATHER YEARS
# ============================================================

with open(
    COMMON_YEARS_FILE,
    "r",
    encoding="utf-8",
) as f:
    common_info = json.load(f)


common_years = set(
    int(y)
    for y in common_info["common_complete_years"]
)


required_years = set(
    DEVELOPMENT_YEARS + FINAL_TEST_YEARS
)


missing_required = sorted(
    required_years - common_years
)


if missing_required:
    raise RuntimeError(
        "Protocol cannot be frozen because these "
        f"required years are incomplete: {missing_required}"
    )


# ============================================================
# LEAKAGE CHECKS
# ============================================================

dev_set = set(DEVELOPMENT_YEARS)
test_set = set(FINAL_TEST_YEARS)


assert dev_set.isdisjoint(test_set)

for fold_name, fold in TEMPORAL_FOLDS.items():

    train_set = set(fold["train_years"])
    val_set = set(fold["validation_years"])

    assert train_set.isdisjoint(val_set)

    assert train_set.issubset(dev_set)
    assert val_set.issubset(dev_set)

    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


# ============================================================
# VERIFY EACH CLIMATE-YEAR
# ============================================================

annual = pd.read_csv(ANNUAL_FILE)

annual["Year"] = annual["Year"].astype(int)


availability_rows = []

for climate in CLIMATES:

    for year in sorted(required_years):

        matches = annual[
            (annual["Site"] == climate)
            &
            (annual["Year"] == year)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one row for "
                f"{climate}, {year}; found {len(matches)}."
            )

        complete = bool(
            matches.iloc[0]["CompleteYear"]
        )

        if not complete:
            raise RuntimeError(
                f"Incomplete climate-year detected: "
                f"{climate} {year}"
            )

        if year in FINAL_TEST_YEARS:
            partition = "FINAL_TEST"
        else:
            partition = "DEVELOPMENT"

        availability_rows.append(
            {
                "Climate": climate,
                "Year": year,
                "Partition": partition,
                "Complete": True,
            }
        )


availability_df = pd.DataFrame(
    availability_rows
)


# ============================================================
# BUILD FOLD MANIFEST
# ============================================================

fold_rows = []

for fold_name, fold in TEMPORAL_FOLDS.items():

    for climate in CLIMATES:

        for year in fold["train_years"]:

            fold_rows.append(
                {
                    "Fold": fold_name,
                    "Climate": climate,
                    "Year": year,
                    "Role": "TRAIN",
                }
            )

        for year in fold["validation_years"]:

            fold_rows.append(
                {
                    "Fold": fold_name,
                    "Climate": climate,
                    "Year": year,
                    "Role": "VALIDATION",
                }
            )


fold_manifest = pd.DataFrame(fold_rows)


# ============================================================
# FROZEN STATISTICAL PRINCIPLES
# ============================================================

STATISTICAL_PROTOCOL = {

    "primary_experimental_unit":
        "climate-year",

    "ppo_seed_handling":
        (
            "Final PPO policies will be averaged within "
            "each climate-year before controller-level "
            "inferential testing."
        ),

    "seed_pseudoreplication":
        False,

    "final_test_selection":
        (
            "No model, PPO seed, reward, hyperparameter, "
            "training horizon, controller parameter, or "
            "analysis decision may be selected using "
            "2019-2025 results."
        ),

    "climate_reporting":
        (
            "Results will be reported both separately by "
            "climate and across the complete multi-climate "
            "test panel."
        ),
}


# ============================================================
# PROTOCOL OBJECT
# ============================================================

protocol = {

    "study":
        "Multi-climate shared irrigation-water allocation",

    "protocol_status":
        "FROZEN_BEFORE_CONTROLLER_EVALUATION",

    "climates": CLIMATES,

    "complete_weather_period":
        [1984, 2025],

    "development_years":
        DEVELOPMENT_YEARS,

    "final_test_years":
        FINAL_TEST_YEARS,

    "temporal_validation_folds":
        TEMPORAL_FOLDS,

    "agronomic_design":
        AGRONOMIC_DESIGN,

    "controllers":
        CONTROLLERS,

    "scarcity_levels":
        SCARCITY_LEVELS,

    "ppo_primary_training_scarcity":
        PPO_PRIMARY_TRAINING_SCARCITY,

    "statistical_protocol":
        STATISTICAL_PROTOCOL,

    "integrity_rules": [
        "2019-2025 must not be used during PPO training.",
        "2019-2025 must not be used during hyperparameter tuning.",
        "2019-2025 must not be used during reward tuning.",
        "2019-2025 must not be used for training-horizon selection.",
        "2019-2025 must not be used to tune Equal.",
        "2019-2025 must not be used to tune Priority.",
        "2019-2025 must not be used for seed selection.",
        "All final PPO seeds must be retained.",
        "Controller configuration must be frozen before final test.",
    ],
}


# ============================================================
# SAVE
# ============================================================

protocol_file = (
    PROTOCOL_DIR
    / "multiclimate_protocol_frozen.json"
)

with open(
    protocol_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        protocol,
        f,
        indent=2,
    )


availability_file = (
    PROTOCOL_DIR
    / "climate_year_partition.csv"
)

availability_df.to_csv(
    availability_file,
    index=False,
)


fold_file = (
    PROTOCOL_DIR
    / "temporal_fold_manifest.csv"
)

fold_manifest.to_csv(
    fold_file,
    index=False,
)


# ============================================================
# TERMINAL REPORT
# ============================================================

print()
print("=" * 78)
print("EXPERIMENTAL PROTOCOL FREEZE")
print("MULTI-CLIMATE EXPERIMENTAL PROTOCOL FROZEN")
print("=" * 78)

print()

print("CLIMATES")
for climate, info in CLIMATES.items():

    print(
        f"  {climate:<8} : "
        f"{info['role']}"
    )


print()
print("TEMPORAL PARTITION")

print(
    f"  Development : "
    f"{DEVELOPMENT_YEARS[0]}-"
    f"{DEVELOPMENT_YEARS[-1]} "
    f"({len(DEVELOPMENT_YEARS)} years)"
)

print(
    f"  Final test  : "
    f"{FINAL_TEST_YEARS[0]}-"
    f"{FINAL_TEST_YEARS[-1]} "
    f"({len(FINAL_TEST_YEARS)} years)"
)


print()
print("TEMPORAL VALIDATION FOLDS")

for fold_name, fold in TEMPORAL_FOLDS.items():

    train = fold["train_years"]
    val = fold["validation_years"]

    print(
        f"  {fold_name}: "
        f"train {train[0]}-{train[-1]} "
        f"({len(train)} y), "
        f"validate {val[0]}-{val[-1]} "
        f"({len(val)} y)"
    )


print()
print("MULTI-CLIMATE SAMPLE COUNTS")

print(
    "  Development climate-years :",
    len(DEVELOPMENT_YEARS) * len(CLIMATES),
)

print(
    "  Final-test climate-years  :",
    len(FINAL_TEST_YEARS) * len(CLIMATES),
)

for fold_name, fold in TEMPORAL_FOLDS.items():

    n_val = (
        len(fold["validation_years"])
        * len(CLIMATES)
    )

    print(
        f"  {fold_name} validation cases  : {n_val}"
    )


print()
print("SCARCITY LEVELS")

for scarcity in SCARCITY_LEVELS:
    print(f"  {scarcity * 100:.0f}%")


print()
print("FINAL TEST INTEGRITY")

print("  2019-2025 controller results used : NO")
print("  PPO training performed            : NO")
print("  Equal evaluated                   : NO")
print("  Priority evaluated                : NO")
print("  Final test remains unopened       : YES")


print()
print("Saved:")
print(f"  {protocol_file}")
print(f"  {availability_file}")
print(f"  {fold_file}")


print()
print("=" * 78)
print("EXPERIMENTAL PROTOCOL FREEZE COMPLETE")
print("=" * 78)

print()
print(
    "NEXT: modify the irrigation environment for "
    "multi-climate training while keeping 2019-2025 "
    "hard-protected."
)