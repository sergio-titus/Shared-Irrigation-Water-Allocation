# ============================================================
# FINAL-TEST REFERENCE GENERATION
# FINAL-TEST REFERENCE LIBRARY (2019-2025)
#
# PURPOSE
# -------
# Generate the fully irrigated reference yields and irrigation
# totals required to evaluate Equal, Priority, and the frozen
# 10-seed PPO ensemble on the common 2019-2025 final test.
#
# IMPORTANT
# ---------
# - All 10 final PPO models MUST already be frozen.
# - No PPO training occurs here.
# - No PPO inference occurs here.
# - Equal and Priority are NOT evaluated here.
# - No controller performance is inspected.
# - The reference definition is exactly the same as Milestone 53C:
#       Maize
#       SandyLoam
#       Initial water = FC
#       simulation = May 1 to Oct 31
#       irrigation_method = 1
#       SMT = [100,100,100,100]
#       MaxIrr = 25 mm
#       MaxIrrSeason = 10000 mm
#
# Scientifically, 2019-2025 weather is accessed in this milestone
# only to construct deterministic reference denominators AFTER all
# final PPO models have been frozen.
# ============================================================

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from aquacrop import (
    AquaCropModel,
    Soil,
    Crop,
    InitialWaterContent,
    IrrigationManagement,
)

import multiclimate_optimization_core as core


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

FINAL_TRAINING_DIR = (
    ROOT
    / "final_training"
)

FINAL_TRAINING_SUMMARY = (
    FINAL_TRAINING_DIR
    / "ppo_final_training_final_training_summary.csv"
)

FINAL_TRAINING_INTEGRITY = (
    FINAL_TRAINING_DIR
    / "ppo_final_training_final_training_integrity.json"
)

FINAL_MODELS_DIR = (
    FINAL_TRAINING_DIR
    / "models"
)

OUTPUT_DIR = (
    ROOT
    / "final_test_reference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIELD_FILE = (
    OUTPUT_DIR
    / "final_test_reference_fields.csv"
)

CLIMATE_YEAR_FILE = (
    OUTPUT_DIR
    / "final_test_reference_by_climate_year.csv"
)

CLIMATE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_test_reference_climate_summary.csv"
)

INTEGRITY_FILE = (
    OUTPUT_DIR
    / "final_test_reference_final_test_reference_integrity.json"
)


# ============================================================
# 2. FROZEN DESIGN
# ============================================================

CLIMATES = [
    "Tunis",
    "Niamey",
    "Cotonou",
]

FINAL_TEST_YEARS = list(
    range(
        2019,
        2026,
    )
)

FIELD_CONFIGS = [
    {
        "field": "F1",
        "planting_date": "05/01",
    },
    {
        "field": "F2",
        "planting_date": "05/08",
    },
    {
        "field": "F3",
        "planting_date": "05/15",
    },
    {
        "field": "F4",
        "planting_date": "05/22",
    },
]

FIELD_NAMES = [
    cfg["field"]
    for cfg in FIELD_CONFIGS
]

MAX_FIELD_IRRIGATION = 25.0

EXPECTED_FINAL_SEEDS = [
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

EXPECTED_SELECTED_HORIZON = 102400


# ============================================================
# 3. HELPERS
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

            digest.update(
                chunk
            )

    return digest.hexdigest()


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


# ============================================================
# 4. VERIFY THAT FINAL PPO MODELS ARE FROZEN FIRST
# ============================================================

if not FINAL_TRAINING_SUMMARY.exists():

    raise FileNotFoundError(
        f"Missing final PPO training summary: "
        f"{FINAL_TRAINING_SUMMARY}"
    )


training_df = pd.read_csv(
    FINAL_TRAINING_SUMMARY
)


required_training_columns = {
    "Seed",
    "Selected_horizon",
    "Actual_timesteps",
    "Final_test_used",
}


missing_training_columns = (
    required_training_columns
    - set(
        training_df.columns
    )
)


if missing_training_columns:

    raise RuntimeError(
        "53G summary missing columns: "
        f"{sorted(missing_training_columns)}"
    )


training_df[
    "Seed"
] = training_df[
    "Seed"
].astype(int)


if len(
    training_df
) != 10:

    raise RuntimeError(
        "Expected 10 frozen final PPO models."
    )


if set(
    training_df[
        "Seed"
    ].tolist()
) != set(
    EXPECTED_FINAL_SEEDS
):

    raise RuntimeError(
        "Final PPO seed set differs from "
        "the frozen 10-seed specification."
    )


if not (
    training_df[
        "Selected_horizon"
    ].astype(int)
    == EXPECTED_SELECTED_HORIZON
).all():

    raise RuntimeError(
        "One or more final models use "
        "the wrong selected horizon."
    )


if not (
    training_df[
        "Actual_timesteps"
    ].astype(int)
    == EXPECTED_SELECTED_HORIZON
).all():

    raise RuntimeError(
        "One or more final PPO models have "
        "the wrong actual timestep count."
    )


if (
    training_df[
        "Final_test_used"
    ].astype(bool)
    .any()
):

    raise RuntimeError(
        "53G reports final-test use."
    )


training_integrity = load_json(
    FINAL_TRAINING_INTEGRITY
)


if (
    training_integrity.get(
        "final_test_used",
        None,
    )
    is not False
):

    raise RuntimeError(
        "53G integrity file does not certify "
        "that final-test data were unused."
    )


if (
    training_integrity.get(
        "all_final_seeds_retained",
        None,
    )
    is not True
):

    raise RuntimeError(
        "53G integrity file does not certify "
        "retention of all final PPO seeds."
    )


missing_models = []


for seed in EXPECTED_FINAL_SEEDS:

    model_file = (
        FINAL_MODELS_DIR
        / f"multiclimate_ppo_seed_{seed}.zip"
    )

    if not model_file.exists():

        missing_models.append(
            str(
                model_file
            )
        )


if missing_models:

    raise RuntimeError(
        "Missing final PPO model files:\n"
        + "\n".join(
            missing_models
        )
    )


# ============================================================
# 5. VERIFY FINAL-TEST WEATHER AVAILABILITY
#
# The core weather CSVs contain 1984-2025 data.
# We use only 2019-2025 here.
# ============================================================

WEATHER_BY_CLIMATE = (
    core.WEATHER_BY_CLIMATE
)


for climate in CLIMATES:

    if climate not in WEATHER_BY_CLIMATE:

        raise RuntimeError(
            f"Missing weather for climate: "
            f"{climate}"
        )

    weather_df = (
        WEATHER_BY_CLIMATE[
            climate
        ]
        .copy()
    )

    available_years = set(
        weather_df[
            "Date"
        ]
        .dt.year
        .astype(int)
        .tolist()
    )

    missing_years = sorted(
        set(
            FINAL_TEST_YEARS
        )
        - available_years
    )

    if missing_years:

        raise RuntimeError(
            f"{climate}: missing final-test "
            f"weather years {missing_years}"
        )


# ============================================================
# 6. CREATE FULLY IRRIGATED REFERENCE FIELD
# ============================================================

def create_reference_field(
    climate,
    year,
    planting_date,
):

    climate = str(
        climate
    )

    year = int(
        year
    )


    if climate not in CLIMATES:

        raise RuntimeError(
            f"Unknown climate: "
            f"{climate}"
        )


    if year not in FINAL_TEST_YEARS:

        raise RuntimeError(
            f"53H is restricted to "
            f"2019-2025; received {year}."
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
        planting_date=(
            planting_date
        ),
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
            irrigation_method=1,
            SMT=[
                100,
                100,
                100,
                100,
            ],
            MaxIrr=(
                MAX_FIELD_IRRIGATION
            ),
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
        weather_df=(
            weather
        ),
        soil=(
            soil
        ),
        crop=(
            crop
        ),
        initial_water_content=(
            initial_water
        ),
        irrigation_management=(
            irrigation
        ),
    )


    return model


# ============================================================
# 7. RUN ONE REFERENCE FIELD
# ============================================================

def run_reference_field(
    climate,
    year,
    field_name,
    planting_date,
):

    model = create_reference_field(
        climate=climate,
        year=year,
        planting_date=planting_date,
    )


    model.run_model(
        till_termination=True,
    )


    results = (
        model.get_simulation_results()
    )


    if (
        results is False
        or results is None
        or len(
            results
        ) == 0
    ):

        raise RuntimeError(
            f"No AquaCrop result for "
            f"{climate} {year} {field_name}."
        )


    crop_yield = float(
        results.iloc[0][
            "Dry yield (tonne/ha)"
        ]
    )


    irrigation = float(
        results.iloc[0][
            "Seasonal irrigation (mm)"
        ]
    )


    if not np.isfinite(
        crop_yield
    ):

        raise RuntimeError(
            f"Non-finite reference yield: "
            f"{climate} {year} {field_name}"
        )


    if not np.isfinite(
        irrigation
    ):

        raise RuntimeError(
            f"Non-finite reference irrigation: "
            f"{climate} {year} {field_name}"
        )


    if crop_yield < 0.0:

        raise RuntimeError(
            f"Negative reference yield: "
            f"{climate} {year} {field_name}"
        )


    if irrigation < -1e-9:

        raise RuntimeError(
            f"Negative reference irrigation: "
            f"{climate} {year} {field_name}"
        )


    return {
        "Climate":
            climate,

        "Year":
            int(
                year
            ),

        "Field":
            field_name,

        "Planting_date":
            planting_date,

        "Reference_yield_t_ha":
            crop_yield,

        "Reference_irrigation_mm":
            irrigation,
    }


# ============================================================
# 8. MAIN
# ============================================================

def main():

    start = (
        time.perf_counter()
    )


    print()
    print("#" * 78)
    print("FINAL-TEST REFERENCE GENERATION")
    print(
        "FINAL-TEST REFERENCE LIBRARY "
        "(2019-2025)"
    )
    print("#" * 78)


    print()
    print(
        "All 10 final PPO models frozen : PASS"
    )

    print(
        "Selected PPO horizon           : "
        f"{EXPECTED_SELECTED_HORIZON:,}"
    )

    print(
        "Climates                       : "
        f"{CLIMATES}"
    )

    print(
        "Final-test years               : "
        f"{FINAL_TEST_YEARS}"
    )

    print(
        "Controller evaluation          : NO"
    )

    print(
        "PPO inference                  : NO"
    )


    rows = []


    total_expected = (
        len(
            CLIMATES
        )
        * len(
            FINAL_TEST_YEARS
        )
        * len(
            FIELD_CONFIGS
        )
    )


    counter = 0


    for climate in CLIMATES:

        print()
        print("=" * 78)

        print(
            f"CLIMATE: "
            f"{climate}"
        )

        print("=" * 78)


        for year in FINAL_TEST_YEARS:

            year_yield = 0.0
            year_irrigation = 0.0


            for cfg in FIELD_CONFIGS:

                counter += 1


                result = (
                    run_reference_field(
                        climate=climate,
                        year=year,
                        field_name=(
                            cfg[
                                "field"
                            ]
                        ),
                        planting_date=(
                            cfg[
                                "planting_date"
                            ]
                        ),
                    )
                )


                rows.append(
                    result
                )


                year_yield += (
                    result[
                        "Reference_yield_t_ha"
                    ]
                )


                year_irrigation += (
                    result[
                        "Reference_irrigation_mm"
                    ]
                )


            print(
                f"{climate:<8} {year}: "
                f"yield={year_yield:8.3f} t/ha | "
                f"irrigation={year_irrigation:8.3f} mm"
            )


    field_df = pd.DataFrame(
        rows
    )


    if len(
        field_df
    ) != total_expected:

        raise RuntimeError(
            f"Expected {total_expected} field rows, "
            f"got {len(field_df)}."
        )


    if field_df.duplicated(
        subset=[
            "Climate",
            "Year",
            "Field",
        ]
    ).any():

        raise RuntimeError(
            "Duplicate Climate-Year-Field reference key."
        )


    # ========================================================
    # 9. BUILD ONE ROW PER CLIMATE-YEAR
    # ========================================================

    climate_year_rows = []


    for climate in CLIMATES:

        for year in FINAL_TEST_YEARS:

            d = (
                field_df[
                    (
                        field_df[
                            "Climate"
                        ]
                        == climate
                    )
                    &
                    (
                        field_df[
                            "Year"
                        ]
                        == year
                    )
                ]
                .copy()
            )


            if len(
                d
            ) != 4:

                raise RuntimeError(
                    f"Expected four fields for "
                    f"{climate} {year}."
                )


            row = {
                "Climate":
                    climate,

                "Year":
                    int(
                        year
                    ),
            }


            for field_name in FIELD_NAMES:

                frow = (
                    d[
                        d[
                            "Field"
                        ]
                        == field_name
                    ]
                )


                if len(
                    frow
                ) != 1:

                    raise RuntimeError(
                        f"Expected one row for "
                        f"{climate} {year} "
                        f"{field_name}."
                    )


                frow = (
                    frow.iloc[0]
                )


                row[
                    f"{field_name}_reference_yield"
                ] = float(
                    frow[
                        "Reference_yield_t_ha"
                    ]
                )


                row[
                    f"{field_name}_reference_irrigation"
                ] = float(
                    frow[
                        "Reference_irrigation_mm"
                    ]
                )


            row[
                "Total reference yield (tonne/ha)"
            ] = float(
                sum(
                    row[
                        f"{field_name}_reference_yield"
                    ]
                    for field_name
                    in FIELD_NAMES
                )
            )


            row[
                "Total reference irrigation (mm)"
            ] = float(
                sum(
                    row[
                        f"{field_name}_reference_irrigation"
                    ]
                    for field_name
                    in FIELD_NAMES
                )
            )


            climate_year_rows.append(
                row
            )


    climate_year_df = pd.DataFrame(
        climate_year_rows
    )


    expected_climate_years = (
        len(
            CLIMATES
        )
        * len(
            FINAL_TEST_YEARS
        )
    )


    if len(
        climate_year_df
    ) != expected_climate_years:

        raise RuntimeError(
            f"Expected {expected_climate_years} "
            f"climate-year rows, "
            f"got {len(climate_year_df)}."
        )


    if climate_year_df.duplicated(
        subset=[
            "Climate",
            "Year",
        ]
    ).any():

        raise RuntimeError(
            "Duplicate Climate-Year key."
        )


    # ========================================================
    # 10. CLIMATE SUMMARY
    # ========================================================

    climate_summary = (
        climate_year_df
        .groupby(
            "Climate",
            as_index=False,
        )
        .agg(
            Years=(
                "Year",
                "nunique",
            ),

            Mean_reference_yield_t_ha=(
                "Total reference yield (tonne/ha)",
                "mean",
            ),

            SD_reference_yield_t_ha=(
                "Total reference yield (tonne/ha)",
                "std",
            ),

            Mean_reference_irrigation_mm=(
                "Total reference irrigation (mm)",
                "mean",
            ),

            SD_reference_irrigation_mm=(
                "Total reference irrigation (mm)",
                "std",
            ),

            Min_reference_irrigation_mm=(
                "Total reference irrigation (mm)",
                "min",
            ),

            Max_reference_irrigation_mm=(
                "Total reference irrigation (mm)",
                "max",
            ),
        )
    )


    # ========================================================
    # 11. SAVE
    # ========================================================

    field_df.to_csv(
        FIELD_FILE,
        index=False,
    )


    climate_year_df.to_csv(
        CLIMATE_YEAR_FILE,
        index=False,
    )


    climate_summary.to_csv(
        CLIMATE_SUMMARY_FILE,
        index=False,
    )


    runtime_seconds = (
        time.perf_counter()
        - start
    )


    integrity = {

        "milestone":
            "53H",

        "status":
            "COMPLETE",

        "reference_definition":
            {
                "crop":
                    "Maize",

                "soil":
                    "SandyLoam",

                "initial_water":
                    "FC",

                "simulation_window":
                    "May 1-October 31",

                "irrigation_method":
                    1,

                "SMT":
                    [
                        100,
                        100,
                        100,
                        100,
                    ],

                "MaxIrr_mm":
                    float(
                        MAX_FIELD_IRRIGATION
                    ),

                "MaxIrrSeason_mm":
                    10000,
            },

        "climates":
            CLIMATES,

        "final_test_years":
            FINAL_TEST_YEARS,

        "climate_year_rows":
            int(
                len(
                    climate_year_df
                )
            ),

        "field_rows":
            int(
                len(
                    field_df
                )
            ),

        "all_final_ppo_models_frozen_before_reference_generation":
            True,

        "number_of_frozen_final_ppo_models":
            10,

        "ppo_training_performed":
            False,

        "ppo_inference_performed":
            False,

        "equal_evaluated":
            False,

        "priority_evaluated":
            False,

        "controller_performance_seen":
            False,

        "final_test_weather_accessed_for_reference_generation":
            True,

        "controller_final_test_opened":
            False,

        "final_training_summary_sha256":
            sha256_file(
                FINAL_TRAINING_SUMMARY
            ),

        "final_training_integrity_sha256":
            sha256_file(
                FINAL_TRAINING_INTEGRITY
            ),

        "reference_by_climate_year_sha256":
            sha256_file(
                CLIMATE_YEAR_FILE
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
    # 12. TERMINAL REPORT
    # ========================================================

    print()
    print("#" * 78)
    print("FINAL-TEST REFERENCE GENERATION COMPLETE")
    print("#" * 78)

    print()
    print(
        "FINAL-TEST REFERENCE SUMMARY"
    )

    print()

    print(
        climate_summary.to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.3f}"
            ),
        )
    )


    print()
    print("=" * 78)
    print("INTEGRITY")
    print("=" * 78)

    print(
        f"Expected climate-year rows : "
        f"{expected_climate_years}"
    )

    print(
        f"Actual climate-year rows   : "
        f"{len(climate_year_df)}"
    )

    print(
        f"Expected field rows        : "
        f"{total_expected}"
    )

    print(
        f"Actual field rows          : "
        f"{len(field_df)}"
    )

    print(
        "Unique climate-year keys   : PASS"
    )

    print(
        "Final PPO models frozen     : YES"
    )

    print(
        "PPO training performed      : NO"
    )

    print(
        "PPO inference performed     : NO"
    )

    print(
        "Equal evaluated             : NO"
    )

    print(
        "Priority evaluated          : NO"
    )

    print(
        "Controller performance seen : NO"
    )

    print(
        "2019-2025 weather accessed  : YES "
        "(reference generation only)"
    )

    print(
        "Controller final test opened: NO"
    )

    print(
        f"Runtime                     : "
        f"{runtime_seconds:.2f} s "
        f"({runtime_seconds / 60.0:.2f} min)"
    )


    print()
    print("Saved:")

    print(
        f"  {FIELD_FILE}"
    )

    print(
        f"  {CLIMATE_YEAR_FILE}"
    )

    print(
        f"  {CLIMATE_SUMMARY_FILE}"
    )

    print(
        f"  {INTEGRITY_FILE}"
    )


    print()
    print(
        "NEXT: freeze the common final-evaluation "
        "manifest, then evaluate Equal, Priority and "
        "all 10 frozen PPO seeds on the same "
        "2019-2025 climate-years and scarcity levels."
    )


if __name__ == "__main__":

    main()
