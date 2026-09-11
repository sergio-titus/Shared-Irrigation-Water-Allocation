# ============================================================
# DEVELOPMENT REFERENCE GENERATION
# MULTI-CLIMATE DEVELOPMENT REFERENCE LIBRARY
#
# PURPOSE
# -------
# Generate fully irrigated AquaCrop reference yield and
# irrigation for every DEVELOPMENT climate-year combination:
#
#     3 climates x 35 years = 105 climate-years
#     4 fields each         = 420 field simulations
#
# FINAL TEST 2019-2025 IS HARD-PROTECTED.
#
# NO Equal evaluation.
# NO Priority evaluation.
# NO PPO training.
# NO final-test reference generation.
# ============================================================

from pathlib import Path
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


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

WEATHER_DIR = (
    ROOT / "weather" / "processed"
)

PROTOCOL_FILE = (
    ROOT
    / "protocol"
    / "multiclimate_protocol_frozen.json"
)

REFERENCE_DIR = (
    ROOT / "reference"
)

REFERENCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. LOAD FROZEN PROTOCOL
# ============================================================

if not PROTOCOL_FILE.exists():
    raise FileNotFoundError(
        f"Missing frozen protocol: {PROTOCOL_FILE}"
    )


with open(
    PROTOCOL_FILE,
    "r",
    encoding="utf-8",
) as f:
    protocol = json.load(f)


DEVELOPMENT_YEARS = [
    int(y)
    for y in protocol["development_years"]
]

FINAL_TEST_YEARS = [
    int(y)
    for y in protocol["final_test_years"]
]


CLIMATES = list(
    protocol["climates"].keys()
)


# ============================================================
# 3. HARD TEST PROTECTION
# ============================================================

if set(DEVELOPMENT_YEARS) & set(FINAL_TEST_YEARS):
    raise RuntimeError(
        "Development/final-test overlap detected."
    )


if any(
    year >= 2019
    for year in DEVELOPMENT_YEARS
):
    raise RuntimeError(
        "Protected final-test year entered development set."
    )


assert DEVELOPMENT_YEARS == list(
    range(1984, 2019)
)

assert FINAL_TEST_YEARS == list(
    range(2019, 2026)
)


# ============================================================
# 4. FIELD CONFIGURATION
# ============================================================

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


# Preserve the original agronomic design.
SOIL_TYPE = "SandyLoam"
CROP_NAME = "Maize"

MAX_FIELD_IRRIGATION = 25.0


# ============================================================
# 5. WEATHER FILES
# ============================================================

WEATHER_FILES = {
    "Tunis":
        WEATHER_DIR
        / "tunis_aquacrop_weather.csv",

    "Niamey":
        WEATHER_DIR
        / "niamey_aquacrop_weather.csv",

    "Cotonou":
        WEATHER_DIR
        / "cotonou_aquacrop_weather.csv",
}


for climate, path in WEATHER_FILES.items():

    if not path.exists():

        raise FileNotFoundError(
            f"Missing weather file for "
            f"{climate}: {path}"
        )


# ============================================================
# 6. LOAD WEATHER
# ============================================================

def load_weather(path):

    df = pd.read_csv(path)

    required = [
        "MinTemp",
        "MaxTemp",
        "Precipitation",
        "ReferenceET",
        "Date",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing weather columns "
            f"in {path}: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    return df


WEATHER_BY_CLIMATE = {
    climate: load_weather(path)
    for climate, path
    in WEATHER_FILES.items()
}


# ============================================================
# 7. VERIFY NO TEST WEATHER WILL BE USED
# ============================================================

for climate, df in WEATHER_BY_CLIMATE.items():

    available_years = set(
        df["Date"].dt.year.astype(int)
    )

    missing_dev = sorted(
        set(DEVELOPMENT_YEARS)
        - available_years
    )

    if missing_dev:
        raise RuntimeError(
            f"{climate}: missing development "
            f"years {missing_dev}"
        )


# ============================================================
# 8. CREATE FULLY IRRIGATED REFERENCE FIELD
# ============================================================

def create_reference_model(
    climate,
    year,
    planting_date,
):

    year = int(year)

    # --------------------------------------------------------
    # HARD FINAL-TEST PROTECTION
    # --------------------------------------------------------

    if year in FINAL_TEST_YEARS:
        raise RuntimeError(
            f"FINAL TEST ACCESS BLOCKED: "
            f"{climate} {year}"
        )

    if year >= 2019:
        raise RuntimeError(
            f"Protected year requested: "
            f"{climate} {year}"
        )

    if year not in DEVELOPMENT_YEARS:
        raise RuntimeError(
            f"Year {year} is outside "
            f"the frozen development period."
        )

    weather_all = (
        WEATHER_BY_CLIMATE[
            climate
        ]
    )

    # AquaCrop receives the weather record;
    # the simulation dates below select the target season.
    weather = weather_all.copy()

    soil = Soil(
        soil_type=SOIL_TYPE
    )

    crop = Crop(
        CROP_NAME,
        planting_date=planting_date,
    )

    initial_water = (
        InitialWaterContent(
            value=["FC"]
        )
    )

    # --------------------------------------------------------
    # FULL-IRRIGATION REFERENCE
    #
    # Method 1 = soil-moisture-target irrigation.
    #
    # SMT=100% maintains the crop close to field capacity
    # and defines the unconstrained reference condition.
    #
    # The same 25-mm/day field maximum is retained.
    # --------------------------------------------------------

    irrigation = (
        IrrigationManagement(
            irrigation_method=1,
            SMT=[
                100,
                100,
                100,
                100,
            ],
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
        initial_water_content=(
            initial_water
        ),
        irrigation_management=(
            irrigation
        ),
    )

    return model


# ============================================================
# 9. RUN ONE REFERENCE FIELD
# ============================================================

def run_reference_field(
    climate,
    year,
    field_name,
    planting_date,
):

    model = create_reference_model(
        climate=climate,
        year=year,
        planting_date=planting_date,
    )

    model.run_model(
        till_termination=True
    )

    result = (
        model.get_simulation_results()
    )

    if (
        result is False
        or len(result) == 0
    ):
        raise RuntimeError(
            f"No AquaCrop final result for "
            f"{climate} {year} {field_name}"
        )

    row = result.iloc[0]

    dry_yield = float(
        row[
            "Dry yield (tonne/ha)"
        ]
    )

    irrigation = float(
        row[
            "Seasonal irrigation (mm)"
        ]
    )

    if (
        not np.isfinite(dry_yield)
        or dry_yield <= 0
    ):
        raise RuntimeError(
            f"Invalid yield: "
            f"{climate} {year} "
            f"{field_name} = {dry_yield}"
        )

    if (
        not np.isfinite(irrigation)
        or irrigation < 0
    ):
        raise RuntimeError(
            f"Invalid irrigation: "
            f"{climate} {year} "
            f"{field_name} = {irrigation}"
        )

    harvest_date = ""

    if (
        "Harvest Date (YYYY/MM/DD)"
        in result.columns
    ):
        harvest_date = str(
            row[
                "Harvest Date (YYYY/MM/DD)"
            ]
        )

    potential_yield = np.nan

    if (
        "Yield potential (tonne/ha)"
        in result.columns
    ):
        potential_yield = float(
            row[
                "Yield potential (tonne/ha)"
            ]
        )

    return {
        "Climate": climate,
        "Year": int(year),
        "Field": field_name,
        "Planting_date":
            planting_date,
        "Reference_yield_t_ha":
            dry_yield,
        "Reference_irrigation_mm":
            irrigation,
        "Potential_yield_t_ha":
            potential_yield,
        "Harvest_date":
            harvest_date,
    }


# ============================================================
# 10. MAIN SIMULATION
# ============================================================

def main():

    print()
    print("#" * 78)
    print("DEVELOPMENT REFERENCE GENERATION")
    print(
        "MULTI-CLIMATE DEVELOPMENT "
        "REFERENCE LIBRARY"
    )
    print("#" * 78)

    print()
    print(
        f"Development years : "
        f"{DEVELOPMENT_YEARS[0]}-"
        f"{DEVELOPMENT_YEARS[-1]}"
    )

    print(
        f"Protected test    : "
        f"{FINAL_TEST_YEARS[0]}-"
        f"{FINAL_TEST_YEARS[-1]}"
    )

    print(
        f"Climates          : "
        f"{CLIMATES}"
    )

    print(
        f"Climate-years     : "
        f"{len(CLIMATES) * len(DEVELOPMENT_YEARS)}"
    )

    print(
        f"Field simulations : "
        f"{len(CLIMATES) * len(DEVELOPMENT_YEARS) * len(FIELD_CONFIGS)}"
    )

    start_time = time.perf_counter()

    field_rows = []

    # ========================================================
    # RUN ONLY DEVELOPMENT CLIMATE-YEARS
    # ========================================================

    for climate in CLIMATES:

        print()
        print("=" * 78)
        print(
            f"CLIMATE: {climate}"
        )
        print("=" * 78)

        for year in DEVELOPMENT_YEARS:

            year_results = []

            for cfg in FIELD_CONFIGS:

                r = run_reference_field(
                    climate=climate,
                    year=year,
                    field_name=(
                        cfg["field"]
                    ),
                    planting_date=(
                        cfg[
                            "planting_date"
                        ]
                    ),
                )

                field_rows.append(r)
                year_results.append(r)

            total_yield = sum(
                x[
                    "Reference_yield_t_ha"
                ]
                for x in year_results
            )

            total_irrigation = sum(
                x[
                    "Reference_irrigation_mm"
                ]
                for x in year_results
            )

            print(
                f"{climate:<8} "
                f"{year}: "
                f"yield={total_yield:7.3f} t/ha | "
                f"irrigation={total_irrigation:8.2f} mm"
            )

    # ========================================================
    # FIELD TABLE
    # ========================================================

    field_df = pd.DataFrame(
        field_rows
    )

    expected_field_rows = (
        len(CLIMATES)
        *
        len(DEVELOPMENT_YEARS)
        *
        len(FIELD_CONFIGS)
    )

    if len(field_df) != expected_field_rows:

        raise RuntimeError(
            f"Expected {expected_field_rows} "
            f"field rows, got {len(field_df)}."
        )

    # Exactly four fields per climate-year.
    field_counts = (
        field_df
        .groupby(
            [
                "Climate",
                "Year",
            ]
        )
        .size()
    )

    if not (
        field_counts == 4
    ).all():

        raise RuntimeError(
            "Each development climate-year "
            "must contain exactly four fields."
        )


    # ========================================================
    # YEAR-LEVEL REFERENCE TABLE
    # ========================================================

    year_rows = []

    for (
        climate,
        year
    ), group in field_df.groupby(
        [
            "Climate",
            "Year",
        ],
        sort=True,
    ):

        row = {
            "Climate":
                climate,

            "Year":
                int(year),

            "Split":
                "Development",

            "Total reference yield (tonne/ha)":
                float(
                    group[
                        "Reference_yield_t_ha"
                    ].sum()
                ),

            "Total reference irrigation (mm)":
                float(
                    group[
                        "Reference_irrigation_mm"
                    ].sum()
                ),
        }

        total_irr = row[
            "Total reference irrigation (mm)"
        ]

        row[
            "Budget 100% (mm)"
        ] = total_irr

        row[
            "Budget 60% (mm)"
        ] = (
            0.60 * total_irr
        )

        row[
            "Budget 40% (mm)"
        ] = (
            0.40 * total_irr
        )

        for field_name in FIELD_NAMES:

            frow = group[
                group["Field"]
                == field_name
            ]

            if len(frow) != 1:
                raise RuntimeError(
                    f"Expected one row for "
                    f"{climate} {year} "
                    f"{field_name}"
                )

            frow = frow.iloc[0]

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

        year_rows.append(row)


    reference_df = pd.DataFrame(
        year_rows
    )


    # ========================================================
    # STRICT INTEGRITY CHECKS
    # ========================================================

    expected_year_rows = (
        len(CLIMATES)
        *
        len(DEVELOPMENT_YEARS)
    )

    if len(reference_df) != expected_year_rows:

        raise RuntimeError(
            f"Expected {expected_year_rows} "
            f"reference rows, "
            f"got {len(reference_df)}."
        )


    # Every climate-year key must be unique.
    duplicate_keys = (
        reference_df
        .duplicated(
            subset=[
                "Climate",
                "Year",
            ]
        )
    )

    if duplicate_keys.any():

        raise RuntimeError(
            "Duplicate climate-year "
            "reference detected."
        )


    # Absolutely no test years.
    if reference_df[
        "Year"
    ].isin(
        FINAL_TEST_YEARS
    ).any():

        raise RuntimeError(
            "FINAL TEST LEAKAGE in "
            "reference library."
        )


    if (
        reference_df["Year"]
        >= 2019
    ).any():

        raise RuntimeError(
            "Protected year found "
            "in development reference."
        )


    # ========================================================
    # SAVE
    # ========================================================

    field_file = (
        REFERENCE_DIR
        / "development_reference_fields.csv"
    )

    year_file = (
        REFERENCE_DIR
        / "development_reference_by_climate_year.csv"
    )


    field_df.to_csv(
        field_file,
        index=False,
    )

    reference_df.to_csv(
        year_file,
        index=False,
    )


    # ========================================================
    # CLIMATE SUMMARY
    # ========================================================

    summary = (
        reference_df
        .groupby(
            "Climate",
            as_index=False,
        )
        .agg(
            Years=(
                "Year",
                "count",
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


    summary_file = (
        REFERENCE_DIR
        / "development_reference_climate_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )


    # ========================================================
    # SAVE INTEGRITY RECORD
    # ========================================================

    integrity = {
        "milestone":
            "53C",

        "reference_scope":
            "development_only",

        "climates":
            CLIMATES,

        "development_years":
            DEVELOPMENT_YEARS,

        "protected_final_test_years":
            FINAL_TEST_YEARS,

        "development_climate_years":
            int(
                len(reference_df)
            ),

        "field_simulations":
            int(
                len(field_df)
            ),

        "final_test_reference_generated":
            False,

        "equal_evaluated":
            False,

        "priority_evaluated":
            False,

        "ppo_training_performed":
            False,

        "final_test_opened":
            False,
    }


    integrity_file = (
        REFERENCE_DIR
        / "development_reference_integrity.json"
    )

    with open(
        integrity_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            integrity,
            f,
            indent=2,
        )


    runtime = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "DEVELOPMENT REFERENCE SUMMARY"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.3f}"
            ),
        )
    )


    print()
    print("=" * 78)
    print(
        "REFERENCE LIBRARY INTEGRITY"
    )
    print("=" * 78)

    print(
        f"Expected climate-year rows : "
        f"{expected_year_rows}"
    )

    print(
        f"Actual climate-year rows   : "
        f"{len(reference_df)}"
    )

    print(
        f"Expected field rows        : "
        f"{expected_field_rows}"
    )

    print(
        f"Actual field rows          : "
        f"{len(field_df)}"
    )

    print(
        "Unique climate-year keys   : PASS"
    )

    print(
        "2019-2025 reference used   : NO"
    )

    print(
        "Equal evaluated            : NO"
    )

    print(
        "Priority evaluated         : NO"
    )

    print(
        "PPO trained                : NO"
    )

    print(
        "Final test remains unopened: YES"
    )


    print()
    print(
        f"Runtime: "
        f"{runtime:.2f} s "
        f"({runtime / 60:.2f} min)"
    )


    print()
    print("Saved:")
    print(f"  {field_file}")
    print(f"  {year_file}")
    print(f"  {summary_file}")
    print(f"  {integrity_file}")


    print()
    print("=" * 78)
    print(
        "DEVELOPMENT REFERENCE GENERATION COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "NEXT: build and smoke-test the "
        "multi-climate PPO environment using "
        "development climate-years only."
    )


if __name__ == "__main__":
    main()