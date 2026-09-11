from aquacrop import (
    AquaCropModel,
    Soil,
    Crop,
    InitialWaterContent,
    IrrigationManagement,
)
from aquacrop.utils import (
    prepare_weather,
    get_filepath,
)

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

import pandas as pd
import numpy as np

import os
import time
import json
import psutil


from pathlib import Path


# ============================================================
# MULTI-CLIMATE EXTENSION CORE
# ============================================================
# This module is DEVELOPMENT-ONLY.
# Protected final-test years 2019-2025 cannot be instantiated.
# No controller evaluation or PPO training is performed on import.
# ============================================================

# ============================================================
# 1. DEVELOPMENT AND FINAL TEST YEARS
# ============================================================

DEVELOPMENT_YEARS = list(range(1984, 2019))
FINAL_TEST_YEARS = list(range(2019, 2026))

# Compatibility aliases.
TRAIN_YEARS = DEVELOPMENT_YEARS
TEST_YEARS = FINAL_TEST_YEARS


# ============================================================
# 2. TEMPORAL VALIDATION FOLDS
# ============================================================

TEMPORAL_FOLDS = [
    {
        "name": "Fold_1",
        "train_years": list(range(1984, 1998)),
        "validation_years": list(range(1998, 2005)),
    },
    {
        "name": "Fold_2",
        "train_years": list(range(1984, 2005)),
        "validation_years": list(range(2005, 2012)),
    },
    {
        "name": "Fold_3",
        "train_years": list(range(1984, 2012)),
        "validation_years": list(range(2012, 2019)),
    },
]


# ============================================================
# 3. CLIMATES
# ============================================================

CLIMATES = ["Tunis", "Niamey", "Cotonou"]

CLIMATE_TO_ONEHOT = {
    "Tunis": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "Niamey": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "Cotonou": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
}


# ============================================================
# 4. FIELD CONFIGURATION
# ============================================================

FIELD_CONFIGS = [
    {"field": "F1", "planting_date": "05/01"},
    {"field": "F2", "planting_date": "05/08"},
    {"field": "F3", "planting_date": "05/15"},
    {"field": "F4", "planting_date": "05/22"},
]

FIELD_NAMES = [cfg["field"] for cfg in FIELD_CONFIGS]


# ============================================================
# 5. RESOURCE CONSTRAINTS
# ============================================================

SCARCITY_FRACTION = 0.40
DAILY_SYSTEM_CAPACITY = 40.0
MAX_FIELD_IRRIGATION = 25.0
DEPLETION_TRIGGER = 0.40


# ============================================================
# 6. REWARD SETTINGS
# ============================================================
# Defaults are retained from the original configurable core.
# Training scripts may call configure_reward(...) before PPO
# training. Milestone 53D does not train PPO.
# ============================================================

DAILY_STRESS_WEIGHT = 0.10
TERMINAL_YIELD_WEIGHT = 15.0
TERMINAL_JAIN_WEIGHT = 3.0
TERMINAL_WP_WEIGHT = 1.0
TERMINAL_LOSS_WEIGHT = 5.0
TERMINAL_WORST_WEIGHT = 0.0


# ============================================================
# 7. FILE PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

REFERENCE_PATH = (
    ROOT
    / "reference"
    / "development_reference_by_climate_year.csv"
)

WEATHER_FILES = {
    "Tunis": ROOT / "weather" / "processed" / "tunis_aquacrop_weather.csv",
    "Niamey": ROOT / "weather" / "processed" / "niamey_aquacrop_weather.csv",
    "Cotonou": ROOT / "weather" / "processed" / "cotonou_aquacrop_weather.csv",
}


# ============================================================
# 8. LOAD DEVELOPMENT REFERENCE LIBRARY
# ============================================================

if not REFERENCE_PATH.exists():
    raise FileNotFoundError(
        f"Missing multi-climate development reference file: {REFERENCE_PATH}"
    )

reference_df = pd.read_csv(REFERENCE_PATH)

required_reference_columns = {
    "Climate",
    "Year",
    "Total reference yield (tonne/ha)",
    "Total reference irrigation (mm)",
}

for field_name in FIELD_NAMES:
    required_reference_columns.add(f"{field_name}_reference_yield")
    required_reference_columns.add(f"{field_name}_reference_irrigation")

missing_reference_columns = sorted(
    required_reference_columns - set(reference_df.columns)
)

if missing_reference_columns:
    raise RuntimeError(
        "Reference library is missing columns: "
        f"{missing_reference_columns}"
    )

reference_df["Year"] = reference_df["Year"].astype(int)
reference_df["Climate"] = reference_df["Climate"].astype(str)

if reference_df.duplicated(subset=["Climate", "Year"]).any():
    raise RuntimeError("Duplicate Climate-Year key in reference library.")

if reference_df["Year"].isin(FINAL_TEST_YEARS).any():
    raise RuntimeError(
        "FINAL-TEST LEAKAGE: development reference library contains 2019-2025."
    )

available_reference_pairs = set(
    zip(reference_df["Climate"], reference_df["Year"])
)

for climate in CLIMATES:
    for year in DEVELOPMENT_YEARS:
        if (climate, year) not in available_reference_pairs:
            raise RuntimeError(
                f"Missing development reference for {climate} {year}."
            )

print(
    "Multi-climate development reference library loaded: "
    f"{len(reference_df)} climate-years"
)


# ============================================================
# 9. LOAD MULTI-CLIMATE WEATHER
# ============================================================

def _load_weather_csv(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing weather file: {path}")

    df = pd.read_csv(path)

    required = [
        "MinTemp",
        "MaxTemp",
        "Precipitation",
        "ReferenceET",
        "Date",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Weather file {path} is missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


WEATHER_BY_CLIMATE = {
    climate: _load_weather_csv(path)
    for climate, path in WEATHER_FILES.items()
}

for climate, df in WEATHER_BY_CLIMATE.items():
    available_weather_years = set(df["Date"].dt.year.astype(int))
    missing_development = sorted(
        set(DEVELOPMENT_YEARS) - available_weather_years
    )
    if missing_development:
        raise RuntimeError(
            f"{climate}: missing development weather years "
            f"{missing_development}"
        )


# ============================================================
# 10. CLIMATE-YEAR REFERENCE
# ============================================================

def get_climate_year_reference(climate, year):
    climate = str(climate)
    year = int(year)

    if climate not in CLIMATES:
        raise RuntimeError(f"Unknown climate: {climate}")

    if year in FINAL_TEST_YEARS or year >= 2019:
        raise RuntimeError(
            f"FINAL TEST ACCESS BLOCKED: {climate} {year}"
        )

    if year not in DEVELOPMENT_YEARS:
        raise RuntimeError(
            f"Year {year} is outside the frozen development period."
        )

    rows = reference_df[
        (reference_df["Climate"] == climate)
        & (reference_df["Year"] == year)
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one reference row for "
            f"{climate} {year}, found {len(rows)}."
        )

    row = rows.iloc[0]

    reference_yields = {
        field_name: float(row[f"{field_name}_reference_yield"])
        for field_name in FIELD_NAMES
    }

    reference_irrigation = {
        field_name: float(row[f"{field_name}_reference_irrigation"])
        for field_name in FIELD_NAMES
    }

    total_reference_irrigation = float(
        row["Total reference irrigation (mm)"]
    )

    total_reference_yield = float(
        sum(reference_yields.values())
    )

    seasonal_budget = float(
        SCARCITY_FRACTION * total_reference_irrigation
    )

    return {
        "climate": climate,
        "year": year,
        "reference_yields": reference_yields,
        "reference_irrigation": reference_irrigation,
        "total_reference_irrigation": total_reference_irrigation,
        "total_reference_yield": total_reference_yield,
        "seasonal_budget": seasonal_budget,
    }


# Compatibility alias is intentionally NOT provided for the old
# year-only get_year_reference(), because silently dropping climate
# would be scientifically unsafe.


# ============================================================
# 11. CREATE AQUACROP FIELD
# ============================================================

def create_field(climate, year, planting_date):
    climate = str(climate)
    year = int(year)

    if climate not in CLIMATES:
        raise RuntimeError(f"Unknown climate: {climate}")

    if year in FINAL_TEST_YEARS or year >= 2019:
        raise RuntimeError(
            f"FINAL TEST ACCESS BLOCKED while creating field: "
            f"{climate} {year}"
        )

    if year not in DEVELOPMENT_YEARS:
        raise RuntimeError(
            f"Year {year} is outside the frozen development period."
        )

    weather = WEATHER_BY_CLIMATE[climate]

    soil = Soil(soil_type="SandyLoam")

    crop = Crop(
        "Maize",
        planting_date=planting_date,
    )

    initial_water = InitialWaterContent(value=["FC"])

    irrigation = IrrigationManagement(
        irrigation_method=5,
        depth=0,
        MaxIrr=MAX_FIELD_IRRIGATION,
        MaxIrrSeason=10000,
    )

    model = AquaCropModel(
        sim_start_time=f"{year}/05/01",
        sim_end_time=f"{year}/10/31",
        weather_df=weather.copy(),
        soil=soil,
        crop=crop,
        initial_water_content=initial_water,
        irrigation_management=irrigation,
    )

    model._initialize()
    return model

# 12. MULTI-CLIMATE PPO ENVIRONMENT
# ============================================================

class MultiClimateSharedWaterEnv(
    gym.Env
):

    metadata = {
        "render_modes": []
    }


    def __init__(
        self,
        years,
        climates=None,
        seed=42,
    ):

        super().__init__()

        self.years = [int(year) for year in years]

        self.climates = (
            list(CLIMATES)
            if climates is None
            else [str(climate) for climate in climates]
        )

        if not self.years:
            raise RuntimeError("Environment received no years.")

        if not self.climates:
            raise RuntimeError("Environment received no climates.")

        unknown_climates = sorted(
            set(self.climates) - set(CLIMATES)
        )
        if unknown_climates:
            raise RuntimeError(
                f"Unknown climates requested: {unknown_climates}"
            )

        # ----------------------------------------
        # HARD PROTECTION AGAINST FINAL TEST
        # ----------------------------------------
        if any(
            year in FINAL_TEST_YEARS or year >= 2019
            for year in self.years
        ):
            raise RuntimeError(
                "Training/development environment contains "
                "a protected final-test year (2019-2025)."
            )

        invalid_years = sorted(
            set(self.years) - set(DEVELOPMENT_YEARS)
        )
        if invalid_years:
            raise RuntimeError(
                f"Years outside frozen development period: {invalid_years}"
            )

        self.climate_year_pairs = [
            (climate, year)
            for climate in self.climates
            for year in self.years
        ]

        missing_pairs = [
            (climate, year)
            for climate, year in self.climate_year_pairs
            if (climate, year) not in available_reference_pairs
        ]
        if missing_pairs:
            raise RuntimeError(
                "Missing development references for requested pairs: "
                f"{missing_pairs[:10]}"
            )


        self.base_seed = int(
            seed
        )


        self.rng = (
            np.random.default_rng(
                self.base_seed
            )
        )


        # ----------------------------------------
        # ACTION
        #
        # a_i ∈ [0,1]
        #
        # desired_i =
        #     a_i × current_request_i
        # ----------------------------------------

        self.action_space = (
            spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
        )


        # ----------------------------------------
        # OBSERVATION
        #
        # 7 per field × 4 = 28
        #
        # + remaining budget fraction
        # + remaining time fraction
        # + pacing ratio
        # + 3-dim climate one-hot vector
        #
        # = 34
        # ----------------------------------------

        self.observation_space = (
            spaces.Box(
                low=0.0,
                high=1.0,
                shape=(34,),
                dtype=np.float32,
            )
        )


        self.current_year = None

        self.current_climate = None

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


    # ========================================================
    # 12.1 CLIMATE-YEAR SELECTION
    # ========================================================

    def _select_climate_year(self, options=None):
        """
        Select one development Climate-Year pair.

        With options={"climate": ..., "year": ...}, selection
        is deterministic. This is useful for validation and smoke
        tests and remains subject to the same final-test protection.
        """

        if options is not None and (
            "climate" in options or "year" in options
        ):
            if "climate" not in options or "year" not in options:
                raise RuntimeError(
                    "Deterministic reset requires both climate and year."
                )

            climate = str(options["climate"])
            year = int(options["year"])

            if climate not in self.climates:
                raise RuntimeError(
                    f"Requested climate {climate} is not enabled in this environment."
                )

            if year not in self.years:
                raise RuntimeError(
                    f"Requested year {year} is not enabled in this environment."
                )

            if year in FINAL_TEST_YEARS or year >= 2019:
                raise RuntimeError(
                    f"FINAL TEST ACCESS BLOCKED: {climate} {year}"
                )

            if (climate, year) not in available_reference_pairs:
                raise RuntimeError(
                    f"Missing reference for {climate} {year}."
                )

            return climate, year

        pair_index = int(
            self.rng.integers(0, len(self.climate_year_pairs))
        )

        climate, year = self.climate_year_pairs[pair_index]

        if year in FINAL_TEST_YEARS or year >= 2019:
            raise RuntimeError(
                "FINAL TEST climate-year selected during PPO development."
            )

        return str(climate), int(year)


    # ========================================================
    # 13.2 ACTIVE FIELD
    # ========================================================

    def _is_crop_active(
        self,
        model,
    ):

        if (
            model
            ._clock_struct
            .model_is_finished
        ):

            return False


        state = model._init_cond


        dap = float(
            getattr(
                state,
                "dap",
                0.0,
            )
        )


        growing_season = bool(
            getattr(
                state,
                "growing_season",
                False,
            )
        )


        crop_mature = bool(
            getattr(
                state,
                "crop_mature",
                False,
            )
        )


        crop_dead = bool(
            getattr(
                state,
                "crop_dead",
                False,
            )
        )


        return (
            dap > 0
            and growing_season
            and not crop_mature
            and not crop_dead
        )


    # ========================================================
    # 13.3 HARMONIZED REQUEST
    # ========================================================

    def _get_field_request(
        self,
        model,
    ):

        if not self._is_crop_active(
            model
        ):

            return 0.0


        state = model._init_cond


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


        if taw <= 1e-9:

            return 0.0


        q = (
            depletion
            / taw
        )


        if (
            q
            >= DEPLETION_TRIGGER
        ):

            request = min(
                max(
                    depletion,
                    0.0,
                ),
                MAX_FIELD_IRRIGATION,
            )


            return float(
                request
            )


        return 0.0


    # ========================================================
    # 13.4 OBSERVATION
    # ========================================================

    def _get_observation(
        self,
    ):

        obs = []


        for field_name in FIELD_NAMES:

            model = self.models[
                field_name
            ]


            if (
                model
                ._clock_struct
                .model_is_finished
            ):

                obs.extend(
                    [
                        0.0,  # depletion
                        0.0,  # growth stage
                        0.0,  # canopy
                        0.0,  # GDD
                        0.0,  # yield formation
                        0.0,  # transpiration ratio
                        1.0,  # irrigation progress
                    ]
                )

                continue


            state = model._init_cond


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


            growth_stage = float(
                getattr(
                    state,
                    "growth_stage",
                    0.0,
                )
            )


            canopy_cover = float(
                getattr(
                    state,
                    "canopy_cover",
                    0.0,
                )
            )


            gdd_cum = float(
                getattr(
                    state,
                    "gdd_cum",
                    0.0,
                )
            )


            yield_form = float(
                bool(
                    getattr(
                        state,
                        "yield_form",
                        False,
                    )
                )
            )


            tr_ratio = float(
                getattr(
                    state,
                    "tr_ratio",
                    1.0,
                )
            )


            irr_cum = float(
                getattr(
                    state,
                    "irr_cum",
                    0.0,
                )
            )


            field_reference_irrigation = float(
                self.reference_irrigation[
                    field_name
                ]
            )


            if (
                field_reference_irrigation
                > 1e-9
            ):

                irrigation_progress = (
                    irr_cum
                    / field_reference_irrigation
                )

            else:

                irrigation_progress = 0.0


            obs.extend(
                [
                    float(
                        np.clip(
                            depletion_ratio,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            growth_stage
                            / 4.0,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            canopy_cover,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            gdd_cum
                            / 2000.0,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            yield_form,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            tr_ratio,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            irrigation_progress,
                            0.0,
                            1.0,
                        )
                    ),
                ]
            )


        # ----------------------------------------
        # GLOBAL FEATURES
        # ----------------------------------------

        if (
            self.seasonal_budget
            > 1e-9
        ):

            remaining_budget_fraction = (
                self.remaining_budget
                / self.seasonal_budget
            )

        else:

            remaining_budget_fraction = 0.0


        total_days = (
            pd.Timestamp(
                year=self.current_year,
                month=10,
                day=31,
            )
            -
            pd.Timestamp(
                year=self.current_year,
                month=5,
                day=1,
            )
        ).days + 1


        remaining_days = max(
            total_days
            - self.step_count,
            1,
        )


        remaining_time_fraction = (
            remaining_days
            / total_days
        )


        pacing_need = (
            self.remaining_budget
            / remaining_days
        )


        pacing_fraction = (
            pacing_need
            / DAILY_SYSTEM_CAPACITY
        )


        obs.extend(
            [
                float(
                    np.clip(
                        remaining_budget_fraction,
                        0.0,
                        1.0,
                    )
                ),

                float(
                    np.clip(
                        remaining_time_fraction,
                        0.0,
                        1.0,
                    )
                ),

                float(
                    np.clip(
                        pacing_fraction,
                        0.0,
                        1.0,
                    )
                ),
            ]
        )

        # ----------------------------------------
        # CLIMATE IDENTITY
        # ----------------------------------------
        # One-hot identity allows one shared PPO policy to
        # distinguish the three predeclared climatic regimes.
        # No future weather information is exposed.
        climate_onehot = CLIMATE_TO_ONEHOT[
            self.current_climate
        ]

        obs.extend(
            [float(value) for value in climate_onehot]
        )


        observation = np.asarray(
            obs,
            dtype=np.float32,
        )


        if (
            observation.shape
            != (34,)
        ):

            raise RuntimeError(
                f"Invalid observation shape: "
                f"{observation.shape}"
            )


        if not np.all(
            np.isfinite(
                observation
            )
        ):

            raise RuntimeError(
                "Non-finite value "
                "in observation."
            )


        return observation


    # ========================================================
    # 12.5 RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        (
            self.current_climate,
            self.current_year,
        ) = self._select_climate_year(options=options)

        climate_year_ref = get_climate_year_reference(
            self.current_climate,
            self.current_year,
        )

        self.reference_yields = climate_year_ref[
            "reference_yields"
        ]

        self.reference_irrigation = climate_year_ref[
            "reference_irrigation"
        ]

        self.total_reference_irrigation = climate_year_ref[
            "total_reference_irrigation"
        ]

        self.total_reference_yield = climate_year_ref[
            "total_reference_yield"
        ]

        self.seasonal_budget = climate_year_ref[
            "seasonal_budget"
        ]

        self.remaining_budget = float(
            self.seasonal_budget
        )

        self.models = {}

        for cfg in FIELD_CONFIGS:
            self.models[cfg["field"]] = create_field(
                climate=self.current_climate,
                year=self.current_year,
                planting_date=cfg["planting_date"],
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

        observation = self._get_observation()

        info = {
            "climate": self.current_climate,
            "year": self.current_year,
            "episode_number": self.episode_number,
            "seasonal_budget": self.seasonal_budget,
            "total_reference_irrigation": self.total_reference_irrigation,
            "total_reference_yield": self.total_reference_yield,
        }

        return observation, info


    # ========================================================
    # 13.6 DAILY STRESS
    # ========================================================

    def _calculate_daily_stress(
        self,
    ):

        stress_values = []


        for field_name in FIELD_NAMES:

            model = self.models[
                field_name
            ]


            if not self._is_crop_active(
                model
            ):

                continue


            state = model._init_cond


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

                q = (
                    depletion
                    / taw
                )

            else:

                q = 0.0


            # Only penalize depletion above
            # the irrigation trigger.
            #
            # This avoids penalizing normal,
            # non-stressed root-zone depletion.

            if (
                q
                > DEPLETION_TRIGGER
            ):

                excess_stress = (
                    q
                    - DEPLETION_TRIGGER
                ) / (
                    1.0
                    - DEPLETION_TRIGGER
                )

            else:

                excess_stress = 0.0


            stress_values.append(
                float(
                    np.clip(
                        excess_stress,
                        0.0,
                        1.0,
                    )
                )
            )


        if not stress_values:

            return 0.0


        return float(
            np.mean(
                stress_values
            )
        )


    # ========================================================
    # 13.7 FINAL RESULTS
    # ========================================================

    def _get_final_results(
        self,
    ):

        yields = {}

        irrigations = {}

        retentions = {}


        for field_name in FIELD_NAMES:

            results = (
                self.models[
                    field_name
                ]
                .get_simulation_results()
            )


            if (
                results is False
                or len(results) == 0
            ):

                crop_yield = 0.0

                irrigation = 0.0

            else:

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


            reference_yield = float(
                self.reference_yields[
                    field_name
                ]
            )


            if reference_yield > 1e-12:

                retention = (
                    crop_yield
                    / reference_yield
                )

            else:

                retention = 0.0


            yields[
                field_name
            ] = crop_yield


            irrigations[
                field_name
            ] = irrigation


            retentions[
                field_name
            ] = retention


        retention_vector = (
            np.asarray(
                [
                    retentions[field]
                    for field
                    in FIELD_NAMES
                ],
                dtype=np.float64,
            )
        )


        total_yield = float(
            sum(
                yields.values()
            )
        )


        total_irrigation = float(
            sum(
                irrigations.values()
            )
        )


        if (
            self.total_reference_yield
            > 1e-12
        ):

            total_yield_retention = (
                total_yield
                / self.total_reference_yield
            )

        else:

            total_yield_retention = 0.0


        mean_retention = float(
            retention_vector.mean()
        )


        worst_retention = float(
            retention_vector.min()
        )


        denominator = (
            len(
                retention_vector
            )
            *
            np.sum(
                retention_vector ** 2
            )
        )


        if denominator > 1e-12:

            jain = float(
                (
                    np.sum(
                        retention_vector
                    ) ** 2
                )
                /
                denominator
            )

        else:

            jain = 0.0


        if total_irrigation > 1e-12:

            water_productivity = (
                total_yield
                / total_irrigation
            )

        else:

            water_productivity = 0.0


        clipped_retention = (
            np.clip(
                retention_vector,
                0.0,
                1.0,
            )
        )


        squared_loss = float(
            np.mean(
                (
                    1.0
                    - clipped_retention
                ) ** 2
            )
        )


        return {
            "yields":
                yields,

            "irrigations":
                irrigations,

            "retentions":
                retentions,

            "total_yield":
                total_yield,

            "total_yield_retention":
                total_yield_retention,

            "total_irrigation":
                total_irrigation,

            "mean_retention":
                mean_retention,

            "worst_retention":
                worst_retention,

            "jain":
                jain,

            "water_productivity":
                water_productivity,

            "squared_loss":
                squared_loss,
        }


    # ========================================================
    # 13.8 STEP
    # ========================================================

    def step(
        self,
        action,
    ):

        if self.episode_finished:

            raise RuntimeError(
                "step() called after termination."
            )


        action = np.asarray(
            action,
            dtype=np.float64,
        )


        action = np.clip(
            action,
            0.0,
            1.0,
        )


        if action.shape != (4,):

            raise RuntimeError(
                f"Invalid action shape: "
                f"{action.shape}"
            )


        # ----------------------------------------
        # CURRENT REQUESTS
        # ----------------------------------------

        requests = {}


        for field_name in FIELD_NAMES:

            requests[
                field_name
            ] = (
                self._get_field_request(
                    self.models[
                        field_name
                    ]
                )
            )


        request_vector = np.asarray(
            [
                requests[field]
                for field
                in FIELD_NAMES
            ],
            dtype=np.float64,
        )


        total_request = float(
            request_vector.sum()
        )


        self.cumulative_request += (
            total_request
        )


        if total_request > 1e-9:

            self.request_days += 1


        if (
            total_request
            >
            DAILY_SYSTEM_CAPACITY
            + 1e-9
        ):

            self.competition_days += 1


        # ----------------------------------------
        # REQUEST-AWARE PPO ACTION
        # ----------------------------------------

        desired = (
            action
            * request_vector
        )


        available_today = min(
            DAILY_SYSTEM_CAPACITY,
            self.remaining_budget,
        )


        desired_total = float(
            desired.sum()
        )


        if (
            desired_total
            > available_today
            and desired_total
            > 1e-12
        ):

            allocations = (
                desired
                *
                (
                    available_today
                    / desired_total
                )
            )

        else:

            allocations = desired


        total_today = float(
            allocations.sum()
        )


        # ----------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------

        if np.any(
            allocations
            >
            request_vector
            + 1e-8
        ):

            raise RuntimeError(
                "PPO allocation exceeds "
                "field request."
            )


        if (
            total_today
            >
            DAILY_SYSTEM_CAPACITY
            + 1e-8
        ):

            raise RuntimeError(
                "Daily capacity exceeded."
            )


        if (
            total_today
            >
            self.remaining_budget
            + 1e-8
        ):

            raise RuntimeError(
                "Seasonal budget exceeded."
            )


        if total_today > 1e-9:

            self.irrigation_days += 1


        # ----------------------------------------
        # DAILY REWARD BEFORE STEPPING
        # ----------------------------------------

        daily_stress = (
            self._calculate_daily_stress()
        )


        daily_reward = (
            -DAILY_STRESS_WEIGHT
            * daily_stress
        )


        # ----------------------------------------
        # ADVANCE AQUACROP
        # ----------------------------------------

        for index, field_name in enumerate(
            FIELD_NAMES
        ):

            model = self.models[
                field_name
            ]


            if (
                model
                ._clock_struct
                .model_is_finished
            ):

                continue


            depth = float(
                allocations[
                    index
                ]
            )


            model._param_struct.IrrMngt.depth = (
                depth
            )


            model.run_model(
                num_steps=1,
                initialize_model=False,
                process_outputs=False,
            )


        # ----------------------------------------
        # WATER ACCOUNTING
        # ----------------------------------------

        self.remaining_budget -= (
            total_today
        )


        self.remaining_budget = max(
            self.remaining_budget,
            0.0,
        )


        self.total_allocated += (
            total_today
        )


        self.step_count += 1


        if (
            self.remaining_budget
            <= 1e-9
            and
            self.budget_exhaustion_step
            is None
        ):

            self.budget_exhaustion_step = (
                self.step_count
            )


        # ----------------------------------------
        # TERMINATION
        # ----------------------------------------

        all_finished = all(
            model
            ._clock_struct
            .model_is_finished
            for model
            in self.models.values()
        )


        terminated = bool(
            all_finished
        )


        truncated = False


        terminal_reward = 0.0

        final_results = None


        if terminated:

            self.episode_finished = True


            final_results = (
                self._get_final_results()
            )


            # ------------------------------------
            # TERMINAL YIELD TERM
            # ------------------------------------

            yield_component = float(
                np.clip(
                    final_results[
                        "total_yield_retention"
                    ],
                    0.0,
                    1.10,
                )
            )


            # ------------------------------------
            # FAIRNESS TERM
            # ------------------------------------

            jain_component = float(
                np.clip(
                    final_results[
                        "jain"
                    ],
                    0.0,
                    1.0,
                )
            )


            # ------------------------------------
            # WATER PRODUCTIVITY TERM
            #
            # Typical WP here is around 0.04-0.05
            # t/ha/mm.
            #
            # Divide by 0.05 so that a good
            # result is approximately O(1).
            # ------------------------------------

            wp_component = float(
                np.clip(
                    final_results[
                        "water_productivity"
                    ]
                    / 0.05,
                    0.0,
                    1.5,
                )
            )


            # ------------------------------------
            # LOSS TERM
            # ------------------------------------

            loss_component = float(
                final_results[
                    "squared_loss"
                ]
            )


            # ------------------------------------
            # WORST-FIELD RETENTION TERM
            #
            # Already computed from the four field-level
            # yield-retention ratios.
            #
            # The HPO framework may set its coefficient to
            # zero or a positive value.
            # ------------------------------------

            worst_component = float(
                np.clip(
                    final_results[
                        "worst_retention"
                    ],
                    0.0,
                    1.0,
                )
            )


            terminal_reward = (
                TERMINAL_YIELD_WEIGHT
                * yield_component
                +
                TERMINAL_JAIN_WEIGHT
                * jain_component
                +
                TERMINAL_WORST_WEIGHT
                * worst_component
                +
                TERMINAL_WP_WEIGHT
                * wp_component
                -
                TERMINAL_LOSS_WEIGHT
                * loss_component
            )


        reward = float(
            daily_reward
            + terminal_reward
        )


        observation = (
            self._get_observation()
        )


        info = {
            "climate":
                self.current_climate,

            "year":
                self.current_year,

            "episode_number":
                self.episode_number,

            "step":
                self.step_count,

            "requests":
                requests,

            "total_request":
                total_request,

            "allocations": {
                field:
                    float(
                        allocations[index]
                    )
                for index, field
                in enumerate(
                    FIELD_NAMES
                )
            },

            "total_today":
                total_today,

            "remaining_budget":
                self.remaining_budget,

            "seasonal_budget":
                self.seasonal_budget,

            "daily_stress":
                daily_stress,

            "final_results":
                final_results,
        }


        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )




# Compatibility alias for scripts that imported the previous class name.
# Semantics are now multi-climate and still DEVELOPMENT-ONLY.
MultiYearSharedWaterEnv = MultiClimateSharedWaterEnv

# ============================================================
# REWARD CONFIGURATION INTERFACE
# ============================================================

def configure_reward(
    daily_stress_weight,
    terminal_yield_weight,
    terminal_jain_weight,
    terminal_wp_weight,
    terminal_loss_weight,
    terminal_worst_weight=0.0,
):
    """
    Set reward coefficients for one optimization experiment.

    The optimization scripts, not this module, determine
    candidate values.
    """

    global DAILY_STRESS_WEIGHT
    global TERMINAL_YIELD_WEIGHT
    global TERMINAL_JAIN_WEIGHT
    global TERMINAL_WP_WEIGHT
    global TERMINAL_LOSS_WEIGHT
    global TERMINAL_WORST_WEIGHT

    DAILY_STRESS_WEIGHT = float(
        daily_stress_weight
    )

    TERMINAL_YIELD_WEIGHT = float(
        terminal_yield_weight
    )

    TERMINAL_JAIN_WEIGHT = float(
        terminal_jain_weight
    )

    TERMINAL_WP_WEIGHT = float(
        terminal_wp_weight
    )

    TERMINAL_LOSS_WEIGHT = float(
        terminal_loss_weight
    )

    TERMINAL_WORST_WEIGHT = float(
        terminal_worst_weight
    )


def get_reward_configuration():
    """
    Return current reward coefficients.
    """

    return {
        "daily_stress_weight":
            float(
                DAILY_STRESS_WEIGHT
            ),

        "terminal_yield_weight":
            float(
                TERMINAL_YIELD_WEIGHT
            ),

        "terminal_jain_weight":
            float(
                TERMINAL_JAIN_WEIGHT
            ),

        "terminal_wp_weight":
            float(
                TERMINAL_WP_WEIGHT
            ),

        "terminal_loss_weight":
            float(
                TERMINAL_LOSS_WEIGHT
            ),

        "terminal_worst_weight":
            float(
                TERMINAL_WORST_WEIGHT
            ),
    }



# ============================================================
# VALIDATION OF FROZEN MULTI-CLIMATE TEMPORAL PARTITION
# ============================================================

def validate_temporal_partition():
    development = set(DEVELOPMENT_YEARS)
    final_test = set(FINAL_TEST_YEARS)

    if development & final_test:
        raise RuntimeError("Development/test overlap.")

    if DEVELOPMENT_YEARS != list(range(1984, 2019)):
        raise RuntimeError("Development period no longer matches frozen protocol.")

    if FINAL_TEST_YEARS != list(range(2019, 2026)):
        raise RuntimeError("Final-test period no longer matches frozen protocol.")

    for fold in TEMPORAL_FOLDS:
        training = set(fold["train_years"])
        validation = set(fold["validation_years"])

        if training & validation:
            raise RuntimeError(
                f"{fold['name']}: train/validation overlap."
            )

        if not training.issubset(development):
            raise RuntimeError(
                f"{fold['name']}: training year outside development period."
            )

        if not validation.issubset(development):
            raise RuntimeError(
                f"{fold['name']}: validation year outside development period."
            )

        if training & final_test:
            raise RuntimeError(
                f"{fold['name']}: test-year leakage in training."
            )

        if validation & final_test:
            raise RuntimeError(
                f"{fold['name']}: test-year leakage in validation."
            )

        if max(training) >= min(validation):
            raise RuntimeError(
                f"{fold['name']}: temporal order violated."
            )

    expected_folds = [
        (1984, 1997, 1998, 2004),
        (1984, 2004, 2005, 2011),
        (1984, 2011, 2012, 2018),
    ]

    actual_folds = [
        (
            min(fold["train_years"]),
            max(fold["train_years"]),
            min(fold["validation_years"]),
            max(fold["validation_years"]),
        )
        for fold in TEMPORAL_FOLDS
    ]

    if actual_folds != expected_folds:
        raise RuntimeError("Temporal folds no longer match frozen protocol.")

    return True


validate_temporal_partition()
