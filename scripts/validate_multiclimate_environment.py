# ============================================================
# MULTI-CLIMATE ENVIRONMENT VALIDATION
# MULTI-CLIMATE ENVIRONMENT SMOKE TEST
#
# PURPOSE
# -------
# Verify the new development-only multi-climate environment
# before any PPO training or controller comparison.
#
# This script checks:
#   1. Frozen 1984-2018 / 2019-2025 temporal protocol.
#   2. 34-dimensional observation space.
#   3. Climate one-hot identity.
#   4. Deterministic Climate-Year reset.
#   5. AquaCrop execution in all three climates.
#   6. Daily and seasonal water accounting.
#   7. Explicit blocking of 2019-2025.
#   8. Gymnasium/SB3 environment compatibility.
#
# NO PPO TRAINING IS PERFORMED.
# NO Equal/Priority evaluation is performed.
# NO 2019-2025 simulation is performed.
# ============================================================

from pathlib import Path
import json
import time

import numpy as np

from stable_baselines3.common.env_checker import check_env

import multiclimate_optimization_core as mc


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path("results")
    / "multiclimate_extension"
    / "smoke_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN SMOKE CASES
# ============================================================
# All cases are development-only and deliberately spread over
# time and climate. They are not used for model selection.
# ============================================================

SMOKE_CASES = [
    ("Tunis", 1984),
    ("Niamey", 2004),
    ("Cotonou", 2018),
]

EXPECTED_ONEHOT = {
    "Tunis": np.asarray(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    ),
    "Niamey": np.asarray(
        [0.0, 1.0, 0.0],
        dtype=np.float32,
    ),
    "Cotonou": np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float32,
    ),
}


# ============================================================
# HELPERS
# ============================================================

def assert_close(
    a,
    b,
    tolerance=1e-6,
    label="values",
):
    if abs(float(a) - float(b)) > tolerance:
        raise RuntimeError(
            f"Mismatch for {label}: "
            f"{a} vs {b}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 78)
    print("MULTI-CLIMATE ENVIRONMENT VALIDATION")
    print("MULTI-CLIMATE ENVIRONMENT SMOKE TEST")
    print("#" * 78)

    start_time = time.perf_counter()

    # ========================================================
    # 1. FROZEN PROTOCOL
    # ========================================================

    print()
    print("=" * 78)
    print("1. FROZEN PROTOCOL CHECK")
    print("=" * 78)

    if not mc.validate_temporal_partition():
        raise RuntimeError(
            "Temporal partition validation returned False."
        )

    assert mc.DEVELOPMENT_YEARS == list(
        range(1984, 2019)
    )

    assert mc.FINAL_TEST_YEARS == list(
        range(2019, 2026)
    )

    assert mc.CLIMATES == [
        "Tunis",
        "Niamey",
        "Cotonou",
    ]

    print("Development years : 1984-2018")
    print("Protected test    : 2019-2025")
    print(
        f"Climates          : {mc.CLIMATES}"
    )
    print("Frozen protocol   : PASS")


    # ========================================================
    # 2. EXPLICIT FINAL-TEST BLOCKING
    # ========================================================

    print()
    print("=" * 78)
    print("2. FINAL-TEST PROTECTION")
    print("=" * 78)

    constructor_blocked = False

    try:
        mc.MultiClimateSharedWaterEnv(
            years=[2019],
            climates=["Tunis"],
            seed=1,
        )
    except RuntimeError as exc:
        constructor_blocked = True
        print(
            "2019 constructor access blocked: PASS"
        )
        print(f"  Message: {exc}")

    if not constructor_blocked:
        raise RuntimeError(
            "Environment incorrectly allowed 2019."
        )

    reference_blocked = False

    try:
        mc.get_climate_year_reference(
            "Cotonou",
            2025,
        )
    except RuntimeError as exc:
        reference_blocked = True
        print(
            "2025 reference access blocked  : PASS"
        )
        print(f"  Message: {exc}")

    if not reference_blocked:
        raise RuntimeError(
            "Reference function incorrectly allowed 2025."
        )


    # ========================================================
    # 3. OBSERVATION / ENVIRONMENT STRUCTURE
    # ========================================================

    print()
    print("=" * 78)
    print("3. ENVIRONMENT STRUCTURE")
    print("=" * 78)

    structure_env = (
        mc.MultiClimateSharedWaterEnv(
            years=[
                1984,
                1998,
                2005,
                2012,
                2018,
            ],
            climates=mc.CLIMATES,
            seed=5301,
        )
    )

    if structure_env.action_space.shape != (4,):
        raise RuntimeError(
            "Action space is not 4-dimensional."
        )

    if structure_env.observation_space.shape != (34,):
        raise RuntimeError(
            "Observation space is not 34-dimensional."
        )

    print(
        f"Action shape      : "
        f"{structure_env.action_space.shape}"
    )

    print(
        f"Observation shape : "
        f"{structure_env.observation_space.shape}"
    )

    print("Structure check   : PASS")


    # ========================================================
    # 4. SB3/GYMNASIUM CHECK
    # ========================================================

    print()
    print("=" * 78)
    print("4. SB3 / GYMNASIUM ENVIRONMENT CHECK")
    print("=" * 78)

    check_env(
        structure_env,
        warn=True,
        skip_render_check=True,
    )

    print("SB3 check_env     : PASS")


    # ========================================================
    # 5. DETERMINISTIC COMPLETE EPISODES
    # ========================================================

    print()
    print("=" * 78)
    print("5. THREE-CLIMATE COMPLETE-EPISODE SMOKE TEST")
    print("=" * 78)

    smoke_rows = []

    for case_index, (
        climate,
        year,
    ) in enumerate(
        SMOKE_CASES,
        start=1,
    ):

        print()
        print("-" * 78)
        print(
            f"CASE {case_index}: "
            f"{climate} {year}"
        )
        print("-" * 78)

        env = (
            mc.MultiClimateSharedWaterEnv(
                years=[year],
                climates=[climate],
                seed=(
                    5300
                    + case_index
                ),
            )
        )

        obs, info = env.reset(
            seed=(
                5300
                + case_index
            ),
            options={
                "climate": climate,
                "year": year,
            },
        )

        # ----------------------------------------
        # Deterministic reset
        # ----------------------------------------

        if info["climate"] != climate:
            raise RuntimeError(
                "Deterministic climate reset failed."
            )

        if int(info["year"]) != year:
            raise RuntimeError(
                "Deterministic year reset failed."
            )

        # ----------------------------------------
        # Observation
        # ----------------------------------------

        if obs.shape != (34,):
            raise RuntimeError(
                f"Bad observation shape for "
                f"{climate} {year}: {obs.shape}"
            )

        if not np.all(
            np.isfinite(obs)
        ):
            raise RuntimeError(
                "Non-finite observation."
            )

        if np.any(obs < -1e-8) or np.any(
            obs > 1.0 + 1e-8
        ):
            raise RuntimeError(
                "Observation outside [0, 1]."
            )

        observed_onehot = obs[-3:]

        if not np.allclose(
            observed_onehot,
            EXPECTED_ONEHOT[climate],
            atol=1e-8,
        ):
            raise RuntimeError(
                f"Climate one-hot mismatch for "
                f"{climate}: {observed_onehot}"
            )

        print(
            f"Observation      : {obs.shape}"
        )
        print(
            f"Climate one-hot  : "
            f"{observed_onehot.tolist()}"
        )
        print(
            f"Reference water  : "
            f"{info['total_reference_irrigation']:.3f} mm"
        )
        print(
            f"40% budget       : "
            f"{info['seasonal_budget']:.3f} mm"
        )

        expected_budget = (
            mc.SCARCITY_FRACTION
            * info[
                "total_reference_irrigation"
            ]
        )

        assert_close(
            info["seasonal_budget"],
            expected_budget,
            tolerance=1e-8,
            label="40% seasonal budget",
        )

        # ----------------------------------------
        # Run full episode
        # ----------------------------------------
        # action=1 means "satisfy every currently
        # active request as much as shared constraints
        # allow". It is ONLY a mechanics smoke test,
        # not a controller result for the study.
        # ----------------------------------------

        terminated = False
        truncated = False
        final_info = None

        while not (
            terminated
            or truncated
        ):

            action = np.ones(
                4,
                dtype=np.float32,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                final_info,
            ) = env.step(action)

            if env.step_count > 250:
                raise RuntimeError(
                    "Smoke episode exceeded 250 days."
                )

        if truncated:
            raise RuntimeError(
                f"Unexpected truncation: "
                f"{climate} {year}"
            )

        if final_info is None:
            raise RuntimeError(
                "Missing terminal info."
            )

        final_results = final_info[
            "final_results"
        ]

        if final_results is None:
            raise RuntimeError(
                "Missing final_results at termination."
            )

        # ----------------------------------------
        # Water accounting
        # ----------------------------------------

        aqua_irrigation = float(
            final_results[
                "total_irrigation"
            ]
        )

        accounting_difference = (
            float(env.total_allocated)
            - aqua_irrigation
        )

        if abs(
            accounting_difference
        ) > 1e-6:
            raise RuntimeError(
                f"Water accounting failed for "
                f"{climate} {year}: "
                f"difference="
                f"{accounting_difference:.10f} mm"
            )

        if (
            env.total_allocated
            > env.seasonal_budget
            + 1e-6
        ):
            raise RuntimeError(
                "Seasonal budget exceeded."
            )

        if env.remaining_budget < -1e-8:
            raise RuntimeError(
                "Negative remaining budget."
            )

        budget_used_pct = (
            env.total_allocated
            / env.seasonal_budget
            * 100.0
            if env.seasonal_budget > 1e-12
            else 0.0
        )

        print(
            f"Steps            : "
            f"{env.step_count}"
        )
        print(
            f"Allocated water  : "
            f"{env.total_allocated:.6f} mm"
        )
        print(
            f"AquaCrop water   : "
            f"{aqua_irrigation:.6f} mm"
        )
        print(
            f"Accounting diff  : "
            f"{accounting_difference:.10f} mm"
        )
        print(
            f"Budget used      : "
            f"{budget_used_pct:.3f}%"
        )
        print(
            f"Yield retention  : "
            f"{final_results['total_yield_retention'] * 100:.3f}%"
        )
        print(
            f"Worst retention  : "
            f"{final_results['worst_retention'] * 100:.3f}%"
        )
        print(
            f"Jain fairness    : "
            f"{final_results['jain']:.6f}"
        )
        print("Episode mechanics: PASS")

        smoke_rows.append(
            {
                "Climate": climate,
                "Year": year,
                "Observation_dim": 34,
                "Steps": int(
                    env.step_count
                ),
                "Reference_irrigation_mm": float(
                    env.total_reference_irrigation
                ),
                "Seasonal_budget_mm": float(
                    env.seasonal_budget
                ),
                "Allocated_mm": float(
                    env.total_allocated
                ),
                "AquaCrop_irrigation_mm": float(
                    aqua_irrigation
                ),
                "Accounting_difference_mm": float(
                    accounting_difference
                ),
                "Budget_used_pct": float(
                    budget_used_pct
                ),
                "Total_yield_retention_pct": float(
                    final_results[
                        "total_yield_retention"
                    ]
                    * 100.0
                ),
                "Worst_field_retention_pct": float(
                    final_results[
                        "worst_retention"
                    ]
                    * 100.0
                ),
                "Jain": float(
                    final_results[
                        "jain"
                    ]
                ),
                "PASS": True,
            }
        )


    # ========================================================
    # 6. SAVE SMOKE RECORD
    # ========================================================

    import pandas as pd

    smoke_df = pd.DataFrame(
        smoke_rows
    )

    smoke_file = (
        OUTPUT_DIR
        / "environment_validation_multiclimate_smoke_results.csv"
    )

    smoke_df.to_csv(
        smoke_file,
        index=False,
    )

    integrity = {
        "milestone": "53D",
        "development_only": True,
        "observation_dimension": 34,
        "climate_encoding": {
            key: value.tolist()
            for key, value
            in EXPECTED_ONEHOT.items()
        },
        "smoke_cases": [
            {
                "climate": climate,
                "year": year,
            }
            for climate, year
            in SMOKE_CASES
        ],
        "test_constructor_blocked": constructor_blocked,
        "test_reference_blocked": reference_blocked,
        "ppo_training_performed": False,
        "equal_evaluated": False,
        "priority_evaluated": False,
        "final_test_opened": False,
        "all_smoke_cases_passed": bool(
            smoke_df["PASS"].all()
        ),
    }

    integrity_file = (
        OUTPUT_DIR
        / "environment_validation_multiclimate_smoke_integrity.json"
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

    print()
    print("=" * 78)
    print("MULTI-CLIMATE ENVIRONMENT VALIDATION SUMMARY")
    print("=" * 78)

    print(
        smoke_df.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.6f}"
            ),
        )
    )

    print()
    print("FINAL INTEGRITY")
    print("  34-dim observation           : PASS")
    print("  Climate identity             : PASS")
    print("  Three climates executed      : PASS")
    print("  Water accounting             : PASS")
    print("  2019-2025 constructor blocked: PASS")
    print("  2019-2025 reference blocked  : PASS")
    print("  PPO training performed       : NO")
    print("  Equal evaluated              : NO")
    print("  Priority evaluated           : NO")
    print("  Final test remains unopened  : YES")

    print()
    print(
        f"Runtime: {runtime:.2f} s "
        f"({runtime / 60.0:.2f} min)"
    )

    print()
    print("Saved:")
    print(f"  {smoke_file}")
    print(f"  {integrity_file}")

    print()
    print("=" * 78)
    print("MULTI-CLIMATE ENVIRONMENT VALIDATION COMPLETE")
    print("=" * 78)

    print()
    print(
        "NEXT: freeze the multi-climate PPO training "
        "specification before any PPO optimization or training."
    )


if __name__ == "__main__":
    main()
