# ============================================================
# OPTIMIZATION FEASIBILITY ASSESSMENT
# GLOBAL-OPTIMALITY FEASIBILITY ASSESSMENT
#
# PURPOSE
# -------
# Examine whether a genuinely globally optimal offline
# irrigation benchmark is computationally realistic for the
# frozen AquaCrop final-test problem.
#
# This script DOES NOT:
#   - train PPO
#   - perform PPO inference
#   - modify any controller
#   - optimize irrigation
#   - alter final-test results
#   - perform hypothesis tests
#
# It:
#   1. imports the exact frozen 53J evaluator;
#   2. creates Tunis-2019 at 40% scarcity;
#   3. counts the episode horizon and request structure;
#   4. characterizes feasible allocation branching for
#      several discretization levels;
#   5. tests whether the environment/AquaCrop state can be
#      deep-copied and branched reproducibly;
#   6. estimates naive exhaustive-search growth.
#
# IMPORTANT
# ---------
# The branching calculation follows ONE realized trajectory.
# Since AquaCrop requests are endogenous and depend on previous
# irrigation decisions, this is NOT a formal count of the full
# optimization tree. It is a feasibility diagnostic only.
# ============================================================


from pathlib import Path
import importlib.util
import copy
import json
import math
import pickle
import time

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

FINAL_SCRIPT = Path(
    "evaluate_all_controllers.py"
)

CLIMATE = "Tunis"

YEAR = 2019

SCARCITY = 0.40

SEED = 0


DISCRETIZATIONS_MM = [
    5.0,
    2.0,
    1.0,
]


EPS = 1e-9


OUTPUT_DIR = (
    Path("results")
    / "multiclimate_extension"
    / "global_optimality_feasibility"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


STEP_FILE = (
    OUTPUT_DIR
    / "optimization_feasibility_Tunis2019_40_request_structure.csv"
)


BRANCH_FILE = (
    OUTPUT_DIR
    / "optimization_feasibility_branching_analysis.csv"
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "optimization_feasibility_global_optimality_summary.json"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def allocation_vector(
    info,
    field_names,
):
    """
    Convert info["allocations"] into a numeric vector ordered
    according to field_names.

    The validated environment returns allocations as a dict:
        {
            field_name: allocation_mm,
            ...
        }
    """

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
                in field_names
            ],
            dtype=np.float64,
        )

    # Defensive fallback in case a future implementation
    # already returns a numeric sequence.

    arr = np.asarray(
        allocations,
        dtype=np.float64,
    )

    if arr.shape != (
        len(
            field_names
        ),
    ):

        raise RuntimeError(
            "Unexpected allocation structure: "
            f"{type(allocations)}; "
            f"shape={arr.shape}"
        )

    return arr


def count_feasible_allocations(
    requests,
    available,
    quantum,
):
    """
    Exact count of discretized feasible allocation vectors
    for FOUR fields at ONE decision state.

    Each allocation is represented as an integer multiple
    of 'quantum'.

    Constraints:
        allocation_i <= request_i
        sum(allocation_i) <= available

    This is only a local branching-count calculation.
    """

    requests = np.asarray(
        requests,
        dtype=np.float64,
    )

    if requests.shape != (
        4,
    ):

        raise RuntimeError(
            "Expected exactly four field requests."
        )

    max_units = [
        int(
            math.floor(
                max(
                    0.0,
                    float(r),
                )
                /
                quantum
                +
                1e-12
            )
        )
        for r
        in requests
    ]


    available_units = int(
        math.floor(
            max(
                0.0,
                float(
                    available
                ),
            )
            /
            quantum
            +
            1e-12
        )
    )


    count = 0


    # Four fields only.
    #
    # We explicitly enumerate the first three dimensions.
    # For the fourth field, all remaining feasible values
    # can be counted directly.

    for a in range(
        max_units[0]
        + 1
    ):

        for b in range(
            max_units[1]
            + 1
        ):

            for c in range(
                max_units[2]
                + 1
            ):

                used = (
                    a
                    + b
                    + c
                )

                if (
                    used
                    >
                    available_units
                ):

                    continue


                max_d = min(
                    max_units[3],
                    available_units
                    - used,
                )


                count += (
                    max_d
                    + 1
                )


    return int(
        count
    )


# ============================================================
# 3. IMPORT EXACT FROZEN FINAL EVALUATOR
# ============================================================

script_start = (
    time.perf_counter()
)


print()

print(
    "=" * 78
)

print(
    "OPTIMIZATION FEASIBILITY ASSESSMENT"
)

print(
    "GLOBAL-OPTIMALITY FEASIBILITY ASSESSMENT"
)

print(
    "=" * 78
)


if not FINAL_SCRIPT.exists():

    raise FileNotFoundError(
        f"Cannot find "
        f"{FINAL_SCRIPT}"
    )


spec = (
    importlib.util
    .spec_from_file_location(
        "final53j",
        FINAL_SCRIPT,
    )
)


if (
    spec is None
    or spec.loader is None
):

    raise RuntimeError(
        "Could not create import specification "
        "for the frozen 53J evaluator."
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


print()

print(
    f"Imported frozen evaluator: "
    f"{FINAL_SCRIPT}"
)


# ============================================================
# 4. BASIC INTERFACE CHECK
# ============================================================

required_attributes = [
    "FinalTestEnv",
    "current_requests",
    "FIELD_NAMES",
    "DAILY_SYSTEM_CAPACITY",
]


missing_attributes = [
    name
    for name
    in required_attributes
    if not hasattr(
        final53j,
        name,
    )
]


if missing_attributes:

    raise RuntimeError(
        "Frozen evaluator is missing required "
        f"attributes: {missing_attributes}"
    )


FIELD_NAMES = list(
    final53j.FIELD_NAMES
)


if len(
    FIELD_NAMES
) != 4:

    raise RuntimeError(
        "56A expects exactly four fields; "
        f"found {len(FIELD_NAMES)}."
    )


DAILY_CAPACITY = float(
    final53j
    .DAILY_SYSTEM_CAPACITY
)


# ============================================================
# 5. CREATE TEST ENVIRONMENT
# ============================================================

env = (
    final53j.FinalTestEnv(
        climate=CLIMATE,
        year=YEAR,
        scarcity_fraction=SCARCITY,
        seed=SEED,
    )
)


obs, reset_info = env.reset(
    seed=SEED
)


print(
    f"Case                     : "
    f"{CLIMATE} {YEAR}, "
    f"{100 * SCARCITY:.0f}%"
)


print(
    f"Daily shared capacity    : "
    f"{DAILY_CAPACITY:.3f} mm"
)


print(
    f"Seasonal budget          : "
    f"{env.seasonal_budget:.3f} mm"
)


# ============================================================
# 6. FOLLOW A NEUTRAL FULL-REQUEST POLICY
#
# IMPORTANT:
#
# This is NOT an optimization benchmark.
#
# Action [1,1,1,1] asks to satisfy the full endogenous request
# of each field. The validated environment itself applies the
# existing feasibility projection when total desired water
# exceeds daily or seasonal availability.
#
# The purpose is only to obtain ONE representative trajectory
# for branching-complexity diagnostics.
# ============================================================

records = []


terminated = False

truncated = False


episode_start = (
    time.perf_counter()
)


while not (
    terminated
    or truncated
):

    requests_dict = (
        final53j
        .current_requests(
            env
        )
    )


    requests = np.asarray(
        [
            requests_dict[
                name
            ]
            for name
            in FIELD_NAMES
        ],
        dtype=np.float64,
    )


    total_request = float(
        requests.sum()
    )


    remaining_before = float(
        env.remaining_budget
    )


    available_today = float(
        min(
            DAILY_CAPACITY,
            remaining_before,
        )
    )


    request_fields = int(
        np.sum(
            requests
            >
            EPS
        )
    )


    competition = bool(
        total_request
        >
        DAILY_CAPACITY
        +
        EPS
    )


    action = np.ones(
        4,
        dtype=np.float32,
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


    allocations = (
        allocation_vector(
            info=info,
            field_names=FIELD_NAMES,
        )
    )


    executed_total = float(
        allocations.sum()
    )


    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if np.any(
        allocations
        >
        requests
        +
        1e-8
    ):

        raise RuntimeError(
            "Observed allocation exceeds request."
        )


    if (
        executed_total
        >
        DAILY_CAPACITY
        +
        1e-8
    ):

        raise RuntimeError(
            "Observed allocation exceeds daily capacity."
        )


    if (
        executed_total
        >
        remaining_before
        +
        1e-8
    ):

        raise RuntimeError(
            "Observed allocation exceeds remaining "
            "seasonal budget."
        )


    row = {
        "Step":
            int(
                env.step_count
            ),

        "Request_fields":
            request_fields,

        "Total_request_mm":
            total_request,

        "Available_today_mm":
            available_today,

        "Remaining_budget_before_mm":
            remaining_before,

        "Competition_day":
            competition,

        "Executed_total_mm":
            executed_total,
    }


    for i, name in enumerate(
        FIELD_NAMES
    ):

        row[
            f"{name}_request_mm"
        ] = float(
            requests[
                i
            ]
        )


        row[
            f"{name}_executed_mm"
        ] = float(
            allocations[
                i
            ]
        )


    records.append(
        row
    )


episode_runtime = (
    time.perf_counter()
    -
    episode_start
)


trajectory = pd.DataFrame(
    records
)


trajectory.to_csv(
    STEP_FILE,
    index=False,
)


# ============================================================
# 7. BASIC HORIZON CHARACTERIZATION
# ============================================================

n_steps = int(
    len(
        trajectory
    )
)


request_mask = (
    trajectory[
        "Total_request_mm"
    ]
    >
    EPS
)


request_days = int(
    request_mask.sum()
)


competition_days = int(
    trajectory[
        "Competition_day"
    ].sum()
)


request_field_distribution = (
    trajectory.loc[
        request_mask,
        "Request_fields",
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. DISCRETIZED DAILY BRANCHING
#
# For each observed request day and each water quantum q:
#
#   allocation_i in {0, q, 2q, ...}
#
# subject to:
#
#   allocation_i <= request_i
#   sum(allocation_i) <= available water
#
# This is a LOCAL branching diagnostic only.
#
# It does NOT imply that future requests remain identical
# under alternative irrigation histories.
# ============================================================

branch_records = []


for _, row in trajectory.iterrows():

    if (
        float(
            row[
                "Total_request_mm"
            ]
        )
        <=
        EPS
    ):

        continue


    requests = np.asarray(
        [
            row[
                f"{name}_request_mm"
            ]
            for name
            in FIELD_NAMES
        ],
        dtype=np.float64,
    )


    available = float(
        row[
            "Available_today_mm"
        ]
    )


    for quantum in (
        DISCRETIZATIONS_MM
    ):

        count = (
            count_feasible_allocations(
                requests=requests,
                available=available,
                quantum=quantum,
            )
        )


        branch_records.append(
            {
                "Step":
                    int(
                        row[
                            "Step"
                        ]
                    ),

                "Quantum_mm":
                    float(
                        quantum
                    ),

                "Request_fields":
                    int(
                        row[
                            "Request_fields"
                        ]
                    ),

                "Total_request_mm":
                    float(
                        row[
                            "Total_request_mm"
                        ]
                    ),

                "Available_today_mm":
                    available,

                "Feasible_daily_allocations":
                    int(
                        count
                    ),

                "log10_daily_branches":
                    (
                        float(
                            math.log10(
                                count
                            )
                        )
                        if count
                        >
                        0
                        else
                        0.0
                    ),
            }
        )


branch_df = pd.DataFrame(
    branch_records
)


branch_df.to_csv(
    BRANCH_FILE,
    index=False,
)


# ============================================================
# 9. NAIVE TREE-SIZE DIAGNOSTIC
#
# We calculate the product of daily branch counts in log10
# space along this ONE observed trajectory.
#
# Because requests are endogenous, this is NOT the exact
# complete search-tree size and must never be reported as a
# mathematical bound.
# ============================================================

branch_summary = {}


for quantum in (
    DISCRETIZATIONS_MM
):

    qdf = branch_df[
        np.isclose(
            branch_df[
                "Quantum_mm"
            ],
            quantum,
        )
    ].copy()


    counts = qdf[
        "Feasible_daily_allocations"
    ].to_numpy(
        dtype=np.float64
    )


    if len(
        counts
    ) == 0:

        continue


    safe_counts = np.maximum(
        counts,
        1.0,
    )


    log10_tree = float(
        np.sum(
            np.log10(
                safe_counts
            )
        )
    )


    branch_summary[
        str(
            quantum
        )
    ] = {

        "decision_days":
            int(
                len(
                    qdf
                )
            ),

        "median_daily_branches":
            float(
                np.median(
                    counts
                )
            ),

        "mean_daily_branches":
            float(
                np.mean(
                    counts
                )
            ),

        "max_daily_branches":
            int(
                np.max(
                    counts
                )
            ),

        "trajectory_log10_naive_tree_size":
            log10_tree,
    }


# ============================================================
# 10. AQUACROP STATE BRANCHING TEST
#
# Goal:
# determine whether a complete environment can be deep-copied
# at an intermediate state and then branched reproducibly.
#
# First:
#     same state + same action -> same next state
#
# Second:
#     same state + different actions -> independent branches
# ============================================================

branch_env = (
    final53j.FinalTestEnv(
        climate=CLIMATE,
        year=YEAR,
        scarcity_fraction=SCARCITY,
        seed=SEED,
    )
)


branch_obs, _ = (
    branch_env.reset(
        seed=SEED
    )
)


# ------------------------------------------------------------
# Advance to a useful intermediate state.
#
# We try to reach a state with positive irrigation request so
# that the branch test is informative.
# ------------------------------------------------------------

branch_state_found = False

max_search_steps = 120


for _ in range(
    max_search_steps
):

    if (
        branch_env
        .episode_finished
    ):

        break


    req_dict = (
        final53j
        .current_requests(
            branch_env
        )
    )


    req_vector = np.asarray(
        [
            req_dict[
                name
            ]
            for name
            in FIELD_NAMES
        ],
        dtype=np.float64,
    )


    # We prefer a state with at least two requesting fields.
    if (
        np.sum(
            req_vector
            >
            EPS
        )
        >=
        2
    ):

        branch_state_found = True

        break


    (
        branch_obs,
        _,
        term,
        trunc,
        _,
    ) = branch_env.step(
        np.ones(
            4,
            dtype=np.float32,
        )
    )


    if (
        term
        or trunc
    ):

        break


# If no multi-request state was found, the current state is
# still usable for the deterministic copy test.

branch_requests_dict = (
    final53j
    .current_requests(
        branch_env
    )
)


branch_requests = np.asarray(
    [
        branch_requests_dict[
            name
        ]
        for name
        in FIELD_NAMES
    ],
    dtype=np.float64,
)


# ============================================================
# 11. DEEP-COPY SPEED
# ============================================================

copy_start = (
    time.perf_counter()
)


env_a = copy.deepcopy(
    branch_env
)


deepcopy_seconds = (
    time.perf_counter()
    -
    copy_start
)


env_b = copy.deepcopy(
    branch_env
)


# ============================================================
# 12. SAME-STATE / SAME-ACTION REPRODUCIBILITY
# ============================================================

test_action = np.asarray(
    [
        0.25,
        0.50,
        0.75,
        1.00,
    ],
    dtype=np.float32,
)


(
    oa,
    ra,
    ta,
    tra,
    ia,
) = env_a.step(
    test_action
)


(
    ob,
    rb,
    tb,
    trb,
    ib,
) = env_b.step(
    test_action
)


allocations_a = (
    allocation_vector(
        info=ia,
        field_names=FIELD_NAMES,
    )
)


allocations_b = (
    allocation_vector(
        info=ib,
        field_names=FIELD_NAMES,
    )
)


observation_error = float(
    np.max(
        np.abs(
            np.asarray(
                oa,
                dtype=np.float64,
            )
            -
            np.asarray(
                ob,
                dtype=np.float64,
            )
        )
    )
)


allocation_error = float(
    np.max(
        np.abs(
            allocations_a
            -
            allocations_b
        )
    )
)


reward_error = float(
    abs(
        float(
            ra
        )
        -
        float(
            rb
        )
    )
)


termination_match = bool(
    ta
    ==
    tb
)


truncation_match = bool(
    tra
    ==
    trb
)


branch_reproduction_pass = bool(
    observation_error
    <=
    1e-12
    and
    allocation_error
    <=
    1e-12
    and
    reward_error
    <=
    1e-12
    and
    termination_match
    and
    truncation_match
)


# ============================================================
# 13. DIFFERENT-ACTION BRANCH TEST
# ============================================================

env_c = copy.deepcopy(
    branch_env
)


env_d = copy.deepcopy(
    branch_env
)


# Choose deliberately different actions.

action_c = np.asarray(
    [
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


action_d = np.asarray(
    [
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)


(
    oc,
    rc,
    tc,
    trc,
    ic,
) = env_c.step(
    action_c
)


(
    od,
    rd,
    td,
    trd,
    id_,
) = env_d.step(
    action_d
)


allocation_c = (
    allocation_vector(
        info=ic,
        field_names=FIELD_NAMES,
    )
)


allocation_d = (
    allocation_vector(
        info=id_,
        field_names=FIELD_NAMES,
    )
)


different_branch_allocation_distance = float(
    np.sum(
        np.abs(
            allocation_c
            -
            allocation_d
        )
    )
)


different_branch_observation_distance = float(
    np.sum(
        np.abs(
            np.asarray(
                oc,
                dtype=np.float64,
            )
            -
            np.asarray(
                od,
                dtype=np.float64,
            )
        )
    )
)


different_branch_detected = bool(
    (
        different_branch_allocation_distance
        >
        1e-12
    )
    or
    (
        different_branch_observation_distance
        >
        1e-12
    )
)


# ============================================================
# 14. ROUGH NAIVE EXHAUSTIVE-RUNTIME SCALE
#
# This uses:
#
#   observed mean AquaCrop step time
#
# multiplied by:
#
#   the naive trajectory branch-count product.
#
# Again, this is NOT a formal complexity bound.
# ============================================================

seconds_per_step = float(
    episode_runtime
    /
    max(
        n_steps,
        1,
    )
)


runtime_estimates = {}


for (
    quantum_string,
    stats,
) in branch_summary.items():

    log10_nodes = float(
        stats[
            "trajectory_log10_naive_tree_size"
        ]
    )


    log10_seconds = (
        log10_nodes
        +
        math.log10(
            max(
                seconds_per_step,
                1e-15,
            )
        )
    )


    seconds_per_year = (
        365.25
        *
        24.0
        *
        3600.0
    )


    log10_years = (
        log10_seconds
        -
        math.log10(
            seconds_per_year
        )
    )


    runtime_estimates[
        quantum_string
    ] = {

        "log10_naive_seconds":
            float(
                log10_seconds
            ),

        "log10_naive_years":
            float(
                log10_years
            ),
    }


# ============================================================
# 15. STATE SERIALIZATION DIAGNOSTIC
#
# Serialized size provides a rough indication of how expensive
# it would be to store many AquaCrop states in an exact search
# or dynamic-programming method.
# ============================================================

pickle_success = False

serialized_state_bytes = None

pickle_error = None


try:

    state_bytes = pickle.dumps(
        branch_env,
        protocol=(
            pickle
            .HIGHEST_PROTOCOL
        ),
    )


    serialized_state_bytes = int(
        len(
            state_bytes
        )
    )


    pickle_success = True


except Exception as exc:

    pickle_error = repr(
        exc
    )


# ============================================================
# 16. SIMPLE FEASIBILITY FLAGS
#
# These are diagnostics, NOT mathematical proofs.
# ============================================================

max_log10_tree = None


if branch_summary:

    max_log10_tree = float(
        max(
            item[
                "trajectory_log10_naive_tree_size"
            ]
            for item
            in branch_summary.values()
        )
    )


naive_exhaustive_search_obviously_intractable = bool(
    max_log10_tree is not None
    and
    max_log10_tree
    >
    20.0
)


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

total_runtime = (
    time.perf_counter()
    -
    script_start
)


summary = {

    "milestone":
        "56A",

    "purpose":
        (
            "Assess feasibility of obtaining a genuinely "
            "globally optimal offline AquaCrop irrigation "
            "benchmark."
        ),

    "case": {

        "climate":
            CLIMATE,

        "year":
            int(
                YEAR
            ),

        "scarcity_fraction":
            float(
                SCARCITY
            ),

        "daily_capacity_mm":
            float(
                DAILY_CAPACITY
            ),

        "seasonal_budget_mm":
            float(
                env.seasonal_budget
            ),
    },

    "episode": {

        "steps":
            int(
                n_steps
            ),

        "request_days":
            int(
                request_days
            ),

        "competition_days":
            int(
                competition_days
            ),

        "runtime_seconds":
            float(
                episode_runtime
            ),

        "seconds_per_step":
            float(
                seconds_per_step
            ),
    },

    "request_field_distribution": {

        str(
            int(
                key
            )
        ):
            int(
                value
            )

        for key, value
        in request_field_distribution.items()
    },

    "branching":
        branch_summary,

    "naive_runtime":
        runtime_estimates,

    "state_branching": {

        "branch_state_found_with_at_least_two_requests":
            bool(
                branch_state_found
            ),

        "branch_state_requests_mm":
            [
                float(
                    x
                )
                for x
                in branch_requests
            ],

        "deepcopy_seconds":
            float(
                deepcopy_seconds
            ),

        "same_action_observation_error":
            float(
                observation_error
            ),

        "same_action_allocation_error_mm":
            float(
                allocation_error
            ),

        "same_action_reward_error":
            float(
                reward_error
            ),

        "same_action_reproduction_pass":
            bool(
                branch_reproduction_pass
            ),

        "different_action_allocation_L1_mm":
            float(
                different_branch_allocation_distance
            ),

        "different_action_observation_L1":
            float(
                different_branch_observation_distance
            ),

        "different_branch_detected":
            bool(
                different_branch_detected
            ),
    },

    "serialization": {

        "pickle_success":
            bool(
                pickle_success
            ),

        "serialized_state_bytes":
            serialized_state_bytes,

        "serialized_state_MB":
            (
                float(
                    serialized_state_bytes
                    /
                    1024.0
                    /
                    1024.0
                )
                if
                serialized_state_bytes
                is not None
                else
                None
            ),

        "pickle_error":
            pickle_error,
    },

    "diagnostic_flags": {

        "naive_exhaustive_search_obviously_intractable":
            bool(
                naive_exhaustive_search_obviously_intractable
            ),

        "note":
            (
                "This flag is based only on branching observed "
                "along one trajectory and is not a formal "
                "complexity proof."
            ),
    },

    "integrity": {

        "optimization_performed":
            False,

        "ppo_training_performed":
            False,

        "ppo_inference_performed":
            False,

        "controller_modified":
            False,

        "hypothesis_tests_performed":
            False,

        "final_results_modified":
            False,
    },

    "script_runtime_seconds":
        float(
            total_runtime
        ),
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# 18. TERMINAL REPORT
# ============================================================

print()

print(
    "=" * 78
)

print(
    "EPISODE STRUCTURE"
)

print(
    "=" * 78
)


print(
    f"Simulation steps              : "
    f"{n_steps}"
)


print(
    f"Request days                  : "
    f"{request_days}"
)


print(
    f"Competition days              : "
    f"{competition_days}"
)


print(
    f"One episode runtime           : "
    f"{episode_runtime:.3f} s"
)


print(
    f"Mean simulation time / step   : "
    f"{1000 * seconds_per_step:.3f} ms"
)


print()

print(
    "Number of requesting fields "
    "on request days:"
)


if len(
    request_field_distribution
) > 0:

    print(
        request_field_distribution
        .to_string()
    )

else:

    print(
        "No request days."
    )


# ============================================================
# 19. BRANCHING REPORT
# ============================================================

print()

print(
    "=" * 78
)

print(
    "DISCRETIZED DAILY BRANCHING"
)

print(
    "=" * 78
)


for quantum in (
    DISCRETIZATIONS_MM
):

    key = str(
        quantum
    )


    if key not in (
        branch_summary
    ):

        continue


    s = branch_summary[
        key
    ]


    r = runtime_estimates[
        key
    ]


    print()

    print(
        f"Water quantum: "
        f"{quantum:.1f} mm"
    )


    print(
        f"  Decision days             : "
        f"{s['decision_days']}"
    )


    print(
        f"  Median daily branches     : "
        f"{s['median_daily_branches']:.1f}"
    )


    print(
        f"  Mean daily branches       : "
        f"{s['mean_daily_branches']:.1f}"
    )


    print(
        f"  Maximum daily branches    : "
        f"{s['max_daily_branches']:,}"
    )


    print(
        f"  log10 naive tree size     : "
        f"{s['trajectory_log10_naive_tree_size']:.2f}"
    )


    print(
        f"  log10 naive runtime years : "
        f"{r['log10_naive_years']:.2f}"
    )


# ============================================================
# 20. STATE BRANCHING REPORT
# ============================================================

print()

print(
    "=" * 78
)

print(
    "AQUACROP STATE BRANCHING TEST"
)

print(
    "=" * 78
)


print(
    f"Multi-request state found     : "
    f"{'YES' if branch_state_found else 'NO'}"
)


print(
    "Requests at branch state     : "
    +
    np.array2string(
        branch_requests,
        precision=3,
        separator=", ",
    )
    +
    " mm"
)


print(
    f"Deep-copy time                : "
    f"{1000 * deepcopy_seconds:.3f} ms"
)


print(
    f"Same-action observation error : "
    f"{observation_error:.3e}"
)


print(
    f"Same-action allocation error  : "
    f"{allocation_error:.3e} mm"
)


print(
    f"Same-action reward error      : "
    f"{reward_error:.3e}"
)


print(
    f"Branch reproduction           : "
    f"{'PASS' if branch_reproduction_pass else 'FAIL'}"
)


print(
    f"Different-action allocation "
    f"distance: "
    f"{different_branch_allocation_distance:.6f} mm"
)


print(
    f"Different-action observation "
    f"distance: "
    f"{different_branch_observation_distance:.6f}"
)


print(
    f"Independent branch detected   : "
    f"{'YES' if different_branch_detected else 'NO'}"
)


# ============================================================
# 21. SERIALIZATION REPORT
# ============================================================

print()

print(
    "=" * 78
)

print(
    "STATE SERIALIZATION"
)

print(
    "=" * 78
)


print(
    f"Pickle serialization          : "
    f"{'PASS' if pickle_success else 'FAIL'}"
)


if (
    serialized_state_bytes
    is not None
):

    print(
        f"Serialized environment size   : "
        f"{serialized_state_bytes / 1024 / 1024:.3f} MB"
    )

else:

    print(
        f"Serialization error           : "
        f"{pickle_error}"
    )


# ============================================================
# 22. PRELIMINARY FEASIBILITY REPORT
# ============================================================

print()

print(
    "=" * 78
)

print(
    "PRELIMINARY GLOBAL-OPTIMALITY FEASIBILITY"
)

print(
    "=" * 78
)


if (
    naive_exhaustive_search_obviously_intractable
):

    print(
        "Naive exhaustive discretized search: "
        "APPARENTLY INTRACTABLE"
    )

else:

    print(
        "Naive exhaustive discretized search: "
        "NOT YET RULED OUT BY THIS DIAGNOSTIC"
    )


print(
    "NOTE: this is a computational diagnostic, "
    "not a proof of impossibility."
)


# ============================================================
# 23. INTEGRITY
# ============================================================

print()

print(
    "=" * 78
)

print(
    "INTEGRITY"
)

print(
    "=" * 78
)


print(
    "Optimization performed       : NO"
)

print(
    "PPO training performed       : NO"
)

print(
    "PPO inference performed      : NO"
)

print(
    "Controller modified          : NO"
)

print(
    "Hypothesis tests             : NO"
)

print(
    "Frozen final results changed : NO"
)


print()

print(
    f"Total 56A runtime             : "
    f"{total_runtime:.2f} s"
)


print()

print(
    "Saved:"
)


for path in [
    STEP_FILE,
    BRANCH_FILE,
    SUMMARY_FILE,
]:

    print(
        f"  {path}"
    )


print()

print(
    "=" * 78
)

print(
    "OPTIMIZATION FEASIBILITY ASSESSMENT COMPLETE"
)

print(
    "=" * 78
)