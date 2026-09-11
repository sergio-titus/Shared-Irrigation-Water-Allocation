# ============================================================
# PUBLICATION FIGURE GENERATION
# FINAL PORTRAIT-ORIENTED MANUSCRIPT FIGURES
#
# NO AquaCrop simulation
# NO PPO training
# NO PPO inference
# NO controller modification
# NO new statistical testing
#
# Uses frozen outputs from:
#   53F horizon selection
#   53J final evaluation
#   53K statistics
#   53L diagnostics
#
# Main changes relative to 54B:
# - Portrait-oriented figures
# - Old-style line/marker presentation
# - Equal, Priority and PPO shown together
# - 100%, 60%, 40% all retained
# - Year-by-year 2019-2025 results shown
# - Three climates retained within the same figures
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("results") / "multiclimate_extension"

WEATHER_DIR = ROOT / "weather"
HORIZON_DIR = ROOT / "horizon_reconfirmation"
EVAL_DIR = ROOT / "final_evaluation"
STATS_DIR = ROOT / "final_statistics"
DIAG_DIR = ROOT / "final_diagnostics"

OUT_DIR = ROOT / "final_figures_portrait"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIMATE_FILE = (
    WEATHER_DIR /
    "climate_contrast_summary.csv"
)

HORIZON_FILE = (
    HORIZON_DIR /
    "ppo_horizon_selection_horizon_summary.csv"
)

PANEL_FILE = (
    EVAL_DIR /
    "controller_evaluation_final_controller_panel.csv"
)

OVERALL_FILE = (
    STATS_DIR /
    "statistical_analysis_descriptive_summary.csv"
)

PPO_SEED_FILE = (
    EVAL_DIR /
    "controller_evaluation_ppo_seed_robustness_40pct.csv"
)

DIAG_FILE = (
    DIAG_DIR /
    "controller_diagnostics_climate_controller_diagnostics.csv"
)

LATENCY_FILE = (
    DIAG_DIR /
    "controller_diagnostics_decision_latency_summary.csv"
)

INTEGRITY_FILE = (
    OUT_DIR /
    "publication_figures_portrait_figure_integrity.json"
)


# ============================================================
# 2. CHECK INPUTS
# ============================================================

required = [
    CLIMATE_FILE,
    HORIZON_FILE,
    PANEL_FILE,
    OVERALL_FILE,
    PPO_SEED_FILE,
    DIAG_FILE,
    LATENCY_FILE,
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required input: {p}"
        )


# ============================================================
# 3. LOAD DATA
# ============================================================

climate = pd.read_csv(CLIMATE_FILE)
horizon = pd.read_csv(HORIZON_FILE)
panel = pd.read_csv(PANEL_FILE)
overall = pd.read_csv(OVERALL_FILE)
ppo_seeds = pd.read_csv(PPO_SEED_FILE)
diag = pd.read_csv(DIAG_FILE)
latency = pd.read_csv(LATENCY_FILE)


# ============================================================
# 4. CONSTANTS
# ============================================================

CONTROLLERS = [
    "Equal",
    "Priority",
    "Optimized PPO",
]

DISPLAY = {
    "Equal": "Equal",
    "Priority": "Priority",
    "Optimized PPO": "PPO",
}

CLIMATES = [
    "Cotonou",
    "Niamey",
    "Tunis",
]

SCARCITIES = [
    100.0,
    60.0,
    40.0,
]

YEARS = list(
    range(2019, 2026)
)


# ============================================================
# 5. GLOBAL STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "font.size": 9.5,

    "axes.labelsize": 10,

    "axes.titlesize": 10.5,

    "xtick.labelsize": 8.5,

    "ytick.labelsize": 8.5,

    "legend.fontsize": 9,

    "figure.titlesize": 12,

    "axes.spines.top": False,

    "axes.spines.right": False,

    "axes.grid": True,

    "grid.alpha": 0.22,

    "grid.linewidth": 0.6,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,
})


# ============================================================
# 6. HELPERS
# ============================================================

def save_figure(fig, filename):

    png = OUT_DIR / f"{filename}.png"
    pdf = OUT_DIR / f"{filename}.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def panel_label(ax, text):

    ax.text(
        -0.10,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def controller_line(
    ax,
    x,
    y,
    controller,
):

    marker_map = {
        "Equal": "o",
        "Priority": "^",
        "Optimized PPO": "s",
    }

    ax.plot(
        x,
        y,
        marker=marker_map[controller],
        linewidth=1.7,
        markersize=4.8,
        label=DISPLAY[controller],
    )


# ============================================================
# FIGURE 1
# CLIMATE CONTRAST — PORTRAIT
# ============================================================

print()
print("=" * 80)
print("FIGURE 1 — CLIMATE CONTRAST")
print("=" * 80)

cl = (
    climate
    .set_index("Site")
    .loc[CLIMATES]
    .reset_index()
)

x = np.arange(
    len(CLIMATES)
)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(6.5, 9.0),
)


# ------------------------------------------------------------
# A. Rainfall
# ------------------------------------------------------------

ax = axes[0]

ax.bar(
    x,
    cl["MeanAnnualRain_mm"],
)

ax.errorbar(
    x,
    cl["MeanAnnualRain_mm"],
    yerr=cl["SDAnnualRain_mm"],
    fmt="none",
    capsize=4,
    linewidth=1.0,
)

ax.set_xticks(x)
ax.set_xticklabels(CLIMATES)

ax.set_ylabel(
    "Annual rainfall (mm)"
)

ax.set_title(
    "Annual rainfall"
)

panel_label(ax, "a")


# ------------------------------------------------------------
# B. ET0
# ------------------------------------------------------------

ax = axes[1]

ax.bar(
    x,
    cl["MeanAnnualET0_mm"],
)

ax.set_xticks(x)
ax.set_xticklabels(CLIMATES)

ax.set_ylabel(
    "Reference ET$_0$ (mm)"
)

ax.set_title(
    "Atmospheric water demand"
)

panel_label(ax, "b")


# ------------------------------------------------------------
# C. Rain / ET0
# ------------------------------------------------------------

ax = axes[2]

ax.bar(
    x,
    cl["MeanRain_ET0_ratio"],
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
)

ax.set_xticks(x)
ax.set_xticklabels(CLIMATES)

ax.set_ylabel(
    "Rainfall / ET$_0$"
)

ax.set_title(
    "Climatic water balance"
)

panel_label(ax, "c")


fig.suptitle(
    "Contrasting climatic conditions used in the multi-climate experiment",
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)

save_figure(
    fig,
    "Fig01_climate_contrast_portrait",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# PPO TRAINING-HORIZON SELECTION — PORTRAIT
# ============================================================

print()
print("=" * 80)
print("FIGURE 2 — PPO TRAINING-HORIZON SELECTION")
print("=" * 80)

horizon = horizon.sort_values(
    "Checkpoint"
).copy()

steps = horizon[
    "Checkpoint"
].to_numpy()

macro_mean = horizon[
    "Mean_macro_yield_retention_pct"
].to_numpy()

macro_se = horizon[
    "SE_macro_yield_retention_pct"
].to_numpy()

selected_horizon = 102400

best_position = np.argmax(
    macro_mean
)

one_se_threshold = (
    macro_mean[best_position]
    -
    macro_se[best_position]
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(6.8, 8.5),
)


# ------------------------------------------------------------
# A. Macro validation
# ------------------------------------------------------------

ax = axes[0]

ax.errorbar(
    steps,
    macro_mean,
    yerr=macro_se,
    marker="o",
    linewidth=1.7,
    capsize=4,
)

ax.axhline(
    one_se_threshold,
    linestyle="--",
    linewidth=1.2,
    label="One-SE threshold",
)

ax.axvline(
    selected_horizon,
    linestyle=":",
    linewidth=1.4,
    label="Selected: 102,400",
)

eligible = (
    horizon["One_SE_eligible"]
    .astype(bool)
)

ax.scatter(
    horizon.loc[
        eligible,
        "Checkpoint",
    ],
    horizon.loc[
        eligible,
        "Mean_macro_yield_retention_pct",
    ],
    s=80,
    facecolors="none",
    linewidths=1.4,
    label="One-SE eligible",
)

ax.set_xticks(steps)

ax.set_xticklabels(
    [
        f"{int(v / 1000)}k"
        for v in steps
    ]
)

ax.set_xlabel(
    "PPO training timesteps"
)

ax.set_ylabel(
    "Macro-climate yield retention (%)"
)

ax.set_title(
    "Development-fold validation performance"
)

ax.legend(
    frameon=False,
)

panel_label(ax, "a")


# ------------------------------------------------------------
# B. Climate-specific validation
# ------------------------------------------------------------

ax = axes[1]

climate_columns = {

    "Cotonou":
        "Mean_Cotonou_retention_pct",

    "Niamey":
        "Mean_Niamey_retention_pct",

    "Tunis":
        "Mean_Tunis_retention_pct",
}

for climate_name in CLIMATES:

    ax.plot(
        steps,
        horizon[
            climate_columns[
                climate_name
            ]
        ],
        marker="o",
        linewidth=1.7,
        label=climate_name,
    )

ax.axvline(
    selected_horizon,
    linestyle=":",
    linewidth=1.4,
)

ax.set_xticks(steps)

ax.set_xticklabels(
    [
        f"{int(v / 1000)}k"
        for v in steps
    ]
)

ax.set_xlabel(
    "PPO training timesteps"
)

ax.set_ylabel(
    "Yield retention (%)"
)

ax.set_title(
    "Climate-specific validation performance"
)

ax.legend(
    frameon=False,
)

panel_label(ax, "b")


fig.suptitle(
    "Development-only selection of the PPO training horizon",
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)

save_figure(
    fig,
    "Fig02_PPO_horizon_selection_portrait",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# OVERALL CONTROLLER COMPARISON
#
# 3 VERTICAL PANELS
# Equal / Priority / PPO together
# 100 / 60 / 40 together
# ============================================================

print()
print("=" * 80)
print("FIGURE 3 — OVERALL CONTROLLER COMPARISON")
print("=" * 80)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(6.5, 10.0),
)

metric_info = [

    (
        "Yield_retention_mean_pct",
        "Yield retention (%)",
        "Mean yield retention",
        (60, 102),
    ),

    (
        "Worst_field_retention_mean_pct",
        "Worst-field retention (%)",
        "Worst-field yield retention",
        (0, 102),
    ),

    (
        "Jain_mean",
        "Jain fairness index",
        "Allocation fairness",
        (0.85, 1.005),
    ),
]

scarcity_x = np.array(
    [100, 60, 40]
)

for i, (
    column,
    ylabel,
    title,
    ylim,
) in enumerate(metric_info):

    ax = axes[i]

    for controller in CONTROLLERS:

        y = []

        for scarcity in SCARCITIES:

            row = overall[
                (
                    overall[
                        "Scarcity_pct"
                    ] == scarcity
                )
                &
                (
                    overall[
                        "Controller"
                    ] == controller
                )
            ]

            if len(row) != 1:
                raise RuntimeError(
                    f"Missing overall row: "
                    f"{controller}, {scarcity}"
                )

            y.append(
                float(
                    row.iloc[0][column]
                )
            )

        controller_line(
            ax,
            scarcity_x,
            y,
            controller,
        )

    ax.set_xlim(
        105,
        35,
    )

    ax.set_xticks(
        scarcity_x
    )

    ax.set_xticklabels(
        [
            "100%",
            "60%",
            "40%",
        ]
    )

    ax.set_ylim(*ylim)

    ax.set_xlabel(
        "Seasonal water availability"
    )

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    panel_label(
        ax,
        chr(ord("a") + i),
    )

    if i == 0:

        ax.legend(
            frameon=False,
            ncol=3,
            loc="lower left",
        )


fig.suptitle(
    "Overall controller performance across 21 unseen climate-years",
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)

save_figure(
    fig,
    "Fig03_overall_controller_comparison_portrait",
)

plt.close(fig)


# ============================================================
# FIGURE 4
# YEAR-BY-YEAR YIELD RETENTION
#
# Rows:
#   100%
#   60%
#   40%
#
# Columns:
#   Cotonou
#   Niamey
#   Tunis
#
# Each panel:
#   Equal + Priority + PPO
#   2019-2025
#
# Portrait canvas
# ============================================================

print()
print("=" * 80)
print("FIGURE 4 — YEAR-BY-YEAR YIELD RETENTION")
print("=" * 80)

fig, axes = plt.subplots(
    3,
    3,
    figsize=(8.2, 10.8),
    sharex=True,
    sharey=True,
)

for r, scarcity in enumerate(
    SCARCITIES
):

    for c, climate_name in enumerate(
        CLIMATES
    ):

        ax = axes[r, c]

        subset = panel[
            (
                panel[
                    "Scarcity_pct"
                ] == scarcity
            )
            &
            (
                panel[
                    "Climate"
                ] == climate_name
            )
        ].copy()

        for controller in CONTROLLERS:

            d = subset[
                subset[
                    "Controller"
                ] == controller
            ].sort_values(
                "Year"
            )

            controller_line(
                ax,
                d["Year"],
                d[
                    "Yield_retention_pct"
                ],
                controller,
            )

        ax.set_ylim(
            60,
            102,
        )

        ax.set_xticks(
            YEARS
        )

        ax.tick_params(
            axis="x",
            labelrotation=45,
        )

        if r == 0:

            ax.set_title(
                climate_name,
                fontweight="bold",
            )

        if c == 0:

            ax.set_ylabel(
                f"{int(scarcity)}% availability\n"
                "Yield retention (%)"
            )

        if r == 2:

            ax.set_xlabel(
                "Weather year"
            )


# One shared legend
handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.012),
)

fig.suptitle(
    "Year-by-year yield retention across climates and water-availability levels",
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0.055, 1, 0.965]
)

save_figure(
    fig,
    "Fig04_yearly_yield_retention_all_climates_portrait",
)

plt.close(fig)


# ============================================================
# FIGURE 5
# YEAR-BY-YEAR WORST-FIELD RETENTION
# Same architecture as Figure 4
# ============================================================

print()
print("=" * 80)
print("FIGURE 5 — YEAR-BY-YEAR WORST-FIELD RETENTION")
print("=" * 80)

fig, axes = plt.subplots(
    3,
    3,
    figsize=(8.2, 10.8),
    sharex=True,
    sharey=True,
)

for r, scarcity in enumerate(
    SCARCITIES
):

    for c, climate_name in enumerate(
        CLIMATES
    ):

        ax = axes[r, c]

        subset = panel[
            (
                panel[
                    "Scarcity_pct"
                ] == scarcity
            )
            &
            (
                panel[
                    "Climate"
                ] == climate_name
            )
        ].copy()

        for controller in CONTROLLERS:

            d = subset[
                subset[
                    "Controller"
                ] == controller
            ].sort_values(
                "Year"
            )

            controller_line(
                ax,
                d["Year"],
                d[
                    "Worst_field_retention_pct"
                ],
                controller,
            )

        ax.set_ylim(
            0,
            102,
        )

        ax.set_xticks(
            YEARS
        )

        ax.tick_params(
            axis="x",
            labelrotation=45,
        )

        if r == 0:

            ax.set_title(
                climate_name,
                fontweight="bold",
            )

        if c == 0:

            ax.set_ylabel(
                f"{int(scarcity)}% availability\n"
                "Worst-field retention (%)"
            )

        if r == 2:

            ax.set_xlabel(
                "Weather year"
            )


handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.012),
)

fig.suptitle(
    "Year-by-year worst-field yield retention across climates and water-availability levels",
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0.055, 1, 0.965]
)

save_figure(
    fig,
    "Fig05_yearly_worst_field_all_climates_portrait",
)

plt.close(fig)


# ============================================================
# FIGURE 6
# RESOURCE / WATER-ALLOCATION DIAGNOSTICS
#
# Portrait:
# A Cotonou budget utilization
# B Niamey budget utilization
# C Tunis budget utilization
# D Decision latency
# ============================================================

print()
print("=" * 80)
print("FIGURE 6 — WATER-ALLOCATION DIAGNOSTICS")
print("=" * 80)

fig, axes = plt.subplots(
    4,
    1,
    figsize=(6.5, 11.0),
)

for i, climate_name in enumerate(
    CLIMATES
):

    ax = axes[i]

    for controller in CONTROLLERS:

        d = diag[
            (
                diag[
                    "Climate"
                ] == climate_name
            )
            &
            (
                diag[
                    "Controller"
                ] == controller
            )
        ].copy()

        d = (
            d
            .set_index(
                "Scarcity_pct"
            )
            .loc[SCARCITIES]
            .reset_index()
        )

        controller_line(
            ax,
            d[
                "Scarcity_pct"
            ],
            d[
                "Budget_utilization_mean_pct"
            ],
            controller,
        )

    ax.axhline(
        100,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(
        105,
        35,
    )

    ax.set_xticks(
        [100, 60, 40]
    )

    ax.set_xticklabels(
        [
            "100%",
            "60%",
            "40%",
        ]
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_xlabel(
        "Seasonal water availability"
    )

    ax.set_ylabel(
        "Budget utilization (%)"
    )

    ax.set_title(
        f"{climate_name}: seasonal water-budget utilization"
    )

    panel_label(
        ax,
        chr(ord("a") + i),
    )

    if i == 0:

        ax.legend(
            frameon=False,
            ncol=3,
            loc="upper left",
        )


# ------------------------------------------------------------
# D. Decision latency
# ------------------------------------------------------------

ax = axes[3]

lat = (
    latency
    .set_index(
        "Controller"
    )
    .loc[
        CONTROLLERS
    ]
    .reset_index()
)

x = np.arange(
    len(CONTROLLERS)
)

ax.bar(
    x,
    lat[
        "Mean_latency_ms"
    ],
)

ax.errorbar(
    x,
    lat[
        "Mean_latency_ms"
    ],
    yerr=lat[
        "SD_latency_ms"
    ],
    fmt="none",
    capsize=4,
    linewidth=1.0,
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        DISPLAY[c]
        for c in CONTROLLERS
    ]
)

ax.set_yscale(
    "log"
)

ax.set_ylabel(
    "Decision latency (ms)"
)

ax.set_title(
    "Controller decision computation"
)

panel_label(
    ax,
    "d",
)


fig.suptitle(
    "Water-resource use and controller computational requirements",
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)

save_figure(
    fig,
    "Fig06_water_allocation_diagnostics_portrait",
)

plt.close(fig)


# ============================================================
# SUPPLEMENTARY FIGURE S1
# PPO SEED ROBUSTNESS — PORTRAIT
# ============================================================

print()
print("=" * 80)
print("SUPPLEMENTARY FIGURE S1 — PPO SEED ROBUSTNESS")
print("=" * 80)

ppo_seeds = ppo_seeds.sort_values(
    "PPO_seed"
)

x = np.arange(
    len(ppo_seeds)
)

seed_labels = [
    str(int(v))
    for v in ppo_seeds[
        "PPO_seed"
    ]
]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(6.5, 9.5),
)

seed_metrics = [

    (
        "Mean_yield_retention_pct",
        "Mean yield retention (%)",
        "Yield retention",
    ),

    (
        "Worst_field_retention_mean_pct",
        "Worst-field retention (%)",
        "Worst-field protection",
    ),

    (
        "Jain_mean",
        "Jain fairness index",
        "Allocation fairness",
    ),
]

for i, (
    column,
    ylabel,
    title,
) in enumerate(seed_metrics):

    ax = axes[i]

    values = ppo_seeds[
        column
    ].to_numpy()

    ax.plot(
        x,
        values,
        marker="o",
        linestyle="none",
        markersize=6,
    )

    ax.axhline(
        values.mean(),
        linestyle="--",
        linewidth=1.2,
        label="10-seed mean",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        seed_labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    panel_label(
        ax,
        chr(ord("a") + i),
    )

    if i == 2:

        ax.set_xlabel(
            "Final PPO training seed"
        )

    if i == 0:

        ax.legend(
            frameon=False,
        )


fig.suptitle(
    "Robustness across the 10 independently trained final PPO policies at 40% water availability",
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)

save_figure(
    fig,
    "FigS1_PPO_seed_robustness_portrait",
)

plt.close(fig)


# ============================================================
# 7. INTEGRITY
# ============================================================

integrity = {

    "milestone":
        "54C",

    "status":
        "COMPLETE",

    "orientation":
        "portrait",

    "main_figures": [

        "Fig01_climate_contrast_portrait",

        "Fig02_PPO_horizon_selection_portrait",

        "Fig03_overall_controller_comparison_portrait",

        "Fig04_yearly_yield_retention_all_climates_portrait",

        "Fig05_yearly_worst_field_all_climates_portrait",

        "Fig06_water_allocation_diagnostics_portrait",
    ],

    "supplementary_figures": [

        "FigS1_PPO_seed_robustness_portrait",
    ],

    "climates": CLIMATES,

    "final_test_years": YEARS,

    "scarcity_levels_pct": [
        100,
        60,
        40,
    ],

    "controllers": CONTROLLERS,

    "aquacrop_simulation_performed":
        False,

    "ppo_training_performed":
        False,

    "ppo_inference_performed":
        False,

    "controller_modified":
        False,

    "new_statistical_tests_performed":
        False,

    "old_tunis_only_results_used":
        False,

    "frozen_results_only":
        True,
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
# 8. COMPLETE
# ============================================================

print()
print("=" * 80)
print("PUBLICATION FIGURE GENERATION COMPLETE")
print("=" * 80)

print()
print("MAIN FIGURES:")
print()
print("Fig. 1  Climate contrast")
print("Fig. 2  PPO training-horizon selection")
print("Fig. 3  Overall Equal/Priority/PPO comparison")
print("Fig. 4  Yearly yield retention:")
print("        3 climates × 3 scarcity levels")
print("Fig. 5  Yearly worst-field retention:")
print("        3 climates × 3 scarcity levels")
print("Fig. 6  Water-resource diagnostics + latency")

print()
print("SUPPLEMENTARY:")
print()
print("Fig. S1 PPO final-seed robustness")

print()
print("All figures:")
print("  - Portrait oriented")
print("  - PNG 600 dpi")
print("  - PDF vector")
print("  - Frozen results only")

print()
print(f"Output directory:")
print(f"  {OUT_DIR}")

print()
print("No simulation.")
print("No training.")
print("No inference.")
print("No controller tuning.")
print("No new hypothesis tests.")

print()
print("=" * 80)