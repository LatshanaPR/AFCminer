import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# WFCMiner - 14 Week Project Timeline
# ============================================================

# Format:
# ("Task Name", Start Week, End Week)

tasks = [
    ("Literature Survey", 1, 3),
    ("Existing Fair Clique Study", 2, 4),
    ("Problem Formulation", 3, 4),
    ("Dataset Exploration", 4, 6),

    ("System Design", 5, 5),
    ("Baseline Implementation", 5, 6),
    ("Validation against Base paper", 6, 7),
    ("Weak Fairness Implementation", 7, 9),
    ("Dynamic Attribute Tracking", 8, 10),
    ("Testing & Debugging", 9, 12),
    ("Performance Analysis", 11, 13),
    ("Final Report & Presentation", 12, 14),
]


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(16, 8))


# ============================================================
# Draw horizontal guide lines
# ============================================================

for i, (task, start, end) in enumerate(tasks):

    bar_start = start - 1

    if bar_start > 0:

        ax.hlines(
            y=i,
            xmin=0,
            xmax=bar_start,
            linestyle="--",
            linewidth=0.7,
            alpha=0.35
        )


# ============================================================
# Draw task bars
# ============================================================

for i, (task, start, end) in enumerate(tasks):

    # --------------------------------------------------------
    # Completed portion: Weeks 1–5
    # --------------------------------------------------------

    completed_start = start
    completed_end = min(end, 5)

    if completed_start <= completed_end:

        left = completed_start - 1

        width = (
            completed_end
            - completed_start
            + 1
        )

        ax.barh(
            i,
            width,
            left=left,
            height=0.55,
            color="blue"
        )


    # --------------------------------------------------------
    # Planned portion: Weeks 6–14
    # --------------------------------------------------------

    planned_start = max(start, 6)
    planned_end = end

    if planned_start <= planned_end:

        left = planned_start - 1

        width = (
            planned_end
            - planned_start
            + 1
        )

        ax.barh(
            i,
            width,
            left=left,
            height=0.55,
            color="blue",
            alpha=0.35
        )


# ============================================================
# Current position
# ============================================================

# End of Week 5 / beginning of Week 6

ax.axvline(
    x=5,
    linestyle="--",
    linewidth=1.5
)


# ============================================================
# X-axis
# ============================================================

# 14 weeks

ax.set_xlim(0, 14)

# Put labels in the center of each week

ax.set_xticks(
    [i + 0.5 for i in range(14)]
)

ax.set_xticklabels(
    [f"W{i}" for i in range(1, 15)],
    fontsize=11
)

ax.set_xlabel(
    "Project Week",
    fontsize=12
)


# ============================================================
# Y-axis
# ============================================================

ax.set_yticks(
    range(len(tasks))
)

ax.set_yticklabels(
    [task[0] for task in tasks],
    fontsize=10
)

# Week 1 at the top

ax.invert_yaxis()


# ============================================================
# Week boundary grid
# ============================================================

for x in range(15):

    ax.axvline(
        x=x,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5
    )


# ============================================================
# Title
# ============================================================

ax.set_title(
    "WFCMiner - 14 Week Project Timeline",
    fontsize=18,
    fontweight="bold",
    pad=18
)


# ============================================================
# Legend
# ============================================================

legend_elements = [

    Patch(
        color="blue",
        label="Completed (Weeks 1–5)"
    ),

    Patch(
        color="blue",
        alpha=0.35,
        label="Planned (Weeks 6–14)"
    )
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=True,
    fontsize=10
)


# ============================================================
# Layout
# ============================================================

plt.tight_layout()


# ============================================================
# Save
# ============================================================

plt.savefig(
    "WFCMiner_14_Week_Gantt.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "WFCMiner_14_Week_Gantt.pdf",
    bbox_inches="tight"
)


# ============================================================
# Display
# ============================================================

plt.show()