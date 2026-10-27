"""
Generates the README figures in docs/figures/.

The two diagrams are drawn directly. The two charts come from a fresh training
run of the agents in run_demo.py (3 episodes on the sales data, then 3 on the
marketing data), so every plotted value is measured, not typed in.

Usage (from the repository root):
    python scripts/make_figures.py
"""

import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_demo import MultiAgentCoordinator  # noqa: E402

OUT_DIR = "docs/figures"
DPI = 200

# Palette: categorical slots 1-3 of the reference data-viz palette (validated
# for colour-vision deficiency in every pairing), plus neutral text/surfaces.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
BORDER = "#cfcec8"
AGENTS = {
    "statistical": ("Statistical", "#2a78d6"),
    "geographic": ("Geographic", "#eb6834"),
    "temporal": ("Temporal", "#1baf7a"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_2,
    "axes.edgecolor": BORDER,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# --------------------------------------------------------------------------- #
# Diagram helpers
# --------------------------------------------------------------------------- #
def box(ax, x, y, w, h, title, lines=(), edge=BORDER, fill="white", title_color=TEXT):
    """Rounded box centred on (x, y). Returns its (left, right, bottom, top)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1.6, edgecolor=edge, facecolor=fill))
    if lines:
        ax.text(x, y + h / 2 - 0.32, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=title_color)
        ax.text(x, y + h / 2 - 0.62, "\n".join(lines), ha="center", va="top",
                fontsize=9, color=TEXT_2, linespacing=1.5)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=title_color)
    return x - w / 2, x + w / 2, y - h / 2, y + h / 2


def arrow(ax, start, end, color=TEXT_2, style="-|>", rad=0.0, lw=1.6):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}"))


def new_canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- #
# Figure 1: architecture
# --------------------------------------------------------------------------- #
def architecture():
    fig, ax = new_canvas(13, 6.6)
    ax.set_ylim(-0.1, 6.6)
    ax.text(0.4, 6.25, "System architecture", fontsize=15, fontweight="bold", va="center")
    ax.text(0.4, 5.85, "Each agent decides whether the data suits its specialty, runs its analysis, "
            "shares the result and learns from the reward.", fontsize=10, color=TEXT_2, va="center")

    # Column 1: data
    d = box(ax, 1.55, 3.0, 2.3, 1.7, "Business data",
            ["Sales  10,000 rows", "Marketing  5,000 rows", "HR  2,000 rows"])
    # Column 2: context
    c = box(ax, 4.45, 3.0, 2.5, 1.7, "Context",
            ["dataset size", "numeric / categorical cols", "region or date columns?"])
    arrow(ax, (d[1], 3.0), (c[0], 3.0))

    # Column 3: agents
    ys = [4.45, 3.0, 1.55]
    agent_boxes = []
    for (key, (name, color)), y in zip(AGENTS.items(), ys):
        b = box(ax, 7.55, y, 2.7, 1.15, f"{name} agent", ["Q-value per context"],
                edge=color, title_color=color)
        agent_boxes.append(b)
        arrow(ax, (c[1], 3.0), (b[0], y), rad=0.0)

    # Message bus beside the agents
    ax.add_patch(FancyBboxPatch((9.05, 1.0), 0.32, 4.0,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=0, facecolor=GRID))
    ax.text(9.21, 3.0, "shared insights", rotation=90, ha="center", va="center",
            fontsize=9, color=TEXT_2)
    for b, y in zip(agent_boxes, ys):
        ax.plot([b[1], 9.05], [y, y], color=MUTED, lw=1.4, solid_capstyle="butt")

    # Column 4: coordinator, Column 5: output
    k = box(ax, 10.75, 3.9, 2.0, 1.6, "Coordinator",
            ["picks agents", "combines results"])
    o = box(ax, 10.75, 1.75, 2.0, 1.3, "Insights",
            ["findings and", "recommendations"])
    arrow(ax, (9.37, 3.9), (k[0], 3.9))
    arrow(ax, (10.75, k[2]), (10.75, o[3]))

    # Learning loop under the agents
    arrow(ax, (agent_boxes[2][0] + 0.3, agent_boxes[2][2]),
          (agent_boxes[2][1] - 0.3, agent_boxes[2][2]), rad=0.55, color=MUTED, lw=1.2)
    ax.text(7.55, 0.12, "reward  r  →  Q ← Q + α (r − Q)", ha="center", va="center",
            fontsize=9.5, color=TEXT_2, family="DejaVu Sans Mono")

    fig.savefig(f"{OUT_DIR}/01-architecture.png", dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: learning loop
# --------------------------------------------------------------------------- #
def learning_loop():
    fig, ax = new_canvas(13, 3.3)
    ax.text(0.4, 3.0, "One coordination episode", fontsize=15, fontweight="bold", va="center")

    steps = [
        ("1  Observe", ["extract the dataset's", "context"]),
        ("2  Decide", ["each agent checks if", "the data fits its specialty"]),
        ("3  Analyse", ["correlations, regions", "or seasonality"]),
        ("4  Share", ["send the insight to", "the other agents"]),
        ("5  Learn", ["reward the result,", "update the Q-value"]),
    ]
    w, h, gap = 2.15, 1.35, 0.42
    x0 = 0.4 + w / 2
    boxes = []
    for i, (title, lines) in enumerate(steps):
        boxes.append(box(ax, x0 + i * (w + gap), 1.65, w, h, title, lines))
    for a, b in zip(boxes, boxes[1:]):
        arrow(ax, (a[1], 1.65), (b[0], 1.65))
    # Return arrow: next episode
    arrow(ax, (boxes[-1][0] + 0.5, boxes[-1][2]), (boxes[0][1] - 0.5, boxes[0][2]),
          rad=-0.12, color=MUTED, lw=1.2)
    ax.text(6.5, 0.2, "next episode, with updated Q-values", ha="center", va="center",
            fontsize=9, color=TEXT_2)

    fig.savefig(f"{OUT_DIR}/02-learning-loop.png", dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Training run for the charts
# --------------------------------------------------------------------------- #
def train():
    """Run 3 episodes per dataset and record each agent's reward and Q-value per episode."""
    with contextlib.redirect_stdout(io.StringIO()):
        coordinator = MultiAgentCoordinator()
    rows = []
    episode = 0
    for dataset in ("sales", "marketing"):
        data = pd.read_csv(f"data/{dataset}_data.csv")
        for _ in range(3):
            episode += 1
            before = {k: len(a.learning_history) for k, a in coordinator.agents.items()}
            with contextlib.redirect_stdout(io.StringIO()):
                coordinator.coordinate_analysis(data)
            for key, agent in coordinator.agents.items():
                if len(agent.learning_history) > before[key]:
                    last = agent.learning_history[-1]
                    rows.append({"episode": episode, "dataset": dataset, "agent": key,
                                 "reward": last["reward"], "q_value": last["q_value"]})
    return pd.DataFrame(rows), coordinator


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def shade_datasets(ax, top):
    ax.axvspan(0.5, 3.5, color="#f3f2ee", zorder=0, linewidth=0)
    ax.text(2.0, top, "sales data", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.text(5.0, top, "marketing data", ha="center", va="bottom", fontsize=9, color=MUTED)


# --------------------------------------------------------------------------- #
# Figure 3: learning curves (two panels, one shared x-axis, no dual y-axis)
# --------------------------------------------------------------------------- #
def learning_curves(history):
    fig, (ax_q, ax_r) = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Learning over 6 episodes", x=0.065, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.065, 0.885, "Measured from a fresh run of the agents in run_demo.py. Q-values are "
             "kept per dataset, so lines restart on the marketing data. Geographic and Temporal overlap.",
             fontsize=10, color=TEXT_2)

    q_top = history["q_value"].max() * 1.18
    r_top = history["reward"].max() * 1.18
    for ax, col, title, top in ((ax_q, "q_value", "Q-value", q_top),
                                (ax_r, "reward", "Reward per episode", r_top)):
        style_axes(ax)
        shade_datasets(ax, top * 0.97)
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0, top * 1.06)
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Episode")
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=TEXT, pad=10)

        # Draw the overlapping pair first so Statistical sits on top. Each agent
        # keeps a separate Q-value per context, so the line breaks between datasets.
        for key in ("temporal", "geographic", "statistical"):
            name, color = AGENTS[key]
            for i, (_, s) in enumerate(history[history["agent"] == key].groupby("dataset", sort=False)):
                ax.plot(s["episode"], s[col], color=color, lw=2, marker="o", ms=7,
                        markeredgecolor=SURFACE, markeredgewidth=2,
                        label=name if i == 0 else None, zorder=3)

        # Selective direct labels: the last point of each distinct line
        stat = history[history["agent"] == "statistical"].iloc[-1]
        ax.annotate(f"Statistical  {stat[col]:.2f}", (stat["episode"], stat[col]),
                    xytext=(-8, 10), textcoords="offset points", ha="right",
                    fontsize=9.5, color=TEXT)
        pair = history[history["agent"] == "geographic"].iloc[-1]
        offset, align = ((10, -26), "left") if col == "q_value" else ((0, -16), "center")
        ax.annotate(f"Geographic & Temporal  {pair[col]:.2f}", (pair["episode"], pair[col]),
                    xytext=offset, textcoords="offset points", ha=align,
                    fontsize=9.5, color=TEXT)

    handles, labels = ax_q.get_legend_handles_labels()
    order = [labels.index(n) for n in ("Statistical", "Geographic", "Temporal")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(f"{OUT_DIR}/03-learning-curves.png", dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: total reward per agent
# --------------------------------------------------------------------------- #
def agent_summary(coordinator):
    rows = []
    for key in ("statistical", "geographic", "temporal"):
        a = coordinator.agents[key]
        rows.append((AGENTS[key][0], AGENTS[key][1], a.total_reward, len(a.learning_history)))
    total = sum(r[2] for r in rows)

    fig, ax = plt.subplots(figsize=(13, 3.2))
    fig.suptitle("Total reward by agent", x=0.065, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.065, 0.83, f"System total {total:.1f} over 6 episodes. Statistical acts on both "
             "datasets; the other two only on the sales data.", fontsize=10, color=TEXT_2)
    y = range(len(rows))[::-1]
    ax.barh(list(y), [r[2] for r in rows], color=[r[1] for r in rows], height=0.56,
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), [r[0] for r in rows], fontsize=10.5, color=TEXT)
    for yi, (_, _, reward, n) in zip(y, rows):
        ax.text(reward + total * 0.008, yi, f"{reward:.1f}   ({n} episodes)",
                va="center", fontsize=10, color=TEXT)
    ax.set_xlim(0, max(r[2] for r in rows) * 1.25)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(f"{OUT_DIR}/04-reward-by-agent.png", dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    architecture()
    learning_loop()
    history, coordinator = train()
    learning_curves(history)
    agent_summary(coordinator)
    print(history.to_string(index=False))
    print(f"Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
