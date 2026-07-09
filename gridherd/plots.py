"""
plots.py - Figures for visualising demand herding and its solutions.

All figures save to figures/ at 150 dpi so they're article-ready without
extra steps. Filenames are deterministic so re-running just overwrites.

Layout decision: every figure uses a two-panel stacked layout (price on top,
demand below). This pairing is important - seeing the price curve alongside
demand makes it immediately obvious WHY agents cluster in the overnight window.
Without it you'd just see a spike with no context.

Colour assignments are fixed per strategy so the same colour always means the
same thing across figures and CLI runs:
  blue  #2a78d6 - naive (the baseline herding case)
  aqua  #1baf7a - jitter (partial fix)
  amber #eda100 - spread (full fix)
  grey  #898781 - baseline reference line (not a strategy)
"""

from __future__ import annotations

import matplotlib.animation
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridherd.market import period_labels
from gridherd.simulate import SimulationResult

FIGURES_DIR = Path("figures")
DPI = 150

_STRATEGY_COLORS = {
    "naive":  "#2a78d6",
    "jitter": "#1baf7a",
    "spread": "#eda100",
}
_BASELINE_COLOR = "#898781"
_PRICE_COLOR = "#e34948"


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)


def _apply_xticks(ax: plt.Axes, labels: list[str]) -> None:
    """Show a tick every 4 periods (every 2 hours) so the x-axis stays readable."""
    x = np.arange(len(labels))
    tick_positions = x[::4]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels[i] for i in tick_positions], rotation=45, ha="right", fontsize=9)


def _price_panel(ax: plt.Axes, prices: np.ndarray, deadline_period: int) -> None:
    """
    Draw the Agile price curve and shade the eligible charging window.

    The shaded band (00:00-07:00) is where agents are allowed to charge.
    Showing it alongside the price curve immediately explains the herding:
    the cheapest prices and the eligible window overlap perfectly overnight.
    """
    x = np.arange(len(prices))
    deadline_x = deadline_period - 0.5  # boundary sits between periods

    ax.axvspan(0, deadline_x, alpha=0.08, color="#1baf7a",
               label="Eligible window (00:00-07:00)")
    ax.axvline(deadline_x, color="#1baf7a", linewidth=1.0, linestyle=":", alpha=0.5)

    ax.plot(x, prices, color=_PRICE_COLOR, linewidth=1.8, label="Agile price (p/kWh)")
    ax.fill_between(x, prices, alpha=0.10, color=_PRICE_COLOR)

    ax.set_ylabel("Price (p/kWh)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, color="#e1e0d9")


def plot_demand(result: SimulationResult, show: bool = True) -> Path:
    """
    Phase 1 figure: stacked area showing baseline vs total demand.

    Stacked areas let you see both the absolute level and how much the agents
    are adding on top. The herding spike is the coloured layer that rises above
    the blue baseline in the overnight window.
    """
    _ensure_figures_dir()

    labels = period_labels()
    x = np.arange(48)
    strategy_display = result.strategy if result.strategy != "naive" else "Naive"

    fig, (ax_price, ax_demand) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    fig.suptitle(
        f"UK Grid Demand Herding - {result.n_agents:,} {strategy_display.title()} Agents",
        fontsize=14, fontweight="bold", y=0.99,
    )

    _price_panel(ax_price, result.prices, result.deadline_period)

    # baseline in blue, agent demand stacked on top in the strategy colour
    ax_demand.fill_between(x, 0, result.baseline_demand,
                           alpha=0.50, color="#2a78d6", label="Baseline demand")

    agent_color = _STRATEGY_COLORS.get(result.strategy, "#e67e22")
    ax_demand.fill_between(x, result.baseline_demand, result.total_demand,
                           alpha=0.70, color=agent_color,
                           label=f"Agent demand ({result.n_agents:,} {result.strategy} agents)")
    ax_demand.plot(x, result.total_demand, color=agent_color, linewidth=1.0, alpha=0.7)

    # annotate original peak (evening)
    bp = result.baseline_peak_period
    ax_demand.axvline(bp, color="#2a78d6", linestyle="--", linewidth=1.2, alpha=0.6)
    ax_demand.annotate(
        f"Original peak\n{result.baseline_peak_gw:.1f} GW @ {result.baseline_peak_time}",
        xy=(bp, result.baseline_peak_gw),
        xytext=(bp - 7, result.baseline_peak_gw + 2.5),
        fontsize=8.5, color="#1a5276",
        arrowprops=dict(arrowstyle="->", color="#1a5276", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2a78d6", alpha=0.85),
    )

    # annotate the new peak - push text right if peak is near left edge, left otherwise
    tp = result.total_peak_period
    offset_x = tp + 3 if tp < 30 else tp - 14
    ax_demand.axvline(tp, color=agent_color, linestyle="--", linewidth=1.2, alpha=0.6)
    ax_demand.annotate(
        f"New peak\n{result.total_peak_gw:.1f} GW @ {result.total_peak_time}",
        xy=(tp, result.total_peak_gw),
        xytext=(offset_x, result.total_peak_gw - 5),
        fontsize=8.5, color="#5d3a00",
        arrowprops=dict(arrowstyle="->", color="#5d3a00", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=agent_color, alpha=0.85),
    )

    ax_demand.set_ylabel("Demand (GW)", fontsize=10)
    ax_demand.set_ylim(bottom=0)
    ax_demand.legend(loc="upper right", fontsize=9)
    ax_demand.grid(axis="y", alpha=0.3, color="#e1e0d9")
    ax_demand.set_xlabel("Settlement period (half-hourly)", fontsize=10)

    _apply_xticks(ax_demand, labels)
    plt.tight_layout()

    # keep the canonical Phase 1 filename for the naive case so previously saved
    # article assets don't silently get a different name on re-run
    if result.strategy == "naive":
        out_path = FIGURES_DIR / "phase1_demand_herding.png"
    else:
        out_path = FIGURES_DIR / f"phase1_demand_{result.strategy}.png"

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_strategy_comparison(
    results: dict[str, SimulationResult],
    show: bool = True,
) -> Path:
    """
    Phase 2 figure: overlapping total-demand lines for all three strategies.

    Lines rather than stacked areas because we're comparing three totals on the
    same scale, not decomposing one total. The dashed baseline reference shows
    what the grid looks like with no agents at all.

    The stat table in the top-right corner summarises the key numbers so you
    don't have to read them off the axes.
    """
    _ensure_figures_dir()

    first = next(iter(results.values()))
    labels = period_labels()
    x = np.arange(48)

    fig, (ax_price, ax_demand) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.5]},
    )
    fig.suptitle(
        f"UK Grid Demand - Strategy Comparison ({first.n_agents:,} agents each)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    _price_panel(ax_price, first.prices, first.deadline_period)

    # dashed baseline - context, not a series
    ax_demand.plot(x, first.baseline_demand,
                   color=_BASELINE_COLOR, linewidth=1.6, linestyle="--",
                   label="Baseline (no agents)", zorder=2)

    for strategy, result in results.items():
        color = _STRATEGY_COLORS.get(strategy, "#52514e")
        ax_demand.plot(x, result.total_demand,
                       color=color, linewidth=2.2, label=strategy.title(), zorder=3)
        # subtle fill to anchor each line visually
        ax_demand.fill_between(x, first.baseline_demand, result.total_demand,
                               alpha=0.08, color=color)

        # label each peak directly so the reader doesn't have to hunt
        tp = result.total_peak_period
        offset_y = result.total_peak_gw + 0.8
        ax_demand.annotate(
            f"{result.total_peak_gw:.1f} GW\n@ {result.total_peak_time}",
            xy=(tp, result.total_peak_gw),
            xytext=(tp + 1, offset_y),
            fontsize=7.5, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    _add_stat_table(ax_demand, results, first.baseline_peak_gw)

    ax_demand.set_ylabel("Total demand (GW)", fontsize=10)
    ax_demand.set_ylim(bottom=0)
    # legend goes lower-left because the peak annotations are all in the upper region
    ax_demand.legend(loc="lower left", fontsize=9)
    ax_demand.grid(axis="y", alpha=0.3, color="#e1e0d9")
    ax_demand.set_xlabel("Settlement period (half-hourly)", fontsize=10)

    _apply_xticks(ax_demand, labels)
    plt.tight_layout()

    out_path = FIGURES_DIR / "phase2_strategy_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def _add_stat_table(
    ax: plt.Axes,
    results: dict[str, SimulationResult],
    baseline_peak_gw: float,
) -> None:
    """
    Render a peak/cost summary table as a text box inside the demand panel.

    Putting the numbers in the figure means you don't have to flip between
    the plot and the terminal to read the key results.
    """
    lines = ["Strategy    Peak (GW)   Time    Avg cost"]
    lines.append("-" * 44)
    for strategy, r in results.items():
        lines.append(
            f"{strategy:<10}  {r.total_peak_gw:>5.1f} GW  "
            f"{r.total_peak_time}  "
            f"{r.avg_cost_pence:.0f}p (£{r.avg_cost_pence/100:.2f})"
        )

    ax.text(
        0.98, 0.97, "\n".join(lines),
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c3c2b7", alpha=0.88),
    )


# ── Phase 3: feedback loop figures ───────────────────────────────────────────

def _draw_feedback_panel(
    ax: plt.Axes,
    round_result: "SimulationResult",
    baseline: np.ndarray,
    y_max: float,
    round_label: str,
) -> None:
    """Draw one round's demand curve into an axes (used by both static and animated figures)."""
    x = np.arange(48)
    ax.plot(x, baseline, color=_BASELINE_COLOR, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.plot(x, round_result.total_demand, color="#2a78d6", linewidth=1.8)
    ax.fill_between(x, baseline, round_result.total_demand, alpha=0.20, color="#2a78d6")

    tp = round_result.total_peak_period
    ax.axvline(tp, color="#2a78d6", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(
        0.02, 0.97,
        f"{round_label}\npeak {round_result.total_peak_gw:.1f} GW\n@ {round_result.total_peak_time}",
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", fontweight="bold", color="#2a78d6",
    )
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.25, color="#e1e0d9")


def plot_feedback_rounds(
    feedback: "FeedbackSimulation",
    rounds_to_show: tuple[int, ...] = (0, 1, 2, 3, 4, 9),
    show: bool = True,
) -> Path:
    """
    Phase 3 static figure: small multiples showing the spike migrating across rounds.

    Each panel is one round. Consistent y-axis limits across all panels so you can
    compare peak heights directly. The dashed baseline reference in each panel shows
    how much demand the agents are adding.

    The spike doesn't dissolve - it just moves. That's because naive agents always
    flee together to the next cheapest cluster, creating a new spike there instead.
    Price signals shift the herding destination but don't break the herding.
    """
    _ensure_figures_dir()

    labels = period_labels()
    n_rounds = len(feedback.rounds)
    indices = [i for i in rounds_to_show if i < n_rounds]

    y_max = max(r.total_demand.max() for r in feedback.rounds) * 1.08

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7), sharex=True, sharey=True)
    fig.suptitle(
        f"UK Grid Demand - Spike Migration Under Price Feedback"
        f" ({feedback.n_agents:,} naive agents, sensitivity={feedback.sensitivity} p/GW)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax_i, round_idx in enumerate(indices):
        ax = axes.flat[ax_i]
        _draw_feedback_panel(ax, feedback.rounds[round_idx], feedback.baseline, y_max,
                             f"Round {round_idx + 1}")
        _apply_xticks(ax, labels)
        ax.set_ylabel("Demand (GW)", fontsize=9)

    # hide unused panels if rounds_to_show has fewer than nrows*ncols entries
    for ax_i in range(len(indices), nrows * ncols):
        axes.flat[ax_i].set_visible(False)

    fig.text(0.5, -0.01, "Settlement period (half-hourly)", ha="center", fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / "phase3_feedback_rounds.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_feedback_animation(
    feedback: "FeedbackSimulation",
    show: bool = False,
) -> Path:
    """
    Phase 3 animated GIF - shows the herding spike migrating round by round.

    Each frame is one round of the feedback loop. The spike walks across the
    overnight window because agents flee the periods made expensive by last round's
    spike and all land in the next cheapest cluster together. Price feedback moves
    the spike around but doesn't spread it out - only the spread strategy does that.

    Saves using PillowWriter. Requires: pip install pillow
    """
    _ensure_figures_dir()

    labels = period_labels()
    x = np.arange(48)
    baseline = feedback.baseline
    y_max = max(r.total_demand.max() for r in feedback.rounds) * 1.08

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        f"UK Grid Demand Herding - Price Feedback Loop ({feedback.n_agents:,} naive agents)",
        fontsize=12, fontweight="bold",
    )

    _apply_xticks(ax, labels)
    ax.set_ylabel("Demand (GW)", fontsize=10)
    ax.set_xlabel("Settlement period (half-hourly)", fontsize=10)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.25, color="#e1e0d9")

    # static elements drawn once
    ax.plot(x, baseline, color=_BASELINE_COLOR, linewidth=1.2,
            linestyle="--", alpha=0.7, label="Baseline")
    demand_line, = ax.plot([], [], color="#2a78d6", linewidth=2.0, label="Total demand")
    fill_ref = [ax.fill_between(x, baseline, baseline, alpha=0.20, color="#2a78d6")]
    vline = ax.axvline(0, color="#2a78d6", linestyle=":", linewidth=1.0, alpha=0.0)
    round_text = ax.text(
        0.01, 0.97, "", transform=ax.transAxes,
        fontsize=10, fontweight="bold", color="#2a78d6", verticalalignment="top",
    )
    ax.legend(loc="lower left", fontsize=9)

    def _update(frame_idx: int):
        result = feedback.rounds[frame_idx]
        demand_line.set_data(x, result.total_demand)

        # fill_between can't be updated in-place so remove and redraw each frame
        fill_ref[0].remove()
        fill_ref[0] = ax.fill_between(x, baseline, result.total_demand,
                                       alpha=0.20, color="#2a78d6")

        tp = result.total_peak_period
        vline.set_xdata([tp, tp])
        vline.set_alpha(0.6)
        round_text.set_text(
            f"Round {frame_idx + 1}   peak {result.total_peak_gw:.1f} GW @ {result.total_peak_time}"
        )
        return demand_line, vline, round_text

    anim = matplotlib.animation.FuncAnimation(
        fig, _update, frames=len(feedback.rounds), interval=600, blit=False,
    )

    out_path = FIGURES_DIR / "phase3_feedback_animation.gif"
    try:
        writer = matplotlib.animation.PillowWriter(fps=1.5)
        anim.save(out_path, writer=writer, dpi=DPI)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            f"GIF save failed: {exc}\nInstall Pillow with:  pip install pillow"
        ) from exc

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path
