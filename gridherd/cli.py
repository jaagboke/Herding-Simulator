"""
cli.py - Command-line entry point for gridherd.

Install with `pip install -e .` then run:

    gridherd run --agents 3000000 --plot
    gridherd run --agents 3000000 --strategy jitter --plot
    gridherd run --agents 3000000 --strategy all --plot
    gridherd run --agents 3000000 --mix naive=0.5,jitter=0.5 --plot
    gridherd run --agents 3000000 --feedback --rounds 10 --plot
"""

import argparse
import sys

from gridherd.agents import JITTER_DEFAULT_MAX_PERIODS, SPREAD_DEFAULT_K_POOL, HouseholdAgent
from gridherd.market import agile_price_curve, baseline_demand
from gridherd.simulate import run_feedback_simulation, run_mixed_simulation, run_simulation


def _parse_mix(s: str) -> dict[str, float]:
    """
    Parse --mix 'naive=0.5,jitter=0.5' into {'naive': 0.5, 'jitter': 0.5}.

    Normalises fractions to sum to 1, so you can write --mix naive=1,jitter=1
    and get 50/50 without doing the maths yourself.
    """
    result = {}
    try:
        for item in s.split(","):
            name, frac = item.strip().split("=")
            result[name.strip()] = float(frac.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Cannot parse mix {s!r}. Expected format: 'naive=0.5,jitter=0.5'"
        )
    total = sum(result.values())
    return {k: v / total for k, v in result.items()}


def _period_time(period: int) -> str:
    from gridherd.market import period_label
    return period_label(period)


def _print_result_row(label: str, result) -> None:
    delta_sign = "+" if result.peak_increase_gw >= 0 else ""
    cost_pounds = result.avg_cost_pence / 100
    print(
        f"  {label:<10}  peak {result.total_peak_gw:>5.1f} GW @ {result.total_peak_time}"
        f"  ({delta_sign}{result.peak_increase_gw:.1f} GW, {delta_sign}{result.peak_increase_pct:.1f}%)"
        f"  avg cost {result.avg_cost_pence:.0f}p (£{cost_pounds:.2f})"
    )


def _print_feedback_table(feedback) -> None:
    """Print a per-round summary table for the feedback simulation."""
    print()
    print("  Round  Peak (GW)  Time   vs baseline")
    print("  " + "-" * 38)
    baseline_peak = feedback.baseline.max()
    for i, result in enumerate(feedback.rounds):
        delta = result.total_peak_gw - baseline_peak
        sign = "+" if delta >= 0 else ""
        print(
            f"  {i+1:>5}  {result.total_peak_gw:>8.1f}  "
            f"{result.total_peak_time}  {sign}{delta:.1f} GW"
        )


def cmd_run(args: argparse.Namespace) -> None:
    prices = agile_price_curve(seed=args.seed)
    baseline = baseline_demand()
    agent = HouseholdAgent()

    print()
    print("gridherd - UK Grid Demand Herding Simulator")
    print(f"  Agents: {args.agents:,}   Seed: {args.seed}")
    print(f"  Baseline peak: {baseline.max():.1f} GW @ {_period_time(int(baseline.argmax()))}")
    print()

    if args.feedback:
        print(f"  Feedback mode: {args.rounds} rounds, sensitivity={args.sensitivity} p/GW")
        feedback = run_feedback_simulation(
            args.agents, prices, baseline, agent,
            sensitivity=args.sensitivity,
            rounds=args.rounds,
        )
        _print_feedback_table(feedback)

        if args.plot:
            from gridherd.plots import plot_feedback_animation, plot_feedback_rounds
            p1 = plot_feedback_rounds(feedback, show=not args.no_display)
            print(f"\n  Small-multiples figure: {p1}")
            p2 = plot_feedback_animation(feedback, show=False)
            print(f"  Animation GIF:          {p2}")
        return

    if args.mix:
        result = run_mixed_simulation(
            args.agents, prices, baseline, agent, args.mix,
            jitter_max_periods=args.jitter_max,
            k_pool=args.k_pool,
        )
        print(f"  Mixed population: {args.mix}")
        _print_result_row("mixed", result)

        if args.plot:
            from gridherd.plots import plot_demand
            path = plot_demand(result, show=not args.no_display)
            print(f"\n  Figure saved to: {path}")

    elif args.strategy == "all":
        results = {}
        for strategy in ("naive", "jitter", "spread"):
            results[strategy] = run_simulation(
                args.agents, prices, baseline, agent,
                strategy=strategy,
                jitter_max_periods=args.jitter_max,
                k_pool=args.k_pool,
            )

        print("  Strategy comparison:")
        for strategy, result in results.items():
            _print_result_row(strategy, result)

        naive_cost = results["naive"].avg_cost_pence
        print()
        print("  Cost of de-synchronisation (vs naive):")
        for strategy, result in results.items():
            delta = result.avg_cost_pence - naive_cost
            sign = "+" if delta >= 0 else ""
            print(f"    {strategy:<8}  {sign}{delta:.0f}p per charge ({sign}{100*delta/naive_cost:.1f}%)")

        if args.plot:
            from gridherd.plots import plot_strategy_comparison
            path = plot_strategy_comparison(results, show=not args.no_display)
            print(f"\n  Comparison figure saved to: {path}")

    else:
        result = run_simulation(
            args.agents, prices, baseline, agent,
            strategy=args.strategy,
            jitter_max_periods=args.jitter_max,
            k_pool=args.k_pool,
        )
        _print_result_row(args.strategy, result)

        if args.plot:
            from gridherd.plots import plot_demand
            path = plot_demand(result, show=not args.no_display)
            print(f"\n  Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gridherd",
        description="Simulate UK electricity grid demand herding from price-responsive devices.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the simulation.")
    run_parser.add_argument(
        "--agents", type=int, default=3_000_000, metavar="N",
        help="Number of household agents (default: 3,000,000).",
    )
    run_parser.add_argument(
        "--strategy", choices=["naive", "jitter", "spread", "all"], default="naive",
        help="Charging strategy. 'all' compares all three and generates a comparison plot.",
    )
    run_parser.add_argument(
        "--mix", type=_parse_mix, default=None, metavar="STRATEGY=FRAC,...",
        help="Mixed population, e.g. --mix naive=0.5,jitter=0.5. Overrides --strategy.",
    )
    run_parser.add_argument(
        "--jitter-max", type=int, default=JITTER_DEFAULT_MAX_PERIODS, metavar="P",
        dest="jitter_max",
        help=f"Max jitter delay in half-hour periods (default: {JITTER_DEFAULT_MAX_PERIODS}).",
    )
    run_parser.add_argument(
        "--k-pool", type=int, default=SPREAD_DEFAULT_K_POOL, metavar="K",
        dest="k_pool",
        help=f"Spread pool size in periods (default: {SPREAD_DEFAULT_K_POOL}).",
    )
    run_parser.add_argument(
        "--feedback", action="store_true",
        help="Enable iterative price-feedback loop (Phase 3).",
    )
    run_parser.add_argument(
        "--rounds", type=int, default=10, metavar="N",
        help="Feedback rounds to simulate (default: 10). Only used with --feedback.",
    )
    run_parser.add_argument(
        "--sensitivity", type=float, default=0.5, metavar="S",
        help="Price feedback sensitivity in p/GW of total demand (default: 0.5).",
    )
    run_parser.add_argument(
        "--plot", action="store_true",
        help="Generate and save figures to figures/.",
    )
    run_parser.add_argument(
        "--no-display", action="store_true",
        help="Save figures without opening windows (useful for headless/CI).",
    )
    run_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the price curve (default: 42).",
    )

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
