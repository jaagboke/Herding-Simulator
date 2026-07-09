"""
simulate.py - Runs the simulation and collects results.

The core question this module answers: if N agents all apply strategy S,
what does aggregate demand look like across the 48 settlement periods?

For phases 1 and 2, the answer is O(1) in N: compute one agent's expected
schedule and multiply by N. No need to loop over millions of agents.

Phase 3 adds a feedback loop where the previous round's aggregate demand
feeds back into the prices agents see next round. This makes the spike
migrate rather than stay put - and it never converges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gridherd.agents import (
    JITTER_DEFAULT_MAX_PERIODS,
    SPREAD_DEFAULT_K_POOL,
    VALID_STRATEGIES,
    HouseholdAgent,
    expected_per_agent_schedule,
    schedule_cost_pence,
)
from gridherd.market import PERIODS, period_label


@dataclass
class SimulationResult:
    """Everything in and out of a single simulation run."""

    prices: np.ndarray           # p/kWh, shape (48,)
    baseline_demand: np.ndarray  # GW before agents, shape (48,)
    agent_demand: np.ndarray     # GW added by agents, shape (48,)
    n_agents: int
    strategy: str = "naive"
    avg_cost_pence: float = 0.0
    deadline_period: int = 14    # stored so plots can shade the eligible window

    @property
    def total_demand(self) -> np.ndarray:
        return self.baseline_demand + self.agent_demand

    @property
    def baseline_peak_gw(self) -> float:
        return float(self.baseline_demand.max())

    @property
    def baseline_peak_period(self) -> int:
        return int(self.baseline_demand.argmax())

    @property
    def baseline_peak_time(self) -> str:
        return period_label(self.baseline_peak_period)

    @property
    def total_peak_gw(self) -> float:
        return float(self.total_demand.max())

    @property
    def total_peak_period(self) -> int:
        return int(self.total_demand.argmax())

    @property
    def total_peak_time(self) -> str:
        return period_label(self.total_peak_period)

    @property
    def peak_increase_gw(self) -> float:
        return self.total_peak_gw - self.baseline_peak_gw

    @property
    def peak_increase_pct(self) -> float:
        return 100.0 * self.peak_increase_gw / self.baseline_peak_gw


def run_simulation(
    n_agents: int,
    prices: np.ndarray,
    baseline: np.ndarray,
    agent_template: HouseholdAgent,
    strategy: str = "naive",
    jitter_max_periods: int = JITTER_DEFAULT_MAX_PERIODS,
    k_pool: int = SPREAD_DEFAULT_K_POOL,
) -> SimulationResult:
    """
    Simulate N identical agents all using the same strategy.

    Works by computing the expected per-agent schedule once and scaling by N.
    This is exact (not an approximation) because all agents are identical - there
    is no per-agent variance to worry about. For 3 million agents this takes
    the same time as computing it for 1 agent.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. Valid: {VALID_STRATEGIES}")

    per_agent_kw = expected_per_agent_schedule(
        strategy, agent_template, prices,
        jitter_max_periods=jitter_max_periods,
        k_pool=k_pool,
    )

    # scale from kW per agent to GW across the whole fleet
    agent_demand_gw = (per_agent_kw * n_agents) / 1_000_000
    cost = schedule_cost_pence(per_agent_kw, prices)

    return SimulationResult(
        prices=prices,
        baseline_demand=baseline,
        agent_demand=agent_demand_gw,
        n_agents=n_agents,
        strategy=strategy,
        avg_cost_pence=cost,
        deadline_period=agent_template.deadline_period,
    )


def run_mixed_simulation(
    n_agents: int,
    prices: np.ndarray,
    baseline: np.ndarray,
    agent_template: HouseholdAgent,
    mix: dict[str, float],
    jitter_max_periods: int = JITTER_DEFAULT_MAX_PERIODS,
    k_pool: int = SPREAD_DEFAULT_K_POOL,
) -> SimulationResult:
    """
    Simulate a population where different fractions use different strategies.

    mix maps strategy name to the fraction of agents using it, e.g.
    {"naive": 0.5, "jitter": 0.5} for a 50/50 split. Fractions must sum to 1.

    Each sub-population's demand is computed independently and added up.
    The reported average cost is the population-weighted average.
    """
    total_mix = sum(mix.values())
    if abs(total_mix - 1.0) > 1e-9:
        raise ValueError(f"mix fractions must sum to 1.0, got {total_mix:.4f}")

    total_agent_demand = np.zeros(PERIODS)
    total_cost = 0.0

    for strategy, fraction in mix.items():
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy in mix: {strategy!r}")

        per_agent_kw = expected_per_agent_schedule(
            strategy, agent_template, prices,
            jitter_max_periods=jitter_max_periods,
            k_pool=k_pool,
        )
        total_agent_demand += (per_agent_kw * n_agents * fraction) / 1_000_000
        total_cost += schedule_cost_pence(per_agent_kw, prices) * fraction

    label = ",".join(f"{s}={f:.0%}" for s, f in mix.items())

    return SimulationResult(
        prices=prices,
        baseline_demand=baseline,
        agent_demand=total_agent_demand,
        n_agents=n_agents,
        strategy=f"mixed({label})",
        avg_cost_pence=total_cost,
        deadline_period=agent_template.deadline_period,
    )


@dataclass
class FeedbackSimulation:
    """
    Results of an iterative price-feedback simulation.

    rounds[0]  = naive scheduling against the original prices (no feedback yet)
    rounds[k]  = naive scheduling after k rounds of price feedback
    effective_prices[k] = the prices agents actually saw during round k

    The feedback formula is:
        effective_price[p] = base_price[p] + sensitivity * total_demand[p]

    where total_demand comes from the previous round. Agents respond by moving
    to whatever is now cheapest, which creates a new spike there, which makes
    those periods expensive next round, and so on. No convergence.
    """
    rounds: list[SimulationResult]
    effective_prices: list[np.ndarray]
    base_prices: np.ndarray
    baseline: np.ndarray
    sensitivity: float  # p/GW - how much price rises per GW of demand
    n_agents: int


def run_feedback_simulation(
    n_agents: int,
    prices: np.ndarray,
    baseline: np.ndarray,
    agent_template: HouseholdAgent,
    sensitivity: float = 0.5,
    rounds: int = 10,
) -> FeedbackSimulation:
    """
    Run naive scheduling iteratively with linear price-demand feedback.

    After each round, the price signal is updated:
        next_price[p] = original_price[p] + sensitivity * this_round_total_demand[p]

    Prices always reset from the original base - they don't compound. This is
    closer to how balancing market prices actually work (recalculated each half-hour
    from the wholesale spot price), and it also avoids prices growing without bound.

    Note: real imbalance pricing is nonlinear and step-function shaped (reserve
    tiers activate at specific MW thresholds). Linear is a reasonable first pass
    that shows the direction of the effect without needing to model market microstructure.

    With sensitivity=0 this degenerates to running the same naive simulation N times,
    which is useful as a regression test.
    """
    results = []
    effective_prices_list = []
    current_prices = prices.copy()

    for _ in range(rounds):
        effective_prices_list.append(current_prices.copy())
        result = run_simulation(n_agents, current_prices, baseline, agent_template,
                                strategy="naive")
        results.append(result)
        # update prices for next round using this round's total demand
        current_prices = prices + sensitivity * result.total_demand

    return FeedbackSimulation(
        rounds=results,
        effective_prices=effective_prices_list,
        base_prices=prices,
        baseline=baseline,
        sensitivity=sensitivity,
        n_agents=n_agents,
    )
