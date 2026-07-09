"""
agents.py - Household device agents and their charging strategies.

Each agent represents one price-responsive home device - an EV charger, home
battery, or smart heat pump. They all see the same 48 Agile prices and decide
independently when to charge.

The herding problem emerges naturally from this setup: if every agent runs the
same greedy "charge in the cheapest periods" strategy, they all pick the same
periods. Scale that to 3 million devices and you get a 21 GW spike in the middle
of the night - exactly when the grid is least prepared for it.

Phases 2 and 3 add two alternative strategies to break the synchronisation:

  jitter - each agent randomly delays its start by up to N periods. Agents with
    longer delays can't reach the cheapest overnight slots, so they spread out a
    bit. Problem: most agents still get short delays and herd anyway. The UK's 2021
    Smart Charging Regulations mandate 10-minute jitter, which is barely 0.3 periods
    - basically useless at scale.

  spread - all agents agree on a pool of K cheap periods and each randomly picks
    which K/n_needed of those to use. Since the picks are independent, aggregate
    demand evens out across the whole pool. This actually works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridherd.market import PERIOD_HOURS, PERIODS

# default jitter window: 4 hours (8 periods). Deliberately generous -
# the regulation only mandates 10 min but we use 4h to show it still doesn't work
JITTER_DEFAULT_MAX_PERIODS = 8

# pool size for spread strategy: twice as many periods as an agent actually needs.
# with 4 periods needed and k=8, each pool period gets half the fleet's demand
SPREAD_DEFAULT_K_POOL = 8

VALID_STRATEGIES = ("naive", "jitter", "spread")


@dataclass
class HouseholdAgent:
    """
    Represents one price-responsive home charging device.

    Defaults are calibrated to a typical UK EV owner:
    - 7 kW charger (standard single-phase Type 2)
    - needs 14 kWh to get ready for a 30-mile commute
    - arrives home flat (SoC = 0) and needs to be done by 07:00 (period 14)

    We keep all agents identical so the herding effect is as clear as possible.
    In reality: different arrival times, battery sizes, and mileages would spread
    demand out naturally. That's Phase 4 territory.
    """
    charge_rate_kw: float = 7.0
    energy_required_kwh: float = 14.0
    state_of_charge_kwh: float = 0.0
    deadline_period: int = 14         # period 14 = 07:00, when the owner leaves
    battery_capacity_kwh: float = 75.0  # not used in scheduling, just for realism

    @property
    def energy_needed_kwh(self) -> float:
        """How much charge the agent still needs (can't be negative)."""
        return max(0.0, self.energy_required_kwh - self.state_of_charge_kwh)


def _greedy_schedule(
    agent: HouseholdAgent,
    prices: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """
    Fill cheapest eligible periods first until the agent's energy need is met.

    This is the core of the herding mechanism. Because every agent calls this
    with the same prices, they all sort periods the same way and pick the same
    slots. The last period might be partial (if the remaining energy need is
    less than a full 30-minute slot).

    Shared between naive and jitter so the selection logic is only written once.
    """
    schedule = np.zeros(PERIODS)
    remaining = agent.energy_needed_kwh

    if remaining <= 0 or len(eligible) == 0:
        return schedule

    # sort eligible periods by price, cheapest first
    for period in eligible[np.argsort(prices[eligible])]:
        if remaining <= 0:
            break
        energy_this_period = agent.charge_rate_kw * PERIOD_HOURS
        # partial charging if less than a full period's worth is left
        actual_energy = min(energy_this_period, remaining)
        schedule[period] = actual_energy / PERIOD_HOURS  # back to kW
        remaining -= actual_energy

    return schedule


def naive_schedule(agent: HouseholdAgent, prices: np.ndarray) -> np.ndarray:
    """
    The basic strategy: just charge in the cheapest available periods.

    This is what any sensible cost-minimising smart charger would do, and it's
    exactly what causes the problem. Every agent uses periods 0 through deadline-1
    as eligible, sorts by price, and picks the cheapest N. They all pick the same N.
    """
    return _greedy_schedule(agent, prices, np.arange(agent.deadline_period))


def _expected_jitter_schedule(
    agent: HouseholdAgent,
    prices: np.ndarray,
    jitter_max_periods: int,
) -> np.ndarray:
    """
    Average schedule across all possible random start delays.

    Each agent picks a delay d uniformly from {0, 1, ..., jitter_max_periods}
    and only uses periods from d onwards as eligible. Instead of simulating each
    agent individually, we just average over all d values - this gives the expected
    per-agent schedule directly, which we can then scale by N.

    Why this barely helps: with jitter_max=8, most agents (those with d=0..5) can
    still reach the cheapest overnight periods. Only the 2-3 agents with the longest
    delays get pushed off the cheap slots. So about 7/9 of the fleet still herds.
    """
    n_delays = jitter_max_periods + 1
    accumulated = np.zeros(PERIODS)

    # average over each possible delay value
    for delay in range(n_delays):
        eligible = np.arange(delay, agent.deadline_period)
        accumulated += _greedy_schedule(agent, prices, eligible)

    return accumulated / n_delays


def get_spread_pool(
    eligible: np.ndarray,
    prices: np.ndarray,
    k_pool: int,
) -> np.ndarray:
    """
    Find the cheapest contiguous block of k_pool periods within eligible.

    We use a sliding window sum (convolution) to find the starting index of
    the k-period run with the lowest total price, then return those periods.

    Why contiguous rather than the k individually cheapest periods?
    Price noise occasionally flips the order of two adjacent overnight periods
    (e.g. period 1 ends up ranked cheaper than period 2 due to random noise,
    even though the underlying curve has it the other way). If we pick scattered
    cheapest periods we'd get a non-contiguous set like {1, 3, 4, 5, 6, 7, 8, 9}
    with a gap at period 2. That gap creates a demand dip in the middle of the
    overnight window - visible as a zigzag in plots. A contiguous block avoids
    the gap and is also more physically sensible (charge across the cheapest
    4-hour window, not in random scattered slots).
    """
    k = min(k_pool, len(eligible))
    if k >= len(eligible):
        return eligible.copy()

    # rolling sum of k consecutive prices, find the cheapest starting position
    window_sums = np.convolve(prices[eligible], np.ones(k, dtype=float), mode='valid')
    best_start = int(np.argmin(window_sums))
    return eligible[best_start : best_start + k]


def _expected_spread_schedule(
    agent: HouseholdAgent,
    prices: np.ndarray,
    k_pool: int,
) -> np.ndarray:
    """
    Average schedule for an agent that picks randomly from the cheapest pool.

    All agents see the same k_pool cheap periods (same prices = same pool), but
    each independently and randomly picks which of those periods to actually use.
    Since the draws are independent, aggregate demand spreads uniformly across
    the pool - no herding spike.

    The maths: if an agent needs n periods out of a pool of k, the expected
    fraction of any single pool period it uses is n/k (hypergeometric mean).
    So expected power at each pool period = charge_rate * (n_needed / k).
    We skip the simulation and use this formula directly.
    """
    eligible = np.arange(agent.deadline_period)
    if len(eligible) == 0:
        return np.zeros(PERIODS)

    pool = get_spread_pool(eligible, prices, k_pool)
    k = len(pool)

    energy_per_period = agent.charge_rate_kw * PERIOD_HOURS
    n_needed = agent.energy_needed_kwh / energy_per_period
    # fraction of each pool period that the average agent occupies
    fraction = min(1.0, n_needed / k)

    schedule = np.zeros(PERIODS)
    schedule[pool] = agent.charge_rate_kw * fraction
    return schedule


def expected_per_agent_schedule(
    strategy: str,
    agent: HouseholdAgent,
    prices: np.ndarray,
    jitter_max_periods: int = JITTER_DEFAULT_MAX_PERIODS,
    k_pool: int = SPREAD_DEFAULT_K_POOL,
) -> np.ndarray:
    """
    Return the expected per-agent schedule (kW, length 48) for a given strategy.

    For naive the schedule is deterministic so 'expected' just means actual.
    For jitter and spread it's the mathematical expectation over the randomness
    in each strategy. Multiply by N agents and divide by 1e6 to get GW.
    """
    if strategy == "naive":
        return naive_schedule(agent, prices)
    elif strategy == "jitter":
        return _expected_jitter_schedule(agent, prices, jitter_max_periods)
    elif strategy == "spread":
        return _expected_spread_schedule(agent, prices, k_pool)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Valid choices: {VALID_STRATEGIES}")


def schedule_cost_pence(schedule: np.ndarray, prices: np.ndarray) -> float:
    """
    Total cost of a charging schedule in pence.

    cost = sum over periods of (power_kW * 0.5h * price_p_per_kWh)
         = kWh consumed * p/kWh = pence

    For jitter and spread schedules (which are expectations), this gives the
    expected cost per agent. Naive always has the lowest cost by construction
    since it greedily picks the cheapest slots - any other strategy pays more
    to achieve some grid benefit.
    """
    return float(np.dot(schedule * PERIOD_HOURS, prices))
