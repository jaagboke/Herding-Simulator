"""
Tests for Phase 2 strategies: jitter, spread, and mixed populations.

Three categories:
  - Energy conservation: each strategy delivers the right amount of energy
  - Structure: jitter respects deadlines, spread is uniform across its pool
  - Cost ordering: naive is always cheapest (it's designed to be)

The spread tests use get_spread_pool directly so both the test and the
implementation agree on what the pool looks like.
"""

import numpy as np
import pytest

from gridherd.agents import (
    JITTER_DEFAULT_MAX_PERIODS,
    SPREAD_DEFAULT_K_POOL,
    HouseholdAgent,
    expected_per_agent_schedule,
    get_spread_pool,
    schedule_cost_pence,
)
from gridherd.market import PERIOD_HOURS, PERIODS, agile_price_curve, baseline_demand
from gridherd.simulate import run_mixed_simulation, run_simulation


# --- jitter ---

def test_jitter_delivers_correct_energy():
    """Jitter schedule should deliver exactly the required energy in expectation."""
    agent = HouseholdAgent(energy_required_kwh=14.0, state_of_charge_kwh=0.0)
    prices = agile_price_curve()
    schedule = expected_per_agent_schedule("jitter", agent, prices)

    assert schedule.sum() * PERIOD_HOURS == pytest.approx(14.0, abs=1e-9)


def test_jitter_zero_delay_equals_naive():
    """With max delay = 0 every agent has zero delay, so jitter == naive."""
    agent = HouseholdAgent()
    prices = agile_price_curve()

    naive = expected_per_agent_schedule("naive", agent, prices)
    jitter_zero = expected_per_agent_schedule("jitter", agent, prices, jitter_max_periods=0)

    np.testing.assert_array_equal(naive, jitter_zero)


def test_jitter_stays_within_deadline():
    """No charging should appear at or after the deadline, regardless of delay."""
    agent = HouseholdAgent(deadline_period=14)
    schedule = expected_per_agent_schedule("jitter", agent, agile_price_curve(),
                                           jitter_max_periods=8)

    assert schedule[14:].sum() == pytest.approx(0.0)


def test_jitter_reduces_peak_vs_naive():
    """
    Large jitter should reduce (but not eliminate) the peak concentration.
    With jitter_max=8, not all agents can access the very cheapest period,
    so the peak power in any single period should drop below the naive level.
    """
    agent = HouseholdAgent()
    prices = agile_price_curve()

    naive_sched = expected_per_agent_schedule("naive", agent, prices)
    jitter_sched = expected_per_agent_schedule("jitter", agent, prices, jitter_max_periods=8)

    assert jitter_sched.max() < naive_sched.max()


def test_jitter_nonnegative():
    """Power draw is never negative."""
    schedule = expected_per_agent_schedule("jitter", HouseholdAgent(), agile_price_curve())
    assert (schedule >= 0).all()


# --- spread ---

def test_spread_delivers_correct_energy():
    """Spread schedule should deliver the required energy in expectation."""
    agent = HouseholdAgent(energy_required_kwh=14.0, state_of_charge_kwh=0.0)
    schedule = expected_per_agent_schedule("spread", agent, agile_price_curve(), k_pool=8)

    assert schedule.sum() * PERIOD_HOURS == pytest.approx(14.0, abs=1e-9)


def test_spread_uniform_across_pool():
    """
    Every period in the spread pool should have identical expected power.

    This is the key property that makes spread actually work: no single period
    is more likely to be chosen than another, so the 21 GW of flexible demand
    spreads evenly across all K pool periods rather than piling into one.
    """
    agent = HouseholdAgent(energy_required_kwh=14.0, state_of_charge_kwh=0.0, deadline_period=14)
    prices = agile_price_curve()
    k_pool = 8
    schedule = expected_per_agent_schedule("spread", agent, prices, k_pool=k_pool)

    eligible = np.arange(agent.deadline_period)
    pool = get_spread_pool(eligible, prices, k_pool)
    pool_powers = schedule[pool]

    np.testing.assert_allclose(
        pool_powers, pool_powers[0], rtol=1e-10,
        err_msg="all pool periods should have equal expected power",
    )


def test_spread_zero_outside_pool():
    """Periods outside the pool should have zero demand."""
    agent = HouseholdAgent(deadline_period=14)
    prices = agile_price_curve()
    k_pool = 6
    schedule = expected_per_agent_schedule("spread", agent, prices, k_pool=k_pool)

    eligible = np.arange(agent.deadline_period)
    pool = set(get_spread_pool(eligible, prices, k_pool))
    non_pool = [p for p in range(PERIODS) if p not in pool]

    assert schedule[non_pool].sum() == pytest.approx(0.0)


def test_spread_k1_equals_naive():
    """With pool size 1, there's only one period to pick, same as naive would pick."""
    agent = HouseholdAgent(energy_required_kwh=3.5)  # needs exactly 1 period
    prices = agile_price_curve()

    naive = expected_per_agent_schedule("naive", agent, prices)
    spread_k1 = expected_per_agent_schedule("spread", agent, prices, k_pool=1)

    np.testing.assert_allclose(naive, spread_k1, rtol=1e-10)


def test_spread_nonnegative():
    """Power draw is never negative."""
    schedule = expected_per_agent_schedule("spread", HouseholdAgent(), agile_price_curve())
    assert (schedule >= 0).all()


# --- cost ordering ---

def test_naive_has_lowest_cost():
    """
    Naive greedily picks the cheapest periods so it should have the lowest
    cost by construction. Jitter and spread pay a premium for grid benefit.
    """
    agent = HouseholdAgent()
    prices = agile_price_curve()

    naive_cost  = schedule_cost_pence(expected_per_agent_schedule("naive",  agent, prices), prices)
    jitter_cost = schedule_cost_pence(expected_per_agent_schedule("jitter", agent, prices), prices)
    spread_cost = schedule_cost_pence(expected_per_agent_schedule("spread", agent, prices), prices)

    assert naive_cost <= jitter_cost, "jitter shouldn't be cheaper than naive"
    assert naive_cost <= spread_cost, "spread shouldn't be cheaper than naive"


def test_spread_costs_more_than_naive():
    """Spread uses some periods that aren't the absolute cheapest, so it costs more."""
    agent = HouseholdAgent()
    prices = agile_price_curve()

    naive_cost  = schedule_cost_pence(expected_per_agent_schedule("naive",  agent, prices), prices)
    spread_cost = schedule_cost_pence(expected_per_agent_schedule("spread", agent, prices, k_pool=8), prices)

    assert spread_cost > naive_cost


def test_all_costs_positive():
    """Every strategy should produce a positive charging cost."""
    agent = HouseholdAgent()
    prices = agile_price_curve()
    for strategy in ("naive", "jitter", "spread"):
        cost = schedule_cost_pence(expected_per_agent_schedule(strategy, agent, prices), prices)
        assert cost > 0, f"expected positive cost for {strategy}"


# --- herding reduction ---

def test_spread_eliminates_overnight_spike():
    """
    With spread, the total peak should stay at or below the original baseline
    peak. This means the herding spike is completely gone.
    """
    prices = agile_price_curve()
    baseline = baseline_demand()
    result = run_simulation(3_000_000, prices, baseline, HouseholdAgent(), strategy="spread")

    assert result.total_peak_gw <= result.baseline_peak_gw, (
        f"Spread peak {result.total_peak_gw:.1f} GW should be at most "
        f"baseline peak {result.baseline_peak_gw:.1f} GW"
    )


def test_jitter_reduces_but_not_eliminates_spike():
    """
    Jitter should reduce the herding spike compared to naive, but shouldn't
    fully eliminate it - most agents with short delays still herd.
    """
    prices = agile_price_curve()
    baseline = baseline_demand()
    agent = HouseholdAgent()

    naive_result  = run_simulation(3_000_000, prices, baseline, agent, strategy="naive")
    jitter_result = run_simulation(3_000_000, prices, baseline, agent, strategy="jitter")

    assert jitter_result.total_peak_gw < naive_result.total_peak_gw, (
        "jitter should reduce the peak compared to naive"
    )


# --- mixed populations ---

def test_mixed_demand_is_weighted_average():
    """50/50 naive+spread should produce exactly the average of the two pure demands."""
    prices = agile_price_curve()
    baseline = baseline_demand()
    agent = HouseholdAgent()
    n = 2_000

    naive_r  = run_simulation(n, prices, baseline, agent, strategy="naive")
    spread_r = run_simulation(n, prices, baseline, agent, strategy="spread")
    mixed_r  = run_mixed_simulation(n, prices, baseline, agent, mix={"naive": 0.5, "spread": 0.5})

    expected = (naive_r.agent_demand + spread_r.agent_demand) / 2
    np.testing.assert_allclose(mixed_r.agent_demand, expected, rtol=1e-10)


def test_mixed_cost_is_weighted_average():
    """Mixed population cost should be the fraction-weighted average of each strategy's cost."""
    prices = agile_price_curve()
    baseline = baseline_demand()
    agent = HouseholdAgent()

    naive_r  = run_simulation(1000, prices, baseline, agent, strategy="naive")
    jitter_r = run_simulation(1000, prices, baseline, agent, strategy="jitter")
    mixed_r  = run_mixed_simulation(1000, prices, baseline, agent,
                                    mix={"naive": 0.5, "jitter": 0.5})

    expected = 0.5 * naive_r.avg_cost_pence + 0.5 * jitter_r.avg_cost_pence
    assert mixed_r.avg_cost_pence == pytest.approx(expected, rel=1e-9)


def test_mix_fractions_get_normalised():
    """--mix naive=1,jitter=1 should be treated as 50/50 after normalisation."""
    from gridherd.cli import _parse_mix
    mix = _parse_mix("naive=1,jitter=1")
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    assert mix["naive"] == pytest.approx(0.5)
    assert mix["jitter"] == pytest.approx(0.5)


def test_mix_fractions_not_summing_to_one_raises():
    """run_mixed_simulation should reject mixes that don't sum to 1.0."""
    prices = agile_price_curve()
    baseline = baseline_demand()
    agent = HouseholdAgent()
    with pytest.raises(ValueError, match="sum to 1.0"):
        run_mixed_simulation(100, prices, baseline, agent, mix={"naive": 0.3, "spread": 0.3})


def test_unknown_strategy_raises():
    """Passing an invalid strategy name should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        expected_per_agent_schedule("turbo", HouseholdAgent(), agile_price_curve())
