"""
Tests for the Phase 3 price-feedback loop in gridherd/simulate.py.

The feedback loop updates prices after each round based on aggregate demand,
causing naive agents to reschedule. Key things to check:

  - With sensitivity=0 nothing changes between rounds (regression against Phase 1)
  - Non-zero sensitivity actually changes the prices in high-demand periods
  - The spike actually moves (it's a migration, not just noise)
  - The correct number of rounds is returned
  - Baseline demand stays constant throughout (only prices change)
"""

import numpy as np
import pytest

from gridherd.agents import HouseholdAgent
from gridherd.market import agile_price_curve, baseline_demand
from gridherd.simulate import run_feedback_simulation, run_simulation


@pytest.fixture(scope="module")
def prices():
    return agile_price_curve()


@pytest.fixture(scope="module")
def baseline():
    return baseline_demand()


@pytest.fixture(scope="module")
def agent():
    return HouseholdAgent()


def test_zero_sensitivity_matches_naive(prices, baseline, agent):
    """
    When sensitivity=0 the price signal never changes, so every round
    should produce the same result as a plain naive simulation.

    This is the regression guard: the feedback loop must degenerate to
    Phase 1 when there's no actual feedback.
    """
    naive = run_simulation(1000, prices, baseline, agent, strategy="naive")
    feedback = run_feedback_simulation(1000, prices, baseline, agent,
                                       sensitivity=0.0, rounds=5)

    for i, r in enumerate(feedback.rounds):
        np.testing.assert_allclose(
            r.total_demand, naive.total_demand, rtol=1e-12,
            err_msg=f"round {i+1} with sensitivity=0 should match naive exactly",
        )


def test_prices_rise_in_high_demand_periods(prices, baseline, agent):
    """
    After round 1, the effective prices for round 2 should be higher at
    periods where demand was high. That's the whole point of the feedback.
    """
    feedback = run_feedback_simulation(1000, prices, baseline, agent,
                                       sensitivity=0.5, rounds=2)

    round0 = feedback.rounds[0]
    prices_round1 = feedback.effective_prices[1]
    spike_period = round0.total_peak_period

    # price at the spike period should have gone up
    assert prices_round1[spike_period] > prices[spike_period], (
        "effective price at the spike period should exceed the original price"
    )

    # and the spiked periods should have risen more than the unspiked ones
    unspiked = [p for p in range(14) if round0.agent_demand[p] < 0.01]
    if unspiked:
        avg_rise_elsewhere = (prices_round1[unspiked] - prices[unspiked]).mean()
        rise_at_spike = prices_round1[spike_period] - prices[spike_period]
        assert rise_at_spike > avg_rise_elsewhere


def test_spike_migrates_between_rounds(prices, baseline, agent):
    """
    The peak period in round 2 should be different from round 1.

    This is the core finding: price feedback moves the spike to the next
    cheapest cluster, but all agents move together so it's still a spike.
    The herding mechanism is intact, just pointing somewhere new.
    """
    feedback = run_feedback_simulation(3_000_000, prices, baseline, agent,
                                       sensitivity=0.5, rounds=3)

    peak_r1 = feedback.rounds[0].total_peak_period
    peak_r2 = feedback.rounds[1].total_peak_period

    assert peak_r1 != peak_r2, (
        f"spike should have moved by round 2 but is still at period {peak_r2}"
    )


def test_correct_number_of_rounds_returned(prices, baseline, agent):
    """run_feedback_simulation(rounds=N) should return exactly N results."""
    for n in (1, 5, 10):
        feedback = run_feedback_simulation(100, prices, baseline, agent, rounds=n)
        assert len(feedback.rounds) == n
        assert len(feedback.effective_prices) == n


def test_baseline_stays_constant(prices, baseline, agent):
    """
    The baseline demand should be the same in every round. Only the agent
    scheduling changes between rounds, not the fixed background demand.
    """
    feedback = run_feedback_simulation(100, prices, baseline, agent,
                                       sensitivity=1.0, rounds=5)

    for i, r in enumerate(feedback.rounds):
        np.testing.assert_array_equal(
            r.baseline_demand, baseline,
            err_msg=f"baseline changed unexpectedly in round {i+1}",
        )
