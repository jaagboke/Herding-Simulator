"""
Tests for simulation aggregation in gridherd/simulate.py.

Mostly checking mathematical properties: linearity with agent count, correct
baseline/agent/total relationships, and that the herding effect actually shows
up in the output (a new overnight peak higher than the original evening one).
"""

import numpy as np
import pytest

from gridherd.agents import HouseholdAgent
from gridherd.market import PERIODS, agile_price_curve, baseline_demand
from gridherd.simulate import SimulationResult, run_simulation


# --- aggregation maths ---

def test_total_is_baseline_plus_agents():
    """total_demand must equal baseline + agent_demand at every single period."""
    prices = agile_price_curve()
    baseline = baseline_demand()
    result = run_simulation(1_000, prices, baseline, HouseholdAgent())

    np.testing.assert_array_equal(
        result.total_demand,
        result.baseline_demand + result.agent_demand,
    )


def test_agent_demand_scales_with_n():
    """Doubling the agent count should exactly double the agent demand."""
    prices = agile_price_curve()
    baseline = baseline_demand()
    agent = HouseholdAgent()

    r1 = run_simulation(1_000, prices, baseline, agent)
    r2 = run_simulation(2_000, prices, baseline, agent)

    np.testing.assert_allclose(r2.agent_demand, r1.agent_demand * 2, rtol=1e-10)


def test_zero_agents_leaves_baseline_intact():
    """With no agents, total demand should be identical to the baseline."""
    prices = agile_price_curve()
    baseline = baseline_demand()
    result = run_simulation(0, prices, baseline, HouseholdAgent())

    np.testing.assert_array_equal(result.total_demand, baseline)


def test_agent_demand_never_negative():
    """Agents only consume power, so demand added should never drop below zero."""
    result = run_simulation(1_000, agile_price_curve(), baseline_demand(), HouseholdAgent())
    assert (result.agent_demand >= 0).all()


# --- output shapes ---

def test_all_arrays_are_48_periods():
    """Every array in the result should be shape (48,)."""
    result = run_simulation(100, agile_price_curve(), baseline_demand(), HouseholdAgent())

    assert result.prices.shape == (PERIODS,)
    assert result.baseline_demand.shape == (PERIODS,)
    assert result.agent_demand.shape == (PERIODS,)
    assert result.total_demand.shape == (PERIODS,)


# --- peak properties ---

def test_baseline_peak_is_in_the_evening():
    """Without smart devices, the grid's peak should be in the afternoon/evening."""
    result = run_simulation(0, agile_price_curve(), baseline_demand(), HouseholdAgent())

    # period 28 = 14:00, so we're checking it's afternoon or later
    assert result.baseline_peak_period >= 28, (
        f"Baseline peak should be in the evening, got period {result.baseline_peak_period}"
    )


def test_herding_creates_overnight_peak():
    """
    With 3M agents (21 GW of flexible load), the overnight cheap periods
    accumulate enough demand to create a new peak above the evening one.
    This is the whole point of the simulation.
    """
    prices = agile_price_curve()
    baseline = baseline_demand()
    result = run_simulation(3_000_000, prices, baseline, HouseholdAgent())

    # the herding spike should be in the charging window, before 07:00
    assert result.total_peak_period < 14, (
        f"Expected herding peak before 07:00 (period 14), "
        f"got period {result.total_peak_period} ({result.total_peak_time})"
    )

    # and it should be worse than the original evening peak
    assert result.total_peak_gw > result.baseline_peak_gw, (
        f"Herding peak ({result.total_peak_gw:.1f} GW) should exceed "
        f"baseline peak ({result.baseline_peak_gw:.1f} GW)"
    )


def test_peak_time_matches_peak_period():
    """The string peak time labels should be derived from the period indices."""
    from gridherd.market import period_label

    result = run_simulation(1_000_000, agile_price_curve(), baseline_demand(), HouseholdAgent())

    assert result.baseline_peak_time == period_label(result.baseline_peak_period)
    assert result.total_peak_time == period_label(result.total_peak_period)


def test_peak_increase_properties():
    """peak_increase_gw and peak_increase_pct should be consistent with the raw peaks."""
    result = run_simulation(1_000_000, agile_price_curve(), baseline_demand(), HouseholdAgent())

    expected_gw = result.total_peak_gw - result.baseline_peak_gw
    expected_pct = 100.0 * expected_gw / result.baseline_peak_gw

    assert result.peak_increase_gw == pytest.approx(expected_gw, abs=1e-9)
    assert result.peak_increase_pct == pytest.approx(expected_pct, abs=1e-9)
