"""
Tests for the agent scheduling logic in gridherd/agents.py.

Focus is on naive_schedule since that's the herding mechanism. We check that
it picks the right periods, stops when the energy need is met, respects the
deadline, and handles partial charging correctly.

The determinism test at the bottom is the formal statement of why herding exists:
same inputs always produce the same output, so N identical agents produce N
identical schedules that all stack on top of each other.
"""

import numpy as np
import pytest

from gridherd.agents import HouseholdAgent, naive_schedule
from gridherd.market import PERIOD_HOURS, PERIODS


def make_agent(**kwargs) -> HouseholdAgent:
    """Build an agent with default values, overriding any fields passed in."""
    defaults = dict(
        charge_rate_kw=7.0,
        energy_required_kwh=14.0,
        state_of_charge_kwh=0.0,
        deadline_period=14,
    )
    defaults.update(kwargs)
    return HouseholdAgent(**defaults)


# --- period selection ---

def test_charges_cheapest_periods():
    """Agent should prefer the periods with the lowest price."""
    prices = np.full(PERIODS, 20.0)
    prices[3] = 5.0  # cheapest in eligible window
    prices[7] = 6.0  # second cheapest

    # needs exactly 2 periods (7.0 kWh / 3.5 kWh per period)
    agent = make_agent(energy_required_kwh=7.0)
    schedule = naive_schedule(agent, prices)

    assert schedule[3] > 0, "should charge in cheapest period (3)"
    assert schedule[7] > 0, "should charge in second-cheapest period (7)"


def test_respects_deadline():
    """Nothing should be scheduled on or after the deadline, even if it's cheaper."""
    prices = np.full(PERIODS, 20.0)
    prices[20] = 1.0  # very cheap, but period 20 is past the 07:00 deadline

    agent = make_agent(energy_required_kwh=3.5)  # exactly 1 period
    schedule = naive_schedule(agent, prices)

    assert schedule[14:].sum() == 0.0
    assert schedule[20] == 0.0, "period 20 must not be used even though it's cheapest"


def test_stops_once_requirement_is_met():
    """Agent shouldn't keep scheduling after it's got enough energy."""
    # prices go 0, 1, 2, 3 ... so periods 0 and 1 are cheapest
    prices = np.arange(PERIODS, dtype=float)
    agent = make_agent(energy_required_kwh=7.0)  # needs 2 periods

    schedule = naive_schedule(agent, prices)

    assert schedule[0] == pytest.approx(7.0)
    assert schedule[1] == pytest.approx(7.0)
    assert schedule[2:].sum() == pytest.approx(0.0), "should stop after 2 periods"


# --- energy accuracy ---

def test_meets_energy_requirement_exactly():
    """Total charged energy should match the agent's deficit exactly."""
    agent = make_agent(energy_required_kwh=14.0, state_of_charge_kwh=0.0)
    prices = np.linspace(20, 10, PERIODS)

    schedule = naive_schedule(agent, prices)
    total_kwh = schedule.sum() * PERIOD_HOURS

    assert total_kwh == pytest.approx(14.0, abs=1e-9)


def test_only_charges_the_deficit():
    """Should only charge the gap between required and current SoC."""
    agent = make_agent(energy_required_kwh=10.0, state_of_charge_kwh=3.0)  # needs 7 kWh
    schedule = naive_schedule(agent, np.full(PERIODS, 20.0))

    assert schedule.sum() * PERIOD_HOURS == pytest.approx(7.0, abs=1e-9)


def test_partial_last_period():
    """
    If the remaining energy is less than a full half-hour, the last period
    should run at reduced power rather than over-charging.

    5.25 kWh = 1.5 periods at 7 kW. So: first period full, second at half rate.
    """
    agent = make_agent(energy_required_kwh=5.25)
    prices = np.arange(PERIODS, dtype=float)  # period 0 cheapest, period 1 next

    schedule = naive_schedule(agent, prices)

    assert schedule[0] == pytest.approx(7.0), "first period should be full rate"
    assert 0 < schedule[1] < 7.0, "second period should be partial"
    assert schedule[1] == pytest.approx(3.5)
    assert schedule[2] == pytest.approx(0.0)
    assert schedule.sum() * PERIOD_HOURS == pytest.approx(5.25, abs=1e-9)


# --- edge cases ---

def test_fully_charged_agent_schedules_nothing():
    """An agent that already has enough charge should produce an empty schedule."""
    agent = make_agent(energy_required_kwh=10.0, state_of_charge_kwh=15.0)
    schedule = naive_schedule(agent, np.full(PERIODS, 20.0))

    assert schedule.sum() == 0.0


def test_energy_needed_floors_at_zero():
    """energy_needed_kwh shouldn't go negative when SoC exceeds the requirement."""
    agent = make_agent(energy_required_kwh=5.0, state_of_charge_kwh=5.0)
    assert agent.energy_needed_kwh == 0.0


# --- herding determinism ---

def test_same_inputs_always_same_output():
    """
    This is the herding condition: same prices always produce the same schedule.
    With N identical agents this means all N schedules are identical, so their
    aggregate is N times the individual schedule - a perfect concentrated spike.
    """
    prices = np.random.default_rng(42).uniform(10, 40, PERIODS)
    agent = HouseholdAgent()

    sched_a = naive_schedule(agent, prices)
    sched_b = naive_schedule(agent, prices)

    np.testing.assert_array_equal(sched_a, sched_b)


def test_schedule_shape():
    """Output should always be a (48,) array."""
    schedule = naive_schedule(HouseholdAgent(), np.full(PERIODS, 20.0))
    assert schedule.shape == (PERIODS,)
