"""
market.py - Synthetic price and demand curves for the UK electricity grid.

The UK electricity market splits each day into 48 half-hour 'settlement periods'
(SP1 = 00:00-00:30, SP48 = 23:30-24:00). Octopus Agile passes the wholesale
half-hourly price directly to consumers, so a smart EV charger that sees these
prices will naturally want to charge overnight when prices are lowest.

That's the setup for the herding problem this whole project is about.
"""

import numpy as np

# 48 half-hour periods in a day - this is the fundamental grid unit
PERIODS = 48
PERIOD_HOURS = 0.5


def period_label(period: int) -> str:
    """Turn a period index (0-47) into a readable time string like '02:30'."""
    hour = period // 2
    minute = (period % 2) * 30
    return f"{hour:02d}:{minute:02d}"


def period_labels() -> list[str]:
    """All 48 labels in order, '00:00' through '23:30'."""
    return [period_label(p) for p in range(PERIODS)]


def agile_price_curve(seed: int = 42) -> np.ndarray:
    """
    Synthetic Agile-style price curve for a typical UK winter weekday.

    Shape: cheap overnight (around 12p), moderate daytime (24p), expensive
    evening spike around 17:30 (up to ~37p). Small Gaussian noise on top
    makes it look realistic without going to the extremes of real Agile
    (which can go negative in high-wind periods).

    We clip to [4p, 60p] to avoid negative prices or extreme caps that would
    need special handling in the agent code. Real Agile can hit both limits
    but they're edge cases not relevant to the herding demo.
    """
    rng = np.random.default_rng(seed)
    # hours[p] = the midpoint time of period p in hours
    hours = np.arange(PERIODS) * PERIOD_HOURS  # 0.0, 0.5, ..., 23.5

    base = 24.0  # moderate daytime price in p/kWh

    # Overnight dip: wind generation usually peaks at night, demand is low.
    # Gaussian centred at 02:30 with σ=1.5h knocks ~12p off the base price.
    overnight_dip = -12.0 * np.exp(-0.5 * ((hours - 2.5) / 1.5) ** 2)

    # Evening peak: cooking, heating, and lighting all hit at once around 17:30.
    # Adds up to ~13p on top of the base.
    evening_spike = 13.0 * np.exp(-0.5 * ((hours - 17.5) / 1.5) ** 2)

    # Small noise so adjacent periods aren't perfectly tied in price.
    # σ=1.5p is intentionally modest - enough to break ties without masking
    # the overnight/evening structure that drives agent decisions.
    noise = rng.normal(0, 1.5, PERIODS)

    prices = base + overnight_dip + evening_spike + noise
    return np.clip(prices, 4.0, 60.0)


def baseline_demand() -> np.ndarray:
    """
    Baseline national demand curve in GW for a UK winter weekday.

    'Baseline' = demand before any smart devices respond to prices.
    Think of it as what the grid looked like before EVs and smart chargers
    existed - fixed schedules, manual control, no automation.

    Loosely based on National Grid ESO winter weekday data:
    - overnight trough around 25 GW (02:00-05:00)
    - evening peak around 40 GW at 17:30 (heating + cooking + lighting)

    We use a two-cosine model (main 24h cycle + morning harmonic) rather than
    real data because it avoids a download step and keeps the baseline legible
    in plots. The exact shape doesn't matter much - what matters is that the
    overnight baseline is ~25 GW, well below what 3M EV chargers will add to it.
    """
    hours = np.arange(PERIODS) * PERIOD_HOURS

    # Main 24h cycle centred on the 17:30 evening peak.
    # Amplitude 8.5 GW gives roughly 40 GW peak, 25 GW trough.
    evening = 8.5 * np.cos(2 * np.pi * (hours - 17.5) / 24)

    # Morning harmonic (centred 09:00) adds the secondary shoulder you see
    # in real UK demand data. Without it the morning would look too flat.
    morning = 1.5 * np.cos(2 * np.pi * (hours - 9.0) / 24)

    # Mean UK winter weekday load is roughly 32 GW
    return 32.0 + evening + morning
