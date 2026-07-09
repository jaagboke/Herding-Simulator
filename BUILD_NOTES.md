# gridherd - Build Notes

Raw notes to feed the accompanying Medium article. Written during development,
not polished. Each phase appended after completion and sign-off.

---

## Phase 1 - Core simulation and herding demonstration

**What was built:**

- `gridherd/market.py` — synthetic 48-period Agile-style price curve and
  two-cosine baseline demand curve
- `gridherd/agents.py` — `HouseholdAgent` dataclass + `naive_schedule()`,
  the greedy sort-by-price strategy that is the herding mechanism
- `gridherd/simulate.py` — `run_simulation()` aggregator and `SimulationResult`
  dataclass with peak-reporting properties
- `gridherd/plots.py` — two-panel stacked-area figure saved to `figures/` at 150 dpi
- `gridherd/cli.py` — `gridherd run --agents N --plot` entry point
- `tests/test_agents.py` + `tests/test_simulate.py` — 19 unit tests, all passing

**Key design decisions and alternatives rejected:**

*Settlement periods as plain integers (0-47), not datetime objects.*
Array indexing stays `schedule[period]`, not `schedule[datetime(...)]`. The
`period_label()` function converts to "HH:MM" only at display time. Rejected
a pandas DatetimeIndex: it adds a dependency, requires timezone handling (UK
DST makes this genuinely tricky — Agile prices go from 48 to 46 periods in
March), and obscures the simple indexing logic that is the whole point of the
tutorial.

*O(1) aggregation: N agents → one schedule × N.*
All agents are identical and deterministic, so the aggregate is just a scalar
multiple of a single agent's schedule. Runs instantly for 3M agents. Rejected
explicit per-agent simulation (O(N) loop): 144M operations for no additional
insight at this stage. Phase 2 will need the loop when agent strategies diverge.

*Gaussian price curve, not piecewise linear.*
Smooth Gaussians for the overnight dip and evening spike are easier to explain
("prices follow a bell-curve around the cheapest window") than hard transitions.
The downside: real Agile prices are jagged and can go negative on windy nights.
We clip to [4p, 60p] to keep Phase 1 clean. This simplification is documented
in the code docstring so readers aren't misled about real Agile behaviour.

*Two-cosine baseline demand.*
Fundamental 24h cycle (evening-centred) + 12h harmonic (morning-centred)
gives the characteristic UK double-hump shape in three lines of numpy. Rejected
real National Grid data: adds a download step, requires handling missing values,
and the exact demand shape doesn't change the herding result — what matters is
that overnight baseline is ~25 GW, well below the 21 GW of agent load we're
adding.

*energy_required_kwh as absolute target SoC, state_of_charge_kwh as current SoC.*
`energy_needed = max(0, required - current)`. Default: required=14, current=0
→ needs 14 kWh → 4 half-hour periods at 7 kW. This makes herding maximally
visible: all 3M agents charge in exactly the same 4 cheapest overnight periods.
Could have made energy_needed a direct field, but the SoC framing is physically
correct and will matter in Phase 3 when SoC evolves across the feedback loop.

**Headline result (3,000,000 naive agents, default parameters, seed=42):**

- Baseline peak:    39.7 GW at 17:00 (evening — what the grid is designed around)
- New peak:         47.4 GW at 02:00 (overnight — the herding spike)
- Increase:         +7.8 GW / +19.6%
- Average cost:     163p (£1.63) per agent per charge

The peak didn't just grow — it relocated. 17:00 is the evening peak the grid is
designed to handle, with fast-response gas peakers and interconnector imports
lined up. 02:00 is the hour planners expect to be the quietest of the day, with
minimal headroom and no fast-response margin held in reserve.

Sanity check: 3M agents × 7 kW = 21 GW maximum simultaneous draw. Overnight
baseline is ~26 GW at 02:00. Observed new peak = 47.4 GW ≈ 26 + 21 = 47 GW.
The model is self-consistent: the agents genuinely concentrate all 21 GW into the
same four periods, producing the full theoretical maximum.

**Surprising things:**

The effect threshold is sharp: at 1M agents (~7 GW flexible), the overnight spike
barely registers above baseline because 7 GW < the 13.7 GW gap between overnight
baseline (~26 GW) and the evening peak (39.7 GW). At 3M agents (21 GW) the
overnight total (47.4 GW) comfortably beats the original peak. The transition from
"interesting but harmless" to "worse than the original problem" happens somewhere
around 2M agents.

The UK 2021 Smart Charging Regulations mandate "up to 10 minutes" of randomised
delay. That is a surprisingly small number. With periods of 30 minutes and agents
concentrated in 4 of them, a 10-minute jitter shifts some agents by at most one
period. Phase 2 tested this and found that even 4 hours (8 periods) of jitter
leaves a significant spike. The regulation's 10 minutes is almost certainly
insufficient on its own.

---

## Phase 2 - Additional strategies, mixed populations, cost quantification

**What was built:**

- `gridherd/agents.py` — two new expected-schedule functions:
  - `_expected_jitter_schedule()`: averages over uniform delay distribution
  - `_expected_spread_schedule()`: uniform occupancy across k_pool cheapest periods
  - `expected_per_agent_schedule()` dispatcher; `schedule_cost_pence()` utility
- `gridherd/simulate.py` — updated `run_simulation()` accepts `--strategy`;
  new `run_mixed_simulation()` for mixed populations; `SimulationResult` gains
  `strategy`, `avg_cost_pence`, `deadline_period` fields
- `gridherd/plots.py` — `plot_strategy_comparison()`: overlapping demand lines with
  inline stat table showing peak and cost per strategy; fixed the duplicate
  axvspan/text annotation bug from Phase 1
- `gridherd/cli.py` — `--strategy naive|jitter|spread|all`, `--mix`, `--jitter-max`,
  `--k-pool`; cost-of-de-synchronisation table in terminal output
- `tests/test_strategies.py` — 20 new tests covering energy conservation, pool
  uniformity, cost ordering, herding reduction, and mixed-population linearity

**Key design decisions:**

*Expected schedules, not per-agent simulation.*
Both jitter and spread return the EXPECTED per-agent schedule (averaged over the
strategy's randomness). For jitter: average over the discrete uniform distribution
of delays. For spread: the hypergeometric expectation gives fraction = n_needed/k_pool
per pool period. This keeps the O(1)-in-N property and produces clean, noise-free
demand curves for the comparison plot. Alternative (simulate each of 3M agents
individually): would take minutes and add noise without changing the expected result.

*Jitter shifts the eligible window, not the price.*
The UK regulation specifies a start-time delay, not a price perturbation. We
model it directly: agent with delay d considers periods d..deadline-1 as eligible,
then greedy-optimises within that window. This matches the regulatory intent and
produces the correct qualitative result — agents with large delays miss the cheapest
periods, but agents with small delays still herd perfectly.

*Spread uses k_pool=8 by default (twice the number of periods needed).*
With 4 periods needed and 8 in the pool, expected occupancy = 50%. Demand splits
across 8 overnight periods (2.625 GW each) instead of 4 (5.25 GW each). Rejected
k_pool = 14 (all eligible periods): would include periods at 20-24p, pushing agent
cost up substantially for diminishing de-synchronisation benefit.

**Headline result (3,000,000 agents, seed=42):**

| Strategy | New peak (GW) | Time  | vs baseline | Avg cost | Premium |
|----------|---------------|-------|-------------|----------|---------|
| Naive    | 47.4          | 02:00 | +19.6%      | 163p     | —       |
| Jitter   | 43.5          | 03:30 | +9.6%       | 194p     | +31p (+19%) |
| Spread   | 39.7          | 17:00 | +0.0%       | 200p     | +37p (+23%) |

Spread's new peak is 39.7 GW at 17:00 — identical to the baseline peak. The
herding spike is completely eliminated: overnight demand never exceeds the
original evening peak that the grid is already designed to handle. The cost
to each household is 37p extra per charge (£0.37 per fill-up, ~23% premium).

Jitter (4 hours of random delay) reduces the spike from +19.6% to +9.6%, but
does not eliminate it. The overnight peak moves from 02:00 to 03:30 but remains
above the baseline peak. This is the key finding: start-time randomisation
de-synchronises some of the fleet but not enough, because the majority of agents
still have early delays that let them access the same cheapest overnight window.

The UK's 10-minute mandatory delay corresponds to 0.33 periods. With jitter
reduced from 8 periods (4 hours) to 0.33 periods, the de-synchronisation effect
would be negligible — essentially identical to naive herding. The regulation
alone is not sufficient; the spread mechanism (or an equivalent: staggered
deadlines, diverse user profiles, TOU tariff diversity) is what actually works.

**Mixed populations barely help:**

Testing a 50/50 naive+jitter fleet (1.5M of each): peak 44.7 GW at 03:30,
vs 47.4 GW pure-naive and 43.5 GW pure-jitter. The improvement is marginal
because the 1.5M naive agents still herd perfectly in the cheapest overnight
cluster — they just share the spike with a jitter sub-fleet that partially
de-synchronises. Halving the naive fraction doesn't halve the spike; it reduces
it by less than 3 GW because the naive component retains full correlated demand.
Mixed populations are not a substitute for fleet-wide strategy change.

**Spread eliminates the spike completely:**

The spread strategy's new peak is 39.7 GW at 17:00 — identical to the baseline
peak. The herding spike is gone: overnight demand never exceeds the original
evening ceiling the grid is already designed to handle. Spread beats jitter by a
full 3.8 GW (39.7 vs 43.5 GW). The mechanism: spread breaks the herding by using
a contiguous pool of 8 cheap periods and distributing demand uniformly across all
8 (rather than piling into the 4 cheapest). Each pool period gets only 10.5 GW of
agent demand instead of 21 GW, never tipping over the evening peak.

**Cost trade-off confirmed (163p naive, 194p jitter, 200p spread):**

The premium for spread over naive is 37p per charge (23%). Over a full year of
daily charges: £135 per household. The 10-minute UK mandate adds less than 1%
de-synchronisation benefit — essentially no premium but also essentially no
benefit. The honest comparison is: 37p per charge per household vs the
alternative of building additional grid capacity or activating expensive fast-
response balancing reserves every overnight charging window, once EVs become
widespread.

**Implementation note — zigzag fix:**

The spread curve had a visible zigzag at 00:30-01:00. Root cause: the ±1.5p
Gaussian noise in the synthetic price curve inverted the natural ordering of
periods 1 (00:30) and 2 (01:00), making the cheapest-8 pool skip period 2 and
include period 1 instead — a non-contiguous set {1,3,4,5,6,7,8,9} producing
demand step-down at period 2. Fixed by switching from "k individually cheapest"
to "cheapest contiguous window of k" (sliding-window convolution). Both more
correct and more realistic: devices charge across "the cheapest 4-hour block",
not scattered non-adjacent slots. The zigzag was NOT sampling noise — it was a
deterministic artefact of the noise model interacting with period ordering.

**Surprising things:**

The spread peak is not just below the naive spike — it equals the baseline
peak exactly (39.7 GW at 17:00), meaning the agent population adds zero
peak increase to the grid. The overnight periods now handle 2.6 GW of agent
demand each on top of ~26-28 GW baseline, peaking around 30 GW: comfortably
below the 39.7 GW evening ceiling.

The jitter peak migrates from 02:00 (naive) to 03:30. This is not immediately
obvious but makes sense on reflection: with jitter, agents with longer delays
are pushed later into the overnight window. The most popular period shifts
toward the later edge of the cheap band because later periods are accessible
to more delay values. The effect is like gravity toward the right edge of the
eligible window.

The cost premium for spread (37p, £0.37 per charge) is small relative to the
grid benefit. At current UK electricity prices, 37p is about 15-20 minutes of
charging. If this saved even one balancing market action by National Grid ESO,
the savings to the system would dwarf the household premium by orders of magnitude.
This is a textbook case of a positive externality not captured in the private cost.

---

## Phase 3 - Price feedback loop and spike migration

**What was built:**

- `gridherd/simulate.py` — `FeedbackSimulation` dataclass and
  `run_feedback_simulation()`: runs N rounds of naive scheduling, updating the
  effective price signal after each round using a linear demand-response model
- `gridherd/plots.py` — two new figures:
  - `plot_feedback_rounds()`: 2×3 small-multiples static figure, rounds 1,2,3,4,5,10
  - `plot_feedback_animation()`: animated GIF via FuncAnimation + PillowWriter (2 fps)
- `gridherd/cli.py` — `--feedback`, `--rounds N`, `--sensitivity S` arguments
- `tests/test_feedback.py` — 5 tests covering zero-sensitivity regression guard,
  price response, spike migration, round count, and baseline invariance

**Key design decisions:**

*Linear price feedback: effective_price[p] = base_price[p] + sensitivity × total_demand[p]*

Sensitivity has units pence/GW. The linear model is a first-order approximation;
real balancing-market imbalance pricing is a step function — reserve tiers activate
at specific MW thresholds, making the effective cost function highly nonlinear and
discontinuous. Linear is instructive but understates the nonlinearity of the real
incentive. Default sensitivity = 0.5 p/GW: a 21 GW overnight spike adds ~23.5p to
those periods' effective prices, enough to push naive agents away.

*Prices reset to base each round (not compounded):*
Effective price[k] = base_price + sensitivity × demand[k-1]. Compounding (each
round building on the last round's already-inflated price) would diverge rapidly
and obscure the oscillation. The non-compounding form is closer to how balancing
markets actually work: settlement prices are recalculated from wholesale spot
prices each half-hour.

*Naive strategy only for the feedback loop:*
Jitter and spread would also migrate, but with more complex dynamics (the pool
shifts, not just the peak). Naive agents show the clearest oscillation because
their herding is total: all 3M move together, creating the maximum price signal,
which causes all 3M to move together again in the opposite direction.

**Headline result (3,000,000 naive agents, sensitivity=0.5 p/GW, 10 rounds, seed=42):**

| Round | Peak (GW) | Time  | vs baseline |
|-------|-----------|-------|-------------|
| 1     | 47.4      | 02:00 | +7.8 GW     |
| 2     | 49.0      | 01:00 | +9.3 GW     |
| 3     | 47.4      | 02:00 | +7.8 GW     |
| 4     | 49.0      | 01:00 | +9.3 GW     |
| ...   | ...       | ...   | ...         |
| 10    | 49.0      | 01:00 | +9.3 GW     |

The system settles immediately into a 2-cycle oscillation: round 1 spikes at
02:00 (period 4, as in Phase 1). The feedback makes period 4 expensive, so
round 2's agents move to 01:00 (period 2) — which wasn't in the original
cheapest-4 pool but is now cheapest given the price adjustment. The round 2
peak of 49.0 GW is HIGHER than round 1 (47.4 GW): the feedback pushes agents
into a period with higher baseline demand (~27.9 GW vs 26.3 GW), so the total
is worse. Then round 2's spike makes period 2 expensive, agents return to
period 4 in round 3, and the cycle repeats indefinitely.

**Why the spike migrates but does not dissolve:**

The fundamental issue is that price feedback changes the cheapest period's
relative rank, but all agents see the same updated price signal and respond
identically. The herding mechanism is intact — only its destination changes.
There is no mechanism that would cause agents to spread out across the
overnight window; they still sort by price and pick the cheapest N periods
in unison. A price signal that successfully deters charging in period X merely
designates period Y as the next victim.

Contrast with the spread strategy (Phase 2): spread works because it IGNORES
price differences within the pool and forces a uniform random draw. The
feedback loop has no equivalent mechanism — it still uses pure price ranking,
which preserves the herding structure.

**Surprising things:**

The feedback makes the grid situation WORSE in round 2 (49.0 GW > 47.4 GW).
This is counterintuitive — the price signal is supposed to discourage the spike,
not increase it. The effect happens because the round-1 feedback prices are
computed from total demand (including 26 GW baseline), so the cheapest
remaining slot (01:00) happens to have slightly higher baseline demand than
02:00. The agents' new spike lands on top of more baseline demand than their
original spike. An open-loop price-feedback mechanism can actively worsen the
peak before it even begins to equilibrate.

The 2-cycle oscillation period is striking: the spike alternates between
adjacent overnight periods (01:00 and 02:00) forever, with no convergence
toward the equilibrium (which would require the spike to dissolve into spread
demand across many periods). This is consistent with theory: for linear price
feedback and a greedy fleet, the equilibrium requires agents to randomise over
periods, which greedy agents never do. The fixed point does not exist under
this strategy.

Figures saved to:
- `figures/phase3_feedback_rounds.png` — small-multiples static (rounds 1,2,3,4,5,10)
- `figures/phase3_feedback_animation.gif` — animated GIF (10 rounds, 2 fps)
