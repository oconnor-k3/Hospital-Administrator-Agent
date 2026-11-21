# MDP Specification (prototype v0.1)

Everything below is implemented in `src/hospital_cao/env/`. Treat it as a draft to argue about, not a final design.

## Time and episode
- 1 tick = 1 simulated hour. Episode = 168 ticks (one week). Terminates at the horizon.
- Several decisions can happen within one tick; time advances only on the "end turn" action.

## State (observation, `obs_dim = 5 + 3K`, all in [0, 1])
| Feature | Meaning |
|---|---|
| `obs[0..2]` | free doctors, nurses, beds (fraction of total) |
| `obs[3]` | time fraction of episode |
| `obs[4]` | total queue length / 50 (capped) |
| per slot k < K: occupied, acuity/2, min(wait,24)/24 | the K visible waiting patients, ordered by `queue_order` (`arrival` = oldest first, `acuity` = sickest first) |

Hidden from the agent: exact remaining treatment time of patients in care, anything beyond the first K queued patients.

## Actions (`Discrete(K+1)`)
- `k in [0, K)`: start treating the patient in visible slot k (needs resources, see below).
- `K`: end turn, advance one tick.
- Invalid choice (empty slot or not enough resources): ignored, reward -0.01, time does not advance.
- `action_mask()` marks the valid actions. Use it in the agent (mask logits / Q-values).
- After 20 decisions in one tick the env forces time forward (prevents idle loops).

## Patients and resources
| Acuity | Needs (doctors, nurses, beds) | Mean treatment (ticks) | Waits before worsening |
|---|---|---|---|
| 0 low | 1, 0, 0 | 1 | 8 -> becomes medium |
| 1 medium | 1, 1, 1 | 3 | 5 -> becomes critical |
| 2 critical | 1, 2, 1 | 6 | 3 -> dies |

Resources are held for the whole treatment and released on discharge. Treatment time is `1 + Poisson(mean - 1)`, sampled when treatment starts (so a patient who deteriorated while waiting gets the longer treatment).

## Transitions
Order within an advance: utilisation recorded -> treatments progress and discharge -> waiting patients age, deteriorate, or die -> `t += 1` -> new arrivals -> reward computed.
Arrivals: Poisson with rate `arrival_rate * (1 + 0.4 * sin(2*pi*(t mod 24)/24))`; acuity drawn from (0.50, 0.35, 0.15).

## Reward (swap via `HospitalConfig.objective`)
- `risk_weighted`: +1/+3/+10 per discharge by acuity, -50 per death, -0.1 x acuity weight per waiting patient per tick.
- `throughput`: +1 per discharge.
- `fairness`: +0.5 per discharge, -0.002 x wait^2 per waiting patient, -5 per death.

## Helpers for agents
- `env.action_mask()`: boolean mask over the K+1 actions.
- `env.tabular_state(max_count=3)`: hashable `(free_doctors, free_nurses, free_beds, n_low, n_medium, n_critical)` with counts capped. Use with `configs/tiny.yaml` for tabular Q-learning (set `queue_order: acuity` so slot 0 = most urgent).
- `HospitalEnv(cfg, record_history=True)`: logs one record per tick in `env.history` (queue by acuity, busy resources, discharges, deaths, deteriorations, reward).
- `env.render_text()`: one-snapshot text view for debugging.

## Baselines (`src/hospital_cao/baselines/`)
Rules use patient attributes, not slot position, so they behave the same under either `queue_order`.
- `random`: uniform over valid starts and end-turn.
- `fifo`: oldest arrival first.
- `acuity_first`: sickest first, ties by wait.
- `earliest_deadline`: fewest ticks left before the patient deteriorates or dies.

## Baseline results (30 seeded episodes, risk_weighted return, higher is better)
| Config | random | fifo | acuity_first | earliest_deadline |
|---|---|---|---|---|
| default (1.0 arrivals/hr) | -122 | 313 | 294 | **343** |
| overloaded (1.5/hr) | -1352 | **-88** | -1281 | -495 |
| tiny (tabular testbed) | -155 | -45 | -52 | **-43** |
