# Hospital CAO Agent (environment + baselines)

Simulated hospital resource-allocation MDP for a reinforcement-learning agent. See `docs/mdp_spec.md` for state, action, transition, and reward definitions, baseline results, and known limitations.

# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this file, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install numpy pyyaml matplotlib      # environment, configs, plots
# pip install torch                      # later, for DQN
```

## Run
```
python tests/run_tests.py                                   # invariant tests (or `pytest tests`)
python scripts/run_baselines.py                             # all baselines x all objectives, default config
python scripts/run_baselines.py --config configs/overloaded.yaml --episodes 50 --tag overloaded
python scripts/demo_episode.py --policy earliest_deadline   # text trace + episode plot
```
Outputs go to `results/figures/` and `results/logs/`.

## Configs
`configs/*.yaml` define hospitals and load (`small_hospital`, `large_hospital`, `overloaded`, `tiny`); `configs/objectives/*.yaml` pick the reward. Merge any combination:
```python
from hospital_cao.env import HospitalEnv, load_config
cfg = load_config("configs/small_hospital.yaml", "configs/objectives/fairness.yaml", arrival_rate=1.2)
env = HospitalEnv(cfg)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)   # info["action_mask"] marks valid actions
```

## Layout
```
configs/                      hospital, load, and objective YAML files
src/hospital_cao/env/         environment, config loading, patients, reward functions
src/hospital_cao/baselines/   random, FIFO, acuity-first, earliest-deadline
src/hospital_cao/evaluation/  seeded evaluation, episode recording, plots
src/hospital_cao/utils/       seeding, result saving
scripts/                      entry points
tests/                        environment invariants and config tests
docs/                         MDP specification
results/                      generated figures and logs
```

## Next
Tabular Q-learning on `configs/tiny.yaml` (use `env.tabular_state()`), then DQN with action masking in `src/hospital_cao/agents/`.
