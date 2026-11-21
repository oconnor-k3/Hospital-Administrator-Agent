import numpy as np
from hospital_cao.env import HospitalEnv


def run_episode(env, policy, seed):
    """Play one seeded episode with policy.act(env) -> action. Returns metrics."""
    env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        _, r, done, _, _ = env.step(policy.act(env))
        total += r
    out = env.summary()
    out["return"] = total
    return out


def evaluate_runs(config, policy, n_episodes=20, seed0=0):
    """Per-episode metrics (list of dicts). Same seeds => same patient streams
    across policies, so comparisons are paired."""
    env = HospitalEnv(config)
    return [dict(run_episode(env, policy, seed0 + i), seed=seed0 + i) for i in range(n_episodes)]


def evaluate(config, policy, n_episodes=20, seed0=0):
    """Mean metrics over seeded episodes."""
    runs = evaluate_runs(config, policy, n_episodes, seed0)
    keys = [k for k in runs[0] if k != "seed"]
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


def evaluate_with_std(config, policy, n_episodes=20, seed0=0):
    """Mean and std per metric, for error bars."""
    runs = evaluate_runs(config, policy, n_episodes, seed0)
    keys = [k for k in runs[0] if k != "seed"]
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
            for k in keys}


def record_episode(config, policy, seed=0):
    """Run one episode with per-tick history on. Returns (history, summary)."""
    env = HospitalEnv(config, record_history=True)
    summary = run_episode(env, policy, seed)
    return env.history, summary
