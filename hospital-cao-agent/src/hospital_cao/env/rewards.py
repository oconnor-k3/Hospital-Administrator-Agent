"""Pluggable reward functions. Each takes (env, events) where events holds the
patients that were started / discharged / died / deteriorated during the tick
that just elapsed. Swap objectives by changing HospitalConfig.objective."""

# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this project, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------

ACUITY_WEIGHT = (1.0, 3.0, 10.0)


def risk_weighted(env, ev):
    """Prioritise the sickest: value discharges by acuity, heavily penalise
    deaths, and charge a waiting cost that scales with acuity."""
    r = sum(ACUITY_WEIGHT[p.acuity] for p in ev["discharged"])
    r -= 50.0 * len(ev["deaths"])
    r -= 0.1 * sum(ACUITY_WEIGHT[p.acuity] for p in env.queue)
    return r


def throughput(env, ev):
    """Maximise patients treated, ignoring who they are."""
    return float(len(ev["discharged"]))


def fairness(env, ev):
    """Penalise long waits superlinearly so no one is left waiting forever."""
    r = 0.5 * len(ev["discharged"])
    r -= 0.002 * sum(p.wait ** 2 for p in env.queue)
    r -= 5.0 * len(ev["deaths"])
    return r


REWARDS = {"risk_weighted": risk_weighted, "throughput": throughput, "fairness": fairness}


def get_reward_fn(name):
    if name not in REWARDS:
        raise ValueError(f"Unknown objective '{name}'. Options: {list(REWARDS)}")
    return REWARDS[name]
