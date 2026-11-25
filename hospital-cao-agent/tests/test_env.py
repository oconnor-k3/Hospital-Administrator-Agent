from hospital_cao.env import HospitalEnv, HospitalConfig
from hospital_cao.baselines import ALL_BASELINES

# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this project, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------

def test_resources_conserved_and_nonnegative():
    env = HospitalEnv(HospitalConfig(), seed=0)
    env.reset()
    pol = ALL_BASELINES["random"](seed=1)
    done = False
    while not done:
        _, _, done, _, _ = env.step(pol.act(env))
        held = [sum(p.held[i] for p in env.in_treatment) for i in range(3)]
        total = [env.cfg.n_doctors, env.cfg.n_nurses, env.cfg.n_beds]
        assert all(f >= 0 for f in env.free)
        assert [f + h for f, h in zip(env.free, held)] == total


def test_patients_accounted_for():
    env = HospitalEnv(HospitalConfig(), seed=0)
    env.reset()
    pol = ALL_BASELINES["acuity_first"]()
    done = False
    while not done:
        _, _, done, _, _ = env.step(pol.act(env))
    s = env.stats
    in_system = len(env.queue) + len(env.in_treatment)
    assert s["arrived"] == sum(s["discharged"]) + s["deaths"] + in_system


def test_seeded_runs_reproducible():
    def run():
        env = HospitalEnv(HospitalConfig())
        env.reset(seed=7)
        pol = ALL_BASELINES["fifo"]()
        total, done = 0.0, False
        while not done:
            _, r, done, _, _ = env.step(pol.act(env))
            total += r
        return total
    assert run() == run()


def test_invalid_action_does_not_advance_time():
    env = HospitalEnv(HospitalConfig())
    env.reset(seed=0)
    t0 = env.t
    _, r, _, _, info = env.step(env.K - 1 if not env.action_mask()[env.K - 1] else 0)
    assert env.t == t0


def test_idle_loop_is_forced_forward():
    env = HospitalEnv(HospitalConfig(max_decisions_per_tick=5))
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.K - 1)
    assert env.t >= 1


def test_obs_shape_and_range():
    env = HospitalEnv(HospitalConfig())
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    assert info["action_mask"].shape == (env.n_actions,)


# ---------------------------------------------------------------- new tests
from pathlib import Path
from hospital_cao.env import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_all_yaml_configs_load():
    for f in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(f)
        HospitalEnv(cfg).reset(seed=0)
    for f in sorted((CONFIG_DIR / "objectives").glob("*.yaml")):
        load_config(CONFIG_DIR / "small_hospital.yaml", f)


def test_config_merge_and_override_order():
    cfg = load_config(CONFIG_DIR / "small_hospital.yaml",
                      CONFIG_DIR / "objectives" / "fairness.yaml", arrival_rate=0.7)
    assert cfg.objective == "fairness" and cfg.arrival_rate == 0.7 and cfg.n_doctors == 4


def test_bad_config_rejected():
    for bad in ({"acuity_probs": (0.5, 0.5, 0.5)}, {"queue_order": "random"},
                {"n_beds": 0}, {"mean_duration": (1.0, 2.0)}):
        try:
            HospitalConfig().with_overrides(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad config {bad}")


def test_acuity_order_puts_sickest_first():
    env = HospitalEnv(HospitalConfig(queue_order="acuity", arrival_rate=3.0))
    env.reset(seed=3)
    for _ in range(4):               # let a queue build without treating anyone
        env.step(env.K)
    view = env.visible_queue()
    keys = [(-p.acuity, -p.wait) for p in view]
    assert keys == sorted(keys)


def test_acuity_order_action_starts_that_patient():
    env = HospitalEnv(HospitalConfig(queue_order="acuity", arrival_rate=3.0))
    env.reset(seed=3)
    for _ in range(3):
        env.step(env.K)
    target = env.visible_queue()[0]
    mask = env.action_mask()
    if mask[0]:
        env.step(0)
        assert target in env.in_treatment


def test_history_recorded_per_tick():
    env = HospitalEnv(HospitalConfig(horizon=24), record_history=True)
    env.reset(seed=0)
    pol = ALL_BASELINES["fifo"]()
    done = False
    while not done:
        _, _, done, _, _ = env.step(pol.act(env))
    assert len(env.history) == 24
    assert sum(h["deaths"] for h in env.history) == env.stats["deaths"]


def test_tabular_state_is_hashable_and_bounded():
    cfg = load_config(CONFIG_DIR / "tiny.yaml")
    env = HospitalEnv(cfg)
    env.reset(seed=0)
    seen = set()
    pol = ALL_BASELINES["acuity_first"]()
    done = False
    while not done:
        s = env.tabular_state()
        seen.add(s)
        assert all(0 <= v for v in s) and all(c <= 3 for c in s[3:])
        _, _, done, _, _ = env.step(pol.act(env))
    assert len(seen) > 1


def test_every_objective_runs_and_returns_finite():
    import math
    for obj in ("risk_weighted", "throughput", "fairness"):
        env = HospitalEnv(HospitalConfig(objective=obj, horizon=48))
        env.reset(seed=1)
        pol = ALL_BASELINES["fifo"]()
        done, total = False, 0.0
        while not done:
            _, r, done, _, _ = env.step(pol.act(env))
            total += r
        assert math.isfinite(total)
