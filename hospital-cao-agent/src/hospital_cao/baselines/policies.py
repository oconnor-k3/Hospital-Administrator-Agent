"""Hand-written heuristics the RL agent has to beat. Each policy picks, among
visible waiting patients that can start right now, one by its own rule; if
none can start it ends the turn. Rules use patient attributes, not slot
position, so results don't depend on HospitalConfig.queue_order."""
import numpy as np


class _Base:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def candidates(self, env):
        mask = env.action_mask()
        return [k for k in range(env.K) if mask[k]]

    def act(self, env):
        raise NotImplementedError


class RandomPolicy(_Base):
    """Uniformly random among valid starts and 'end turn'."""
    def act(self, env):
        opts = self.candidates(env) + [env.K]
        return int(self.rng.choice(opts))


class FIFOPolicy(_Base):
    """Oldest arrival first (equivalently, longest total wait first)."""
    def act(self, env):
        c = self.candidates(env)
        if not c:
            return env.K
        q = env.visible_queue()
        return min(c, key=lambda k: (q[k].arrival_t, q[k].pid))


class AcuityFirstPolicy(_Base):
    """Sickest first, ties broken by longer wait. Greedy over what is affordable."""
    def act(self, env):
        c = self.candidates(env)
        if not c:
            return env.K
        q = env.visible_queue()
        return max(c, key=lambda k: (q[k].acuity, q[k].wait))


class EarliestDeadlinePolicy(_Base):
    """Most urgent by time-to-deterioration: ticks left before the patient gets
    worse (or dies). Mixes acuity and wait, and protects cheap patients from
    escalating into expensive ones."""
    def act(self, env):
        c = self.candidates(env)
        if not c:
            return env.K
        q = env.visible_queue()
        slack = lambda p: env.cfg.deteriorate_after[p.acuity] - p.level_wait
        return min(c, key=lambda k: (slack(q[k]), -q[k].acuity))
