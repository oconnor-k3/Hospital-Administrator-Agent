"""All knobs for the three variability axes live here:
(1) healthcare situation (arrival rate, acuity mix),
(2) hospital conditions (doctors, nurses, beds),
(3) objective (which reward function is used).

Configs can be built in code, or loaded/merged from YAML files:
    cfg = load_config("configs/small_hospital.yaml", "configs/objectives/fairness.yaml",
                      arrival_rate=1.2)      # keyword overrides win
"""

# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this project, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Tuple

import yaml

ACUITY_NAMES = ("low", "medium", "critical")
QUEUE_ORDERS = ("arrival", "acuity")


@dataclass
class HospitalConfig:
    # --- hospital conditions ---
    n_doctors: int = 4
    n_nurses: int = 6
    n_beds: int = 8

    # --- healthcare situation ---
    arrival_rate: float = 1.0            # mean arrivals per tick (1 tick = 1 hour); 1.5 = overloaded stress test
    diurnal_amplitude: float = 0.4       # arrival rate swings +/- this fraction over 24h
    acuity_probs: Tuple[float, ...] = (0.50, 0.35, 0.15)

    # Resources needed to START treatment: (doctors, nurses, beds), by acuity.
    requirements: Tuple[Tuple[int, int, int], ...] = ((1, 0, 0), (1, 1, 1), (1, 2, 1))
    # Mean treatment duration in ticks, by acuity.
    mean_duration: Tuple[float, ...] = (1.0, 3.0, 6.0)
    # Ticks a patient can wait at a given acuity before getting worse.
    # low -> medium, medium -> critical, critical -> dies.
    deteriorate_after: Tuple[int, ...] = (8, 5, 3)

    # --- episode / interface ---
    horizon: int = 168                   # one simulated week
    queue_view: int = 10                 # K: waiting patients visible/actionable
    # How the K visible slots are ordered:
    #   "arrival": oldest first (slot 0 = longest in queue)
    #   "acuity" : sickest first, ties broken by wait (slot 0 = most urgent)
    queue_order: str = "arrival"
    max_decisions_per_tick: int = 20     # safety valve against idle-looping
    invalid_action_penalty: float = 0.01

    # --- objective ---
    objective: str = "risk_weighted"     # risk_weighted | throughput | fairness

    def __post_init__(self):
        n = len(ACUITY_NAMES)
        if abs(sum(self.acuity_probs) - 1.0) > 1e-6:
            raise ValueError("acuity_probs must sum to 1")
        for name in ("acuity_probs", "requirements", "mean_duration", "deteriorate_after"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have {n} entries (one per acuity level)")
        if any(len(r) != 3 for r in self.requirements):
            raise ValueError("each requirement must be (doctors, nurses, beds)")
        if self.queue_order not in QUEUE_ORDERS:
            raise ValueError(f"queue_order must be one of {QUEUE_ORDERS}")
        if self.queue_view < 1 or self.horizon < 1:
            raise ValueError("queue_view and horizon must be >= 1")
        # Every acuity level must be treatable by a fully free hospital.
        caps = (self.n_doctors, self.n_nurses, self.n_beds)
        for a, req in enumerate(self.requirements):
            if any(r > c for r, c in zip(req, caps)):
                raise ValueError(f"acuity {a} needs more resources than the hospital has")

    def with_overrides(self, **kw):
        return replace(self, **_tuplify(kw))

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _tuplify(x):
    """YAML gives lists; the dataclass wants tuples (recursively)."""
    if isinstance(x, dict):
        return {k: _tuplify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return tuple(_tuplify(v) for v in x)
    return x


def load_config(*paths, **overrides) -> HospitalConfig:
    """Merge YAML files left-to-right (later files win), then apply overrides."""
    merged = {}
    for p in paths:
        with open(Path(p)) as f:
            data = yaml.safe_load(f) or {}
        merged.update(data)
    merged.update(overrides)
    known = {f.name for f in fields(HospitalConfig)}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return HospitalConfig(**_tuplify(merged))
