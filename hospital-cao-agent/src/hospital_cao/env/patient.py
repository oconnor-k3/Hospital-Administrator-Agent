from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Patient:
    pid: int
    acuity: int                 # 0 low, 1 medium, 2 critical
    arrival_t: int
    wait: int = 0               # total ticks waited
    level_wait: int = 0         # ticks waited at current acuity
    remaining: int = 0          # ticks of treatment left
    start_t: Optional[int] = None
    held: Tuple[int, int, int] = (0, 0, 0)   # resources currently held

# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this project, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------
