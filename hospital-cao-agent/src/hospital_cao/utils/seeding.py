import random
import numpy as np


def set_global_seed(seed: int):
    """Seed python, numpy and (if installed) torch. The env has its own RNG,
    seeded via env.reset(seed=...)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

