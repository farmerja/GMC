from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np


DEFAULT_MC_SEED = 42
_NUMPY_RANDOM_LOCK = threading.RLock()


def resolve_mc_seed(seed: Optional[int], default: int = DEFAULT_MC_SEED) -> int:
    resolved = int(default if seed is None else seed)
    resolved %= (2**31 - 1)
    return resolved if resolved != 0 else 1


@contextmanager
def seeded_numpy_execution(seed: Optional[int], default: int = DEFAULT_MC_SEED) -> Iterator[int]:
    seed_i = resolve_mc_seed(seed, default=default)
    with _NUMPY_RANDOM_LOCK:
        np_state = np.random.get_state()
        np.random.seed(seed_i)
        try:
            yield seed_i
        finally:
            np.random.set_state(np_state)
