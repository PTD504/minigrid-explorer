from __future__ import annotations

from typing import Any

import numpy as np

from explorer.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Uniform random baseline selecting uniformly from Discrete(3) navigation actions.

    Acts as the naive lower bound for exploration coverage and survival.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset RNG state if seed was provided."""
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        """Sample an action uniformly from {0: left, 1: right, 2: forward}."""
        return int(self.rng.integers(0, 3))

