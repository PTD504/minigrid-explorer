from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all exploration agents and baselines."""

    @abstractmethod
    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        """Select an action given current observation and optional step info.

        Args:
            obs: Observation dictionary (from ExploreEnv or ObservationMemoryWrapper).
            info: Optional info dictionary returned by step() or reset().

        Returns:
            action: Discrete action index (0=left, 1=right, 2=forward).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal agent state/memory at the start of a new episode."""
