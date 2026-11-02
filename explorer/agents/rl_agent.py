from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from explorer.agents.base import BaseAgent


class RLAgent(BaseAgent):
    """Autonomous exploration agent powered by a trained RL checkpoint.

    Supports both standard feedforward (PPO) and recurrent (RecurrentPPO) models.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        deterministic: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        self.deterministic = deterministic
        self.device = device

        self.model: Any = None
        self.is_recurrent: bool = False

        # Attempt loading with PPO first; fallback to RecurrentPPO if policy is recurrent
        try:
            self.model = PPO.load(str(self.checkpoint_path), device=self.device)
        except (ValueError, TypeError, RuntimeError, KeyError):
            try:
                self.model = RecurrentPPO.load(str(self.checkpoint_path), device=self.device)
                self.is_recurrent = True
            except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint '{self.checkpoint_path}' with PPO or RecurrentPPO: {exc}"
                ) from exc

        self.lstm_states: Any = None
        self.episode_starts: np.ndarray = np.ones((1,), dtype=bool)

    def reset(self) -> None:
        """Reset internal recurrent hidden states on episode start."""
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        """Select action using policy inference."""
        if self.is_recurrent:
            action, self.lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=self.deterministic,
            )
            self.episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)

        if isinstance(action, (np.ndarray, list)):
            return int(action.item() if hasattr(action, "item") else action[0])
        return int(action)
