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

        # Inspect checkpoint zip metadata to detect recurrent policies reliably.
        # Note: PPO.load can deserialize recurrent policies without raising an exception,
        # which would otherwise leave is_recurrent=False and drop recurrent state tracking.
        is_recurrent_ckpt = False
        try:
            import json
            import zipfile

            with zipfile.ZipFile(self.checkpoint_path, "r") as archive:
                if "data" in archive.namelist():
                    data = json.loads(archive.read("data").decode("utf-8"))
                    policy_info = str(data.get("policy_class", "")).lower()
                    if "recurrent" in policy_info:
                        is_recurrent_ckpt = True
        except Exception:
            pass

        if is_recurrent_ckpt:
            try:
                self.model = RecurrentPPO.load(str(self.checkpoint_path), device=self.device)
                self.is_recurrent = True
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load recurrent checkpoint '{self.checkpoint_path}': {exc}"
                ) from exc
        else:
            try:
                self.model = PPO.load(str(self.checkpoint_path), device=self.device)
                # Verify that the loaded policy is indeed non-recurrent
                if "recurrent" in self.model.policy.__class__.__name__.lower():
                    self.model = RecurrentPPO.load(str(self.checkpoint_path), device=self.device)
                    self.is_recurrent = True
            except Exception:
                # Fallback to RecurrentPPO if PPO.load failed
                try:
                    self.model = RecurrentPPO.load(str(self.checkpoint_path), device=self.device)
                    self.is_recurrent = True
                except Exception as exc:
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
