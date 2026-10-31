from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym

from explorer.utils.config import get_reward_config


class RewardEngine:
    """Computes dense exploration rewards with anti-reward hacking safeguards.

    Mathematical formulation:
        R_t = w_reveal * ΔN_t + w_poi * ΔP_t - c_step - c_bump * 1[bump_t]
              - c_loop * 1[loop_t] - c_death * 1[death_t]
              + b_done * (1 - t/T) * 1[coverage_done_t]

    Anti-hacking invariants & guarantees:
    1. Telescoping property: newly revealed walkable tiles (ΔN_t) are non-overlapping,
       preventing reward farming by moving back and forth or turning.
    2. Strict anti-suicide invariant: c_death > c_step * max_steps across all difficulties.
    3. Loop detection: penalizes repeating a recent (x, y, dir) state within
       a circular buffer of length `loop_window` without gaining new information
       (ΔN_t == 0 and ΔP_t == 0).
    4. Information boundary: zero normalization against true map size |W| or total POIs.
    5. Diagnostic telemetry: returns an exact component breakdown dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_reward_config()
        self.w_reveal: float = float(self.config.get("w_reveal", 0.02))
        self.w_poi: float = float(self.config.get("w_poi", 1.0))
        self.c_step: float = float(self.config.get("c_step", 0.002))
        self.c_bump: float = float(self.config.get("c_bump", 0.02))
        self.c_loop: float = float(self.config.get("c_loop", 0.01))
        self.c_death: float = float(self.config.get("c_death", 2.0))
        self.b_done: float = float(self.config.get("b_done", 1.0))
        self.loop_window_size: int = int(self.config.get("loop_window", 20))

        # Circular buffer with O(1) state frequency tracking
        self.state_history: deque[tuple[int, int, int]] = deque(maxlen=self.loop_window_size)
        self.state_counts: dict[tuple[int, int, int], int] = {}

    def reset(self, initial_state: tuple[int, int, int] | None = None) -> None:
        """Reset state history for a new episode.

        Args:
            initial_state: Optional (x, y, dir) state of the agent at reset.
        """
        self.state_history.clear()
        self.state_counts.clear()
        if initial_state is not None and self.loop_window_size > 0:
            self.state_history.append(initial_state)
            self.state_counts[initial_state] = 1

    def compute_reward(
        self,
        agent_pos: tuple[int, int],
        agent_dir: int,
        delta_revealed: int,
        delta_pois: int,
        bump: bool,
        terminated: bool,
        end_reason: str | None,
        step_count: int,
        max_steps: int,
        coverage_done: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Calculate total step reward and diagnostic breakdown.

        Args:
            agent_pos: Current (x, y) coordinates of the agent.
            agent_dir: Current direction index (0=right, 1=down, 2=left, 3=up).
            delta_revealed: Count of newly observed walkable tiles (ΔN_t).
            delta_pois: Count of newly discovered POIs (ΔP_t).
            bump: Whether forward movement bumped into a wall or map boundary.
            terminated: Whether episode ended due to an environmental trigger.
            end_reason: Termination reason ("death", "coverage", "timeout", etc.).
            step_count: Current episode step count (t).
            max_steps: Maximum allowed episode steps (T).
            coverage_done: Whether coverage threshold was newly achieved this step.

        Returns:
            total_reward: Computed scalar step reward R_t.
            breakdown: Detailed dictionary of all component contributions.
        """
        curr_state = (int(agent_pos[0]), int(agent_pos[1]), int(agent_dir))

        # 1. State loop detection with O(1) lookup
        is_loop = False
        if self.loop_window_size > 0:
            if (curr_state in self.state_counts) and (delta_revealed == 0) and (delta_pois == 0):
                is_loop = True

            # Evict oldest entry from counts if deque is at maxlen
            if (
                self.state_history.maxlen is not None
                and len(self.state_history) == self.state_history.maxlen
                and len(self.state_history) > 0
            ):
                old_state = self.state_history.popleft()
                cnt = self.state_counts[old_state] - 1
                if cnt <= 0:
                    del self.state_counts[old_state]
                else:
                    self.state_counts[old_state] = cnt

            self.state_history.append(curr_state)
            self.state_counts[curr_state] = self.state_counts.get(curr_state, 0) + 1

        # 2. Individual reward and penalty components
        reward_reveal = self.w_reveal * float(delta_revealed)
        reward_poi = self.w_poi * float(delta_pois)
        penalty_step = -self.c_step
        penalty_bump = -self.c_bump if bump else 0.0
        penalty_loop = -self.c_loop if is_loop else 0.0
        penalty_death = -self.c_death if (end_reason == "death") else 0.0

        bonus_done = 0.0
        if coverage_done and max_steps > 0:
            remaining_ratio = max(0.0, 1.0 - float(step_count) / float(max_steps))
            bonus_done = self.b_done * remaining_ratio

        total_reward = (
            reward_reveal
            + reward_poi
            + penalty_step
            + penalty_bump
            + penalty_loop
            + penalty_death
            + bonus_done
        )

        breakdown = {
            "reward_reveal": float(reward_reveal),
            "reward_poi": float(reward_poi),
            "penalty_step": float(penalty_step),
            "penalty_bump": float(penalty_bump),
            "penalty_loop": float(penalty_loop),
            "penalty_death": float(penalty_death),
            "bonus_done": float(bonus_done),
            "total_reward": float(total_reward),
        }

        return float(total_reward), breakdown


class RewardWrapper(gym.Wrapper):
    """Gymnasium environment wrapper for modular reward calculation.

    Wraps any ExploreEnv to compute or override exploration rewards and injects
    'reward_breakdown' into step info.
    """

    def __init__(
        self,
        env: gym.Env,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_engine = RewardEngine(config)
        self._coverage_bonus_awarded = False
        unwrapped = env.unwrapped
        env_config = getattr(unwrapped, "config", {})
        self.target_coverage: float = env_config.get("target_coverage", 0.95)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.reward_engine.reset()
        self._coverage_bonus_awarded = False
        return obs, info

    def step(
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped

        curr_revealed_cov = info.get(
            "revealed_coverage",
            getattr(getattr(unwrapped, "tracker", None), "revealed_coverage", 0.0),
        )
        coverage_done = False
        if not self._coverage_bonus_awarded and curr_revealed_cov >= self.target_coverage:
            coverage_done = True
            self._coverage_bonus_awarded = True

        reward, breakdown = self.reward_engine.compute_reward(
            agent_pos=getattr(unwrapped, "agent_pos", (0, 0)),
            agent_dir=getattr(unwrapped, "agent_dir", 0),
            delta_revealed=info.get("delta_revealed", 0),
            delta_pois=info.get("delta_pois", 0),
            bump=info.get("bump", False),
            terminated=terminated,
            end_reason=info.get("end_reason"),
            step_count=getattr(unwrapped, "step_count", 0),
            max_steps=getattr(unwrapped, "max_steps", 1),
            coverage_done=coverage_done,
        )
        info["reward_breakdown"] = breakdown
        return obs, reward, terminated, truncated, info

