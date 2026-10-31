from __future__ import annotations

from typing import Any

from explorer.agents.base import BaseAgent

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2

# Egocentric coordinates in 7x7 view (center at x=3, agent at y=6, facing UP)
POS_LEFT = (2, 6)
POS_FRONT = (3, 5)
POS_FRONT_LEFT = (2, 5)
POS_RIGHT = (4, 6)


class WallFollowerAgent(BaseAgent):
    """Deterministic left-hand wall-following baseline.

    Maintains contact with the wall on the agent's left hand:
    - Seeks a wall by moving forward until an obstacle is encountered.
    - Follows outer and inner wall contours deterministically.
    - Avoids monsters visible in the immediate field of view.
    """

    def __init__(self) -> None:
        self.last_action: int | None = None
        self.has_contact_wall: bool = False

    def reset(self) -> None:
        """Reset internal wall contact state for a new episode."""
        self.last_action = None
        self.has_contact_wall = False

    def _is_obstacle(self, obs: dict[str, Any], x: int, y: int) -> bool:
        """Check if cell (x, y) in the 7x7 egocentric view contains an obstacle."""
        if "view" in obs:
            view = obs["view"]
            # Channel 2: wall, Channel 4: monster
            return bool(view[x, y, 2] > 0.5 or view[x, y, 4] > 0.5)
        elif "image" in obs:
            img = obs["image"]
            # MiniGrid OBJECT_TO_IDX["wall"] == 2
            return bool(img[x, y, 0] == 2)
        return False

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        """Select action adhering to the left-hand rule."""
        left_blocked = self._is_obstacle(obs, *POS_LEFT)
        front_blocked = self._is_obstacle(obs, *POS_FRONT)
        front_left_blocked = self._is_obstacle(obs, *POS_FRONT_LEFT)

        # 1. Wall-seeking mode if not yet adjacent to any wall
        if not self.has_contact_wall:
            if left_blocked:
                self.has_contact_wall = True
            elif front_blocked:
                self.has_contact_wall = True
                self.last_action = ACTION_RIGHT
                return ACTION_RIGHT
            else:
                self.last_action = ACTION_FORWARD
                return ACTION_FORWARD

        # 2. Wall-following mode (hugging left wall)
        if left_blocked:
            if front_blocked:
                action = ACTION_RIGHT
            else:
                action = ACTION_FORWARD
        else:
            # Left side is open
            if self.last_action == ACTION_LEFT:
                # Just rounded a left corner; step forward into the opening
                action = ACTION_RIGHT if front_blocked else ACTION_FORWARD
            else:
                if front_left_blocked and not front_blocked:
                    action = ACTION_FORWARD
                else:
                    action = ACTION_LEFT

        self.last_action = action
        return action
