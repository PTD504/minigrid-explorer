from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym

from explorer.agents.base import BaseAgent

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2

# Cardinal directions: (dx, dy) and corresponding MiniGrid dir index
# 0: Right (1, 0), 1: Down (0, 1), 2: Left (-1, 0), 3: Up (0, -1)
CARDINAL_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DELTA_TO_DIR = {
    (1, 0): 0,
    (0, 1): 1,
    (-1, 0): 2,
    (0, -1): 3,
}


class FrontierExplorerAgent(BaseAgent):
    """Robotics-standard Frontier Exploration baseline.

    Finds the nearest frontier tile (known walkable tile adjacent to unrevealed space)
    via Breadth-First Search (BFS) over the accumulated explored map.
    Respects strict fairness constraints: only navigates through tiles revealed
    by the agent's line-of-sight. Avoids dynamic threats (monsters).
    """

    def __init__(self, env: gym.Env | None = None) -> None:
        self.env = env
        self.planned_path: list[tuple[int, int]] = []

    def set_env(self, env: gym.Env) -> None:
        """Associate environment instance for reading accumulated explored memory."""
        self.env = env

    def reset(self) -> None:
        """Reset internal path cache for a new episode."""
        self.planned_path.clear()

    def _find_closest_frontier(
        self,
        start_pos: tuple[int, int],
        known_walkable: set[tuple[int, int]],
        revealed_tiles: set[tuple[int, int]],
        blocked_tiles: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        """Find the shortest BFS path from start_pos to the closest frontier tile."""
        def is_frontier(pos: tuple[int, int]) -> bool:
            for dx, dy in CARDINAL_DIRS:
                nbr = (pos[0] + dx, pos[1] + dy)
                if nbr not in revealed_tiles:
                    return True
            return False

        # If current position is a frontier, check if we need to turn to face unseen neighbors
        queue: deque[tuple[int, int]] = deque([start_pos])
        visited: set[tuple[int, int]] = {start_pos}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start_pos: None}
        target_pos: tuple[int, int] | None = None

        while queue:
            curr = queue.popleft()
            if curr != start_pos and is_frontier(curr):
                target_pos = curr
                break

            cx, cy = curr
            for dx, dy in CARDINAL_DIRS:
                nbr = (cx + dx, cy + dy)
                if (
                    nbr in known_walkable
                    and nbr not in visited
                    and nbr not in blocked_tiles
                ):
                    visited.add(nbr)
                    parent[nbr] = curr
                    queue.append(nbr)

        if target_pos is None:
            # Check if start_pos itself is a frontier
            if is_frontier(start_pos):
                return [start_pos]
            return None

        # Reconstruct path from start_pos to target_pos
        path: list[tuple[int, int]] = []
        curr_node: tuple[int, int] | None = target_pos
        while curr_node is not None:
            path.append(curr_node)
            curr_node = parent.get(curr_node)
        path.reverse()
        return path

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        """Select next navigation action towards the closest frontier."""
        if self.env is None:
            # Fallback if env is not linked: rotate safely
            return ACTION_FORWARD

        unwrapped = self.env.unwrapped
        agent_pos = tuple(unwrapped.agent_pos)
        agent_dir = unwrapped.agent_dir

        revealed_tiles = set(unwrapped.tracker.revealed_tiles)
        known_walkable = revealed_tiles.intersection(unwrapped.walkable_tiles)

        # Dynamic obstacles (Fair baseline POMDP rule):
        # The agent only knows and blocks monster positions that are currently visible
        visible_monsters: set[tuple[int, int]] = set()
        if info is not None and "monsters" in info:
            visible_monsters = {tuple(m["pos"]) for m in info["monsters"]}
        elif hasattr(unwrapped, "tracker") and hasattr(unwrapped.tracker, "current_visible_tiles"):
            visible_monsters = {
                m.pos
                for m in unwrapped.monsters
                if m.pos in unwrapped.tracker.current_visible_tiles
            }
        blocked_tiles = visible_monsters

        path = self._find_closest_frontier(
            start_pos=agent_pos,
            known_walkable=known_walkable,
            revealed_tiles=revealed_tiles,
            blocked_tiles=blocked_tiles,
        )

        if not path or len(path) < 2:
            # If no path to a distant frontier, check if standing on a frontier facing walls
            # Turn to scan remaining unseen neighbors
            for dx, dy in CARDINAL_DIRS:
                nbr = (agent_pos[0] + dx, agent_pos[1] + dy)
                if nbr not in revealed_tiles:
                    target_dir = DELTA_TO_DIR[(dx, dy)]
                    if agent_dir != target_dir:
                        diff = (target_dir - agent_dir) % 4
                        return ACTION_RIGHT if diff == 1 else ACTION_LEFT

            # If completely explored or surrounded, take safe turn
            return ACTION_RIGHT

        # Target next waypoint along BFS path
        next_pos = path[1]
        delta = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])
        target_dir = DELTA_TO_DIR[delta]

        if agent_dir == target_dir:
            return ACTION_FORWARD
        diff = (target_dir - agent_dir) % 4
        return ACTION_RIGHT if diff == 1 else ACTION_LEFT

