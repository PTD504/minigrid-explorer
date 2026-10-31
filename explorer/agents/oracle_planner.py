from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym

from explorer.agents.base import BaseAgent

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2

CARDINAL_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DELTA_TO_DIR = {
    (1, 0): 0,
    (0, 1): 1,
    (-1, 0): 2,
    (0, -1): 3,
}


class OraclePlannerAgent(BaseAgent):
    """Privileged Oracle Planner establishing the theoretical upper bound of performance.

    Operates with full global ground-truth knowledge of the dungeon map and monster positions.
    Computes shortest-path traversals directly to unrevealed floor tiles and undiscovered POIs.
    Reported separately as an upper-bound baseline (§11 of design.md).
    """

    def __init__(self, env: gym.Env | None = None) -> None:
        self.env = env
        self.current_target: tuple[int, int] | None = None

    def set_env(self, env: gym.Env) -> None:
        """Associate environment instance with the oracle planner."""
        self.env = env

    def reset(self) -> None:
        self.current_target = None

    def _find_shortest_path_to_targets(
        self,
        start_pos: tuple[int, int],
        walkable_tiles: set[tuple[int, int]],
        targets: set[tuple[int, int]],
        blocked_tiles: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        """Compute shortest BFS path from start_pos to any target in targets."""
        if start_pos in targets:
            return [start_pos]

        queue: deque[tuple[int, int]] = deque([start_pos])
        visited: set[tuple[int, int]] = {start_pos}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start_pos: None}
        reached_target: tuple[int, int] | None = None

        while queue:
            curr = queue.popleft()
            if curr in targets:
                reached_target = curr
                break

            cx, cy = curr
            for dx, dy in CARDINAL_DIRS:
                nbr = (cx + dx, cy + dy)
                if (
                    nbr in walkable_tiles
                    and nbr not in visited
                    and nbr not in blocked_tiles
                ):
                    visited.add(nbr)
                    parent[nbr] = curr
                    queue.append(nbr)

        if reached_target is None:
            return None

        path: list[tuple[int, int]] = []
        curr_node: tuple[int, int] | None = reached_target
        while curr_node is not None:
            path.append(curr_node)
            curr_node = parent.get(curr_node)
        path.reverse()
        return path

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        """Compute optimal step with privileged knowledge."""
        if self.env is None:
            return ACTION_FORWARD

        unwrapped = self.env.unwrapped
        agent_pos = tuple(unwrapped.agent_pos)
        agent_dir = unwrapped.agent_dir

        walkable_tiles = set(unwrapped.walkable_tiles)
        revealed_tiles = set(unwrapped.tracker.revealed_tiles)
        unrevealed_walkable = walkable_tiles - revealed_tiles
        blocked_tiles = {m.pos for m in unwrapped.monsters}

        path = None
        # Stick to committed target if still unrevealed and reachable
        if self.current_target is not None and self.current_target in unrevealed_walkable:
            path = self._find_shortest_path_to_targets(
                start_pos=agent_pos,
                walkable_tiles=walkable_tiles,
                targets={self.current_target},
                blocked_tiles=blocked_tiles,
            )

        if path is None and unrevealed_walkable:
            path = self._find_shortest_path_to_targets(
                start_pos=agent_pos,
                walkable_tiles=walkable_tiles,
                targets=unrevealed_walkable,
                blocked_tiles=blocked_tiles,
            )
            if path is not None and len(path) > 0:
                self.current_target = path[-1]

        if path is None or len(path) < 2:
            self.current_target = None
            # Check remaining undiscovered POIs
            undiscovered_pois = set(unwrapped.poi_positions) - set(unwrapped.tracker.discovered_pois)
            if undiscovered_pois:
                path = self._find_shortest_path_to_targets(
                    start_pos=agent_pos,
                    walkable_tiles=walkable_tiles,
                    targets=undiscovered_pois,
                    blocked_tiles=blocked_tiles,
                )

        if not path or len(path) < 2:
            # All explored or blocked; scan remaining directions
            return ACTION_RIGHT

        next_pos = path[1]
        delta = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])
        target_dir = DELTA_TO_DIR[delta]

        if agent_dir == target_dir:
            return ACTION_FORWARD
        diff = (target_dir - agent_dir) % 4
        return ACTION_RIGHT if diff == 1 else ACTION_LEFT

