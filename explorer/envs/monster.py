from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from minigrid.core.grid import Grid

from explorer.envs.objects import Monster, PointOfInterest
from explorer.envs.visibility import has_line_of_sight

if TYPE_CHECKING:
    from explorer.envs.map_generator import Room

# Fixed direction priority for deterministic BFS tie-breaking: Up, Right, Down, Left
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def compute_bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable_tiles: set[tuple[int, int]],
    blocked_tiles: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Compute the shortest path using Breadth-First Search (BFS).

    Guarantees absolute determinism through fixed direction tie-breaking.
    Blocked tiles (e.g. other monsters) are respected, but the goal tile itself
    is allowed so monsters can target the agent or waypoint directly.
    """
    if start == goal:
        return [start]

    queue: deque[tuple[int, int]] = deque([start])
    visited: set[tuple[int, int]] = {start}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    reached = False
    while queue:
        curr = queue.popleft()
        if curr == goal:
            reached = True
            break

        cx, cy = curr
        for dx, dy in DIRECTIONS:
            nbr = (cx + dx, cy + dy)
            if nbr not in walkable_tiles:
                continue
            if blocked_tiles and nbr in blocked_tiles:
                continue
            if nbr not in visited:
                visited.add(nbr)
                came_from[nbr] = curr
                queue.append(nbr)

    if not reached and goal not in came_from:
        return None

    # Reconstruct path from goal to start
    path: list[tuple[int, int]] = []
    curr_node: tuple[int, int] | None = goal
    while curr_node is not None:
        path.append(curr_node)
        curr_node = came_from.get(curr_node)
    path.reverse()
    return path


def compute_bfs_distance_map(
    start: tuple[int, int],
    walkable_tiles: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Compute BFS distance from start to all reachable walkable tiles."""
    dist_map: dict[tuple[int, int], int] = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        curr = queue.popleft()
        cdist = dist_map[curr]
        cx, cy = curr
        for dx, dy in DIRECTIONS:
            nbr = (cx + dx, cy + dy)
            if nbr in walkable_tiles and nbr not in dist_map:
                dist_map[nbr] = cdist + 1
                queue.append(nbr)

    return dist_map


@dataclass
class MonsterEntity:
    """State and parameters for a single monster instance."""

    id: int
    pos: tuple[int, int]
    period: int
    detect_radius: int
    waypoints: list[tuple[int, int]]
    current_waypoint_idx: int = 0
    state: str = "patrol"  # 'patrol' or 'chase'
    world_obj: Monster = field(default_factory=Monster)


class MonsterManager:
    """Manages spawning, deterministic pathfinding, patrol, and chase behaviors

    for all dynamic threats in the environment.
    """

    def __init__(
        self,
        config: dict[str, Any],
        seed: int,
        rooms: list[Room],
        walkable_tiles: set[tuple[int, int]],
        agent_start_pos: tuple[int, int],
        grid: Grid,
        poi_positions: set[tuple[int, int]] | None = None,
    ):
        self.config = config
        self.num_monsters = config.get("num_monsters", 1)
        self.period = config.get("monster_period", 3)
        self.detect_radius = config.get("monster_detect_radius", 5)
        self.min_dist = config.get("monster_min_dist", 5)

        self.walkable_tiles = walkable_tiles
        self.poi_positions = poi_positions or set()

        # Dedicated independent RNG stream for monster initialization
        self.rng = np.random.Generator(np.random.PCG64(seed + 9999))

        self.monsters: list[MonsterEntity] = []
        self._spawn_monsters(rooms, agent_start_pos, grid)

    def _spawn_monsters(
        self,
        rooms: list[Room],
        agent_start_pos: tuple[int, int],
        grid: Grid,
    ) -> None:
        """Spawn monsters respecting minimum distance from agent and deterministic waypoints."""
        if self.num_monsters <= 0:
            return

        # Compute BFS distances from agent start position
        dist_map = compute_bfs_distance_map(agent_start_pos, self.walkable_tiles)

        # Filter candidates meeting minimum distance requirement
        candidates = [
            pos for pos, dist in dist_map.items() if dist >= self.min_dist
        ]

        # Fallback if no tile is far enough (e.g. tiny maps)
        if not candidates:
            max_d = max(dist_map.values()) if dist_map else 0
            candidates = [pos for pos, dist in dist_map.items() if dist == max_d]

        # Sort candidates deterministically to guarantee reproducible selection
        candidates.sort()

        # Base waypoints from room centers (or walkable tiles if fewer rooms)
        room_centers = [r.center for r in rooms] if rooms else [candidates[0]]

        occupied_spawns: set[tuple[int, int]] = set()

        for idx in range(self.num_monsters):
            # Select spawn tile not already occupied by another monster
            available = [c for c in candidates if c not in occupied_spawns]
            if not available:
                available = [c for c in candidates]

            spawn_idx = int(self.rng.integers(0, len(available)))
            spawn_pos = available[spawn_idx]
            occupied_spawns.add(spawn_pos)

            # Build patrol waypoints cycle for this monster
            # Offset waypoint sequence per monster to avoid synchronized crowding
            shifted_waypoints = list(room_centers)
            if len(shifted_waypoints) > 1:
                shift = (idx * (len(shifted_waypoints) // max(1, self.num_monsters))) % len(shifted_waypoints)
                shifted_waypoints = shifted_waypoints[shift:] + shifted_waypoints[:shift]

            # Ensure current spawn position is not identical to first waypoint if possible
            if len(shifted_waypoints) > 1 and shifted_waypoints[0] == spawn_pos:
                shifted_waypoints = shifted_waypoints[1:] + [shifted_waypoints[0]]

            monster = MonsterEntity(
                id=idx,
                pos=spawn_pos,
                period=self.period,
                detect_radius=self.detect_radius,
                waypoints=shifted_waypoints,
                current_waypoint_idx=0,
                state="patrol",
                world_obj=Monster(),
            )
            self.monsters.append(monster)

            # Place monster object on grid
            grid.set(spawn_pos[0], spawn_pos[1], monster.world_obj)

    def step(
        self,
        grid: Grid,
        agent_pos: tuple[int, int],
        step_count: int,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Update monster positions if current step matches monster period.

        Returns:
            List of (old_pos, new_pos) for each monster.
        """
        movements: list[tuple[tuple[int, int], tuple[int, int]]] = []

        for monster in self.monsters:
            old_pos = monster.pos

            # Monsters only move once every `monster.period` environment steps
            if step_count % monster.period != 0:
                movements.append((old_pos, old_pos))
                continue

            # Other monsters act as dynamic obstacles
            other_monsters = {m.pos for m in self.monsters if m.id != monster.id}

            # 1. Check for Line of Sight and distance to agent
            chase_target = None
            if has_line_of_sight(grid, monster.pos, agent_pos):
                path_to_agent = compute_bfs_path(
                    monster.pos,
                    agent_pos,
                    self.walkable_tiles,
                    blocked_tiles=other_monsters,
                )
                if path_to_agent is not None and (len(path_to_agent) - 1) <= monster.detect_radius:
                    chase_target = agent_pos
                    monster.state = "chase"

            if chase_target is not None and path_to_agent is not None and len(path_to_agent) > 1:
                next_pos = path_to_agent[1]
            else:
                # 2. Patrol mode
                monster.state = "patrol"
                if monster.pos == monster.waypoints[monster.current_waypoint_idx]:
                    monster.current_waypoint_idx = (
                        monster.current_waypoint_idx + 1
                    ) % len(monster.waypoints)

                target_waypoint = monster.waypoints[monster.current_waypoint_idx]
                path_to_waypoint = compute_bfs_path(
                    monster.pos,
                    target_waypoint,
                    self.walkable_tiles,
                    blocked_tiles=other_monsters,
                )

                if path_to_waypoint is not None and len(path_to_waypoint) > 1:
                    next_pos = path_to_waypoint[1]
                else:
                    next_pos = old_pos

            # Invariant: Never step onto a tile currently occupied by another monster
            if next_pos in other_monsters:
                next_pos = old_pos

            # Update monster position on the grid
            if next_pos != old_pos:
                # Clear old position (restore POI if one was present)
                if old_pos in self.poi_positions:
                    grid.set(old_pos[0], old_pos[1], PointOfInterest())
                else:
                    grid.set(old_pos[0], old_pos[1], None)

                # Set new position
                grid.set(next_pos[0], next_pos[1], monster.world_obj)
                monster.pos = next_pos

            movements.append((old_pos, next_pos))

        return movements

