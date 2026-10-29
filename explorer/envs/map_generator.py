from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall

from explorer.envs.objects import PointOfInterest


@dataclass
class Room:
    """Represents a rectangular room inside the grid."""
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def overlaps(self, other: Room, padding: int = 1) -> bool:
        """Check if this room overlaps with another room including padding."""
        return not (
            self.x + self.w + padding <= other.x
            or other.x + other.w + padding <= self.x
            or self.y + self.h + padding <= other.y
            or other.y + other.h + padding <= self.y
        )


@dataclass
class MapMetadata:
    """Contains structural metadata for the generated map."""
    rooms: list[Room]
    walkable_tiles: set[tuple[int, int]]
    dead_ends: list[tuple[int, int]]
    junctions: list[tuple[int, int]]
    poi_positions: list[tuple[int, int]]
    agent_start_pos: tuple[int, int]
    agent_start_dir: int
    walkable_ratio: float


class ProceduralMapGenerator:
    """Procedurally generates dungeon-style grid maps with rooms, corridors,

    junctions, dead ends, and points of interest (POIs).
    
    Guarantees:
    1. 100% reachability across all walkable tiles (verified via BFS).
    2. Deterministic reproducibility given a fixed NumPy RNG stream.
    3. Meaningful navigational structures (rooms, corridors, loops, dead ends).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def generate(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
        max_attempts: int = 50,
    ) -> tuple[Grid, MapMetadata]:
        """Generate a procedural grid with guaranteed connectivity and structures."""
        for attempt in range(max_attempts):
            result = self._try_generate_layout(width, height, rng)
            if result is not None:
                grid, metadata = result
                return grid, metadata

        raise RuntimeError(
            f"Failed to generate valid map after {max_attempts} attempts. "
            "Please check environment parameters or room constraints."
        )

    def _try_generate_layout(
        self, width: int, height: int, rng: np.random.Generator
    ) -> tuple[Grid, MapMetadata] | None:
        # Extract configuration with sensible defaults
        num_rooms_min = self.config.get("num_rooms_min", 3)
        num_rooms_max = self.config.get("num_rooms_max", 4)
        room_min_size = self.config.get("room_min_size", 4)
        room_max_size = self.config.get("room_max_size", 6)
        num_pois = self.config.get("num_pois", 2)
        poi_min_dist = self.config.get("poi_min_dist", 4)
        num_dead_ends_target = self.config.get("num_dead_ends", 2)
        num_extra_corridors = self.config.get("num_extra_corridors", 1)

        # 1. Initialize grid filled completely with walls
        grid = Grid(width, height)
        for x in range(width):
            for y in range(height):
                grid.set(x, y, Wall())

        # 2. Place non-overlapping rooms via rejection sampling
        target_num_rooms = rng.integers(num_rooms_min, num_rooms_max + 1)
        rooms: list[Room] = []
        max_room_attempts = 150

        for _ in range(max_room_attempts):
            if len(rooms) >= target_num_rooms:
                break
            rw = rng.integers(room_min_size, room_max_size + 1)
            rh = rng.integers(room_min_size, room_max_size + 1)
            # Ensure within grid boundaries with 1-tile outer margin
            if width - 2 - rw <= 1 or height - 2 - rh <= 1:
                continue
            rx = rng.integers(1, width - 1 - rw)
            ry = rng.integers(1, height - 1 - rh)
            candidate = Room(rx, ry, rw, rh)

            if not any(candidate.overlaps(r, padding=1) for r in rooms):
                rooms.append(candidate)

        if len(rooms) < num_rooms_min:
            return None

        # Carve out rooms
        walkable: set[tuple[int, int]] = set()
        for room in rooms:
            for x in range(room.x, room.x + room.w):
                for y in range(room.y, room.y + room.h):
                    grid.set(x, y, None)
                    walkable.add((x, y))

        # 3. Connect rooms using Minimum Spanning Tree (MST)
        mst_edges = self._compute_mst(rooms)
        for r1_idx, r2_idx in mst_edges:
            self._carve_corridor(grid, rooms[r1_idx].center, rooms[r2_idx].center, walkable, rng)

        # 4. Add extra corridors to introduce cycles and junctions
        if len(rooms) > 2 and num_extra_corridors > 0:
            existing_pairs = set(mst_edges)
            candidates = [
                (i, j)
                for i in range(len(rooms))
                for j in range(i + 1, len(rooms))
                if (i, j) not in existing_pairs
            ]
            rng.shuffle(candidates)
            for i, j in candidates[:num_extra_corridors]:
                self._carve_corridor(grid, rooms[i].center, rooms[j].center, walkable, rng)

        # 5. Carve dead-end branches
        self._carve_dead_ends(grid, width, height, walkable, num_dead_ends_target, rng)

        # 6. Analyze topology (dead ends, junctions)
        dead_ends: list[tuple[int, int]] = []
        junctions: list[tuple[int, int]] = []
        for x, y in walkable:
            nbrs = self._count_walkable_neighbors(x, y, walkable)
            if nbrs == 1:
                dead_ends.append((x, y))
            elif nbrs >= 3:
                junctions.append((x, y))

        # 7. Select agent starting position in room 0
        agent_room = rooms[0]
        start_x = rng.integers(agent_room.x, agent_room.x + agent_room.w)
        start_y = rng.integers(agent_room.y, agent_room.y + agent_room.h)
        agent_start_pos = (int(start_x), int(start_y))
        agent_start_dir = int(rng.integers(0, 4))

        # 8. Verify full BFS connectivity
        reachable = self._bfs_reachable(agent_start_pos, walkable)
        if reachable != walkable:
            # Not all walkable tiles are reachable
            return None

        # 9. Place POIs (prioritize dead-end tips and rooms, respecting min distance)
        poi_positions = self._place_pois(
            grid, walkable, dead_ends, agent_start_pos, poi_min_dist, num_pois, rng
        )
        if len(poi_positions) < num_pois:
            return None

        total_tiles = width * height
        walkable_ratio = len(walkable) / float(total_tiles)

        metadata = MapMetadata(
            rooms=rooms,
            walkable_tiles=walkable,
            dead_ends=dead_ends,
            junctions=junctions,
            poi_positions=poi_positions,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            walkable_ratio=walkable_ratio,
        )

        return grid, metadata

    def _compute_mst(self, rooms: list[Room]) -> list[tuple[int, int]]:
        """Compute Minimum Spanning Tree over room centers using Prim's algorithm."""
        n = len(rooms)
        in_mst = [False] * n
        in_mst[0] = True
        edges: list[tuple[int, int]] = []

        for _ in range(n - 1):
            min_dist = float("inf")
            best_pair = (-1, -1)
            for i in range(n):
                if not in_mst[i]:
                    continue
                c1 = rooms[i].center
                for j in range(n):
                    if in_mst[j]:
                        continue
                    c2 = rooms[j].center
                    dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (min(i, j), max(i, j))
            if best_pair[0] != -1:
                in_mst[best_pair[1] if in_mst[best_pair[0]] else best_pair[0]] = True
                edges.append(best_pair)

        return edges

    def _carve_corridor(
        self,
        grid: Grid,
        start: tuple[int, int],
        end: tuple[int, int],
        walkable: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> None:
        """Carve an L-shaped corridor connecting two points."""
        x1, y1 = start
        x2, y2 = end

        # 50% chance horizontal then vertical, or vertical then horizontal
        if rng.random() < 0.5:
            # Horizontal then vertical
            for x in range(min(x1, x2), max(x1, x2) + 1):
                grid.set(x, y1, None)
                walkable.add((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                grid.set(x2, y, None)
                walkable.add((x2, y))
        else:
            # Vertical then horizontal
            for y in range(min(y1, y2), max(y1, y2) + 1):
                grid.set(x1, y, None)
                walkable.add((x1, y))
            for x in range(min(x1, x2), max(x1, x2) + 1):
                grid.set(x, y2, None)
                walkable.add((x, y2))

    def _carve_dead_ends(
        self,
        grid: Grid,
        width: int,
        height: int,
        walkable: set[tuple[int, int]],
        count: int,
        rng: np.random.Generator,
    ) -> None:
        """Carve dead-end branches extending from walkable spaces into walls."""
        attempts = 0
        created = 0
        max_attempts = 100
        walkable_list = list(walkable)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while created < count and attempts < max_attempts:
            attempts += 1
            idx = rng.integers(0, len(walkable_list))
            sx, sy = walkable_list[idx]

            dir_idx = rng.integers(0, len(directions))
            dx, dy = directions[dir_idx]

            branch_len = rng.integers(2, 4)
            current_x, current_y = sx, sy
            branch_cells: list[tuple[int, int]] = []
            can_carve = True

            for _ in range(branch_len):
                nx, ny = current_x + dx, current_y + dy
                # Stay inside perimeter walls
                if nx <= 1 or nx >= width - 2 or ny <= 1 or ny >= height - 2:
                    can_carve = False
                    break
                # Only carve if neighboring cells (except predecessor) are walls
                neighbor_count = 0
                for cdx, cdy in directions:
                    adj = (nx + cdx, ny + cdy)
                    if adj in walkable and adj != (current_x, current_y):
                        neighbor_count += 1
                if neighbor_count > 0:
                    can_carve = False
                    break

                branch_cells.append((nx, ny))
                current_x, current_y = nx, ny

            if can_carve and branch_cells:
                for bx, by in branch_cells:
                    grid.set(bx, by, None)
                    walkable.add((bx, by))
                    walkable_list.append((bx, by))
                created += 1

    def _count_walkable_neighbors(self, x: int, y: int, walkable: set[tuple[int, int]]) -> int:
        return sum(
            1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if (x + dx, y + dy) in walkable
        )

    def _bfs_reachable(
        self, start: tuple[int, int], walkable: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Compute the set of all tiles reachable from start using BFS."""
        if start not in walkable:
            return set()
        visited: set[tuple[int, int]] = {start}
        queue = [start]
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nbr = (cx + dx, cy + dy)
                if nbr in walkable and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return visited

    def _place_pois(
        self,
        grid: Grid,
        walkable: set[tuple[int, int]],
        dead_ends: list[tuple[int, int]],
        agent_start: tuple[int, int],
        poi_min_dist: int,
        num_pois: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Place POI objects prioritizing dead ends and distant walkable tiles."""
        def manhattan(p1: tuple[int, int], p2: tuple[int, int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Prioritize dead ends that are far enough from the agent
        valid_dead_ends = [
            p for p in dead_ends if p != agent_start and manhattan(p, agent_start) >= poi_min_dist
        ]
        rng.shuffle(valid_dead_ends)

        # Other walkable candidates
        other_candidates = [
            p
            for p in walkable
            if p != agent_start
            and manhattan(p, agent_start) >= poi_min_dist
            and p not in valid_dead_ends
        ]
        rng.shuffle(other_candidates)

        chosen: list[tuple[int, int]] = []
        for p in valid_dead_ends:
            if len(chosen) >= num_pois:
                break
            chosen.append(p)

        for p in other_candidates:
            if len(chosen) >= num_pois:
                break
            chosen.append(p)

        # Place POI objects into the grid
        for px, py in chosen:
            grid.set(px, py, PointOfInterest(color="purple"))

        return chosen
