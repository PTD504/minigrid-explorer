from __future__ import annotations

from typing import Any

import numpy as np


def _build_rotation_tables(sz: int = 7) -> dict[int, np.ndarray]:
    """Precompute coordinate transformations for MiniGrid view subgrid rotation.

    In MiniGrid, subgrid of shape (sz, sz) is sliced at (top_x, top_y) and then rotated left
    (agent_dir + 1) times counter-clockwise.
    Given coordinate (vx, vy) in the rotated subgrid, this table maps directly to
    the unrotated local slice offset (rx, ry) such that:
    world_pos = (top_x + rx, top_y + ry).
    """
    coords = np.zeros((sz, sz, 2), dtype=int)
    for i in range(sz):
        for j in range(sz):
            coords[i, j] = [i, j]

    tables: dict[int, np.ndarray] = {}
    for agent_dir in range(4):
        c = coords.copy()
        for _ in range(agent_dir + 1):
            w, h, _ = c.shape
            out = np.zeros((h, w, 2), dtype=c.dtype)
            for i in range(w):
                for j in range(h):
                    out[j, h - 1 - i] = c[i, j]
            c = out
        tables[agent_dir] = c
    return tables


# Cached rotation tables for standard 7x7 MiniGrid view
_ROTATION_TABLES_7X7 = _build_rotation_tables(7)


class ExplorationTracker:
    """Tracks agent exploration progress, revealed/visited areas, and POI discoveries.

    Maintains strict separation between:
    1. Agent-facing telemetry (revealed tiles, delta newly seen tiles, discovered POIs).
    2. Evaluator-only benchmark metrics (revealed coverage, visited coverage normalized by ground-truth W).
    """

    def __init__(
        self,
        walkable_tiles: set[tuple[int, int]],
        poi_positions: set[tuple[int, int]],
    ):
        self.walkable_tiles = set(walkable_tiles)
        self.total_walkable = len(self.walkable_tiles)
        self.poi_positions = set(poi_positions)
        self.total_pois = len(self.poi_positions)

        # Internal state
        self.revealed_tiles: set[tuple[int, int]] = set()
        self.visited_tiles: set[tuple[int, int]] = set()
        self.discovered_pois: set[tuple[int, int]] = set()

    def reset(
        self,
        walkable_tiles: set[tuple[int, int]],
        poi_positions: set[tuple[int, int]],
    ) -> None:
        """Reset tracker state for a new episode."""
        self.walkable_tiles = set(walkable_tiles)
        self.total_walkable = len(self.walkable_tiles)
        self.poi_positions = set(poi_positions)
        self.total_pois = len(self.poi_positions)

        self.revealed_tiles.clear()
        self.visited_tiles.clear()
        self.discovered_pois.clear()

    def update(
        self,
        agent_pos: tuple[int, int],
        agent_dir: int,
        top_x: int,
        top_y: int,
        vis_mask: np.ndarray,
        agent_view_size: int = 7,
    ) -> tuple[int, int]:
        """Update exploration status from the current view observation.

        Args:
            agent_pos: Current agent world coordinates (x, y).
            agent_dir: Current agent orientation (0: right, 1: down, 2: left, 3: up).
            top_x: Top-left X coordinate of the agent's view slice in world coordinates.
            top_y: Top-left Y coordinate of the agent's view slice in world coordinates.
            vis_mask: 2D boolean array (sz x sz) indicating visible cells in the rotated subgrid.
            agent_view_size: Size of the local observation window (default 7).

        Returns:
            delta_revealed: Number of *new* walkable tiles revealed on this step (ΔN_t).
            delta_pois: Number of *new* POIs discovered on this step (ΔP_t).
        """
        # 1. Update visited tiles (agent stands on this tile)
        self.visited_tiles.add(agent_pos)

        # 2. Map visible cells from the egocentric subgrid to world coordinates
        if agent_view_size == 7:
            rot_table = _ROTATION_TABLES_7X7[agent_dir]
        else:
            rot_table = _build_rotation_tables(agent_view_size)[agent_dir]

        w, h = vis_mask.shape
        step_revealed_walkable = 0
        step_discovered_pois = 0

        for vx in range(w):
            for vy in range(h):
                if not vis_mask[vx, vy]:
                    continue

                rx, ry = rot_table[vx, vy]
                world_x = top_x + int(rx)
                world_y = top_y + int(ry)
                world_pos = (world_x, world_y)

                # Track newly revealed tiles
                if world_pos not in self.revealed_tiles:
                    self.revealed_tiles.add(world_pos)
                    if world_pos in self.walkable_tiles:
                        step_revealed_walkable += 1

                # Track newly discovered POIs (revealed in agent's line-of-sight)
                if world_pos in self.poi_positions and world_pos not in self.discovered_pois:
                    self.discovered_pois.add(world_pos)
                    step_discovered_pois += 1

        return step_revealed_walkable, step_discovered_pois

    @property
    def revealed_coverage(self) -> float:
        """Ground-truth revealed area coverage: |R_t ∩ W| / |W|."""
        if self.total_walkable == 0:
            return 0.0
        revealed_walkable = len(self.revealed_tiles.intersection(self.walkable_tiles))
        return float(revealed_walkable) / float(self.total_walkable)

    @property
    def visited_coverage(self) -> float:
        """Ground-truth visited area coverage: |V_t ∩ W| / |W|."""
        if self.total_walkable == 0:
            return 0.0
        visited_walkable = len(self.visited_tiles.intersection(self.walkable_tiles))
        return float(visited_walkable) / float(self.total_walkable)

    @property
    def poi_discovery_rate(self) -> float:
        """Proportion of POIs discovered: |discovered_pois| / |total_pois|."""
        if self.total_pois == 0:
            return 1.0
        return float(len(self.discovered_pois)) / float(self.total_pois)

    def get_eval_metrics(self) -> dict[str, Any]:
        """Compile comprehensive evaluator telemetry."""
        return {
            "revealed_coverage": self.revealed_coverage,
            "visited_coverage": self.visited_coverage,
            "poi_discovery_rate": self.poi_discovery_rate,
            "revealed_count": len(self.revealed_tiles.intersection(self.walkable_tiles)),
            "visited_count": len(self.visited_tiles.intersection(self.walkable_tiles)),
            "discovered_pois": len(self.discovered_pois),
            "total_walkable": self.total_walkable,
            "total_pois": self.total_pois,
        }
