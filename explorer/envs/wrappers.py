from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX

from explorer.envs.tracking import _ROTATION_TABLES_7X7, _build_rotation_tables
from explorer.utils.config import get_obs_config

# View channels: one-hot categorical encoding
VIEW_UNSEEN = 0
VIEW_EMPTY = 1
VIEW_WALL = 2
VIEW_POI = 3
VIEW_MONSTER = 4
VIEW_NUM_CHANNELS = 5

# Memory channels: accumulated binary knowledge
MEM_REVEALED = 0
MEM_VISITED = 1
MEM_KNOWN_WALL = 2
MEM_KNOWN_POI = 3
MEM_NUM_CHANNELS = 4

# Direction unit vectors: 0=Right, 1=Down, 2=Left, 3=Up
DIR_TO_VEC = [
    np.array((1, 0)),
    np.array((0, 1)),
    np.array((-1, 0)),
    np.array((0, -1)),
]


class ObservationMemoryWrapper(gym.ObservationWrapper):
    """Observation wrapper for ExploreEnv enforcing strict partial observability.

    Transforms raw MiniGrid observations into an egocentric Dict representation:
    - 'view': (7, 7, 5) one-hot matrix of visible tiles in the agent's line-of-sight.
    - 'memory': (K, K, 4) egocentric crop of accumulated spatial memory (agent at center).
    - 'step_ratio': (1,) normalized step counter (t / max_steps).

    Guarantees:
    - Strips 'mission' and absolute world coordinates completely.
    - Out-of-bounds padded regions are indistinguishable from unexplored interior tiles.
    - Entities outside line-of-sight (hidden POIs, roaming monsters) never leak into observation.
    """

    def __init__(
        self,
        env: gym.Env,
        crop_size: int | None = None,
        use_memory: bool | None = None,
        include_step_ratio: bool | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(env)

        # Load defaults from config if not explicitly passed
        obs_cfg = config or get_obs_config()

        self.crop_size: int = crop_size if crop_size is not None else obs_cfg.get("crop_size", 15)
        self.use_memory: bool = (
            use_memory if use_memory is not None else obs_cfg.get("use_memory", True)
        )
        self.include_step_ratio: bool = (
            include_step_ratio
            if include_step_ratio is not None
            else obs_cfg.get("include_step_ratio", True)
        )

        if self.crop_size % 2 == 0:
            raise ValueError(f"crop_size must be an odd integer, got {self.crop_size}")

        self.view_size: int = getattr(self.env.unwrapped, "agent_view_size", 7)
        self.center: int = self.crop_size // 2

        # Precompute meshgrid offsets for egocentric memory crop
        u_grid, v_grid = np.meshgrid(
            np.arange(self.crop_size), np.arange(self.crop_size), indexing="ij"
        )
        self.dx_ego = u_grid - self.center  # Lateral offset: positive is agent's right
        self.dy_ego = self.center - v_grid  # Longitudinal offset: positive is agent's forward

        # Initialize internal accumulated memory maps
        self.width: int = getattr(self.env.unwrapped, "width", 0)
        self.height: int = getattr(self.env.unwrapped, "height", 0)
        self._reset_memory()

        # Define Observation Space
        space_dict: dict[str, spaces.Box] = {
            "view": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.view_size, self.view_size, VIEW_NUM_CHANNELS),
                dtype=np.float32,
            ),
        }
        if self.use_memory:
            space_dict["memory"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.crop_size, self.crop_size, MEM_NUM_CHANNELS),
                dtype=np.float32,
            )
        if self.include_step_ratio:
            space_dict["step_ratio"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(space_dict)

    def _reset_memory(self) -> None:
        """Reset internal 2D world memory grids for a new episode."""
        unwrapped = self.env.unwrapped
        self.width = getattr(unwrapped, "width", 0)
        self.height = getattr(unwrapped, "height", 0)
        if self.width > 0 and self.height > 0:
            self.world_revealed = np.zeros((self.width, self.height), dtype=bool)
            self.world_visited = np.zeros((self.width, self.height), dtype=bool)
            self.world_walls = np.zeros((self.width, self.height), dtype=bool)
            self.world_pois = np.zeros((self.width, self.height), dtype=bool)
        else:
            self.world_revealed = np.zeros((0, 0), dtype=bool)
            self.world_visited = np.zeros((0, 0), dtype=bool)
            self.world_walls = np.zeros((0, 0), dtype=bool)
            self.world_pois = np.zeros((0, 0), dtype=bool)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment and initialize memory with step-0 field of view."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_memory()
        return self.observation(obs), info

    def _update_memory(self) -> None:
        """Update accumulated 2D memory from current step's raycasted line-of-sight."""
        unwrapped = self.env.unwrapped
        agent_pos = tuple(unwrapped.agent_pos)
        agent_dir = unwrapped.agent_dir

        # Record visited position
        if 0 <= agent_pos[0] < self.width and 0 <= agent_pos[1] < self.height:
            self.world_visited[agent_pos[0], agent_pos[1]] = True

        sub_grid, vis_mask = unwrapped.gen_obs_grid()
        top_x, top_y, _, _ = unwrapped.get_view_exts()

        view_size = unwrapped.agent_view_size
        if view_size == 7:
            rot_table = _ROTATION_TABLES_7X7[agent_dir]
        else:
            rot_table = _build_rotation_tables(view_size)[agent_dir]

        w, h = vis_mask.shape
        for vx in range(w):
            for vy in range(h):
                if not vis_mask[vx, vy]:
                    continue
                rx, ry = rot_table[vx, vy]
                wx = top_x + int(rx)
                wy = top_y + int(ry)

                if 0 <= wx < self.width and 0 <= wy < self.height:
                    self.world_revealed[wx, wy] = True
                    obj = sub_grid.get(vx, vy)
                    if obj is not None:
                        if obj.type == "wall":
                            self.world_walls[wx, wy] = True
                        elif obj.type == "poi":
                            self.world_pois[wx, wy] = True

        # Sync with ground-truth revealed tracker if available
        if hasattr(unwrapped, "tracker"):
            tracker = unwrapped.tracker
            for px, py in tracker.discovered_pois:
                if 0 <= px < self.width and 0 <= py < self.height:
                    self.world_pois[px, py] = True

    def _get_memory_crop(self) -> np.ndarray:
        """Extract egocentric KxK crop centered on agent and rotated to agent direction."""
        unwrapped = self.env.unwrapped
        agent_pos = unwrapped.agent_pos
        agent_dir = unwrapped.agent_dir

        fwd = DIR_TO_VEC[agent_dir]
        right = np.array([-fwd[1], fwd[0]])

        # Vectorized mapping from egocentric crop coordinates (u, v) to world coordinates (wx, wy)
        world_x = agent_pos[0] + self.dy_ego * fwd[0] + self.dx_ego * right[0]
        world_y = agent_pos[1] + self.dy_ego * fwd[1] + self.dx_ego * right[1]

        valid = (
            (world_x >= 0)
            & (world_x < self.width)
            & (world_y >= 0)
            & (world_y < self.height)
        )

        crop = np.zeros((self.crop_size, self.crop_size, MEM_NUM_CHANNELS), dtype=np.float32)
        valid_wx = world_x[valid]
        valid_wy = world_y[valid]

        crop[valid, MEM_REVEALED] = self.world_revealed[valid_wx, valid_wy].astype(np.float32)
        crop[valid, MEM_VISITED] = self.world_visited[valid_wx, valid_wy].astype(np.float32)
        crop[valid, MEM_KNOWN_WALL] = self.world_walls[valid_wx, valid_wy].astype(np.float32)
        crop[valid, MEM_KNOWN_POI] = self.world_pois[valid_wx, valid_wy].astype(np.float32)

        return crop

    def _get_view(self, obs: dict[str, Any]) -> np.ndarray:
        """Extract one-hot encoded 7x7 local field of view."""
        obj_types = obs["image"][:, :, 0]
        view = np.zeros((self.view_size, self.view_size, VIEW_NUM_CHANNELS), dtype=np.float32)

        is_unseen = obj_types == OBJECT_TO_IDX["unseen"]
        is_wall = obj_types == OBJECT_TO_IDX["wall"]
        is_poi = obj_types == OBJECT_TO_IDX["poi"]
        is_monster = obj_types == OBJECT_TO_IDX["monster"]
        is_empty = (~is_unseen) & (~is_wall) & (~is_poi) & (~is_monster)

        view[:, :, VIEW_UNSEEN] = is_unseen.astype(np.float32)
        view[:, :, VIEW_EMPTY] = is_empty.astype(np.float32)
        view[:, :, VIEW_WALL] = is_wall.astype(np.float32)
        view[:, :, VIEW_POI] = is_poi.astype(np.float32)
        view[:, :, VIEW_MONSTER] = is_monster.astype(np.float32)

        return view

    def _get_step_ratio(self) -> np.ndarray:
        """Compute normalized time step ratio t / max_steps."""
        unwrapped = self.env.unwrapped
        step_count = getattr(unwrapped, "step_count", 0)
        max_steps = getattr(unwrapped, "max_steps", 1)
        ratio = float(step_count) / float(max_steps)
        return np.array([np.clip(ratio, 0.0, 1.0)], dtype=np.float32)

    def observation(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert raw environment observation to structured, non-leaking policy observation."""
        self._update_memory()

        policy_obs: dict[str, np.ndarray] = {
            "view": self._get_view(obs),
        }

        if self.use_memory:
            policy_obs["memory"] = self._get_memory_crop()

        if self.include_step_ratio:
            policy_obs["step_ratio"] = self._get_step_ratio()

        return policy_obs


# Alias for concise import
ExploreObsWrapper = ObservationMemoryWrapper

