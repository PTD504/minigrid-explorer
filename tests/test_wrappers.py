from __future__ import annotations

import numpy as np
import pytest

from explorer.envs.explore_env import ExploreEnv
from explorer.envs.objects import Monster, PointOfInterest
from explorer.envs.wrappers import (
    MEM_KNOWN_POI,
    MEM_KNOWN_WALL,
    MEM_REVEALED,
    MEM_VISITED,
    VIEW_MONSTER,
    ExploreObsWrapper,
    ObservationMemoryWrapper,
)


def test_obs_space_and_dict_keys():
    """Verify that observation has exact Dict format and strips privileged fields."""
    env = ExploreEnv(difficulty="easy")
    wrapped = ObservationMemoryWrapper(env)
    obs, _ = wrapped.reset(seed=42)

    # Required keys in policy observation
    assert set(obs.keys()) == {"view", "memory", "step_ratio"}
    assert "mission" not in obs
    assert "image" not in obs
    assert "direction" not in obs

    # Shapes and dtypes
    assert obs["view"].shape == (7, 7, 5)
    assert obs["view"].dtype == np.float32

    assert obs["memory"].shape == (15, 15, 4)
    assert obs["memory"].dtype == np.float32

    assert obs["step_ratio"].shape == (1,)
    assert obs["step_ratio"].dtype == np.float32

    # Verify gymnasium space compliance
    assert wrapped.observation_space.contains(obs)

    # Verify strictly one-hot representation for view
    view_sum = np.sum(obs["view"], axis=-1)
    assert np.allclose(view_sum, 1.0), "Every cell in view must be strictly one-hot"


def test_alias_compatibility():
    """Verify ExploreObsWrapper is a drop-in alias for ObservationMemoryWrapper."""
    assert ExploreObsWrapper is ObservationMemoryWrapper
    env = ExploreEnv(difficulty="easy")
    wrapped = ExploreObsWrapper(env)
    obs, _ = wrapped.reset(seed=42)
    assert "memory" in obs


def test_ablation_flags():
    """Verify that memory and step_ratio can be enabled/disabled for ablation studies."""
    env = ExploreEnv(difficulty="easy")

    # Ablation: use_memory = False
    no_mem_env = ObservationMemoryWrapper(env, use_memory=False)
    obs, _ = no_mem_env.reset(seed=42)
    assert "memory" not in obs
    assert set(obs.keys()) == {"view", "step_ratio"}
    assert "memory" not in no_mem_env.observation_space.spaces
    assert no_mem_env.observation_space.contains(obs)

    # Ablation: include_step_ratio = False
    no_ratio_env = ObservationMemoryWrapper(env, include_step_ratio=False)
    obs, _ = no_ratio_env.reset(seed=42)
    assert "step_ratio" not in obs
    assert set(obs.keys()) == {"view", "memory"}
    assert "step_ratio" not in no_ratio_env.observation_space.spaces
    assert no_ratio_env.observation_space.contains(obs)

    # Custom crop_size
    custom_crop_env = ObservationMemoryWrapper(env, crop_size=11)
    obs, _ = custom_crop_env.reset(seed=42)
    assert obs["memory"].shape == (11, 11, 4)

    # Even crop_size should raise ValueError
    with pytest.raises(ValueError, match="crop_size must be an odd integer"):
        ObservationMemoryWrapper(env, crop_size=12)


def test_step_ratio_progression():
    """Verify that step_ratio tracks t / max_steps monotonically within [0, 1]."""
    env = ExploreEnv(difficulty="easy", max_steps=20)
    wrapped = ObservationMemoryWrapper(env)
    obs, _ = wrapped.reset(seed=42)

    assert obs["step_ratio"][0] == 0.0

    for step in range(1, 10):
        obs, _, terminated, truncated, _ = wrapped.step(0)  # Turn left
        expected_ratio = float(step) / 20.0
        assert np.isclose(obs["step_ratio"][0], expected_ratio)
        if terminated or truncated:
            break


def test_egocentric_memory_rotation():
    """Verify that memory crop rotates egocentrically with agent orientation.

    When agent rotates:
    - Facing Up: wall in front is at (center, center - 1).
    - Turn Right (facing Right): wall is now at (center - 1, center) [left].
    - Turn Right (facing Down): wall is now at (center, center + 1) [behind].
    - Turn Right (facing Left): wall is now at (center + 1, center) [right].
    """
    env = ExploreEnv(difficulty="easy")
    wrapped = ObservationMemoryWrapper(env)
    obs, _ = wrapped.reset(seed=42)

    center = wrapped.center
    assert center == 7

    # In seed 42, agent starts at (5, 5) facing UP (3), with a wall at (5, 4)
    # Check that wall is in front in memory
    assert obs["memory"][center, center - 1, MEM_KNOWN_WALL] == 1.0

    # 1. Turn Right (action 1) -> now facing RIGHT (dir 0)
    obs, _, _, _, _ = wrapped.step(1)
    assert obs["memory"][center - 1, center, MEM_KNOWN_WALL] == 1.0, "Wall must rotate to the left"

    # 2. Turn Right (action 1) -> now facing DOWN (dir 1)
    obs, _, _, _, _ = wrapped.step(1)
    assert obs["memory"][center, center + 1, MEM_KNOWN_WALL] == 1.0, "Wall must rotate behind"

    # 3. Turn Right (action 1) -> now facing LEFT (dir 2)
    obs, _, _, _, _ = wrapped.step(1)
    assert obs["memory"][center + 1, center, MEM_KNOWN_WALL] == 1.0, "Wall must rotate to the right"


def test_memory_accumulation_and_visited():
    """Verify that memory accumulates visited, revealed, and known POI tiles."""
    env = ExploreEnv(difficulty="easy")
    wrapped = ObservationMemoryWrapper(env)
    obs, _ = wrapped.reset(seed=42)

    center = wrapped.center
    # Step 0: agent's own position is visited
    assert obs["memory"][center, center, MEM_VISITED] == 1.0
    assert obs["memory"][center, center, MEM_REVEALED] == 1.0

    # Step forward or turn and verify memory accumulates
    initial_revealed_count = np.sum(obs["memory"][:, :, MEM_REVEALED])
    for _ in range(5):
        obs, _, _, _, _ = wrapped.step(env.action_space.sample())

    later_revealed_count = np.sum(obs["memory"][:, :, MEM_REVEALED])
    assert later_revealed_count >= initial_revealed_count

    # If POI is discovered in memory, verify channel value
    if len(env.tracker.discovered_pois) > 0:
        assert np.any(obs["memory"][:, :, MEM_KNOWN_POI] == 1.0)


def test_no_privileged_leak_hidden_poi():
    """Verify changing a hidden POI outside line-of-sight does NOT alter observation."""
    env = ExploreEnv(difficulty="easy")
    wrapped = ObservationMemoryWrapper(env)
    _obs_initial, _ = wrapped.reset(seed=42)

    # Find an unrevealed, unoccupied interior floor tile
    unrevealed_floor: tuple[int, int] | None = None
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            if (x, y) not in env.tracker.revealed_tiles and env.grid.get(x, y) is None:
                unrevealed_floor = (x, y)
                break
        if unrevealed_floor is not None:
            break

    assert unrevealed_floor is not None, "Could not find unrevealed tile for leak test"

    obs_before = wrapped.observation(env.gen_obs())

    # Add a POI in that unrevealed room
    env.grid.set(*unrevealed_floor, PointOfInterest())

    obs_after = wrapped.observation(env.gen_obs())

    # Strictly no change in policy observation
    assert np.array_equal(obs_before["view"], obs_after["view"]), "Hidden POI leaked into view!"
    assert np.array_equal(
        obs_before["memory"], obs_after["memory"]
    ), "Hidden POI leaked into memory!"


def test_no_privileged_leak_hidden_monster():
    """Verify moving a monster outside line-of-sight does NOT alter observation."""
    env = ExploreEnv(difficulty="easy")
    wrapped = ObservationMemoryWrapper(env)
    obs_initial, _ = wrapped.reset(seed=42)

    # Ensure monster exists
    assert len(env.monsters) > 0
    monster = env.monsters[0]

    # Verify monster is not in initial view
    assert np.sum(obs_initial["view"][:, :, VIEW_MONSTER]) == 0.0

    obs_before = wrapped.observation(env.gen_obs())

    # Find an unrevealed tile to teleport the monster to
    unrevealed_tile = None
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            if (x, y) not in env.tracker.revealed_tiles and (x, y) in env.walkable_tiles:
                unrevealed_tile = (x, y)
                break
        if unrevealed_tile is not None:
            break

    assert unrevealed_tile is not None

    # Move monster to that hidden tile
    env.grid.set(*monster.pos, None)
    monster.pos = unrevealed_tile
    env.grid.set(*unrevealed_tile, Monster())

    obs_after = wrapped.observation(env.gen_obs())

    assert np.array_equal(
        obs_before["view"], obs_after["view"]
    ), "Hidden monster leaked into view!"
    assert np.array_equal(
        obs_before["memory"], obs_after["memory"]
    ), "Hidden monster leaked into memory!"


def test_out_of_bounds_padding_does_not_leak_map_size():
    """Verify that out-of-bounds padded cells in memory are all 0 (unknown)."""
    env = ExploreEnv(difficulty="easy")
    wrapped = ObservationMemoryWrapper(env, crop_size=15)
    wrapped.reset(seed=42)

    # Place agent at top-left corner (1, 1) facing UP
    env.agent_pos = (1, 1)
    env.agent_dir = 3

    obs = wrapped.observation(env.gen_obs())
    mem = obs["memory"]
    center = wrapped.center  # 7

    # Since agent is at (1, 1) facing UP:
    # 2 steps forward is world y = -1 (outside map)
    # The cell at (center, center - 2) = (7, 5) is out-of-bounds
    out_of_bounds_cell = mem[center, center - 2]

    # Must be all zeros (identical to unrevealed interior tiles)
    assert np.all(out_of_bounds_cell == 0.0), "Out of bounds cells must be padded with 0"
