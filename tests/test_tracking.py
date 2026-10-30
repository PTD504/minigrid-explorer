from __future__ import annotations

import pytest

from explorer.envs.explore_env import ExploreEnv


def test_initial_tracking_step_zero():
    """At step 0 (reset), tracker initializes with nonzero revealed tiles and valid coverage."""
    env = ExploreEnv(difficulty="easy")
    _obs, info = env.reset(seed=42)

    assert "revealed_coverage" in info
    assert "visited_coverage" in info
    assert "poi_discovery_rate" in info
    assert "delta_revealed" in info
    assert "delta_pois" in info

    # Step-0 field of view reveals at least the agent tile and immediately visible neighbors
    assert info["revealed_coverage"] > 0.0
    assert info["revealed_coverage"] <= 1.0
    assert info["delta_revealed"] > 0

    # Visited coverage at step 0 is exactly 1 tile (agent's start position)
    assert info["visited_count"] == 1
    assert info["visited_coverage"] == 1.0 / len(env.walkable_tiles)


def test_tracking_monotonic_coverage():
    """Revealed coverage and visited coverage must be monotonically non-decreasing."""
    env = ExploreEnv(difficulty="easy")
    _, info = env.reset(seed=101)

    prev_revealed_cov = info["revealed_coverage"]
    prev_visited_cov = info["visited_coverage"]

    for _ in range(60):
        action = env.action_space.sample()
        _, _, term, trunc, step_info = env.step(action)

        curr_revealed_cov = step_info["revealed_coverage"]
        curr_visited_cov = step_info["visited_coverage"]

        # Invariant: Coverage never decreases
        assert curr_revealed_cov >= prev_revealed_cov - 1e-9, "Revealed coverage decreased!"
        assert curr_visited_cov >= prev_visited_cov - 1e-9, "Visited coverage decreased!"
        assert curr_revealed_cov <= 1.0 + 1e-9, "Revealed coverage exceeded 100%!"
        assert curr_visited_cov <= 1.0 + 1e-9, "Visited coverage exceeded 100%!"

        # Invariant: delta_revealed is non-negative
        assert step_info["delta_revealed"] >= 0

        prev_revealed_cov = curr_revealed_cov
        prev_visited_cov = curr_visited_cov

        if term or trunc:
            break


def test_tracking_no_increase_on_pure_rotation():
    """Rotating in place multiple full circles without moving does not reveal new tiles once 360 deg is scanned."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    # First do a full 360-degree rotation (4 turns) to reveal all 4 directions
    for _ in range(4):
        env.step(0)  # Turn left

    coverage_after_360 = env.tracker.revealed_coverage

    # Subsequent turns must reveal 0 new tiles
    for _ in range(8):
        _, _, _, _, step_info = env.step(0)
        assert step_info["delta_revealed"] == 0, "Turning revealed new tiles after full 360 degree scan!"
        assert step_info["revealed_coverage"] == coverage_after_360


def test_poi_discovery_only_when_in_los():
    """POI is only discovered when it enters agent's true line-of-sight."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    # At reset, check if POI positions are far enough and not immediately all discovered
    # In easy.yaml, poi_min_dist is 4, so not all POIs should be seen immediately
    assert len(env.poi_positions) == env.config["num_pois"]
    assert env.tracker.poi_discovery_rate < 1.0

    # Pick a POI that has not been discovered yet
    undiscovered = list(env.poi_positions - env.tracker.discovered_pois)
    assert len(undiscovered) > 0, "Expected at least one undiscovered POI at start"
    target_poi = undiscovered[0]

    # Verify that moving agent adjacent to POI and facing it triggers discovery
    px, py = target_poi
    # Find adjacent walkable neighbor
    from explorer.envs.monster import DIRECTIONS
    for dx, dy in DIRECTIONS:
        adj = (px - dx, py - dy)
        if adj in env.walkable_tiles and env.grid.get(*adj) is None:
            env.agent_pos = adj
            # Face towards POI
            # DIRECTIONS: 0=(0,-1)[up/dir 3], 1=(1,0)[right/dir 0], 2=(0,1)[down/dir 1], 3=(-1,0)[left/dir 2]
            dir_map = {(0, -1): 3, (1, 0): 0, (0, 1): 1, (-1, 0): 2}
            env.agent_dir = dir_map[(dx, dy)]

            # Step in place (turn left then right, or just check update)
            prev_discovered_count = len(env.tracker.discovered_pois)
            _, _, _, _, info = env.step(0)

            assert target_poi in env.tracker.discovered_pois
            assert info["poi_discovery_rate"] > 0
            assert len(env.tracker.discovered_pois) >= prev_discovered_count
            return

    pytest.skip("Could not find adjacent tile for targeted POI LOS test")


def test_terminal_strictly_death_or_timeout():
    """Episode termination occurs strictly on monster collision or max_steps timeout (no coverage cutoff)."""
    env = ExploreEnv(difficulty="easy", max_steps=15)
    env.reset(seed=42)

    # Artificially mark all walkable tiles as revealed
    env.tracker.revealed_tiles = set(env.walkable_tiles)
    assert env.tracker.revealed_coverage == 1.0

    # Step 1: should NOT terminate simply because coverage is 100%
    # Ensure monster doesn't kill agent on this step
    env.monsters[0].pos = (env.width - 2, env.height - 2)
    _, _, term, trunc, info = env.step(0)

    assert term is False, "Episode must not terminate early on coverage during training!"
    assert trunc is False

    # Continue until timeout (15 steps)
    for _ in range(14):
        _, _, term, trunc, info = env.step(0)
        if term or trunc:
            break

    assert trunc is True
    assert info["end_reason"] == "timeout"
