from __future__ import annotations

import math

import pytest

from explorer.envs.explore_env import ExploreEnv
from explorer.envs.monster import DIRECTIONS
from explorer.rewards import RewardEngine, RewardWrapper
from explorer.utils.config import (
    get_env_config,
    get_reward_config,
    validate_config_invariants,
)


def test_anti_suicide_invariant():
    """Verify that c_death > c_step * max_steps holds across all difficulty levels with safety margin."""
    reward_cfg = get_reward_config()
    c_death = reward_cfg["c_death"]
    c_step = reward_cfg["c_step"]

    difficulties = ["easy", "medium", "hard"]
    for diff in difficulties:
        env_cfg = get_env_config(diff)
        max_steps = env_cfg["max_steps"]
        cumulative_step_cost = c_step * max_steps

        # Strict invariant: c_death must exceed maximum possible episode step cost
        assert (
            c_death > cumulative_step_cost
        ), f"Anti-suicide invariant failed for {diff}: c_death={c_death} <= {cumulative_step_cost}"

        # Ensure safety margin (> 1.0x) so suicide into monster is never optimal
        margin = c_death / cumulative_step_cost
        assert margin > 1.0, f"Safety margin violated for {diff}: {margin:.2f}x"

        # Validate with config validator utility
        validate_config_invariants(env_cfg, reward_cfg)

    # Negative test: intentionally violating invariant raises AssertionError
    bad_reward_cfg = dict(reward_cfg, c_death=0.1)
    with pytest.raises(AssertionError, match="Anti-suicide invariant violated"):
        validate_config_invariants(get_env_config("easy"), bad_reward_cfg)


def test_telescoping_property():
    """Exploration reward telescopes: sum(delta_N) over episode equals total newly revealed walkable tiles.

    Re-visiting or re-scanning known tiles yields zero reveal reward.
    """
    env = ExploreEnv(difficulty="easy")
    _obs, info = env.reset(seed=42)

    initial_revealed = info["revealed_count"]
    w_reveal = env.reward_engine.w_reveal

    total_reveal_reward = 0.0

    # 1. Do a 360-degree rotation (4 turns). Initial turns may reveal tiles, subsequent will not.
    for _ in range(4):
        _obs, _step_reward, term, trunc, step_info = env.step(0)
        bd = step_info["reward_breakdown"]
        total_reveal_reward += bd["reward_reveal"]
        assert bd["reward_reveal"] == w_reveal * step_info["delta_revealed"]
        if term or trunc:
            break

    # 2. Rotate another 4 turns in place - must yield exactly 0 new reveals and 0 reveal reward
    for _ in range(4):
        _obs, _step_reward, _term, _trunc, step_info = env.step(0)
        bd = step_info["reward_breakdown"]
        assert step_info["delta_revealed"] == 0
        assert bd["reward_reveal"] == 0.0
        total_reveal_reward += bd["reward_reveal"]

    # 3. Step forward and backward
    for _ in range(10):
        _obs, _step_reward, term, trunc, step_info = env.step(2)  # Forward (might bump or move)
        bd = step_info["reward_breakdown"]
        total_reveal_reward += bd["reward_reveal"]
        if term or trunc:
            break

    # Telescoping property verification: cumulative reveal reward == w_reveal * (|R_T ∩ W| - |R_0 ∩ W|)
    final_revealed = step_info["revealed_count"]
    revealed_during_episode = final_revealed - initial_revealed
    expected_reveal_reward = w_reveal * revealed_during_episode

    assert math.isclose(
        total_reveal_reward, expected_reveal_reward, abs_tol=1e-7
    ), f"Telescoping violated: got {total_reveal_reward}, expected {expected_reveal_reward}"


def test_poi_reward():
    """POI discovery reward triggers exactly once per discovered POI with value w_poi."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    w_poi = env.reward_engine.w_poi
    undiscovered = list(env.poi_positions - env.tracker.discovered_pois)
    assert len(undiscovered) > 0, "Expected at least one undiscovered POI at start"

    target_poi = undiscovered[0]
    px, py = target_poi

    # Find open tile adjacent to POI
    placed = False
    for dx, dy in DIRECTIONS:
        adj = (px - dx, py - dy)
        if adj in env.walkable_tiles and env.grid.get(*adj) is None:
            env.agent_pos = adj
            dir_map = {(0, -1): 3, (1, 0): 0, (0, 1): 1, (-1, 0): 2}
            env.agent_dir = dir_map[(dx, dy)]
            placed = True
            break

    assert placed, "Could not find adjacent tile to test POI reward"

    # Step facing POI -> POI is discovered in line of sight
    _obs, _step_reward, _term, _trunc, step_info = env.step(0)  # Turn in place to trigger LOS update
    bd = step_info["reward_breakdown"]

    assert target_poi in env.tracker.discovered_pois
    assert step_info["delta_pois"] == 1
    assert bd["reward_poi"] == w_poi

    # Step again while still facing or seeing the POI -> must NOT re-award POI reward
    _obs, _step_reward_2, _term, _trunc, step_info_2 = env.step(0)
    bd2 = step_info_2["reward_breakdown"]

    assert step_info_2["delta_pois"] == 0
    assert bd2["reward_poi"] == 0.0


def test_bump_penalty():
    """Attempting forward action into a wall triggers c_bump penalty and sets bump=True."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    c_bump = env.reward_engine.c_bump

    # Find a direction where front cell is a wall
    ax, ay = env.agent_pos
    dir_to_fwd = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    wall_dir = None
    for d, (fdx, fdy) in enumerate(dir_to_fwd):
        nx, ny = ax + fdx, ay + fdy
        cell = env.grid.get(nx, ny)
        if cell is not None and not cell.can_overlap():
            wall_dir = d
            break

    assert wall_dir is not None, "Could not find adjacent wall from start position"

    # Align agent to face the wall
    env.agent_dir = wall_dir

    # Execute forward action into wall
    _obs, _step_reward, _term, _trunc, info = env.step(2)
    bd = info["reward_breakdown"]

    assert info["bump"] is True
    assert bd["penalty_bump"] == -c_bump

    # Now face an open floor tile if available and move forward
    for d, (fdx, fdy) in enumerate(dir_to_fwd):
        nx, ny = ax + fdx, ay + fdy
        cell = env.grid.get(nx, ny)
        if cell is None:  # Open floor
            env.agent_dir = d
            _obs, _step_reward_free, _term, _trunc, info_free = env.step(2)
            assert info_free["bump"] is False
            assert info_free["reward_breakdown"]["penalty_bump"] == 0.0
            break


def test_loop_penalty():
    """Repeating a state within loop_window without new reveals triggers c_loop penalty."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    c_loop = env.reward_engine.c_loop

    # 1. First 4 turns in place reveal all directions around starting position
    for _ in range(4):
        env.step(0)

    # 2. Next turn left: state was visited recently in loop_window, and ΔN = 0, ΔP = 0
    _obs, _step_reward, _term, _trunc, info = env.step(0)
    bd = info["reward_breakdown"]

    assert info["delta_revealed"] == 0
    assert info["delta_pois"] == 0
    assert bd["penalty_loop"] == -c_loop

    # 3. Test loop suppression: when state is in window BUT new tile is revealed, no loop penalty
    engine = RewardEngine()
    engine.reset()

    # Step 1: visit (5, 5, 0) with new reveal
    _r1, bd1 = engine.compute_reward(
        agent_pos=(5, 5),
        agent_dir=0,
        delta_revealed=2,
        delta_pois=0,
        bump=False,
        terminated=False,
        end_reason=None,
        step_count=1,
        max_steps=250,
    )
    assert bd1["penalty_loop"] == 0.0

    # Step 2: return to (5, 5, 0) with zero reveals -> loop penalty applied
    _r2, bd2 = engine.compute_reward(
        agent_pos=(5, 5),
        agent_dir=0,
        delta_revealed=0,
        delta_pois=0,
        bump=False,
        terminated=False,
        end_reason=None,
        step_count=2,
        max_steps=250,
    )
    assert bd2["penalty_loop"] == -c_loop

    # Step 3: return to (5, 5, 0) with non-zero reveal -> loop penalty suppressed!
    _r3, bd3 = engine.compute_reward(
        agent_pos=(5, 5),
        agent_dir=0,
        delta_revealed=1,
        delta_pois=0,
        bump=False,
        terminated=False,
        end_reason=None,
        step_count=3,
        max_steps=250,
    )
    assert bd3["penalty_loop"] == 0.0


def test_death_penalty():
    """Colliding with a monster yields -c_death terminal penalty."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    c_death = env.reward_engine.c_death
    monster = env.monsters[0]
    mx, my = monster.pos

    # Put agent adjacent to monster facing monster
    placed = False
    for dx, dy, required_dir in [(0, -1, 3), (1, 0, 0), (0, 1, 1), (-1, 0, 2)]:
        adj = (mx - dx, my - dy)
        if adj in env.walkable_tiles and env.grid.get(*adj) is None:
            env.agent_pos = adj
            env.agent_dir = required_dir
            env.step_count = 0  # Step 1, monster stays in place (period=3)
            placed = True
            break

    assert placed, "Could not position agent adjacent to monster"

    # Step forward into monster
    _obs, _step_reward, term, _trunc, info = env.step(2)
    bd = info["reward_breakdown"]

    assert term is True
    assert info["end_reason"] == "death"
    assert bd["penalty_death"] == -c_death


def test_reward_breakdown():
    """Total reward returned by step() exactly equals the sum of all breakdown components."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=123)

    expected_keys = {
        "reward_reveal",
        "reward_poi",
        "penalty_step",
        "penalty_bump",
        "penalty_loop",
        "penalty_death",
        "bonus_done",
        "total_reward",
    }

    for action in [0, 1, 2, 0, 2, 2, 1, 2]:
        _obs, reward, term, trunc, info = env.step(action)
        bd = info["reward_breakdown"]

        # Check all required keys exist
        assert set(bd.keys()) == expected_keys

        # Check total_reward matches scalar reward
        assert math.isclose(reward, bd["total_reward"], abs_tol=1e-7)

        # Check sum of all individual components matches total_reward
        component_sum = (
            bd["reward_reveal"]
            + bd["reward_poi"]
            + bd["penalty_step"]
            + bd["penalty_bump"]
            + bd["penalty_loop"]
            + bd["penalty_death"]
            + bd["bonus_done"]
        )
        assert math.isclose(component_sum, bd["total_reward"], abs_tol=1e-7)

        if term or trunc:
            break


def test_coverage_done_bonus():
    """Reaching target coverage threshold triggers one-time b_done * (1 - t/T) bonus."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    b_done = env.reward_engine.b_done
    max_steps = env.max_steps

    # Manually simulate reaching coverage threshold
    env.tracker.revealed_tiles = set(env.walkable_tiles)
    assert env.tracker.revealed_coverage == 1.0 >= env.target_coverage

    # Step 1: should receive early completion bonus
    _obs, _step_reward, _term, _trunc, info = env.step(0)
    bd = info["reward_breakdown"]

    step_t = env.step_count
    expected_bonus = b_done * (1.0 - float(step_t) / float(max_steps))
    assert math.isclose(bd["bonus_done"], expected_bonus, abs_tol=1e-6)
    assert bd["bonus_done"] > 0.0

    # Step 2: coverage is still >= target_coverage, but bonus must NOT trigger again
    _obs, _step_reward_2, _term, _trunc, info_2 = env.step(0)
    bd2 = info_2["reward_breakdown"]
    assert bd2["bonus_done"] == 0.0


def test_reward_wrapper():
    """RewardWrapper correctly computes reward and attaches reward_breakdown."""
    base_env = ExploreEnv(difficulty="easy")
    wrapped_env = RewardWrapper(base_env)

    obs, _info = wrapped_env.reset(seed=42)
    assert obs is not None

    obs, reward, _term, _trunc, step_info = wrapped_env.step(0)
    assert "reward_breakdown" in step_info
    assert math.isclose(reward, step_info["reward_breakdown"]["total_reward"], abs_tol=1e-7)

    # Test coverage bonus through RewardWrapper
    wrapped_env.unwrapped.tracker.revealed_tiles = set(wrapped_env.unwrapped.walkable_tiles)
    obs, _reward_cov, _term, _trunc, step_info_cov = wrapped_env.step(0)
    assert step_info_cov["reward_breakdown"]["bonus_done"] > 0.0


def test_reward_engine_reset_with_initial_state():
    """RewardEngine reset with optional initial state tracks it in history."""
    engine = RewardEngine()
    engine.reset(initial_state=(3, 4, 1))
    assert (3, 4, 1) in engine.state_counts
    assert len(engine.state_history) == 1


def test_no_normalization_leakage():
    """w_reveal is a fixed scalar constant and never scaled by total walkable area |W|."""
    engine = RewardEngine()
    engine.reset()

    # Revealing 5 tiles always yields exactly 5 * w_reveal regardless of map size
    _, bd1 = engine.compute_reward(
        agent_pos=(1, 1),
        agent_dir=0,
        delta_revealed=5,
        delta_pois=0,
        bump=False,
        terminated=False,
        end_reason=None,
        step_count=1,
        max_steps=250,
    )
    assert math.isclose(bd1["reward_reveal"], 5 * engine.w_reveal, abs_tol=1e-7)
