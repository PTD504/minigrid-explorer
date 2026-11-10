from __future__ import annotations

import numpy as np
import pytest
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.world_object import Wall

from explorer.envs.explore_env import ExploreEnv
from explorer.envs.monster import compute_bfs_distance_map
from explorer.envs.objects import Monster


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_monster_spawn_min_dist(difficulty: str):
    """Monsters must spawn at BFS distance >= monster_min_dist from the agent's start position."""
    for seed in [1, 42, 123, 999]:
        env = ExploreEnv(difficulty=difficulty)
        env.reset(seed=seed)

        expected_min_dist = env.config.get("monster_min_dist", 5)
        dist_map = compute_bfs_distance_map(env.agent_pos, env.walkable_tiles)

        assert len(env.monsters) == env.config.get("num_monsters", 1)
        for monster in env.monsters:
            actual_dist = dist_map.get(monster.pos)
            assert actual_dist is not None, f"Monster at {monster.pos} is not on a reachable walkable tile!"
            max_possible = max(dist_map.values())
            assert actual_dist >= min(expected_min_dist, max_possible)


def test_monster_no_wall_clip():
    """Monsters must respect walls and boundaries, never clipping through obstacles."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    for _ in range(80):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        for monster in env.monsters:
            mx, my = monster.pos
            assert 1 <= mx < env.width - 1
            assert 1 <= my < env.height - 1
            assert (mx, my) in env.walkable_tiles
            cell = env.grid.get(mx, my)
            assert not isinstance(cell, Wall)
            assert isinstance(cell, Monster)
        if term or trunc:
            env.reset(seed=42)


def test_monster_determinism():
    """Fixed seed and identical agent action sequence produce identical monster trajectories."""
    def run_rollout(seed: int, actions: list[int]):
        env = ExploreEnv(difficulty="easy")
        env.reset(seed=seed)
        trajectory = []
        for a in actions:
            _, _, term, trunc, _ = env.step(a)
            trajectory.append([(m.id, m.pos, m.state) for m in env.monsters])
            if term or trunc:
                break
        return trajectory

    actions = [2, 0, 2, 2, 1, 2, 0, 2, 1, 2] * 4  # 40 steps
    traj1 = run_rollout(seed=101, actions=actions)
    traj2 = run_rollout(seed=101, actions=actions)

    assert len(traj1) == len(traj2)
    for step_idx, (t1, t2) in enumerate(zip(traj1, traj2)):
        assert t1 == t2, f"Discrepancy at step {step_idx}: {t1} vs {t2}"


def test_monster_chase_and_los():
    """Monster enters chase mode when agent is in line of sight and within detect_radius."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    monster = env.monsters[0]
    mx, my = monster.pos

    target_pos = None
    for dx in range(1, min(monster.detect_radius, 4)):
        cand = (mx + dx, my)
        if cand in env.walkable_tiles and env.grid.get(*cand) is None:
            from explorer.envs.visibility import has_line_of_sight
            if has_line_of_sight(env.grid, monster.pos, cand):
                target_pos = cand
                break

    if target_pos is not None:
        env.agent_pos = target_pos
        env.step_count = monster.period - 1
        _, _, _, _, _ = env.step(0)  # Agent turns (stays in place)

        assert monster.state == "chase"


def test_monster_collision_same_tile():
    """Agent and monster occupying the same tile terminates the episode with end_reason='death'."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    monster = env.monsters[0]
    mx, my = monster.pos

    for dx, dy, required_dir in [(0, -1, 3), (1, 0, 0), (0, 1, 1), (-1, 0, 2)]:
        adj = (mx - dx, my - dy)
        if adj in env.walkable_tiles and env.grid.get(*adj) is None:
            env.agent_pos = adj
            env.agent_dir = required_dir
            env.step_count = 0  # Next step is 1, monster won't move
            _, _, term, _, info = env.step(2)  # Move forward into monster tile
            assert term is True
            assert info["end_reason"] == "death"
            return

    pytest.skip("Could not find adjacent open floor tile for direct collision test")


def test_monster_collision_swap():
    """Agent and monster swapping positions in the same step triggers collision and death."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    monster = env.monsters[0]
    mx, my = monster.pos

    for dx, dy, required_dir in [(0, -1, 3), (1, 0, 0), (0, 1, 1), (-1, 0, 2)]:
        adj = (mx + dx, my + dy)
        if adj in env.walkable_tiles and env.grid.get(*adj) is None:
            monster.waypoints = [adj]
            monster.current_waypoint_idx = 0
            monster.state = "patrol"

            opposite_dir = (required_dir + 2) % 4
            env.agent_pos = adj
            env.agent_dir = opposite_dir
            env.step_count = monster.period - 1

            _, _, term, _, info = env.step(2)
            assert term is True
            assert info["end_reason"] == "death"
            return

    pytest.skip("Could not find suitable open configuration for swap collision test")


def test_multiple_monsters_no_overlap():
    """On Hard difficulty with 2 monsters, monsters must never overlap on the same tile."""
    env = ExploreEnv(difficulty="hard")
    env.reset(seed=777)

    assert len(env.monsters) == 2

    for step in range(120):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        m1_pos = env.monsters[0].pos
        m2_pos = env.monsters[1].pos
        assert m1_pos != m2_pos, f"Monsters overlapped at {m1_pos} on step {step}!"
        if term or trunc:
            env.reset(seed=777 + step)


def test_monster_in_observation_when_visible():
    """Monster must be encoded with OBJECT_TO_IDX['monster'] in obs['image'] when visible."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    monster = env.monsters[0]
    mx, my = monster.pos

    target_agent_pos = (mx, my + 2)
    if target_agent_pos in env.walkable_tiles:
        env.agent_pos = target_agent_pos
        env.agent_dir = 3  # North (up)

        obs = env.gen_obs()
        img = obs["image"]
        monster_id = OBJECT_TO_IDX["monster"]
        assert np.any(img[:, :, 0] == monster_id), "Monster should be visible in observation!"


def test_info_monsters_strict_pomdp_filtering():
    """info['monsters'] must only expose monsters currently in the agent's visible FoV."""
    env = ExploreEnv(difficulty="easy")
    _obs, _info = env.reset(seed=42)

    monster = env.monsters[0]

    # Case 1: Place monster far away in an unrevealed corner behind walls
    far_tiles = [
        t for t in env.walkable_tiles
        if abs(t[0] - env.agent_pos[0]) + abs(t[1] - env.agent_pos[1]) > 10
    ]
    if far_tiles:
        monster.pos = far_tiles[0]
        # Step in place (turn left)
        _, _, _, _, step_info = env.step(0)
        # Monster is outside FoV, info['monsters'] must be empty!
        assert len(step_info["monsters"]) == 0, (
            f"Hidden monster at {monster.pos} was leaked in info: {step_info['monsters']}"
        )

    # Case 2: Place monster directly 1 tile in front of agent
    ax, ay = env.agent_pos
    # Agent facing right (0)
    env.agent_dir = 0
    monster.pos = (ax + 1, ay)
    _, _, _, _, step_info2 = env.step(0)
    # Monster is visible in front of agent, so it must be present in step_info['monsters']
    visible_monster_positions = [m["pos"] for m in step_info2["monsters"]]
    assert monster.pos in visible_monster_positions, (
        f"Visible monster at {monster.pos} was not reported in info: {step_info2['monsters']}"
    )


def test_monster_mode_off_no_monsters():
    """monster_mode='off' ensures no monsters spawn or exist in the environment."""
    for diff in ["easy", "medium", "hard"]:
        env = ExploreEnv(difficulty=diff, monster_mode="off")
        obs, info = env.reset(seed=123)

        assert env.monster_manager is None
        assert len(env.monsters) == 0
        assert len(info["monsters"]) == 0
        assert info["near_monster"] is False
        assert info["monster_collisions"] == 0

        # Step and verify no monster appears
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        assert len(env.monsters) == 0
        assert len(info["monsters"]) == 0
        assert info["monster_collision"] is False


def test_monster_mode_reset_reproducible():
    """Identical seeds produce identical layouts for off and invulnerable modes."""
    # Test 'off' mode reproducibility
    env1 = ExploreEnv(difficulty="medium", monster_mode="off")
    env2 = ExploreEnv(difficulty="medium", monster_mode="off")
    obs1, _ = env1.reset(seed=777)
    obs2, _ = env2.reset(seed=777)
    assert np.array_equal(obs1["image"], obs2["image"])
    assert env1.agent_pos == env2.agent_pos
    assert env1.poi_positions == env2.poi_positions

    # Test 'invulnerable' mode reproducibility
    env_inv1 = ExploreEnv(difficulty="medium", monster_mode="invulnerable")
    env_inv2 = ExploreEnv(difficulty="medium", monster_mode="invulnerable")
    obs_inv1, _ = env_inv1.reset(seed=777)
    obs_inv2, _ = env_inv2.reset(seed=777)
    assert np.array_equal(obs_inv1["image"], obs_inv2["image"])
    assert [m.pos for m in env_inv1.monsters] == [m.pos for m in env_inv2.monsters]


def test_monster_mode_invulnerable_collision_survives():
    """monster_mode='invulnerable' counts collision but does NOT terminate episode or penalize death."""
    env = ExploreEnv(difficulty="easy", monster_mode="invulnerable")
    env.reset(seed=42)

    monster = env.monsters[0]
    ax, ay = env.agent_pos

    # Face East (dir=0) and place monster directly 1 step in front of agent
    env.agent_dir = 0
    front_pos = (ax + 1, ay)
    if front_pos in env.walkable_tiles:
        monster.pos = front_pos
        env.grid.set(front_pos[0], front_pos[1], monster.world_obj)

        # Action 2 is forward (step directly into monster tile)
        obs, rew, terminated, truncated, info = env.step(2)

        assert info["monster_collision"] is True
        assert info["monster_collisions"] >= 1
        assert terminated is False, "Invulnerable mode should not terminate on monster collision!"
        assert info["end_reason"] != "death"
        assert info["reward_breakdown"].get("penalty_death", 0.0) == 0.0


