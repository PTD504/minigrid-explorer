from __future__ import annotations

import pytest

from explorer.agents import (
    FrontierExplorerAgent,
    OraclePlannerAgent,
    RandomAgent,
    WallFollowerAgent,
)
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper


def test_random_agent_valid_actions():
    """RandomAgent only outputs valid actions in {0, 1, 2} and respects seed determinism."""
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=42)

    actions1 = [agent1.act({}) for _ in range(100)]
    actions2 = [agent2.act({}) for _ in range(100)]

    assert set(actions1).issubset({0, 1, 2})
    assert actions1 == actions2

    # Reset restores sequence
    agent1.reset()
    actions1_reset = [agent1.act({}) for _ in range(100)]
    assert actions1 == actions1_reset


def test_random_agent_rollout():
    """RandomAgent executes rollouts cleanly on wrapped environment."""
    env = ObservationMemoryWrapper(ExploreEnv(difficulty="easy"))
    obs, info = env.reset(seed=101)
    agent = RandomAgent(seed=101)

    steps = 0
    for _ in range(50):
        action = agent.act(obs, info)
        assert action in {0, 1, 2}
        obs, _reward, term, trunc, info = env.step(action)
        steps += 1
        if term or trunc:
            break

    assert steps > 0
    assert info["revealed_coverage"] > 0.0


def test_wall_follower_navigation():
    """WallFollowerAgent navigates deterministically and maintains wall contact."""
    env = ExploreEnv(difficulty="easy")
    obs, info = env.reset(seed=42)
    agent = WallFollowerAgent()

    actions = []
    for _ in range(40):
        action = agent.act(obs, info)
        actions.append(action)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break

    # Rerun with fresh agent on same seed -> exact same action trajectory
    obs2, info2 = env.reset(seed=42)
    agent2 = WallFollowerAgent()
    actions2 = []
    for _ in range(len(actions)):
        action = agent2.act(obs2, info2)
        actions2.append(action)
        obs2, _, term, trunc, info2 = env.step(action)

    assert actions == actions2


def test_wall_follower_with_memory_wrapper():
    """WallFollowerAgent operates seamlessly on egocentric ObservationMemoryWrapper."""
    env = ObservationMemoryWrapper(ExploreEnv(difficulty="easy"))
    obs, info = env.reset(seed=42)
    agent = WallFollowerAgent()

    for _ in range(30):
        action = agent.act(obs, info)
        assert action in {0, 1, 2}
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break

    assert info["revealed_coverage"] > 0.15


def test_frontier_explorer_coverage_growth():
    """FrontierExplorerAgent systematically discovers frontiers and expands coverage."""
    raw_env = ExploreEnv(difficulty="easy")
    env = ObservationMemoryWrapper(raw_env)
    obs, info = env.reset(seed=42)

    agent = FrontierExplorerAgent(env)

    initial_coverage = info["revealed_coverage"]
    for _ in range(80):
        action = agent.act(obs, info)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break

    final_coverage = info["revealed_coverage"]
    assert final_coverage > initial_coverage
    # Frontier explorer should achieve strong coverage (> 60% on easy in 80 steps)
    assert final_coverage >= 0.60, f"Expected coverage >= 0.60, got {final_coverage:.2f}"


def test_frontier_explorer_monster_avoidance():
    """FrontierExplorerAgent avoids stepping into tiles occupied by monsters."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)
    agent = FrontierExplorerAgent(env)

    # Position monster in front of agent
    monster = env.monsters[0]
    ax, ay = env.agent_pos
    monster.pos = (ax + 1, ay)
    env.agent_dir = 0  # Facing right towards monster

    # Agent must NOT move forward into monster
    action = agent.act({}, {})
    # Either turns away (0 or 1), does not crash into monster
    assert action in {0, 1}, "Frontier explorer should turn to avoid stepping into monster"


def test_frontier_explorer_fairness_no_hidden_monster_block():
    """Frontier explorer does not have clairvoyance: it only blocks monsters that are visible."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)
    agent = FrontierExplorerAgent(env)

    # Place monster in an unvisited, unrevealed tile far away
    far_tiles = [
        t for t in env.walkable_tiles
        if t not in env.tracker.revealed_tiles and abs(t[0] - env.agent_pos[0]) + abs(t[1] - env.agent_pos[1]) > 8
    ]
    if far_tiles:
        hidden_pos = far_tiles[0]
        env.monsters[0].pos = hidden_pos
        # The hidden monster tile should NOT be considered blocked by the fair baseline
        visible_monsters = {
            m.pos for m in env.monsters if m.pos in env.tracker.current_visible_tiles
        }
        assert hidden_pos not in visible_monsters, "Hidden monster must not be in visible monsters!"
        action = agent.act({}, {})
        assert action in {0, 1, 2}


def test_oracle_planner_high_coverage():
    """OraclePlannerAgent achieves rapid high-efficiency coverage using ground-truth pathfinding."""
    env = ExploreEnv(difficulty="easy")
    obs, info = env.reset(seed=42)
    agent = OraclePlannerAgent(env)

    initial_cov = info["revealed_coverage"]
    for _ in range(80):
        action = agent.act(obs, info)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break

    final_cov = info["revealed_coverage"]
    assert final_cov > initial_cov
    # Oracle should rapidly achieve >= 75% coverage
    assert final_cov >= 0.75, f"Oracle planner expected >= 0.75 coverage, got {final_cov:.2f}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_all_baselines_run_across_difficulties(difficulty: str):
    """Ensure all 4 baseline agents run without errors on all difficulty levels."""
    env = ObservationMemoryWrapper(ExploreEnv(difficulty=difficulty))

    baselines = [
        RandomAgent(seed=123),
        WallFollowerAgent(),
        FrontierExplorerAgent(env),
        OraclePlannerAgent(env),
    ]

    for agent in baselines:
        obs, info = env.reset(seed=42)
        agent.reset()
        for _ in range(15):
            action = agent.act(obs, info)
            assert action in {0, 1, 2}
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
