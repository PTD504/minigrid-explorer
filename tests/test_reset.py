import numpy as np

from explorer.envs.explore_env import ExploreEnv


def test_reset_reproducible():
    """Verify that resetting the environment with the same seed produces identical state and observations."""
    e1, e2 = ExploreEnv(difficulty="easy"), ExploreEnv(difficulty="easy")
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)

    assert np.array_equal(o1["image"], o2["image"])
    assert tuple(e1.agent_pos) == tuple(e2.agent_pos)
    assert e1.agent_dir == e2.agent_dir
    assert e1.walkable_tiles == e2.walkable_tiles
    assert e1.poi_positions == e2.poi_positions


def test_different_seeds_differ():
    """Verify that different seeds produce distinct environments and starting coordinates."""
    e = ExploreEnv(difficulty="easy")
    positions = set()
    for s in range(10):
        e.reset(seed=s)
        positions.add(tuple(e.agent_pos))
    assert len(positions) > 1


def test_action_space_is_three():
    """Verify action space is restricted to turn left, turn right, and move forward."""
    assert ExploreEnv().action_space.n == 3