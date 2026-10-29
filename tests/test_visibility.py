import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall

from explorer.envs.explore_env import ExploreEnv
from explorer.envs.visibility import compute_raycast_vis_mask


def test_wall_blocks_visibility_strictly():
    """Verify that a wall completely occludes tiles located behind it."""
    # 7x7 grid with agent at bottom-center (3, 6)
    grid = Grid(7, 7)
    # Put a horizontal wall across row 4 with an opening at column 6
    for i in range(6):
        grid.set(i, 4, Wall())

    vis_mask = compute_raycast_vis_mask(grid, agent_pos=(3, 6))

    # Agent is at (3, 6)
    assert vis_mask[3, 6] is True or vis_mask[3, 6] == 1

    # Row 4 walls in front of agent must be visible (the front surface)
    for i in range(6):
        assert vis_mask[i, 4] is True or vis_mask[i, 4] == 1

    # Behind the wall (rows 0, 1, 2, 3 in columns 0..4) MUST NOT be visible!
    for r in range(4):
        for c in range(5):
            assert not vis_mask[c, r], (
                f"Occlusion leak detected! Tile ({c}, {r}) behind wall was marked visible."
            )


def test_diagonal_corner_occlusion_no_leak():
    """Verify that rays do not leak diagonally through wall corners (supercover test)."""
    env = ExploreEnv(difficulty="medium")
    env.reset(seed=42)
    # Exact scenario from user inspection: agent at (8, 12) facing West
    env.agent_pos = (8, 12)
    env.agent_dir = 2  # West

    _, vis_mask = env.gen_obs_grid()

    # Map to world coordinates to check the POI at (7, 14) behind wall at (7, 13)
    f_vec = env.dir_vec
    r_vec = env.right_vec
    top_left = (
        env.agent_pos
        + f_vec * (env.agent_view_size - 1)
        - r_vec * (env.agent_view_size // 2)
    )

    highlight_mask = np.zeros(shape=(env.width, env.height), dtype=bool)
    for vis_j in range(env.agent_view_size):
        for vis_i in range(env.agent_view_size):
            if not vis_mask[vis_i, vis_j]:
                continue
            abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
            if 0 <= abs_i < env.width and 0 <= abs_j < env.height:
                highlight_mask[abs_i, abs_j] = True

    # POI at (7, 14) must NOT be visible because wall at (7, 13) blocks it
    assert not highlight_mask[7, 14], "POI at (7, 14) leaked through corner!"
    assert not highlight_mask[7, 15], "Tile at (7, 15) leaked through corner!"
    assert highlight_mask[7, 13], "Wall at (7, 13) should be visible as an obstacle boundary."


def test_explore_env_visibility_no_see_through_walls():
    """Verify that in ExploreEnv, tiles directly behind walls are unseen."""
    env = ExploreEnv(difficulty="easy")
    env.reset(seed=42)

    sub_grid, vis_mask = env.gen_obs_grid()
    assert vis_mask.shape == (7, 7)

    # Validate that for any unseen tile, sub_grid at that location is None (obscured)
    for tx in range(7):
        for ty in range(7):
            if not vis_mask[tx, ty]:
                assert sub_grid.get(tx, ty) is None
