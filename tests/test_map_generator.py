import numpy as np
import pytest

from explorer.envs.map_generator import ProceduralMapGenerator
from explorer.utils.config import get_env_config


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_map_generator_connectivity_and_structures(difficulty: str):
    """Test that generated maps across difficulties satisfy structural and connectivity invariants."""
    config = get_env_config(difficulty)
    generator = ProceduralMapGenerator(config)
    rng = np.random.default_rng(100)

    _grid, meta = generator.generate(config["size"], config["size"], rng)

    # 1. 100% reachability from agent start position
    start_pos = meta.agent_start_pos
    assert start_pos in meta.walkable_tiles

    visited = {start_pos}
    queue = [start_pos]
    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nbr = (cx + dx, cy + dy)
            if nbr in meta.walkable_tiles and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    assert visited == meta.walkable_tiles, "All walkable tiles must be reachable via BFS!"

    # 2. Check room count constraints
    assert len(meta.rooms) >= config["num_rooms_min"]

    # 3. Check POI count and placement
    assert len(meta.poi_positions) == config["num_pois"]
    for poi_pos in meta.poi_positions:
        assert poi_pos in meta.walkable_tiles
        assert poi_pos != start_pos

    # 4. Check dead ends and junctions exist
    assert len(meta.dead_ends) >= 1
    assert len(meta.junctions) >= 1


def test_map_generator_determinism():
    """Verify that using the same RNG seed generates identical map topology and coordinates."""
    config = get_env_config("easy")
    gen = ProceduralMapGenerator(config)

    rng1 = np.random.default_rng(12345)
    _, meta1 = gen.generate(config["size"], config["size"], rng1)

    rng2 = np.random.default_rng(12345)
    _, meta2 = gen.generate(config["size"], config["size"], rng2)

    assert meta1.agent_start_pos == meta2.agent_start_pos
    assert meta1.agent_start_dir == meta2.agent_start_dir
    assert meta1.walkable_tiles == meta2.walkable_tiles
    assert meta1.poi_positions == meta2.poi_positions
    assert meta1.dead_ends == meta2.dead_ends


def test_map_generator_different_seeds():
    """Verify that different seeds produce distinct layouts."""
    config = get_env_config("easy")
    gen = ProceduralMapGenerator(config)

    rng1 = np.random.default_rng(1)
    _, meta1 = gen.generate(config["size"], config["size"], rng1)

    rng2 = np.random.default_rng(2)
    _, meta2 = gen.generate(config["size"], config["size"], rng2)

    assert meta1.walkable_tiles != meta2.walkable_tiles or meta1.agent_start_pos != meta2.agent_start_pos
