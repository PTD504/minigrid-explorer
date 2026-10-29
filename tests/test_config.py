from explorer.utils.config import (
    get_env_config,
    get_reward_config,
    get_seeds_config,
    validate_config_invariants,
    validate_seed_splits,
)


def test_seed_splits_disjoint():
    """Ensure train, val, and test seed ranges are strictly disjoint."""
    seeds_cfg = get_seeds_config()
    validate_seed_splits(seeds_cfg)


def test_anti_suicide_invariant_all_difficulties():
    """Verify that penalty for death is strictly greater than cumulative step penalty."""
    reward_cfg = get_reward_config()
    for diff in ["easy", "medium", "hard"]:
        env_cfg = get_env_config(diff)
        validate_config_invariants(env_cfg, reward_cfg)
