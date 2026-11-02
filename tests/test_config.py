from explorer.utils.config import (
    get_env_config,
    get_obs_config,
    get_reward_config,
    get_seeds_config,
    get_train_config,
    validate_config_invariants,
    validate_seed_splits,
)


def test_obs_config_loaded():
    """Verify default observation config loads correct values."""
    obs_cfg = get_obs_config()
    assert obs_cfg.get("crop_size") == 15
    assert obs_cfg.get("use_memory") is True
    assert obs_cfg.get("include_step_ratio") is True


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


def test_train_config_loaded():
    """Verify default ppo train config loads expected fields."""
    train_cfg = get_train_config("ppo")
    assert train_cfg.get("algo") == "ppo"
    assert "learning_rate" in train_cfg
    assert "n_steps" in train_cfg
    assert "batch_size" in train_cfg
    assert "n_envs" in train_cfg
