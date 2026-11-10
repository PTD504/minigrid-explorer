import pytest

from explorer.utils.config import (
    get_env_config,
    get_obs_config,
    get_reward_config,
    get_seed_range,
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
    """Ensure train, val, test, and dev seed ranges are strictly disjoint."""
    seeds_cfg = get_seeds_config()
    assert "dev" in seeds_cfg
    assert seeds_cfg["dev"]["start"] == 3_000_000
    assert seeds_cfg["dev"]["count"] == 100
    validate_seed_splits(seeds_cfg)

    dev_seeds = get_seed_range("dev")
    assert len(dev_seeds) == 100
    assert dev_seeds[0] == 3_000_000
    assert dev_seeds[-1] == 3_000_099


def test_seed_splits_dev_overlap_detected():
    """Ensure validate_seed_splits raises ValueError if dev overlaps with other splits."""
    overlapping_cfg = {
        "train": {"start": 0, "count": 1000},
        "val": {"start": 1000, "count": 50},
        "test": {"start": 2000, "count": 50},
        "dev": {"start": 1020, "count": 50},  # Overlaps with val (1000..1050)
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_splits(overlapping_cfg)


def test_seed_splits_rejects_legacy_end():
    """Ensure validate_seed_splits raises ValueError on legacy 'end' key."""
    legacy_cfg = {
        "train": {"start": 0, "end": 1000},
        "val": {"start": 1000, "count": 50},
        "test": {"start": 2000, "count": 50},
        "dev": {"start": 3000, "count": 50},
    }
    with pytest.raises(ValueError, match="Legacy key 'end'"):
        validate_seed_splits(legacy_cfg)


def test_seed_splits_rejects_unknown_or_missing_keys():
    """Ensure validate_seed_splits catches missing required splits or extra keys."""
    missing_cfg = {
        "train": {"start": 0, "count": 1000},
        "val": {"start": 1000, "count": 50},
    }
    with pytest.raises(ValueError, match="Missing required split"):
        validate_seed_splits(missing_cfg)

    extra_cfg = {
        "train": {"start": 0, "count": 1000},
        "val": {"start": 1000, "count": 50},
        "test": {"start": 2000, "count": 50},
        "dev": {"start": 3000, "count": 50},
        "unknown": {"start": 4000, "count": 10},
    }
    with pytest.raises(ValueError, match="Unknown split"):
        validate_seed_splits(extra_cfg)




def test_anti_suicide_invariant_all_difficulties():
    """Verify that penalty for death is strictly greater than cumulative step penalty."""
    reward_cfg = get_reward_config()
    for diff in ["easy", "medium", "hard"]:
        env_cfg = get_env_config(diff)
        validate_config_invariants(env_cfg, reward_cfg)


def test_reward_config_ablation_profiles():
    """Verify reward config loads default parameters and dynamic ablation overrides."""
    # Default baseline
    cfg_default = get_reward_config("default")
    assert cfg_default["c_bump"] == 0.02
    assert cfg_default["c_loop"] == 0.01
    assert cfg_default["c_step"] == 0.002
    assert cfg_default["c_death"] == 2.0

    # Named profile: no_bump
    cfg_no_bump = get_reward_config("no_bump")
    assert cfg_no_bump["c_bump"] == 0.0
    assert cfg_no_bump["c_loop"] == 0.01

    # Named profile: no_loop
    cfg_no_loop = get_reward_config("no_loop")
    assert cfg_no_loop["c_bump"] == 0.02
    assert cfg_no_loop["c_loop"] == 0.0

    # Named profile: both (no_bump_no_loop & no_loop_no_bump)
    cfg_both1 = get_reward_config("no_bump_no_loop")
    assert cfg_both1["c_bump"] == 0.0
    assert cfg_both1["c_loop"] == 0.0

    cfg_both2 = get_reward_config("no_loop_no_bump")
    assert cfg_both2["c_bump"] == 0.0
    assert cfg_both2["c_loop"] == 0.0

    # Keyword argument overrides
    cfg_kw_bump = get_reward_config(no_bump=True)
    assert cfg_kw_bump["c_bump"] == 0.0
    assert cfg_kw_bump["c_loop"] == 0.01

    cfg_kw_loop = get_reward_config(no_loop=True)
    assert cfg_kw_loop["c_bump"] == 0.02
    assert cfg_kw_loop["c_loop"] == 0.0

    cfg_kw_both = get_reward_config(no_bump=True, no_loop=True)
    assert cfg_kw_both["c_bump"] == 0.0
    assert cfg_kw_both["c_loop"] == 0.0

    # Numeric overrides
    cfg_custom = get_reward_config(c_bump=0.05, c_loop=0.03)
    assert cfg_custom["c_bump"] == 0.05
    assert cfg_custom["c_loop"] == 0.03


def test_train_config_loaded():
    """Verify default ppo train config loads expected fields."""
    train_cfg = get_train_config("ppo")
    assert train_cfg.get("algo") == "ppo"
    assert "learning_rate" in train_cfg
    assert "n_steps" in train_cfg
    assert "batch_size" in train_cfg
    assert "n_envs" in train_cfg
