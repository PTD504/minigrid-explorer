from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.is_absolute():
        path = CONFIGS_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_env_config(difficulty: str = "easy") -> dict[str, Any]:
    """Load environment configuration by difficulty level ('easy', 'medium', 'hard')."""
    return load_yaml(f"env/{difficulty}.yaml")


def get_reward_config(config_name: str = "default") -> dict[str, Any]:
    """Load reward configuration."""
    return load_yaml(f"reward/{config_name}.yaml")


def get_obs_config(config_name: str = "default") -> dict[str, Any]:
    """Load observation configuration."""
    return load_yaml(f"obs/{config_name}.yaml")


def get_seeds_config() -> dict[str, Any]:
    """Load train, val, and test seed split configurations."""
    return load_yaml("seeds.yaml")


def validate_config_invariants(env_config: dict[str, Any], reward_config: dict[str, Any]) -> None:
    """Validate critical domain invariants.
    
    1. Anti-suicide invariant: c_death > c_step * max_steps.
    """
    max_steps = env_config.get("max_steps", 250)
    c_step = reward_config.get("c_step", 0.002)
    c_death = reward_config.get("c_death", 2.0)
    assert c_death > c_step * max_steps, (
        f"Anti-suicide invariant violated: c_death ({c_death}) <= c_step * max_steps "
        f"({c_step} * {max_steps} = {c_step * max_steps})"
    )


def validate_seed_splits(seeds_config: dict[str, Any]) -> None:
    """Validate that train, val, and test seed ranges are mutually disjoint."""
    train_range = range(seeds_config["train"]["start"], seeds_config["train"]["end"])
    val_range = range(
        seeds_config["val"]["start"],
        seeds_config["val"]["start"] + seeds_config["val"]["count"],
    )
    test_range = range(
        seeds_config["test"]["start"],
        seeds_config["test"]["start"] + seeds_config["test"]["count"],
    )

    # Check intersection
    assert set(train_range).isdisjoint(set(val_range)), "Train and Val seeds overlap!"
    assert set(train_range).isdisjoint(set(test_range)), "Train and Test seeds overlap!"
    assert set(val_range).isdisjoint(set(test_range)), "Val and Test seeds overlap!"
