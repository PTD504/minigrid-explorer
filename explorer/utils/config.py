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


def get_reward_config(
    config_name: str = "default",
    *,
    no_bump: bool = False,
    no_loop: bool = False,
    c_bump: float | None = None,
    c_loop: float | None = None,
) -> dict[str, Any]:
    """Load reward configuration with dynamic ablation overrides.

    Ablation profiles supported via config_name:
        - 'default': Baseline reward parameters from configs/reward/default.yaml
        - 'no_bump': Disables wall bump penalty (c_bump = 0.0)
        - 'no_loop': Disables loop state repetition penalty (c_loop = 0.0)
        - 'no_bump_no_loop' (or 'no_loop_no_bump'): Disables both c_bump and c_loop (0.0)
        - '<custom>': If configs/reward/<custom>.yaml exists, loads it.

    Keyword arguments (no_bump, no_loop, c_bump, c_loop) take highest precedence.
    """
    config_file = CONFIGS_DIR / f"reward/{config_name}.yaml"
    if config_file.exists():
        cfg = load_yaml(f"reward/{config_name}.yaml")
    else:
        cfg = load_yaml("reward/default.yaml")

    # Name-based profile resolution
    name_norm = config_name.lower().replace("-", "_")
    if name_norm in ("no_bump", "nobump"):
        cfg["c_bump"] = 0.0
    elif name_norm in ("no_loop", "noloop"):
        cfg["c_loop"] = 0.0
    elif name_norm in ("no_bump_no_loop", "no_loop_no_bump", "none", "no_bump_loop"):
        cfg["c_bump"] = 0.0
        cfg["c_loop"] = 0.0

    # Explicit flag overrides
    if no_bump:
        cfg["c_bump"] = 0.0
    if no_loop:
        cfg["c_loop"] = 0.0
    if c_bump is not None:
        cfg["c_bump"] = float(c_bump)
    if c_loop is not None:
        cfg["c_loop"] = float(c_loop)

    return cfg


def get_obs_config(config_name: str = "default") -> dict[str, Any]:
    """Load observation configuration."""
    return load_yaml(f"obs/{config_name}.yaml")


def get_seeds_config() -> dict[str, Any]:
    """Load train, val, test, and dev seed split configurations."""
    return load_yaml("seeds.yaml")


def get_split(split_name: str) -> range:
    """Get the range of seeds for a given split ('train', 'val', 'test', 'dev')."""
    seeds_cfg = get_seeds_config()
    validate_seed_splits(seeds_cfg)
    if split_name not in seeds_cfg:
        raise KeyError(
            f"Seed split '{split_name}' not found in seeds.yaml. Available: {list(seeds_cfg.keys())}"
        )
    cfg = seeds_cfg[split_name]
    return range(int(cfg["start"]), int(cfg["start"]) + int(cfg["count"]))


def get_seed_range(split: str = "dev") -> list[int]:
    """Get list of seeds for a given split ('train', 'val', 'test', 'dev')."""
    return list(get_split(split))


def get_train_config(config_name: str = "ppo") -> dict[str, Any]:
    """Load training configuration."""
    return load_yaml(f"train/{config_name}.yaml")


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


def validate_seed_splits(seeds_config: dict[str, Any] | None = None) -> None:
    """Validate that train, val, test, and dev seed ranges are strictly valid and mutually disjoint."""
    if seeds_config is None:
        seeds_config = get_seeds_config()

    expected_splits = ["train", "val", "test", "dev"]
    # Reject unknown top-level keys
    for k in seeds_config:
        if k not in expected_splits:
            raise ValueError(f"Unknown split '{k}' in seeds config. Allowed: {expected_splits}")

    # Ensure all required splits are present
    for k in expected_splits:
        if k not in seeds_config:
            raise ValueError(f"Missing required split '{k}' in seeds config.")

    ranges: list[tuple[str, range]] = []
    for key in expected_splits:
        cfg = seeds_config[key]
        if not isinstance(cfg, dict):
            raise ValueError(f"Split '{key}' configuration must be a dictionary.")

        # Reject legacy 'end' or unexpected keys
        if "end" in cfg:
            raise ValueError(f"Legacy key 'end' is no longer supported in split '{key}'. Use 'count' instead.")

        for field in cfg:
            if field not in ("start", "count"):
                raise ValueError(f"Unexpected field '{field}' in split '{key}'. Only 'start' and 'count' allowed.")

        if "start" not in cfg or "count" not in cfg:
            raise ValueError(f"Split '{key}' must contain both 'start' and 'count'.")

        start = int(cfg["start"])
        count = int(cfg["count"])
        if start < 0:
            raise ValueError(f"Split '{key}' start must be non-negative, got {start}.")
        if count <= 0:
            raise ValueError(f"Split '{key}' count must be > 0, got {count}.")

        ranges.append((key, range(start, start + count)))

    # Pairwise disjointness check
    for i in range(len(ranges)):
        name1, r1 = ranges[i]
        s1 = set(r1)
        for j in range(i + 1, len(ranges)):
            name2, r2 = ranges[j]
            if not s1.isdisjoint(set(r2)):
                raise ValueError(f"Seed split overlap detected between '{name1}' and '{name2}'!")


