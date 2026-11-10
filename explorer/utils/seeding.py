from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def make_run_seeds(global_seed: int, seed_count: int) -> list[np.random.SeedSequence]:
    """Deterministically derive a sequence of child SeedSequences from a single global seed.

    Guarantees prefix invariance: make_run_seeds(g, 3) == make_run_seeds(g, 5)[:3].
    """
    if seed_count <= 0:
        raise ValueError(f"seed_count must be > 0, got {seed_count}")
    return list(np.random.SeedSequence(global_seed).spawn(seed_count))


def get_run_seed_int(ss: np.random.SeedSequence) -> int:
    """Extract a reproducible 32-bit unsigned integer state from a SeedSequence."""
    return int(ss.generate_state(1)[0])


def build_run_rngs(
    run_ss: np.random.SeedSequence,
    n_envs: int,
    resume_count: int = 0,
) -> tuple[int, list[np.random.SeedSequence]]:
    """Stateless generator for algorithmic seed and per-worker map SeedSequences.

    Uses explicit spawn_key indexing:
    - k=0: algorithm seed (for SB3, PyTorch, NumPy model logic)
    - k=1..n_envs: independent child SeedSequences for environment map samplers

    Calling multiple times with identical arguments returns identical sequences.
    """
    if n_envs <= 0:
        raise ValueError(f"n_envs must be > 0, got {n_envs}")
    if resume_count < 0:
        raise ValueError(f"resume_count must be >= 0, got {resume_count}")

    base_key = tuple(run_ss.spawn_key)

    algo_ss = np.random.SeedSequence(
        entropy=run_ss.entropy,
        spawn_key=base_key + (resume_count, 0),
    )
    algo_seed = int(algo_ss.generate_state(1)[0])

    map_sss = [
        np.random.SeedSequence(
            entropy=run_ss.entropy,
            spawn_key=base_key + (resume_count, k),
        )
        for k in range(1, n_envs + 1)
    ]

    return algo_seed, map_sss


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of run hyperparameters and environment specs."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def get_git_commit_hash() -> str | None:
    """Retrieve the current Git commit hash if in a Git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return res.stdout.strip()
    except Exception:
        return None


def atomic_save_json(path: Path | str, data: dict[str, Any]) -> None:
    """Write JSON data atomically via temporary file and replace."""
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_save_model(model: Any, path: Path | str) -> None:
    """Save an SB3 model checkpoint atomically via temporary zip file and replace."""
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    try:
        model.save(str(tmp_path))
        # Stable-Baselines3 appends .zip if not present
        real_tmp = tmp_path if tmp_path.exists() else tmp_path.with_suffix(".zip")
        real_target = (
            target_path if target_path.name.endswith(".zip") else target_path.with_suffix(".zip")
        )
        os.replace(real_tmp, real_target)
    finally:
        for p in [tmp_path, tmp_path.with_suffix(".zip")]:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

