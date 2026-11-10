"""Reinforcement Learning Training Pipeline for MiniGrid Explorer.

Stateless upfront multi-seed generation, SubprocVecEnv worker map sampling,
dual TensorBoard & W&B logging, atomic checkpoints, robust resume & fine-tuning.

NOTE: VecNormalize is NOT used in this pipeline. Observations are normalized
locally in wrappers (egocentric memory crop, step ratio, RGB scaling) and reward
bounds are fixed by design invariants.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from explorer.agents.policy import get_policy_kwargs
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper, TrainMapSampler
from explorer.utils.config import (
    get_env_config,
    get_obs_config,
    get_reward_config,
    get_seeds_config,
    get_split,
    get_train_config,
    validate_seed_splits,
)
from explorer.utils.seeding import (
    atomic_save_json,
    atomic_save_model,
    build_run_rngs,
    compute_config_hash,
    get_git_commit_hash,
    get_run_seed_int,
    make_run_seeds,
)

# Supported algorithm names
SUPPORTED_ALGOS: tuple[str, ...] = ("ppo", "recurrent_ppo")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule decaying from initial_value to 0."""

    def _schedule(progress_remaining: float) -> float:
        return float(progress_remaining * initial_value)

    return _schedule


def check_checkpoint_compatibility(
    checkpoint_path: Path | str,
    expected_algo: str,
) -> None:
    """Validate that an SB3 checkpoint exists and matches expected algorithm class.

    Used by --init-from to ensure fine-tuning target matches current network architecture.
    """
    chk = Path(checkpoint_path).resolve()
    if not chk.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {chk}")
    if not chk.is_file():
        raise ValueError(f"Checkpoint path is not a file: {chk}")

    expected_algo = expected_algo.lower().strip()
    with zipfile.ZipFile(chk, "r") as archive:
        if "data" not in archive.namelist():
            raise ValueError(f"Invalid SB3 checkpoint: missing 'data' in {chk}")
        data = json.loads(archive.read("data"))
        policy_class_info = data.get("policy_class", {})
        module_name = policy_class_info.get("__module__", "")

        if expected_algo == "ppo":
            if "recurrent" in module_name:
                raise ValueError(
                    f"Algorithm mismatch: config specifies 'ppo' but checkpoint {chk} "
                    f"is a RecurrentPPO model ({module_name})."
                )
        elif expected_algo == "recurrent_ppo":
            if "recurrent" not in module_name:
                raise ValueError(
                    f"Algorithm mismatch: config specifies 'recurrent_ppo' but checkpoint {chk} "
                    f"is not a recurrent model ({module_name})."
                )


class DungeonEvalCallback(BaseCallback):
    """Custom evaluation callback for MiniGrid Explorer.

    Evaluates the current policy on a fixed deterministic set of validation seeds,
    records detailed exploration and survival metrics, logs them to TensorBoard
    and Weights & Biases (if active), and saves best and latest model checkpoints atomically.

    Checkpoints are ranked by:
    1. Primary: mean revealed area coverage (mean_revealed_coverage).
    2. Secondary (tie-breaker): mean POI discovery rate (mean_poi_discovery_rate).
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        val_seeds: list[int] | None = None,
        best_model_save_path: str | Path | None = None,
        deterministic: bool = True,
        target_coverage: float = 0.95,
        verbose: int = 1,
        best_coverage: float = -1.0,
        best_poi_rate: float = -1.0,
        best_timestep: int = 0,
        run_state_file: Path | str | None = None,
        run_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.val_seeds = val_seeds
        self.best_model_save_path = (
            Path(best_model_save_path).resolve() if best_model_save_path else None
        )
        self.deterministic = deterministic
        self.target_coverage = target_coverage

        self.best_coverage = float(best_coverage)
        self.best_poi_rate = float(best_poi_rate)
        self.best_timestep = int(best_timestep)
        self.eval_history: list[dict[str, Any]] = []

        self.run_state_file = (
            Path(run_state_file).resolve() if run_state_file else None
        )
        self.run_state = run_state or {}

        val_range = get_split("val")
        if self.val_seeds is None:
            if len(val_range) < self.n_eval_episodes:
                raise ValueError(
                    f"n_eval_episodes ({self.n_eval_episodes}) exceeds available validation seeds ({len(val_range)})"
                )
            self.val_seeds = list(val_range)[: self.n_eval_episodes]
        else:
            if len(self.val_seeds) < self.n_eval_episodes:
                raise ValueError(
                    f"n_eval_episodes ({self.n_eval_episodes}) exceeds available validation seeds ({len(self.val_seeds)})"
                )

    def _init_callback(self) -> None:
        if self.best_model_save_path:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Run evaluation rollouts across validation seeds
        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        ep_revealed_cov: list[float] = []
        ep_visited_cov: list[float] = []
        ep_poi_rates: list[float] = []
        ep_deaths: list[int] = []
        ep_timeouts: list[int] = []
        ep_completions: list[int] = []

        for i in range(self.n_eval_episodes):
            seed = self.val_seeds[i]
            obs, info = self.eval_env.reset(seed=seed)
            done = False
            total_rew = 0.0
            steps = 0

            # State tracking supports both Recurrent policies and Standard policies
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

            while not done:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=self.deterministic,
                )
                episode_starts = np.zeros((1,), dtype=bool)

                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_rew += float(reward)
                steps += 1
                done = terminated or truncated

            # Record final episode telemetry
            ep_rewards.append(total_rew)
            ep_lengths.append(steps)
            rev_cov = float(info.get("revealed_coverage", 0.0))
            ep_revealed_cov.append(rev_cov)
            ep_visited_cov.append(float(info.get("visited_coverage", 0.0)))
            ep_poi_rates.append(float(info.get("poi_discovery_rate", 0.0)))

            end_reason = info.get("end_reason")
            ep_deaths.append(1 if end_reason == "death" else 0)
            ep_timeouts.append(1 if end_reason == "timeout" else 0)
            ep_completions.append(
                1
                if (rev_cov >= self.target_coverage or end_reason == "coverage")
                else 0
            )

        # Aggregate statistics
        mean_reward = float(np.mean(ep_rewards))
        mean_revealed_cov = float(np.mean(ep_revealed_cov))
        mean_visited_cov = float(np.mean(ep_visited_cov))
        mean_poi_rate = float(np.mean(ep_poi_rates))
        death_rate = float(np.mean(ep_deaths))
        timeout_rate = float(np.mean(ep_timeouts))
        completion_rate = float(np.mean(ep_completions))
        mean_ep_length = float(np.mean(ep_lengths))

        # 1. Log to SB3 Logger (TensorBoard)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_revealed_coverage", mean_revealed_cov)
        self.logger.record("eval/mean_visited_coverage", mean_visited_cov)
        self.logger.record("eval/mean_poi_discovery_rate", mean_poi_rate)
        self.logger.record("eval/death_rate", death_rate)
        self.logger.record("eval/timeout_rate", timeout_rate)
        self.logger.record("eval/completion_rate", completion_rate)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.dump(step=self.num_timesteps)

        # 2. Log to Weights & Biases (if active)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/mean_reward": mean_reward,
                        "eval/mean_revealed_coverage": mean_revealed_cov,
                        "eval/mean_visited_coverage": mean_visited_cov,
                        "eval/mean_poi_discovery_rate": mean_poi_rate,
                        "eval/death_rate": death_rate,
                        "eval/timeout_rate": timeout_rate,
                        "eval/completion_rate": completion_rate,
                        "eval/mean_ep_length": mean_ep_length,
                        "train/timesteps": self.num_timesteps,
                    }
                )
        except ImportError:
            pass

        metrics_record = {
            "timesteps": self.num_timesteps,
            "mean_reward": mean_reward,
            "mean_revealed_coverage": mean_revealed_cov,
            "mean_visited_coverage": mean_visited_cov,
            "mean_poi_discovery_rate": mean_poi_rate,
            "death_rate": death_rate,
            "timeout_rate": timeout_rate,
            "completion_rate": completion_rate,
            "mean_ep_length": mean_ep_length,
        }
        self.eval_history.append(metrics_record)

        if self.verbose > 0:
            print(
                f"[Eval @ {self.num_timesteps} steps] "
                f"Revealed: {mean_revealed_cov * 100:.1f}% | "
                f"POI: {mean_poi_rate * 100:.1f}% | "
                f"Death: {death_rate * 100:.1f}% | "
                f"Reward: {mean_reward:.3f}"
            )

        # 3. Model Checkpoint Selection (D1: primary coverage, secondary POI rate)
        is_better = mean_revealed_cov > self.best_coverage or (
            np.isclose(mean_revealed_cov, self.best_coverage, atol=1e-4)
            and mean_poi_rate > self.best_poi_rate
        )

        if is_better and self.best_model_save_path:
            self.best_coverage = mean_revealed_cov
            self.best_poi_rate = mean_poi_rate
            self.best_timestep = self.num_timesteps
            best_file = self.best_model_save_path / "best_model.zip"
            atomic_save_model(self.model, best_file)
            if self.verbose > 0:
                print(
                    f"--> New best model saved to {best_file} "
                    f"(Coverage: {mean_revealed_cov * 100:.2f}%, POI: {mean_poi_rate * 100:.2f}%)"
                )

        # 4. Save latest model checkpoint atomically
        if self.best_model_save_path:
            latest_file = self.best_model_save_path / "latest_model.zip"
            atomic_save_model(self.model, latest_file)

        # 5. Update run_state.json atomically
        if self.run_state_file:
            self.run_state.update(
                {
                    "current_timestep": self.num_timesteps,
                    "best_score": self.best_coverage,
                    "best_poi_rate": self.best_poi_rate,
                    "best_timestep": self.best_timestep,
                    "updated_at": datetime.now().isoformat(),
                }
            )
            atomic_save_json(self.run_state_file, self.run_state)

        return True


def make_env(
    difficulty: str = "easy",
    map_ss: np.random.SeedSequence | None = None,
    worker_idx: int = 0,
    log_dir: Path | str | None = None,
    resume_count: int = 0,
    is_train: bool = True,
    max_steps_override: int | None = None,
    use_memory: bool = True,
    crop_size: int = 15,
    reward_config_name: str = "default",
    env_config_override: dict[str, Any] | None = None,
    seed: int | None = None,
    rank: int = 0,
) -> Callable[[], gym.Env]:
    """Create a single wrapped MiniGrid Explorer environment instance."""

    def _init() -> gym.Env:
        cfg_override = dict(env_config_override or {})
        if max_steps_override is not None:
            cfg_override["max_steps"] = int(max_steps_override)

        env: gym.Env = ExploreEnv(
            difficulty=difficulty,
            config=cfg_override if cfg_override else None,
            reward_config=get_reward_config(reward_config_name),
        )

        if is_train and map_ss is not None:
            worker_log = (
                Path(log_dir).resolve() / f"worker_{worker_idx}.jsonl"
                if log_dir is not None
                else None
            )
            env = TrainMapSampler(
                env,
                seed_pool=get_split("train"),
                rng=np.random.default_rng(map_ss),
                log_file=worker_log,
                resume_count=resume_count,
            )
        elif seed is not None:
            # Backwards compatibility fallback if map_ss is not provided
            env.reset(seed=seed + rank)

        env = ObservationMemoryWrapper(
            env,
            crop_size=crop_size,
            use_memory=use_memory,
            include_step_ratio=True,
        )
        return env

    return _init


def make_vec_env_dungeon(
    difficulty: str = "easy",
    n_envs: int = 8,
    map_sss: list[np.random.SeedSequence] | None = None,
    log_dir: Path | str | None = None,
    resume_count: int = 0,
    max_steps_override: int | None = None,
    use_memory: bool = True,
    crop_size: int = 15,
    reward_config_name: str = "default",
    env_config_override: dict[str, Any] | None = None,
    seed: int | None = 42,
    vec_env_type: str = "subproc",
) -> VecEnv:
    """Build a vectorized parallel environment wrapped in VecMonitor.

    If map_sss is supplied, each worker uses an independent TrainMapSampler with its SeedSequence.
    Otherwise falls back to legacy seed + rank.
    """
    env_fns = [
        make_env(
            difficulty=difficulty,
            map_ss=map_sss[i] if map_sss is not None and i < len(map_sss) else None,
            worker_idx=i,
            log_dir=log_dir,
            resume_count=resume_count,
            is_train=True,
            max_steps_override=max_steps_override,
            use_memory=use_memory,
            crop_size=crop_size,
            reward_config_name=reward_config_name,
            env_config_override=env_config_override,
            seed=seed,
            rank=i,
        )
        for i in range(n_envs)
    ]

    vec_cls = (
        SubprocVecEnv if (vec_env_type == "subproc" and n_envs > 1) else DummyVecEnv
    )
    venv = vec_cls(env_fns)
    return VecMonitor(venv)


def train_one(
    run_ss: np.random.SeedSequence,
    run_idx: int,
    run_dir: Path,
    exp_dir: Path,
    algo: str = "ppo",
    difficulty: str = "easy",
    total_timesteps: int = 500000,
    n_envs: int = 8,
    use_memory: bool = True,
    features_dim: int = 256,
    learning_rate: float | None = None,
    lr_schedule_type: str | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    ent_coef: float | None = None,
    clip_range: float | None = None,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    save_freq: int | None = None,
    reward_config_name: str = "default",
    max_steps_override: int | None = None,
    resume: bool = False,
    restart: bool = False,
    init_from: str | Path | None = None,
    track: bool = False,
    wandb_project: str = "minigrid-explorer",
    wandb_entity: str | None = None,
    exp_name: str | None = None,
    device: str = "auto",
    verbose: int = 1,
    extra_callbacks: list[BaseCallback] | None = None,
) -> tuple[BaseAlgorithm | None, float, list[dict[str, Any]]]:
    """Train a single training seed run within an experiment."""
    algo_key = algo.lower().strip()
    if algo_key not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unsupported algorithm '{algo}'. Supported options: {list(SUPPORTED_ALGOS)}"
        )

    run_dir = Path(run_dir).resolve()
    exp_dir = Path(exp_dir).resolve()
    done_file = run_dir / "DONE"
    run_state_file = run_dir / "run_state.json"
    latest_model_file = run_dir / "latest_model.zip"

    # 1. State Directory Guards
    if done_file.exists():
        if verbose > 0:
            print(f"[Run {run_idx}] Already marked DONE ({done_file}). Skipping.")
        best_cov = -1.0
        if run_state_file.exists():
            try:
                with open(run_state_file, "r", encoding="utf-8") as f:
                    best_cov = float(json.load(f).get("best_score", -1.0))
            except Exception:
                pass
        return None, best_cov, []

    if run_dir.exists() and any(run_dir.iterdir()):
        if restart:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_dir = run_dir.with_name(f"{run_dir.name}.incomplete_{timestamp_str}")
            run_dir.rename(archived_dir)
            if verbose > 0:
                print(
                    f"[Run {run_idx}] Incomplete run directory archived to {archived_dir.name}. Restarting fresh."
                )
            run_dir.mkdir(parents=True, exist_ok=True)
        elif not resume:
            raise ValueError(
                f"Run directory '{run_dir}' exists and is incomplete. "
                f"Use --resume to continue or --restart to archive and restart."
            )

    # 2. Compatibility Check for --init-from
    if init_from is not None:
        check_checkpoint_compatibility(init_from, algo_key)

    # 3. Determine Resume Parameters
    resume_count = 0
    best_score = -1.0
    best_poi_rate = -1.0
    best_timestep = 0
    wandb_run_id: str | None = None
    is_resuming = False

    if resume:
        if not latest_model_file.exists() and not run_state_file.exists():
            if verbose > 0:
                print(
                    f"[Run {run_idx}] --resume specified but no checkpoint found in {run_dir}. "
                    f"Training fresh from step 0."
                )
            is_resuming = False
        else:
            if not latest_model_file.exists():
                raise FileNotFoundError(
                    f"Cannot resume run {run_idx}: missing latest checkpoint {latest_model_file}"
                )
            is_resuming = True
            if run_state_file.exists():
                with open(run_state_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                best_score = float(st.get("best_score", -1.0))
                best_poi_rate = float(st.get("best_poi_rate", -1.0))
                best_timestep = int(st.get("best_timestep", 0))
                resume_count = int(st.get("resume_count", 0)) + 1
                wandb_run_id = st.get("wandb_run_id")

    # 4. Stateless Seed Derivation
    algo_seed, map_sss = build_run_rngs(run_ss, n_envs, resume_count=resume_count)
    set_random_seed(algo_seed, using_cuda=torch.cuda.is_available())

    # 5. Load Hyperparameters
    train_cfg = get_train_config(algo_key)
    obs_cfg = get_obs_config()
    env_cfg = get_env_config(difficulty)

    crop_size = int(obs_cfg.get("crop_size", 15))
    lr = (
        learning_rate
        if learning_rate is not None
        else float(train_cfg.get("learning_rate", 3e-4))
    )
    sched = (
        lr_schedule_type
        if lr_schedule_type is not None
        else str(train_cfg.get("lr_schedule", "linear"))
    )
    steps_per_env = (
        n_steps if n_steps is not None else int(train_cfg.get("n_steps", 128))
    )
    m_batch = (
        batch_size if batch_size is not None else int(train_cfg.get("batch_size", 64))
    )
    epochs = n_epochs if n_epochs is not None else int(train_cfg.get("n_epochs", 4))
    entropy_c = (
        ent_coef if ent_coef is not None else float(train_cfg.get("ent_coef", 0.015))
    )
    clipping = (
        clip_range
        if clip_range is not None
        else float(train_cfg.get("clip_range", 0.2))
    )
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
    eval_step_freq = (
        eval_freq if eval_freq is not None else int(train_cfg.get("eval_freq", 10000))
    )
    eval_eps = (
        n_eval_episodes
        if n_eval_episodes is not None
        else int(train_cfg.get("n_eval_episodes", 20))
    )

    # 6. Setup Logging and Directories
    run_dir.mkdir(parents=True, exist_ok=True)
    map_seeds_dir = run_dir / "map_seeds"
    map_seeds_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = exp_dir / "tb_logs"
    tb_run_name = run_dir.name

    full_config = {
        "algo": algo_key,
        "difficulty": difficulty,
        "run_idx": run_idx,
        "algo_seed": algo_seed,
        "total_timesteps": total_timesteps,
        "use_memory": use_memory,
        "crop_size": crop_size,
        "features_dim": features_dim,
        "learning_rate": lr,
        "lr_schedule": sched,
        "n_envs": n_envs,
        "n_steps": steps_per_env,
        "batch_size": m_batch,
        "n_epochs": epochs,
        "ent_coef": entropy_c,
        "clip_range": clipping,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "target_coverage": float(env_cfg.get("target_coverage", 0.95)),
        "reward_config": reward_config_name,
    }
    cfg_hash = compute_config_hash(full_config)

    # Handle Weights & Biases
    if track:
        import wandb

        if os.environ.get("WANDB_MODE") == "disabled":
            del os.environ["WANDB_MODE"]

        w_name = f"{exp_name or exp_dir.name}_{run_dir.name}"
        wandb_init_kwargs: dict[str, Any] = {
            "project": wandb_project,
            "entity": wandb_entity,
            "name": w_name,
            "group": exp_dir.name,
            "tags": [algo_key, difficulty, f"run_{run_idx}", f"algo_seed_{algo_seed}"],
            "config": full_config,
            "sync_tensorboard": True,
        }
        if wandb_run_id is not None:
            wandb_init_kwargs["id"] = wandb_run_id
            wandb_init_kwargs["resume"] = "allow"

        wandb.init(**wandb_init_kwargs)
        if wandb.run is not None:
            wandb_run_id = wandb.run.id
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Immediately save initial run_state.json with wandb_run_id
    run_state = {
        "run_idx": run_idx,
        "algo_seed": algo_seed,
        "status": "running",
        "current_timestep": 0 if not is_resuming else None,
        "total_timesteps": total_timesteps,
        "best_score": best_score,
        "best_poi_rate": best_poi_rate,
        "best_timestep": best_timestep,
        "resume_count": resume_count,
        "wandb_run_id": wandb_run_id,
        "config_hash": cfg_hash,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    atomic_save_json(run_state_file, run_state)

    # 7. Create Parallel Training and Validation Environments
    train_vec_env = make_vec_env_dungeon(
        difficulty=difficulty,
        n_envs=n_envs,
        map_sss=map_sss,
        log_dir=map_seeds_dir,
        resume_count=resume_count,
        max_steps_override=max_steps_override,
        use_memory=use_memory,
        crop_size=crop_size,
        reward_config_name=reward_config_name,
    )

    eval_single_env = make_env(
        difficulty=difficulty,
        is_train=False,
        max_steps_override=max_steps_override,
        use_memory=use_memory,
        crop_size=crop_size,
        reward_config_name=reward_config_name,
    )()

    # 8. Setup Evaluation Callback
    eval_callback_freq = max(eval_step_freq // n_envs, 1)
    eval_callback = DungeonEvalCallback(
        eval_env=eval_single_env,
        eval_freq=eval_callback_freq,
        n_eval_episodes=eval_eps,
        best_model_save_path=run_dir,
        target_coverage=float(env_cfg.get("target_coverage", 0.95)),
        verbose=verbose,
        best_coverage=best_score,
        best_poi_rate=best_poi_rate,
        best_timestep=best_timestep,
        run_state_file=run_state_file,
        run_state=run_state,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    save_step_freq = save_freq if save_freq is not None else train_cfg.get("save_freq")
    if save_step_freq is not None and int(save_step_freq) > 0:
        save_cb_freq = max(int(save_step_freq) // n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_cb_freq,
            save_path=str(run_dir),
            name_prefix="model_step",
            verbose=0,
        )
        callbacks.append(checkpoint_callback)

    # 9. Build or Load Policy
    lr_fn: Callable[[float], float] | float = (
        linear_schedule(lr) if sched == "linear" else lr
    )

    algo_cls = RecurrentPPO if algo_key == "recurrent_ppo" else PPO
    model: BaseAlgorithm

    if is_resuming:
        if verbose > 0:
            print(f"[Run {run_idx}] Resuming from checkpoint: {latest_model_file}")
        model = algo_cls.load(
            str(latest_model_file),
            env=train_vec_env,
            device=device,
            tensorboard_log=str(tb_log_dir),
            verbose=verbose,
        )
        remaining_timesteps = total_timesteps - model.num_timesteps
        if verbose > 0:
            print(
                f"[Run {run_idx}] Current step: {model.num_timesteps}/{total_timesteps}. "
                f"Remaining to learn: {remaining_timesteps}"
            )
    elif init_from is not None:
        if verbose > 0:
            print(f"[Run {run_idx}] Initializing model weights from: {init_from}")
        model = algo_cls.load(
            str(init_from),
            env=train_vec_env,
            device=device,
            tensorboard_log=str(tb_log_dir),
            verbose=verbose,
        )
        model.num_timesteps = 0
        model._num_timesteps_at_start = 0
        remaining_timesteps = total_timesteps
    else:
        policy_kwargs = get_policy_kwargs(
            features_dim=features_dim,
            use_memory=use_memory,
        )
        if algo_key == "recurrent_ppo":
            policy_kwargs.update(
                {
                    "lstm_hidden_size": int(train_cfg.get("lstm_hidden_size", 256)),
                    "n_lstm_layers": int(train_cfg.get("n_lstm_layers", 1)),
                    "shared_lstm": bool(train_cfg.get("shared_lstm", False)),
                    "enable_critic_lstm": bool(train_cfg.get("enable_critic_lstm", True)),
                }
            )

        common_model_kwargs: dict[str, Any] = {
            "env": train_vec_env,
            "learning_rate": lr_fn,
            "n_steps": steps_per_env,
            "batch_size": m_batch,
            "n_epochs": epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clipping,
            "ent_coef": entropy_c,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": policy_kwargs,
            "tensorboard_log": str(tb_log_dir),
            "verbose": verbose,
            "device": device,
            "seed": algo_seed,
        }

        if algo_key == "ppo":
            model = PPO(policy="MultiInputPolicy", **common_model_kwargs)
        else:
            model = RecurrentPPO(policy="MultiInputLstmPolicy", **common_model_kwargs)
        remaining_timesteps = total_timesteps

    # 10. Run Training Loop
    try:
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                reset_num_timesteps=not is_resuming,
                tb_log_name=tb_run_name,
            )
        else:
            if verbose > 0:
                print(f"[Run {run_idx}] Timestep budget already fulfilled.")

        # Save final model and latest model atomically
        final_file = run_dir / "final_model.zip"
        latest_file = run_dir / "latest_model.zip"
        atomic_save_model(model, final_file)
        atomic_save_model(model, latest_file)

        # Update run state
        run_state["status"] = "completed"
        run_state["current_timestep"] = model.num_timesteps
        run_state["best_score"] = eval_callback.best_coverage
        run_state["best_poi_rate"] = eval_callback.best_poi_rate
        run_state["best_timestep"] = eval_callback.best_timestep
        run_state["updated_at"] = datetime.now().isoformat()
        atomic_save_json(run_state_file, run_state)

        # Touch DONE file ONLY after all checkpoints are saved
        done_file.touch()

        if verbose > 0:
            print(f"[Run {run_idx}] Completed. Checkpoint saved to {final_file}")
    finally:
        train_vec_env.close()
        eval_single_env.close()

        if track:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, eval_callback.best_coverage, eval_callback.eval_history


def aggregate_val_results(exp_dir: Path | str) -> dict[str, Any]:
    """Aggregate validation metrics across all runs in an experiment directory.

    Produces a summary JSON file and displays a terminal table clearly marked:
    [VAL ONLY - QUICK INSPECTION].
    """
    exp_dir = Path(exp_dir).resolve()
    val_summary_file = exp_dir / "val_summary.json"

    run_dirs = sorted(
        [
            d
            for d in exp_dir.iterdir()
            if d.is_dir() and (d.name.startswith("seed_") or d.name.startswith("run_"))
        ]
    )
    run_results: list[dict[str, Any]] = []

    for d in run_dirs:
        state_file = d / "run_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                run_results.append(
                    {
                        "run": d.name,
                        "status": st.get("status", "unknown"),
                        "best_score": float(st.get("best_score", -1.0)),
                        "best_poi_rate": float(st.get("best_poi_rate", -1.0)),
                        "best_timestep": int(st.get("best_timestep", 0)),
                        "current_timestep": int(st.get("current_timestep", 0)),
                    }
                )
            except Exception:
                pass

    scores = [r["best_score"] for r in run_results if r["best_score"] >= 0]
    pois = [r["best_poi_rate"] for r in run_results if r["best_poi_rate"] >= 0]

    cov_mean = float(np.mean(scores)) if scores else 0.0
    cov_std = float(np.std(scores)) if scores else 0.0
    poi_mean = float(np.mean(pois)) if pois else 0.0
    poi_std = float(np.std(pois)) if pois else 0.0

    valid_runs = [r for r in run_results if r["best_score"] >= 0]
    top_run = (
        max(valid_runs, key=lambda r: (r["best_score"], r["best_poi_rate"]))
        if valid_runs
        else None
    )

    summary_data = {
        "experiment": exp_dir.name,
        "num_runs": len(run_results),
        "coverage_mean": cov_mean,
        "coverage_std": cov_std,
        "poi_mean": poi_mean,
        "poi_std": poi_std,
        "top_run": top_run,
        "runs": run_results,
        "updated_at": datetime.now().isoformat(),
    }
    atomic_save_json(val_summary_file, summary_data)

    print("\n" + "=" * 86)
    print("                    [VAL ONLY - QUICK INSPECTION] Validation Summary")
    print(f" Experiment: {exp_dir.name} | Total Runs: {len(run_results)}")
    print("=" * 86)
    print(
        f" {'Run':<23} | {'Status':<11} | {'Best Coverage':<15} | {'Best POI Rate':<15} | {'Best @ Step':<12}"
    )
    print("-" * 86)
    for r in run_results:
        cov_str = f"{r['best_score'] * 100:.2f}%" if r["best_score"] >= 0 else "N/A"
        poi_str = f"{r['best_poi_rate'] * 100:.2f}%" if r["best_poi_rate"] >= 0 else "N/A"
        print(
            f" {r['run']:<23} | {r['status']:<11} | {cov_str:<15} | {poi_str:<15} | {r['best_timestep']:<12}"
        )
    print("-" * 86)
    print(
        f" Mean ± Std: Coverage = {cov_mean * 100:.2f}% ± {cov_std * 100:.2f}% | "
        f"POI Discovery = {poi_mean * 100:.2f}% ± {poi_std * 100:.2f}%"
    )
    if top_run:
        print(
            f" Top Run   : {top_run['run']} (Coverage: {top_run['best_score'] * 100:.2f}%, "
            f"POI: {top_run['best_poi_rate'] * 100:.2f}% @ step {top_run['best_timestep']})"
        )
    print("=" * 86 + "\n")

    return summary_data


def train_multirun(
    algo: str = "ppo",
    difficulty: str = "easy",
    seed: int = 2026,
    seed_count: int = 1,
    total_timesteps: int = 500000,
    n_envs: int | None = None,
    use_memory: bool | None = None,
    features_dim: int = 256,
    learning_rate: float | None = None,
    lr_schedule_type: str | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    ent_coef: float | None = None,
    clip_range: float | None = None,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    save_freq: int | None = None,
    reward_config_name: str = "default",
    max_steps_override: int | None = None,
    resume: bool = False,
    restart: bool = False,
    init_from: str | Path | None = None,
    track: bool = False,
    wandb_project: str = "minigrid-explorer",
    wandb_entity: str | None = None,
    exp_name: str | None = None,
    output_dir: str | Path = "results",
    device: str = "auto",
    verbose: int = 1,
    extra_callbacks: list[BaseCallback] | None = None,
) -> tuple[BaseAlgorithm | None, float, list[dict[str, Any]]]:
    """Execute sequential multi-run training upfront derived from a single global seed."""
    # 1. Enforce strict split disjointness and absence of invalid keys
    validate_seed_splits()

    if resume and init_from:
        raise ValueError("--resume and --init-from are mutually exclusive.")
    if resume and restart:
        raise ValueError("--resume and --restart are mutually exclusive.")
    if restart and init_from:
        raise ValueError("--restart and --init-from are mutually exclusive.")

    algo_key = algo.lower().strip()
    if algo_key not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unsupported algorithm '{algo}'. Supported options: {list(SUPPORTED_ALGOS)}"
        )

    # Resolve default memory mode:
    # - PPO: True (explicit 15x15 spatial memory)
    # - RecurrentPPO: False (implicit LSTM memory, view-only)
    effective_use_memory = (
        (algo_key != "recurrent_ppo") if use_memory is None else bool(use_memory)
    )

    train_cfg = get_train_config(algo_key)
    num_envs = n_envs if n_envs is not None else int(train_cfg.get("n_envs", 8))

    # 2. Derive run seeds upfront
    run_seeds = make_run_seeds(seed, seed_count)

    # 3. Setup experiment directory
    output_root = Path(output_dir).resolve()
    reward_tag = f"-{reward_config_name}" if reward_config_name != "default" else ""
    mem_tag = "full" if effective_use_memory else "nomem"

    if exp_name:
        experiment_name = exp_name
    else:
        experiment_name = f"{algo_key}-{difficulty}-{mem_tag}{reward_tag}_seed{seed}"

    exp_dir = output_root / "checkpoints" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 4. Manifest management with prefix and parameter checks
    manifest_file = exp_dir / "run_manifest.json"
    manifest_seeds = [get_run_seed_int(ss) for ss in run_seeds]

    config_for_hash = {
        "algo": algo_key,
        "difficulty": difficulty,
        "total_timesteps": total_timesteps,
        "use_memory": effective_use_memory,
        "features_dim": features_dim,
        "reward_config": reward_config_name,
        "n_envs": num_envs,
    }
    cfg_hash = compute_config_hash(config_for_hash)

    current_manifest = {
        "global_seed": seed,
        "seed_count": seed_count,
        "seeds": manifest_seeds,
        "difficulty": difficulty,
        "algo": algo_key,
        "use_memory": effective_use_memory,
        "reward_config": reward_config_name,
        "n_envs": num_envs,
        "config_hash": cfg_hash,
        "git_commit": get_git_commit_hash(),
        "created_at": datetime.now().isoformat(),
    }

    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            existing_manifest = json.load(f)

        # Compare hyperparameters (strictly excluding seed_count)
        check_keys = [
            "global_seed",
            "difficulty",
            "algo",
            "use_memory",
            "reward_config",
            "n_envs",
            "config_hash",
        ]
        for k in check_keys:
            if existing_manifest.get(k) != current_manifest.get(k):
                raise ValueError(
                    f"Run manifest mismatch for '{k}': existing={existing_manifest.get(k)}, "
                    f"current={current_manifest.get(k)} in {manifest_file}"
                )

        existing_seeds = existing_manifest.get("seeds", [])
        if len(existing_seeds) > len(manifest_seeds):
            raise ValueError(
                f"Existing run manifest has {len(existing_seeds)} seeds, cannot decrease to {len(manifest_seeds)}."
            )
        if manifest_seeds[: len(existing_seeds)] != existing_seeds:
            raise ValueError(
                f"Existing seeds in manifest {existing_seeds} do not match prefix of new seeds {manifest_seeds}."
            )

        # Allow expanding seed_count
        if len(manifest_seeds) > len(existing_seeds):
            existing_manifest["seed_count"] = seed_count
            existing_manifest["seeds"] = manifest_seeds
            existing_manifest["updated_at"] = datetime.now().isoformat()
            atomic_save_json(manifest_file, existing_manifest)
    else:
        atomic_save_json(manifest_file, current_manifest)

    # 5. Sequentially execute runs
    last_model = None
    last_coverage = -1.0
    last_history: list[dict[str, Any]] = []

    for run_idx in range(seed_count):
        # Free previous model and cached tensors before initializing new run
        last_model = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        run_ss = run_seeds[run_idx]
        base_algo_seed, _ = build_run_rngs(run_ss, num_envs, resume_count=0)
        default_run_dir = exp_dir / f"run_{run_idx}_seed_{base_algo_seed}"
        legacy_run_dir = exp_dir / f"seed_{run_idx}"
        run_dir = legacy_run_dir if legacy_run_dir.exists() else default_run_dir

        if verbose > 0:
            print(f"\n>>> Starting Run {run_idx + 1}/{seed_count} ({run_dir.name}) in {run_dir} <<<")

        model, cov, history = train_one(
            run_ss=run_ss,
            run_idx=run_idx,
            run_dir=run_dir,
            exp_dir=exp_dir,
            algo=algo_key,
            difficulty=difficulty,
            total_timesteps=total_timesteps,
            n_envs=num_envs,
            use_memory=effective_use_memory,
            features_dim=features_dim,
            learning_rate=learning_rate,
            lr_schedule_type=lr_schedule_type,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            clip_range=clip_range,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_freq=save_freq,
            reward_config_name=reward_config_name,
            max_steps_override=max_steps_override,
            resume=resume,
            restart=restart,
            init_from=init_from,
            track=track,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            exp_name=exp_name,
            device=device,
            verbose=verbose,
        )
        if model is not None:
            last_model = model
            last_coverage = cov
            last_history = history

    # 6. Aggregate validation results across runs
    summary_data = aggregate_val_results(exp_dir)

    # 7. Select top run across ALL seeds and copy its best_model.zip to experiment root
    run_subdirs = sorted(
        [
            d
            for d in exp_dir.iterdir()
            if d.is_dir() and (d.name.startswith("seed_") or d.name.startswith("run_"))
        ]
    )
    top_run = summary_data.get("top_run")
    if top_run:
        top_best_file = exp_dir / top_run["run"] / "best_model.zip"
        if top_best_file.exists():
            shutil.copy2(top_best_file, exp_dir / "best_model.zip")
            if verbose > 0:
                print(
                    f"--> Experiment best_model.zip updated from top performer: "
                    f"{top_run['run']} ({top_run['best_score'] * 100:.2f}%)"
                )
    elif run_subdirs:
        first_best = run_subdirs[0] / "best_model.zip"
        if first_best.exists():
            shutil.copy2(first_best, exp_dir / "best_model.zip")

    # Copy final_model from the last executed run
    if run_subdirs:
        last_final = run_subdirs[-1] / "final_model.zip"
        if last_final.exists():
            shutil.copy2(last_final, exp_dir / "final_model.zip")

    mean_cov = float(summary_data.get("coverage_mean", last_coverage))
    return last_model, mean_cov, last_history


# Backwards compatibility aliases
def train(
    *args: Any,
    seed_count: int = 1,
    **kwargs: Any,
) -> tuple[BaseAlgorithm | None, float, list[dict[str, Any]]]:
    """Compatibility wrapper calling train_multirun."""
    return train_multirun(*args, seed_count=seed_count, **kwargs)


def train_ppo(
    *args: Any,
    **kwargs: Any,
) -> tuple[BaseAlgorithm | None, float, list[dict[str, Any]]]:
    """Convenience alias for PPO training."""
    kwargs.setdefault("algo", "ppo")
    return train_multirun(*args, **kwargs)
