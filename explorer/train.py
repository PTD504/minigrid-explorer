from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from explorer.agents.policy import get_policy_kwargs
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper
from explorer.utils.config import (
    get_env_config,
    get_obs_config,
    get_reward_config,
    get_seeds_config,
    get_train_config,
)

# Supported algorithm names
SUPPORTED_ALGOS: tuple[str, ...] = ("ppo", "recurrent_ppo")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule decaying from initial_value to 0."""

    def _schedule(progress_remaining: float) -> float:
        return float(progress_remaining * initial_value)

    return _schedule


class DungeonEvalCallback(BaseCallback):
    """Custom evaluation callback for MiniGrid Explorer.

    Evaluates the current policy on a fixed deterministic set of validation seeds,
    records detailed exploration and survival metrics, logs them to TensorBoard
    and Weights & Biases (if active), and saves the best model checkpoint.

    Compatible with both feedforward policies (PPO) and recurrent policies (RecurrentPPO).
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
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.val_seeds = val_seeds
        self.best_model_save_path = (
            Path(best_model_save_path) if best_model_save_path else None
        )
        self.deterministic = deterministic
        self.target_coverage = target_coverage

        self.best_coverage: float = -1.0
        self.best_poi_rate: float = -1.0
        self.eval_history: list[dict[str, Any]] = []

    def _init_callback(self) -> None:
        if self.best_model_save_path:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)

        if self.val_seeds is None:
            # Fall back to val seed split from seeds.yaml
            seeds_cfg = get_seeds_config()
            val_start = seeds_cfg.get("val", {}).get("start", 1000000)
            self.val_seeds = [val_start + i for i in range(self.n_eval_episodes)]
        elif len(self.val_seeds) < self.n_eval_episodes:
            self.n_eval_episodes = len(self.val_seeds)

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
            seed = self.val_seeds[i] if self.val_seeds else (1000000 + i)
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
            best_file = self.best_model_save_path / "best_model.zip"
            self.model.save(str(best_file))
            if self.verbose > 0:
                print(
                    f"--> New best model saved to {best_file} "
                    f"(Coverage: {mean_revealed_cov * 100:.2f}%, POI: {mean_poi_rate * 100:.2f}%)"
                )

        return True


def make_env(
    difficulty: str = "easy",
    seed: int | None = None,
    use_memory: bool = True,
    crop_size: int = 15,
    reward_config_name: str = "default",
    env_config_override: dict[str, Any] | None = None,
    rank: int = 0,
) -> Callable[[], gym.Env]:
    """Create a single wrapped MiniGrid Explorer environment instance."""

    def _init() -> gym.Env:
        env = ExploreEnv(
            difficulty=difficulty,
            config=env_config_override,
            reward_config=get_reward_config(reward_config_name),
        )
        env = ObservationMemoryWrapper(
            env,
            crop_size=crop_size,
            use_memory=use_memory,
            include_step_ratio=True,
        )
        if seed is not None:
            env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_env_dungeon(
    difficulty: str = "easy",
    n_envs: int = 8,
    seed: int = 42,
    use_memory: bool = True,
    crop_size: int = 15,
    reward_config_name: str = "default",
    env_config_override: dict[str, Any] | None = None,
    vec_env_type: str = "dummy",
) -> VecEnv:
    """Build a vectorized parallel environment wrapped in VecMonitor."""
    env_fns = [
        make_env(
            difficulty=difficulty,
            seed=seed,
            use_memory=use_memory,
            crop_size=crop_size,
            reward_config_name=reward_config_name,
            env_config_override=env_config_override,
            rank=i,
        )
        for i in range(n_envs)
    ]

    vec_cls = (
        SubprocVecEnv if (vec_env_type == "subproc" and n_envs > 1) else DummyVecEnv
    )
    venv = vec_cls(env_fns)
    return VecMonitor(venv)


def train(
    algo: str = "ppo",
    difficulty: str = "easy",
    seed: int = 42,
    total_timesteps: int = 500000,
    n_envs: int | None = None,
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
    track: bool = False,
    wandb_project: str = "minigrid-explorer",
    wandb_entity: str | None = None,
    exp_name: str | None = None,
    output_dir: str | Path = "results",
    device: str = "auto",
    verbose: int = 1,
) -> tuple[BaseAlgorithm, float, list[dict[str, Any]]]:
    """Train a reinforcement learning agent with custom extractor and dual TB/W&B logging.

    Supports algorithms registered in ALGO_REGISTRY: 'ppo', 'recurrent_ppo'.

    Args:
        algo: RL algorithm ('ppo' or 'recurrent_ppo').
        difficulty: Dungeon difficulty ('easy', 'medium', 'hard').
        seed: Random seed for training environments.
        total_timesteps: Total environment interaction steps.
        n_envs: Number of parallel environments (default from config).
        use_memory: Whether to provide egocentric memory crop (False for ablation).
        features_dim: Output dimension of multi-modal feature extractor.
        learning_rate: Optimizer learning rate.
        lr_schedule_type: Learning rate decay schedule ('linear' or 'constant').
        n_steps: Rollout steps per environment before update.
        batch_size: Mini-batch size for SGD updates.
        n_epochs: Optimization epochs per rollout.
        ent_coef: Entropy bonus coefficient.
        clip_range: PPO clipping threshold.
        eval_freq: Timesteps between validation evaluations.
        n_eval_episodes: Number of validation episodes per evaluation.
        track: Whether to sync logs to Weights & Biases (default: False).
        wandb_project: W&B project name.
        wandb_entity: W&B entity/organization name.
        exp_name: Custom experiment name suffix.
        output_dir: Root directory for checkpoints and logs.
        device: PyTorch device ('auto', 'cpu', 'cuda').
        verbose: Verbosity level (0=silent, 1=info).

    Returns:
        tuple of (trained_model, best_eval_revealed_coverage, eval_history)
    """
    algo_key = algo.lower().strip()
    if algo_key not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unsupported algorithm '{algo}'. Supported options: {list(SUPPORTED_ALGOS)}"
        )

    # 1. Load configurations
    train_cfg = get_train_config(algo_key)
    obs_cfg = get_obs_config()
    env_cfg = get_env_config(difficulty)
    seeds_cfg = get_seeds_config()

    crop_size = int(obs_cfg.get("crop_size", 15))
    num_envs = n_envs if n_envs is not None else int(train_cfg.get("n_envs", 8))
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

    # 2. Setup hierarchical run directories & names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "full" if use_memory else "nomem"
    custom_tag = f"_{exp_name}" if exp_name else ""
    group_name = f"{algo_key}-{difficulty}" + ("-no-memory" if not use_memory else "")
    run_subname = f"seed{seed}_{mode_tag}_{timestamp}{custom_tag}"
    run_id = f"{group_name}_{run_subname}"

    output_root = Path(output_dir)
    tb_log_dir = output_root / "tb_logs" / group_name
    checkpoint_dir = output_root / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Handle Weights & Biases (W&B) Dual-Logging
    full_config = {
        "algo": algo_key,
        "difficulty": difficulty,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "use_memory": use_memory,
        "crop_size": crop_size,
        "features_dim": features_dim,
        "learning_rate": lr,
        "lr_schedule": sched,
        "n_envs": num_envs,
        "n_steps": steps_per_env,
        "batch_size": m_batch,
        "n_epochs": epochs,
        "ent_coef": entropy_c,
        "clip_range": clipping,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "target_coverage": float(env_cfg.get("target_coverage", 0.95)),
    }

    if track:
        import wandb

        # Reset WANDB_MODE if previously disabled in the same process
        if os.environ.get("WANDB_MODE") == "disabled":
            del os.environ["WANDB_MODE"]

        tags = [algo_key, difficulty, f"seed-{seed}"]
        if not use_memory:
            tags.append("ablation-no-memory")

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_id,
            group=group_name,
            tags=tags,
            config=full_config,
            sync_tensorboard=True,
        )
    else:
        # Prevent wandb from prompting login or attempting network connection
        os.environ["WANDB_MODE"] = "disabled"

    # 4. Create Vectorized Training and Evaluation Environments
    train_vec_env = make_vec_env_dungeon(
        difficulty=difficulty,
        n_envs=num_envs,
        seed=seed,
        use_memory=use_memory,
        crop_size=crop_size,
    )

    eval_single_env = make_env(
        difficulty=difficulty,
        seed=None,
        use_memory=use_memory,
        crop_size=crop_size,
    )()

    # 5. Build Policy Architecture
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

    lr_fn: Callable[[float], float] | float = (
        linear_schedule(lr) if sched == "linear" else lr
    )

    # 6. Initialize Agent Model
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
        "seed": seed,
    }

    model: BaseAlgorithm
    if algo_key == "ppo":
        model = PPO(policy="MultiInputPolicy", **common_model_kwargs)
    elif algo_key == "recurrent_ppo":
        model = RecurrentPPO(policy="MultiInputLstmPolicy", **common_model_kwargs)
    else:
        raise ValueError(f"Unsupported algorithm '{algo}'.")

    # 7. Setup Evaluation Callback
    val_start = seeds_cfg.get("val", {}).get("start", 1000000)
    val_seeds = [val_start + i for i in range(eval_eps)]

    # Normalize eval frequency so it triggers every eval_step_freq environment steps
    eval_callback_freq = max(eval_step_freq // num_envs, 1)

    eval_callback = DungeonEvalCallback(
        eval_env=eval_single_env,
        eval_freq=eval_callback_freq,
        n_eval_episodes=eval_eps,
        val_seeds=val_seeds,
        best_model_save_path=checkpoint_dir,
        target_coverage=float(env_cfg.get("target_coverage", 0.95)),
        verbose=verbose,
    )

    # 8. Setup Checkpoint Callback (periodic saving)
    callbacks: list[BaseCallback] = [eval_callback]
    save_step_freq = save_freq if save_freq is not None else train_cfg.get("save_freq")
    if save_step_freq is not None and int(save_step_freq) > 0:
        save_cb_freq = max(int(save_step_freq) // num_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_cb_freq,
            save_path=str(checkpoint_dir),
            name_prefix="model_step",
            verbose=0,
        )
        callbacks.append(checkpoint_callback)

    # 9. Run Training Loop
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_subname,
        )
    finally:
        # Save final checkpoint
        final_file = checkpoint_dir / "final_model.zip"
        model.save(str(final_file))
        if verbose > 0:
            print(f"Training completed. Final model saved to {final_file}")

        # Cleanup environments
        train_vec_env.close()
        eval_single_env.close()

        # Finish W&B run if active
        if track:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass

    return model, eval_callback.best_coverage, eval_callback.eval_history


def train_ppo(
    *args: Any, **kwargs: Any
) -> tuple[BaseAlgorithm, float, list[dict[str, Any]]]:
    """Convenience alias for train(algo='ppo', ...) for backwards compatibility."""
    kwargs.setdefault("algo", "ppo")
    return train(*args, **kwargs)
