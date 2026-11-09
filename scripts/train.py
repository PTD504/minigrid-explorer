"""Unified CLI training script for MiniGrid Explorer RL agents.

Supports:
- Upfront reproducible multi-run seeding (--seed, --seed-count)
- Fault-tolerant resume (--resume)
- Incomplete run archiving and restart (--restart)
- Transfer / fine-tuning from existing checkpoints (--init-from <path>)
- Smoke test override (--max-steps-override)

Usage:
  # Standard single run with default global seed 2026:
  uv run python scripts/train.py --difficulty easy --total-timesteps 250000

  # Multi-seed study (3 runs derived from global seed 2026):
  uv run python scripts/train.py --difficulty easy --seed 2026 --seed-count 3

  # Resume an interrupted run:
  uv run python scripts/train.py --difficulty easy --seed 2026 --seed-count 3 --resume

  # Fine-tune / transfer from an existing checkpoint:
  uv run python scripts/train.py --difficulty medium --init-from results/checkpoints/.../best_model.zip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from explorer.train import train_multirun
from explorer.utils.config import get_reward_config, validate_seed_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RL agent on MiniGrid Explorer with multi-seed orchestration and dual TB/W&B logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core algorithm & environment arguments
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "recurrent_ppo"],
        help="Reinforcement learning algorithm.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Dungeon difficulty level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Global random seed from which all training run seeds are derived.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Number of sequential training runs to derive and execute.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500000,
        help="Total environment interaction steps per run.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel vectorized environments (SubprocVecEnv).",
    )

    # Resume, Restart, and Transfer flags
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution of incomplete runs from latest_model.zip and run_state.json.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Archive incomplete run directories and restart training fresh from step 0.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Initialize policy weights from an existing checkpoint (mutually exclusive with --resume).",
    )
    parser.add_argument(
        "--max-steps-override",
        type=int,
        default=None,
        help="Override max_steps per episode (strictly for rapid verification / smoke tests without editing configs).",
    )

    # Ablation & Architecture
    mem_group = parser.add_mutually_exclusive_group()
    mem_group.add_argument(
        "--no-memory",
        dest="use_memory",
        action="store_false",
        default=None,
        help="Explicitly disable egocentric memory crop (view-only).",
    )
    mem_group.add_argument(
        "--with-memory",
        dest="use_memory",
        action="store_true",
        default=None,
        help="Explicitly enable egocentric memory crop.",
    )
    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Feature extractor latent dimension.",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default="default",
        help="Reward configuration profile ('default', 'no_bump', 'no_loop', 'no_bump_no_loop').",
    )
    parser.add_argument(
        "--no-bump",
        action="store_true",
        default=False,
        help="Ablation flag: disable wall bump penalty (c_bump = 0.0).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        default=False,
        help="Ablation flag: disable loop state penalty (c_loop = 0.0).",
    )

    # Optimization overrides
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override (if None, read from configs/train/<algo>.yaml).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Steps per environment per rollout update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="SGD mini-batch size.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Entropy bonus coefficient override.",
    )

    # Evaluation & Checkpointing
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation interval in total environment steps.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=20,
        help="Number of deterministic validation episodes per evaluation.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Periodic checkpoint saving interval in environment steps.",
    )

    # Dual Logging: TensorBoard (default) + W&B (opt-in)
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable Weights & Biases (W&B) cloud tracking alongside TensorBoard.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="minigrid-explorer",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/username (optional).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Custom experiment name tag for checkpoint and log directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output root directory for checkpoints and logs.",
    )

    # Execution device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run PyTorch tensors on ('auto', 'cpu', 'cuda').",
    )

    args = parser.parse_args()

    # Mutual exclusion checks
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive.")
    if args.resume and args.restart:
        parser.error("--resume and --restart are mutually exclusive.")
    if args.restart and args.init_from:
        parser.error("--restart and --init-from are mutually exclusive.")

    # Environment variable fallback for max-steps-override
    env_max_steps = os.environ.get("MINIGRID_MAX_STEPS_OVERRIDE")
    if args.max_steps_override is None and env_max_steps is not None:
        try:
            args.max_steps_override = int(env_max_steps)
        except ValueError:
            pass

    return args


def main() -> None:
    # 1. Enforce strict split disjointness and absence of invalid keys
    validate_seed_splits()

    args = parse_args()
    if args.use_memory is not None:
        use_memory = args.use_memory
    else:
        # Default policy architecture:
        # - PPO: uses 15x15 explicit spatial memory (use_memory=True)
        # - RecurrentPPO: uses implicit LSTM sequence memory, no spatial memory (use_memory=False)
        use_memory = (args.algo != "recurrent_ppo")

    # Resolve reward configuration profile
    reward_config_name = args.reward_config
    if args.no_bump and args.no_loop:
        reward_config_name = "no_bump_no_loop"
    elif args.no_bump:
        reward_config_name = "no_bump"
    elif args.no_loop:
        reward_config_name = "no_loop"

    reward_cfg = get_reward_config(reward_config_name)

    print("=" * 70)
    print("MiniGrid Explorer - Reinforcement Learning Training Pipeline")
    print(
        f"Algorithm: {args.algo.upper()} | Difficulty: {args.difficulty.upper()} | Global Seed: {args.seed}"
    )
    print(
        f"Seed Count: {args.seed_count} | Steps/Run: {args.total_timesteps:,} | Parallel Envs: {args.n_envs}"
    )
    print(f"Memory Branch Active: {use_memory} (use_memory={use_memory})")
    print(
        f"Reward Profile: {reward_config_name} (c_bump={reward_cfg.get('c_bump', 0.02)}, c_loop={reward_cfg.get('c_loop', 0.01)})"
    )
    if args.resume:
        print("Mode: RESUME INCOMPLETE RUNS")
    elif args.restart:
        print("Mode: RESTART (ARCHIVE INCOMPLETE)")
    elif args.init_from:
        print(f"Mode: FINE-TUNE FROM {args.init_from}")
    else:
        print("Mode: FRESH MULTI-RUN")

    if args.max_steps_override:
        print(f"Max Steps Override (Smoke Test): {args.max_steps_override}")

    print(
        f"Tracking: Local TensorBoard [YES] | W&B Cloud [{'ENABLED' if args.track else 'OFFLINE/DISABLED'}]"
    )
    print("=" * 70)

    _model, best_cov, _history = train_multirun(
        algo=args.algo,
        difficulty=args.difficulty,
        seed=args.seed,
        seed_count=args.seed_count,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        use_memory=use_memory,
        features_dim=args.features_dim,
        reward_config_name=reward_config_name,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        max_steps_override=args.max_steps_override,
        resume=args.resume,
        restart=args.restart,
        init_from=args.init_from,
        track=args.track,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Load aggregate summary for final banner
    exp_dir = (
        Path(args.output_dir)
        / "checkpoints"
        / (
            args.exp_name
            or f"{args.algo}-{args.difficulty}_{'full' if use_memory else 'nomem'}_{reward_config_name}_seed{args.seed}"
        )
    )
    val_summary_file = exp_dir / "val_summary.json"
    summary_data = {}
    if val_summary_file.exists():
        try:
            with open(val_summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("ALL RUNS PROCESSED!")
    if summary_data and summary_data.get("num_runs", 0) > 0:
        cov_m = summary_data.get("coverage_mean", 0.0)
        cov_s = summary_data.get("coverage_std", 0.0)
        poi_m = summary_data.get("poi_mean", 0.0)
        poi_s = summary_data.get("poi_std", 0.0)
        top_r = summary_data.get("top_run")
        print(
            f"  • Multi-Run Mean: {cov_m * 100:.2f}% ± {cov_s * 100:.2f}% Coverage | "
            f"{poi_m * 100:.2f}% ± {poi_s * 100:.2f}% POI"
        )
        if top_r:
            print(
                f"  • Top Checkpoint: {top_r['best_score'] * 100:.2f}% Coverage "
                f"({top_r['run']} @ step {top_r['best_timestep']})"
            )
    else:
        print(f"  • Best Validation Coverage: {best_cov * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
