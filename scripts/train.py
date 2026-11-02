#!/usr/bin/env python3
"""Unified CLI training script for MiniGrid Explorer RL agents.

Supports standard PPO and RecurrentPPO (PPO + LSTM ablation).

Usage:
  # Reviewer mode (Offline TensorBoard only, zero W&B auth required):
  uv run python scripts/train.py --algo ppo --difficulty easy --seed 42 --total-timesteps 250000

  # Research mode with Weights & Biases cloud tracking:
  uv run python scripts/train.py --algo ppo --difficulty easy --seed 42 --track

  # RecurrentPPO (PPO + LSTM ablation study):
  uv run python scripts/train.py --algo recurrent_ppo --difficulty easy --seed 42 --track

  # Ablation experiment (No egocentric memory crop):
  uv run python scripts/train.py --algo ppo --difficulty easy --seed 42 --no-memory --track
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from explorer.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RL agent on MiniGrid Explorer with dual TB/W&B logging.",
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
        default=42,
        help="Base random seed for training environments.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500000,
        help="Total environment interaction steps.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel vectorized environments.",
    )

    # Ablation & Architecture
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable egocentric memory crop (Ablation study: view only).",
    )
    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Feature extractor latent dimension.",
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
        help="Custom experiment name tag appended to run ID.",
    )

    # Execution device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run PyTorch tensors on ('auto', 'cpu', 'cuda').",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_memory = not args.no_memory

    print("=" * 70)
    print("MiniGrid Explorer - Reinforcement Learning Training Pipeline")
    print(
        f"Algorithm: {args.algo.upper()} | Difficulty: {args.difficulty.upper()} | Seed: {args.seed}"
    )
    print(f"Total Steps: {args.total_timesteps:,} | Parallel Envs: {args.n_envs}")
    print(f"Memory Branch Active: {use_memory} (use_memory={use_memory})")
    print(
        f"Tracking: Local TensorBoard [YES] | W&B Cloud [{'ENABLED' if args.track else 'OFFLINE/DISABLED'}]"
    )
    print("=" * 70)

    _model, best_cov, _history = train(
        algo=args.algo,
        difficulty=args.difficulty,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        use_memory=use_memory,
        features_dim=args.features_dim,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        track=args.track,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        exp_name=args.exp_name,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print(
        f"TRAINING COMPLETE! Best Validation Revealed Coverage: {best_cov * 100:.2f}%"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
