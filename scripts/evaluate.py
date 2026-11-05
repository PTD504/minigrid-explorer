#!/usr/bin/env python3
"""Interactive RL Model & Baseline Evaluation CLI for MiniGrid Explorer.

Evaluate one or more trained RL model checkpoints (with smart aliases) or baselines,
measuring coverage distribution, death rates, temporal efficiency, and reward breakdowns.

Usage Examples:
    # Quick evaluation of a single model on Medium difficulty:
    uv run python scripts/evaluate.py --checkpoint medium --difficulty medium

    # Compare memory vs no-memory model on 50 seeds:
    uv run python scripts/evaluate.py --checkpoints medium nomem --difficulty medium --n-episodes 50

    # Compare all 3 models on Easy:
    uv run python scripts/evaluate.py --checkpoints easy nomem recurrent --difficulty easy --n-episodes 50

    # Include Frontier BFS baseline for comparison:
    uv run python scripts/evaluate.py --checkpoints medium nomem --include-baseline frontier --difficulty medium
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from explorer.agents.frontier_explorer import FrontierExplorerAgent
from explorer.agents.oracle_planner import OraclePlannerAgent
from explorer.agents.random_agent import RandomAgent
from explorer.agents.rl_agent import RLAgent
from explorer.agents.wall_follower import WallFollowerAgent
from explorer.eval.evaluator import (
    EvalResult,
    evaluate_single_agent,
    format_results_terminal,
    save_evaluation_report,
)
from explorer.ui.app import resolve_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniGrid Explorer Model Inference & Telemetry Evaluator."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Single checkpoint path or alias ('medium', 'nomem', 'recurrent', 'easy')",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="List of multiple checkpoint paths or aliases to compare side-by-side",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Map difficulty setting (default: medium)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes / seeds (default: 50)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1_000_000,
        help="Starting seed index (default: 1,000,000 for validation split)",
    )
    parser.add_argument(
        "--include-baseline",
        nargs="*",
        choices=["frontier", "wall_follower", "random", "oracle", "all"],
        default=None,
        help="Optional baseline agents to include for comparison",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy inference (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/eval",
        help="Directory to save evaluation report files (default: results/eval)",
    )

    args = parser.parse_args()

    # Collect checkpoint targets
    target_ckpts: list[str] = []
    if args.checkpoints:
        target_ckpts.extend(args.checkpoints)
    elif args.checkpoint:
        target_ckpts.append(args.checkpoint)
    else:
        # Default: compare medium memory and medium no-memory
        target_ckpts = ["medium", "nomem"]

    seeds = [args.seed_start + i for i in range(args.n_episodes)]

    print("\n=================================================================")
    print(f" MiniGrid Explorer Evaluation: {len(target_ckpts)} Model(s) on {args.difficulty.upper()}")
    print(f" Seeds: {args.seed_start}..{args.seed_start + args.n_episodes - 1} ({args.n_episodes} episodes)")
    print("=================================================================\n")

    results: list[EvalResult] = []

    # 1. Evaluate RL Models
    for ckpt_input in target_ckpts:
        resolved_path = resolve_checkpoint_path(ckpt_input)
        if resolved_path is None or not resolved_path.exists():
            print(f"Error: Checkpoint '{ckpt_input}' could not be resolved! Skipping.")
            continue

        run_name = resolved_path.parent.name
        is_nomem = "nomem" in run_name or "no-memory" in run_name
        is_rec = "recurrent" in run_name
        alias_tag = (
            f"RL: {run_name[:24]}... ({'RecPPO' if is_rec else 'PPO-NoMem' if is_nomem else 'PPO'})"
        )

        print(f"--> Evaluating RL Checkpoint: {resolved_path.name} in {run_name}")
        agent = RLAgent(resolved_path, deterministic=args.deterministic)
        res = evaluate_single_agent(
            agent_or_factory=agent,
            agent_name=alias_tag,
            difficulty=args.difficulty,
            seeds=seeds,
            deterministic=args.deterministic,
            verbose=True,
        )
        results.append(res)
        print(
            f"    Result: Coverage = {res.coverage_mean:.2f}% ± {res.coverage_std:.2f}% "
            f"| Success: {res.completion_rate:.1f}% | Deaths: {res.death_rate:.1f}%\n"
        )

    # 2. Evaluate Optional Baselines
    if args.include_baseline:
        baseline_keys = args.include_baseline
        if "all" in baseline_keys:
            baseline_keys = ["frontier", "wall_follower", "random", "oracle"]

        for b_key in baseline_keys:
            print(f"--> Evaluating Baseline: {b_key}")
            if b_key == "random":
                factory = lambda env, s: RandomAgent(seed=s)
            elif b_key == "wall_follower":
                factory = lambda env, s: WallFollowerAgent()
            elif b_key == "frontier":
                factory = lambda env, s: FrontierExplorerAgent(env)
            elif b_key == "oracle":
                factory = lambda env, s: OraclePlannerAgent(env)
            else:
                continue

            res = evaluate_single_agent(
                agent_or_factory=factory,
                agent_name=f"Baseline: {b_key}",
                difficulty=args.difficulty,
                seeds=seeds,
                verbose=True,
            )
            results.append(res)
            print(
                f"    Result: Coverage = {res.coverage_mean:.2f}% ± {res.coverage_std:.2f}% "
                f"| Success: {res.completion_rate:.1f}% | Deaths: {res.death_rate:.1f}%\n"
            )

    if not results:
        print("No models were successfully evaluated.")
        return

    # Print summary table
    print("\n" + format_results_terminal(results) + "\n")

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"eval_{args.difficulty}_{timestamp}"
    json_path, md_path = save_evaluation_report(results, args.output_dir, prefix)
    print(f"Full JSON report saved:     {json_path}")
    print(f"Markdown report saved:      {md_path}\n")


if __name__ == "__main__":
    main()
