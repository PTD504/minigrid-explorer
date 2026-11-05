#!/usr/bin/env python3
"""Official Benchmark Suite for MiniGrid Explorer on 100 Held-Out Test Seeds.

Evaluates all trained RL policies and baseline algorithms across the official 100 test seeds
(seeds 2,000,000 to 2,000,099 as configured in configs/seeds.yaml).

Produces:
1. Console summary table.
2. Markdown benchmark report (results/benchmarks/benchmark_100_seeds_<diff>.md).
3. Raw JSON dataset (results/benchmarks/benchmark_100_seeds_<diff>.json).
4. High-resolution comparison figure (results/benchmarks/benchmark_100_seeds_<diff>.png).

Usage:
    uv run python scripts/benchmark_test_set.py --difficulty medium
    uv run python scripts/benchmark_test_set.py --difficulty easy
    uv run python scripts/benchmark_test_set.py --difficulty medium --rl-only
    uv run python scripts/benchmark_test_set.py --difficulty medium --checkpoints medium nomem
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
    get_available_checkpoints,
    save_evaluation_report,
)
from explorer.ui.app import resolve_checkpoint_path
from explorer.utils.config import get_seeds_config


def plot_benchmark_distribution(results: list[EvalResult], out_path: Path, difficulty: str) -> None:
    """Generate high-resolution boxplot comparing coverage distributions across 100 seeds."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 2]})

    # Palette
    colors = [
        "#0284c7",  # Sky blue
        "#059669",  # Emerald
        "#7c3aed",  # Purple
        "#ea580c",  # Orange
        "#dc2626",  # Red
        "#4b5563",  # Gray
        "#ca8a04",  # Yellow
    ]

    labels = [r.agent_name for r in results]
    data = [r.raw_coverages for r in results]

    # Plot 1: Boxplot of Revealed Coverage
    try:
        bplot = ax1.boxplot(
            data,
            patch_artist=True,
            tick_labels=labels,
            vert=False,
            notch=True,
            medianprops={"color": "#111827", "linewidth": 2},
        )
    except TypeError:
        bplot = ax1.boxplot(
            data,
            patch_artist=True,
            labels=labels,
            vert=False,
            notch=True,
            medianprops={"color": "#111827", "linewidth": 2},
        )
    for patch, c in zip(bplot["boxes"], (colors * 3)[: len(results)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Reference target line at 95%
    ax1.axvline(95.0, color="#dc2626", linestyle="--", linewidth=1.8, label="95% Target Coverage")
    ax1.set_xlabel("Revealed Area Coverage (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Coverage Distribution on 100 Held-Out Seeds ({difficulty.upper()} 25x25)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xlim(0, 105)
    ax1.legend(loc="lower left", frameon=True)

    # Plot 2: Bar chart of Completion Rate vs Death Rate
    y_pos = np.arange(len(results))
    height = 0.35

    comp_rates = [r.completion_rate for r in results]
    death_rates = [r.death_rate for r in results]

    ax2.barh(y_pos - height / 2, comp_rates, height, label="Success Rate (≥95%)", color="#059669", alpha=0.85)
    ax2.barh(y_pos + height / 2, death_rates, height, label="Death Rate by Monster", color="#dc2626", alpha=0.85)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Safety & Task Completion Comparison", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 105)
    ax2.legend(loc="upper right", frameon=True)

    for i in range(len(results)):
        ax2.text(comp_rates[i] + 1.5, i - height / 2, f"{comp_rates[i]:.1f}%", va="center", fontsize=9)
        if death_rates[i] > 0:
            ax2.text(death_rates[i] + 1.5, i + height / 2, f"{death_rates[i]:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Benchmark distribution plot saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official MiniGrid Explorer Benchmark on 100 Held-Out Test Seeds."
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Evaluation difficulty (default: medium)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Optional specific checkpoints or aliases to evaluate (defaults to all trained models)",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Only evaluate RL checkpoints (skip baseline agents)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks",
        help="Directory to save benchmark reports (default: results/benchmarks)",
    )
    parser.add_argument(
        "--generalization",
        action="store_true",
        help="Also evaluate on training seed suite (0..99) to measure generalization gap",
    )

    args = parser.parse_args()

    # Load 100 test seeds from configs/seeds.yaml
    seeds_cfg = get_seeds_config()
    test_start = seeds_cfg.get("test", {}).get("start", 2_000_000)
    test_count = seeds_cfg.get("test", {}).get("count", 100)
    test_seeds = [test_start + i for i in range(test_count)]

    print("\n==========================================================================")
    print(" MiniGrid Explorer: Benchmark on 100 Held-Out Test Seeds")
    print(f" Difficulty: {args.difficulty.upper()} | Seeds: {test_start}..{test_start + test_count - 1} (N={test_count})")
    print("==========================================================================\n")

    eval_targets: list[tuple[str, Any, str]] = []

    # 1. Gather RL Models
    if args.checkpoints:
        for c in args.checkpoints:
            p = resolve_checkpoint_path(c)
            if p and p.exists():
                name = f"RL: {p.parent.name[:24]}..."
                eval_targets.append((name, RLAgent(p), "rl"))
    else:
        # Default: Discover best models from results/checkpoints
        all_ckpts = get_available_checkpoints()
        bests = [p for p in all_ckpts if p.name == "best_model.zip"]
        for p in bests:
            run_name = p.parent.name
            if "nomem" in run_name or "no-memory" in run_name:
                arch_tag = "PPO-NoMem"
            elif "recurrent" in run_name:
                arch_tag = "RecurrentPPO"
            else:
                arch_tag = "PPO"

            # Create readable name e.g. "PPO (Medium with Memory)"
            if "medium" in run_name:
                d_tag = "Medium"
            elif "easy" in run_name:
                d_tag = "Easy"
            else:
                d_tag = "Hard"

            label = f"{arch_tag} ({d_tag})"
            eval_targets.append((label, RLAgent(p), "rl"))

    # 2. Gather Baseline Algorithms (unless --rl-only)
    if not args.rl_only:
        eval_targets.append(("Frontier (BFS)", lambda env, s: FrontierExplorerAgent(env), "baseline"))
        eval_targets.append(("Wall Follower", lambda env, s: WallFollowerAgent(), "baseline"))
        eval_targets.append(("Random", lambda env, s: RandomAgent(seed=s), "baseline"))
        eval_targets.append(("Oracle (Upper)", lambda env, s: OraclePlannerAgent(env), "baseline"))

    results: list[EvalResult] = []

    for name, agent_target, kind in eval_targets:
        print(f"\n--> Running Evaluation: {name} on {len(test_seeds)} held-out seeds...")
        res = evaluate_single_agent(
            agent_or_factory=agent_target,
            agent_name=name,
            difficulty=args.difficulty,
            seeds=test_seeds,
            verbose=True,
        )
        results.append(res)
        print(
            f"    Coverage: {res.coverage_mean:.2f}% ± {res.coverage_std:.2f}% | "
            f"Median: {res.coverage_median:.2f}% | Success: {res.completion_rate:.1f}% | "
            f"Deaths: {res.death_rate:.1f}% | Visited: {res.visited_path_mean:.1f}%"
        )

    # Print summary table
    print("\n\n" + "=" * 90)
    print(f" OFFICIAL BENCHMARK RESULTS (100 HELD-OUT SEEDS - {args.difficulty.upper()})")
    print("=" * 90)
    print(format_results_terminal(results))

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"benchmark_100_seeds_{args.difficulty}_{timestamp}"
    json_path, md_path = save_evaluation_report(results, args.output_dir, prefix)
    print(f"\n[Saved] Full JSON data:  {json_path}")
    print(f"[Saved] Markdown report: {md_path}")

    # Generate and save publication-quality plot
    plot_path = Path(args.output_dir) / f"{prefix}.png"
    plot_benchmark_distribution(results, plot_path, args.difficulty)

    # Generalization Comparison (Train Seeds vs Held-out Test Seeds)
    if args.generalization:
        print("\n" + "=" * 90)
        print(" GENERALIZATION ANALYSIS: TRAIN MAPS (SEEDS 0..99) VS TEST MAPS (SEEDS 2M..)")
        print("=" * 90)
        train_start = seeds_cfg.get("train", {}).get("start", 0)
        train_seeds = [train_start + i for i in range(test_count)]
        gen_results: list[tuple[str, float, float, float]] = []

        for (name, agent_target, _kind), test_res in zip(eval_targets, results):
            print(f"--> Evaluating {name} on {len(train_seeds)} train seeds...")
            tr_res = evaluate_single_agent(
                agent_or_factory=agent_target,
                agent_name=f"{name} [Train]",
                difficulty=args.difficulty,
                seeds=train_seeds,
                verbose=False,
            )
            gap = tr_res.coverage_mean - test_res.coverage_mean
            gen_results.append((name, tr_res.coverage_mean, test_res.coverage_mean, gap))
            print(f"    Train: {tr_res.coverage_mean:.2f}% | Test: {test_res.coverage_mean:.2f}% | Gap: {gap:+.2f}%")

        gen_md = "\n\n### Generalization Analysis: Training Maps vs Held-Out Maps\n\n"
        gen_md += "| Agent / Policy | Train Coverage (N=100) | Test Coverage (N=100) | Generalization Gap (Δ%) |\n"
        gen_md += "| :--- | :---: | :---: | :---: |\n"
        for name, tr_cov, te_cov, gap in gen_results:
            gen_md += f"| **{name}** | {tr_cov:.2f}% | {te_cov:.2f}% | {gap:+.2f}% |\n"

        with open(md_path, "a", encoding="utf-8") as f:
            f.write(gen_md)
        print(f"\n[Updated] Appended Generalization Analysis to: {md_path}")


if __name__ == "__main__":
    main()
