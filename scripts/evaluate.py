"""MiniGrid Explorer: Unified Multi-Run & Baseline Evaluation Engine.

Single source of truth for benchmark evaluation, statistical ensemble aggregation,
and comparative reporting across trained RL policies and heuristic/algorithmic baselines.

Usage Examples:
    # 1. Smoke test: 1 run, 5 episodes on Easy difficulty
    uv run python scripts/evaluate.py --checkpoint ppo_easy_5seeds --baselines frontier --difficulty easy --smoke-test

    # 2. Benchmark multi-seed ensembles on 100 held-out test seeds
    uv run python scripts/evaluate.py --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds --difficulty easy

    # 3. Comprehensive benchmark comparing RL models with classical baselines on Medium
    uv run python scripts/evaluate.py --checkpoints ppo_medium_5seeds ppo_nomem_medium_5seeds \\
        --baselines random frontier wall_follower oracle --difficulty medium

    # 4. Evaluate specific baselines only (no RL checkpoints required)
    uv run python scripts/evaluate.py --baselines random frontier wall_follower --difficulty easy --n-episodes 50

    # 5. Cross-difficulty generalization (evaluate easy-trained models on medium maps)
    uv run python scripts/evaluate.py --checkpoints ppo_easy_5seeds --difficulty medium --n-episodes 100
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from explorer.agents.base import BaseAgent
from explorer.agents.frontier_explorer import FrontierExplorerAgent
from explorer.agents.oracle_planner import OraclePlannerAgent
from explorer.agents.random_agent import RandomAgent
from explorer.agents.rl_agent import RLAgent
from explorer.agents.wall_follower import WallFollowerAgent
from explorer.eval.evaluator import (
    EvalResult,
    evaluate_single_agent,
)
from explorer.utils.config import get_split


def parse_run_index(dir_name: str) -> int:
    """Extract numeric run index for sorting run folders naturally (run_0, run_1, etc.)."""
    m = re.search(r"run_(\d+)", dir_name)
    return int(m.group(1)) if m else 999999


def find_run_checkpoints(model_dir: Path, max_runs: int = 5) -> list[tuple[str, Path]]:
    """Locate best_model.zip files for up to max_runs inside a model directory.

    Args:
        model_dir: Path to directory containing run subfolders or direct model zips.
        max_runs: Cap N on the number of runs to evaluate. Evaluates min(len(found), max_runs).
    """
    checkpoints: list[tuple[str, Path]] = []
    if not model_dir.exists():
        return checkpoints

    # 1. Scan for run_* subdirectories (e.g. run_0_seed_..., run_1_seed_...)
    run_dirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and "run_" in d.name],
        key=lambda d: parse_run_index(d.name),
    )

    for idx, d in enumerate(run_dirs[:max_runs]):
        best_zip = d / "best_model.zip"
        if best_zip.exists():
            checkpoints.append((f"run_{idx}", best_zip))
            continue
        final_zip = d / "final_model.zip"
        if final_zip.exists():
            checkpoints.append((f"run_{idx}", final_zip))
            continue
        zips = sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if zips:
            checkpoints.append((f"run_{idx}", zips[0]))

    # 2. Fallback to root model zip if no run subfolders exist
    if not checkpoints:
        for fname in ["best_model.zip", "final_model.zip"]:
            root_zip = model_dir / fname
            if root_zip.exists():
                checkpoints.append(("best", root_zip))
                break
        if not checkpoints:
            zips = sorted(model_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if zips:
                checkpoints.append(("best", zips[0]))

    return checkpoints


def resolve_model_target(
    target_str: str,
    ckpt_root: Path,
    max_runs: int = 5,
) -> tuple[str, list[tuple[str, Path]]] | None:
    """Resolve a target argument string dynamically into a display name and checkpoint runs.

    Supports:
      - Direct file or directory paths (absolute or relative to current working dir).
      - Directory names inside results/checkpoints/ (e.g. 'ppo_easy_5seeds').
      - Dynamic substring matching within results/checkpoints/.
    """
    cleaned = target_str.strip()
    if not cleaned:
        return None

    resolved_path: Path | None = None
    target_path = Path(cleaned)

    # 1. Direct path
    if target_path.exists():
        resolved_path = target_path
    # 2. Inside results/checkpoints/
    elif (ckpt_root / cleaned).exists():
        resolved_path = ckpt_root / cleaned
    else:
        # 3. Dynamic substring matching in ckpt_root
        matches = [
            d for d in ckpt_root.iterdir()
            if d.is_dir() and cleaned.lower() in d.name.lower()
        ]
        if matches:
            matches.sort(key=lambda p: len(p.name))
            resolved_path = matches[0]

    if resolved_path is None or not resolved_path.exists():
        return None

    # Derive display name dynamically from directory or file name
    if resolved_path.is_file():
        if resolved_path.name in ("best_model.zip", "final_model.zip"):
            display_name = resolved_path.parent.name
        else:
            display_name = resolved_path.stem
        return display_name, [("model", resolved_path)]

    # If it's a directory
    display_name = resolved_path.name
    runs = find_run_checkpoints(resolved_path, max_runs=max_runs)
    if not runs:
        return None

    return display_name, runs


def evaluate_baseline_agent(
    agent_name: str,
    difficulty: str,
    seeds: list[int],
) -> EvalResult:
    """Instantiate and evaluate a baseline agent across evaluation seeds."""
    agent_key = agent_name.lower().replace(" ", "_").replace(":", "")

    if "random" in agent_key:
        factory = lambda env, s: RandomAgent(seed=s)
        display_label = "Baseline: Random"
    elif "frontier" in agent_key:
        factory = lambda env, s: FrontierExplorerAgent(env)
        display_label = "Baseline: Frontier BFS"
    elif "wall" in agent_key:
        factory = lambda env, s: WallFollowerAgent()
        display_label = "Baseline: Wall Follower"
    elif "oracle" in agent_key:
        factory = lambda env, s: OraclePlannerAgent(env)
        display_label = "Baseline: Oracle Planner (Privileged)"
    else:
        raise ValueError(f"Unknown baseline: {agent_name}")

    return evaluate_single_agent(
        agent_or_factory=factory,
        agent_name=display_label,
        difficulty=difficulty,
        seeds=seeds,
        deterministic=True,
        verbose=False,
    )


def aggregate_model_family_results(
    family_name: str,
    run_results: list[EvalResult],
) -> dict[str, Any]:
    """Compute ensemble statistics across training seed runs (or map seeds for single runs)."""
    if not run_results:
        return {}

    n = len(run_results)
    cov_means = [r.coverage_mean for r in run_results]
    visited_means = [r.visited_path_mean for r in run_results]
    poi_means = [r.poi_discovery_mean for r in run_results]
    death_rates = [r.death_rate for r in run_results]
    completion_rates = [r.completion_rate for r in run_results]
    rewards = [r.mean_reward for r in run_results]
    ep_lens = [r.mean_ep_length for r in run_results]
    bump_rates = [r.bump_rate for r in run_results]
    loop_rates = [r.loop_rate for r in run_results]
    rev_effs = [r.revealed_efficiency for r in run_results]
    latencies = [r.latency_ms for r in run_results]
    fps_vals = [r.fps for r in run_results]

    cov_std = float(np.std(cov_means, ddof=1)) if n > 1 else float(run_results[0].coverage_std)
    rew_std = float(np.std(rewards, ddof=1)) if n > 1 else float(run_results[0].std_reward)

    return {
        "family_name": family_name,
        "n_runs": n,
        "coverage_mean": float(np.mean(cov_means)),
        "coverage_std": cov_std,
        "visited_mean": float(np.mean(visited_means)),
        "visited_std": float(np.std(visited_means)),
        "poi_mean": float(np.mean(poi_means)),
        "poi_std": float(np.std(poi_means)),
        "death_rate_mean": float(np.mean(death_rates)),
        "death_rate_std": float(np.std(death_rates)),
        "completion_rate_mean": float(np.mean(completion_rates)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": rew_std,
        "ep_length_mean": float(np.mean(ep_lens)),
        "bump_rate_mean": float(np.mean(bump_rates)),
        "loop_rate_mean": float(np.mean(loop_rates)),
        "revealed_efficiency_mean": float(np.mean(rev_effs)),
        "latency_ms_mean": float(np.mean(latencies)),
        "fps_mean": float(np.mean(fps_vals)),
        "milestones": {
            m: float(np.mean([r.coverage_milestones.get(m, 0.0) for r in run_results]))
            for m in ["25%", "50%", "75%", "100%"]
        },
        "runs": [r.to_dict() for r in run_results],
    }


def format_results_terminal(
    aggregated: list[dict[str, Any]],
    difficulty: str,
    n_episodes: int,
    seed_range: str,
) -> str:
    """Format evaluation results into 2 clean ASCII tables for terminal output."""
    div_len = 118
    lines: list[str] = [
        "=" * div_len,
        f" MiniGrid Explorer Evaluation: {difficulty.upper()} Suite ({n_episodes} Test Episodes: {seed_range})",
        "=" * div_len,
        "",
        "Table 1: Primary Exploration & Survival (Mean ± Std across Training Seeds)",
        "-" * div_len,
        f"{'Model / Agent':<36} {'Runs':<5} {'Coverage (Mean ± Std)':<23} {'POI (%)':<9} {'Success (%)':<12} {'Deaths (%)':<11} {'Steps':<8} {'Return':<9} {'Latency':<9}",
        "-" * div_len,
    ]

    for item in aggregated:
        name = item["family_name"]
        runs = str(item["n_runs"])
        cov = f"{item['coverage_mean']:.2f}% ± {item['coverage_std']:.2f}%"
        poi = f"{item['poi_mean']:.1f}%"
        succ = f"{item['completion_rate_mean']:.1f}%"
        deaths = f"{item['death_rate_mean']:.1f}%"
        steps = f"{item['ep_length_mean']:.1f}"
        ret = f"{item['reward_mean']:+.3f}"
        lat = f"{item['latency_ms_mean']:.2f} ms"
        lines.append(
            f"{name:<36} {runs:<5} {cov:<23} {poi:<9} {succ:<12} {deaths:<11} {steps:<8} {ret:<9} {lat:<9}"
        )

    lines.extend([
        "-" * div_len,
        "",
        "Table 2: Navigation Regularization & Diagnostic Telemetry",
        "-" * div_len,
        f"{'Model / Agent':<36} {'Wall Bump (%)':<15} {'Loop Rate (%)':<15} {'Revealed Eff (tiles/step)':<28} {'25% Cov':<10} {'50% Cov':<10} {'FPS':<6}",
        "-" * div_len,
    ])

    for item in aggregated:
        name = item["family_name"]
        bump = f"{item['bump_rate_mean']:.2f}%"
        loop = f"{item['loop_rate_mean']:.2f}%"
        eff = f"{item['revealed_efficiency_mean']:.3f}"
        m25 = f"{item['milestones'].get('25%', 0.0):.1f}%"
        m50 = f"{item['milestones'].get('50%', 0.0):.1f}%"
        fps = f"{item['fps_mean']:.0f}"
        lines.append(
            f"{name:<36} {bump:<15} {loop:<15} {eff:<28} {m25:<10} {m50:<10} {fps:<6}"
        )

    lines.append("-" * div_len)
    return "\n".join(lines)


def format_results_markdown(
    aggregated: list[dict[str, Any]],
    difficulty: str,
    n_episodes: int,
    seed_range: str,
) -> str:
    """Format aggregated results into standardized GitHub Flavored Markdown tables."""
    lines: list[str] = [
        f"# MiniGrid Explorer Evaluation Report: {difficulty.upper()} Suite",
        "",
        f"- **Generated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Difficulty Tier:** `{difficulty}`",
        f"- **Episodes Evaluated:** `{n_episodes}` (Seeds: `{seed_range}`)",
        "",
        "## Table 1: Primary Exploration & Survival Metrics",
        "",
        "| Model / Agent | Runs | Revealed Coverage (%) | Visited Floor (%) | POI Discovery (%) | Success (≥95%) | Death Rate (%) | Steps | Return | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in aggregated:
        name = item["family_name"]
        runs = item["n_runs"]
        cov = f"**{item['coverage_mean']:.2f}% ± {item['coverage_std']:.2f}%**"
        vis = f"{item['visited_mean']:.2f}%"
        poi = f"{item['poi_mean']:.1f}%"
        succ = f"**{item['completion_rate_mean']:.1f}%**"
        death = f"{item['death_rate_mean']:.1f}%"
        steps = f"{item['ep_length_mean']:.1f}"
        rew = f"{item['reward_mean']:+.3f} ± {item['reward_std']:.3f}"
        lat = f"{item['latency_ms_mean']:.2f} ms"
        lines.append(
            f"| **{name}** | {runs} | {cov} | {vis} | {poi} | {succ} | {death} | {steps} | {rew} | {lat} |"
        )

    lines.extend([
        "",
        "## Table 2: Navigation Regularization & Diagnostic Telemetry",
        "",
        "| Model / Agent | Wall Bump Rate (%) | Loop Rate (%) | Revealed Eff (tiles/step) | 25% Horizon Cov | 50% Horizon Cov | 75% Horizon Cov | FPS |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for item in aggregated:
        name = item["family_name"]
        bump = f"{item['bump_rate_mean']:.2f}%"
        loop = f"{item['loop_rate_mean']:.2f}%"
        eff = f"{item['revealed_efficiency_mean']:.3f}"
        m25 = f"{item['milestones'].get('25%', 0.0):.1f}%"
        m50 = f"{item['milestones'].get('50%', 0.0):.1f}%"
        m75 = f"{item['milestones'].get('75%', 0.0):.1f}%"
        fps = f"{item['fps_mean']:.0f}"
        lines.append(
            f"| **{name}** | {bump} | {loop} | {eff} | {m25} | {m50} | {m75} | {fps} |"
        )

    lines.append("")
    return "\n".join(lines)


def save_reports(
    aggregated: list[dict[str, Any]],
    output_dir: Path,
    difficulty: str,
    n_episodes: int,
    seed_range: str,
) -> tuple[Path, Path]:
    """Atomically save Markdown and JSON reports with timestamp and latest pointers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Markdown Report
    md_content = format_results_markdown(aggregated, difficulty, n_episodes, seed_range)
    md_file = output_dir / f"eval_{difficulty}_{timestamp}.md"
    md_file.write_text(md_content, encoding="utf-8")

    latest_md = output_dir / f"eval_{difficulty}_latest.md"
    shutil.copyfile(md_file, latest_md)

    # 2. JSON Report
    json_payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "difficulty": difficulty,
            "n_episodes": n_episodes,
            "seed_range": seed_range,
        },
        "results": aggregated,
    }
    json_file = output_dir / f"eval_{difficulty}_{timestamp}.json"
    json_file.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    latest_json = output_dir / f"eval_{difficulty}_latest.json"
    shutil.copyfile(json_file, latest_json)

    return md_file, json_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified MiniGrid Explorer Policy & Baseline Evaluation Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="List of checkpoint folder names (e.g. ppo_easy_5seeds), directories, or .zip files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Single checkpoint folder name, directory, or .zip file (shortcut for --checkpoints).",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Evaluation map difficulty tier.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train", "dev"],
        help="Seed split from configs/seeds.yaml.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate (defaults to entire split, e.g. 100 for test).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Override starting seed manually instead of using predefined split.",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        choices=["random", "frontier", "wall_follower", "oracle", "all", "none"],
        help="Baseline agents to evaluate alongside RL models.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Maximum number of training runs to evaluate per folder (cap: min(found, n_runs)).",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy action selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "eval",
        help="Directory to save generated evaluation reports.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast verification mode: evaluates 5 episodes and at most 1 run per model.",
    )

    args = parser.parse_args()

    # Determine seed sequence
    if args.seed_start is not None:
        n_ep = args.n_episodes if args.n_episodes is not None else (5 if args.smoke_test else 50)
        seeds = [args.seed_start + i for i in range(n_ep)]
    else:
        split_seeds = list(get_split(args.split))
        if args.smoke_test:
            n_ep = 5
        elif args.n_episodes is not None:
            n_ep = min(args.n_episodes, len(split_seeds))
        else:
            n_ep = len(split_seeds)
        seeds = split_seeds[:n_ep]

    max_runs = 1 if args.smoke_test else args.n_runs
    seed_range = f"{seeds[0]}..{seeds[-1]}"
    ckpt_root = ROOT_DIR / "results" / "checkpoints"

    # Collect checkpoint targets
    raw_targets: list[str] = []
    if args.checkpoints:
        raw_targets.extend(args.checkpoints)
    elif args.checkpoint:
        raw_targets.append(args.checkpoint)

    # Collect baseline targets
    baseline_targets: list[str] = []
    if args.baselines:
        for b in args.baselines:
            if b == "all":
                baseline_targets.extend(["random", "frontier", "wall_follower", "oracle"])
            elif b != "none":
                baseline_targets.append(b)
        # Deduplicate while preserving order
        seen_b: set[str] = set()
        dedup_b: list[str] = []
        for b in baseline_targets:
            if b not in seen_b:
                seen_b.add(b)
                dedup_b.append(b)
        baseline_targets = dedup_b

    # If neither checkpoint nor baselines specified, prompt clearly and exit
    if not raw_targets and not baseline_targets:
        print("\n[Error] No evaluation target specified!")
        print("Please provide at least one checkpoint with '--checkpoints <name/path>' or baseline with '--baselines <name>'.")
        print("Example: uv run python scripts/evaluate.py --checkpoints ppo_easy_5seeds --difficulty easy\n")
        sys.exit(1)

    print("\n" + "=" * 90)
    print(f" MiniGrid Explorer Evaluator: Initializing Evaluation Suite [{args.difficulty.upper()}]")
    print(f" Test Seeds:       {len(seeds)} episodes ({seed_range}) | Split: '{args.split}'")
    print(f" Max Runs/Model:   {max_runs}")
    print(f" Deterministic:    {args.deterministic}")
    print(f" Output Directory: {args.output_dir}")
    print("=" * 90 + "\n")

    aggregated_all: list[dict[str, Any]] = []

    # 1. Evaluate Baseline Agents (if requested)
    if baseline_targets:
        print("--> Evaluating Baseline Agents...")
        for b_name in baseline_targets:
            t0 = datetime.now()
            print(f"  [Baseline] Running '{b_name}' across {len(seeds)} episodes... ", end="", flush=True)
            res = evaluate_baseline_agent(b_name, args.difficulty, seeds)
            dt = (datetime.now() - t0).total_seconds()
            print(f"Done ({dt:.1f}s) | Cov: {res.coverage_mean:.2f}% ± {res.coverage_std:.2f}% | Succ: {res.completion_rate:.1f}%")
            agg = aggregate_model_family_results(res.agent_name, [res])
            aggregated_all.append(agg)

    # 2. Evaluate RL Checkpoints
    if raw_targets:
        print("--> Evaluating RL Policy Checkpoints...")
        for target_str in raw_targets:
            resolved = resolve_model_target(target_str, ckpt_root, max_runs=max_runs)
            if resolved is None:
                print(f"  [Warning] Checkpoint target '{target_str}' could not be resolved! Skipping.")
                continue

            family_name, run_ckpts = resolved
            print(f"  [RL Family] '{family_name}': Discovered {len(run_ckpts)} run(s) (cap={max_runs})")

            run_results: list[EvalResult] = []
            for run_label, ckpt_path in run_ckpts:
                t0 = datetime.now()
                print(f"    Evaluating {run_label} ({ckpt_path.parent.name}) ... ", end="", flush=True)
                agent = RLAgent(ckpt_path, deterministic=args.deterministic)
                res = evaluate_single_agent(
                    agent_or_factory=agent,
                    agent_name=f"{family_name} ({run_label})",
                    difficulty=args.difficulty,
                    seeds=seeds,
                    deterministic=args.deterministic,
                    verbose=False,
                )
                dt = (datetime.now() - t0).total_seconds()
                print(f"Done ({dt:.1f}s) | Cov: {res.coverage_mean:.2f}% | Succ: {res.completion_rate:.1f}%")
                run_results.append(res)

            if run_results:
                agg = aggregate_model_family_results(family_name, run_results)
                aggregated_all.append(agg)

    if not aggregated_all:
        print("\n[Error] No valid agents or models were evaluated.")
        sys.exit(1)

    # Print summary tables to terminal
    term_table = format_results_terminal(aggregated_all, args.difficulty, len(seeds), seed_range)
    print("\n" + term_table + "\n")

    # Save reports to normalized output directory
    md_path, json_path = save_reports(
        aggregated_all,
        args.output_dir,
        args.difficulty,
        len(seeds),
        seed_range,
    )
    print("Saved Evaluation Reports:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON:     {json_path}")
    print(f"  - Latest:   {args.output_dir / f'eval_{args.difficulty}_latest.md'}\n")


if __name__ == "__main__":
    main()
