#!/usr/bin/env python3
"""Run and benchmark baseline exploration agents on MiniGrid Explorer.

Supports:
1. RandomAgent: Uniform random exploration (naive lower bound).
2. WallFollowerAgent: Deterministic left-hand wall-following heuristic.
3. FrontierExplorerAgent: Classical BFS frontier exploration (primary RL target).
4. OraclePlannerAgent: Privileged shortest-path planner (theoretical upper bound).

Evaluates key metrics across episodes:
- Revealed area coverage (|R_T ∩ W| / |W|)
- Visited path coverage (|V_T ∩ W| / |W|)
- POI discovery rate
- Survival rate & collision rate
- Temporal efficiency (steps, bump rate, reward)
- Step inference latency (ms)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from explorer.agents import (
    BaseAgent,
    FrontierExplorerAgent,
    OraclePlannerAgent,
    RandomAgent,
    WallFollowerAgent,
)
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper
from explorer.utils.config import get_seeds_config

AGENT_REGISTRY = {
    "random": RandomAgent,
    "wall_follower": WallFollowerAgent,
    "frontier": FrontierExplorerAgent,
    "oracle": OraclePlannerAgent,
}


def create_agent(agent_name: str, env: Any, seed: int) -> BaseAgent:
    """Instantiate a baseline agent with appropriate configuration."""
    if agent_name == "random":
        return RandomAgent(seed=seed)
    elif agent_name == "wall_follower":
        return WallFollowerAgent()
    elif agent_name == "frontier":
        return FrontierExplorerAgent(env)
    elif agent_name == "oracle":
        return OraclePlannerAgent(env)
    else:
        raise ValueError(f"Unknown agent: {agent_name}. Valid choices: {list(AGENT_REGISTRY.keys())}")


def evaluate_single_episode(
    agent_name: str,
    difficulty: str,
    seed: int,
    max_steps: int | None = None,
    terminate_on_coverage: bool = True,
    track_trajectory: bool = False,
) -> dict[str, Any]:
    """Run a single evaluation episode for a given agent and seed."""
    raw_env = ExploreEnv(
        difficulty=difficulty,
        max_steps=max_steps,
        terminate_on_coverage=terminate_on_coverage,
    )
    env = ObservationMemoryWrapper(raw_env)
    agent = create_agent(agent_name, env, seed=seed)

    obs, info = env.reset(seed=seed)
    agent.reset()

    total_reward = 0.0
    bumps = 0
    step_latencies: list[float] = []

    # Telemetry tracking for visualization
    agent_traj = [tuple(raw_env.agent_pos)] if track_trajectory else []
    coverage_curve = [info["revealed_coverage"] * 100.0] if track_trajectory else []
    monster_trajs: dict[int, list[tuple[int, int]]] = (
        {m.id: [m.pos] for m in raw_env.monsters} if track_trajectory else {}
    )

    term = False
    trunc = False

    while not (term or trunc):
        # Measure inference latency
        t0 = time.perf_counter()
        action = agent.act(obs, info)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(dt_ms)

        obs, reward, term, trunc, info = env.step(action)
        total_reward += float(reward)
        if info.get("bump", False):
            bumps += 1

        if track_trajectory:
            agent_traj.append(tuple(raw_env.agent_pos))
            coverage_curve.append(info["revealed_coverage"] * 100.0)
            for m in raw_env.monsters:
                monster_trajs[m.id].append(m.pos)

    steps = raw_env.step_count
    end_reason = info.get("end_reason", "unknown")
    survived = 1.0 if end_reason != "death" else 0.0

    episode_data: dict[str, Any] = {
        "agent": agent_name,
        "seed": seed,
        "revealed_coverage": info["revealed_coverage"] * 100.0,
        "visited_coverage": info["visited_coverage"] * 100.0,
        "poi_discovery_rate": info["poi_discovery_rate"] * 100.0,
        "survived": survived,
        "steps": steps,
        "total_reward": total_reward,
        "bump_rate": (bumps / max(1, steps)) * 100.0,
        "end_reason": end_reason,
        "latency_ms": float(np.mean(step_latencies)) if step_latencies else 0.0,
    }

    if track_trajectory:
        episode_data["agent_traj"] = agent_traj
        episode_data["coverage_curve"] = coverage_curve
        episode_data["monster_trajs"] = monster_trajs
        episode_data["walkable_tiles"] = list(raw_env.walkable_tiles)
        episode_data["revealed_tiles"] = list(raw_env.tracker.revealed_tiles)
        episode_data["poi_positions"] = list(raw_env.poi_positions)
        episode_data["discovered_pois"] = list(raw_env.tracker.discovered_pois)
        episode_data["grid_size"] = raw_env.width

    return episode_data


def run_benchmark(
    agents: list[str],
    difficulty: str = "easy",
    episodes: int = 5,
    start_seed: int = 1000000,
    max_steps: int | None = None,
    terminate_on_coverage: bool = True,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run multi-episode benchmark across multiple agents."""
    benchmark_results: dict[str, dict[str, Any]] = {}

    for agent_name in agents:
        ep_results: list[dict[str, Any]] = []
        if verbose:
            print(f"\nEvaluating Agent: {agent_name.upper()} ({episodes} episodes on {difficulty})...")

        for i in range(episodes):
            seed = start_seed + i
            res = evaluate_single_episode(
                agent_name,
                difficulty,
                seed,
                max_steps=max_steps,
                terminate_on_coverage=terminate_on_coverage,
            )
            ep_results.append(res)
            if verbose:
                print(
                    f"  Episode {i+1:2d} (Seed {seed}): "
                    f"Revealed={res['revealed_coverage']:5.1f}% | "
                    f"POI={res['poi_discovery_rate']:5.1f}% | "
                    f"Steps={res['steps']:3d} | "
                    f"Reward={res['total_reward']:+6.2f} | "
                    f"End={res['end_reason']}"
                )

        # Aggregate statistics (mean and std)
        metrics = [
            "revealed_coverage",
            "visited_coverage",
            "poi_discovery_rate",
            "survived",
            "steps",
            "total_reward",
            "bump_rate",
            "latency_ms",
        ]
        agg: dict[str, Any] = {"raw_episodes": ep_results}
        for m in metrics:
            vals = [r[m] for r in ep_results]
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))

        benchmark_results[agent_name] = agg

    return benchmark_results


def print_summary_table(results: dict[str, dict[str, Any]], difficulty: str, episodes: int) -> None:
    """Print clean formatted Markdown/ASCII comparison table."""
    print("\n" + "=" * 105)
    print(f"BASELINE BENCHMARK REPORT: difficulty={difficulty.upper()} ({episodes} episodes)")
    print("=" * 105)
    header = (
        f"{'Agent':<18} | "
        f"{'Revealed Cov (%)':<16} | "
        f"{'Visited Cov (%)':<16} | "
        f"{'POI Rate (%)':<14} | "
        f"{'Survival (%)':<12} | "
        f"{'Steps':<12} | "
        f"{'Reward':<14}"
    )
    print(header)
    print("-" * 105)

    for agent_name, data in results.items():
        rev_str = f"{data['revealed_coverage_mean']:.1f} ± {data['revealed_coverage_std']:.1f}"
        vis_str = f"{data['visited_coverage_mean']:.1f} ± {data['visited_coverage_std']:.1f}"
        poi_str = f"{data['poi_discovery_rate_mean']:.1f} ± {data['poi_discovery_rate_std']:.1f}"
        surv_str = f"{data['survived_mean'] * 100:.1f}%"
        step_str = f"{data['steps_mean']:.1f} ± {data['steps_std']:.1f}"
        rew_str = f"{data['total_reward_mean']:+.2f} ± {data['total_reward_std']:.2f}"

        row = (
            f"{agent_name:<18} | "
            f"{rev_str:<16} | "
            f"{vis_str:<16} | "
            f"{poi_str:<14} | "
            f"{surv_str:<12} | "
            f"{step_str:<12} | "
            f"{rew_str:<14}"
        )
        print(row)
    print("=" * 105 + "\n")


def generate_comparison_plot(
    agents: list[str],
    difficulty: str,
    seed: int,
    output_path: str,
    max_steps: int | None = None,
    terminate_on_coverage: bool = True,
) -> None:
    """Generate side-by-side diagnostic plots of trajectories and coverage curves."""
    print(f"Generating visual diagnostic comparison on seed {seed}...")
    trajectories: dict[str, dict[str, Any]] = {}

    for name in agents:
        data = evaluate_single_episode(
            name,
            difficulty,
            seed,
            max_steps=max_steps,
            terminate_on_coverage=terminate_on_coverage,
            track_trajectory=True,
        )
        trajectories[name] = data

    # Create figure: Left 2x2 grid for map trajectories, Right for coverage curves
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    # Subplots for the 4 agents
    agent_display_names = {
        "random": "Random Agent",
        "wall_follower": "Wall Follower Agent",
        "frontier": "Frontier Explorer Agent",
        "oracle": "Oracle Planner Agent",
    }
    agent_colors = {
        "random": "#e74c3c",
        "wall_follower": "#e67e22",
        "frontier": "#2ecc71",
        "oracle": "#3498db",
    }

    grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (agent_key, (r, c)) in enumerate(zip(agents, grid_positions)):
        ax = fig.add_subplot(gs[r, c])
        data = trajectories[agent_key]
        grid_sz = data["grid_size"]

        # Background grid
        bg = np.ones((grid_sz, grid_sz, 3), dtype=float) * 0.15

        # Draw revealed vs unrevealed walkable tiles
        for x, y in data["walkable_tiles"]:
            if (x, y) in data["revealed_tiles"]:
                bg[y, x] = [0.85, 0.85, 0.85]  # Light gray for revealed
            else:
                bg[y, x] = [0.35, 0.35, 0.35]  # Dim gray for unrevealed

        # Draw POIs
        for px, py in data["poi_positions"]:
            if (px, py) in data["discovered_pois"]:
                bg[py, px] = [0.6, 0.2, 0.8]  # Discovered purple
            else:
                bg[py, px] = [0.3, 0.1, 0.4]  # Undiscovered dark purple

        ax.imshow(bg)

        # Plot agent trajectory
        traj = np.array(data["agent_traj"])
        if len(traj) > 0:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=agent_colors[agent_key],
                linewidth=1.8,
                alpha=0.85,
                label="Path",
            )
            ax.scatter(traj[0, 0], traj[0, 1], c="blue", s=50, marker="o", label="Start", zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c="red", s=60, marker="*", label="End", zorder=5)

        # Plot monster trajectories
        for mpath in data["monster_trajs"].values():
            if len(mpath) > 0:
                marr = np.array(mpath)
                ax.plot(marr[:, 0], marr[:, 1], color="#e74c3c", linestyle=":", linewidth=1.2, alpha=0.6)
                ax.scatter(marr[-1, 0], marr[-1, 1], c="#c0392b", s=40, marker="s", zorder=4)

        ax.set_title(
            f"{agent_display_names.get(agent_key, agent_key)}\n"
            f"Cov: {data['revealed_coverage']:.1f}% | Steps: {data['steps']} | {data['end_reason']}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Coverage curves over time (spanning the right 2 columns)
    ax_curve = fig.add_subplot(gs[:, 2:])
    for agent_key in agents:
        curve = trajectories[agent_key]["coverage_curve"]
        steps = np.arange(len(curve))
        ax_curve.plot(
            steps,
            curve,
            label=f"{agent_display_names.get(agent_key, agent_key)} ({curve[-1]:.1f}%)",
            color=agent_colors[agent_key],
            linewidth=2.5,
        )

    ax_curve.axhline(95.0, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Target Coverage (95%)")
    ax_curve.set_xlabel("Environment Steps (t)", fontsize=12, fontweight="bold")
    ax_curve.set_ylabel("Revealed Area Coverage (%)", fontsize=12, fontweight="bold")
    ax_curve.set_title(
        f"Exploration Progress Over Time (Difficulty: {difficulty.upper()}, Seed: {seed})",
        fontsize=13,
        fontweight="bold",
    )
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax_curve.set_ylim(0, 105)

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_p, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot successfully saved to: {output_p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and evaluate baseline agents on MiniGrid Explorer."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        choices=["random", "wall_follower", "frontier", "oracle", "all"],
        help="Agent to evaluate (default: all)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Dungeon difficulty preset (default: easy)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per agent (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Starting evaluation seed (default: validation start seed from seeds.yaml)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max episode steps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode diagnostic metrics",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export raw JSON benchmark data",
    )
    parser.add_argument(
        "--terminate-on-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate episode when revealed coverage reaches target threshold (default: True)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparative trajectory and coverage curve plot",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="results/baseline_comparison.png",
        help="Destination path for generated diagnostic plot",
    )

    args = parser.parse_args()

    # Determine seed range
    if args.seed is None:
        seeds_cfg = get_seeds_config()
        start_seed = seeds_cfg.get("val", {}).get("start", 1000000)
    else:
        start_seed = args.seed

    agents_to_run = (
        ["random", "wall_follower", "frontier", "oracle"]
        if args.agent == "all"
        else [args.agent]
    )

    # Run benchmarks
    results = run_benchmark(
        agents=agents_to_run,
        difficulty=args.difficulty,
        episodes=args.episodes,
        start_seed=start_seed,
        max_steps=args.max_steps,
        terminate_on_coverage=args.terminate_on_coverage,
        verbose=args.verbose,
    )

    # Print summary table
    print_summary_table(results, difficulty=args.difficulty, episodes=args.episodes)

    # Export JSON if requested
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Exported raw benchmark metrics to: {out_p}")

    # Generate diagnostic plot if requested
    if args.plot:
        generate_comparison_plot(
            agents=agents_to_run,
            difficulty=args.difficulty,
            seed=start_seed,
            output_path=args.plot_output,
            max_steps=args.max_steps,
            terminate_on_coverage=args.terminate_on_coverage,
        )


if __name__ == "__main__":
    main()
