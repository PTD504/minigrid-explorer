"""Run pure exploration calibration on MiniGrid Explorer without monster kill bias.

Measures the empirical steps required for FrontierExplorerAgent and OraclePlannerAgent
to reach key exploration thresholds (90%, 95%, 100% revealed coverage) across independent seeds
in split 'dev' under two monster regimes:
- 'off': no monsters spawned.
- 'invulnerable': monsters roam normally, but collisions do not cause death/termination or penalty.

Produces:
- results/calibration/episodes.csv: Detailed metrics per episode.
- results/calibration/coverage_curves.csv: Revealed coverage trajectory sampled every 5 steps.

Usage:
    uv run python scripts/calibrate_max_steps.py
    uv run python scripts/calibrate_max_steps.py --difficulty easy --agent frontier --monster-mode off
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from explorer.agents.frontier_explorer import FrontierExplorerAgent
from explorer.agents.oracle_planner import OraclePlannerAgent
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper
from explorer.utils.config import get_env_config, get_seed_range


def run_calibration_episode(
    difficulty: str,
    agent_type: str,
    monster_mode: str,
    seed: int,
    max_steps_safety: int,
    sample_interval: int = 5,
    early_stop_100: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a single calibration episode, recording threshold milestones and curve telemetry."""
    raw_env = ExploreEnv(
        difficulty=difficulty,
        monster_mode=monster_mode,
        max_steps=max_steps_safety,
        terminate_on_coverage=False,
    )
    env = ObservationMemoryWrapper(raw_env)

    if agent_type == "frontier":
        agent = FrontierExplorerAgent(raw_env)
    elif agent_type == "oracle":
        agent = OraclePlannerAgent(raw_env)
    else:
        raise ValueError(f"Unknown agent_type '{agent_type}'")

    obs, info = env.reset(seed=seed)
    agent.reset()

    total_pois = len(raw_env.poi_positions)
    step_90: int | None = None
    step_95: int | None = None
    step_100: int | None = None
    step_first_poi: int | None = None
    step_all_pois: int | None = None

    wall_collisions = 0
    loop_steps = 0
    monster_collisions = 0

    curve_points: list[dict[str, Any]] = []

    # Initial step 0 state
    init_cov = float(info.get("revealed_coverage", 0.0)) * 100.0
    discovered_pois = int(info.get("discovered_pois", 0))

    if init_cov >= 90.0 and step_90 is None:
        step_90 = 0
    if init_cov >= 95.0 and step_95 is None:
        step_95 = 0
    if init_cov >= 100.0 and step_100 is None:
        step_100 = 0
    if discovered_pois >= 1 and step_first_poi is None:
        step_first_poi = 0
    if discovered_pois >= total_pois and step_all_pois is None:
        step_all_pois = 0

    curve_points.append({
        "difficulty": difficulty,
        "agent": agent_type,
        "monster_mode": monster_mode,
        "seed": seed,
        "step": 0,
        "revealed_coverage": round(init_cov, 2),
    })

    done = False
    step_count = 0
    end_reason = "timeout"

    while not done:
        action = agent.act(obs, info)
        obs, _rew, term, trunc, info = env.step(action)
        step_count = raw_env.step_count

        cov = float(info.get("revealed_coverage", 0.0)) * 100.0
        discovered_pois = int(info.get("discovered_pois", 0))

        if info.get("bump", False):
            wall_collisions += 1
        if info.get("is_loop", False):
            loop_steps += 1
        if info.get("monster_collision", False):
            monster_collisions += 1

        if cov >= 90.0 and step_90 is None:
            step_90 = step_count
        if cov >= 95.0 and step_95 is None:
            step_95 = step_count
        if cov >= 100.0 and step_100 is None:
            step_100 = step_count
        if discovered_pois >= 1 and step_first_poi is None:
            step_first_poi = step_count
        if discovered_pois >= total_pois and step_all_pois is None:
            step_all_pois = step_count

        if step_count % sample_interval == 0:
            curve_points.append({
                "difficulty": difficulty,
                "agent": agent_type,
                "monster_mode": monster_mode,
                "seed": seed,
                "step": step_count,
                "revealed_coverage": round(cov, 2),
            })

        if early_stop_100 and cov >= 100.0:
            end_reason = "coverage_100"
            done = True
        elif term:
            end_reason = info.get("end_reason", "terminated")
            done = True
        elif trunc or step_count >= max_steps_safety:
            end_reason = "timeout"
            done = True

    # Ensure last step is present in curve
    if curve_points[-1]["step"] != step_count:
        curve_points.append({
            "difficulty": difficulty,
            "agent": agent_type,
            "monster_mode": monster_mode,
            "seed": seed,
            "step": step_count,
            "revealed_coverage": round(cov, 2),
        })

    final_coverage = round(float(info.get("revealed_coverage", 0.0)) * 100.0, 2)

    ep_meta = {
        "difficulty": difficulty,
        "agent": agent_type,
        "monster_mode": monster_mode,
        "seed": seed,
        "step_90": step_90 if step_90 is not None else "",
        "step_95": step_95 if step_95 is not None else "",
        "step_100": step_100 if step_100 is not None else "",
        "step_first_poi": step_first_poi if step_first_poi is not None else "",
        "step_all_pois": step_all_pois if step_all_pois is not None else "",
        "monster_collisions": monster_collisions,
        "wall_collisions": wall_collisions,
        "loop_steps": loop_steps,
        "total_steps": step_count,
        "final_coverage": final_coverage,
        "end_reason": end_reason,
    }

    return ep_meta, curve_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate max_steps via pure exploration baseline runs on dev split."
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty tier to evaluate (default: all)",
    )
    parser.add_argument(
        "--agent",
        choices=["frontier", "oracle", "all"],
        default="all",
        help="Agent baseline to evaluate (default: all)",
    )
    parser.add_argument(
        "--monster-mode",
        choices=["off", "invulnerable", "all"],
        default="all",
        help="Monster mode to evaluate (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Seed split to load from configs/seeds.yaml (default: dev)",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=3.0,
        help="Safety ceiling multiplier on existing max_steps (default: 3.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/calibration",
        help="Output directory for generated calibration CSVs (default: results/calibration)",
    )
    parser.add_argument(
        "--no-early-stop-100",
        action="store_true",
        help="Do not terminate early when 100% coverage is reached",
    )

    args = parser.parse_args()

    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]
    agents = ["frontier", "oracle"] if args.agent == "all" else [args.agent]
    monster_modes = ["off", "invulnerable"] if args.monster_mode == "all" else [args.monster_mode]

    # Load seeds strictly from configs/seeds.yaml via explorer.utils.config
    seeds = get_seed_range(args.split)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_file = out_dir / "episodes.csv"
    curves_file = out_dir / "coverage_curves.csv"

    print("==========================================================================")
    print(" MiniGrid Explorer: Pure Exploration Step Calibration")
    print(f" Split: '{args.split}' ({len(seeds)} seeds: {seeds[0]}..{seeds[-1]})")
    print(f" Difficulties: {difficulties}")
    print(f" Agents:       {agents}")
    print(f" Monster Modes: {monster_modes}")
    print(f" Safety Ceiling Multiplier: {args.multiplier}x")
    print("==========================================================================\n")

    all_episodes: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []

    total_runs = len(difficulties) * len(agents) * len(monster_modes) * len(seeds)
    run_idx = 0
    t_start = time.perf_counter()

    for diff in difficulties:
        env_cfg = get_env_config(diff)
        base_max_steps = int(env_cfg.get("max_steps", 250))
        safety_max_steps = int(base_max_steps * args.multiplier)

        for ag in agents:
            for m_mode in monster_modes:
                print(
                    f"--> Running: diff={diff.upper():<6} | agent={ag:<8} | "
                    f"mode={m_mode:<12} | safety_limit={safety_max_steps}"
                )
                t_sub = time.perf_counter()

                for s in seeds:
                    run_idx += 1
                    ep_meta, curves = run_calibration_episode(
                        difficulty=diff,
                        agent_type=ag,
                        monster_mode=m_mode,
                        seed=s,
                        max_steps_safety=safety_max_steps,
                        sample_interval=5,
                        early_stop_100=not args.no_early_stop_100,
                    )
                    all_episodes.append(ep_meta)
                    all_curves.extend(curves)

                dt_sub = time.perf_counter() - t_sub
                # Compute quick summary for terminal
                steps_95 = [int(e["step_95"]) for e in all_episodes[-len(seeds):] if e["step_95"] != ""]
                pct_95 = (len(steps_95) / len(seeds)) * 100.0
                mean_95 = (sum(steps_95) / len(steps_95)) if steps_95 else 0.0
                print(
                    f"    Done {len(seeds)} seeds in {dt_sub:.2f}s | "
                    f"Reaching 95%: {pct_95:.1f}% ({len(steps_95)}/{len(seeds)}) | "
                    f"Mean steps to 95%: {mean_95:.1f}\n"
                )

    # Write episodes.csv
    ep_fieldnames = [
        "difficulty",
        "agent",
        "monster_mode",
        "seed",
        "step_90",
        "step_95",
        "step_100",
        "step_first_poi",
        "step_all_pois",
        "monster_collisions",
        "wall_collisions",
        "loop_steps",
        "total_steps",
        "final_coverage",
        "end_reason",
    ]
    with open(episodes_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ep_fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)
    print(f"Saved episode calibration metrics to: {episodes_file} ({len(all_episodes)} rows)")

    # Write coverage_curves.csv
    curve_fieldnames = [
        "difficulty",
        "agent",
        "monster_mode",
        "seed",
        "step",
        "revealed_coverage",
    ]
    with open(curves_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fieldnames)
        writer.writeheader()
        writer.writerows(all_curves)
    print(f"Saved coverage curves trajectory to:  {curves_file} ({len(all_curves)} rows)")

    total_time = time.perf_counter() - t_start
    print(f"\nAll {total_runs} calibration episodes completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()

