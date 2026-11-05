from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from explorer.agents.base import BaseAgent
from explorer.agents.rl_agent import RLAgent
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_available_checkpoints() -> list[Path]:
    """Find all available trained checkpoints sorted with best/final models first."""
    ckpt_root = ROOT_DIR / "results" / "checkpoints"
    if not ckpt_root.exists():
        return []

    bests = sorted(ckpt_root.glob("**/best_model.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    finals = sorted(ckpt_root.glob("**/final_model.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    others = sorted(
        [p for p in ckpt_root.glob("**/*.zip") if p.name not in ("best_model.zip", "final_model.zip")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    all_ckpts: list[Path] = []
    seen: set[str] = set()
    for p in bests + finals + others:
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            all_ckpts.append(p)
    return all_ckpts


@dataclass
class EvalResult:
    """Detailed benchmark evaluation metrics for an exploration agent."""

    agent_name: str
    difficulty: str
    num_episodes: int
    coverage_mean: float
    coverage_std: float
    coverage_median: float
    coverage_min: float
    coverage_max: float
    coverage_q25: float
    coverage_q75: float
    completion_rate: float  # Percentage of episodes reaching >= 95% target
    death_rate: float  # Percentage of episodes ending in death by monster
    timeout_rate: float  # Percentage of episodes reaching max steps
    visited_path_mean: float  # Percentage of walkable floor visited
    poi_discovery_mean: float  # Percentage of POIs discovered
    mean_ep_length: float  # Average survival duration (steps)
    mean_reward: float
    std_reward: float = 0.0

    # Ather Labs Spec Section 7: Required Extended Metrics
    wall_collisions_mean: float = 0.0  # Average wall collisions / bumps per episode
    bump_rate: float = 0.0  # Percentage of steps that resulted in a bump
    revealed_efficiency: float = 0.0  # Newly revealed walkable tiles per step
    visited_efficiency: float = 0.0  # Unique visited walkable tiles per step
    loop_steps_mean: float = 0.0  # Average steps in repeated state loop per episode
    loop_rate: float = 0.0  # Percentage of steps in repeated state loop
    monster_encounters_mean: float = 0.0  # Average steps with monster in proximity (dist <= 1)
    monster_encounter_rate: float = 0.0  # Percentage of steps with monster in proximity
    coverage_milestones: dict[str, float] = field(default_factory=dict)  # Mean coverage at 25%, 50%, 75%, 100%

    reward_breakdown: dict[str, float] = field(default_factory=dict)
    end_reasons: dict[str, int] = field(default_factory=dict)
    fps: float = 0.0
    latency_ms: float = 0.0
    raw_coverages: list[float] = field(default_factory=list)
    raw_rewards: list[float] = field(default_factory=list)
    raw_lengths: list[int] = field(default_factory=list)
    raw_bumps: list[int] = field(default_factory=list)
    raw_loops: list[int] = field(default_factory=list)
    raw_monster_encounters: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


def detect_use_memory(agent: BaseAgent) -> bool:
    """Determine whether an agent requires memory wrapper based on policy inspection."""
    if isinstance(agent, RLAgent):
        model = getattr(agent, "model", None)
        if hasattr(model, "observation_space") and hasattr(model.observation_space, "spaces"):
            return "memory" in model.observation_space.spaces
    return True


def evaluate_single_agent(
    agent_or_factory: BaseAgent | Callable[[Any, int], BaseAgent],
    agent_name: str,
    difficulty: str = "medium",
    seeds: list[int] | None = None,
    use_memory: bool | None = None,
    deterministic: bool = True,
    verbose: bool = True,
) -> EvalResult:
    """Run comprehensive evaluation for an agent over a list of seeds."""
    if seeds is None:
        seeds = [2_000_000 + i for i in range(100)]

    raw_coverages: list[float] = []
    raw_visited: list[float] = []
    raw_pois: list[float] = []
    raw_rewards: list[float] = []
    raw_lengths: list[int] = []
    raw_bumps: list[int] = []
    raw_loops: list[int] = []
    raw_monster_encounters: list[int] = []
    raw_revealed_efficiencies: list[float] = []
    raw_visited_efficiencies: list[float] = []
    milestone_cov_accum: dict[str, list[float]] = {"25%": [], "50%": [], "75%": [], "100%": []}

    end_reasons: dict[str, int] = {"coverage": 0, "timeout": 0, "death": 0}
    reward_breakdown_accum: dict[str, float] = {}

    total_actions = 0
    t_start = time.perf_counter()

    for i, seed in enumerate(seeds):
        raw_env = ExploreEnv(difficulty=difficulty, terminate_on_coverage=True)

        # Instantiate agent
        if callable(agent_or_factory) and not isinstance(agent_or_factory, BaseAgent):
            agent = agent_or_factory(raw_env, seed)
        else:
            agent = agent_or_factory

        # Auto-detect memory wrapper requirement if not explicitly specified
        mem_flag = use_memory if use_memory is not None else detect_use_memory(agent)
        env = ObservationMemoryWrapper(
            raw_env,
            use_memory=mem_flag,
            include_step_ratio=True,
        )

        obs, info = env.reset(seed=seed)
        agent.reset()
        done = False

        ep_reward = 0.0
        ep_bumps = 0
        ep_loops = 0
        ep_encounters = 0
        ep_milestones: dict[str, float] = {}
        max_steps = raw_env.max_steps
        step_idx = 0

        while not done:
            action = agent.act(obs, info)
            obs, rew, term, trunc, info = env.step(action)
            step_idx += 1
            total_actions += 1
            ep_reward += float(rew)

            if info.get("bump", False):
                ep_bumps += 1
            if info.get("is_loop", False):
                ep_loops += 1
            if info.get("near_monster", False):
                ep_encounters += 1

            cov_now = info.get("revealed_coverage", 0.0) * 100.0
            if "25%" not in ep_milestones and step_idx >= max_steps * 0.25:
                ep_milestones["25%"] = cov_now
            if "50%" not in ep_milestones and step_idx >= max_steps * 0.50:
                ep_milestones["50%"] = cov_now
            if "75%" not in ep_milestones and step_idx >= max_steps * 0.75:
                ep_milestones["75%"] = cov_now

            done = term or trunc

        final_cov = info.get("revealed_coverage", 0.0) * 100.0
        for m in ["25%", "50%", "75%", "100%"]:
            if m not in ep_milestones:
                ep_milestones[m] = final_cov
            milestone_cov_accum[m].append(ep_milestones[m])

        steps = raw_env.step_count
        vis = info.get("visited_coverage", 0.0) * 100.0
        tot_pois = max(1, info.get("total_pois", len(raw_env.poi_positions)))
        poi_pct = (info.get("discovered_pois", 0) / tot_pois) * 100.0

        raw_coverages.append(final_cov)
        raw_visited.append(vis)
        raw_pois.append(poi_pct)
        raw_rewards.append(ep_reward)
        raw_lengths.append(steps)
        raw_bumps.append(ep_bumps)
        raw_loops.append(ep_loops)
        raw_monster_encounters.append(ep_encounters)

        revealed_count = info.get(
            "revealed_count",
            len(raw_env.tracker.revealed_tiles.intersection(raw_env.walkable_tiles)),
        )
        visited_count = info.get(
            "visited_count",
            len(raw_env.tracker.visited_tiles.intersection(raw_env.walkable_tiles)),
        )
        raw_revealed_efficiencies.append(revealed_count / max(1, steps))
        raw_visited_efficiencies.append(visited_count / max(1, steps))

        reason = info.get("end_reason", "unknown")
        end_reasons[reason] = end_reasons.get(reason, 0) + 1

        # Accumulate breakdown
        for k, v in info.get("reward_breakdown", {}).items():
            reward_breakdown_accum[k] = reward_breakdown_accum.get(k, 0.0) + float(v)

        if verbose and (i + 1) % max(1, len(seeds) // 5) == 0:
            print(f"  [{agent_name}] Seed {i + 1}/{len(seeds)}: Current Mean Cov = {np.mean(raw_coverages):.2f}%")

    total_time = max(1e-5, time.perf_counter() - t_start)
    fps = total_actions / total_time
    latency_ms = (total_time / max(1, total_actions)) * 1000.0

    n_episodes = len(seeds)
    breakdown_mean = {k: v / n_episodes for k, v in reward_breakdown_accum.items()}

    completed_count = end_reasons.get("coverage", 0)
    death_count = end_reasons.get("death", 0)
    timeout_count = end_reasons.get("timeout", 0)

    total_steps = max(1, int(np.sum(raw_lengths)))
    mean_bumps = float(np.mean(raw_bumps))
    bump_rate = (float(np.sum(raw_bumps)) / total_steps) * 100.0
    mean_loops = float(np.mean(raw_loops))
    loop_rate = (float(np.sum(raw_loops)) / total_steps) * 100.0
    mean_encounters = float(np.mean(raw_monster_encounters))
    encounter_rate = (float(np.sum(raw_monster_encounters)) / total_steps) * 100.0
    revealed_eff = float(np.mean(raw_revealed_efficiencies))
    visited_eff = float(np.mean(raw_visited_efficiencies))
    coverage_milestones = {
        m: float(np.mean(vals)) for m, vals in milestone_cov_accum.items()
    }

    return EvalResult(
        agent_name=agent_name,
        difficulty=difficulty,
        num_episodes=n_episodes,
        coverage_mean=float(np.mean(raw_coverages)),
        coverage_std=float(np.std(raw_coverages)),
        coverage_median=float(np.median(raw_coverages)),
        coverage_min=float(np.min(raw_coverages)),
        coverage_max=float(np.max(raw_coverages)),
        coverage_q25=float(np.percentile(raw_coverages, 25)),
        coverage_q75=float(np.percentile(raw_coverages, 75)),
        completion_rate=(completed_count / n_episodes) * 100.0,
        death_rate=(death_count / n_episodes) * 100.0,
        timeout_rate=(timeout_count / n_episodes) * 100.0,
        visited_path_mean=float(np.mean(raw_visited)),
        poi_discovery_mean=float(np.mean(raw_pois)),
        mean_ep_length=float(np.mean(raw_lengths)),
        mean_reward=float(np.mean(raw_rewards)),
        std_reward=float(np.std(raw_rewards)),
        wall_collisions_mean=mean_bumps,
        bump_rate=bump_rate,
        revealed_efficiency=revealed_eff,
        visited_efficiency=visited_eff,
        loop_steps_mean=mean_loops,
        loop_rate=loop_rate,
        monster_encounters_mean=mean_encounters,
        monster_encounter_rate=encounter_rate,
        coverage_milestones=coverage_milestones,
        reward_breakdown=breakdown_mean,
        end_reasons=end_reasons,
        fps=fps,
        latency_ms=latency_ms,
        raw_coverages=raw_coverages,
        raw_rewards=raw_rewards,
        raw_lengths=raw_lengths,
        raw_bumps=raw_bumps,
        raw_loops=raw_loops,
        raw_monster_encounters=raw_monster_encounters,
    )


def format_results_markdown(results: list[EvalResult]) -> str:
    """Format benchmark evaluation results into structured GitHub Flavored Markdown tables."""
    lines: list[str] = [
        "### 1. Primary Exploration & Survival Metrics",
        "",
        "| Agent / Policy | Difficulty | Mean Coverage (%) | Median (%) | Success Rate (≥95%) | Death Rate | Visited Path | Steps | Reward (Mean ± Std) | FPS |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| **{r.agent_name}** | {r.difficulty.upper()} | "
            f"**{r.coverage_mean:.2f} ± {r.coverage_std:.2f}** | "
            f"{r.coverage_median:.2f} | "
            f"**{r.completion_rate:.1f}%** | "
            f"{r.death_rate:.1f}% | "
            f"{r.visited_path_mean:.1f}% | "
            f"{r.mean_ep_length:.1f} | "
            f"{r.mean_reward:.2f} ± {r.std_reward:.2f} | "
            f"{r.fps:.0f} |"
        )

    lines.extend([
        "",
        "### 2. Navigation Efficiency, Wall Collisions & Threat Interactions",
        "",
        "| Agent / Policy | Wall Bumps (Ep) | Bump Rate (%) | Loop Rate (%) | Revealed / Step | Threat Encounters (Ep) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in results:
        lines.append(
            f"| **{r.agent_name}** | "
            f"{r.wall_collisions_mean:.1f} | "
            f"{r.bump_rate:.1f}% | "
            f"{r.loop_rate:.1f}% | "
            f"{r.revealed_efficiency:.2f} | "
            f"{r.monster_encounters_mean:.1f} ({r.monster_encounter_rate:.1f}%) |"
        )

    lines.extend([
        "",
        "### 3. Step Progression Milestones (Mean Coverage %)",
        "",
        "| Agent / Policy | Cov @ 25% Steps | Cov @ 50% Steps | Cov @ 75% Steps | Final Cov |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for r in results:
        m25 = r.coverage_milestones.get("25%", r.coverage_mean)
        m50 = r.coverage_milestones.get("50%", r.coverage_mean)
        m75 = r.coverage_milestones.get("75%", r.coverage_mean)
        m100 = r.coverage_milestones.get("100%", r.coverage_mean)
        lines.append(
            f"| **{r.agent_name}** | {m25:.1f}% | {m50:.1f}% | {m75:.1f}% | **{m100:.1f}%** |"
        )

    return "\n".join(lines)


def format_results_terminal(results: list[EvalResult]) -> str:
    """Format benchmark evaluation results into a structured ASCII table for terminal display."""
    header = (
        f"{'Agent / Model':<28} {'Diff':<8} {'Coverage (Mean ± Std)':<23} "
        f"{'Success':<9} {'Death':<7} {'Reward (Mean±Std)':<18} {'Bumps':<7} {'Loop%':<7} {'FPS':<6}"
    )
    divider = "-" * len(header)
    rows = [divider, header, divider]
    for r in results:
        cov_str = f"{r.coverage_mean:.1f}% ± {r.coverage_std:.1f}%"
        rew_str = f"{r.mean_reward:.2f}±{r.std_reward:.2f}"
        rows.append(
            f"{r.agent_name:<28} {r.difficulty.upper():<8} {cov_str:<23} "
            f"{r.completion_rate:.1f}%{'':<3} "
            f"{r.death_rate:.1f}%{'':<2} "
            f"{rew_str:<18} "
            f"{r.wall_collisions_mean:<7.1f} "
            f"{r.loop_rate:<6.1f}% "
            f"{r.fps:<6.0f}"
        )
    rows.append(divider)
    return "\n".join(rows)


def save_evaluation_report(
    results: list[EvalResult],
    output_dir: Path | str,
    prefix: str = "eval_results",
) -> tuple[Path, Path]:
    """Persist evaluation results as JSON and Markdown files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"

    # Save JSON with full raw data
    data = [r.to_dict() for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Save Markdown summary
    md_content = f"# Benchmark Evaluation Report: {prefix}\n\n"
    md_content += f"- **Evaluated Agents:** {len(results)}\n"
    md_content += f"- **Episodes per Agent:** {results[0].num_episodes if results else 0}\n\n"
    md_content += format_results_markdown(results)
    md_content += "\n\n### Episode Termination Breakdown\n\n"
    md_content += "| Agent | Coverage Reached | Timeout | Monster Death |\n"
    md_content += "| :--- | :---: | :---: | :---: |\n"
    for r in results:
        md_content += f"| **{r.agent_name}** | {r.end_reasons.get('coverage', 0)} | {r.end_reasons.get('timeout', 0)} | {r.end_reasons.get('death', 0)} |\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path
