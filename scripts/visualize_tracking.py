"""Visualize exploration coverage tracking and monster dynamics across episodes.

Generates visual diagnostic figures demonstrating:
1. Revealed area overlay on the dungeon map.
2. Agent trajectory (visited tiles path) for RL policies or baselines.
3. Discovered vs undiscovered Points of Interest (POIs).
4. Monster patrol trajectories and encounters.

Supports:
- RL Policies (PPO, RecurrentPPO, No-Memory) via checkpoints.
- Algorithmic Baselines: random, wall_follower, frontier (BFS), oracle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from explorer.agents import (
    BaseAgent,
    FrontierExplorerAgent,
    OraclePlannerAgent,
    RandomAgent,
    WallFollowerAgent,
)
from explorer.agents.rl_agent import RLAgent
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper
from explorer.eval.evaluator import detect_use_memory
from explorer.ui.app import find_default_checkpoint, resolve_checkpoint_path


def create_agent(
    agent_type: str,
    env: Any,
    seed: int,
    checkpoint: str | Path | None = None,
    deterministic: bool = True,
) -> tuple[BaseAgent, Path | None, bool]:
    """Create agent instance and determine memory requirements.

    Returns:
        (agent, resolved_checkpoint_path, use_memory)
    """
    agent_type = agent_type.lower()
    if agent_type == "rl":
        resolved_ckpt = resolve_checkpoint_path(checkpoint)
        if resolved_ckpt is None or not resolved_ckpt.exists():
            raise FileNotFoundError(
                f"Could not resolve a valid checkpoint for RL agent: '{checkpoint}'. "
                "Please specify an existing path or valid alias (e.g. 'medium', 'easy', 'nomem')."
            )
        rl_agent = RLAgent(checkpoint_path=resolved_ckpt, deterministic=deterministic)
        use_mem = detect_use_memory(rl_agent)
        return rl_agent, resolved_ckpt, use_mem

    resolved_ckpt = None
    use_mem = True

    if agent_type == "random":
        agent = RandomAgent(seed=seed)
    elif agent_type == "wall_follower":
        agent = WallFollowerAgent()
    elif agent_type == "frontier":
        agent = FrontierExplorerAgent(env)
    elif agent_type == "oracle":
        agent = OraclePlannerAgent(env)
    else:
        raise ValueError(
            f"Unsupported agent type '{agent_type}'. "
            "Supported options: 'rl', 'random', 'wall_follower', 'frontier', 'oracle'."
        )

    return agent, resolved_ckpt, use_mem


def run_and_render_rollout(
    difficulty: str = "easy",
    seed: int = 42,
    num_steps: int = 60,
    agent_type: str = "random",
    checkpoint: str | Path | None = None,
    deterministic: bool = True,
    output_path: str = "assets/maps/exploration_tracking_demo.png",
) -> None:
    """Run an exploration episode and render side-by-side diagnostic plots."""
    # Pre-resolve agent to configure observation wrapper correctly
    temp_env = ExploreEnv(difficulty=difficulty, render_mode="rgb_array")
    agent, resolved_ckpt, use_memory = create_agent(
        agent_type=agent_type,
        env=temp_env,
        seed=seed,
        checkpoint=checkpoint,
        deterministic=deterministic,
    )
    temp_env.close()

    raw_env = ExploreEnv(difficulty=difficulty, render_mode="rgb_array")
    env = ObservationMemoryWrapper(
        raw_env,
        use_memory=use_memory,
        include_step_ratio=True,
    )

    # Re-link env if agent needs unwrapped or wrapped env (e.g. frontier/oracle)
    if hasattr(agent, "env"):
        agent.env = env

    obs, info = env.reset(seed=seed)
    agent.reset()

    agent_trajectory = [tuple(raw_env.agent_pos)]
    monster_trajectories: dict[int, list[tuple[int, int]]] = {
        m.id: [m.pos] for m in raw_env.monsters
    }

    # Step simulation loop
    term = False
    trunc = False
    for step_idx in range(num_steps):
        if agent_type == "random":
            action = raw_env.action_space.sample()
        else:
            action = agent.act(obs, info)

        obs, _r, term, trunc, info = env.step(action)
        agent_trajectory.append(tuple(raw_env.agent_pos))
        for m in raw_env.monsters:
            monster_trajectories[m.id].append(m.pos)

        if term or trunc:
            break

    # Generate environment base frame
    base_frame = raw_env.get_frame(highlight=False, tile_size=32, agent_pov=False)
    tile_size = 32

    # Create composite overlay image
    overlay_frame = base_frame.copy().astype(float)

    # Darken unrevealed walkable tiles to visualize fog of war
    for x, y in raw_env.walkable_tiles:
        if (x, y) not in raw_env.tracker.revealed_tiles:
            x1, y1 = x * tile_size, y * tile_size
            x2, y2 = (x + 1) * tile_size, (y + 1) * tile_size
            overlay_frame[y1:y2, x1:x2] = overlay_frame[y1:y2, x1:x2] * 0.25

    # Highlight revealed tiles with a soft golden hue
    for x, y in raw_env.tracker.revealed_tiles:
        if (x, y) in raw_env.walkable_tiles:
            x1, y1 = x * tile_size, y * tile_size
            x2, y2 = (x + 1) * tile_size, (y + 1) * tile_size
            overlay_frame[y1:y2, x1:x2, 0] = np.clip(overlay_frame[y1:y2, x1:x2, 0] * 1.15 + 20, 0, 255)
            overlay_frame[y1:y2, x1:x2, 1] = np.clip(overlay_frame[y1:y2, x1:x2, 1] * 1.15 + 20, 0, 255)

    overlay_frame = overlay_frame.astype(np.uint8)

    # Build agent title label
    if agent_type == "rl" and resolved_ckpt is not None:
        model_name = resolved_ckpt.parent.name if resolved_ckpt.parent.name != "checkpoints" else resolved_ckpt.name
        agent_label = f"RL: {model_name}"
    else:
        agent_label = agent_type.replace("_", " ").title()

    # Plot 2-panel diagnostic figure
    _fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: Full Ground Truth Map with Monster Path
    ax1 = axes[0]
    ax1.imshow(base_frame)
    ax1.set_title(
        f"Ground Truth Map (Seed {seed})\nWalkable: {len(raw_env.walkable_tiles)} | POIs: {len(raw_env.poi_positions)} | Monsters: {len(raw_env.monsters)}",
        fontsize=12,
        fontweight="bold",
    )
    # Overlay monster paths
    colors = ["red", "magenta", "orange"]
    for m_id, path in monster_trajectories.items():
        px = [p[0] * tile_size + tile_size // 2 for p in path]
        py = [p[1] * tile_size + tile_size // 2 for p in path]
        ax1.plot(px, py, color=colors[m_id % len(colors)], linewidth=2.5, linestyle="--", label=f"Monster {m_id} Path")
        ax1.scatter([px[0]], [py[0]], color="cyan", s=70, marker="o", label="Monster Spawn" if m_id == 0 else None)
        ax1.scatter([px[-1]], [py[-1]], color=colors[m_id % len(colors)], s=120, marker="X", label=f"Monster {m_id} Final")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.axis("off")

    # Panel 2: Exploration State & Fog of War
    ax2 = axes[1]
    ax2.imshow(overlay_frame)
    rev_pct = raw_env.tracker.revealed_coverage * 100
    vis_pct = raw_env.tracker.visited_coverage * 100
    pois_found = f"{len(raw_env.tracker.discovered_pois)}/{len(raw_env.poi_positions)}"

    status_tag = ""
    if term:
        # Check termination reason
        if raw_env.tracker.revealed_coverage >= 0.95:
            status_tag = " [TARGET REACHED]"
        elif any(tuple(raw_env.agent_pos) == m.pos for m in raw_env.monsters):
            status_tag = " [DEATH BY MONSTER]"
        else:
            status_tag = " [TERMINATED]"
    elif trunc:
        status_tag = " [TIMEOUT]"

    ax2.set_title(
        f"Exploration Progress - {agent_label} (Step {raw_env.step_count}){status_tag}\n"
        f"Revealed Cov: {rev_pct:.1f}% | Visited Cov: {vis_pct:.1f}% | POIs: {pois_found}",
        fontsize=12,
        fontweight="bold",
    )

    # Plot agent trajectory
    ax_coords = [p[0] * tile_size + tile_size // 2 for p in agent_trajectory]
    ay_coords = [p[1] * tile_size + tile_size // 2 for p in agent_trajectory]
    ax2.plot(ax_coords, ay_coords, color="lime", linewidth=2.0, alpha=0.85, label=f"Agent Path ({agent_label})")
    ax2.scatter([ax_coords[0]], [ay_coords[0]], color="yellow", s=100, marker="*", label="Agent Start")
    ax2.scatter([ax_coords[-1]], [ay_coords[-1]], color="lime", s=100, marker="o", label="Agent Current")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    raw_env.close()
    print(f"[Success] Saved exploration tracking visualization ({agent_label}) to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize exploration and tracking telemetry")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--agent",
        choices=["rl", "random", "wall_follower", "frontier", "oracle"],
        default="random",
        help="Agent type to run (default: random).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path or alias to trained RL model checkpoint (e.g. 'medium', 'easy', 'nomem', or .zip path).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling for RL policy (default: deterministic).",
    )
    parser.add_argument("--output", type=str, default="assets/maps/exploration_tracking_demo.png")
    args = parser.parse_args()

    # If checkpoint is passed but agent is default 'random', switch agent to 'rl'
    agent_type = args.agent
    if args.checkpoint is not None and agent_type == "random":
        agent_type = "rl"

    run_and_render_rollout(
        difficulty=args.difficulty,
        seed=args.seed,
        num_steps=args.steps,
        agent_type=agent_type,
        checkpoint=args.checkpoint,
        deterministic=not args.stochastic,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
