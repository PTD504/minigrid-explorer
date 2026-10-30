#!/usr/bin/env python3
"""Visualize exploration coverage tracking and monster dynamics across episodes.

Generates visual diagnostic figures demonstrating:
1. Revealed area overlay on the dungeon map.
2. Agent trajectory (visited tiles path).
3. Discovered vs undiscovered Points of Interest (POIs).
4. Monster patrol trajectories and encounters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from explorer.envs.explore_env import ExploreEnv


def run_and_render_rollout(
    difficulty: str = "easy",
    seed: int = 42,
    num_steps: int = 60,
    output_path: str = "assets/maps/exploration_tracking_demo.png",
) -> None:
    """Run an exploration episode and render side-by-side diagnostic plots."""
    env = ExploreEnv(difficulty=difficulty, render_mode="rgb_array")
    _obs, _initial_info = env.reset(seed=seed)

    agent_trajectory = [tuple(env.agent_pos)]
    monster_trajectories: dict[int, list[tuple[int, int]]] = {
        m.id: [m.pos] for m in env.monsters
    }

    # Step random exploratory actions
    for _ in range(num_steps):
        action = env.action_space.sample()
        _obs, _r, term, trunc, _info = env.step(action)
        agent_trajectory.append(tuple(env.agent_pos))
        for m in env.monsters:
            monster_trajectories[m.id].append(m.pos)
        if term or trunc:
            break

    # Generate environment base frame
    base_frame = env.get_frame(highlight=False, tile_size=32, agent_pov=False)
    tile_size = 32

    # Create composite overlay image
    overlay_frame = base_frame.copy().astype(float)

    # Darken unrevealed walkable tiles to visualize fog of war
    for x, y in env.walkable_tiles:
        if (x, y) not in env.tracker.revealed_tiles:
            x1, y1 = x * tile_size, y * tile_size
            x2, y2 = (x + 1) * tile_size, (y + 1) * tile_size
            overlay_frame[y1:y2, x1:x2] = overlay_frame[y1:y2, x1:x2] * 0.25

    # Highlight revealed tiles with a soft golden hue
    for x, y in env.tracker.revealed_tiles:
        if (x, y) in env.walkable_tiles:
            x1, y1 = x * tile_size, y * tile_size
            x2, y2 = (x + 1) * tile_size, (y + 1) * tile_size
            overlay_frame[y1:y2, x1:x2, 0] = np.clip(overlay_frame[y1:y2, x1:x2, 0] * 1.15 + 20, 0, 255)
            overlay_frame[y1:y2, x1:x2, 1] = np.clip(overlay_frame[y1:y2, x1:x2, 1] * 1.15 + 20, 0, 255)

    overlay_frame = overlay_frame.astype(np.uint8)

    # Plot 2-panel diagnostic figure
    _fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: Full Ground Truth Map with Monster Path
    ax1 = axes[0]
    ax1.imshow(base_frame)
    ax1.set_title(
        f"Ground Truth Map (Seed {seed})\nWalkable: {len(env.walkable_tiles)} | POIs: {len(env.poi_positions)} | Monsters: {len(env.monsters)}",
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
    rev_pct = env.tracker.revealed_coverage * 100
    vis_pct = env.tracker.visited_coverage * 100
    pois_found = f"{len(env.tracker.discovered_pois)}/{len(env.poi_positions)}"
    ax2.set_title(
        f"Exploration Progress (Step {env.step_count})\nRevealed Cov: {rev_pct:.1f}% | Visited Cov: {vis_pct:.1f}% | POIs: {pois_found}",
        fontsize=12,
        fontweight="bold",
    )

    # Plot agent trajectory
    ax_coords = [p[0] * tile_size + tile_size // 2 for p in agent_trajectory]
    ay_coords = [p[1] * tile_size + tile_size // 2 for p in agent_trajectory]
    ax2.plot(ax_coords, ay_coords, color="lime", linewidth=2.0, alpha=0.85, label="Agent Path")
    ax2.scatter([ax_coords[0]], [ay_coords[0]], color="yellow", s=100, marker="*", label="Agent Start")
    ax2.scatter([ax_coords[-1]], [ay_coords[-1]], color="lime", s=100, marker="o", label="Agent Current")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Success] Saved exploration tracking visualization to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize exploration and tracking telemetry")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output", type=str, default="assets/maps/exploration_tracking_demo.png")
    args = parser.parse_args()

    run_and_render_rollout(
        difficulty=args.difficulty,
        seed=args.seed,
        num_steps=args.steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
