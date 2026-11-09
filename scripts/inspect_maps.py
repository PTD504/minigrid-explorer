"""Batch map visualizer to inspect and compare procedural map generation across seeds and difficulties."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from explorer.envs.explore_env import ExploreEnv


def inspect_seeds(difficulty: str = "easy", seeds: list[int] | None = None, output_path: str | None = None, show: bool = False) -> None:
    """Generate a multi-panel grid comparing different seeds of the same difficulty level."""
    if seeds is None:
        seeds = [7, 42, 101, 2026]

    num_maps = len(seeds)
    cols = min(4, num_maps)
    rows = (num_maps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_maps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, seed in enumerate(seeds):
        env = ExploreEnv(difficulty=difficulty, render_mode="rgb_array")
        env.reset(seed=seed)
        frame = env.get_frame(highlight=True, tile_size=32, agent_pov=False)

        ax = axes[idx]
        ax.imshow(frame)
        num_rooms = len(env.map_metadata.rooms) if env.map_metadata else 0
        walkable = len(env.walkable_tiles)
        pois = len(env.poi_positions)
        monsters = len(env.monsters)
        ax.set_title(f"Seed {seed} | Rooms: {num_rooms} | POIs: {pois} | Monsters: {monsters} | Walkable: {walkable}", fontsize=9)
        ax.axis("off")
        env.close()

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    # Leave headroom at the top for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle(
        f"Procedural Map Generation Variety: [{difficulty.upper()}] Across Multiple Seeds",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Success] Saved seed comparison figure to: {output_path}")

    if show:
        plt.show()
    plt.close()


def compare_difficulties(seed: int = 42, output_path: str | None = None, show: bool = False) -> None:
    """Generate a side-by-side comparison across Easy, Medium, and Hard configurations with the same seed."""
    difficulties = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, diff in enumerate(difficulties):
        env = ExploreEnv(difficulty=diff, render_mode="rgb_array")
        env.reset(seed=seed)
        frame = env.get_frame(highlight=True, tile_size=32, agent_pov=False)

        ax = axes[idx]
        ax.imshow(frame)
        num_rooms = len(env.map_metadata.rooms) if env.map_metadata else 0
        walkable = len(env.walkable_tiles)
        pois = len(env.poi_positions)
        monsters = len(env.monsters)
        ax.set_title(
            f"{diff.upper()} ({env.width}x{env.height})\nRooms: {num_rooms} | POIs: {pois} | Monsters: {monsters} | Walkable: {walkable}",
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")
        env.close()

    # Leave headroom at the top for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle(
        f"MiniGrid Explorer: Difficulty Scaling Comparison (Seed {seed})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Success] Saved difficulty comparison figure to: {output_path}")

    if show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect procedural map generation across seeds and difficulties")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy", help="Difficulty level")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 42, 123, 777], help="List of seeds to compare")
    parser.add_argument("--compare-difficulties", action="store_true", help="Compare easy, medium, and hard side-by-side")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for difficulty comparison")
    parser.add_argument("--output", type=str, default="results/map_inspection.png", help="Path to save output figure")
    parser.add_argument("--show", action="store_true", help="Display window directly")
    args = parser.parse_args()

    if args.compare_difficulties:
        compare_difficulties(seed=args.seed, output_path=args.output, show=args.show)
    else:
        inspect_seeds(difficulty=args.difficulty, seeds=args.seeds, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
