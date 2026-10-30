#!/usr/bin/env python3
"""Diagnostic script to inspect and visualize ObservationMemoryWrapper in action.

Produces:
1. Terminal ASCII rendering of local view (7x7) and egocentric memory crop (15x15).
2. Diagnostic image saved to disk comparing Global World, Local View, and Memory Channels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import (
    MEM_KNOWN_POI,
    MEM_KNOWN_WALL,
    MEM_REVEALED,
    MEM_VISITED,
    VIEW_EMPTY,
    VIEW_MONSTER,
    VIEW_POI,
    VIEW_UNSEEN,
    VIEW_WALL,
    ObservationMemoryWrapper,
)

ACTION_NAMES = {
    0: "Turn Left",
    1: "Turn Right",
    2: "Move Forward",
}


def render_ascii_view(view: np.ndarray) -> str:
    """Render 7x7 local view as ASCII characters.

    Symbols:
    '?' = unseen
    '.' = empty floor
    '#' = wall
    'P' = POI
    'M' = monster
    'A' = agent (at 3, 6)
    """
    h, w, _ = view.shape
    lines = []
    lines.append("   " + " ".join(f"{x}" for x in range(w)))
    for y in range(h):
        row = [f"{y:2d} "]
        for x in range(w):
            if x == 3 and y == 6:
                row.append("▲")  # Agent facing forward
            elif view[x, y, VIEW_WALL] > 0.5:
                row.append("#")
            elif view[x, y, VIEW_POI] > 0.5:
                row.append("P")
            elif view[x, y, VIEW_MONSTER] > 0.5:
                row.append("M")
            elif view[x, y, VIEW_EMPTY] > 0.5:
                row.append(".")
            elif view[x, y, VIEW_UNSEEN] > 0.5:
                row.append("·")
            else:
                row.append("?")
        lines.append(" ".join(row))
    return "\n".join(lines)


def render_ascii_memory(memory: np.ndarray, center: int = 7) -> str:
    """Render 15x15 egocentric memory crop as ASCII characters.

    Symbols:
    '▲' = Agent (at center)
    '#' = known wall
    'P' = known POI
    '*' = visited tile
    '.' = revealed tile (seen but not visited)
    ' ' = unknown / unrevealed
    """
    h, w, _ = memory.shape
    lines = []
    lines.append("   " + " ".join(f"{x%10}" for x in range(w)))
    for v in range(h):
        row = [f"{v:2d} "]
        for u in range(w):
            if u == center and v == center:
                row.append("▲")
            elif memory[u, v, MEM_KNOWN_WALL] > 0.5:
                row.append("#")
            elif memory[u, v, MEM_KNOWN_POI] > 0.5:
                row.append("P")
            elif memory[u, v, MEM_VISITED] > 0.5:
                row.append("*")
            elif memory[u, v, MEM_REVEALED] > 0.5:
                row.append(".")
            else:
                row.append(" ")
        lines.append(" ".join(row))
    return "\n".join(lines)


def run_visualization(
    difficulty: str = "easy",
    seed: int = 42,
    num_steps: int = 15,
    save_path: str = "assets/demo/observation_wrapper_demo.png",
) -> None:
    """Run an interactive rollout and save a multi-panel visual comparison."""
    base_env = ExploreEnv(difficulty=difficulty, render_mode="rgb_array")
    wrapped_env = ObservationMemoryWrapper(base_env, crop_size=15)

    obs, info = wrapped_env.reset(seed=seed)
    agent_trajectory = [tuple(base_env.agent_pos)]

    print("=" * 70)
    print("MINIGRID EXPLORER - OBSERVATION MEMORY WRAPPER DEMO")
    print("=" * 70)
    print(f"Difficulty: {difficulty.upper()} | Seed: {seed}")
    print(f"Observation Space: {wrapped_env.observation_space}")
    print(f"Keys in obs: {list(obs.keys())}")
    print(f"View shape: {obs['view'].shape} (dtype: {obs['view'].dtype})")
    print(f"Memory shape: {obs['memory'].shape} (dtype: {obs['memory'].dtype})")
    print(f"Step ratio shape: {obs['step_ratio'].shape} (value: {obs['step_ratio'][0]:.3f})")
    print("-" * 70)

    # Perform rollout with forward-biased exploratory actions to avoid bumping into walls
    for step_idx in range(1, num_steps + 1):
        fwd_cell = base_env.grid.get(*base_env.front_pos)
        can_forward = fwd_cell is None or fwd_cell.can_overlap()
        if can_forward and np.random.rand() < 0.75:
            action = 2  # Move forward
        else:
            action = int(np.random.choice([0, 1]))  # Turn left or right

        obs, _reward, terminated, truncated, info = wrapped_env.step(action)
        agent_trajectory.append(tuple(base_env.agent_pos))
        action_name = ACTION_NAMES.get(action, f"Action {action}")

        if step_idx in (1, num_steps // 2, num_steps) or info.get("delta_revealed", 0) > 0:
            print(f"\n[Step {step_idx:02d}] Action: {action_name:<12} | Step Ratio: {obs['step_ratio'][0]:.2f}")
            print(f"Agent Pos: {base_env.agent_pos} | Dir: {base_env.agent_dir} (0:R, 1:D, 2:L, 3:U)")
            print(f"Revealed Tiles: {len(base_env.tracker.revealed_tiles)} | Visited: {len(base_env.tracker.visited_tiles)}")
            print("\n--- Local View (7x7 One-Hot) ---")
            print(render_ascii_view(obs["view"]))
            print("\n--- Egocentric Memory Crop (15x15 Centered at ▲) ---")
            print(render_ascii_memory(obs["memory"]))

        if terminated or truncated:
            print(f"\nEpisode finished early at step {step_idx}: {info.get('end_reason')}")
            break

    # Save visual comparison figure with Matplotlib
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1], height_ratios=[1, 1])

    # 1. Full World Map (Ground Truth)
    ax_world = fig.add_subplot(gs[:, 0])
    full_frame = base_env.get_frame(highlight=True, tile_size=32, agent_pov=False)
    ax_world.imshow(full_frame)

    # Overlay agent trajectory on world map
    tile_size = 32
    if len(agent_trajectory) > 1:
        tx = [p[0] * tile_size + tile_size // 2 for p in agent_trajectory]
        ty = [p[1] * tile_size + tile_size // 2 for p in agent_trajectory]
        ax_world.plot(tx, ty, color="#00FF66", linewidth=2.5, linestyle="-", alpha=0.85, label="Agent Path")
        ax_world.scatter([tx[0]], [ty[0]], color="cyan", s=70, marker="o", edgecolors="black", label="Start Pos")
        ax_world.legend(loc="upper right", fontsize=8)

    ax_world.set_title(
        f"1. Global World Map (Ground Truth)\nAgent: {base_env.agent_pos} facing {base_env.agent_dir}\n"
        f"Walkable: {len(base_env.walkable_tiles)} | Visited: {len(base_env.tracker.visited_tiles)} | Revealed: {len(base_env.tracker.revealed_tiles)}",
        fontsize=11,
        fontweight="bold",
    )
    ax_world.axis("off")

    # 2. Local 7x7 View (Color-coded)
    ax_view = fig.add_subplot(gs[0, 1])
    view = obs["view"]
    # Synthesize RGB view: unseen=gray, empty=white, wall=dark gray, poi=purple, monster=red
    view_rgb = np.full((7, 7, 3), 0.3)  # unseen = dark gray
    for vx in range(7):
        for vy in range(7):
            if view[vx, vy, VIEW_EMPTY] > 0.5:
                view_rgb[vy, vx] = [0.9, 0.9, 0.9]  # empty floor
            elif view[vx, vy, VIEW_WALL] > 0.5:
                view_rgb[vy, vx] = [0.15, 0.2, 0.25]  # wall
            elif view[vx, vy, VIEW_POI] > 0.5:
                view_rgb[vy, vx] = [0.7, 0.1, 0.9]  # POI purple
            elif view[vx, vy, VIEW_MONSTER] > 0.5:
                view_rgb[vy, vx] = [0.9, 0.1, 0.1]  # Monster red

    ax_view.imshow(view_rgb)
    # Mark agent position at (3, 6)
    ax_view.scatter([3], [6], marker="^", color="blue", s=180, edgecolors="white", label="Agent")
    ax_view.set_title("2. Local View (7x7 One-Hot)\nAgent at bottom facing UP", fontsize=11, fontweight="bold")
    ax_view.set_xticks(range(7))
    ax_view.set_yticks(range(7))
    ax_view.legend(loc="lower right", fontsize=8)

    # 3. Egocentric Memory Crop (15x15 Composite)
    ax_mem = fig.add_subplot(gs[0, 2])
    mem = obs["memory"]
    mem_rgb = np.zeros((15, 15, 3), dtype=float)  # unknown = black
    for u in range(15):
        for v in range(15):
            if mem[u, v, MEM_REVEALED] > 0.5:
                mem_rgb[v, u] = [0.5, 0.6, 0.8]  # revealed floor (soft blue)
            if mem[u, v, MEM_VISITED] > 0.5:
                mem_rgb[v, u] = [0.2, 0.8, 0.3]  # visited path (green)
            if mem[u, v, MEM_KNOWN_WALL] > 0.5:
                mem_rgb[v, u] = [0.2, 0.2, 0.2]  # known wall
            if mem[u, v, MEM_KNOWN_POI] > 0.5:
                mem_rgb[v, u] = [0.8, 0.1, 0.9]  # known POI (purple)

    ax_mem.imshow(mem_rgb)
    ax_mem.scatter([7], [7], marker="^", color="red", s=180, edgecolors="white", label="Agent (Center)")
    ax_mem.set_title("3. Memory Crop (15x15 Egocentric)\nAgent at CENTER (7, 7)", fontsize=11, fontweight="bold")
    ax_mem.set_xticks([0, 7, 14])
    ax_mem.set_yticks([0, 7, 14])
    ax_mem.legend(loc="lower right", fontsize=8)

    # 4. Individual Memory Channels (4 sub-plots)
    channels_info = [
        ("Ch 0: Revealed Map", mem[:, :, MEM_REVEALED].T, "Blues", gs[0, 3]),
        ("Ch 1: Visited Path", mem[:, :, MEM_VISITED].T, "Greens", gs[1, 1]),
        ("Ch 2: Known Walls", mem[:, :, MEM_KNOWN_WALL].T, "gray", gs[1, 2]),
        ("Ch 3: Known POIs", mem[:, :, MEM_KNOWN_POI].T, "Purples", gs[1, 3]),
    ]

    for title, data, cmap, pos in channels_info:
        ax = fig.add_subplot(pos)
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.scatter([7], [7], marker="^", color="red", s=40)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"ObservationMemoryWrapper Inspection (Difficulty: {difficulty.capitalize()} | Seed {seed} | Step {step_idx})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 70)
    print(f"SUCCESS: Visual diagnostic saved to: {out_file.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize ObservationMemoryWrapper.")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--save", type=str, default="assets/demo/observation_wrapper_demo.png")
    args = parser.parse_args()

    run_visualization(
        difficulty=args.difficulty,
        seed=args.seed,
        num_steps=args.steps,
        save_path=args.save,
    )
