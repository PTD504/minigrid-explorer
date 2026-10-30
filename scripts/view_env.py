#!/usr/bin/env python3
"""Interactive manual player for MiniGrid Explorer.

Inherits directly from MiniGrid's official ManualControl class, adding WASD navigation,
dynamic seed randomization (N), and difficulty switching (1, 2, 3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from minigrid.manual_control import ManualControl

from explorer.envs.explore_env import ExploreEnv
from explorer.utils.config import get_env_config


class ExploreManualControl(ManualControl):
    """Clean manual controller inheriting directly from MiniGrid's ManualControl.

    Adds support for WASD keys, dynamic difficulty switching (1, 2, 3),
    randomized seed stepping (N), and custom monster parameters.
    """

    def __init__(
        self,
        env: ExploreEnv,
        seed: int = 42,
        difficulty: str = "easy",
        custom_config: dict | None = None,
    ):
        super().__init__(env, seed=seed)
        self.difficulty = difficulty
        self.custom_config = custom_config or {}
        self.rng = np.random.default_rng()

    def switch_difficulty(self, difficulty: str) -> None:
        """Switch environment difficulty and regenerate map."""
        print(f"\n[Difficulty Switched] -> {difficulty.upper()} (Seed: {self.seed})")
        self.difficulty = difficulty
        self.env.close()
        config = get_env_config(difficulty)
        config.update(self.custom_config)
        self.env = ExploreEnv(difficulty=difficulty, config=config, render_mode="human")
        self.reset(self.seed)

    def step(self, action: int) -> None:
        """Execute action, inspect step feedback, and handle death/termination."""
        _obs, _reward, terminated, truncated, info = self.env.step(action)
        monsters_str = ", ".join(f"M{m.id}@{m.pos}[{m.state}]" for m in self.env.monsters)
        rev_cov = info.get("revealed_coverage", 0.0) * 100
        pois_found = f"{info.get('discovered_pois', 0)}/{info.get('total_pois', 0)}"
        status_msg = f"Step {self.env.step_count} | Agent@{self.env.agent_pos} | Cov: {rev_cov:.1f}% | POIs: {pois_found}"
        if monsters_str:
            status_msg += f" | {monsters_str}"
        if info.get("bump"):
            status_msg += " | [BUMP WALL]"
        print(status_msg)

        if terminated:
            end_reason = info.get("end_reason")
            if end_reason == "death":
                print("\n" + "!" * 50)
                print("  [GAME OVER] You were slain by a monster!")
                print("!" * 50 + "\n")
            else:
                print(f"\n[Episode Terminated] Reason: {end_reason}")
            self.reset(self.seed)
        elif truncated:
            print("\n[Episode Truncated] Time limit reached.")
            self.reset(self.seed)
        else:
            self.env.render()


    def key_handler(self, event) -> None:
        key: str = event.key


        # WASD navigation bindings
        if key == "w":
            self.step(self.env.actions.forward)
            return
        elif key == "a":
            self.step(self.env.actions.left)
            return
        elif key == "d":
            self.step(self.env.actions.right)
            return

        # Seed randomization
        elif key == "n":
            new_seed = int(self.rng.integers(1, 100000))
            print(f"\n[Random Seed Generated] -> Seed {new_seed}")
            self.seed = new_seed
            self.reset(self.seed)
            return

        # Fast difficulty switching
        elif key == "1" and self.difficulty != "easy":
            self.switch_difficulty("easy")
            return
        elif key == "2" and self.difficulty != "medium":
            self.switch_difficulty("medium")
            return
        elif key == "3" and self.difficulty != "hard":
            self.switch_difficulty("hard")
            return

        # Handle clean exit on escape key
        if key == "escape":
            self.closed = True
            self.env.close()
            return

        # Delegate standard controls (arrow keys, backspace) to parent
        super().key_handler(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual control player for MiniGrid Explorer")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy", help="Map difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Map seed")
    parser.add_argument("--num-monsters", type=int, default=None, help="Override number of monsters (e.g. 1, 2, 3)")
    parser.add_argument("--monster-period", type=int, default=None, help="Override monster movement speed (moves every N steps)")
    parser.add_argument("--monster-detect-radius", type=int, default=None, help="Override monster chase detection radius")
    args = parser.parse_args()

    custom_config: dict[str, int] = {}
    if args.num_monsters is not None:
        custom_config["num_monsters"] = args.num_monsters
    if args.monster_period is not None:
        custom_config["monster_period"] = args.monster_period
    if args.monster_detect_radius is not None:
        custom_config["monster_detect_radius"] = args.monster_detect_radius

    config = get_env_config(args.difficulty)
    config.update(custom_config)

    print("\n" + "=" * 60)
    print("  MINIGRID EXPLORER - MANUAL CONTROL")
    print("=" * 60)
    print(f"  Configuration : {args.difficulty.upper()} (Seed: {args.seed})")
    print(f"  Monsters      : {config.get('num_monsters', 1)} | Period: {config.get('monster_period', 3)} | Detect Radius: {config.get('monster_detect_radius', 5)}")
    print("  Controls:")
    print("    - [W] or [Up Arrow]    : Move Forward")
    print("    - [A] or [Left Arrow]  : Turn Left")
    print("    - [D] or [Right Arrow] : Turn Right")
    print("    - [1], [2], [3]        : Switch Difficulty (Easy/Medium/Hard)")
    print("    - [N]                  : Generate Next Random Seed")
    print("    - [Backspace]          : Reset Current Episode")
    print("    - [ESC]                : Exit")
    print("=" * 60 + "\n")

    env = ExploreEnv(difficulty=args.difficulty, config=config, render_mode="human")
    controller = ExploreManualControl(
        env,
        seed=args.seed,
        difficulty=args.difficulty,
        custom_config=custom_config,
    )
    controller.start()



if __name__ == "__main__":
    main()