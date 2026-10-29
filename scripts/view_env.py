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


class ExploreManualControl(ManualControl):
    """Clean manual controller inheriting directly from MiniGrid's ManualControl.

    Adds support for WASD keys, dynamic difficulty switching (1, 2, 3), and
    randomized seed stepping (N).
    """

    def __init__(
        self,
        env: ExploreEnv,
        seed: int = 42,
        difficulty: str = "easy",
    ):
        super().__init__(env, seed=seed)
        self.difficulty = difficulty
        self.rng = np.random.default_rng()

    def switch_difficulty(self, difficulty: str) -> None:
        """Switch environment difficulty and regenerate map."""
        print(f"\n[Difficulty Switched] -> {difficulty.upper()} (Seed: {self.seed})")
        self.difficulty = difficulty
        self.env.close()
        self.env = ExploreEnv(difficulty=difficulty, render_mode="human")
        self.reset(self.seed)

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
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MINIGRID EXPLORER - MANUAL CONTROL")
    print("=" * 60)
    print("  Controls:")
    print("    - [W] or [Up Arrow]    : Move Forward")
    print("    - [A] or [Left Arrow]  : Turn Left")
    print("    - [D] or [Right Arrow] : Turn Right")
    print("    - [1], [2], [3]        : Switch Difficulty (Easy/Medium/Hard)")
    print("    - [N]                  : Generate Next Random Seed")
    print("    - [Backspace]          : Reset Current Episode")
    print("    - [ESC]                : Exit")
    print("=" * 60 + "\n")

    env = ExploreEnv(difficulty=args.difficulty, render_mode="human")
    controller = ExploreManualControl(env, seed=args.seed, difficulty=args.difficulty)
    controller.start()


if __name__ == "__main__":
    main()