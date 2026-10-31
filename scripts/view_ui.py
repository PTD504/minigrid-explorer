#!/usr/bin/env python3
"""Interactive GUI launcher for MiniGrid Explorer Research UI.

Usage:
    uv run python scripts/view_ui.py
    uv run python scripts/view_ui.py --difficulty medium --agent frontier
    uv run python scripts/view_ui.py --difficulty hard --seed 42 --agent oracle
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from explorer.ui.app import main

if __name__ == "__main__":
    main()

