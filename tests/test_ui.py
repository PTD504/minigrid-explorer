from __future__ import annotations

import os
from pathlib import Path

# Force dummy video driver for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

from explorer.ui.app import ExplorerUIApp


def test_ui_app_initialization():
    """Verify ExplorerUIApp initializes correctly with all panels and buttons."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="frontier", headless=True)
    assert app.raw_env is not None
    assert app.raw_env.width == 17
    assert app.agent is not None
    assert len(app.buttons) > 10
    assert len(app.trajectory) == 1


def test_ui_app_step():
    """Verify simulation steps update telemetry and trajectory."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="frontier", headless=True)
    initial_traj_len = len(app.trajectory)

    app.step()

    assert len(app.trajectory) == initial_traj_len + 1
    assert "revealed_coverage" in app.info
    assert isinstance(app.last_reward, float)


def test_ui_app_difficulty_switch():
    """Verify switching difficulty resizes grid layout accordingly."""
    app = ExplorerUIApp(difficulty="easy", seed=42, headless=True)
    assert app.raw_env.width == 17

    app.reset_env(difficulty="medium")
    assert app.raw_env.width == 25
    assert app.difficulty == "medium"

    app.reset_env(difficulty="hard")
    assert app.raw_env.width == 33
    assert app.difficulty == "hard"


def test_ui_app_agent_switch():
    """Verify switching agent updates active agent controller."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="manual", headless=True)
    assert app.agent is None  # Manual mode has no automated agent

    for agent_key in ["frontier", "wall_follower", "random", "oracle"]:
        app.agent_type = agent_key
        app._init_agent()
        assert app.agent is not None


def test_ui_app_button_clicks():
    """Verify clicking UI buttons triggers appropriate state changes."""
    app = ExplorerUIApp(difficulty="easy", seed=100, agent_type="frontier", headless=True)

    # Click next seed
    app._handle_button_click("seed_next")
    assert app.seed == 101

    # Click prev seed
    app._handle_button_click("seed_prev")
    assert app.seed == 100

    # Click difficulty
    app._handle_button_click("diff_medium")
    assert app.difficulty == "medium"

    # Click play / pause toggle
    assert app.is_playing is False
    app._handle_button_click("ctrl_play")
    assert app.is_playing is True
    app._handle_button_click("ctrl_play")
    assert app.is_playing is False


def test_ui_app_render_and_screenshot(tmp_path: Path):
    """Verify render pipeline and screenshot export."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="frontier", headless=True)

    for _ in range(5):
        app.step()

    app.render()
    out_file = tmp_path / "test_ui.png"
    app.save_screenshot(str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 1000

