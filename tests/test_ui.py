from __future__ import annotations

import os
from pathlib import Path

# Force dummy video driver for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

from stable_baselines3 import PPO

from explorer.agents.rl_agent import RLAgent
from explorer.train import make_vec_env_dungeon
from explorer.ui.app import ExplorerUIApp, resolve_checkpoint_path


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


def test_ui_app_rl_agent_with_checkpoint(tmp_path: Path):
    """Verify ExplorerUIApp loads trained RL model checkpoint and auto-steps."""
    # Create a small dummy PPO model and save to disk
    train_venv = make_vec_env_dungeon(difficulty="easy", n_envs=1, seed=42, use_memory=True)
    model = PPO("MultiInputPolicy", train_venv, n_steps=8, batch_size=8, n_epochs=1, verbose=0)
    model.learn(total_timesteps=8)
    ckpt_path = tmp_path / "test_model.zip"
    model.save(str(ckpt_path))
    train_venv.close()

    app = ExplorerUIApp(
        difficulty="easy",
        seed=42,
        checkpoint_path=ckpt_path,
        headless=True,
    )

    assert app.agent_type == "rl"
    assert isinstance(app.agent, RLAgent)

    # Step simulation with RL agent
    initial_traj_len = len(app.trajectory)
    app.step()
    assert len(app.trajectory) == initial_traj_len + 1

    # Render scene
    app.render()


def test_ui_app_rl_agent_missing_checkpoint():
    """Verify ExplorerUIApp handles missing checkpoint gracefully without crashing."""
    app = ExplorerUIApp(
        difficulty="easy",
        seed=42,
        agent_type="rl",
        checkpoint_path="/nonexistent/path/model.zip",
        headless=True,
    )
    assert app.agent_type == "rl"
    assert app.agent is None
    # Step should be a no-op waiting for manual input or loaded model
    traj_len = len(app.trajectory)
    app.step()
    assert len(app.trajectory) == traj_len
    app.render()


def test_resolve_checkpoint_path_shortcuts():
    """Verify resolve_checkpoint_path resolves aliases and directory paths."""
    # Test shortcut aliases matching real checkpoints in results/checkpoints/
    res_med = resolve_checkpoint_path("medium")
    assert res_med is not None
    assert "medium" in res_med.parent.name
    assert res_med.suffix == ".zip"

    res_easy = resolve_checkpoint_path("easy")
    assert res_easy is not None
    assert "easy" in res_easy.parent.name
    assert "nomem" not in res_easy.parent.name

    res_nomem = resolve_checkpoint_path("nomem")
    assert res_nomem is not None
    assert "nomem" in res_nomem.parent.name or "no-memory" in res_nomem.parent.name

    res_rec = resolve_checkpoint_path("recurrent")
    assert res_rec is not None
    assert "recurrent" in res_rec.parent.name

    # Test direct file path
    assert resolve_checkpoint_path(res_med) == res_med.resolve()

    # Test default fallback
    assert resolve_checkpoint_path(None) is not None


def test_ui_app_with_alias_checkpoint():
    """Verify ExplorerUIApp initializes and steps using alias checkpoint string."""
    app = ExplorerUIApp(
        difficulty="medium",
        seed=42,
        checkpoint_path="medium",
        headless=True,
    )
    assert app.agent_type == "rl"
    assert isinstance(app.agent, RLAgent)
    app.step()
    assert len(app.trajectory) == 2
    app.render()


def test_ui_app_terminal_lock_on_death():
    """Verify simulation locks permanently when agent dies, preventing further moves."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="manual", headless=True)
    assert app.raw_env is not None
    assert not app.is_done

    # Place agent directly next to a monster to force collision on forward move
    monster = app.raw_env.monsters[0]
    mx, my = monster.pos
    app.raw_env.agent_pos = (mx - 1, my)
    app.raw_env.agent_dir = 0  # Facing Right towards monster

    # Advance 1 step forward into monster
    app.step(2)

    assert app.is_done is True
    assert app.final_end_reason == "death"
    traj_len_at_death = len(app.trajectory)
    step_count_at_death = app.raw_env.step_count

    # Attempt to step again via autonomous agent step
    app.step()
    assert len(app.trajectory) == traj_len_at_death
    assert app.raw_env.step_count == step_count_at_death

    # Attempt to step via manual WASD action
    app.step(2)
    assert len(app.trajectory) == traj_len_at_death
    assert app.raw_env.step_count == step_count_at_death

    # Ensure UI rendering displays dead status and button states remain locked
    app.render()
    assert app.btn_play.custom_bg == (51, 65, 85)


def test_ui_app_terminal_lock_on_coverage():
    """Verify simulation locks permanently when target coverage is reached."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="manual", headless=True)
    assert app.raw_env is not None

    # Simulate achieving target coverage (>= 95%)
    app.raw_env.tracker.revealed_tiles = set(app.raw_env.walkable_tiles)
    app.step(0)  # Turn in place

    assert app.is_done is True
    assert app.final_end_reason == "coverage"
    traj_len = len(app.trajectory)

    # Further steps must be no-ops
    app.step()
    app.step(1)
    assert len(app.trajectory) == traj_len
    app.render()


def test_ui_app_terminal_lock_on_timeout():
    """Verify simulation locks permanently when max steps are reached."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="manual", headless=True)
    assert app.raw_env is not None

    # Set step_count right at max_steps - 1
    app.raw_env.step_count = app.raw_env.max_steps - 1
    app.step(0)  # Turn in place to hit max_steps

    assert app.is_done is True
    assert app.final_end_reason == "timeout"
    traj_len = len(app.trajectory)
    step_count = app.raw_env.step_count

    # Further steps must be no-ops
    app.step()
    assert len(app.trajectory) == traj_len
    assert app.raw_env.step_count == step_count
    app.render()


def test_ui_app_reset_clears_terminal_lock():
    """Verify resetting the environment clears the terminal lock and permits stepping."""
    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="frontier", headless=True)
    app.is_done = True
    app.final_end_reason = "death"

    # Reset environment
    app.reset_env()
    assert app.is_done is False
    assert app.final_end_reason is None

    # Should be able to step now
    initial_traj_len = len(app.trajectory)
    app.step()
    assert len(app.trajectory) == initial_traj_len + 1


def test_ui_app_hotkeys_manual_controls():
    """Verify manual navigation supports Arrow keys and WASD without key collision."""
    import pygame

    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="manual", headless=True)
    assert app.raw_env is not None
    init_dir = app.raw_env.agent_dir

    # Test Right Arrow turns right
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    app.handle_events()
    assert app.raw_env.agent_dir == (init_dir + 1) % 4

    # Test Left Arrow turns left
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    app.handle_events()
    assert app.raw_env.agent_dir == init_dir

    # Test D key turns right
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
    app.handle_events()
    assert app.raw_env.agent_dir == (init_dir + 1) % 4

    # Test A key turns left
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
    app.handle_events()
    assert app.raw_env.agent_dir == init_dir


def test_ui_app_hotkeys_agent_mode_no_navigation_leak():
    """Verify that in autonomous agent mode, pressing S or Arrow keys does NOT step the agent."""
    import pygame

    app = ExplorerUIApp(difficulty="easy", seed=42, agent_type="frontier", headless=True)
    initial_traj_len = len(app.trajectory)

    # Pressing S must NOT step the agent
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
    app.handle_events()
    assert len(app.trajectory) == initial_traj_len

    # Pressing Right Arrow must NOT step the agent
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    app.handle_events()
    assert len(app.trajectory) == initial_traj_len

    # Pressing Enter MUST step the agent
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    app.handle_events()
    assert len(app.trajectory) == initial_traj_len + 1




