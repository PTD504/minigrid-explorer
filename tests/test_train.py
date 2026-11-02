from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from stable_baselines3 import PPO

from explorer.train import (
    DungeonEvalCallback,
    linear_schedule,
    make_env,
    make_vec_env_dungeon,
    train,
    train_ppo,
)


def test_linear_schedule() -> None:
    """Verify linear schedule decays from initial value to 0."""
    sched = linear_schedule(0.001)
    assert pytest.approx(sched(1.0)) == 0.001
    assert pytest.approx(sched(0.5)) == 0.0005
    assert pytest.approx(sched(0.0)) == 0.0


def test_make_env_instance() -> None:
    """Verify make_env returns a callable creating a valid wrapped environment."""
    env_fn = make_env(difficulty="easy", seed=10, use_memory=True, crop_size=15)
    env = env_fn()

    obs, _info = env.reset(seed=10)
    assert "view" in obs
    assert "memory" in obs
    assert "step_ratio" in obs
    assert obs["view"].shape == (7, 7, 5)
    assert obs["memory"].shape == (15, 15, 4)
    assert obs["step_ratio"].shape == (1,)
    env.close()


def test_make_vec_env_dungeon() -> None:
    """Verify make_vec_env_dungeon creates a valid batched VecEnv."""
    venv = make_vec_env_dungeon(
        difficulty="easy",
        n_envs=2,
        seed=42,
        use_memory=True,
        crop_size=15,
        vec_env_type="dummy",
    )
    obs = venv.reset()
    assert obs["view"].shape == (2, 7, 7, 5)
    assert obs["memory"].shape == (2, 15, 15, 4)
    assert obs["step_ratio"].shape == (2, 1)

    actions = [1, 2]
    next_obs, rewards, dones, infos = venv.step(actions)
    assert next_obs["view"].shape == (2, 7, 7, 5)
    assert len(rewards) == 2
    assert len(dones) == 2
    assert len(infos) == 2
    venv.close()


def test_dungeon_eval_callback(tmp_path: Path) -> None:
    """Verify DungeonEvalCallback evaluates policy, aggregates metrics, and saves checkpoint."""
    eval_env = make_env(difficulty="easy", seed=None, use_memory=True)()
    train_venv = make_vec_env_dungeon(
        difficulty="easy", n_envs=2, seed=1, use_memory=True
    )

    cb = DungeonEvalCallback(
        eval_env=eval_env,
        eval_freq=16,
        n_eval_episodes=2,
        val_seeds=[1000000, 1000001],
        best_model_save_path=tmp_path / "checkpoints",
        verbose=0,
    )

    model = PPO(
        "MultiInputPolicy",
        train_venv,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        verbose=0,
    )

    model.learn(total_timesteps=32, callback=cb)

    assert len(cb.eval_history) >= 1
    record = cb.eval_history[-1]
    assert "mean_revealed_coverage" in record
    assert "mean_poi_discovery_rate" in record
    assert "death_rate" in record
    assert "timeout_rate" in record
    assert "mean_reward" in record

    # Best model file should exist in save path
    best_file = tmp_path / "checkpoints" / "best_model.zip"
    assert best_file.exists()

    eval_env.close()
    train_venv.close()


def test_train_ppo_short_run(tmp_path: Path) -> None:
    """Verify train_ppo executes an end-to-end training cycle and outputs checkpoints."""
    model, best_cov, history = train_ppo(
        difficulty="easy",
        seed=99,
        total_timesteps=32,
        n_envs=2,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        eval_freq=16,
        n_eval_episodes=2,
        track=False,
        output_dir=tmp_path,
        verbose=0,
    )

    assert isinstance(model, PPO)
    assert 0.0 <= best_cov <= 1.0
    assert len(history) >= 1

    # Verify final model saved
    ckpts = list(tmp_path.glob("checkpoints/*/final_model.zip"))
    assert len(ckpts) == 1
    assert ckpts[0].exists()


def test_train_ppo_ablation_no_memory(tmp_path: Path) -> None:
    """Verify train_ppo ablation experiment with use_memory=False."""
    model, _best_cov, _history = train_ppo(
        difficulty="easy",
        seed=101,
        total_timesteps=32,
        n_envs=2,
        use_memory=False,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        eval_freq=16,
        n_eval_episodes=2,
        track=False,
        output_dir=tmp_path,
        verbose=0,
    )

    assert isinstance(model, PPO)
    extractor = model.policy.features_extractor
    assert getattr(extractor, "use_memory", None) is False


def test_train_ppo_with_wandb_mock(tmp_path: Path) -> None:
    """Verify train_ppo correctly calls wandb.init and wandb.finish when track=True."""
    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        _model, _best_cov, _history = train_ppo(
            difficulty="easy",
            seed=42,
            total_timesteps=32,
            n_envs=2,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_freq=16,
            n_eval_episodes=2,
            track=True,
            wandb_project="test-project",
            output_dir=tmp_path,
            verbose=0,
        )

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["sync_tensorboard"] is True
        mock_wandb.finish.assert_called_once()


def test_train_recurrent_ppo_short_run(tmp_path: Path) -> None:
    """Verify train with algo='recurrent_ppo' runs end-to-end and outputs checkpoints."""
    from sb3_contrib import RecurrentPPO

    model, best_cov, history = train(
        algo="recurrent_ppo",
        difficulty="easy",
        seed=123,
        total_timesteps=32,
        n_envs=2,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        eval_freq=16,
        n_eval_episodes=2,
        track=False,
        output_dir=tmp_path,
        verbose=0,
    )

    assert isinstance(model, RecurrentPPO)
    assert 0.0 <= best_cov <= 1.0
    assert len(history) >= 1


def test_train_invalid_algo() -> None:
    """Verify train raises ValueError on unsupported algorithm name."""
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        train(algo="dqn")
