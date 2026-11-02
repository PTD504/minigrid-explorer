from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3 import PPO

from explorer.agents.rl_agent import RLAgent
from explorer.train import make_env, make_vec_env_dungeon


def test_rl_agent_missing_checkpoint():
    """Verify RLAgent raises FileNotFoundError when checkpoint path does not exist."""
    with pytest.raises(FileNotFoundError):
        RLAgent(checkpoint_path="/nonexistent/model.zip")


def test_rl_agent_ppo_inference(tmp_path: Path):
    """Verify RLAgent loads PPO checkpoint, produces valid actions, and resets."""
    train_venv = make_vec_env_dungeon(difficulty="easy", n_envs=1, seed=42, use_memory=True)
    model = PPO("MultiInputPolicy", train_venv, n_steps=8, batch_size=8, n_epochs=1, verbose=0)
    model.learn(total_timesteps=8)
    ckpt_path = tmp_path / "dummy_ppo.zip"
    model.save(str(ckpt_path))
    train_venv.close()

    agent = RLAgent(checkpoint_path=ckpt_path)
    assert agent.is_recurrent is False
    assert agent.model is not None

    env = make_env(difficulty="easy", seed=42, use_memory=True)()
    obs, info = env.reset(seed=42)

    action = agent.act(obs, info)
    assert isinstance(action, int)
    assert 0 <= action <= 2

    agent.reset()
    env.close()

