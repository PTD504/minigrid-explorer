from __future__ import annotations

import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from explorer.agents.policy import CustomCombinedExtractor, get_policy_kwargs
from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper


def test_custom_extractor_init_and_forward_with_wrapper() -> None:
    """Test extractor initialization and forward pass using sample observations from wrapper."""
    env = ObservationMemoryWrapper(ExploreEnv(), use_memory=True, crop_size=15)
    obs, _ = env.reset(seed=42)

    extractor = CustomCombinedExtractor(
        env.observation_space, features_dim=256, use_memory=True
    )

    # Convert single observation into PyTorch batched tensor dictionary (batch size = 1)
    obs_tensor = {
        k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()
    }

    out = extractor(obs_tensor)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 256)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_custom_extractor_batch_forward() -> None:
    """Test extractor forward pass with batch size > 1."""
    env = ObservationMemoryWrapper(ExploreEnv(), use_memory=True, crop_size=15)
    obs_space = env.observation_space
    extractor = CustomCombinedExtractor(obs_space, features_dim=256, use_memory=True)

    batch_size = 8
    dummy_obs = {
        "view": torch.rand(batch_size, 7, 7, 5, dtype=torch.float32),
        "memory": torch.rand(batch_size, 15, 15, 4, dtype=torch.float32),
        "step_ratio": torch.rand(batch_size, 1, dtype=torch.float32),
    }

    out = extractor(dummy_obs)
    assert out.shape == (batch_size, 256)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_custom_extractor_ablation_use_memory_false() -> None:
    """Test extractor ablation flag use_memory=False when memory is present in observation space."""
    env = ObservationMemoryWrapper(ExploreEnv(), use_memory=True, crop_size=15)
    obs, _ = env.reset(seed=123)

    extractor = CustomCombinedExtractor(
        env.observation_space, features_dim=256, use_memory=False
    )
    assert extractor.use_memory is False
    assert extractor.memory_cnn is None

    obs_tensor = {
        k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()
    }
    out = extractor(obs_tensor)
    assert out.shape == (1, 256)
    assert not torch.isnan(out).any()


def test_custom_extractor_without_memory_in_space() -> None:
    """Test extractor when observation space does not have memory at all."""
    env = ObservationMemoryWrapper(ExploreEnv(), use_memory=False)
    obs, _ = env.reset(seed=999)
    assert "memory" not in obs

    extractor = CustomCombinedExtractor(
        env.observation_space, features_dim=128, use_memory=False
    )
    obs_tensor = {
        k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()
    }
    out = extractor(obs_tensor)
    assert out.shape == (1, 128)
    assert not torch.isnan(out).any()


def test_custom_extractor_backward_gradients() -> None:
    """Verify that gradients can be computed and are finite (no NaN or Inf)."""
    env = ObservationMemoryWrapper(ExploreEnv(), use_memory=True, crop_size=15)
    extractor = CustomCombinedExtractor(
        env.observation_space, features_dim=256, use_memory=True
    )

    batch_size = 4
    dummy_obs = {
        "view": torch.rand(batch_size, 7, 7, 5, requires_grad=False),
        "memory": torch.rand(batch_size, 15, 15, 4, requires_grad=False),
        "step_ratio": torch.rand(batch_size, 1, requires_grad=False),
    }

    out = extractor(dummy_obs)
    loss = out.sum()
    loss.backward()

    # Check all parameters have valid gradients
    for name, param in extractor.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
        assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient"


def test_custom_extractor_invalid_inputs() -> None:
    """Test extractor raises appropriate exceptions on invalid inputs."""
    # Non-Dict observation space
    box_space = spaces.Box(low=0, high=1, shape=(7, 7, 5))
    with pytest.raises(TypeError, match="expects a spaces.Dict"):
        CustomCombinedExtractor(box_space)  # type: ignore[arg-type]

    # Dict observation space missing 'view'
    dict_missing_view = spaces.Dict(
        {"memory": spaces.Box(low=0, high=1, shape=(15, 15, 4))}
    )
    with pytest.raises(KeyError, match="must contain 'view' key"):
        CustomCombinedExtractor(dict_missing_view)


def test_ppo_initialization_and_learning_compatibility() -> None:
    """Verify end-to-end integration with Stable-Baselines3 PPO."""
    env = ObservationMemoryWrapper(ExploreEnv(), use_memory=True, crop_size=15)
    policy_kwargs = get_policy_kwargs(features_dim=256, use_memory=True)

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        verbose=0,
    )

    # Verify extractor is an instance of CustomCombinedExtractor
    extractor = model.policy.features_extractor
    assert isinstance(extractor, CustomCombinedExtractor)
    assert extractor.features_dim == 256

    # Execute a few learning steps to verify rollout collection and optimization loop
    model.learn(total_timesteps=32)

    # Predict an action from raw environment reset observation
    obs, _ = env.reset(seed=101)
    action, _states = model.predict(obs, deterministic=True)
    assert action in (0, 1, 2, 3)
