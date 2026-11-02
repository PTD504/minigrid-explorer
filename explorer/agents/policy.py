from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Custom multi-modal feature extractor for MiniGrid Explorer.

    Processes heterogeneous observation spaces produced by ObservationMemoryWrapper:
    - 'view': (B, 7, 7, 5) egocentric visual line-of-sight grid.
    - 'memory': (B, K, K, 4) accumulated egocentric spatial memory crop.
    - 'step_ratio': (B, 1) normalized temporal budget scalar.

    Architecture:
    - View CNN: Conv2D(3x3, s=1) -> ReLU -> Flatten (preserves immediate spatial layout without pooling).
    - Memory CNN: Conv2D(3x3, s=1) -> ReLU -> Conv2D(3x3, s=1) -> ReLU -> Flatten (receptive field 5x5).
    - Temporal modulation: Passthrough scalar concatenated directly with flattened CNN latents.
    - Latent projection trunk: Linear(total_dim, features_dim) -> ReLU (default 256-dim).

    Supports ablation testing via the `use_memory` flag. When False, the memory branch is
    bypassed and only 'view' + 'step_ratio' are projected to the latent features.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        use_memory: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                f"CustomCombinedExtractor expects a spaces.Dict observation space, got {type(observation_space)}"
            )

        if "view" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'view' key.")

        # Determine if memory branch should be activated
        self.use_memory: bool = use_memory and ("memory" in observation_space.spaces)

        # ---------------------------------------------------------------------
        # 1. Local View CNN Branch (7x7x5)
        # ---------------------------------------------------------------------
        view_space = observation_space.spaces["view"]
        view_shape = view_space.shape
        # Support both (H, W, C) and (C, H, W)
        if len(view_shape) == 3:
            if view_shape[-1] in (1, 3, 4, 5):
                self.view_channels = view_shape[-1]
                self.view_h, self.view_w = view_shape[0], view_shape[1]
                self.view_channel_last = True
            else:
                self.view_channels = view_shape[0]
                self.view_h, self.view_w = view_shape[1], view_shape[2]
                self.view_channel_last = False
        else:
            raise ValueError(f"Unexpected view shape: {view_shape}")

        self.view_cnn = nn.Sequential(
            nn.Conv2d(self.view_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_view = torch.zeros(1, self.view_channels, self.view_h, self.view_w)
            view_out_dim = int(self.view_cnn(dummy_view).shape[1])

        # ---------------------------------------------------------------------
        # 2. Egocentric Memory CNN Branch (KxKx4)
        # ---------------------------------------------------------------------
        self.memory_cnn: nn.Module | None = None
        mem_out_dim = 0
        self.mem_channels = 0
        self.mem_channel_last = True

        if self.use_memory:
            mem_space = observation_space.spaces["memory"]
            mem_shape = mem_space.shape
            if len(mem_shape) == 3:
                if mem_shape[-1] in (1, 3, 4, 5):
                    self.mem_channels = mem_shape[-1]
                    self.mem_h, self.mem_w = mem_shape[0], mem_shape[1]
                    self.mem_channel_last = True
                else:
                    self.mem_channels = mem_shape[0]
                    self.mem_h, self.mem_w = mem_shape[1], mem_shape[2]
                    self.mem_channel_last = False
            else:
                raise ValueError(f"Unexpected memory shape: {mem_shape}")

            self.memory_cnn = nn.Sequential(
                nn.Conv2d(self.mem_channels, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                dummy_mem = torch.zeros(1, self.mem_channels, self.mem_h, self.mem_w)
                mem_out_dim = int(self.memory_cnn(dummy_mem).shape[1])

        # ---------------------------------------------------------------------
        # 3. Temporal Modulation (step_ratio scalar)
        # ---------------------------------------------------------------------
        step_dim = 0
        self.has_step_ratio = "step_ratio" in observation_space.spaces
        if self.has_step_ratio:
            step_space = observation_space.spaces["step_ratio"]
            step_dim = int(step_space.shape[0]) if len(step_space.shape) > 0 else 1

        # ---------------------------------------------------------------------
        # 4. Latent Fusion Trunk
        # ---------------------------------------------------------------------
        total_concat_dim = view_out_dim + mem_out_dim + step_dim
        self.fc = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multi-modal feature extractor.

        Args:
            observations: Dictionary of observation tensors from the environment.

        Returns:
            Latent feature tensor of shape (batch_size, features_dim).
        """
        # 1. Process View Branch
        view = observations["view"].float()
        if view.dim() == 3:
            if self.view_channel_last:
                view = view.permute(2, 0, 1).unsqueeze(0)
            else:
                view = view.unsqueeze(0)
        elif (
            view.dim() == 4
            and self.view_channel_last
            and view.shape[1] != self.view_channels
        ):
            view = view.permute(0, 3, 1, 2)

        view_latent = self.view_cnn(view)
        feature_list = [view_latent]

        # 2. Process Memory Branch (if enabled)
        if self.use_memory and self.memory_cnn is not None and "memory" in observations:
            mem = observations["memory"].float()
            if mem.dim() == 3:
                if self.mem_channel_last:
                    mem = mem.permute(2, 0, 1).unsqueeze(0)
                else:
                    mem = mem.unsqueeze(0)
            elif (
                mem.dim() == 4
                and self.mem_channel_last
                and mem.shape[1] != self.mem_channels
            ):
                mem = mem.permute(0, 3, 1, 2)

            mem_latent = self.memory_cnn(mem)
            feature_list.append(mem_latent)

        # 3. Process Temporal Step Ratio (if present)
        if self.has_step_ratio and "step_ratio" in observations:
            sr = observations["step_ratio"].float()
            if sr.dim() == 0:
                sr = sr.view(1, 1)
            elif sr.dim() == 1:
                # If batch size > 1 but shape is (B,), unsqueeze to (B, 1)
                sr = sr.unsqueeze(-1)
            feature_list.append(sr)

        # 4. Concatenate and Project
        fused = torch.cat(feature_list, dim=1)
        return self.fc(fused)


def get_policy_kwargs(
    features_dim: int = 256,
    use_memory: bool = True,
    net_arch: dict[str, list[int]] | list[int] | None = None,
) -> dict[str, Any]:
    """Helper to generate SB3 policy_kwargs dictionary with CustomCombinedExtractor.

    Args:
        features_dim: Output dimension of feature extractor (default: 256).
        use_memory: Whether to process the egocentric memory crop (default: True).
        net_arch: Actor-critic head architecture specification (default: dict(pi=[64, 64], vf=[64, 64])).

    Returns:
        policy_kwargs dict ready to pass to PPO("MultiInputPolicy", env, policy_kwargs=...)
    """
    if net_arch is None:
        net_arch = {"pi": [64, 64], "vf": [64, 64]}

    return {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "use_memory": use_memory,
        },
        "net_arch": net_arch,
    }
