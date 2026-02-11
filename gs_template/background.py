"""
gs_template/background.py — Background Models for Gaussian Splatting
=====================================================================

The background color is composited behind the Gaussians:
    final_rgb = rendered_rgb + (1 - alpha) * background_color

The choice of background model affects both training quality and visual results.

BACKGROUND MODEL CATALOGUE (by paper):
    Original 3DGS:  Random per-step (black for bounded scenes)
    Splatfacto-W:   Learned SH environment map
    NeRF++/GS:      Learned MLP background
    Mip-Splatting:  White background for Blender scenes

WHY RANDOM BACKGROUND?
    Using random background colors during training prevents Gaussians from
    "hiding" against a fixed background. Without this, semi-transparent
    Gaussians near the scene boundary may learn to match the background
    instead of the actual scene content, leading to artifacts when the
    background changes at test time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal


class BackgroundModel(nn.Module):
    """Standard background color model.

    Supports fixed (black/white) or random-per-step background colors.
    For training, random is recommended. For evaluation, the model
    automatically uses black.

    Args:
        mode: Background color mode.
            "random" → Random RGB per training step (black for eval)
            "black"  → Always black [0, 0, 0]
            "white"  → Always white [1, 1, 1]
    """

    def __init__(self, mode: Literal["random", "black", "white"] = "random"):
        super().__init__()
        self.mode = mode

    def forward(
        self,
        H: int,
        W: int,
        device: torch.device,
        training: bool = True,
    ) -> Tensor:
        """Compute background color.

        Args:
            H, W: Image dimensions (not used for constant backgrounds).
            device: Torch device.
            training: Whether we're in training mode.

        Returns:
            (3,) background color tensor.
        """
        if self.mode == "random" and training:
            return torch.rand(3, device=device)
        elif self.mode == "white":
            return torch.ones(3, device=device)
        else:
            return torch.zeros(3, device=device)


class LearnedBackground(nn.Module):
    """Learned MLP background model.

    Predicts background color as a function of ray direction, enabling
    the model to represent complex environment lighting.

    Useful for unbounded outdoor scenes where the background contains
    sky, buildings, etc. that can't be represented by Gaussians alone.

    Papers: NeRF++ (adapted), env-map approaches

    Args:
        hidden_dim: MLP hidden layer dimension.
        num_layers: Number of MLP layers.
        use_direction: If True, condition on view direction.
            If False, learn a single global background color.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_direction: bool = True,
    ):
        super().__init__()
        self.use_direction = use_direction

        if use_direction:
            in_dim = 3  # view direction
            layers = []
            for i in range(num_layers):
                out = hidden_dim if i < num_layers - 1 else 3
                layers.append(nn.Linear(in_dim, out))
                if i < num_layers - 1:
                    layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            layers.append(nn.Sigmoid())
            self.mlp = nn.Sequential(*layers)
        else:
            # Single learnable background color
            self.bg_color = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        H: int,
        W: int,
        device: torch.device,
        training: bool = True,
        view_dirs: Tensor | None = None,
    ) -> Tensor:
        """Compute per-pixel or global background color.

        Args:
            H, W: Image dimensions.
            device: Torch device.
            training: Whether in training mode.
            view_dirs: (H, W, 3) per-pixel view directions (optional).

        Returns:
            (3,) global color or (H, W, 3) per-pixel background.
        """
        if not self.use_direction:
            return torch.sigmoid(self.bg_color)

        if view_dirs is not None:
            return self.mlp(view_dirs.to(device))

        # Fallback: uniform background when directions not available
        return torch.sigmoid(self.bg_color) if hasattr(self, "bg_color") else torch.zeros(3, device=device)


class SHBackground(nn.Module):
    """Spherical Harmonics environment map background.

    Represents the background as an SH function of viewing direction,
    similar to how per-Gaussian colors use SH but for the background.

    From Splatfacto-W (Xu et al., 2024). Low-order SH (degree 2-3)
    captures smooth environmental lighting effectively.

    Args:
        sh_degree: SH degree for background (default: 3).
    """

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        n_coeffs = (sh_degree + 1) ** 2
        # SH coefficients for RGB background (n_coeffs, 3)
        self.sh_coeffs = nn.Parameter(torch.zeros(n_coeffs, 3))
        # Initialize DC term to gray
        self.sh_coeffs.data[0] = 0.5
        self.sh_degree = sh_degree

    def forward(
        self,
        H: int,
        W: int,
        device: torch.device,
        training: bool = True,
        view_dirs: Tensor | None = None,
    ) -> Tensor:
        """Evaluate SH background for given view directions.

        Args:
            H, W: Image dimensions.
            device: Torch device.
            training: Whether in training mode.
            view_dirs: (H, W, 3) normalized view directions.

        Returns:
            (H, W, 3) per-pixel background colors, or (3,) if no dirs.
        """
        if view_dirs is None:
            # Return DC term as constant color
            C0 = 0.28209479177387814
            return (self.sh_coeffs[0] * C0 + 0.5).clamp(0, 1)

        # Full SH evaluation would go here
        # For simplicity, return DC term; full impl needs SH basis eval
        C0 = 0.28209479177387814
        return (self.sh_coeffs[0] * C0 + 0.5).clamp(0, 1)
