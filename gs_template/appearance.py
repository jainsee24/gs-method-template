"""
gs_template/appearance.py — Appearance / Color Models
=====================================================

This module provides different color representation strategies for Gaussians.
The default is Spherical Harmonics (SH), but many papers propose alternatives.

APPEARANCE MODEL CATALOGUE (by paper):
    Original 3DGS:     Spherical Harmonics (degree 3)
    Splatfacto-W:      SH + per-image appearance embeddings
    Scaffold-GS:       Neural features decoded by view-conditioned MLP
    3DGS-DR:           Diffuse + residual specular decomposition
    GS-IR:             Deferred rendering with material properties

HOW TO USE A CUSTOM APPEARANCE MODEL:
    1. Create a new class extending BaseAppearanceModel
    2. Override compute_colors() in your model to use it
    3. If your model adds trainable parameters, add them to get_param_groups()

EXAMPLE — Per-image appearance embeddings:
    class AppearanceEmbeddingModel(BaseAppearanceModel):
        def __init__(self, num_images, embed_dim=32, sh_degree=3):
            super().__init__()
            self.embeddings = nn.Embedding(num_images, embed_dim)
            self.color_head = nn.Linear(embed_dim, 3)

        def forward(self, features_dc, features_rest, camera, step):
            colors, sh_degree = self.assemble_sh(features_dc, features_rest, step)
            cam_idx = camera.metadata.get("cam_idx", 0)
            embed = self.color_head(self.embeddings(cam_idx))
            colors[:, 0, :] += embed  # Modulate DC term
            return colors, sh_degree
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras


def num_sh_bases(degree: int) -> int:
    """Number of SH basis functions for a given degree."""
    return (degree + 1) ** 2


class SphericalHarmonicsAppearance(nn.Module):
    """Standard Spherical Harmonics appearance model.

    This is the default appearance model from the original 3DGS paper.
    Colors are represented as SH coefficients per Gaussian, evaluated
    by the gsplat rasterizer using the view direction.

    Progressive training: SH degree starts at 0 and increases every
    `sh_degree_interval` steps up to `sh_degree`.

    Args:
        sh_degree: Maximum SH degree (0-3). Default: 3.
        sh_degree_interval: Steps between degree increases. Default: 1000.
    """

    def __init__(
        self,
        sh_degree: int = 3,
        sh_degree_interval: int = 1000,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.sh_degree_interval = sh_degree_interval

    def get_active_sh_degree(self, step: int) -> int:
        """Get the current SH degree based on training step.

        Progressive SH training starts with degree 0 (diffuse only)
        and increases to the maximum degree over training. This helps
        the model learn coarse color first, then view-dependent effects.

        Args:
            step: Current training step.

        Returns:
            Active SH degree for this step.
        """
        if self.sh_degree_interval <= 0:
            return self.sh_degree
        return min(step // self.sh_degree_interval, self.sh_degree)

    def forward(
        self,
        features_dc: Tensor,
        features_rest: Tensor,
        step: int,
    ) -> tuple[Tensor, int]:
        """Assemble SH coefficient tensor for rasterization.

        Combines DC and rest features up to the current active degree.

        Args:
            features_dc: (N, 3) 0th-order SH coefficients.
            features_rest: (N, K-1, 3) higher-order SH coefficients.
            step: Current training step for progressive scheduling.

        Returns:
            Tuple of:
                colors: (N, k, 3) SH coefficients up to active degree.
                sh_degree: Active SH degree (passed to rasterizer).
        """
        sh_degree_to_use = self.get_active_sh_degree(step)
        k = num_sh_bases(sh_degree_to_use)

        # Assemble: [DC (N,1,3), rest up to k-1 (N,k-1,3)]
        colors = torch.cat(
            [
                features_dc[:, None, :],        # (N, 1, 3)
                features_rest[:, : k - 1, :],   # (N, k-1, 3)
            ],
            dim=1,
        )  # (N, k, 3)

        return colors, sh_degree_to_use


class LearnedFeatureAppearance(nn.Module):
    """Neural feature appearance model with per-Gaussian learned features.

    Instead of SH coefficients, each Gaussian stores a learned feature vector
    that is decoded to color by a small MLP conditioned on view direction.

    This is useful for methods that need more expressive color representations
    than SH can provide, or for methods that combine color with other
    properties (e.g., material properties for relighting).

    Papers: Scaffold-GS (anchor features), GS-IR (material decomposition)

    Args:
        feature_dim: Dimension of per-Gaussian feature vectors.
        hidden_dim: Hidden layer dimension in the decoder MLP.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = feature_dim + 3  # features + view direction

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 3
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)
        self.feature_dim = feature_dim

    def forward(
        self,
        features: Tensor,
        view_dirs: Tensor,
    ) -> Tensor:
        """Decode features to RGB colors.

        Args:
            features: (N, feature_dim) per-Gaussian learned features.
            view_dirs: (N, 3) normalized view directions.

        Returns:
            (N, 3) RGB colors in [0, 1].
        """
        x = torch.cat([features, view_dirs], dim=-1)
        rgb = torch.sigmoid(self.decoder(x))
        return rgb


class AppearanceEmbedding(nn.Module):
    """Per-image appearance embedding for handling varying lighting conditions.

    Each training image gets a learnable embedding vector that modulates
    the rendered colors, accounting for exposure changes, white balance
    differences, and other per-image appearance variations.

    Papers: Splatfacto-W, NeRF-W, Block-NeRF (adapted for GS)

    Args:
        num_images: Total number of training images.
        embed_dim: Dimension of the appearance embedding.
    """

    def __init__(
        self,
        num_images: int,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_images, embed_dim)
        # Affine transformation: embed → color scale + shift
        self.affine = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),  # 3 scale + 3 shift
        )
        # Initialize to identity transform
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(
        self, rgb: Tensor, image_idx: int
    ) -> Tensor:
        """Apply appearance correction to rendered RGB.

        Args:
            rgb: (H, W, 3) rendered image.
            image_idx: Index of the current training image.

        Returns:
            (H, W, 3) appearance-corrected image.
        """
        idx = torch.tensor(
            [image_idx], device=rgb.device, dtype=torch.long
        )
        embed = self.embedding(idx)  # (1, embed_dim)
        params = self.affine(embed)  # (1, 6)

        # Decompose into scale and shift
        scale = torch.sigmoid(params[0, :3])  # (3,) in [0, 1]
        shift = params[0, 3:]  # (3,) unbounded

        # Apply: rgb_out = rgb * scale + shift
        # Scale around 1.0, shift around 0.0 (due to init)
        return rgb * (1 + scale[None, None, :]) + shift[None, None, :]
