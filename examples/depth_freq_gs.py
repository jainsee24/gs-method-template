"""
examples/depth_freq_gs.py — Example: A method combining depth supervision + frequency loss
============================================================================================

This fictional example shows how to combine MULTIPLE extension points to
implement a more complex paper. It demonstrates:

    1. Adding new loss terms (depth + frequency)
    2. Adding training callbacks (frequency annealing schedule)
    3. Adding new config fields
    4. Adding new parameter groups (appearance embedding)

This covers the pattern needed for papers like:
    - FreGS (frequency loss + annealing)
    - Depth-supervised 3DGS (depth loss)
    - Splatfacto-W (appearance embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from torch.nn import Parameter

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import DepthLoss, FrequencyLoss
from gs_template.appearance import AppearanceEmbedding


# ============================================================================
# Step 1: Extended Config
# ============================================================================

@dataclass
class DepthFreqGSModelConfig(GSTemplateModelConfig):
    """Config for our combined depth + frequency method."""

    _target: Type = field(default_factory=lambda: DepthFreqGSModel)

    # --- New config fields for our paper ---
    # Depth supervision
    use_depth_loss: bool = True
    depth_loss_weight: float = 0.5
    depth_loss_mode: str = "l1"

    # Frequency regularization (FreGS-style)
    use_freq_loss: bool = True
    freq_loss_weight: float = 0.1
    freq_annealing_steps: int = 15_000  # Steps to reach full frequency band

    # Appearance embedding (Splatfacto-W-style)
    use_appearance_embedding: bool = True
    appearance_embed_dim: int = 32


# ============================================================================
# Step 2: Model with multiple extension point overrides
# ============================================================================

class DepthFreqGSModel(GSTemplateModel):
    """Example model combining depth supervision, frequency loss,
    and appearance embeddings.

    Extension points overridden:
        - populate_modules(): Add loss modules + appearance embedding
        - compute_regularization_losses(): Add depth + frequency losses
        - get_training_callbacks(): Add frequency annealing schedule
        - get_param_groups(): Add appearance embedding parameters
    """

    config: DepthFreqGSModelConfig

    # ------------------------------------------------------------------
    # Override populate_modules to add our custom modules
    # ------------------------------------------------------------------
    def populate_modules(self) -> None:
        # IMPORTANT: Call super() first to set up all base Gaussian params
        super().populate_modules()

        # Add loss modules
        if self.config.use_depth_loss:
            self.depth_loss_fn = DepthLoss(mode=self.config.depth_loss_mode)

        if self.config.use_freq_loss:
            self.freq_loss_fn = FrequencyLoss()
            # Track current frequency band for progressive annealing
            self._freq_band: float = 0.0

        # Add appearance embedding (creates a learnable nn.Module)
        if self.config.use_appearance_embedding:
            self.appearance_embed = AppearanceEmbedding(
                num_images=self.num_train_data,
                embed_dim=self.config.appearance_embed_dim,
            )

    # ------------------------------------------------------------------
    # Override compute_regularization_losses (EXTENSION POINT 8)
    # ------------------------------------------------------------------
    def compute_regularization_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Add depth and frequency losses on top of base regularization."""
        # Get base losses (scale reg, opacity reg if enabled)
        losses = super().compute_regularization_losses(outputs, batch)

        # --- Depth supervision loss ---
        if self.config.use_depth_loss and "depth_image" in batch:
            depth_pred = outputs["depth"]
            depth_gt = batch["depth_image"].to(self.device)

            # Create validity mask (depth_gt > 0)
            mask = (depth_gt.squeeze() > 0).float()

            losses["depth_loss"] = (
                self.config.depth_loss_weight
                * self.depth_loss_fn(depth_pred, depth_gt, mask)
            )

        # --- Frequency spectrum loss ---
        if self.config.use_freq_loss:
            gt_rgb = self._composite_gt_with_background(
                batch["image"], outputs["background"]
            )
            losses["freq_loss"] = (
                self.config.freq_loss_weight
                * self.freq_loss_fn(
                    outputs["rgb"],
                    gt_rgb,
                    freq_band=self._freq_band,
                )
            )

        return losses

    # ------------------------------------------------------------------
    # Override get_training_callbacks (EXTENSION POINT 9)
    # ------------------------------------------------------------------
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> List[TrainingCallback]:
        """Add frequency annealing schedule on top of densification callbacks."""
        # Get base callbacks (densification strategy hooks)
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # Add frequency annealing callback
        if self.config.use_freq_loss:

            def _update_freq_band(step: int) -> None:
                """Progressively increase the frequency band.

                Start with low frequencies only (smooth loss landscape),
                gradually include higher frequencies for fine detail.
                """
                progress = min(
                    step / self.config.freq_annealing_steps, 1.0
                )
                self._freq_band = progress

            callbacks.append(
                TrainingCallback(
                    where_to_run=[
                        TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                    ],
                    func=_update_freq_band,
                )
            )

        return callbacks

    # ------------------------------------------------------------------
    # Override get_param_groups (EXTENSION POINT 10)
    # ------------------------------------------------------------------
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Add appearance embedding parameters to optimizer groups.

        NOTE: You must also add a corresponding entry in config.py optimizers:
            "appearance_embed": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        """
        groups = super().get_param_groups()

        if self.config.use_appearance_embedding:
            groups["appearance_embed"] = list(
                self.appearance_embed.parameters()
            )

        return groups
