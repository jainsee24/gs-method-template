"""
examples/absgs.py — Example: Implementing AbsGS using the template
===================================================================

AbsGS: Recovering Fine Details for Gaussian Splatting (Ye et al., 2024)
Paper: https://arxiv.org/abs/2404.10484

WHAT'S NOVEL:
    AbsGS proposes using the ABSOLUTE VALUE of 2D gradients for densification
    instead of the gradient norm. This better identifies under-reconstructed
    regions and produces finer detail in the rendered images.

WHAT TO OVERRIDE:
    Almost nothing! AbsGS only changes the densification gradient computation,
    which is already supported as a FLAG in gsplat's DefaultStrategy.

    Changes needed:
    1. config.py: Set use_absgrad=True, densify_grad_thresh=0.0008
    2. That's it — gsplat handles the rest internally.

    This example shows the SIMPLEST possible paper implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    NerfstudioDataParserConfig,
)

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ============================================================================
# Step 1: Config (optional — could just set flags on GSTemplateModelConfig)
# ============================================================================
# For AbsGS, we don't even need a custom model class. We just change
# config defaults. But we show the pattern for completeness.

@dataclass
class AbsGSModelConfig(GSTemplateModelConfig):
    """AbsGS-specific config. Only changes densification parameters."""

    _target: Type = field(default_factory=lambda: AbsGSModel)

    # KEY CHANGE: Use absolute gradients with adjusted threshold
    use_absgrad: bool = True
    densify_grad_thresh: float = 0.0008  # Higher than default (0.0002) for absgrad


# ============================================================================
# Step 2: Model (minimal — inherits everything from GSTemplateModel)
# ============================================================================

class AbsGSModel(GSTemplateModel):
    """AbsGS model.

    The ONLY difference from the base template is that absgrad=True is passed
    to gsplat's rasterization(), which computes absolute gradients instead
    of gradient norms for densification. The DefaultStrategy then uses these
    absolute gradients for its split/clone decisions.

    This is handled entirely through config flags — no method overrides needed.
    We create this class mainly for clarity and to support paper-specific
    extensions in the future.
    """

    config: AbsGSModelConfig

    # No overrides needed! Everything is handled by:
    # 1. config.use_absgrad = True → passed to rasterization(absgrad=True)
    # 2. config.densify_grad_thresh = 0.0008 → used by DefaultStrategy
    # 3. DefaultStrategy(absgrad=True) → uses absolute gradient accumulation


# ============================================================================
# Step 3: Method registration
# ============================================================================

absgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="absgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        max_num_iterations=30_000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=AbsGSModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6, max_steps=30_000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.000125, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
        },
    ),
    description="AbsGS: Recovering Fine Details with Absolute Gradient Densification",
)

# To register: add to pyproject.toml:
# [project.entry-points.'nerfstudio.method_configs']
# absgs = 'examples.absgs:absgs_method'
