"""
gs_template/config.py — Method Registration and Configuration
==============================================================

This file defines the MethodSpecification objects that nerfstudio discovers
via the entry-point system. It wires together:

    TrainerConfig
    └── PipelineConfig (VanillaPipelineConfig)
        ├── DataManagerConfig (FullImageDatamanagerConfig)
        │   └── DataParserConfig (NerfstudioDataParserConfig)
        └── ModelConfig (GSTemplateModelConfig)
    └── optimizers: Dict[str, {optimizer, scheduler}]

KEY DESIGN DECISIONS:
    - FullImageDatamanager: Gaussian Splatting trains on full images, NOT rays.
      Unlike NeRF which samples random rays, GS rasterizes entire images.
    - load_3D_points=True: Essential for SfM-based initialization (COLMAP).
    - mixed_precision=False: GS does NOT use mixed precision training.
      The gradients for Gaussian parameters are sensitive to precision.
    - Per-parameter optimizers: Each Gaussian attribute (means, scales, quats,
      features_dc, features_rest, opacities) gets its OWN optimizer with
      distinct learning rates. This matches the original 3DGS paper.

OPTIMIZER LEARNING RATES (from original 3DGS paper):
    means:         1.6e-4  (with exponential decay to 1.6e-6)
    features_dc:   0.0025  (0th-order SH, highest color LR)
    features_rest:  0.000125 (higher-order SH, 20x lower than DC)
    opacities:     0.05    (high LR for fast opacity convergence)
    scales:        0.005
    quats:         0.001

HOW TO ADD A NEW METHOD VARIANT:
    1. Create a new MethodSpecification with adjusted defaults
    2. Add entry to pyproject.toml [project.entry-points.'nerfstudio.method_configs']
    3. Run `pip install -e . && ns-install-cli`
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

# --------------------------------------------------------------------------
# Use nerfstudio's built-in pipeline and data components
# GS methods do NOT need custom pipelines or datamanagers in most cases
# --------------------------------------------------------------------------
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    NerfstudioDataParserConfig,
)

# --------------------------------------------------------------------------
# Import our custom model config
# --------------------------------------------------------------------------
from gs_template.model import GSTemplateModelConfig


# ============================================================================
# METHOD VARIANT 1: Default ADC Strategy (standard 3DGS densification)
# ============================================================================
# This is the standard Gaussian Splatting configuration using Adaptive
# Density Control (ADC) — the split/clone/prune strategy from the original
# 3DGS paper (Kerbl et al., 2023).
#
# Use this as your base for most papers. Papers that only modify losses,
# initialization, or rendering can use this config directly.
# ============================================================================

gs_template_method = MethodSpecification(
    config=TrainerConfig(
        method_name="gs-template",
        # ---- Training schedule ----
        steps_per_eval_image=100,      # Evaluate on one image every N steps
        steps_per_eval_batch=0,        # 0 = disabled (GS uses full images)
        steps_per_eval_all_images=1000,  # Full eval every N steps
        steps_per_save=2000,           # Checkpoint frequency
        max_num_iterations=30_000,     # Standard 3DGS: 30k iterations
        # ---- IMPORTANT: GS does NOT use mixed precision ----
        # Gaussian parameters (especially means/scales) need full fp32
        mixed_precision=False,
        # ---- Pipeline ----
        pipeline=VanillaPipelineConfig(
            # -- DataManager: Full-image training for splatting --
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    # CRITICAL: Load SfM points for Gaussian initialization
                    load_3D_points=True,
                ),
                # Cache images as uint8 to save GPU memory.
                # They are converted to float32 on-the-fly during training.
                cache_images_type="uint8",
            ),
            # -- Model: Our extensible GS model --
            model=GSTemplateModelConfig(
                # Default config matches original 3DGS
                # Override these for your paper's specific settings
            ),
        ),
        # ---- Per-Parameter Optimizers ----
        # Each Gaussian attribute gets its own Adam optimizer.
        # This is NOT optional — different parameters need different LRs.
        optimizers={
            # Position: Low LR with exponential decay
            # (positions are sensitive; too-high LR causes instability)
            "means": {
                "optimizer": AdamOptimizerConfig(
                    lr=1.6e-4, eps=1e-15,
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30_000,
                ),
            },
            # 0th-order SH (base color): Moderate LR
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            # Higher-order SH (view-dependent effects): 20x lower than DC
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.000125, eps=1e-15),
                "scheduler": None,
            },
            # Opacity: High LR for fast convergence of visibility
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            # Scale: Moderate LR
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            # Rotation quaternions: Lower LR for stability
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
        },
        # ---- Viewer ----
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    ),
    description=(
        "Gaussian Splatting template with ADC densification strategy. "
        "Extend this for most static 3DGS papers."
    ),
)


# ============================================================================
# METHOD VARIANT 2: MCMC Strategy (3DGS as Markov Chain Monte Carlo)
# ============================================================================
# Alternative densification using stochastic Langevin dynamics instead of
# heuristic split/clone. From "3D Gaussian Splatting as Markov Chain
# Monte Carlo" (Kheradmand et al., 2024).
#
# Key differences from ADC:
#   - No split/clone heuristics; uses stochastic perturbation
#   - Fixed maximum number of Gaussians (cap_max)
#   - Opacity regularization is important for this strategy
#   - Generally more robust but may need more iterations
# ============================================================================

gs_template_mcmc_method = MethodSpecification(
    config=TrainerConfig(
        method_name="gs-template-mcmc",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_eval_all_images=1000,
        steps_per_save=2000,
        max_num_iterations=30_000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=GSTemplateModelConfig(
                # MCMC-specific settings
                strategy_type="mcmc",
                mcmc_cap_max=200_000,
                mcmc_noise_lr=5e4,
                mcmc_opacity_reg=0.01,
                # MCMC benefits from higher initial opacity
                initial_opacity=0.5,
            ),
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
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    ),
    description=(
        "Gaussian Splatting template with MCMC densification strategy. "
        "Uses stochastic Langevin dynamics instead of ADC split/clone."
    ),
)
