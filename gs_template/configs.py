"""
gs_template/configs.py — Method Registrations
===============================================

All MethodSpecification entries for nerfstudio's entry-point discovery.
Each config wires together: Trainer → Pipeline → DataManager → Model → Optimizers.

Registered methods:
    gs-template   — vanilla base (identical to splatfacto)
    absgs         — AbsGS (absgrad densification)
    depth-reg-gs  — depth-regularized GS
    feature-gs    — feature distillation GS
"""

from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    NerfstudioDataParserConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from gs_template.model import GSTemplateModelConfig
from gs_template.methods.absgs import AbsGSModelConfig
from gs_template.methods.depth_reg import DepthRegGSModelConfig
from gs_template.methods.feature_gs import FeatureGSModelConfig
from gs_template.datamanagers import DepthDataManagerConfig


# ═══════════════════════════════════════════════════════════════════
# Shared optimizer / datamanager / viewer building blocks
# ═══════════════════════════════════════════════════════════════════

def _base_datamanager() -> FullImageDatamanagerConfig:
    """Standard full-image datamanager for GS methods."""
    return FullImageDatamanagerConfig(
        dataparser=NerfstudioDataParserConfig(load_3D_points=True),
        cache_images_type="uint8",
    )


def _base_optimizers(max_steps: int = 30000) -> dict:
    """Standard 3DGS optimizer set (from the original paper)."""
    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6, max_steps=max_steps,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
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
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps,
                warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
    }


def _base_viewer() -> ViewerConfig:
    return ViewerConfig(num_rays_per_chunk=1 << 15)


# ═══════════════════════════════════════════════════════════════════
# Method 1: gs-template (vanilla base)
# ═══════════════════════════════════════════════════════════════════

gs_template_method = MethodSpecification(
    config=TrainerConfig(
        method_name="gs-template",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=GSTemplateModelConfig(),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="Gaussian Splatting base template (equivalent to splatfacto).",
)


# ═══════════════════════════════════════════════════════════════════
# Method 2: absgs (absolute gradient densification)
# ═══════════════════════════════════════════════════════════════════

absgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="absgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=AbsGSModelConfig(),  # absgrad=True by default
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="AbsGS: absolute gradient densification for fine detail recovery.",
)


# ═══════════════════════════════════════════════════════════════════
# Method 3: depth-reg-gs (depth-regularized)
# ═══════════════════════════════════════════════════════════════════

def _depth_datamanager() -> DepthDataManagerConfig:
    """DataManager that runs monocular depth estimation + caching."""
    return DepthDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(load_3D_points=True),
        cache_images_type="uint8",
        depth_model="depth_anything_v2",
        depth_model_size="vitb",
        enable_cache=True,
    )


depth_reg_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="depth-reg-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_depth_datamanager(),  # ← runs depth estimation
            model=DepthRegGSModelConfig(),
        ),
        optimizers=_base_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="Depth-regularized GS with Pearson depth loss + opacity entropy.",
)


# ═══════════════════════════════════════════════════════════════════
# Method 4: feature-gs (feature distillation)
# ═══════════════════════════════════════════════════════════════════

def _feature_gs_optimizers() -> dict:
    """Base optimizers + distill_features + feature_mlp."""
    opts = _base_optimizers()
    opts["distill_features"] = {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=5e-4, max_steps=10000,
        ),
    }
    opts["feature_mlp"] = {
        "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
        "scheduler": None,
    }
    return opts


feature_gs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="feature-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=_base_datamanager(),
            model=FeatureGSModelConfig(sh_degree=0),
        ),
        optimizers=_feature_gs_optimizers(),
        viewer=_base_viewer(),
        vis="viewer",
    ),
    description="Feature GS: distill CLIP/DINO features into per-Gaussian latents.",
)
