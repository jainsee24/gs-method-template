"""
gs_template — Gaussian Splatting Method Template for Nerfstudio
================================================================

Minimal, extensible template for implementing 3DGS research papers
as nerfstudio plugins. Extends SplatfactoModel.

Installable methods:
    ns-train gs-template    --data <path>   # vanilla base
    ns-train absgs          --data <path>   # AbsGS (absgrad densification)
    ns-train depth-reg-gs   --data <path>   # Depth-regularized GS
    ns-train feature-gs     --data <path>   # Feature distillation GS
"""

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.datamanagers import DepthDataManager, DepthDataManagerConfig
from gs_template.losses import cosine_loss, pearson_depth_loss, opacity_entropy_loss
from gs_template.rendering import prepare_camera, rasterize_gaussians, assemble_outputs

__all__ = [
    "GSTemplateModel",
    "GSTemplateModelConfig",
    "DepthDataManager",
    "DepthDataManagerConfig",
    "cosine_loss",
    "pearson_depth_loss",
    "opacity_entropy_loss",
    "prepare_camera",
    "rasterize_gaussians",
    "assemble_outputs",
]
