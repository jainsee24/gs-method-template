"""
gs_template/losses.py — Reusable Loss Functions
=================================================

A library of common loss functions used across Gaussian Splatting papers.
Import and combine these in your compute_custom_losses() override.

Usage:
    from gs_template.losses import cosine_loss, pearson_depth_loss

    class MyModel(GSTemplateModel):
        def compute_custom_losses(self, outputs, batch, metrics_dict=None):
            return {"feat_loss": 0.001 * cosine_loss(pred, target)}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# Feature Distillation Losses
# ═══════════════════════════════════════════════════════════════════

def cosine_loss(pred: Tensor, target: Tensor, dim: int = 0) -> Tensor:
    """Cosine similarity loss: 1 - cos_sim, averaged over pixels.

    Used in: Feature Splatting, LangSplat, LEGaussians.

    Args:
        pred:   (C, H, W) or (H, W, C) predicted features
        target: same shape as pred, ground-truth features
        dim:    dimension along which to compute cosine similarity
    """
    return (1 - F.cosine_similarity(pred, target, dim=dim)).mean()


def l2_feature_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L2 feature reconstruction loss, per-pixel averaged.

    Used in: DINO-Splatting, generic feature distillation.
    """
    return F.mse_loss(pred, target)


# ═══════════════════════════════════════════════════════════════════
# Depth Losses
# ═══════════════════════════════════════════════════════════════════

def pearson_depth_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Scale-invariant depth loss via Pearson correlation.

    Used in: DepthRegGS, MonoGS, DNGaussian.
    Ignores absolute scale — only penalizes structural disagreement.

    Args:
        pred:   (N,) or (H,W,1) predicted depth (valid pixels only)
        target: same shape, ground-truth or monocular depth
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_c = pred_flat - pred_flat.mean()
    target_c = target_flat - target_flat.mean()
    numer = (pred_c * target_c).sum()
    denom = pred_c.norm() * target_c.norm() + 1e-6
    return 1.0 - numer / denom


def l1_depth_loss(
    pred: Tensor, target: Tensor, valid_mask: Tensor | None = None
) -> Tensor:
    """L1 depth loss with optional validity mask.

    Args:
        pred:       (H, W, 1) predicted depth
        target:     (H, W, 1) ground-truth depth
        valid_mask: (H, W, 1) bool, True where depth is valid
    """
    if valid_mask is not None and valid_mask.sum() > 0:
        return F.l1_loss(pred[valid_mask], target[valid_mask])
    return F.l1_loss(pred, target)


# ═══════════════════════════════════════════════════════════════════
# Regularization Losses
# ═══════════════════════════════════════════════════════════════════

def opacity_entropy_loss(opacities_logit: Tensor) -> Tensor:
    """Encourages opacities toward 0 or 1 (binary), reducing floaters.

    Used in: Many anti-artifact / compactness papers.

    Args:
        opacities_logit: (N, 1) raw logit opacities (pre-sigmoid)
    """
    o = torch.sigmoid(opacities_logit)
    return -(o * torch.log(o + 1e-6) + (1 - o) * torch.log(1 - o + 1e-6)).mean()


def scale_flatten_loss(scales_log: Tensor) -> Tensor:
    """Encourages Gaussians to be flat (one small axis), reducing spikes.

    Used in: 2DGS-inspired methods, SuGaR.

    Args:
        scales_log: (N, 3) log-space scales
    """
    scales = torch.exp(scales_log)
    # Penalize ratio between largest and smallest scale
    sorted_scales, _ = torch.sort(scales, dim=-1)
    ratio = sorted_scales[:, 2] / (sorted_scales[:, 0] + 1e-8)
    return ratio.mean()


def distortion_loss(weights: Tensor, t_mids: Tensor) -> Tensor:
    """Distortion loss from Mip-NeRF 360, adapted for Gaussians.

    Encourages weights to be concentrated (not spread along ray).

    Args:
        weights: (N_rays, N_samples) blending weights
        t_mids:  (N_rays, N_samples) midpoints of sample intervals
    """
    interval = t_mids[:, 1:] - t_mids[:, :-1]
    w_outer = weights[:, :-1] * weights[:, 1:]
    t_diff = (t_mids[:, :-1] - t_mids[:, 1:]).abs()
    return (w_outer * t_diff).sum(dim=-1).mean()


def total_variation_loss(features: Tensor) -> Tensor:
    """Total variation regularization on rendered feature maps.

    Encourages spatial smoothness of per-pixel features.

    Args:
        features: (H, W, C) rendered feature map
    """
    h_diff = (features[1:, :, :] - features[:-1, :, :]).abs().mean()
    w_diff = (features[:, 1:, :] - features[:, :-1, :]).abs().mean()
    return h_diff + w_diff
