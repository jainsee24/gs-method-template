"""
gs_template/losses.py — Composable Loss Modules for Gaussian Splatting
=======================================================================

This module provides reusable loss functions commonly used across 3DGS papers.
Each loss is a standalone class that can be mixed and matched in
compute_regularization_losses().

LOSS CATALOGUE (by paper):
    Original 3DGS:     L1 + SSIM (base loss, always active)
    2DGS:              + DepthDistortionLoss + NormalConsistencyLoss
    GOF:               + DepthDistortionLoss + NormalConsistencyLoss
    FreGS:             + FrequencyLoss
    Compact3D:         + OpacityEntropyLoss
    GaussianPro:       + PlanarLoss
    PhysGaussian:      + ScaleRegularizationLoss
    Depth-supervised:  + DepthLoss

USAGE IN YOUR MODEL:
    class MyModel(GSTemplateModel):
        def populate_modules(self):
            super().populate_modules()
            self.depth_loss = DepthLoss(mode="l1")
            self.normal_loss = NormalConsistencyLoss()

        def compute_regularization_losses(self, outputs, batch):
            losses = super().compute_regularization_losses(outputs, batch)
            losses["depth"] = 0.5 * self.depth_loss(outputs["depth"], batch["depth"])
            losses["normal"] = 0.1 * self.normal_loss(outputs["normals"], outputs["depth"])
            return losses
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_msssim import SSIM


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss.

    Used in the base 3DGS loss: L = (1-λ)*L1 + λ*(1-SSIM).
    Wraps pytorch_msssim.SSIM for convenience.

    Input images should be (H, W, 3) and are automatically permuted to
    (1, 3, H, W) for the SSIM computation.
    """

    def __init__(self, data_range: float = 1.0, channel: int = 3):
        super().__init__()
        self.ssim_module = SSIM(
            data_range=data_range,
            size_average=True,
            channel=channel,
        )

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute SSIM between predicted and ground truth images.

        Args:
            pred: (H, W, 3) predicted image in [0, 1].
            gt: (H, W, 3) ground truth image in [0, 1].

        Returns:
            Scalar SSIM value in [0, 1] (1 = perfect match).
        """
        # pytorch_msssim expects (B, C, H, W)
        pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
        gt_bchw = gt.permute(2, 0, 1).unsqueeze(0)
        return self.ssim_module(pred_bchw, gt_bchw)

    def compute_ssim(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Alias for forward (used in metrics computation)."""
        return self.forward(pred, gt)


class DepthLoss(nn.Module):
    """Depth supervision loss.

    Compares rendered depth against ground truth depth (e.g., from
    a depth sensor, monocular depth estimation, or SfM sparse depth).

    Papers: Depth-supervised 3DGS, DS-NeRF adapted for GS, etc.

    Args:
        mode: Loss type — "l1", "l2", or "log_l1" (log-space L1).
        normalize: Whether to normalize both depths to [0, 1] before comparison.
    """

    def __init__(
        self, mode: str = "l1", normalize: bool = False
    ):
        super().__init__()
        self.mode = mode
        self.normalize = normalize

    def forward(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute depth loss.

        Args:
            pred_depth: (H, W, 1) or (H, W) rendered depth.
            gt_depth: (H, W, 1) or (H, W) ground truth depth.
            mask: Optional (H, W) or (H, W, 1) valid pixel mask.

        Returns:
            Scalar depth loss value.
        """
        pred = pred_depth.squeeze()
        gt = gt_depth.squeeze()

        if self.normalize:
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

        if self.mode == "l1":
            loss = torch.abs(pred - gt)
        elif self.mode == "l2":
            loss = (pred - gt) ** 2
        elif self.mode == "log_l1":
            loss = torch.abs(torch.log(pred + 1e-8) - torch.log(gt + 1e-8))
        else:
            raise ValueError(f"Unknown depth loss mode: {self.mode}")

        if mask is not None:
            mask = mask.squeeze().bool()
            loss = loss[mask]

        return loss.mean()


class DepthDistortionLoss(nn.Module):
    """Depth distortion loss for encouraging compact depth distributions.

    From Mip-NeRF 360 (Barron et al., 2022), adapted for 2DGS (Huang et al., 2024).
    Penalizes the weighted variance of depth values along each ray, encouraging
    Gaussians to concentrate at surfaces rather than spread across depth.

    L_distort = Σ_i Σ_j w_i * w_j * |d_i - d_j|

    This loss requires distortion maps from the rasterizer (available in 2DGS mode).

    Papers: 2DGS, GOF, Mip-Splatting
    """

    def __init__(self):
        super().__init__()

    def forward(self, distortion_map: Tensor) -> Tensor:
        """Compute distortion loss from pre-computed distortion map.

        Args:
            distortion_map: (H, W, 1) distortion values from rasterization_2dgs().

        Returns:
            Scalar distortion loss.
        """
        return distortion_map.mean()


class NormalConsistencyLoss(nn.Module):
    """Normal consistency loss for surface quality.

    Encourages rendered normals to be consistent with depth-derived normals
    and across multiple views. Used by 2DGS and GOF for better geometry.

    L_normal = 1 - <n_render, n_depth>

    where n_render is the rendered normal and n_depth is computed from
    the depth gradient.

    Papers: 2DGS, GOF, Neuralangelo-GS
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        rendered_normals: Tensor,
        depth_normals: Tensor | None = None,
        depth: Tensor | None = None,
        K: Tensor | None = None,
    ) -> Tensor:
        """Compute normal consistency loss.

        Args:
            rendered_normals: (H, W, 3) normals from rasterization.
            depth_normals: (H, W, 3) normals computed from depth gradients.
                If None, computed from depth + K.
            depth: (H, W, 1) depth map (used if depth_normals is None).
            K: (3, 3) camera intrinsics (used if depth_normals is None).

        Returns:
            Scalar normal consistency loss.
        """
        if depth_normals is None and depth is not None and K is not None:
            depth_normals = self._depth_to_normals(depth, K)

        if depth_normals is None:
            raise ValueError(
                "Either depth_normals or (depth, K) must be provided."
            )

        # Normalize both normal maps
        rn = F.normalize(rendered_normals, dim=-1)
        dn = F.normalize(depth_normals, dim=-1)

        # Cosine similarity → loss = 1 - cos_sim
        cos_sim = (rn * dn).sum(dim=-1)
        return (1.0 - cos_sim).mean()

    @staticmethod
    def _depth_to_normals(depth: Tensor, K: Tensor) -> Tensor:
        """Compute normals from depth gradients.

        Args:
            depth: (H, W, 1) depth map.
            K: (3, 3) intrinsics matrix.

        Returns:
            (H, W, 3) normal map.
        """
        d = depth.squeeze(-1)  # (H, W)
        H, W = d.shape

        # Compute depth gradients
        dz_dx = torch.zeros_like(d)
        dz_dy = torch.zeros_like(d)
        dz_dx[:, 1:] = d[:, 1:] - d[:, :-1]
        dz_dy[1:, :] = d[1:, :] - d[:-1, :]

        # Convert to 3D normals using intrinsics
        fx, fy = K[0, 0], K[1, 1]
        normals = torch.stack([-dz_dx * fy, -dz_dy * fx, torch.ones_like(d)], dim=-1)
        normals = F.normalize(normals, dim=-1)

        return normals


class FrequencyLoss(nn.Module):
    """Frequency-domain loss for spectral regularization.

    From FreGS (Zhang et al., 2024). Computes loss in the Fourier domain
    to encourage matching the frequency spectrum of the ground truth image.
    Uses progressive frequency annealing to focus on low frequencies first.

    L_freq = ||FFT(pred) - FFT(gt)||_1 * freq_mask

    Papers: FreGS
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        freq_band: float = 1.0,
    ) -> Tensor:
        """Compute frequency-domain loss.

        Args:
            pred: (H, W, 3) predicted image.
            gt: (H, W, 3) ground truth image.
            freq_band: Fraction of frequency spectrum to include [0, 1].
                0 = only DC, 1 = full spectrum. Use for progressive annealing.

        Returns:
            Scalar frequency loss.
        """
        # Convert to (B, C, H, W)
        pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
        gt_bchw = gt.permute(2, 0, 1).unsqueeze(0)

        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred_bchw, norm="ortho")
        gt_fft = torch.fft.fft2(gt_bchw, norm="ortho")

        # Create frequency mask for progressive annealing
        _, _, H, W = pred_bchw.shape
        freq_mask = self._create_freq_mask(H, W, freq_band, pred.device)

        # L1 loss in frequency domain
        diff = torch.abs(pred_fft - gt_fft) * freq_mask
        return diff.mean()

    @staticmethod
    def _create_freq_mask(
        H: int, W: int, band: float, device: torch.device
    ) -> Tensor:
        """Create a circular frequency mask for progressive annealing.

        Args:
            H, W: spatial dimensions.
            band: fraction of max frequency to include [0, 1].
            device: torch device.

        Returns:
            (1, 1, H, W) binary mask.
        """
        max_freq = min(H, W) // 2
        radius = int(max_freq * band)

        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device).float() - cy
        x = torch.arange(W, device=device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(xx**2 + yy**2)

        mask = (dist <= radius).float()
        # Shift to match FFT layout
        mask = torch.fft.fftshift(mask)
        return mask.unsqueeze(0).unsqueeze(0)


class PlanarLoss(nn.Module):
    """Planar constraint loss for surface regularization.

    From GaussianPro (Cheng et al., 2024). Encourages Gaussians to align
    with locally planar surfaces by penalizing deviations from fitted planes.

    Papers: GaussianPro, SuGaR
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, means: Tensor, normals: Tensor, k: int = 8
    ) -> Tensor:
        """Compute planar loss from Gaussian positions and normals.

        Args:
            means: (N, 3) Gaussian positions.
            normals: (N, 3) Gaussian normals (e.g., from smallest scale axis).
            k: Number of nearest neighbors for local plane fitting.

        Returns:
            Scalar planar constraint loss.
        """
        # For each Gaussian, compute distance to local plane
        # defined by its normal and position
        # This is a simplified version — full GaussianPro uses MVS propagation
        with torch.no_grad():
            # Find k nearest neighbors
            diffs = means.unsqueeze(1) - means.unsqueeze(0)  # (N, N, 3)
            dists = diffs.norm(dim=-1)  # (N, N)
            _, idx = dists.topk(k + 1, largest=False, dim=-1)  # (N, k+1)
            idx = idx[:, 1:]  # skip self

        # Distance from neighbors to local plane
        neighbors = means[idx]  # (N, k, 3)
        local_diff = neighbors - means[:, None, :]  # (N, k, 3)
        plane_dist = (local_diff * normals[:, None, :]).sum(dim=-1)  # (N, k)

        return plane_dist.abs().mean()
