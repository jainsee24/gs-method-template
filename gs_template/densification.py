"""
gs_template/densification.py — Custom Densification Strategies
===============================================================

gsplat provides two built-in strategies:
    DefaultStrategy → ADC (Adaptive Density Control) from original 3DGS
    MCMCStrategy    → Stochastic Langevin dynamics

For papers that propose novel densification approaches, you can either:
1. Subclass gsplat's base Strategy class
2. Override get_training_callbacks() in the model to inject custom logic

This file provides examples and utilities for custom densification.

DENSIFICATION STRATEGY CATALOGUE (by paper):
    Original 3DGS:  ADC (split large + clone small with high gradients)
    AbsGS:          ADC with absolute gradient accumulation
    Mini-Splatting:  Blur-split + depth-guided reinitialization
    Scaffold-GS:    Anchor point growing and pruning
    GaussianPro:    MVS-guided propagation from multi-view stereo
    3DGS-MCMC:      MCMC (Markov Chain Monte Carlo) with Langevin dynamics
    Pixel-GS:       Gradient-aware pixel-based growing

HOW DENSIFICATION WORKS (ADC):
    Every `refine_every` steps between `refine_start_iter` and `refine_stop_iter`:
    1. GROW: For each Gaussian, accumulate the 2D image-space gradient magnitude.
       If avg gradient > `grow_grad2d` threshold:
       - If 3D scale > `grow_scale3d` → SPLIT: Replace with 2 smaller Gaussians
       - If 3D scale ≤ `grow_scale3d` → CLONE: Duplicate the Gaussian
    2. PRUNE: Remove Gaussians with:
       - opacity < `prune_opa` threshold
       - max scale > `prune_scale3d` (too large)
    3. RESET: Every `reset_every` refinement steps, reset all opacities to low value
       (forces the model to re-justify each Gaussian's existence)

HOW TO IMPLEMENT A CUSTOM STRATEGY:
    Option A: Modify parameters of DefaultStrategy (covers most papers)
        - AbsGS: Use absgrad=True flag
        - Different thresholds: Adjust grow_grad2d, prune_opa, etc.

    Option B: Override model callbacks for extra logic
        - GaussianPro: Add propagation step in AFTER_TRAIN_ITERATION callback
        - Mini-Splatting: Add multi-phase scheduling

    Option C: Create a new Strategy subclass (advanced)
        - Scaffold-GS: Fundamentally different anchor-based approach
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor


class DensificationUtils:
    """Utility functions for custom densification strategies.

    These are building blocks that can be used in custom get_training_callbacks()
    implementations.
    """

    @staticmethod
    def grow_gaussians_from_depth(
        gauss_params: Dict[str, torch.nn.Parameter],
        depth_map: Tensor,
        camera_intrinsics: Tensor,
        camera_pose: Tensor,
        num_new: int = 1000,
        depth_threshold: float = 0.1,
    ) -> Dict[str, Tensor]:
        """Grow new Gaussians at depth discontinuities.

        Used by GaussianPro and depth-guided initialization methods.
        Identifies regions where depth changes rapidly (likely object boundaries)
        and spawns new Gaussians there.

        Args:
            gauss_params: Current Gaussian parameters.
            depth_map: (H, W, 1) rendered depth map.
            camera_intrinsics: (3, 3) camera K matrix.
            camera_pose: (4, 4) camera-to-world matrix.
            num_new: Maximum number of new Gaussians to add.
            depth_threshold: Gradient threshold for depth discontinuity.

        Returns:
            Dict of new parameter tensors to concatenate with existing ones.
        """
        depth = depth_map.squeeze()  # (H, W)
        H, W = depth.shape

        # Compute depth gradients
        grad_x = torch.abs(depth[:, 1:] - depth[:, :-1])
        grad_y = torch.abs(depth[1:, :] - depth[:-1, :])

        # Find pixels with high depth gradients
        # Pad to maintain original size
        grad_x_pad = torch.nn.functional.pad(grad_x, (0, 1), value=0)
        grad_y_pad = torch.nn.functional.pad(grad_y, (1, 0), value=0)
        grad_mag = torch.sqrt(grad_x_pad**2 + grad_y_pad**2)

        # Select top-k gradient locations
        flat_grad = grad_mag.flatten()
        k = min(num_new, (flat_grad > depth_threshold).sum().item())

        if k == 0:
            return {}

        _, top_indices = flat_grad.topk(k)
        rows = top_indices // W
        cols = top_indices % W

        # Unproject to 3D
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        z = depth[rows, cols]
        x = (cols.float() - cx) * z / fx
        y = (rows.float() - cy) * z / fy

        points_cam = torch.stack([x, y, z], dim=-1)  # (k, 3)

        # Transform to world space
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        points_world = (R @ points_cam.T).T + t  # (k, 3)

        return {
            "means": points_world,
            "num_new": k,
        }

    @staticmethod
    def importance_prune(
        gauss_params: Dict[str, torch.nn.Parameter],
        visibility_count: Tensor,
        min_visibility: int = 3,
    ) -> Tensor:
        """Compute pruning mask based on visibility across training views.

        Gaussians that are visible in very few training views are likely
        floaters and should be pruned.

        Args:
            gauss_params: Current Gaussian parameters.
            visibility_count: (N,) number of views each Gaussian is visible in.
            min_visibility: Minimum number of views for a Gaussian to survive.

        Returns:
            (N,) boolean mask — True for Gaussians to KEEP.
        """
        return visibility_count >= min_visibility

    @staticmethod
    def compute_gaussian_normals(
        scales: Tensor, quats: Tensor
    ) -> Tensor:
        """Compute the normal direction of each Gaussian.

        The normal is defined as the direction of the SMALLEST scale axis.
        This is used by methods that need Gaussian normals for surface
        regularization (2DGS, GOF, GaussianPro).

        Args:
            scales: (N, 3) activated (positive) scale values.
            quats: (N, 4) normalized quaternions (wxyz).

        Returns:
            (N, 3) unit normal vectors.
        """
        # Find the smallest scale axis for each Gaussian
        min_axis = scales.argmin(dim=-1)  # (N,)

        # Convert quaternions to rotation matrices
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        R = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)  # (N, 3, 3)

        # Extract the column corresponding to the smallest scale axis
        normals = torch.zeros_like(scales)
        for i in range(3):
            mask = min_axis == i
            if mask.any():
                normals[mask] = R[mask, :, i]

        return torch.nn.functional.normalize(normals, dim=-1)


class MultiPhaseDensification:
    """Multi-phase training schedule for densification.

    Some papers use different densification parameters in different
    training phases. For example, Mini-Splatting uses:
    - Phase 1 (0-5k): Aggressive growing with low threshold
    - Phase 2 (5k-15k): Standard ADC
    - Phase 3 (15k-30k): No growing, only pruning

    Usage in get_training_callbacks():
        multi_phase = MultiPhaseDensification(phases=[
            {"start": 0, "end": 5000, "grow_grad2d": 0.0001},
            {"start": 5000, "end": 15000, "grow_grad2d": 0.0002},
            {"start": 15000, "end": 30000, "grow_grad2d": float("inf")},
        ])

        def update_strategy(step):
            params = multi_phase.get_params(step)
            if params:
                self.strategy.grow_grad2d = params["grow_grad2d"]
    """

    def __init__(self, phases: list[dict]):
        """Initialize multi-phase schedule.

        Args:
            phases: List of dicts, each with "start", "end", and
                parameter overrides for that phase.
        """
        self.phases = phases

    def get_params(self, step: int) -> Optional[dict]:
        """Get active parameters for the current step.

        Args:
            step: Current training step.

        Returns:
            Dict of parameter overrides, or None if no phase is active.
        """
        for phase in self.phases:
            if phase["start"] <= step < phase["end"]:
                return {
                    k: v
                    for k, v in phase.items()
                    if k not in ("start", "end")
                }
        return None
