"""
gs_template/compression.py — Compression and Export Utilities
=============================================================

Post-training compression reduces the storage size of Gaussian Splatting
models without significantly degrading rendering quality. This is critical
for deployment (web viewers, mobile, etc.).

COMPRESSION CATALOGUE (by paper):
    Compact3D:      Vector quantization of SH + learned codebooks
    LightGaussian:  SH pruning + quantization + knowledge distillation
    C3DGS:          Learnable masks + entropy-constrained quantization
    Mini-Splatting:  Importance-based Gaussian removal
    HAC:            Hash-grid assisted compression

COMMON COMPRESSION TECHNIQUES:
    1. SH Pruning:
       Remove higher-order SH coefficients (keep only degree 0 or 1).
       Reduces storage by 4-12x per Gaussian with modest quality loss.

    2. Opacity-based Pruning:
       Remove Gaussians with very low opacity (< 0.01) or very small scale.
       These contribute negligibly to the final image.

    3. Quantization:
       Reduce float32 → float16 or int8 for Gaussian attributes.
       Position: float16 is usually sufficient.
       Colors: int8 with codebook quantization.

    4. Vector Quantization:
       Cluster SH coefficients and store cluster indices + codebook.
       From Compact3D.

USAGE:
    After training, compress and export:
        model = GSTemplateModel.load_from_checkpoint(...)
        compressor = GaussianCompressor(model)
        compressor.prune_by_opacity(min_opacity=0.01)
        compressor.prune_sh_degree(max_degree=1)
        compressor.quantize(bits=16)
        compressor.save_ply("compressed_output.ply")
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


class GaussianCompressor:
    """Post-training compression utilities for Gaussian Splatting models.

    This class operates on the gauss_params dictionary extracted from
    a trained GSTemplateModel and provides various compression operations.

    Args:
        gauss_params: Dict of Gaussian parameter tensors.
            Must contain: "means", "scales", "quats", "features_dc",
            "features_rest", "opacities".
    """

    def __init__(self, gauss_params: Dict[str, Tensor]):
        # Clone all parameters to avoid modifying the original model
        self.params = {
            k: v.detach().clone().cpu() for k, v in gauss_params.items()
        }
        self._original_count = self.params["means"].shape[0]

    @property
    def num_gaussians(self) -> int:
        """Current number of Gaussians."""
        return self.params["means"].shape[0]

    @property
    def compression_ratio(self) -> float:
        """Ratio of current to original number of Gaussians."""
        return self.num_gaussians / self._original_count

    # ====================================================================
    # PRUNING OPERATIONS
    # ====================================================================

    def prune_by_opacity(
        self, min_opacity: float = 0.01
    ) -> int:
        """Remove Gaussians with opacity below threshold.

        Args:
            min_opacity: Minimum sigmoid(raw_opacity) to keep.

        Returns:
            Number of Gaussians removed.
        """
        opacities = torch.sigmoid(self.params["opacities"]).squeeze(-1)
        mask = opacities >= min_opacity
        return self._apply_mask(mask)

    def prune_by_scale(
        self,
        max_scale: float = 1.0,
        min_scale: float = 1e-6,
    ) -> int:
        """Remove Gaussians with extreme scale values.

        Args:
            max_scale: Maximum exp(raw_scale) in any dimension.
            min_scale: Minimum exp(raw_scale) in any dimension.

        Returns:
            Number of Gaussians removed.
        """
        scales = torch.exp(self.params["scales"])
        max_s = scales.max(dim=-1).values
        min_s = scales.min(dim=-1).values
        mask = (max_s <= max_scale) & (min_s >= min_scale)
        return self._apply_mask(mask)

    def prune_by_importance(
        self,
        keep_ratio: float = 0.9,
    ) -> int:
        """Remove least important Gaussians based on opacity * volume.

        Importance = opacity * product(scales). Gaussians with the lowest
        importance contribute least to the rendered image.

        Args:
            keep_ratio: Fraction of Gaussians to keep (0-1).

        Returns:
            Number of Gaussians removed.
        """
        opacities = torch.sigmoid(self.params["opacities"]).squeeze(-1)
        scales = torch.exp(self.params["scales"])
        volume = scales.prod(dim=-1)
        importance = opacities * volume

        k = int(self.num_gaussians * keep_ratio)
        _, top_idx = importance.topk(k)
        mask = torch.zeros(self.num_gaussians, dtype=torch.bool)
        mask[top_idx] = True
        return self._apply_mask(mask)

    def prune_sh_degree(self, max_degree: int = 1) -> None:
        """Reduce SH degree by truncating higher-order coefficients.

        This is one of the most effective compression operations.
        Degree 3→0 removes 15/16 of SH storage per Gaussian.

        Args:
            max_degree: Maximum SH degree to keep (0-3).
        """
        if "features_rest" not in self.params:
            return

        from gs_template.model import num_sh_bases

        k = num_sh_bases(max_degree) - 1  # -1 because DC is separate
        if k <= 0:
            # Only keep DC (degree 0)
            self.params["features_rest"] = torch.zeros(
                self.num_gaussians, 0, 3
            )
        else:
            rest = self.params["features_rest"]
            self.params["features_rest"] = rest[:, :k, :]

    # ====================================================================
    # QUANTIZATION
    # ====================================================================

    def quantize(self, bits: int = 16) -> None:
        """Quantize Gaussian parameters to reduce precision.

        Args:
            bits: Target bit width (16 for float16, 8 for int8).
        """
        if bits == 16:
            for key in self.params:
                self.params[key] = self.params[key].half()
        elif bits == 8:
            # Only quantize SH coefficients (positions need higher precision)
            for key in ["features_dc", "features_rest"]:
                if key in self.params:
                    data = self.params[key]
                    # Normalize to [0, 255] and store as uint8
                    vmin, vmax = data.min(), data.max()
                    scale = (vmax - vmin) / 255.0
                    quantized = ((data - vmin) / scale).round().byte()
                    # Store quantization parameters for dequantization
                    self.params[f"{key}_qscale"] = torch.tensor([scale])
                    self.params[f"{key}_qmin"] = torch.tensor([vmin])
                    self.params[key] = quantized

    # ====================================================================
    # EXPORT
    # ====================================================================

    def save_ply(self, output_path: str | Path) -> None:
        """Export compressed Gaussians to PLY format.

        Compatible with standard Gaussian Splatting viewers
        (e.g., antimatter15, PlayCanvas, Unity plugin).

        The PLY format stores:
        - positions (x, y, z)
        - normals (nx, ny, nz) — set to 0 for compatibility
        - SH coefficients (f_dc_0..2, f_rest_0..N)
        - opacity (opacity)
        - scales (scale_0..2)
        - rotations (rot_0..3)

        Args:
            output_path: Path to save the PLY file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        N = self.num_gaussians
        means = self.params["means"].float().numpy()
        scales = self.params["scales"].float().numpy()
        quats = self.params["quats"].float().numpy()
        opacities = self.params["opacities"].float().numpy()
        features_dc = self.params["features_dc"].float().numpy()

        # Construct PLY header
        attributes = [
            "x", "y", "z",
            "nx", "ny", "nz",
            "f_dc_0", "f_dc_1", "f_dc_2",
        ]

        # Add SH rest coefficients
        if "features_rest" in self.params and self.params["features_rest"].numel() > 0:
            rest = self.params["features_rest"].float().numpy()
            n_rest = rest.shape[1] * rest.shape[2]
            for i in range(n_rest):
                attributes.append(f"f_rest_{i}")
        else:
            rest = None
            n_rest = 0

        attributes.extend(["opacity", "scale_0", "scale_1", "scale_2"])
        attributes.extend(["rot_0", "rot_1", "rot_2", "rot_3"])

        # Write PLY file
        with open(output_path, "wb") as f:
            # Header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {N}\n".encode())
            for attr in attributes:
                f.write(f"property float {attr}\n".encode())
            f.write(b"end_header\n")

            # Data
            normals = np.zeros((N, 3), dtype=np.float32)
            for i in range(N):
                # Position
                f.write(struct.pack("<fff", *means[i]))
                # Normals (zero)
                f.write(struct.pack("<fff", *normals[i]))
                # DC SH
                f.write(struct.pack("<fff", *features_dc[i]))
                # Rest SH
                if rest is not None and n_rest > 0:
                    rest_flat = rest[i].flatten()
                    f.write(struct.pack(f"<{n_rest}f", *rest_flat))
                # Opacity
                f.write(struct.pack("<f", opacities[i, 0]))
                # Scales
                f.write(struct.pack("<fff", *scales[i]))
                # Quaternion
                f.write(struct.pack("<ffff", *quats[i]))

        print(
            f"Saved {N} Gaussians to {output_path} "
            f"({output_path.stat().st_size / 1e6:.1f} MB)"
        )

    def _apply_mask(self, mask: Tensor) -> int:
        """Apply a boolean mask to all parameters, keeping only True entries.

        Args:
            mask: (N,) boolean tensor.

        Returns:
            Number of Gaussians removed.
        """
        removed = (~mask).sum().item()
        for key in list(self.params.keys()):
            if key.startswith(("_", "qscale", "qmin")):
                continue
            param = self.params[key]
            if param.shape[0] == mask.shape[0]:
                self.params[key] = param[mask]
        return removed

    def summary(self) -> str:
        """Print compression summary."""
        lines = [
            f"Gaussian Compression Summary:",
            f"  Original count: {self._original_count:,}",
            f"  Current count:  {self.num_gaussians:,}",
            f"  Compression:    {self.compression_ratio:.1%}",
            f"  Parameters:",
        ]
        total_bytes = 0
        for key, val in self.params.items():
            size = val.numel() * val.element_size()
            total_bytes += size
            lines.append(
                f"    {key:20s}: {list(val.shape)} "
                f"({val.dtype}, {size/1024:.1f} KB)"
            )
        lines.append(f"  Total size: {total_bytes/1e6:.2f} MB")
        return "\n".join(lines)
