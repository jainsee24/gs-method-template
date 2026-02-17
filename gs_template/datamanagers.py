"""
gs_template/datamanagers.py — Custom DataManagers
====================================================

Nerfstudio's architecture: DataManager produces batches, Model consumes them.
Any preprocessing (depth estimation, feature extraction) lives HERE, not
in the model. This follows the feature-splatting pattern.

DataManager Pipeline:
    1. __init__: Run heavy preprocessing (depth estimation), cache to disk
    2. next_train(): Inject preprocessed data into each training batch
    3. next_eval():  Same for evaluation

Included:
    DepthDataManager — Runs monocular depth estimation on all images,
                       caches results, injects batch["depth_image"].
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE


# ═══════════════════════════════════════════════════════════════════
# Depth Estimation DataManager
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DepthDataManagerConfig(FullImageDatamanagerConfig):
    """DataManager that runs monocular depth estimation on all images."""

    _target: Type = field(default_factory=lambda: DepthDataManager)

    depth_model: Literal["depth_anything_v2", "zoedepth", "midas"] = "depth_anything_v2"
    """Which pretrained monocular depth model to use."""

    depth_model_size: Literal["vits", "vitb", "vitl"] = "vitb"
    """Model size variant (for Depth Anything v2)."""

    enable_cache: bool = True
    """Cache estimated depths to disk for fast reload."""


class DepthDataManager(FullImageDatamanager):
    """Estimates monocular depth for all images and injects into batches.

    On first run, estimates depth for every image using the chosen model
    and caches results to {data_dir}/depth_cache_{model}.pt.
    On subsequent runs, loads from cache.

    Injects batch["depth_image"] at each training/eval step.

    Architecture (same pattern as Feature Splatting):
        __init__() → estimate_all_depths() → cache
        next_train() → inject depth into batch
        next_eval()  → inject depth into batch
    """

    config: DepthDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Estimate/load depth maps for all images
        all_depths = self._estimate_all_depths()

        # Split train/eval
        n_train = len(self.train_dataset)
        self.train_depths = all_depths[:n_train]
        self.eval_depths = all_depths[n_train:]
        assert len(self.eval_depths) == len(self.eval_dataset), (
            f"Depth count mismatch: {len(self.eval_depths)} vs {len(self.eval_dataset)}"
        )

        # Free GPU memory after estimation
        torch.cuda.empty_cache()
        gc.collect()

    def _estimate_all_depths(self) -> Tensor:
        """Run monocular depth estimation on all train+eval images.

        Returns:
            (N, H, W, 1) tensor of estimated depths for all images.
        """
        image_fnames = (
            self.train_dataset.image_filenames
            + self.eval_dataset.image_filenames
        )

        # Check cache
        cache_dir = self.config.dataparser.data
        cache_name = f"depth_cache_{self.config.depth_model}_{self.config.depth_model_size}.pt"
        cache_path = cache_dir / cache_name

        if self.config.enable_cache and cache_path.exists():
            CONSOLE.print(f"Loading cached depths from {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            cached_fnames = cache.get("image_fnames")
            if cached_fnames == image_fnames:
                CONSOLE.print(
                    f"  ✓ Loaded {len(cache['depths'])} depth maps from cache"
                )
                return cache["depths"]
            else:
                CONSOLE.print("  Cache invalidated (image list changed), re-estimating...")

        # Estimate depths
        CONSOLE.print(
            f"Estimating depths with {self.config.depth_model} "
            f"({self.config.depth_model_size}) for {len(image_fnames)} images..."
        )
        depths = _run_depth_estimation(
            image_fnames,
            model_name=self.config.depth_model,
            model_size=self.config.depth_model_size,
        )

        # Cache
        if self.config.enable_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"image_fnames": image_fnames, "depths": depths},
                cache_path,
            )
            CONSOLE.print(f"  ✓ Cached depths to {cache_path}")

        return depths

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth_image into training batch."""
        camera, data = super().next_train(step)
        cam_idx = camera.metadata["cam_idx"]
        data["depth_image"] = self.train_depths[cam_idx]
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth_image into eval batch."""
        camera, data = super().next_eval(step)
        cam_idx = camera.metadata["cam_idx"]
        data["depth_image"] = self.eval_depths[cam_idx]
        return camera, data


# ═══════════════════════════════════════════════════════════════════
# Depth Estimation Backend
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_depth_estimation(
    image_paths: list,
    model_name: str = "depth_anything_v2",
    model_size: str = "vitb",
) -> Tensor:
    """Run monocular depth estimation on a list of images.

    Supports multiple backends. Falls back gracefully if a model
    is not installed.

    NOTE: Many depth models (ZoeDepth, MiDAS) store numpy.int64 dimensions
    internally which newer PyTorch rejects in F.interpolate. We globally
    patch F.interpolate to auto-cast sizes to Python int for the duration
    of depth estimation, then restore the original.

    Returns:
        (N, H, W, 1) float32 tensor of metric (or relative) depth maps.
    """
    import torch.nn.functional as F
    import torch.nn as nn

    # ── Patch F.interpolate to accept numpy.int64 sizes ──────────
    _original_interpolate = F.interpolate

    def _safe_interpolate(input, size=None, **kwargs):
        if size is not None:
            if isinstance(size, (list, tuple)):
                size = tuple(int(s) for s in size)
            elif not isinstance(size, int):
                size = int(size)
        return _original_interpolate(input, size=size, **kwargs)

    F.interpolate = _safe_interpolate
    nn.functional.interpolate = _safe_interpolate

    try:
        if model_name == "depth_anything_v2":
            return _depth_anything_v2(image_paths, model_size)
        elif model_name == "zoedepth":
            return _zoedepth(image_paths)
        elif model_name == "midas":
            return _midas(image_paths)
        else:
            raise ValueError(
                f"Unknown depth model: {model_name}. "
                f"Supported: depth_anything_v2, zoedepth, midas"
            )
    finally:
        # Always restore original
        F.interpolate = _original_interpolate
        nn.functional.interpolate = _original_interpolate


def _depth_anything_v2(image_paths: list, model_size: str) -> Tensor:
    """Depth Anything v2 — strong default, metric depth."""
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Depth Anything v2 requires `transformers`. "
            "Install with: pip install transformers"
        )
    from PIL import Image

    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    CONSOLE.print(f"  Loading {model_id}...")
    pipe = pipeline("depth-estimation", model=model_id, device="cuda")

    depths = []
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            CONSOLE.print(f"  Processing image {i+1}/{len(image_paths)}...")
        img = Image.open(path).convert("RGB")
        result = pipe(img)
        # result["depth"] is a PIL Image of predicted depth
        depth_pil = result["depth"]
        import numpy as np
        depth_np = np.array(depth_pil, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_np)
        # Normalize to reasonable range
        dmax = depth_tensor.max()
        if dmax > 0:
            depth_tensor = depth_tensor / dmax
        depths.append(depth_tensor.unsqueeze(-1))  # (H, W, 1)

    del pipe
    torch.cuda.empty_cache()
    return torch.stack(depths)  # (N, H, W, 1)


def _zoedepth(image_paths: list) -> Tensor:
    """ZoeDepth — metric depth estimation."""
    try:
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    except Exception:
        raise ImportError(
            "ZoeDepth failed to load. Install with: "
            "pip install timm==0.6.7 && pip install git+https://github.com/isl-org/ZoeDepth"
        )
    from PIL import Image

    model = model.cuda().eval()

    depths = []
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            CONSOLE.print(f"  Processing image {i+1}/{len(image_paths)}...")
        img = Image.open(path).convert("RGB")
        depth = model.infer_pil(img)  # safe — F.interpolate patched by caller
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(-1)
        depths.append(depth_tensor)  # (H, W, 1)

    del model
    torch.cuda.empty_cache()
    return torch.stack(depths)


def _midas(image_paths: list) -> Tensor:
    """MiDAS v3.1 — relative depth (widely supported fallback)."""
    try:
        model = torch.hub.load("intel-isl/MiDAS", "DPT_BEiT_L_384", pretrained=True)
    except Exception:
        raise ImportError(
            "MiDAS failed to load. Install with: "
            "pip install timm"
        )
    from PIL import Image
    import torchvision.transforms as T

    model = model.cuda().eval()
    transform = T.Compose([
        T.Resize(384),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    depths = []
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            CONSOLE.print(f"  Processing image {i+1}/{len(image_paths)}...")
        img = Image.open(path).convert("RGB")
        orig_h, orig_w = int(img.height), int(img.width)  # force Python int
        input_tensor = transform(img).unsqueeze(0).cuda()
        depth = model(input_tensor)
        # Resize back to original resolution (use Python ints!)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu()
        # MiDAS outputs inverse depth — invert
        depth = 1.0 / (depth + 1e-6)
        depth = depth / depth.max()
        depths.append(depth.float().unsqueeze(-1))

    del model
    torch.cuda.empty_cache()
    return torch.stack(depths)
