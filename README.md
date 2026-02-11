# gs-method-template

**A Nerfstudio plugin template for implementing Gaussian Splatting (3DGS) research papers using the gsplat backend.**

This template provides a production-quality, extensible codebase that lets you implement any static 3DGS paper as a nerfstudio plugin. It exposes **11 well-defined extension points** — one for each aspect of the 3DGS pipeline that papers commonly modify — so you only need to override the methods relevant to your paper's contribution.

## Quick Start

```bash
# 1. Install nerfstudio (if not already)
pip install nerfstudio

# 2. Clone and install this template
git clone https://github.com/jainsee24/gs-method-template.git
cd gs-method-template
pip install -e .
ns-install-cli

# 3. Train on your data
ns-train gs-template --data /path/to/your/data

# Or use MCMC strategy
ns-train gs-template-mcmc --data /path/to/your/data
```

## Project Structure

```
gs-method-template/
├── gs_template/
│   ├── __init__.py          # Package exports and documentation
│   ├── config.py            # MethodSpecification + TrainerConfig (entry point)
│   ├── model.py             # GSTemplateModel — core model with 11 extension points
│   ├── losses.py            # Composable loss modules (L1, SSIM, depth, normal, etc.)
│   ├── densification.py     # Custom densification strategies and utilities
│   ├── appearance.py        # Color/appearance models (SH, neural features, embeddings)
│   ├── background.py        # Background models (constant, learned, env-map)
│   └── compression.py       # Post-training compression and PLY export
├── pyproject.toml           # Package config + nerfstudio entry points
└── README.md
```

## The 11 Extension Points

Each extension point is a method in `GSTemplateModel` that you override in your subclass. Search for `EXTENSION POINT` in `model.py` to find them all.

| # | Method | What It Controls | Default Behavior |
|---|--------|-----------------|-----------------|
| 1 | `create_initial_gaussians()` | Initialization strategy | SfM points from COLMAP |
| 2 | `opacity_activation()` | Opacity activation function | `sigmoid` |
| 3 | `scale_activation()` | Scale activation function | `exp` |
| 4 | `rasterize_gaussians()` | Rendering backend | `gsplat.rasterization()` |
| 5 | `compute_colors()` | Appearance / color model | Progressive SH (degree 0→3) |
| 6 | `compute_background()` | Background model | Random (training) / Black (eval) |
| 7 | `compute_regularization_losses()` | Custom loss terms | Scale regularization (optional) |
| 8 | `get_additional_outputs()` | Extra render outputs | None (override for normals, etc.) |
| 9 | `get_training_callbacks()` | Training schedule hooks | ADC or MCMC densification |
| 10 | `get_param_groups()` | Optimizer parameter groups | Per-attribute Gaussian params |
| 11 | `export_gaussians()` | Compression / export | Standard PLY (via nerfstudio) |

## Paper → Extension Point Mapping

This table shows which extension points each major 3DGS paper modifies:

| Paper | Init | Repr | Render | Densify | Loss | Other |
|-------|:----:|:----:|:------:|:-------:|:----:|:-----:|
| **Original 3DGS** (Kerbl 2023) | SfM | SH deg3 | classic | ADC | L1+SSIM | — |
| **Mip-Splatting** (Yu 2024) | — | — | `antialiased` | — | — | 3D filter param |
| **2D GS** (Huang 2024) | — | 2D disks | `rasterization_2dgs` | — | depth distort + normal | normals output |
| **AbsGS** (Ye 2024) | — | — | `absgrad=True` | absgrad ADC | — | — |
| **Scaffold-GS** (Lu 2024) | — | anchor+MLP | neural decode | anchor grow | — | MLP params |
| **Mini-Splatting** (Fang 2024) | — | — | — | multi-phase | — | blur-split |
| **Compact3D** (Lee 2024) | — | low SH | — | — | opacity reg | VQ compress |
| **GaussianPro** (Cheng 2024) | — | — | — | MVS propag | planar | periodic propag |
| **GOF** (Yu 2024) | — | — | normal output | combined grad | depth distort + normal | mesh extract |
| **FreGS** (Zhang 2024) | — | — | — | — | freq spectrum | freq annealing |
| **Splatfacto-W** (Xu 2024) | — | +appear embed | — | — | — | SH background |
| **Depth-3DGS** | — | — | — | — | depth supervision | — |

## Implementation Examples

### Example 1: Mip-Splatting (minimal — config only)

For Mip-Splatting, you only need to change one config flag:

```python
# mip_splatting/config.py
from gs_template.model import GSTemplateModelConfig

mip_splatting_method = MethodSpecification(
    config=TrainerConfig(
        method_name="mip-splatting",
        pipeline=VanillaPipelineConfig(
            model=GSTemplateModelConfig(
                rasterize_mode="antialiased",  # ← This is the key change
            ),
            # ... rest same as gs-template
        ),
    ),
)
```

### Example 2: Depth-Supervised 3DGS (loss only)

```python
# depth_3dgs/model.py
from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import DepthLoss

class Depth3DGSModel(GSTemplateModel):
    def populate_modules(self):
        super().populate_modules()
        self.depth_loss_fn = DepthLoss(mode="l1")

    def compute_regularization_losses(self, outputs, batch):
        losses = super().compute_regularization_losses(outputs, batch)
        if "depth_image" in batch:
            losses["depth_loss"] = 0.5 * self.depth_loss_fn(
                outputs["depth"], batch["depth_image"]
            )
        return losses
```

### Example 3: 2DGS (rendering + losses)

```python
# two_dgs/model.py
from gsplat import rasterization_2dgs

class TwoDGSModel(GSTemplateModel):
    def rasterize_gaussians(self, means, quats, scales, opacities, colors,
                            viewmat, K, W, H, sh_degree, **kwargs):
        # Use 2DGS rasterizer instead of standard
        colors_out, alphas, normals, surf_normals, distort, med_depth, meta = (
            rasterization_2dgs(
                means=means, quats=quats, scales=scales[:, :2],  # 2D only
                opacities=opacities, colors=colors,
                viewmats=viewmat, Ks=K, width=W, height=H,
                sh_degree=sh_degree,
            )
        )
        # Store extra outputs in meta for loss computation
        meta["normals"] = normals
        meta["distortion"] = distort
        render_colors = torch.cat([colors_out, med_depth], dim=-1)
        return render_colors, alphas, meta

    def get_additional_outputs(self, render_colors, render_alphas, info, camera):
        return {
            "normals": info.get("normals"),
            "distortion": info.get("distortion"),
        }

    def compute_regularization_losses(self, outputs, batch):
        losses = super().compute_regularization_losses(outputs, batch)
        if outputs.get("distortion") is not None:
            losses["distortion"] = 0.1 * outputs["distortion"].mean()
        return losses
```

## Config CLI Arguments

All config fields are exposed as CLI arguments:

```bash
# Adjust SH degree
ns-train gs-template --pipeline.model.sh-degree 2

# Use antialiased rendering (Mip-Splatting)
ns-train gs-template --pipeline.model.rasterize-mode antialiased

# Use absolute gradients (AbsGS)
ns-train gs-template --pipeline.model.use-absgrad True \
                     --pipeline.model.densify-grad-thresh 0.0008

# Enable scale regularization
ns-train gs-template --pipeline.model.use-scale-regularization True

# Train longer
ns-train gs-template --max-num-iterations 50000

# White background (for Blender synthetic scenes)
ns-train gs-template --pipeline.model.background-color white
```

## Gaussian Parameter Layout

All Gaussian attributes are stored in `self.gauss_params` (a `ParameterDict`):

| Parameter | Shape | Storage Space | Activation | Description |
|-----------|-------|--------------|------------|-------------|
| `means` | (N, 3) | World coords | None | 3D position |
| `scales` | (N, 3) | Log-space | `exp()` | Anisotropic scales |
| `quats` | (N, 4) | Unnormalized | `q/‖q‖` | Rotation (wxyz) |
| `features_dc` | (N, 3) | SH space | gsplat eval | Base color (0th SH) |
| `features_rest` | (N, K-1, 3) | SH space | gsplat eval | Higher-order SH |
| `opacities` | (N, 1) | Logit-space | `sigmoid()` | Transparency |

**K** = `(sh_degree + 1)²`. For degree 3: K=16, so features_rest is (N, 15, 3).

## Requirements

- Python ≥ 3.10
- nerfstudio ≥ 1.1.0
- gsplat ≥ 1.0.0
- CUDA-compatible GPU with ≥ 8GB VRAM

## License

Apache 2.0
