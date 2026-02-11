"""
gs_template: A Nerfstudio Plugin Template for Gaussian Splatting Papers
========================================================================

This package provides a structured, extensible template for implementing
Gaussian Splatting (3DGS) research papers as Nerfstudio plugins using the
gsplat rasterization backend.

ARCHITECTURE OVERVIEW:
    config.py        → MethodSpecification + TrainerConfig (entry point)
    model.py         → GSTemplateModel: core model with 11 extension points
    losses.py        → Composable loss modules (L1, SSIM, depth, normal, etc.)
    densification.py → Custom densification strategies beyond ADC/MCMC
    appearance.py    → Color/appearance models (SH, neural features, embeddings)
    background.py    → Background models (constant, learned, env-map)
    compression.py   → Post-training compression and export hooks

EXTENSION POINTS (override these in your subclass):
    1.  create_initial_gaussians()      → Initialization strategy
    2.  get_gaussian_param_names()      → Parameter representation
    3.  rasterize_gaussians()           → Rendering backend
    4.  compute_colors()                → Appearance / color model
    5.  compute_background()            → Background model
    6.  opacity_activation()            → Opacity activation fn
    7.  scale_activation()              → Scale activation fn
    8.  compute_regularization_losses() → Custom loss terms
    9.  get_training_callbacks()        → Training schedule hooks
    10. get_additional_outputs()        → Extra render outputs (normals, etc.)
    11. export_gaussians()              → Compression / export

USAGE:
    pip install -e .
    ns-install-cli
    ns-train gs-template --data <DATA_DIR>
"""

from gs_template.config import gs_template_method, gs_template_mcmc_method

__all__ = ["gs_template_method", "gs_template_mcmc_method"]
