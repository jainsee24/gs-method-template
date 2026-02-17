"""
gs_template.methods — Fully integrated example methods
=======================================================

Each method is a self-contained model + config demonstrating
the template at different complexity levels:

    absgs       ★☆☆☆☆  Config-only (absgrad=True)
    depth_reg   ★★☆☆☆  Custom loss function
    feature_gs  ★★★★☆  New params, rendering, loss, densification
"""

from gs_template.methods.absgs import AbsGSModel, AbsGSModelConfig
from gs_template.methods.depth_reg import DepthRegGSModel, DepthRegGSModelConfig
from gs_template.methods.feature_gs import FeatureGSModel, FeatureGSModelConfig

__all__ = [
    "AbsGSModel",
    "AbsGSModelConfig",
    "DepthRegGSModel",
    "DepthRegGSModelConfig",
    "FeatureGSModel",
    "FeatureGSModelConfig",
]
