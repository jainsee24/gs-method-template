"""
gs_template/model.py — Core Gaussian Splatting Model Template
=============================================================

This is the central file of the template. It implements a fully functional
Gaussian Splatting model that trains, converges, and renders correctly
out-of-the-box, while exposing 11 clearly-marked extension points that
paper authors override to implement their novel contributions.

ARCHITECTURE:
    GSTemplateModel extends nerfstudio's Model base class. Unlike NeRF models
    that process RayBundles, GS models receive a Cameras object and rasterize
    full images using gsplat's CUDA-accelerated rasterizer.

    The model stores all Gaussian attributes in a torch.nn.ParameterDict
    called `gauss_params`. Each parameter is stored in an UNCONSTRAINED space
    (log-space for scales, logit-space for opacities, unnormalized quaternions)
    and activated at render time. This ensures stable gradient optimization.

GAUSSIAN PARAMETER STORAGE:
    gauss_params["means"]         → (N, 3)    Gaussian centers in world space
    gauss_params["scales"]        → (N, 3)    Log-space scales (exp → positive)
    gauss_params["quats"]         → (N, 4)    Unnormalized quaternions (wxyz)
    gauss_params["opacities"]     → (N, 1)    Logit-space opacities (sigmoid → [0,1])
    gauss_params["features_dc"]   → (N, 3)    0th-order SH coefficients (base color)
    gauss_params["features_rest"] → (N, K-1, 3) Higher-order SH coefficients

RENDERING PIPELINE (get_outputs):
    1. Extract camera parameters → build viewmat (world-to-camera) and K (intrinsics)
    2. Activate parameters: exp(scales), sigmoid(opacities), normalize(quats)
    3. Assemble SH coefficients via compute_colors()
    4. Call gsplat.rasterization() → render_colors, render_alphas, meta
    5. Composite with background
    6. Return {"rgb", "depth", "accumulation", "background", ...}

EXTENSION POINTS (search for "EXTENSION POINT" to find them all):
    Each extension point is a method that can be overridden in a subclass.
    The default implementation provides standard 3DGS behavior.

EXAMPLE — Implementing Mip-Splatting:
    class MipSplattingModel(GSTemplateModel):
        config: MipSplattingModelConfig

        def populate_modules(self):
            super().populate_modules()
            # Add 3D smoothing filter parameter
            self.filter_3d = torch.nn.Parameter(torch.zeros(self.num_points, 1))

        def rasterize_gaussians(self, means, quats, scales, opacities, colors,
                                viewmat, K, W, H, sh_degree, **kwargs):
            # Use antialiased mode (2D Mip filter built into gsplat)
            return rasterization(
                means=means, quats=quats,
                scales=scales * (1 + torch.exp(self.filter_3d)),  # 3D filter
                opacities=opacities, colors=colors,
                viewmats=viewmat, Ks=K, width=W, height=H,
                sh_degree=sh_degree, render_mode="RGB+ED",
                rasterize_mode="antialiased",  # key change for Mip-Splatting
            )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.models.base_model import Model, ModelConfig

# Local imports
from gs_template.losses import SSIMLoss
from gs_template.appearance import SphericalHarmonicsAppearance
from gs_template.background import BackgroundModel


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def num_sh_bases(degree: int) -> int:
    """Number of SH basis functions for a given degree.

    Degree 0 → 1 basis  (constant color)
    Degree 1 → 4 bases  (+ directional)
    Degree 2 → 9 bases  (+ specular-like)
    Degree 3 → 16 bases (full view-dependent)

    Most papers use degree 3 for best quality, degree 0 for fastest training.
    """
    return (degree + 1) ** 2


def RGB2SH(rgb: Tensor) -> Tensor:
    """Convert linear RGB [0,1] to 0th-order SH coefficient.

    The 0th SH basis Y_0^0 = 0.28209... (the C0 constant).
    To store color as SH: sh_dc = (rgb - 0.5) / C0
    To recover color from SH: rgb = sh_dc * C0 + 0.5

    Args:
        rgb: (N, 3) tensor of RGB colors in [0, 1].

    Returns:
        (N, 3) tensor of 0th-order SH coefficients.
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh: Tensor) -> Tensor:
    """Convert 0th-order SH coefficient back to linear RGB.

    Args:
        sh: (N, 3) tensor of 0th-order SH coefficients.

    Returns:
        (N, 3) tensor of RGB colors in [0, 1].
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def random_quat_tensor(N: int) -> Tensor:
    """Generate N uniformly random unit quaternions (wxyz convention).

    Uses the Shoemake uniform random rotation method. gsplat expects
    quaternions in (w, x, y, z) format.

    Args:
        N: Number of quaternions to generate.

    Returns:
        (N, 4) tensor of random unit quaternions.
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack([
        torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
        torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
        torch.sqrt(u) * torch.sin(2 * math.pi * w),
        torch.sqrt(u) * torch.cos(2 * math.pi * w),
    ], dim=-1)


def k_nearest_sklearn(data: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Find k nearest neighbors for each point using sklearn.

    Used during initialization to set initial Gaussian scales from
    the average distance to the k nearest SfM points.

    Args:
        data: (N, 3) point positions.
        k: Number of neighbors to find.

    Returns:
        Tuple of (distances, indices) each of shape (N, k).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(data.cpu().numpy())
    distances, indices = nn.kneighbors(data.cpu().numpy())
    # Skip self (index 0) — distance to self is 0
    return (
        torch.tensor(distances[:, 1:], dtype=torch.float32),
        torch.tensor(indices[:, 1:], dtype=torch.long),
    )


# ============================================================================
# MODEL CONFIG
# ============================================================================

@dataclass
class GSTemplateModelConfig(ModelConfig):
    """Configuration for the Gaussian Splatting template model.

    This config controls ALL aspects of Gaussian Splatting behavior.
    Paper authors should:
      1. Add new config fields for their paper's hyperparameters
      2. Set appropriate defaults that reproduce their paper's results
      3. Document which paper/equation each parameter corresponds to

    Config fields are automatically exposed as CLI arguments:
        ns-train gs-template --pipeline.model.sh-degree 2 --pipeline.model.ssim-lambda 0.5
    """

    # The class that will be instantiated from this config.
    # Override this in your subclass config to point to your model class.
    _target: Type = field(default_factory=lambda: GSTemplateModel)

    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    # Controls how Gaussians are initialized at the start of training.
    # Most papers use SfM points from COLMAP. Some papers propose novel
    # initialization strategies (depth-guided, dense matching, etc.)

    random_init: bool = False
    """If True, initialize Gaussians randomly instead of from SfM points.
    Set to True only if your paper proposes a random initialization strategy
    or if testing without COLMAP data."""

    num_random: int = 50_000
    """Number of random Gaussians to initialize when random_init=True.
    Ignored when using SfM initialization."""

    random_scale: float = 10.0
    """Scale factor for random initialization volume.
    Points are sampled in [-random_scale/2, random_scale/2]^3."""

    initial_opacity: float = 0.1
    """Initial opacity value for all Gaussians (before logit transform).
    Range (0, 1). Higher values (e.g., 0.5) for MCMC, lower (0.1) for ADC."""

    # ========================================================================
    # REPRESENTATION — Spherical Harmonics
    # ========================================================================
    # Controls the SH degree used for view-dependent color.
    # Degree 0 = diffuse only (1 coeff), Degree 3 = full specular (16 coeffs).
    # Many papers start with degree 0 and increase during training.

    sh_degree: int = 3
    """Maximum spherical harmonics degree for color representation.
    0=diffuse, 1=+directional, 2=+specular-like, 3=full (original 3DGS).
    
    Papers that reduce SH degree:
    - Compact3D: Uses degree 0-1 for compression
    - Some few-shot methods: Lower degree to prevent overfitting"""

    sh_degree_interval: int = 1000
    """Steps between SH degree increases during progressive training.
    At step 0, SH degree = 0. At step sh_degree_interval, degree = 1, etc.
    Set to 0 to use full SH degree from the start."""

    # ========================================================================
    # RENDERING
    # ========================================================================
    # Controls the gsplat rasterization backend behavior.

    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """Rasterization mode:
    - 'classic': Standard 3DGS rasterization (Kerbl et al., 2023)
    - 'antialiased': Mip-Splatting-style 2D anti-aliasing filter
      (computes opacity compensation: ρ = √(Det(Σ)/Det(Σ+εI)))
    
    Set to 'antialiased' for: Mip-Splatting, any method needing alias-free rendering."""

    render_mode: Literal["RGB", "RGB+D", "RGB+ED"] = "RGB+ED"
    """What gsplat renders:
    - 'RGB': Color only (fastest)
    - 'RGB+D': Color + depth (per-Gaussian depth)
    - 'RGB+ED': Color + expected depth (alpha-weighted, most useful)
    
    Use 'RGB+ED' if your method needs depth output for losses or visualization."""

    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    """Camera model for rasterization. Most datasets use pinhole."""

    near_plane: float = 0.01
    """Near clipping plane. Gaussians closer than this are culled."""

    far_plane: float = 1e10
    """Far clipping plane. Gaussians farther than this are culled."""

    # ========================================================================
    # BACKGROUND
    # ========================================================================

    background_color: Literal["random", "black", "white"] = "random"
    """Background color composited behind the Gaussians.
    - 'random': Random color per training step (regularization, recommended)
    - 'black': Black background (for synthetic scenes with known background)
    - 'white': White background (for some synthetic datasets)
    
    Random is recommended during training as it prevents Gaussians from
    'hiding' against a fixed background color."""

    # ========================================================================
    # DENSIFICATION STRATEGY
    # ========================================================================
    # Controls how Gaussians are added, split, cloned, and pruned during
    # training. This is one of the most commonly modified components.

    strategy_type: Literal["default", "mcmc"] = "default"
    """Densification strategy:
    - 'default': ADC (Adaptive Density Control) from original 3DGS
      Uses gradient-based split/clone + opacity-based pruning
    - 'mcmc': Markov Chain Monte Carlo from 3DGS-MCMC paper
      Uses stochastic Langevin dynamics, no split/clone heuristics
    
    Papers that modify densification:
    - AbsGS: Uses absolute gradients (set use_absgrad=True with 'default')
    - Mini-Splatting: Custom blur-split + depth-guided reinit
    - Scaffold-GS: Anchor-based growing (needs custom strategy class)
    - GaussianPro: MVS-guided propagation"""

    # --- ADC (DefaultStrategy) parameters ---
    warmup_length: int = 500
    """Steps before densification begins (refine_start_iter).
    Allows initial Gaussians to settle before splitting/cloning."""

    refine_every: int = 100
    """Densification check frequency (steps between refine operations)."""

    stop_split_at: int = 15_000
    """Step after which no more splitting/cloning occurs.
    Only pruning continues. Usually set to 50% of max_iterations."""

    densify_grad_thresh: float = 0.0002
    """2D gradient magnitude threshold for growing Gaussians.
    Gaussians with avg 2D gradient > this threshold are grown.
    
    AbsGS: Set to 0.0008 and use_absgrad=True for absolute gradients."""

    densify_size_thresh: float = 0.01
    """3D scale threshold for split vs. clone decision:
    - Gaussians with scale > thresh AND high gradient → SPLIT (divide into 2)
    - Gaussians with scale ≤ thresh AND high gradient → CLONE (duplicate)"""

    cull_alpha_thresh: float = 0.1
    """Opacity threshold for pruning. Gaussians with opacity < thresh are removed.
    Lower values keep more semi-transparent Gaussians (higher quality, larger model)."""

    cull_scale_thresh: float = 0.5
    """World-space scale threshold for pruning. Gaussians larger than this
    fraction of the scene extent are removed (prevents giant blobs)."""

    reset_alpha_every: int = 3000
    """Periodically reset all opacities to a low value (forces re-evaluation).
    Set to 0 to disable. Measured in multiples of refine_every."""

    use_absgrad: bool = False
    """Use absolute value of 2D gradient instead of norm for densification.
    From AbsGS (Ye et al., 2024). Requires densify_grad_thresh ≈ 0.0008.
    When True, also passes absgrad=True to gsplat's rasterization()."""

    n_split_samples: int = 2
    """Number of new Gaussians to create when splitting a large Gaussian.
    Standard 3DGS uses 2. Some methods use higher values."""

    # --- MCMC (MCMCStrategy) parameters ---
    mcmc_cap_max: int = 200_000
    """Maximum number of Gaussians for MCMC strategy.
    Unlike ADC which grows without bound, MCMC has a fixed budget."""

    mcmc_noise_lr: float = 5e4
    """Noise learning rate for MCMC Langevin dynamics.
    Controls the magnitude of stochastic perturbation."""

    mcmc_opacity_reg: float = 0.01
    """Opacity regularization weight for MCMC strategy.
    Encourages sparse opacity (pushes Gaussians toward 0 or 1)."""

    mcmc_min_opacity: float = 0.005
    """Minimum opacity for MCMC pruning."""

    # ========================================================================
    # LOSS CONFIGURATION
    # ========================================================================
    # Controls the loss function composition. The base loss is always
    # (1-λ)*L1 + λ*SSIM. Additional regularization terms are added via
    # compute_regularization_losses().

    ssim_lambda: float = 0.2
    """Weight of SSIM loss in the main loss: L = (1-λ)*L1 + λ*SSIM.
    Original 3DGS uses 0.2. Some papers adjust this."""

    # --- Optional regularization ---
    use_scale_regularization: bool = False
    """Penalize Gaussians with extreme aspect ratios (one axis >> others).
    From PhysGaussian. Helps prevent needle-like Gaussians."""

    max_gauss_ratio: float = 10.0
    """Maximum allowed ratio between largest and smallest scale axis.
    Only active when use_scale_regularization=True."""

    use_opacity_regularization: bool = False
    """Penalize intermediate opacities (push toward 0 or 1).
    Useful for Compact3D and other compression methods."""

    opacity_reg_weight: float = 0.01
    """Weight of opacity regularization term."""

    # ========================================================================
    # OUTPUT CONFIGURATION
    # ========================================================================
    output_depth_during_training: bool = True
    """Whether to include depth in training outputs.
    Set to False for speed if depth is not needed for losses."""

    output_normal_during_training: bool = False
    """Whether to compute and output normals during training.
    Needed for: 2DGS, GOF, methods with normal consistency losses.
    Adds computation cost."""


# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================

class GSTemplateModel(Model):
    """Extensible Gaussian Splatting model with 11 well-defined extension points.

    This model implements the complete 3DGS pipeline from initialization through
    rendering and loss computation, with each major component exposed as an
    overridable method.

    The model follows the nerfstudio Model contract:
        populate_modules()        → Called once at construction
        get_param_groups()        → Returns parameter groups for optimizers
        get_training_callbacks()  → Returns training loop hooks
        get_outputs(camera)       → Core rendering (called every training step)
        get_loss_dict(...)        → Loss computation
        get_metrics_dict(...)     → Metrics for logging
        get_image_metrics_and_images(...) → Evaluation metrics and images
    """

    config: GSTemplateModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            seed_points: Optional tuple of (positions: [N,3], colors: [N,3])
                from SfM/COLMAP. Passed by VanillaPipeline from the dataparser's
                metadata["points3D_xyz"] and metadata["points3D_rgb"].
        """
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    # ====================================================================
    # SETUP
    # ====================================================================

    def populate_modules(self) -> None:
        """Initialize all Gaussian parameters and auxiliary modules.

        This is called once during model construction. It:
        1. Calls create_initial_gaussians() to get seed positions and colors
        2. Computes initial scales from KNN distances
        3. Creates the gauss_params ParameterDict
        4. Sets up the densification strategy
        5. Sets up loss and appearance modules

        Override notes:
        - Call super().populate_modules() first, then add your own parameters
        - To add new per-Gaussian parameters, add them to self.gauss_params
        - To add neural network modules, add them as regular nn.Module attributes
        """
        # ==============================================================
        # EXTENSION POINT 1: Initialization Strategy
        # ==============================================================
        means, colors = self.create_initial_gaussians()
        N = means.shape[0]

        # Compute initial scales from k-nearest-neighbor distances
        # This ensures Gaussians start at a size proportional to local point density
        distances, _ = k_nearest_sklearn(means.data, 3)
        avg_dist = distances.mean(dim=-1, keepdim=True)  # (N, 1)

        # Initialize all Gaussian parameters
        # NOTE: All stored in UNCONSTRAINED space for stable optimization
        dim_sh = num_sh_bases(self.config.sh_degree)

        self.gauss_params = torch.nn.ParameterDict(
            {
                # Position in world space (unconstrained)
                "means": torch.nn.Parameter(means),  # (N, 3)
                # Log-scale: actual_scale = exp(scales)
                # Initialized to KNN average distance (log-space)
                "scales": torch.nn.Parameter(
                    torch.log(avg_dist.repeat(1, 3))  # (N, 3)
                ),
                # Quaternion rotation (wxyz, unnormalized)
                # Normalized at render time: q / ||q||
                "quats": torch.nn.Parameter(
                    random_quat_tensor(N)  # (N, 4)
                ),
                # 0th-order SH coefficients (base color)
                # Converted from RGB via: sh = (rgb - 0.5) / C0
                "features_dc": torch.nn.Parameter(
                    RGB2SH(colors)  # (N, 3)
                ),
                # Higher-order SH coefficients (view-dependent effects)
                # Initialized to zero (no view-dependence at start)
                "features_rest": torch.nn.Parameter(
                    torch.zeros(N, dim_sh - 1, 3)  # (N, K-1, 3)
                ),
                # Logit-space opacity: actual_opacity = sigmoid(opacities)
                # Initialized to logit(initial_opacity)
                "opacities": torch.nn.Parameter(
                    torch.logit(
                        self.config.initial_opacity * torch.ones(N, 1)
                    )  # (N, 1)
                ),
            }
        )

        # Store number of points for convenience
        self._num_points = N

        # ==============================================================
        # Setup densification strategy
        # ==============================================================
        self._setup_strategy()

        # ==============================================================
        # Setup loss modules
        # ==============================================================
        self.ssim_loss = SSIMLoss()

        # ==============================================================
        # Setup appearance model (can be overridden)
        # ==============================================================
        self.appearance_model = SphericalHarmonicsAppearance(
            sh_degree=self.config.sh_degree,
            sh_degree_interval=self.config.sh_degree_interval,
        )

        # ==============================================================
        # Setup background model (can be overridden)
        # ==============================================================
        self.background_model = BackgroundModel(
            mode=self.config.background_color
        )

        # Placeholder for render metadata (set in get_outputs, used in callbacks)
        self._render_info: Dict = {}
        self._last_camera: Optional[Cameras] = None

    def _setup_strategy(self) -> None:
        """Initialize the densification strategy based on config.

        The strategy object handles all Gaussian growing/pruning logic.
        gsplat provides two built-in strategies:
        - DefaultStrategy: Standard ADC (split/clone/prune)
        - MCMCStrategy: Stochastic Langevin dynamics

        For custom strategies (e.g., Scaffold-GS anchor growing),
        override this method and set self.strategy to your custom object.
        """
        if self.config.strategy_type == "default":
            self.strategy = DefaultStrategy(
                # When to start/stop densification
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                # Gradient threshold for growing
                grow_grad2d=self.config.densify_grad_thresh,
                # Scale threshold: above → split, below → clone
                grow_scale3d=self.config.densify_size_thresh,
                # Opacity threshold for pruning
                prune_opa=self.config.cull_alpha_thresh,
                # Scale threshold for pruning (remove giant Gaussians)
                prune_scale3d=self.config.cull_scale_thresh,
                # Opacity reset interval
                reset_every=self.config.reset_alpha_every,
                # Number of samples when splitting
                n_split_samples=self.config.n_split_samples,
                # Use absolute gradients (AbsGS)
                absgrad=self.config.use_absgrad,
                verbose=True,
            )
        elif self.config.strategy_type == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.mcmc_cap_max,
                noise_lr=self.config.mcmc_noise_lr,
                refine_every=self.config.refine_every,
                min_opacity=self.config.mcmc_min_opacity,
                verbose=True,
            )
        else:
            raise ValueError(
                f"Unknown strategy type: {self.config.strategy_type}. "
                f"Expected 'default' or 'mcmc'."
            )

        # Initialize strategy state (tracks gradient accumulators, etc.)
        self.strategy_state = self.strategy.initialize_state()

    # ====================================================================
    # PROPERTIES — Convenient access to activated parameters
    # ====================================================================

    @property
    def num_points(self) -> int:
        """Current number of Gaussians (changes during densification)."""
        return self.gauss_params["means"].shape[0]

    @property
    def means(self) -> Tensor:
        """Gaussian centers (N, 3) — no activation needed."""
        return self.gauss_params["means"]

    @property
    def scales(self) -> Tensor:
        """Activated scales (N, 3) — exp(raw_scales)."""
        return self.scale_activation(self.gauss_params["scales"])

    @property
    def quats(self) -> Tensor:
        """Normalized quaternions (N, 4) — q/||q||."""
        q = self.gauss_params["quats"]
        return q / q.norm(dim=-1, keepdim=True)

    @property
    def opacities(self) -> Tensor:
        """Activated opacities (N,) — sigmoid(raw_opacities).
        Note: Returns (N,) not (N,1) as gsplat expects 1D opacities."""
        return self.opacity_activation(self.gauss_params["opacities"])

    # ====================================================================
    # EXTENSION POINT 1: Initialization Strategy
    # ====================================================================

    def create_initial_gaussians(self) -> Tuple[Tensor, Tensor]:
        """Create initial Gaussian positions and colors.

        Returns:
            Tuple of:
                means: (N, 3) float32 tensor of 3D positions
                colors: (N, 3) float32 tensor of RGB colors in [0, 1]

        Override this for custom initialization strategies:
        - Depth-guided initialization (from monocular depth)
        - Dense matching initialization (from RAFT/optical flow)
        - Grid-based initialization
        - Point cloud from other sources (LiDAR, depth sensors)

        Example — Dense random initialization:
            def create_initial_gaussians(self):
                means = torch.rand(100_000, 3) * scene_extent
                colors = torch.ones(100_000, 3) * 0.5  # gray
                return means, colors
        """
        if self.seed_points is not None and not self.config.random_init:
            # SfM initialization (standard path for COLMAP data)
            means = self.seed_points[0].float()  # (N, 3)
            colors = self.seed_points[1].float() / 255.0  # (N, 3) uint8→[0,1]
            colors = colors.clamp(0.0, 1.0)
        else:
            # Random initialization (fallback)
            means = (
                (torch.rand(self.config.num_random, 3) - 0.5)
                * self.config.random_scale
            )
            colors = torch.rand(self.config.num_random, 3)

        return means, colors

    # ====================================================================
    # EXTENSION POINT 2: Activation Functions
    # ====================================================================

    def opacity_activation(self, raw_opacities: Tensor) -> Tensor:
        """Activate raw opacities from logit-space to [0, 1].

        Args:
            raw_opacities: (N, 1) raw parameter values.

        Returns:
            (N,) activated opacities. Note the squeeze — gsplat expects 1D.

        Override for custom activation (e.g., softplus, clamp, etc.)
        """
        return torch.sigmoid(raw_opacities).squeeze(-1)

    def scale_activation(self, raw_scales: Tensor) -> Tensor:
        """Activate raw scales from log-space to positive values.

        Args:
            raw_scales: (N, 3) raw parameter values.

        Returns:
            (N, 3) positive scale values.

        Override for:
        - Lower-bounded scales (Mip-Splatting 3D filter)
        - 2D Gaussians (return only 2 scale dimensions)
        - Clamped scales for stability
        """
        return torch.exp(raw_scales)

    # ====================================================================
    # EXTENSION POINT 3: Rasterization Backend
    # ====================================================================

    def rasterize_gaussians(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        viewmat: Tensor,
        K: Tensor,
        W: int,
        H: int,
        sh_degree: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize Gaussians to produce an image.

        This wraps gsplat.rasterization() and is the main rendering call.

        Args:
            means: (N, 3) Gaussian centers
            quats: (N, 4) normalized quaternions
            scales: (N, 3) positive scales
            opacities: (N,) opacities in [0, 1]
            colors: (N, K, 3) SH coefficients or (N, D) pre-computed features
            viewmat: (1, 4, 4) world-to-camera matrix
            K: (1, 3, 3) camera intrinsics
            W, H: image dimensions
            sh_degree: SH degree to use (may be less than max during progressive training)

        Returns:
            Tuple of:
                render_colors: (1, H, W, X) rendered image (X depends on render_mode)
                render_alphas: (1, H, W, 1) accumulated alpha
                meta: Dict of intermediate tensors for densification

        Override this for:
        - 2DGS: Use gsplat.rasterization_2dgs() instead
        - Custom rendering kernels
        - Multi-pass rendering
        - Deferred shading

        Example — 2D Gaussian Splatting:
            from gsplat import rasterization_2dgs
            def rasterize_gaussians(self, ...):
                colors, alphas, normals, surf_normals, distort, median_depth, meta = (
                    rasterization_2dgs(means=means, quats=quats, ...)
                )
                # Package into standard format
                render_colors = torch.cat([colors, median_depth], dim=-1)
                meta["normals"] = normals
                meta["distortion"] = distort
                return render_colors, alphas, meta
        """
        return rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            sh_degree=sh_degree,
            render_mode=self.config.render_mode,
            rasterize_mode=self.config.rasterize_mode,
            camera_model=self.config.camera_model,
            # absgrad must be True for DefaultStrategy with use_absgrad
            absgrad=(
                self.config.use_absgrad
                and self.config.strategy_type == "default"
            ),
        )

    # ====================================================================
    # EXTENSION POINT 4: Appearance / Color Model
    # ====================================================================

    def compute_colors(
        self, camera: Cameras, step: int
    ) -> Tuple[Tensor, int]:
        """Compute color features for rasterization.

        Returns SH coefficients by default. Override for:
        - Per-image appearance embeddings (Splatfacto-W)
        - Neural feature decoding (Scaffold-GS)
        - Bilateral grid correction
        - Learned color features (non-SH)

        Args:
            camera: Current training camera.
            step: Current training step (for progressive SH scheduling).

        Returns:
            Tuple of:
                colors: (N, K, 3) SH coefficients, or (N, D) features
                sh_degree: SH degree to use (pass to rasterize_gaussians)
                    Set to None if colors are pre-computed (not SH).

        Example — With appearance embedding:
            def compute_colors(self, camera, step):
                colors, sh_degree = super().compute_colors(camera, step)
                # Add per-image appearance embedding
                cam_idx = camera.metadata["cam_idx"]
                embed = self.appearance_embed(cam_idx)  # (1, D)
                # Modulate DC term
                colors[:, 0, :] += embed.expand(N, -1)
                return colors, sh_degree
        """
        # Progressive SH degree scheduling:
        # Start with degree 0, increase every sh_degree_interval steps
        if self.config.sh_degree_interval > 0:
            sh_degree_to_use = min(
                step // self.config.sh_degree_interval,
                self.config.sh_degree,
            )
        else:
            sh_degree_to_use = self.config.sh_degree

        # Assemble SH coefficients: [DC, rest_up_to_current_degree]
        k = num_sh_bases(sh_degree_to_use)
        colors = torch.cat(
            [
                self.gauss_params["features_dc"][:, None, :],  # (N, 1, 3)
                self.gauss_params["features_rest"][:, : k - 1, :],  # (N, k-1, 3)
            ],
            dim=1,
        )  # (N, k, 3)

        return colors, sh_degree_to_use

    # ====================================================================
    # EXTENSION POINT 5: Background Model
    # ====================================================================

    def compute_background(self, H: int, W: int) -> Tensor:
        """Compute background color for compositing.

        The rendered image is composited as: rgb + (1 - alpha) * background

        Args:
            H, W: Image dimensions.

        Returns:
            (3,) or (H, W, 3) background color tensor on self.device.

        Override for:
        - Learned background MLP
        - SH environment map (Splatfacto-W)
        - Per-pixel background
        """
        return self.background_model(H, W, self.device, self.training)

    # ====================================================================
    # CORE: get_outputs — The main rendering method
    # ====================================================================

    def get_outputs(
        self, camera: Cameras
    ) -> Dict[str, Union[Tensor, List]]:
        """Rasterize Gaussians for a given camera and return outputs.

        This is called every training step by the pipeline. It:
        1. Converts nerfstudio Camera → gsplat viewmat + intrinsics
        2. Activates Gaussian parameters
        3. Computes colors via the appearance model
        4. Calls rasterize_gaussians()
        5. Composites with background
        6. Calls get_additional_outputs() for any extra outputs

        Args:
            camera: Cameras object from the datamanager.
                Contains camera_to_worlds (3,4), fx, fy, cx, cy, etc.

        Returns:
            Dict with at least:
                "rgb": (H, W, 3) rendered RGB image
                "depth": (H, W, 1) rendered depth (if configured)
                "accumulation": (H, W, 1) alpha accumulation
                "background": (3,) or (H, W, 3) background color
            Additional keys from get_additional_outputs().
        """
        # Handle empty Gaussians (shouldn't happen but defensive)
        if self.num_points == 0:
            return self._get_empty_outputs(camera)

        # Get image dimensions
        W = int(camera.width[0, 0].item())
        H = int(camera.height[0, 0].item())

        # ---- Camera conversion ----
        # Nerfstudio uses OpenGL convention (camera_to_worlds as [R|t])
        # gsplat expects OpenCV convention world-to-camera matrix
        viewmat = self._camera_to_viewmat(camera)  # (1, 4, 4)
        K = self._camera_to_intrinsics(camera)  # (1, 3, 3)

        # ---- Activate parameters ----
        means = self.means                       # (N, 3) — no activation
        quats = self.quats                       # (N, 4) — normalized
        scales = self.scales                     # (N, 3) — exp activated
        opacities = self.opacities               # (N,)   — sigmoid activated

        # ---- Appearance model ----
        colors, sh_degree = self.compute_colors(camera, self.step)

        # ---- Background ----
        background = self.compute_background(H, W)

        # ---- RASTERIZE ----
        render_colors, render_alphas, info = self.rasterize_gaussians(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmat=viewmat,
            K=K,
            W=W,
            H=H,
            sh_degree=sh_degree,
        )

        # ---- Extract outputs ----
        # render_colors shape depends on render_mode:
        #   "RGB"    → (1, H, W, 3)
        #   "RGB+ED" → (1, H, W, 4)  [rgb(3) + expected_depth(1)]
        #   "RGB+D"  → (1, H, W, 4)  [rgb(3) + depth(1)]
        rgb = render_colors[0, ..., :3]  # (H, W, 3)
        alpha = render_alphas[0]  # (H, W, 1)

        # Extract depth if available
        if render_colors.shape[-1] > 3:
            depth = render_colors[0, ..., 3:4]  # (H, W, 1)
        else:
            depth = torch.zeros(H, W, 1, device=self.device)

        # ---- Background compositing ----
        # rgb_final = rgb_rendered + (1 - alpha) * background_color
        if background.dim() == 1:
            rgb = rgb + (1.0 - alpha) * background[None, None, :]
        else:
            rgb = rgb + (1.0 - alpha) * background

        # ---- Store metadata for densification strategy ----
        # The strategy's step_pre_backward / step_post_backward need
        # the render info (2D means, radii, etc.) from rasterization.
        self._render_info = info
        self._last_camera = camera

        # ---- Build output dict ----
        outputs = {
            "rgb": rgb,                # (H, W, 3)
            "depth": depth,            # (H, W, 1)
            "accumulation": alpha,     # (H, W, 1)
            "background": background,  # (3,) or (H, W, 3)
        }

        # ==============================================================
        # EXTENSION POINT 6: Additional Render Outputs
        # ==============================================================
        outputs.update(
            self.get_additional_outputs(render_colors, render_alphas, info, camera)
        )

        return outputs

    def get_additional_outputs(
        self,
        render_colors: Tensor,
        render_alphas: Tensor,
        info: Dict,
        camera: Cameras,
    ) -> Dict[str, Tensor]:
        """Compute additional outputs beyond RGB/depth/alpha.

        Override this to add method-specific outputs that are needed
        for custom losses or visualization.

        Args:
            render_colors: (1, H, W, X) raw rasterization output
            render_alphas: (1, H, W, 1) alpha accumulation
            info: Dict of intermediate tensors from gsplat
            camera: Current camera

        Returns:
            Dict of additional output tensors.

        Override for:
        - Normal maps (2DGS, GOF)
        - Distortion maps (2DGS)
        - Median depth (2DGS)
        - Feature maps for downstream tasks
        """
        return {}

    # ====================================================================
    # EXTENSION POINT 7: Loss Computation
    # ====================================================================

    def get_loss_dict(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute all losses.

        The total loss is the sum of all values in the returned dict.
        Nerfstudio automatically sums them and calls backward.

        Base loss: (1-λ)*L1 + λ*SSIM
        Plus any regularization from compute_regularization_losses().

        Args:
            outputs: Dict from get_outputs()
            batch: Dict from datamanager with "image" key
            metrics_dict: Optional metrics (not used in base implementation)

        Returns:
            Dict of named loss tensors. All will be summed for backprop.
        """
        # Get ground truth image and composite with same background
        gt_rgb = self._composite_gt_with_background(
            batch["image"], outputs["background"]
        )
        pred_rgb = outputs["rgb"]

        # ---- Main loss: (1-λ)*L1 + λ*SSIM ----
        Ll1 = torch.abs(gt_rgb - pred_rgb).mean()
        simloss = 1.0 - self.ssim_loss(gt_rgb, pred_rgb)

        main_loss = (
            (1.0 - self.config.ssim_lambda) * Ll1
            + self.config.ssim_lambda * simloss
        )

        loss_dict = {"main_loss": main_loss}

        # ---- MCMC opacity regularization ----
        if (
            self.config.strategy_type == "mcmc"
            and self.config.mcmc_opacity_reg > 0.0
        ):
            loss_dict["opacity_reg"] = (
                self.config.mcmc_opacity_reg
                * torch.abs(
                    torch.sigmoid(self.gauss_params["opacities"])
                ).mean()
            )

        # ==============================================================
        # EXTENSION POINT 8: Custom Regularization Losses
        # ==============================================================
        loss_dict.update(
            self.compute_regularization_losses(outputs, batch)
        )

        return loss_dict

    def compute_regularization_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute additional regularization losses.

        This is the MOST COMMONLY OVERRIDDEN extension point. Nearly half
        of 3DGS papers add only new loss terms without changing anything else.

        Args:
            outputs: Dict from get_outputs()
            batch: Dict from datamanager

        Returns:
            Dict of named regularization loss tensors.

        Override for:
        - Depth supervision losses (depth-supervised 3DGS)
        - Normal consistency losses (2DGS, GOF)
        - Depth distortion loss (2DGS)
        - Frequency-domain losses (FreGS)
        - Opacity regularization (Compact3D)
        - Planar constraint losses (GaussianPro)
        - Scale regularization (PhysGaussian)

        Example — Adding depth supervision:
            def compute_regularization_losses(self, outputs, batch):
                losses = super().compute_regularization_losses(outputs, batch)
                if "depth_image" in batch:
                    depth_loss = F.l1_loss(
                        outputs["depth"].squeeze(),
                        batch["depth_image"].squeeze(),
                    )
                    losses["depth_loss"] = 0.5 * depth_loss
                return losses
        """
        losses: Dict[str, Tensor] = {}

        # Scale regularization (PhysGaussian)
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = self.scales
            scale_ratio = (
                scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1).clamp(min=1e-8)
            )
            mask = scale_ratio > self.config.max_gauss_ratio
            if mask.any():
                losses["scale_reg"] = 0.01 * scale_ratio[mask].mean()

        # Opacity regularization (push toward binary)
        if self.config.use_opacity_regularization:
            opa = torch.sigmoid(self.gauss_params["opacities"])
            # Binary cross-entropy with uniform prior encourages 0/1
            losses["opacity_reg"] = self.config.opacity_reg_weight * (
                -opa * torch.log(opa + 1e-10)
                - (1 - opa) * torch.log(1 - opa + 1e-10)
            ).mean()

        return losses

    # ====================================================================
    # EXTENSION POINT 9: Training Callbacks
    # ====================================================================

    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> List[TrainingCallback]:
        """Return training loop callbacks for densification and scheduling.

        The densification strategy requires two hooks:
        1. BEFORE_TRAIN_ITERATION: strategy.step_pre_backward()
           (DefaultStrategy only — updates gradient accumulators)
        2. AFTER_TRAIN_ITERATION: strategy.step_post_backward()
           (Both strategies — performs actual split/clone/prune or MCMC updates)

        Override to add:
        - SH degree annealing callbacks
        - Frequency annealing (FreGS)
        - Multi-phase training schedules (Mini-Splatting)
        - Periodic propagation (GaussianPro)
        - Quantization-aware training hooks (Compact3D)
        - Learning rate warmup/scheduling

        Example — Adding progressive frequency annealing:
            def get_training_callbacks(self, attrs):
                cbs = super().get_training_callbacks(attrs)
                def update_freq_band(step):
                    self.current_freq_band = min(step / 10000, 1.0)
                cbs.append(TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    func=update_freq_band,
                ))
                return cbs
        """
        callbacks: List[TrainingCallback] = []

        # ---- Densification: Pre-backward step ----
        # DefaultStrategy needs this to capture 2D gradient info
        # MCMCStrategy does NOT have step_pre_backward
        if hasattr(self.strategy, "step_pre_backward"):

            def _pre_backward(step: int) -> None:
                if self.step < 2:
                    return  # Skip first steps (no render info yet)
                self.strategy.step_pre_backward(
                    params=self.gauss_params,
                    optimizers=training_callback_attributes.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=self._render_info,
                )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[
                        TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                    ],
                    func=_pre_backward,
                )
            )

        # ---- Densification: Post-backward step ----
        # Both strategies use this for actual densification operations

        def _post_backward(step: int) -> None:
            if self.step < 2:
                return
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=training_callback_attributes.optimizers,
                state=self.strategy_state,
                step=step,
                info=self._render_info,
            )

        callbacks.append(
            TrainingCallback(
                where_to_run=[
                    TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                ],
                func=_post_backward,
            )
        )

        return callbacks

    # ====================================================================
    # EXTENSION POINT 10: Export / Compression
    # ====================================================================

    def export_gaussians(self, output_dir: str) -> None:
        """Export Gaussians for external viewers or compressed storage.

        Override for:
        - Vector quantization export (Compact3D)
        - SH pruning (remove unused higher-order SH)
        - Bit quantization
        - Custom PLY format

        The default nerfstudio exporter handles standard PLY export,
        so this is only needed for custom formats.
        """
        # Default: no custom export (nerfstudio handles PLY)
        pass

    # ====================================================================
    # PARAMETER GROUPS
    # ====================================================================

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return parameter groups for the optimizer.

        Each group name must match a key in the config's optimizers dict.
        The optimizer config in config.py maps:
            "means"         → AdamOptimizer(lr=1.6e-4, ...)
            "features_dc"   → AdamOptimizer(lr=0.0025, ...)
            "features_rest"  → AdamOptimizer(lr=0.000125, ...)
            etc.

        Override if you add new parameter groups (e.g., MLP parameters,
        appearance embeddings) that need separate learning rates.

        Example — Adding appearance embedding parameters:
            def get_param_groups(self):
                groups = super().get_param_groups()
                groups["appearance_embed"] = list(self.appearance_embed.parameters())
                return groups
            # Don't forget to add "appearance_embed" to config.py optimizers!
        """
        return {
            name: [param]
            for name, param in self.gauss_params.items()
        }

    # ====================================================================
    # METRICS
    # ====================================================================

    def get_metrics_dict(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Compute training metrics for logging (tensorboard/wandb).

        These are logged every step but NOT used for backprop.

        Returns:
            Dict of scalar metrics.
        """
        gt_rgb = self._composite_gt_with_background(
            batch["image"], outputs["background"]
        )

        metrics = {
            "psnr": self._compute_psnr(outputs["rgb"], gt_rgb),
            "num_gaussians": torch.tensor(
                self.num_points, dtype=torch.float32
            ),
        }
        return metrics

    def get_image_metrics_and_images(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        """Compute evaluation metrics and comparison images.

        Called during evaluation (steps_per_eval_image). Returns both
        scalar metrics and visualization images.

        Returns:
            Tuple of:
                metrics: Dict of float metrics (PSNR, SSIM, LPIPS)
                images: Dict of (H, W, C) tensors for visualization
        """
        gt_rgb = self._composite_gt_with_background(
            batch["image"], outputs["background"]
        )
        pred_rgb = outputs["rgb"]

        # Compute metrics
        psnr_val = float(self._compute_psnr(pred_rgb, gt_rgb))
        ssim_val = float(self.ssim_loss.compute_ssim(pred_rgb, gt_rgb))

        metrics = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "num_gaussians": float(self.num_points),
        }

        # Build side-by-side comparison images
        images = {
            "img": torch.cat([gt_rgb, pred_rgb], dim=1),  # (H, 2W, 3)
        }

        # Add depth visualization if available
        if "depth" in outputs and outputs["depth"] is not None:
            depth = outputs["depth"]
            depth_vis = (depth - depth.min()) / (
                depth.max() - depth.min() + 1e-8
            )
            images["depth"] = depth_vis.repeat(1, 1, 3)  # (H, W, 3)

        # Add accumulation visualization
        if "accumulation" in outputs:
            images["accumulation"] = outputs["accumulation"].repeat(
                1, 1, 3
            )

        return metrics, images

    # ====================================================================
    # CAMERA-BASED RENDERING (for viewer and evaluation)
    # ====================================================================

    def get_outputs_for_camera(
        self,
        camera: Cameras,
        obb_box=None,
    ) -> Dict[str, Tensor]:
        """Render outputs for a given camera (used by viewer and eval).

        For Gaussian Splatting, this simply delegates to get_outputs().
        (NeRF models need special handling for obb_box, but GS doesn't.)
        """
        return self.get_outputs(camera)

    # ====================================================================
    # INTERNAL HELPERS
    # ====================================================================

    def _camera_to_viewmat(self, camera: Cameras) -> Tensor:
        """Convert nerfstudio Camera (OpenGL) to gsplat viewmat (OpenCV).

        Nerfstudio stores camera_to_worlds as (3, 4) [R | t] in OpenGL convention.
        gsplat expects a (4, 4) world-to-camera matrix in OpenCV convention.

        The conversion:
        1. Extract R and t from c2w
        2. Flip Y and Z axes (OpenGL → OpenCV): R *= [1, -1, -1]
        3. Invert: R_inv = R^T, t_inv = -R^T @ t
        4. Build 4x4 homogeneous matrix

        Args:
            camera: Cameras object with camera_to_worlds of shape (1, 3, 4).

        Returns:
            (1, 4, 4) world-to-camera matrix.
        """
        c2w = camera.camera_to_worlds[0]  # (3, 4)

        # Extract rotation and translation
        R = c2w[:3, :3]  # (3, 3)
        T = c2w[:3, 3:4]  # (3, 1)

        # Flip Y and Z for OpenGL → OpenCV convention
        R_flipped = R * torch.tensor(
            [[1, -1, -1]], device=R.device, dtype=R.dtype
        )

        # Invert: world-to-camera = (camera-to-world)^{-1}
        R_inv = R_flipped.T  # (3, 3)
        T_inv = -R_inv @ T  # (3, 1)

        # Build 4x4 homogeneous matrix
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv

        return viewmat.unsqueeze(0)  # (1, 4, 4)

    def _camera_to_intrinsics(self, camera: Cameras) -> Tensor:
        """Extract camera intrinsics matrix K from nerfstudio Camera.

        Args:
            camera: Cameras object with fx, fy, cx, cy.

        Returns:
            (1, 3, 3) intrinsics matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        """
        fx = float(camera.fx[0, 0].item())
        fy = float(camera.fy[0, 0].item())
        cx = float(camera.cx[0, 0].item())
        cy = float(camera.cy[0, 0].item())

        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        return K.unsqueeze(0)  # (1, 3, 3)

    def _composite_gt_with_background(
        self, gt_image: Tensor, background: Tensor
    ) -> Tensor:
        """Composite ground truth image with the same background as rendering.

        For scenes with alpha channels, this ensures GT and prediction use
        the same background for fair comparison.

        Args:
            gt_image: (H, W, 3) or (H, W, 4) ground truth image.
            background: (3,) or (H, W, 3) background color.

        Returns:
            (H, W, 3) composited ground truth image.
        """
        gt = gt_image.to(self.device)

        if gt.shape[-1] == 4:
            # RGBA image — composite with background
            rgb = gt[..., :3]
            alpha = gt[..., 3:4]
            if background.dim() == 1:
                gt = rgb * alpha + (1 - alpha) * background[None, None, :]
            else:
                gt = rgb * alpha + (1 - alpha) * background
        else:
            gt = gt[..., :3]

        return gt

    def _compute_psnr(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute PSNR between predicted and ground truth images.

        Args:
            pred: (H, W, 3) predicted image.
            gt: (H, W, 3) ground truth image.

        Returns:
            Scalar PSNR value.
        """
        mse = torch.mean((pred - gt) ** 2)
        return -10.0 * torch.log10(mse.clamp(min=1e-10))

    def _get_empty_outputs(
        self, camera: Cameras
    ) -> Dict[str, Union[Tensor, List]]:
        """Return empty outputs when there are no Gaussians."""
        W = int(camera.width[0, 0].item())
        H = int(camera.height[0, 0].item())
        background = self.compute_background(H, W)

        return {
            "rgb": background.expand(H, W, -1)
            if background.dim() == 1
            else background,
            "depth": torch.zeros(H, W, 1, device=self.device),
            "accumulation": torch.zeros(H, W, 1, device=self.device),
            "background": background,
        }
