# Context-Free Grammar for Gaussian Splatting Nerfstudio Plugins

This document formalizes the architectural patterns of a Gaussian Splatting
Nerfstudio plugin as a Context-Free Grammar (CFG). This serves as the domain
knowledge base that constrains LLM code generation, ensuring synthesized
code satisfies architectural invariants.

## Grammar Definition

```
<GS_Plugin>         ::= <PyProject> <Package>
<Package>           ::= <InitModule> <ConfigModule> <ModelModule> <AuxModules>*

<PyProject>         ::= ENTRY_POINT(<method_name>, <config_path>:<spec_name>)
<InitModule>        ::= "from" <ConfigModule> "import" <MethodSpec>+

<ConfigModule>      ::= <MethodSpec>+
<MethodSpec>        ::= MethodSpecification(config=<TrainerConfig>, description=STR)

<TrainerConfig>     ::= TrainerConfig(
                            method_name = STR,
                            max_num_iterations = INT,
                            mixed_precision = False,          # INVARIANT: always False for GS
                            pipeline = <PipelineConfig>,
                            optimizers = <OptimizerDict>
                        )

<PipelineConfig>    ::= VanillaPipelineConfig(
                            datamanager = <DataManagerConfig>,
                            model = <ModelConfig>
                        )

<DataManagerConfig> ::= FullImageDatamanagerConfig(          # INVARIANT: full images, not rays
                            dataparser = <DataParserConfig>,
                            cache_images_type = "uint8"
                        )

<DataParserConfig>  ::= NerfstudioDataParserConfig(
                            load_3D_points = True             # INVARIANT: needed for SfM init
                        )

<ModelConfig>       ::= GSTemplateModelConfig(
                            _target = <ModelClass>,
                            <ConfigFields>*
                        )

<OptimizerDict>     ::= {
                            "means"         : <OptimizerEntry>,  # REQUIRED
                            "features_dc"   : <OptimizerEntry>,  # REQUIRED
                            "features_rest"  : <OptimizerEntry>,  # REQUIRED
                            "opacities"     : <OptimizerEntry>,  # REQUIRED
                            "scales"        : <OptimizerEntry>,  # REQUIRED
                            "quats"         : <OptimizerEntry>,  # REQUIRED
                            <CustomParams>*                       # Optional: MLP params, etc.
                        }

<OptimizerEntry>    ::= { "optimizer": AdamOptimizerConfig(lr=FLOAT, eps=1e-15),
                          "scheduler": <Scheduler> | None }

<Scheduler>         ::= ExponentialDecaySchedulerConfig(lr_final=FLOAT, max_steps=INT)
                      | None
```

## Model Grammar

```
<ModelClass>        ::= class <Name>(GSTemplateModel):
                            config: <ModelConfigClass>
                            <PopulateModules>
                            <ExtensionOverrides>*

<PopulateModules>   ::= def populate_modules(self):
                            super().populate_modules()
                            <CustomParams>*
                            <CustomModules>*

<ExtensionOverrides> ::= <InitOverride>
                       | <ActivationOverride>
                       | <RasterizeOverride>
                       | <ColorOverride>
                       | <BackgroundOverride>
                       | <LossOverride>
                       | <OutputOverride>
                       | <CallbackOverride>
                       | <ParamGroupOverride>
                       | <ExportOverride>
```

## Invariants (MUST be satisfied)

```
INV-1: mixed_precision = False
INV-2: DataManager = FullImageDatamanagerConfig (NOT VanillaDataManager)
INV-3: load_3D_points = True (unless random_init = True)
INV-4: get_outputs() receives Cameras, NOT RayBundle
INV-5: gauss_params MUST contain: means, scales, quats, features_dc, features_rest, opacities
INV-6: Each key in gauss_params MUST have a matching entry in optimizers dict
INV-7: opacities stored as (N, 1) logit-space, activated via sigmoid to (N,) for gsplat
INV-8: scales stored as (N, 3) log-space, activated via exp to (N, 3) for gsplat  
INV-9: quats stored as (N, 4) unnormalized, normalized to unit quaternions for gsplat
INV-10: rasterize_gaussians() must return (render_colors, render_alphas, info_dict)
INV-11: _render_info must be stored after each get_outputs() call (needed by strategy)
INV-12: strategy.step_post_backward() must be called in AFTER_TRAIN_ITERATION callback
```

## Rendering Pipeline DAG

```
camera: Cameras
    │
    ├──► _camera_to_viewmat()  ──► viewmat: (1, 4, 4)
    ├──► _camera_to_intrinsics() ──► K: (1, 3, 3)
    │
    ├──► gauss_params["means"]     ──► means: (N, 3)
    ├──► gauss_params["quats"]     ──► normalize ──► quats: (N, 4)
    ├──► gauss_params["scales"]    ──► scale_activation() ──► scales: (N, 3)
    ├──► gauss_params["opacities"] ──► opacity_activation() ──► opacities: (N,)
    │
    ├──► compute_colors(camera, step) ──► colors: (N, K, 3), sh_degree: int
    ├──► compute_background(H, W) ──► background: (3,)
    │
    └──► rasterize_gaussians(means, quats, scales, opacities, colors,
    │                         viewmat, K, W, H, sh_degree)
    │       │
    │       └──► (render_colors, render_alphas, info)
    │
    ├──► extract rgb, depth, alpha from render_colors
    ├──► composite: rgb = rgb + (1 - alpha) * background
    ├──► store info as self._render_info
    │
    └──► outputs = {"rgb", "depth", "accumulation", "background"}
              + get_additional_outputs(render_colors, render_alphas, info, camera)
```

## Loss Composition DAG

```
outputs["rgb"], batch["image"]
    │
    ├──► _composite_gt_with_background() ──► gt_rgb
    │
    ├──► L1 = |pred - gt|.mean()
    ├──► SSIM = 1 - ssim(pred, gt)
    ├──► main_loss = (1-λ)*L1 + λ*SSIM
    │
    ├──► [if mcmc] opacity_reg = mcmc_weight * |sigmoid(opa)|.mean()
    │
    └──► compute_regularization_losses(outputs, batch)
              │
              ├──► [if scale_reg] scale_reg
              ├──► [if opacity_reg] opacity_entropy
              ├──► [OVERRIDE] depth_loss, normal_loss, freq_loss, planar_loss, ...
              │
              └──► loss_dict (all values summed by nerfstudio trainer)
```

## Training Loop Integration

```
for step in range(max_iterations):
    │
    ├──► BEFORE_TRAIN_ITERATION callbacks
    │       └──► strategy.step_pre_backward()  [DefaultStrategy only]
    │
    ├──► camera, batch = datamanager.next_train(step)
    │
    ├──► outputs = model.get_outputs(camera)
    │
    ├──► loss_dict = model.get_loss_dict(outputs, batch)
    │
    ├──► total_loss = sum(loss_dict.values())
    ├──► total_loss.backward()
    │
    ├──► optimizer.step()  [per-parameter-group]
    │
    ├──► AFTER_TRAIN_ITERATION callbacks
    │       └──► strategy.step_post_backward()  [split/clone/prune or MCMC]
    │
    └──► [every N steps] eval, save, log metrics
```
