# FluidSim 1.0.251 examples

These files are generated entirely from Blender primitives by
`../tools/release/generate_examples.py`. They contain no baked simulation data and no
third-party assets. They are distributed under the project GPL-3.0-or-later
license.

| File | Workflow | Expected cost |
|---|---|---|
| `01_live_pour.blend` | Continuous inflow, Preview Water, basin collider | Low; intended for live blocking |
| `02_moving_collider_splash.blend` | Initial fill and animated velocity-transferring collider | Medium; Production Water settings |
| `03_viscous_honey.blend` | Honey preset and implicit viscosity | High; viscosity increases solve cost |
| `04_whitewater_bake_resume.blend` | Typed foam/bubble/spray with combined + separate outputs and cache-v2 bake range | High; intended for production bake/resume |
| `05_custom_effects_vortex.blend` | Artist-facing Vortex Force with controller and supplied-Warp numerical evidence | Low; Custom Effects workflow demonstration |
| `06_custom_effects_stack.blend` | Ordered Radial Pulse and Velocity Damping stack with analytic energy comparison | Low; demonstrates order sensitivity and deterministic execution |
| `07_warp_code_authoring.blend` | Developer-authored Lateral Wave Text datablock, trust state, and package workflow | Low; Warp Code authoring and trust-boundary demonstration |

## Runtime requirements

The authored scenes target Blender 5.2.0 LTS, Linux x64, NVIDIA CUDA, and the
externally provisioned NVIDIA Warp 1.16.0+cu12 runtime. Opening and inspecting
the scenes does not require CUDA. Initializing or advancing the solver does.

## Regeneration

From the repository root, with the source tree importable as a package name:

```bash
/path/to/supplied/blender --background --factory-startup \
  --python-use-system-env --python ../tools/release/generate_examples.py -- \
  --module fluidsim_phase5_repo --output resources/examples \
  --report-json /tmp/fluidsim-example-generation.json
```

The generator performs only registration, guided setup, authored animation,
and `.blend` writes. It does not execute solver kernels or claim CUDA
validation.
