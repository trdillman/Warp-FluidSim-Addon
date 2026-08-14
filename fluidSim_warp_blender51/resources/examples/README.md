# FluidSim Phase 2 examples

These files are generated entirely from Blender primitives by
`release/generate_examples.py`. They contain no baked simulation data and no
third-party assets. They are distributed under the project GPL-3.0-or-later
license.

| File | Workflow | Expected cost |
|---|---|---|
| `01_live_pour.blend` | Continuous inflow, Preview Water, basin collider | Low; intended for live blocking |
| `02_moving_collider_splash.blend` | Initial fill and animated velocity-transferring collider | Medium; Production Water settings |
| `03_viscous_honey.blend` | Honey preset and implicit viscosity | High; viscosity increases solve cost |
| `04_whitewater_bake_resume.blend` | Splash + Whitewater with cache-v2 bake range | High; intended for production bake/resume |

## Runtime requirements

The authored scenes target Blender 5.2.0 LTS, Linux x64, NVIDIA CUDA, and the
externally provisioned NVIDIA Warp 1.16.0+cu12 runtime. Opening and inspecting
the scenes does not require CUDA. Initializing or advancing the solver does.

## Regeneration

From the repository root, with the source tree importable as `fluidsim_dev`:

```bash
/path/to/supplied/blender --background --factory-startup \
  --python-use-system-env --python release/generate_examples.py -- \
  --module fluidsim_dev --output resources/examples \
  --report-json /tmp/fluidsim-example-generation.json
```

The generator performs only registration, guided setup, authored animation,
and `.blend` writes. It does not execute solver kernels or claim CUDA
validation.
