# FluidSim 1.0.251 examples

These files are generated entirely from Blender primitives by
`../tools/release/generate_examples.py`. They contain no baked simulation data and no
third-party assets. They are distributed under the project GPL-3.0-or-later
license.

These are distributed workflow and validation **examples**, not the canonical
playable demos under the repository's `demos/` directory. In particular,
`05`-`07` are static Custom Effects validation/review scenes built from
precomputed numerical fixtures. They do not contain a live FluidSim Domain and
must not be presented as simulation demos. See `docs/DEMO_SCENE_CONTRACT.md` in
the repository root for the reserved demo definition.

| File | Workflow | Expected cost |
|---|---|---|
| `01_live_pour.blend` | Continuous inflow, Preview Water, basin collider | Low; intended for live blocking |
| `02_moving_collider_splash.blend` | Initial fill and animated velocity-transferring collider | Medium; Production Water settings |
| `03_viscous_honey.blend` | Honey preset and implicit viscosity | High; viscosity increases solve cost |
| `04_whitewater_bake_resume.blend` | Typed foam/bubble/spray with combined + separate outputs and cache-v3 bake range | High; intended for production bake/resume |
| `05_custom_effects_vortex.blend` | Static Vortex Force before/after validation fixture with controller and supplied-Warp numerical evidence | Low; review only, no live domain or solver playback |
| `06_custom_effects_stack.blend` | Static ordered-effect fixture with analytic energy comparison | Low; review only, no live domain or solver playback |
| `07_warp_code_authoring.blend` | Static Warp Code package/trust review scene | Low; authoring evidence only, no live domain or solver playback |

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
