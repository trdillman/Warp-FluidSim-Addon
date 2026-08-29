# FluidSim Phase 5 Handoff — Surface Tension and Final Foundation Release

## Release state

- Branch: `phase5/surface-tension-and-final-release`
- Version: `1.0.250`
- Base: Phase 4 commit `5974ddab9144bda15a5ea5418b2e81c85f6dee06`
- Supplied Blender: 5.2.0 LTS, Linux x64
- Supplied Warp: 1.16.0+cu12, module version 1.16.0
- Executed Warp device: CPU
- CUDA certification: open; no usable NVIDIA driver/device was exposed
- Delivery policy: no Blender or Warp runtime payload is bundled

## Delivered

Phase 5 adds an optional continuum-surface-force capillary model derived from
the simulation liquid signed-distance field. The implementation computes an
outward SDF normal, signed curvature, a compact cosine interface delta, and a
cell-centred acceleration. It clamps curvature to `2 / voxel_size`, rejects
non-finite force values, excludes collision interiors, and averages acceleration
onto weighted MAC faces before viscosity and pressure projection.

The force is governed by explicit physical units:

- surface-tension coefficient in N/m;
- liquid density in kg/m^3;
- force-band half-width in simulation voxels;
- timestep safety in `(0, 1]`;
- normalized seven-point normal-smoothing iterations.

The frame substep planner now considers authored minimum substeps, CFL, and the
capillary limit

`dt_cap = safety * sqrt(rho * h^3 / (4*pi*sigma))`.

It reports the selected limiter, requested capillary count, clamp state,
per-frame maximum active interface voxels, curvature, acceleration, submission
time, and optional scratch allocation. Two lazily allocated `vec3` fields are
ping-ponged and reused for acceleration, for 24 bytes per allocated voxel.

Two strict presets are included:

- `Water Surface Tension` — 0.072 N/m physical-scale water starting point.
- `Stylized High Tension` — deliberately nonphysical 0.25 N/m cohesive liquid.

## Deep-review fixes included in 1.0.250

- Applied the capillary timestep constraint before first-frame source emission.
- Invalidated a same-key SDF after continuous emission and after direct particle
  injection, preventing physical/render caches from omitting new particles.
- Invalidated the pre-advection SDF before whitewater/render use.
- Made Time Scale = 0 finite in capillary preview and particle sanitation.
- Released live optional scratch allocation when the coefficient becomes zero.
- Made scratch profiler accounting reversible across enable/reset/reallocation.
- Aggregated maximum capillary diagnostics across substeps on the device.
- Removed duplicate Spray UI labelling.
- Kept dependency-free static tests runnable without importing Warp.
- Expanded normal support by the smoothing-stencil radius so the requested
  force band remains populated after smoothing.
- Made optional scratch allocation transactional and released validation
  solver resources on all success/failure paths.
- Strengthened the supplied-Warp gate to exercise the real frame-level
  limiter: the probe requests 27 capillary substeps and records the authored
  clamp to 8 even before first-frame source emission.
- Removed checkout-name assumptions from all Phase-3/4/5 Warp validators and
  made their final reports strict and atomic.
- Expanded the physical particle SDF and sparse-grid topology halo to cover the
  complete force, smoothing, scalar-filter, and authored splat-radius stencils.
- Included optional capillary scratch and enlarged topology demand in memory
  admission and safe-setting recommendations.
- Made legacy presets explicitly disable capillarity so switching away from a
  Phase-5 preset cannot retain hidden physical state.
- Made release build reports and checksum manifests strict, fsync-backed atomic
  replacements so interrupted packaging cannot expose partial metadata.

## Compatibility contract

- Cache-v1 and older cache-v2 frame playback remain supported.
- Phase-3 whitewater without type state remains legacy-unclassified on combined
  playback; Phase-4 typed frames retain their strict type validation.
- Surface tension adds no persistent stochastic/checkpoint array.
- Production resume remains manifest-, solver-, and settings-signature strict.
  A Phase-4 checkpoint is not promoted to a 1.0.250 resume point; rebake under
  this release before relying on restartable Phase-5 production work.
- Cancellation, atomic manifest commit, corruption rejection, orphan rejection,
  and cache-v1 playback guarantees are unchanged.

## Validation commands

```bash
python -m pytest -q
python -m compileall -q -f .
python ../tools/release/build_extension.py --verify-only --root .

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
python ../tools/release/validate_phase5_warp.py \
  --expected-warp-root /mnt/data/warp_lang_1_16_external \
  --module fluidsim_phase5_repo \
  --device cpu \
  --report-json /mnt/data/phase5_warp_validation.json

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
FLUIDSIM_EXTERNAL_WARP_ROOT=/mnt/data/warp_lang_1_16_external \
/mnt/data/blender-5.2.0-linux-x64/blender \
  --background --factory-startup --offline-mode --python-use-system-env \
  --python-exit-code 1 --python ../tools/release/validate_phase5_blender.py -- \
  --module fluidsim_phase5_repo \
  --work-dir /mnt/data/phase5_blender_validation_work \
  --expected-warp-root /mnt/data/warp_lang_1_16_external \
  --report-json /mnt/data/phase5_blender_validation.json
```

The NVIDIA release workflow invokes the P5 Warp validator with `--device
cuda:0` before the installed-package full solver regression. That gate is
mandatory for a CUDA-certified release and was not executable on this host.

## Verified in the available environment

- Dependency-free and supplied-Warp Python suites both passed: **215 tests**;
  Python compilation also passed.
- Source verification and supplied-Blender extension validation passed.
- Every P5 Warp kernel compiled and executed with the supplied Warp package on
  CPU, including force direction, finite values, curvature clamp, coefficient
  scaling, collision exclusion, MAC integration, device diagnostics, scratch
  lifecycle, and capillary timestep trends.
- Supplied Blender validated registration, transition tiers, strict presets,
  scene-to-solver projection, capillary preview, panel registration, and
  `.blend` save/reopen persistence.
- Established Phase-3 and Phase-4 supplied-Warp/Blender regressions passed,
  including typed whitewater and four Eevee F12 attribute/material renders.
- All four regenerated example scenes reopened successfully with owned
  inputs/outputs, six output shells, materials, presets, cache settings, and
  Geometry Nodes state intact.

## Remaining gates

Run the mandatory NVIDIA workflow using the supplied Warp package on compatible
hardware. Open checks are full CUDA solver numerics, numerical droplet
oscillation, uninterrupted-versus-resumed determinism, production-scale sparse
and particle growth, VRAM admission/out-of-memory behavior, performance, and
NVIDIA renderer behavior. Windows x64 packaging/smoke and Cycles-specific
render validation also remain open.
