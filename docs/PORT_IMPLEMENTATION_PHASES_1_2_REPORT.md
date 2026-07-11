# FluidSim Warp 1.0.236 Port Report

Date: 2026-07-11
Status: phases one and two implemented; phase three remains investigation
Feature label: ST-FLIP Beta

## Phase one: surface-resolution pipeline

Implemented:

- simulation, multiplier, and custom surface-resolution modes
- separate GPU sparse render-surface grid/SDF/filter scratch/extractor
- render-grid leaf allocation scaling
- dense marching-cubes extraction safety limit with actionable failure
- substep, frame, and manual/lazy rebuild cadence
- optional simulation-grid surface SDF for whitewater/snippets
- active pre-filter plus dilate, erode, and smoothing stack
- settings-sensitive mesh invalidation and reuse
- shared selected mesh for viewport, cache, and OBJ export
- cache/export surface-resolution metadata
- Blender UI controls in Realtime Surfacing / Particles

Measured runtime acceptance:

- 500,000 particles, 24 frames: PASS
- 1x surface: 1,824 vertices at voxel size 0.1
- 2x surface: 7,706 vertices at voxel size 0.05
- 2x uses a separate sparse surface grid
- filter changes invalidate and change the mesh

The production default remains simulation-resolution surfacing. A measured 2x
global default raised the canonical mean frame time to 85.8 ms. Keeping 2x
opt-in produced 69.01 ms, 71.74 ms, and 73.75 ms canonical runs. Their 71.50 ms
mean is approximately 39.2% faster than the original 117.66 ms baseline while
retaining all new controls. All runs passed the interactive profile thresholds;
the historical 40% target is within run-to-run variance rather than a guaranteed
minimum after this port.

## Phase two: ST-FLIP beta

Implemented:

- persistent device-resident particle time residual
- bounded stochastic time advancement and residual carryover
- adaptive temporal jitter using local CFL
- separable space-time P2G deposition
- phase-field construction from face-weight accumulators
- optional variable-coefficient ST pressure matrix and projection
- calibrated or manual reference mass
- local particle trajectory substepping
- temporary surface-position re-synchronization
- particle cache/viewport time-offset attribute
- beta UI controls, default disabled
- Blender/Warp validation target and headless acceptance scripts

The default fixed-coefficient pressure path retains its fused x/r/preconditioner
kernel, recorded launches, parity-specific scalar commands, and bounded CUDA
graph cache. The beta variable-coefficient ST path remains separate and does
not claim the same graph optimization yet.

Measured beta acceptance:

- small live Blender/Warp ST validation: PASS
- 500,000 particles, 24-frame three-lane benchmark: PASS
- low-CFL FLIP: 2.129 s, 88 substeps
- high-CFL FLIP: 1.595 s, 24 substeps
- high-CFL ST-FLIP: 1.745 s, 24 substeps
- ST residual active and bounded
- phase field active
- pressure converged
- 500,000 particles retained
- saved-scene timeline/viewport gate: PASS
- timeline solver frame: 24
- output particles: 516,384
- output surface vertices: 8,811
- visible particle and surface objects populated

## Regression and production proof

- phase-one contract/UI tests: PASS
- pressure/performance regression tests: 18/18 PASS
- non-headless pytest lane: 133 passed
- canonical user-path profile: PASS three times, 69.01-73.75 ms
- headless production gate: `HEADLESS-GATE-PASS`
- live Blender 5.1 review: PASS
  - exact canonical `demos/dam_break.blend`
  - add-on version `(1, 0, 235)`
  - add-on enabled and registered through isolated Blender add-on root
  - beta ST-FLIP UI present
  - surface pipeline UI present
  - whitewater UI remains consolidated

## Phase-three investigation lane

Phase three should not block beta shipment. It should determine when ST-FLIP
graduates from beta and whether its variable pressure path warrants its own
recorded-launch/CUDA-graph optimization.

Required work:

1. compare ordinary high-CFL FLIP and ST-FLIP against a low-CFL reference;
2. measure surface-SDF error, particle-distribution error, volume/mass drift,
   kinetic-energy evolution, and collision penetration;
3. repeat across seeds to quantify stochastic variance;
4. cover dam break, slosh, thin sheets/splashes, moving colliders, and emitters;
5. extend runs beyond 100 frames for accumulated residual/conservation drift;
6. validate the variable-coefficient matrix and projection against a small-grid
   CPU/reference formulation;
7. profile and, if justified, record/capture the variable-coefficient PCG path;
8. profile separate surface splat, filter, marching-cubes, and Blender upload
   costs before recommending 2x defaults by scene class.

## Machine-aware Safe Settings recovery

Version 1.0.236 adds a pre-allocation VRAM estimate and recovery UI. Unsafe
configurations stop before solver arrays are created. Leaf overflow and known
CUDA memory errors route to the same **Apply Safe Settings & Retry** button.
Recommendations account for current free/total VRAM, GPU utilization, Windows
CPU load, RAM pressure, and current scene settings. Adjustments cover leaf
capacity, voxel size, liquid/whitewater capacities, particles per voxel,
topology expansion, and secondary render-surface allocation.

## Evidence artifacts

- `artifacts/phase1_surface_resolution_final.json`
- `artifacts/phase2_stflip_green_probe.log`
- `artifacts/phase2_stflip_benchmark.json`
- `artifacts/phase2_stflip_timeline_viewport.json`
- `artifacts/post_port_user_path_profile_fast_default.json`
- `artifacts/post_port_user_path_profile_fast_default_run2.json`
- `artifacts/post_review_user_path_profile.json`
- `artifacts/post_port_production_gate.log`
- `artifacts/phase_port_live_review_stdout.log`
- `artifacts/phase_port_live_review_stderr.log`
