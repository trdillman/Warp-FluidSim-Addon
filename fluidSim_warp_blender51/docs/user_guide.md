# FluidSim Warp — User Guide

GPU liquid simulation for Blender 5.1, powered by NVIDIA Warp/CUDA.
The solver is a sparse-grid PIC/FLIP liquid simulator that runs entirely on
the GPU and plays back interactively in the viewport — no cache baking
required.

## Requirements

- Blender 5.1
- NVIDIA GPU with CUDA support
- `warp-lang` available to Blender's Python (installed in
  `scripts/addons/modules` or site-packages)

## Installation

Copy the `fluidSim_warp_blender51` folder into your Blender addons directory
(`.../Blender/5.1/scripts/addons/`) and enable **Physics: FluidSim Warp** in
Preferences. The add-on preferences show the detected Warp version and CUDA
device; the compute device (default `cuda:0`) can be changed there.

## Quick start

1. Open the **FluidSim** tab in the 3D Viewport sidebar (N panel).
2. Press **Quick Demo Scene** (Simulation panel) — creates a domain, a box
   liquid source, and a sphere collider.
3. Press **Initialize Solver**.
4. Press **Live** and play the timeline (Space). The liquid simulates
   interactively; rewinding to the start frame resets it.

Or set up manually:

1. Create any object to act as the domain bounds and assign it as
   **Domain Object** (Domain / Sparse Grid panel).
2. Select an object, press **Tag Source** (Sources panel). Choose shape
   (Sphere/Cylinder/Box/Pyramid/Cone/Plane), mode (Particles/Field), emit
   rate, initial velocity, and attribute values (temperature, viscosity,
   density, lifespan).
3. Select a mesh, press **Tag Collider** (Colliders / Sinks panel); the mesh
   is converted to a GPU SDF each frame, with optional velocity transfer
   from animated transforms. **Tag Sink** makes an object delete liquid.
4. Initialize, enable Live, play.

## Panels

- **Simulation** — init/release/reset, live toggle, manual stepping, time
  scale, CFL, substep limits, gravity, status readout.
- **Domain / Sparse Grid** — domain object, voxel size, leaf capacity,
  topology expansion, open boundaries, resolution/usage readouts.
- **Particles** — max particles, particles per voxel, PIC/FLIP sample type,
  FLIP ratio, trace mode (Euler/midpoint/RK3), relaxation.
- **Sources** — tag/untag objects, per-object source settings, height-field
  sourcing (flat or profile back modes).
- **Colliders / Sinks** — tag colliders/sinks, friction, restitution,
  velocity transfer.
- **Forces / Attraction** — speed limit (maintain/percent/acceleration),
  per-object force shapes (directional/omnidirectional/spin/radial).
- **Pressure Projection** — Jacobi or PCG, iterations, tolerance, multigrid
  coarse guess, pressure threshold, particle-count stiffness.
- **Viscosity** — explicit smoothing (Jacobi) or implicit solve (PCG),
  constant viscosity or per-particle viscosity attribute.
- **Surface / Whitewater** — surface SDF build radius and smoothing filter
  (dilate/erode/box/mean-curvature), churn parameters, whitewater emission,
  drag, lifespan.
- **Viewport / Render Output** — toggle surface mesh / particle point cloud /
  whitewater outputs, display fraction, materials, output collection,
  optional npz caching and OBJ/PLY export.
- **Custom Warp Snippets** — user GPU kernels (see below).
- **Diagnostics / Profiling** — per-phase GPU timings, overlay toggle,
  report writing.
- **Validation / Benchmarks** — run the 13-test GPU validation suite and
  standard benchmark; compare against a saved baseline.

## Outputs

Simulation outputs live in the **FluidSim Output** collection:

- **FluidSim Surface** — marching-cubes liquid mesh, updated every frame,
  with a glass-like liquid material (override in the Output panel).
- **FluidSim Particles** — point cloud of liquid particles (display
  fraction subsamples for viewport speed).
- **FluidSim Whitewater** — foam/spray/bubble point cloud.

All outputs render with Cycles/EEVEE and update during playback.

## Custom Warp snippets

Add a snippet in the **Custom Warp Snippets** panel. Each snippet is a Warp
kernel body stored in a Blender text block, executed at a chosen point in
the solver step (first step, before sourcing, before forces, before
projection, final step) over voxels or particles.

Available names inside a voxel snippet: `flat_voxel_index`, `leaf_index`,
`local_x/y/z`, `global_x/y/z`, `world_position`, `voxel_size`, `frame`,
`time`, `real_time_step`, `solver_time_step`, `time_scale`, `substeps`, plus
any declared arguments. Particle snippets get `particle_index`, `position`,
`velocity` and the same time context.

Arguments are either **fields** (e.g. `temperature`, `velocity_u`,
`surface_sdf`, `particle_velocity`, ...) with read/write access, or **float/
int constants**. Example voxel snippet that heats a rising column:

```python
if world_position[0] * world_position[0] + world_position[1] * world_position[1] < 0.25:
    temperature[flat_voxel_index] = temperature[flat_voxel_index] + heat_rate * solver_time_step
```

Press **Apply Snippets** to compile; errors appear inline in the panel.

## Profiling, validation, benchmarking

- Enable **Profiling** before initializing to get per-phase GPU timers
  (sourcing, P2G, pressure, advection, ...) in the Diagnostics panel and
  the viewport overlay.
- **Run Validation** executes 13 GPU correctness tests (topology, transfer,
  projection convergence, viscosity stability, whitewater, ...).
- **Run Benchmark** runs the standard dam-break benchmark and reports
  step-time and voxels/second statistics; **Save Baseline** stores a JSON
  baseline for regression comparison.
- **Write Report** saves a JSON diagnostic report (device, memory, timers,
  frame history, benchmark, validation).

## Tips

- Viewport interactivity scales with voxel size; 0.05–0.1 is a good range
  to start.
- The first initialization compiles CUDA kernels (~10 s); subsequent runs
  use the kernel cache and start instantly.
- Rewind to the start frame at any time to reset a live simulation.
- Keep **Profiling Sync** off for maximum playback speed; turn it on for
  accurate per-phase timings.
