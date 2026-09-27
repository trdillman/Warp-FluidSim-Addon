# FluidSim Custom Effects validation examples

These scenes are generated from real Custom Effects v2 kernel executions using the
certified Warp package on the CPU validation backend.

They are static validation/review artifacts, not playable FluidSim demos. They
contain precomputed before/after geometry and evidence from the generator, but
no live Domain-backed simulation that advances through Blender's frame
handlers. Do not use them to satisfy a request for a Custom Effects demo; a real
demo must meet the repository's `docs/DEMO_SCENE_CONTRACT.md`, including live
effect execution in a non-empty solver.

- **05 — Vortex Force:** compares the initial particle fixture with the validated
  Vortex Force result and trajectory lines.
- **06 — Stack Order:** demonstrates that Radial Pulse → Velocity Damping differs
  from Velocity Damping → Radial Pulse, while independently checking the analytic
  kinetic-energy damping ratio.
- **07 — Warp Code:** includes a developer-authored Lateral Wave Text datablock and
  package. The code was compiled and executed for evidence, but remains disabled and
  untrusted in the distributed `.blend` until the user reviews its exact SHA-256.

CPU validation proves compilation, bounded execution, deterministic fixture results,
and Blender publication. It does not prove live solver playback, full FluidSolver CUDA
numerics, or GPU performance; the NVIDIA CUDA release gate remains required on
compatible hardware.
