# FluidSim Custom Effects demos

These scenes are generated from real Custom Effects v2 kernel executions using the
supplied Warp 1.16.0 package on the CPU validation backend.

- **05 — Vortex Force:** compares the initial particle fixture with the validated
  Vortex Force result and trajectory lines.
- **06 — Stack Order:** demonstrates that Radial Pulse → Velocity Damping differs
  from Velocity Damping → Radial Pulse, while independently checking the analytic
  kinetic-energy damping ratio.
- **07 — Warp Code:** includes a developer-authored Lateral Wave Text datablock and
  package. The code was compiled and executed for evidence, but remains disabled and
  untrusted in the distributed `.blend` until the user reviews its exact SHA-256.

CPU validation proves compilation, bounded execution, deterministic fixture results,
and Blender publication. It does not certify full FluidSolver CUDA numerics or GPU
performance; the NVIDIA CUDA release gate remains required on compatible hardware.
