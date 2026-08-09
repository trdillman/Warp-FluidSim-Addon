# Lane: Execution Engine
- **Dimension name:** Execution Engine
- **ID prefix:** EXEC
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/exec.json

Follow `_common.md` in this same directory.

## Scope (review only these files)
- fluidSim_warp_blender51/sim/solver.py  (~1862)
- fluidSim_warp_blender51/sim/pressure.py  (~2268)
- fluidSim_warp_blender51/sim/kernels_fluid.py  (~1892)
- fluidSim_warp_blender51/sim/kernels_core.py  (~680)
- fluidSim_warp_blender51/sim/viscosity.py  (~445)
- fluidSim_warp_blender51/sim/fields.py  (~608)
- fluidSim_warp_blender51/sim/validation.py  (~1197)

## Dimension Focus
- Control-flow correctness in the solver/pressure loop.
- Error handling & propagation: warp launch errors / NaN / exceptions caught & surfaced, or silently swallowed?
- Edge cases: zero particles, empty/degenerate domain, dt=0, CFL violations, pressure-solve non-convergence, restart/resume mid-step.
- Caching correctness & invalidation (stale arrays reused after a parameter change).
- Numerical stability: pressure projection convergence, divergence, accumulation drift, timestep integration.
- Grid indexing: off-by-one in voxel/grid bounds, boundary cells, ghost-cell handling.
- Warp array lifetime: arrays created/zeroed each step vs reused; launch-dim mismatches.
