# Lane: Particle Pipeline
- **Dimension name:** Particle Pipeline
- **ID prefix:** PIPE
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/pipeline.json

Follow `_common.md` in this same directory.

## Scope (review only these files)
- fluidSim_warp_blender51/sim/sourcing.py  (~3145)
- fluidSim_warp_blender51/sim/source_regions.py  (~1411)
- fluidSim_warp_blender51/sim/particles.py  (~265)
- fluidSim_warp_blender51/sim/capacity_planner.py  (~573)
- fluidSim_warp_blender51/sim/memory_safety.py  (~327)
- fluidSim_warp_blender51/sim/whitewater.py  (~721)
- fluidSim_warp_blender51/sim/surface.py  (~883)

## Dimension Focus  (project mandate: behavior must be ROBUST as particle counts / resolution scale up)
- Data flow through emit -> advect -> pressure -> surface/whitewater.
- GPU/memory lifecycle: warp array alloc/free leaks, reuse of stale buffers after resize, double-free, use-after-free, arrays not zeroed, capacity overflow without reallocation.
- Capacity planner / memory_safety: does capacity growth scale correctly with particle count & resolution? Any hard-coded caps or brittle heuristics that break under scale? Is the fail-closed path actually fail-closed (raises rather than silently truncating)?
- Numerical correctness of emission (mass/volume conservation, seeding), advection, whitewater/surface generation.
- Stage transitions: are buffers re-sized/synced when crossing capacity thresholds between stages?
