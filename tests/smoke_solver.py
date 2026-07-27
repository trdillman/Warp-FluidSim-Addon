"""Headless smoke test: full solver pipeline on CUDA (dam break in a box)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp as wp

wp.init()

from fluidSim_warp_blender51.sim.solver import (
    FluidSolver, SolverSettings, SourceShapeDesc,
)

settings = SolverSettings(
    domain_min=(-1.0, -1.0, 0.0),
    domain_max=(1.0, 1.0, 2.0),
    voxel_size=0.05,
    leaf_capacity=2048,
    max_particles=400_000,
    max_whitewater=100_000,
    whitewater_enabled=True,
    surface_enabled=True,
)

solver = FluidSolver(settings)

# Dam-break: a box of liquid in one corner, emitted once (field of particles).
solver.set_sources([
    SourceShapeDesc(
        enabled=True, shape=2, stype=0, mode=1,
        position=(-0.5, -0.5, 0.6),
        half_extents=(0.45, 0.45, 0.55),
        emit_rate=8.0,
    ),
])

t0 = time.perf_counter()
stats = None
for frame in range(8):
    stats = solver.step_frame()
    print(f"frame {frame}: particles={stats.active_particles} "
          f"voxels={stats.active_voxels} substeps={stats.substeps} "
          f"p_iter={stats.pressure_iterations} p_res={stats.pressure_residual:.3e} "
          f"t={stats.step_time*1000:.1f}ms ww={stats.whitewater_particles}")
    if frame == 0:
        # After the first frame the source region must contain particles.
        assert stats.active_particles > 1000, "no particles emitted"

# Stop emitting; let it settle a few frames.
solver.set_sources([])
for frame in range(4):
    stats = solver.step_frame()
    print(f"settle {frame}: particles={stats.active_particles} "
          f"voxels={stats.active_voxels} t={stats.step_time*1000:.1f}ms")

assert stats.active_particles > 1000, "particles vanished"

verts, tris = solver.surface_mesh()
print(f"surface mesh: {len(verts)} verts, {len(tris)} tris")
assert len(verts) > 0 and len(tris) > 0, "no surface mesh extracted"

pos = solver.particle_positions()
assert pos.shape[0] == stats.active_particles
zmin, zmax = float(pos[:, 2].min()), float(pos[:, 2].max())
print(f"particle z-range: {zmin:.3f} .. {zmax:.3f}")
assert zmin > -0.1, "particles fell through the floor"

print(f"TOTAL {time.perf_counter()-t0:.1f}s")
print("SMOKE-SOLVER-PASS")
