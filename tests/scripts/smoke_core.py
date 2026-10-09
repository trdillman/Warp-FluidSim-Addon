import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warp as wp
wp.init()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "support"))
from _test_device import script_device
dev = script_device()
from fluidSim_warp_blender51.sim.fields import SparseGrid, MacVelocityField
from fluidSim_warp_blender51.sim import kernels_core, kernels_fluid, pressure
from fluidSim_warp_blender51.sim.registry import KERNEL_REGISTRY

g = SparseGrid(origin=(-1,-1,-1), size=(2,2,2), voxel_size=0.1, leaf_capacity=512, device=dev)
print("dims", g.voxels_dim, g.leaves_dim, "cap", g.capacity)
g.begin_topology()
g.mark_world_box((-0.5,-0.5,-0.5),(0.5,0.5,0.5))
g.expand(1)
n = g.finalize_topology()
print("leaves", n)

vel = MacVelocityField(g)
import numpy as np
# particles in a small box
pts = np.random.uniform(-0.4, 0.4, size=(5000,3)).astype(np.float32)
pos = wp.array(pts, dtype=wp.vec3, device=dev)
velp = wp.zeros(5000, dtype=wp.vec3, device=dev)
cnt = wp.array(np.array([5000], dtype=np.int32), device=dev)

voxel_type = wp.zeros(g.voxel_capacity, dtype=wp.int32, device=dev)
voxel_counts = wp.zeros(g.voxel_capacity, dtype=wp.int32, device=dev)
collision_vel = wp.zeros(g.voxel_capacity, dtype=wp.vec3, device=dev)

wp.launch(kernels_fluid.k_particle_set_grid_data, dim=5000,
          inputs=[pos, velp, wp.zeros(1, dtype=wp.mat33, device=dev), 1, cnt, g.leaf_table, g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size,
                  vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww], device=dev)
wp.launch(kernels_fluid.k_normalize_mac, dim=g.voxel_capacity,
          inputs=[vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww, g.num_leaves], device=dev)
wp.launch(kernels_fluid.k_particle_to_count, dim=5000,
          inputs=[pos, cnt, g.leaf_table, g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size, voxel_counts], device=dev)
wp.launch(kernels_fluid.k_particle_to_grid_active, dim=g.voxel_capacity,
          inputs=[voxel_counts, g.num_leaves, voxel_type], device=dev)

ps = pressure.PressureSolver(g, dev)
ps.build_rhs(vel, voxel_type, collision_vel, voxel_counts)
ps.build_preconditioner(voxel_type)
ps.solve_pcg(voxel_type, 100, 1e-6)
print("pcg iters", ps.last_iterations, "res", ps.last_residual)
ps.project(vel, voxel_type, collision_vel)
print("registered kernels:", len(KERNEL_REGISTRY))
wp.synchronize()
print("SMOKE OK")
