"""B34: a sink deep inside a narrow-band liquid drains it.

With narrow band on, the deep interior has no particles; its liquid is the
advected level set. The union step kept deep negative phi without a sink
check and the fluid-from-phi mask excluded collision but not sink, so a
bottom drain killed nothing the pressure solve could see: the tank's surface
height and liquid-voxel count stayed constant while the same tank without
narrow band emptied.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

wp = pytest.importorskip("warp")

import _narrow_band_harness as H  # noqa: E402

from fluidSim_warp_blender51.sim import constants as C  # noqa: E402
from fluidSim_warp_blender51.sim.enums import SourceShapeShape, SourceShapeType  # noqa: E402
from fluidSim_warp_blender51.sim.kernels_narrow_band import k_nb_set_fluid_from_phi  # noqa: E402
from fluidSim_warp_blender51.sim.solver import FluidSolver, SourceShapeDesc  # noqa: E402

FRAMES = 10


def _drain(device, narrow_band):
    solver = FluidSolver(H.settings(device, narrow_band=narrow_band))
    try:
        solver.inject_particles(H.lattice(H.FILL_HEIGHT))
        sink = SourceShapeDesc()
        sink.shape = SourceShapeShape.BOX
        sink.stype = SourceShapeType.SINK
        sink.position = (0.25, 0.25, 0.08)  # tank bottom, ~0.6 m below the surface
        sink.half_extents = (0.08, 0.08, 0.06)
        solver.set_source_batch([sink])
        first = None
        for _ in range(FRAMES):
            st = solver.step_frame()
            if first is None:
                first = int(st.liquid_voxels)
        return H.surface_height(solver), first, int(st.liquid_voxels)
    finally:
        solver.release()


def test_deep_sink_drains_a_narrow_band_tank(cuda_device):
    height_nb, first_lv, last_lv = _drain(cuda_device, narrow_band=True)
    height_ref, _, _ = _drain(cuda_device, narrow_band=False)
    assert height_ref < 0.5, "control tank without narrow band did not drain"
    assert height_nb < 0.5, f"narrow-band surface stayed at {height_nb:.3f} m"
    assert abs(height_nb - height_ref) < 0.05
    assert last_lv < 0.7 * first_lv, (first_lv, last_lv)


def test_fluid_from_phi_respects_sink_precedence(cuda_device):
    """collision > sink > fluid, as in sourcing (k_source_resolve_type)."""
    vt = np.array([0, C.VOXEL_SINK, C.VOXEL_COLLISION, 0], np.int32)
    phi = np.array([-1.0, -1.0, -1.0, 1.0], np.float32)
    pad = C.LEAF_VOLUME - vt.size
    voxel_type = wp.array(np.pad(vt, (0, pad)), dtype=wp.int32, device=cuda_device)
    phi_dev = wp.array(np.pad(phi, (0, pad), constant_values=1.0), dtype=float,
                       device=cuda_device)
    num_leaves = wp.array([1], dtype=wp.int32, device=cuda_device)
    wp.launch(k_nb_set_fluid_from_phi, dim=C.LEAF_VOLUME, device=cuda_device,
              inputs=[phi_dev, num_leaves], outputs=[voxel_type])
    out = voxel_type.numpy()[:4]
    fluid = (out & C.VOXEL_FLUID) != 0
    assert fluid.tolist() == [True, False, False, False]
