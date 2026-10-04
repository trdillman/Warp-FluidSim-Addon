"""Voxel-type marking must be additive, matching paradigmParticleToGridActive.

Ground truth (docs/source_shape_spec.md, from fluidSim.cl):

  __kernel void paradigmParticleToGridActive(...) {
      ...
      if (l_voxelIndex != -1) {
          a_voxelType[l_voxelIndex] |= Bit_Core_VoxelType_Active;
      }
  }

It is purely additive. Across all 20,640 lines of fluidSim.cl,
Bit_Core_VoxelType_Fluid is never cleared by any kernel -- it is set by
paradigmParticleSetGridData, propagated by paradigmExpandFluidMask, and reset
only by the whole-buffer coreClearScalarVoxelType.

This port had added an `else` branch clearing VOXEL_FLUID for unoccupied
voxels. That branch has no counterpart in the original and is redundant:
FluidSolver._clear_grid_fields() already zeroes voxel_type[:nvox] every
substep (unconditionally -- it is in the ungated array list). Its only real
effect was destroying bits that sourcing had set earlier in the same substep.
"""

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.constants import (
    VOXEL_ACTIVE,
    VOXEL_COLLISION,
    VOXEL_FLUID,
    VOXEL_SOURCE_REGION_TRACKING,
)
from fluidSim_warp_blender51.sim.kernels_fluid import k_particle_to_grid_active

LEAF_VOLUME = 512


def _run(voxel_counts, voxel_type):
    wp.init()
    device = wp.get_preferred_device()
    n = len(voxel_counts)
    counts = wp.array(np.asarray(voxel_counts, dtype=np.int32),
                      dtype=wp.int32, device=device)
    vtype = wp.array(np.asarray(voxel_type, dtype=np.int32),
                     dtype=wp.int32, device=device)
    num_leaves = wp.array([max(1, -(-n // LEAF_VOLUME))],
                          dtype=wp.int32, device=device)
    wp.launch(k_particle_to_grid_active, dim=n,
              inputs=[counts, num_leaves], outputs=[vtype], device=device)
    return vtype.numpy()


def test_unoccupied_voxel_retains_bits_set_earlier_in_the_substep():
    """The kernel must not clear VOXEL_FLUID it did not set."""
    out = _run([0], [VOXEL_FLUID | VOXEL_ACTIVE | VOXEL_SOURCE_REGION_TRACKING])
    assert out[0] & VOXEL_FLUID, (
        "VOXEL_FLUID was cleared for an unoccupied voxel; "
        "paradigmParticleToGridActive is additive and never clears Fluid")
    assert out[0] & VOXEL_SOURCE_REGION_TRACKING
    assert out[0] & VOXEL_ACTIVE


def test_occupied_voxel_is_marked_fluid_and_active():
    out = _run([3], [0])
    assert out[0] & VOXEL_FLUID
    assert out[0] & VOXEL_ACTIVE


def test_collision_voxel_is_never_marked_fluid():
    out = _run([5], [VOXEL_COLLISION])
    assert not (out[0] & VOXEL_FLUID)
    assert out[0] & VOXEL_COLLISION


def test_empty_voxel_with_no_prior_bits_stays_empty():
    """Regression guard: the common case must be unchanged (all demos)."""
    out = _run([0], [0])
    assert out[0] == 0


@pytest.mark.parametrize("prior", [0, VOXEL_COLLISION])
def test_marking_is_monotonic_for_unoccupied_voxels(prior):
    """No bit may be removed by this kernel."""
    out = _run([0], [prior])
    assert out[0] & prior == prior, f"kernel removed bits from {prior}"
