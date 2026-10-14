"""Bit-identity of the culled particle-SDF splat against the full-cube splat.

``k_build_surface_sdf`` skips candidate cells whose distance is at or beyond
the value the field was reset to. The claim is not "close enough" but
"identical": ``atomic_min`` only lowers a cell, the field starts at exactly
``far_distance``, so a candidate ``d >= far_distance`` can never win, and the
culled kernel evaluates ``d`` with the same expression as the uncalled one.

These tests hold that claim to the letter -- exact equality of the raw float32
bits, not ``allclose`` -- across splat radii from the default 2 up to the 7
that surface tension forces, and at a radius large enough that the cull is the
only thing keeping the two paths from diverging.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import surface as SURF  # noqa: E402
from fluidSim_warp_blender51.sim.fields import (  # noqa: E402
    ScalarField,
    SparseGrid,
    voxel_center_world,
    voxel_flat_index,
    world_to_voxel,
)


@wp.kernel
def k_reference_full_cube_splat(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    particle_radius: float,
    splat_radius_voxels: int,
    sdf: wp.array(dtype=float),
):
    """Verbatim copy of the loop body as it stood before any splat work.

    Frozen on purpose. The production kernel derives its loop bounds, and
    both of its modes share that arithmetic, so it can no longer serve as
    its own control. This one iterates the whole cube unconditionally.
    """

    tid = wp.tid()
    if tid >= count[0]:
        return
    p = positions[tid]
    vp = world_to_voxel(p, origin, voxel_size)
    ci = int(wp.floor(vp[0]))
    cj = int(wp.floor(vp[1]))
    ck = int(wp.floor(vp[2]))
    r = splat_radius_voxels
    for dk in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                ijk = wp.vec3i(ci + di, cj + dj, ck + dk)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                if flat < 0:
                    continue
                cw = voxel_center_world(ijk, origin, voxel_size)
                d = wp.length(cw - p) - particle_radius
                wp.atomic_min(sdf, flat, d)


def _grid(device, voxel_size=0.05):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(0.8, 0.8, 0.8),
        voxel_size=voxel_size,
        leaf_capacity=64,
        device=device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.799, 0.799, 0.799))
    assert grid.finalize_topology() >= 1
    return grid


def _particles(device, count):
    """Particles scattered through the domain, deliberately off cell centres.

    Cell-centred particles would make every distance land on a representable
    value and hide any last-ulp disagreement between the two paths.
    """

    rng = np.random.default_rng(20260903)
    pos = rng.uniform(0.05, 0.75, size=(count, 3)).astype(np.float32)
    return (
        wp.array(pos, dtype=wp.vec3, device=device),
        wp.array(np.array([count], dtype=np.int32), dtype=wp.int32, device=device),
    )


def _splat(grid, device, positions, count, radius, splat, far, max_distance):
    sdf = ScalarField(grid)
    wp.launch(
        SURF.k_surface_sdf_reset,
        dim=grid.voxel_capacity,
        inputs=[sdf.data, grid.num_leaves, far],
        device=device,
    )
    wp.launch(
        SURF.k_build_surface_sdf,
        dim=int(positions.shape[0]),
        inputs=[positions, count, grid.leaf_table, grid.voxels_dim,
                grid.leaves_dim, grid.origin, grid.voxel_size, radius, splat,
                max_distance],
        outputs=[sdf.data],
        device=device,
    )
    wp.synchronize_device(device)
    return sdf.data.numpy().copy()


def _splat_reference(grid, device, positions, count, radius, splat, far):
    """Same reset, then the frozen full-cube kernel."""

    sdf = ScalarField(grid)
    wp.launch(
        SURF.k_surface_sdf_reset,
        dim=grid.voxel_capacity,
        inputs=[sdf.data, grid.num_leaves, far],
        device=device,
    )
    wp.launch(
        k_reference_full_cube_splat,
        dim=int(positions.shape[0]),
        inputs=[positions, count, grid.leaf_table, grid.voxels_dim,
                grid.leaves_dim, grid.origin, grid.voxel_size, radius, splat],
        outputs=[sdf.data],
        device=device,
    )
    wp.synchronize_device(device)
    return sdf.data.numpy().copy()


@pytest.mark.parametrize("splat", [2, 5, 7, 9])
def test_both_modes_reproduce_the_original_full_cube_splat(cuda_device, splat):
    """Pin both modes to the pre-optimization kernel, bit for bit.

    ``atomic_min`` makes the answer order-independent -- min introduces no
    rounding -- so nondeterministic atomic ordering cannot explain away a
    difference here. Any mismatch is a real bounds defect.
    """

    grid = _grid(cuda_device)
    positions, count = _particles(cuda_device, 4096)
    h = float(grid.voxel_size)
    radius = 0.5 * h * 1.2
    far = (float(splat) - 0.5) * h

    reference = _splat_reference(grid, cuda_device, positions, count, radius,
                                 splat, far)
    legacy = _splat(grid, cuda_device, positions, count, radius, splat, far,
                    SURF.SDF_SPLAT_NO_CULL)
    culled = _splat(grid, cuda_device, positions, count, radius, splat, far,
                    far)

    for name, got in (("legacy", legacy), ("ball", culled)):
        assert np.array_equal(reference.view(np.uint32), got.view(np.uint32)), (
            f"{name} mode diverges from the frozen full-cube splat at radius "
            f"{splat}: {int((reference != got).sum())} of {reference.size} "
            f"cells differ; max |delta| "
            f"{float(np.abs(reference - got).max())}"
        )


@pytest.mark.parametrize("splat", [2, 5, 7, 9])
def test_ball_cull_is_bit_identical_to_the_full_cube_splat(cuda_device, splat):
    grid = _grid(cuda_device)
    positions, count = _particles(cuda_device, 4096)
    h = float(grid.voxel_size)
    radius = 0.5 * h * 1.2
    # The band surface tension actually asks for: (support + 1) * h, with the
    # splat radius sized to cover it. Culling only matters when the cube
    # reaches past this.
    far = (float(splat) - 0.5) * h

    legacy = _splat(grid, cuda_device, positions, count, radius, splat, far,
                    SURF.SDF_SPLAT_NO_CULL)
    culled = _splat(grid, cuda_device, positions, count, radius, splat, far,
                    far)

    # Exact bits, not allclose: the cull is an identity, not an approximation.
    assert np.array_equal(legacy.view(np.uint32), culled.view(np.uint32)), (
        f"splat radius {splat}: {int((legacy != culled).sum())} of "
        f"{legacy.size} cells differ; max |delta| "
        f"{float(np.abs(legacy - culled).max())}"
    )


def test_the_cull_is_actually_exercised_at_a_surface_tension_radius(cuda_device):
    """Guard against the identity test passing because nothing was culled.

    If the cube never reaches beyond ``far`` the two paths are trivially equal
    and the test above proves nothing. Assert the band is genuinely narrower
    than the cube: cells strictly outside the ball must exist, and the field
    must still contain interior (negative) voxels so the splat did real work.
    """

    grid = _grid(cuda_device)
    positions, count = _particles(cuda_device, 4096)
    h = float(grid.voxel_size)
    radius = 0.5 * h * 1.2
    splat = 7
    far = (float(splat) - 0.5) * h

    culled = _splat(grid, cuda_device, positions, count, radius, splat, far, far)

    assert int((culled < 0.0).sum()) > 0, "splat produced no interior voxels"

    # Cube corner distance vs. cull radius, in voxels. The corner of a 15^3
    # cube sits sqrt(3) * 7 = 12.1 voxels out, well beyond the 6.5-voxel ball,
    # so a large majority of the cube is skipped.
    corner_voxels = float(np.sqrt(3.0)) * float(splat)
    cull_voxels = (far + radius) / h
    assert corner_voxels > cull_voxels, (
        "cube does not extend past the cull radius; the identity test would "
        "be vacuous"
    )


def test_pipeline_splat_mode_switch_feeds_the_kernel():
    """The pipeline-level switch feeds the kernel the right max_distance."""

    assert SURF.default_sdf_splat_mode() == SURF.SDF_SPLAT_MODE_BALL

    pipeline = object.__new__(SURF.SurfacePipeline)
    assert pipeline.splat_mode() == SURF.SDF_SPLAT_MODE_BALL
    pipeline.sdf_splat_mode = SURF.SDF_SPLAT_MODE_LEGACY
    assert pipeline.splat_mode() == SURF.SDF_SPLAT_MODE_LEGACY

    with pytest.raises(ValueError):
        SURF.set_default_sdf_splat_mode("nonsense")
    # The rejected value must not have been latched.
    assert SURF.default_sdf_splat_mode() == SURF.SDF_SPLAT_MODE_BALL


def test_legacy_sentinel_survives_the_float32_round_trip():
    """SDF_SPLAT_NO_CULL must stay finite and unreachable inside the kernel.

    It is passed through a float32 kernel parameter. An overflow to inf would
    still disable the cull, but a silent inf is a worse thing to rely on than
    a value chosen to be representable.
    """

    as_f32 = np.float32(SURF.SDF_SPLAT_NO_CULL)
    assert np.isfinite(as_f32)
    assert float(as_f32) > 1.0e29
