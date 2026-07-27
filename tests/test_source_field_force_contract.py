"""Primary-source contracts for FIELD mode and analytic FORCE regions.

See docs/source_shape_spec.md, extracted from fluidSim.cl + paradigm.hda.
FIELD mode does not create liquid or write MAC velocity. The Blender-facing
FORCE role is routed through the force stage after P2G, where valid staggered
faces have non-zero weights.
"""

import numpy as np
import warp as wp

from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim.enums import (
    SourceShapeForceType,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
)
from fluidSim_warp_blender51.sim.sourcing import (
    SourceShape,
    SourceShapeDesc,
    k_source_shapes,
)


def _leaf(device):
    return (
        wp.array([(0, 0, 0)], dtype=wp.vec3i, device=device),
        wp.array([1], dtype=wp.int32, device=device),
    )


def test_field_source_does_not_create_liquid_or_mutate_mac_attributes():
    device = wp.get_preferred_device()
    leaves, num_leaves = _leaf(device)
    shape = SourceShapeDesc(
        enabled=True,
        shape=SourceShapeShape.BOX,
        stype=SourceShapeType.SOURCE,
        mode=SourceShapeSourceMode.FIELD,
        position=(0.4, 0.4, 0.4),
        half_extents=(1.0, 1.0, 1.0),
        velocity=(1.0, 2.0, 3.0),
        force_type=SourceShapeForceType.DIRECTIONAL,
        force_strength=4.0,
        force_dir=(1.0, 0.0, 0.0),
        temperature=8.0,
        viscosity=9.0,
    ).to_struct()
    shapes = wp.array([shape], dtype=SourceShape, device=device)
    n = C.LEAF_VOLUME
    voxel_type = wp.zeros(n, dtype=wp.int32, device=device)
    collision_sdf = wp.full(n, 1.0e6, dtype=float, device=device)
    collision_vel = wp.zeros(n, dtype=wp.vec3, device=device)
    u = wp.full(n, 0.25, dtype=float, device=device)
    v = wp.full(n, 0.5, dtype=float, device=device)
    w = wp.full(n, 0.75, dtype=float, device=device)
    temperature = wp.full(n, 2.0, dtype=float, device=device)
    viscosity = wp.full(n, 3.0, dtype=float, device=device)

    wp.launch(
        k_source_shapes, dim=n,
        inputs=[shapes, 1, leaves, num_leaves, wp.vec3(0.0), 0.1, 1.0 / 24.0],
        outputs=[voxel_type, collision_sdf, collision_vel,
                 u, v, w, temperature, viscosity], device=device)

    assert np.count_nonzero(voxel_type.numpy() & (
        C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING)) == 0
    np.testing.assert_allclose(u.numpy(), 0.25)
    np.testing.assert_allclose(v.numpy(), 0.5)
    np.testing.assert_allclose(w.numpy(), 0.75)
    np.testing.assert_allclose(temperature.numpy(), 2.0)
    np.testing.assert_allclose(viscosity.numpy(), 3.0)


def test_analytic_force_region_applies_after_p2g_only_to_weighted_faces():
    device = wp.get_preferred_device()
    leaves, num_leaves = _leaf(device)
    shape = SourceShapeDesc(
        enabled=True,
        shape=SourceShapeShape.BOX,
        stype=SourceShapeType.SOURCE,
        mode=SourceShapeSourceMode.FIELD,
        position=(0.4, 0.4, 0.4),
        half_extents=(1.0, 1.0, 1.0),
        force_type=SourceShapeForceType.DIRECTIONAL,
        force_strength=2.0,
        force_dir=(1.0, 0.0, 0.0),
    ).to_struct()
    shapes = wp.array([shape], dtype=SourceShape, device=device)
    n = C.LEAF_VOLUME
    u = wp.zeros(n, dtype=float, device=device)
    v = wp.zeros(n, dtype=float, device=device)
    w = wp.zeros(n, dtype=float, device=device)
    wu = wp.ones(n, dtype=float, device=device)
    wv = wp.ones(n, dtype=float, device=device)
    ww = wp.ones(n, dtype=float, device=device)
    # One invalid face must remain untouched.
    wu.numpy()  # force allocation before overwrite on all backends
    zero = wp.zeros(1, dtype=float, device=device)
    wp.copy(wu, zero, count=1)

    kernel = getattr(__import__(
        "fluidSim_warp_blender51.sim.sourcing", fromlist=["k_source_shape_forces"]),
        "k_source_shape_forces")
    wp.launch(
        kernel, dim=n,
        inputs=[shapes, 1, leaves, num_leaves, wp.vec3(0.0), 0.1,
                u, v, w, wu, wv, ww, 0.5], device=device)

    host_u = u.numpy()
    assert host_u[0] == 0.0
    np.testing.assert_allclose(host_u[1:], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(v.numpy(), 0.0)
    np.testing.assert_allclose(w.numpy(), 0.0)
