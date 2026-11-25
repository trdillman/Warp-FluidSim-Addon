"""Height fill keeps out of collider meshes (and a margin), open or closed."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim import source_kernels as K
from fluidSim_warp_blender51.sim.height_fill_solids import (
    HeightFillSolids,
    eligible_solids,
)
from fluidSim_warp_blender51.sim.solver import ColliderDesc
from fluidSim_warp_blender51.sim.source_contracts import SourceShapeDesc

DEVICE = "cpu"
H = 0.1
ORIGIN = (0.0, 0.0, 0.0)
LEAVES_PER_AXIS = 3  # 24^3 voxels, a 2.4 m cube starting at the origin
DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (2.4, 2.4, 2.4)


def _grid():
    coords = [(i, j, k) for k in range(LEAVES_PER_AXIS)
              for j in range(LEAVES_PER_AXIS) for i in range(LEAVES_PER_AXIS)]
    return SimpleNamespace(
        leaf_coords=wp.array(np.array(coords, np.int32), dtype=wp.vec3i, device=DEVICE),
        num_leaves=wp.array([len(coords)], dtype=wp.int32, device=DEVICE),
        origin=wp.vec3(*ORIGIN),
        voxel_size=H,
        coords=coords,
    )


def _centers(grid):
    """World centre of every voxel slot, in the grid's slot order."""
    out = []
    for lc in grid.coords:
        for k in range(C.LEAF_DIM):
            for j in range(C.LEAF_DIM):
                for i in range(C.LEAF_DIM):
                    out.append([
                        ORIGIN[0] + (lc[0] * C.LEAF_DIM + i + 0.5) * H,
                        ORIGIN[1] + (lc[1] * C.LEAF_DIM + j + 0.5) * H,
                        ORIGIN[2] + (lc[2] * C.LEAF_DIM + k + 0.5) * H,
                    ])
    return np.array(out)


def _box(lo, hi, *, missing=(), flip=False):
    lo, hi = np.array(lo, np.float32), np.array(hi, np.float32)
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)],
                       np.float32)
    points = lo + corners * (hi - lo)
    quads = {"-z": (0, 2, 3, 1), "+z": (4, 5, 7, 6), "-y": (0, 1, 5, 4),
             "+y": (2, 6, 7, 3), "-x": (0, 4, 6, 2), "+x": (1, 3, 7, 5)}
    tris = []
    for name, (a, b, c, d) in quads.items():
        if name in missing:
            continue
        pair = [[a, b, c], [a, c, d]]
        if flip:
            pair = [[t[0], t[2], t[1]] for t in pair]
        tris += pair
    return points, np.array(tris, np.int32).reshape(-1)


def _collider(points, indices, *, allow=False, moving=False):
    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=DEVICE),
        indices=wp.array(indices, dtype=wp.int32, device=DEVICE),
    )
    col = ColliderDesc(mesh=mesh, allow_height_fill=allow)
    if moving:
        col.base_points = wp.clone(mesh.points)
    return col


def _mask(colliders, *, base_height=2.0, solids=None):
    grid = _grid()
    builder = HeightFillSolids(DEVICE)
    mask = builder.build(
        grid, len(grid.coords) * C.LEAF_VOLUME, H,
        eligible_solids(colliders) if solids is None else solids,
        base_height=base_height, back_mode=1,
        profile=wp.zeros((2, 2), dtype=float, device=DEVICE),
        profile_res=wp.vec2i(2, 2),
        domain_min=wp.vec3(*DOMAIN_MIN), domain_max=wp.vec3(*DOMAIN_MAX),
    )
    return grid, builder, (None if mask is None else mask.numpy().astype(bool))


def _inside(points, lo, hi):
    return np.all((points > lo) & (points < hi), axis=1)


CUBE_LO, CUBE_HI = (0.8, 0.8, 0.8), (1.6, 1.6, 1.6)


def test_closed_mesh_is_flagged_inside_and_in_its_padding_only():
    grid, _, mask = _mask([_collider(*_box(CUBE_LO, CUBE_HI))])
    centers = _centers(grid)
    lo, hi = np.array(CUBE_LO), np.array(CUBE_HI)
    assert mask[_inside(centers, lo, hi)].all()
    pad = 1.5 * H
    # Just outside the surface but within the padding: flagged.
    near = _inside(centers, lo - pad * 0.9, hi + pad * 0.9) & ~_inside(centers, lo, hi)
    assert near.any() and mask[near].all()
    # Clearly beyond the padding: free.
    far = ~_inside(centers, lo - pad - H, hi + pad + H)
    assert not mask[far].any()


@pytest.mark.parametrize("missing", [("+z",), ("-z",), ("+z", "+y")])
def test_open_mesh_still_counts_as_solid(missing):
    grid, _, mask = _mask([_collider(*_box(CUBE_LO, CUBE_HI, missing=missing))])
    centers = _centers(grid)
    interior = _inside(centers, np.array(CUBE_LO) + 0.1, np.array(CUBE_HI) - 0.1)
    assert interior.any() and mask[interior].all()


def test_inside_out_mesh_still_counts_as_solid():
    grid, _, mask = _mask([_collider(*_box(CUBE_LO, CUBE_HI, flip=True))])
    centers = _centers(grid)
    interior = _inside(centers, np.array(CUBE_LO) + 0.1, np.array(CUBE_HI) - 0.1)
    assert mask[interior].all()


def test_floor_plane_does_not_block_the_space_above_it():
    points = np.array([[0, 0, 0.5], [2.4, 0, 0.5], [2.4, 2.4, 0.5], [0, 2.4, 0.5]],
                      np.float32)
    plane = _collider(points, np.array([0, 1, 2, 0, 2, 3], np.int32))
    grid, _, mask = _mask([plane])
    centers = _centers(grid)
    above = centers[:, 2] > 0.5 + 1.5 * H + 1e-3
    assert above.any() and not mask[above].any()
    right_above = (centers[:, 2] > 0.5) & (centers[:, 2] < 0.5 + 1.5 * H)
    assert mask[right_above].all()


def test_gap_between_two_meshes_narrower_than_the_padding_is_closed():
    left = _collider(*_box((0.8, 0.8, 0.8), (1.2, 1.6, 1.6)))
    right = _collider(*_box((1.3, 0.8, 0.8), (1.7, 1.6, 1.6)))  # 0.1 m gap
    grid, _, mask = _mask([left, right])
    centers = _centers(grid)
    gap = ((centers[:, 0] > 1.2) & (centers[:, 0] < 1.3)
           & (centers[:, 1] > 0.9) & (centers[:, 1] < 1.5)
           & (centers[:, 2] > 0.9) & (centers[:, 2] < 1.5))
    assert gap.any() and mask[gap].all()


def test_only_voxels_the_fill_reaches_are_tested():
    # Fill level 1.0 sits inside the cube: nothing above it is flagged.
    grid, _, mask = _mask([_collider(*_box(CUBE_LO, CUBE_HI))], base_height=1.0)
    centers = _centers(grid)
    assert not mask[centers[:, 2] > 1.0].any()
    assert mask[_inside(centers, np.array(CUBE_LO), np.array((1.6, 1.6, 1.0)))].all()


def test_override_leaves_the_mesh_out_of_the_mask():
    container = _collider(*_box(CUBE_LO, CUBE_HI, missing=("+z",)), allow=True)
    wall = _collider(*_box((0.0, 0.0, 0.0), (0.3, 0.3, 0.3)))
    assert eligible_solids([container, wall]) == [wall]
    assert eligible_solids([container]) == []
    grid, _, mask = _mask([container, wall])
    centers = _centers(grid)
    assert not mask[_inside(centers, np.array(CUBE_LO) + 0.1, np.array(CUBE_HI) - 0.1)].any()
    assert mask[_inside(centers, np.zeros(3), np.full(3, 0.3))].all()
    _, _, none = _mask([container])
    assert none is None


def test_static_query_mesh_is_reused_and_moving_one_is_rebuilt():
    static = _collider(*_box(CUBE_LO, CUBE_HI))
    grid, builder, _ = _mask([static])
    first = builder._queries[id(static.mesh)]
    builder.build(
        grid, len(grid.coords) * C.LEAF_VOLUME, H, [static], base_height=2.0,
        back_mode=1, profile=wp.zeros((2, 2), dtype=float, device=DEVICE),
        profile_res=wp.vec2i(2, 2), domain_min=wp.vec3(*DOMAIN_MIN),
        domain_max=wp.vec3(*DOMAIN_MAX))
    assert builder._queries[id(static.mesh)] is first

    moving = _collider(*_box(CUBE_LO, CUBE_HI), moving=True)
    grid, builder, mask = _mask([moving])
    centers = _centers(grid)
    assert mask[_inside(centers, np.array(CUBE_LO), np.array(CUBE_HI))].all()
    # Move the mesh in place; the next build must follow it.
    shifted = moving.mesh.points.numpy() - np.array([0.6, 0.0, 0.0], np.float32)
    moving.mesh.points.assign(shifted)
    moving.mesh.refit()
    mask2 = builder.build(
        grid, len(grid.coords) * C.LEAF_VOLUME, H, [moving], base_height=2.0,
        back_mode=1, profile=wp.zeros((2, 2), dtype=float, device=DEVICE),
        profile_res=wp.vec2i(2, 2), domain_min=wp.vec3(*DOMAIN_MIN),
        domain_max=wp.vec3(*DOMAIN_MAX)).numpy().astype(bool)
    assert mask2[_inside(centers, np.array((0.3, 0.9, 0.9)), np.array((0.9, 1.5, 1.5)))].all()
    assert not mask2[_inside(centers, np.array((1.2, 0.9, 0.9)), np.array((1.5, 1.5, 1.5)))].any()


def _emission_demand(grid, mask, *, exclusion):
    """Run the real per-voxel demand kernel for a flat height-fill shape."""
    shape = SourceShapeDesc(
        enabled=True, shape=2, stype=0, mode=1,
        position=(1.2, 1.2, 1.0), half_extents=(1.2, 1.2, 1.0),
        emit_rate=4.0, emission_mode=1, source_id="builtin:height",
        solid_exclusion=exclusion,
    ).to_struct()
    shapes = wp.array([shape], dtype=K.SourceShape, device=DEVICE)
    n = len(grid.coords) * C.LEAF_VOLUME
    demand = wp.zeros(n, dtype=wp.int32, device=DEVICE)
    counts = wp.zeros(n, dtype=wp.int32, device=DEVICE)
    state = wp.array([0, 1, 0, 0, 0], dtype=wp.int32, device=DEVICE)
    placeholder = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    wp.launch(
        K.k_source_shapes_particle_demand, dim=n,
        inputs=[shapes, 1, grid.leaf_coords, grid.num_leaves, grid.origin, H,
                counts, state, wp.zeros((2, 2), dtype=float, device=DEVICE),
                wp.vec2i(2, 2), wp.vec3(*DOMAIN_MIN), wp.vec3(*DOMAIN_MAX),
                placeholder if mask is None else mask],
        outputs=[demand], device=DEVICE)
    return demand.numpy()


def test_emission_skips_flagged_voxels_and_only_when_the_shape_asks():
    grid, _, flags = _mask([_collider(*_box(CUBE_LO, CUBE_HI))])
    mask = wp.array(flags.astype(np.int32), dtype=wp.int32, device=DEVICE)
    centers = _centers(grid)
    below = centers[:, 2] <= 2.0

    plain = _emission_demand(grid, None, exclusion=False)
    assert (plain[below] == 4).all()

    kept_out = _emission_demand(grid, mask, exclusion=True)
    assert (kept_out[flags] == 0).all()
    assert (kept_out[below & ~flags] == 4).all()
    assert (kept_out[~below] == 0).all()

    # A shape that does not set the flag ignores the mask (other emitters).
    ignored = _emission_demand(grid, mask, exclusion=False)
    assert np.array_equal(ignored, plain)


def _pipeline(colliders):
    from fluidSim_warp_blender51.sim.sourcing import (
        HeightSourceDesc, SourceEmissionMode, SourcePipeline)

    grid = _grid()
    grid.voxel_capacity = len(grid.coords) * C.LEAF_VOLUME
    settings = SimpleNamespace(
        voxel_size=H, domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX)
    particles = SimpleNamespace(capacity=1_000_000, host_count=0, next_uid=0)
    pipeline = SourcePipeline(
        device=DEVICE, grid=grid, particles=particles, settings=settings,
        active_voxel_count=lambda: grid.voxel_capacity,
        current_frame=lambda: 0, ensure_particle_capacity=None,
        invalidate_host_cache=lambda: None, invalidate_particle_surface=lambda: None,
        colliders=lambda: colliders,
    )
    pipeline.set_height_source(HeightSourceDesc(
        enabled=True, base_height=2.0, emit_particles=True, emit_rate=4.0,
        emission_mode=SourceEmissionMode.INITIAL_FILL))
    return pipeline, grid


def test_pipeline_applies_the_mask_to_the_height_shape_only_when_a_mesh_is_in_play():
    solid = _collider(*_box(CUBE_LO, CUBE_HI))
    pipeline, grid = _pipeline([solid])
    pipeline._prepare_legacy_emission()
    assert pipeline._transaction_nofill is not None
    shape = pipeline._emit_shape_array.numpy()[-1]
    assert int(shape["solid_exclusion"]) == 1
    centers = _centers(grid)
    flagged = pipeline._transaction_nofill.numpy().astype(bool)
    assert flagged[_inside(centers, np.array(CUBE_LO), np.array(CUBE_HI))].all()

    container = _collider(*_box(CUBE_LO, CUBE_HI), allow=True)
    pipeline, _ = _pipeline([container])
    pipeline._prepare_legacy_emission()
    assert pipeline._transaction_nofill is None
    assert int(pipeline._emit_shape_array.numpy()[-1]["solid_exclusion"]) == 0

    pipeline, _ = _pipeline([])
    pipeline._prepare_legacy_emission()
    assert pipeline._transaction_nofill is None


def test_dynamic_body_is_queried_at_its_current_pose():
    """A Dynamic collider's mesh stays in its body frame; the pose moves the keep-out."""
    from fluidSim_warp_blender51.sim import rigid_coupling  # noqa: F401
    from fluidSim_warp_blender51.sim.kernels_rigid import BODY_COM, BODY_ROT, BODY_WIDTH

    points, indices = _box((-0.4, -0.4, -0.4), (0.4, 0.4, 0.4))
    col = _collider(points, indices)
    col.dynamic = True
    body = SimpleNamespace(local_mesh=col.mesh, radius=0.7)
    state = np.zeros((1, BODY_WIDTH), np.float64)
    state[0, BODY_ROT:BODY_ROT + 9] = np.eye(3).ravel()
    state[0, BODY_COM:BODY_COM + 3] = (1.2, 1.2, 1.2)
    device_state = wp.array(state, dtype=wp.float64, device=DEVICE)

    grid = _grid()
    builder = HeightFillSolids(DEVICE)
    n = len(grid.coords) * C.LEAF_VOLUME
    centers = _centers(grid)

    def run():
        return builder.build(
            grid, n, H, [col], base_height=2.4, back_mode=1,
            profile=wp.zeros((2, 2), dtype=float, device=DEVICE),
            profile_res=wp.vec2i(2, 2), domain_min=wp.vec3(*DOMAIN_MIN),
            domain_max=wp.vec3(*DOMAIN_MAX),
            pose_binding=lambda _col: (device_state, 0, body)).numpy().astype(bool)

    mask = run()
    assert mask[_inside(centers, np.array((0.9, 0.9, 0.9)), np.array((1.5, 1.5, 1.5)))].all()
    assert not mask[_inside(centers, np.zeros(3), np.array((0.6, 0.6, 0.6)))].any()
    # The body moves on the device; the same query mesh follows without a rebuild.
    state[0, BODY_COM:BODY_COM + 3] = (0.5, 0.5, 0.5)
    device_state.assign(state)
    moved = run()
    assert moved[_inside(centers, np.array((0.2, 0.2, 0.2)), np.array((0.8, 0.8, 0.8)))].all()
    assert not moved[_inside(centers, np.array((1.5, 1.5, 1.5)), np.array((2.2, 2.2, 2.2)))].any()
    assert len(builder._queries) == 1


def test_disabled_height_fill_builds_no_mask():
    pipeline, _ = _pipeline([_collider(*_box(CUBE_LO, CUBE_HI))])
    pipeline.height_source.enabled = False
    pipeline._prepare_legacy_emission()
    assert pipeline._transaction_nofill is None
