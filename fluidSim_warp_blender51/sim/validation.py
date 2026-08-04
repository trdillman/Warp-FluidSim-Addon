"""GPU validation suite implementing the synthesized validation targets.

Each test runs real Warp kernels on the configured device and returns a
``ValidationResult``. ``run_all()`` executes every registered test and is the
backend for the Blender "Run Validation" operator and the headless test
harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import (
    LEAF_DIM,
    LEAF_VOLUME,
    ST_EPS_MASS,
    ST_EPS_RHO,
    VOXEL_ACTIVE,
    VOXEL_COLLISION,
    VOXEL_FLUID,
    VOXEL_SOURCE_REGION_TRACKING,
    VOXEL_SINK,
)
from . import fields as F
from . import kernels_core as KC
from . import kernels_fluid as KF
from . import pressure as P
from . import sourcing as SRC
from . import surface as SURF
from . import whitewater as WW
from .fields import MacVelocityField, ScalarField, SparseGrid
from .particles import ParticleSystem

VALIDATION_TESTS = [
    "sparse tiled leaf-grid coordinate/index roundtrip",
    "leaf-local voxel addressing and global start-vector reconstruction",
    "field precision parity for 16-bit and 32-bit modes",
    "particle-to-grid count/average/active masks",
    "grid-to-particle velocity/attribute sampling",
    "source shape coverage and source particle attribute transfer",
    "height-field sourcing flat/profile behavior",
    "NanoVDB/OpenVDB volume sourcing kernels",
    "collision/sink voxel-type bit behavior",
    "pressure projection divergence reduction",
    "Jacobi and PCG residual convergence",
    "surface SDF build/filter parity",
    "viscosity solve stability",
    "whitewater churn/emission/advection behavior",
    "benchmark profile schema compatibility",
    "spatiotemporal FLIP jitter, phase, and residual boundedness",
    "spatiotemporal velocity extrapolation band growth and validity gate",
    "spatiotemporal solid face apertures from a sub-cell collider interface",
    "spatiotemporal aperture-weighted pressure operator and projection",
    "spatiotemporal surface tension curvature, sign, and sphere stability",
]


@dataclass
class ValidationResult:
    name: str
    passed: bool
    detail: str = ""
    elapsed: float = 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "detail": self.detail,
            "elapsed": self.elapsed,
        }


_TESTS: list = []


def _test(name):
    def deco(fn):
        fn.validation_name = name
        _TESTS.append(fn)
        return fn

    return deco


def _make_grid(device, size=1.2, voxel_size=0.05, capacity=512):
    grid = SparseGrid(
        origin=(-size / 2, -size / 2, 0.0),
        size=(size, size, size),
        voxel_size=voxel_size,
        leaf_capacity=capacity,
        device=device,
    )
    grid.begin_topology()
    grid.mark_world_box(
        (-size / 2, -size / 2, 0.0), (size / 2, size / 2, size)
    )
    grid.finalize_topology()
    return grid


def _seed_particles(grid, device, spacing_frac=0.5, lo=(-0.4, -0.4, 0.1), hi=(0.4, 0.4, 0.7)):
    """Fill a world-space box with a regular particle lattice."""
    h = grid.voxel_size * spacing_frac
    xs = np.arange(lo[0], hi[0], h, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], h, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], h, dtype=np.float32)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    n = len(pts)
    psys = ParticleSystem(capacity=max(n * 2, 1024), device=device)
    # Direct upload into the first n slots.
    pos_np = np.zeros((psys.capacity, 3), dtype=np.float32)
    pos_np[:n] = pts
    wp.copy(psys.position, wp.array(pos_np, dtype=wp.vec3, device=device))
    psys.count.fill_(n)
    psys.host_count = n
    alive = np.zeros(psys.capacity, dtype=np.int32)
    alive[:n] = 1
    wp.copy(psys.alive, wp.array(alive, dtype=wp.int32, device=device))
    return psys, n


def _voxel_centers(grid):
    """Host-side centers for active leaf slots, in validation order."""
    leaf_coords = grid.leaf_coords.numpy()[: grid.active_leaf_count]
    centers = np.zeros((grid.voxel_capacity, 3), dtype=np.float32)
    for slot, lc in enumerate(leaf_coords):
        for lz in range(LEAF_DIM):
            for ly in range(LEAF_DIM):
                for lx in range(LEAF_DIM):
                    flat = slot * LEAF_VOLUME + (lz * LEAF_DIM + ly) * LEAF_DIM + lx
                    centers[flat] = (
                        np.array(grid.origin, dtype=np.float32)
                        + (np.array(
                            [
                                int(lc[0]) * LEAF_DIM + lx,
                                int(lc[1]) * LEAF_DIM + ly,
                                int(lc[2]) * LEAF_DIM + lz,
                            ],
                            dtype=np.float32,
                        ) + 0.5) * grid.voxel_size
                    )
    return centers


# ---------------------------------------------------------------------------
# 1. sparse tiled leaf-grid coordinate/index roundtrip
# ---------------------------------------------------------------------------


@wp.kernel
def _k_roundtrip(
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    failures: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = F.voxel_coord_of_flat(leaf_coords, tid)
    flat = F.voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat != tid:
        wp.atomic_add(failures, 0, 1)


@_test(VALIDATION_TESTS[0])
def test_leaf_grid_roundtrip(device):
    grid = _make_grid(device)
    failures = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _k_roundtrip, dim=grid.voxel_capacity,
        inputs=[grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                grid.leaves_dim, grid.num_leaves, failures],
        device=device,
    )
    bad = int(failures.numpy()[0])
    total = grid.active_voxel_count
    return bad == 0, f"{total} voxels roundtripped, {bad} failures"


# ---------------------------------------------------------------------------
# 2. leaf-local addressing / start vector reconstruction
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[1])
def test_leaf_addressing(device):
    grid = _make_grid(device)
    leaf_coords = grid.leaf_coords.numpy()[: grid.active_leaf_count]
    leaf_table = grid.leaf_table.numpy()
    ld = grid.leaves_dim
    bad = 0
    for slot, lc in enumerate(leaf_coords):
        linear = (int(lc[2]) * ld[1] + int(lc[1])) * ld[0] + int(lc[0])
        if leaf_table[linear] != slot:
            bad += 1
        # start vector reconstruction: global voxel of local (0,0,0)
        gx, gy, gz = int(lc[0]) * LEAF_DIM, int(lc[1]) * LEAF_DIM, int(lc[2]) * LEAF_DIM
        if (gx % LEAF_DIM, gy % LEAF_DIM, gz % LEAF_DIM) != (0, 0, 0):
            bad += 1
    return bad == 0, f"{len(leaf_coords)} leaves validated, {bad} mismatches"


# ---------------------------------------------------------------------------
# 3. field precision parity 16/32-bit
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[2])
def test_precision_parity(device):
    grid = _make_grid(device)
    n = grid.voxel_capacity
    rng = np.random.default_rng(7)
    values = rng.uniform(-10.0, 10.0, n).astype(np.float32)
    f32 = wp.array(values, dtype=wp.float32, device=device)
    f16 = wp.zeros(n, dtype=wp.float16, device=device)
    back = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(KC.k_convert_f32_to_f16, dim=n, inputs=[f32, f16], device=device)
    wp.launch(KC.k_convert_f16_to_f32, dim=n, inputs=[f16, back], device=device)
    err = np.abs(back.numpy() - values)
    rel = err / np.maximum(np.abs(values), 1.0)
    max_rel = float(rel.max())
    # half precision has ~1e-3 relative error at this range
    return max_rel < 5.0e-3, f"max relative error {max_rel:.2e} (limit 5e-3)"


# ---------------------------------------------------------------------------
# 4. particle-to-grid count/average/active masks
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[3])
def test_p2g_masks(device):
    grid = _make_grid(device)
    psys, n = _seed_particles(grid, device)
    voxel_counts = ScalarField(grid, dtype=wp.int32, name="counts")
    voxel_type = ScalarField(grid, dtype=wp.int32, name="type")
    voxel_counts.zero()
    voxel_type.zero()
    wp.launch(
        KF.k_particle_to_count, dim=psys.capacity,
        inputs=[psys.position, psys.count, grid.leaf_table, grid.voxels_dim,
                grid.leaves_dim, grid.origin, grid.voxel_size],
        outputs=[voxel_counts.data], device=device,
    )
    wp.launch(
        KF.k_particle_to_grid_active, dim=grid.voxel_capacity,
        inputs=[voxel_counts.data, grid.num_leaves],
        outputs=[voxel_type.data], device=device,
    )
    counts = voxel_counts.data.numpy()
    types = voxel_type.data.numpy()
    total_counted = int(counts.sum())
    active = int(((types & VOXEL_FLUID) != 0).sum())
    consistent = bool(np.all((counts > 0) == ((types & VOXEL_FLUID) != 0)))
    ok = total_counted == n and active > 0 and consistent
    return ok, (
        f"counted {total_counted}/{n} particles, {active} fluid voxels, "
        f"mask consistent={consistent}"
    )


# ---------------------------------------------------------------------------
# 5. grid-to-particle sampling
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[4])
def test_g2p_sampling(device):
    grid = _make_grid(device)
    psys, n = _seed_particles(grid, device)
    vel = MacVelocityField(grid)
    vel.zero()
    # Constant velocity field: set all face values to (1, 2, 3) with weight 1.
    for arr, val in ((vel.u, 1.0), (vel.v, 2.0), (vel.w, 3.0)):
        arr.fill_(val)
    for arr in (vel.wu, vel.wv, vel.ww):
        arr.fill_(1.0)
    wp.launch(
        KF.k_staggered_field_to_particle, dim=psys.capacity,
        inputs=[psys.position, psys.count,
                vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww,
                vel.u, vel.v, vel.w,  # old == new -> FLIP delta = 0
                grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                grid.origin, grid.voxel_size,
                0, 1.0],  # PIC
        outputs=[psys.velocity], device=device,
    )
    v = psys.velocity.numpy()[:n]
    err = np.abs(v - np.array([1.0, 2.0, 3.0], dtype=np.float32)).max()
    return err < 1.0e-4, f"max abs sampling error {err:.2e} (constant field)"


# ---------------------------------------------------------------------------
# 6. source shape coverage + particle attribute transfer
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[5])
def test_source_shapes(device):
    from .enums import (
        SourceShapeShape,
        SourceShapeSourceMode,
        SourceShapeType,
    )
    from .solver import FluidSolver, SolverSettings, SourceShapeDesc

    settings = SolverSettings()
    settings.domain_min = (-0.6, -0.6, 0.0)
    settings.domain_max = (0.6, 0.6, 1.2)
    settings.voxel_size = 0.05
    settings.max_particles = 200_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.device = device

    src = SourceShapeDesc()
    src.shape = SourceShapeShape.SPHERE
    src.stype = SourceShapeType.SOURCE
    src.mode = SourceShapeSourceMode.PARTICLES
    src.position = (0.0, 0.0, 0.6)
    src.radius = 0.25
    src.emit_rate = 8.0
    src.temperature = 42.0
    src.viscosity = 3.5

    solver = FluidSolver(settings)
    solver.set_sources([src])
    solver.step_frame()
    n = solver.particles.refresh_host_count()
    if n == 0:
        return False, "no particles emitted"
    pos = solver.particles.position.numpy()[:n]
    center = np.array([0.0, 0.0, 0.6])
    # Particles were emitted inside the sphere; after one frame of gravity
    # they may drift slightly, allow a small tolerance band.
    dist = np.linalg.norm(pos - center, axis=1)
    frac_inside = float((dist < 0.25 + 3 * settings.voxel_size).mean())
    temp = solver.particles.attr("temperature").numpy()[:n]
    visc = solver.particles.attr("viscosity").numpy()[:n]
    temp_ok = bool(np.all(np.abs(temp - 42.0) < 1.0e-3))
    visc_ok = bool(np.all(np.abs(visc - 3.5) < 1.0e-3))
    ok = n > 100 and frac_inside > 0.95 and temp_ok and visc_ok
    return ok, (
        f"{n} particles, {frac_inside * 100:.1f}% within source bounds, "
        f"temperature transfer={temp_ok}, viscosity transfer={visc_ok}"
    )


# ---------------------------------------------------------------------------
# 7. height-field sourcing flat/profile behavior
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[6])
def test_height_field_sourcing(device):
    grid = _make_grid(device, size=0.8, voxel_size=0.1, capacity=64)
    centers = _voxel_centers(grid)

    profile = wp.array2d(
        np.array(
            [
                [0.0, 0.0],
                [0.4, 0.4],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device=device,
    )
    empty_profile = wp.zeros((1, 1), dtype=wp.float32, device=device)

    flat_type = ScalarField(grid, dtype=wp.int32, name="height_flat_type")
    flat_u = ScalarField(grid, name="height_flat_u")
    flat_v = ScalarField(grid, name="height_flat_v")
    flat_w = ScalarField(grid, name="height_flat_w")
    for field in (flat_type, flat_u, flat_v, flat_w):
        field.zero()

    wp.launch(
        SRC.k_source_height,
        dim=grid.voxel_capacity,
        inputs=[
            grid.leaf_coords, grid.num_leaves, grid.origin, grid.voxel_size,
            0.25, 1, empty_profile, wp.vec2i(1, 1),
            grid.origin,
            wp.vec3(
                grid.origin[0] + 0.8,
                grid.origin[1] + 0.8,
                grid.origin[2] + 0.8,
            ),
            wp.vec3(1.0, 2.0, 3.0),
            flat_type.data, flat_u.data, flat_v.data, flat_w.data,
        ],
        device=device,
    )

    flat_types = flat_type.data.numpy()
    flat_fluid = (flat_types & VOXEL_FLUID) != 0
    flat_expected = centers[:, 2] <= 0.25 + 1.0e-6
    flat_ok = bool(np.array_equal(flat_fluid, flat_expected))
    flat_tracking = bool(np.all((flat_types[flat_expected] & VOXEL_SOURCE_REGION_TRACKING) != 0))
    flat_velocity = bool(
        np.allclose(flat_u.data.numpy()[flat_expected], 1.0)
        and np.allclose(flat_v.data.numpy()[flat_expected], 2.0)
        and np.allclose(flat_w.data.numpy()[flat_expected], 3.0)
    )

    prof_type = ScalarField(grid, dtype=wp.int32, name="height_profile_type")
    prof_u = ScalarField(grid, name="height_profile_u")
    prof_v = ScalarField(grid, name="height_profile_v")
    prof_w = ScalarField(grid, name="height_profile_w")
    for field in (prof_type, prof_u, prof_v, prof_w):
        field.zero()

    wp.launch(
        SRC.k_source_height,
        dim=grid.voxel_capacity,
        inputs=[
            grid.leaf_coords, grid.num_leaves, grid.origin, grid.voxel_size,
            0.10, 2, profile, wp.vec2i(2, 2),
            grid.origin,
            wp.vec3(
                grid.origin[0] + 0.8,
                grid.origin[1] + 0.8,
                grid.origin[2] + 0.8,
            ),
            wp.vec3(-1.0, 0.5, 4.0),
            prof_type.data, prof_u.data, prof_v.data, prof_w.data,
        ],
        device=device,
    )

    prof_types = prof_type.data.numpy()
    prof_fluid = (prof_types & VOXEL_FLUID) != 0
    fx = np.clip((centers[:, 0] - grid.origin[0]) / 0.8, 0.0, 1.0)
    prof_expected = centers[:, 2] <= 0.10 + 0.4 * fx + 1.0e-6
    gained_from_profile = int((prof_fluid & ~flat_expected).sum())
    profile_ok = bool(np.array_equal(prof_fluid, prof_expected))
    profile_velocity = bool(
        np.allclose(prof_u.data.numpy()[prof_expected], -1.0)
        and np.allclose(prof_v.data.numpy()[prof_expected], 0.5)
        and np.allclose(prof_w.data.numpy()[prof_expected], 4.0)
    )

    ok = flat_ok and flat_tracking and flat_velocity and profile_ok and profile_velocity
    return ok, (
        f"flat marked {int(flat_fluid.sum())} voxels; profile marked "
        f"{int(prof_fluid.sum())} voxels ({gained_from_profile} profile-only), "
        f"flat_ok={flat_ok}, profile_ok={profile_ok}"
    )


# ---------------------------------------------------------------------------
# 8. NanoVDB/OpenVDB volume sourcing kernels
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[7])
def test_volume_sourcing_kernels(device):
    from .registry import lookup

    grid = _make_grid(device, size=0.8, voxel_size=0.1, capacity=64)
    centers = _voxel_centers(grid)
    interior = np.all(
        centers <= np.array(grid.origin, dtype=np.float32) + 0.65 + 1.0e-6,
        axis=1,
    )
    shape = (8, 8, 8)
    fog_np = np.zeros(shape, dtype=np.float32)
    fog_np[:4, :, :] = 1.0
    sdf_np = np.ones(shape, dtype=np.float32)
    sdf_np[:4, :, :] = -1.0
    temp_np = np.full(shape, 37.5, dtype=np.float32)
    scalar_np = np.full(shape, 2.0, dtype=np.float32)
    vector_np = np.zeros(shape + (3,), dtype=np.float32)
    vector_np[..., 0] = 1.25
    vector_np[..., 1] = -2.5
    vector_np[..., 2] = 4.0

    volume_kwargs = {
        "min_world": tuple(float(x) for x in grid.origin),
        "voxel_size": grid.voxel_size,
        "device": device,
    }
    fog_volume = wp.Volume.load_from_numpy(fog_np, bg_value=0.0, **volume_kwargs)
    sdf_volume = wp.Volume.load_from_numpy(sdf_np, bg_value=1.0, **volume_kwargs)
    temp_volume = wp.Volume.load_from_numpy(temp_np, bg_value=0.0, **volume_kwargs)
    scalar_volume = wp.Volume.load_from_numpy(scalar_np, bg_value=0.0, **volume_kwargs)
    vector_volume = wp.Volume.load_from_numpy(
        vector_np, bg_value=(0.0, 0.0, 0.0), **volume_kwargs
    )

    scalar_dst = ScalarField(grid, name="volume_scalar")
    scalar_dst.zero()
    wp.launch(
        SRC.k_source_volume_float,
        dim=grid.voxel_capacity,
        inputs=[
            scalar_volume.id, grid.leaf_coords, grid.num_leaves,
            grid.origin, grid.voxel_size, 1, 1.0, 3.0, scalar_dst.data,
        ],
        device=device,
    )

    vec_u = ScalarField(grid, name="volume_u")
    vec_v = ScalarField(grid, name="volume_v")
    vec_w = ScalarField(grid, name="volume_w")
    for field in (vec_u, vec_v, vec_w):
        field.zero()
    wp.launch(
        SRC.k_source_volume_vector,
        dim=grid.voxel_capacity,
        inputs=[
            vector_volume.id, grid.leaf_coords, grid.num_leaves,
            grid.origin, grid.voxel_size, 1, 1.0, 2.0,
            vec_u.data, vec_v.data, vec_w.data,
        ],
        device=device,
    )

    component_dst = ScalarField(grid, name="volume_component")
    component_dst.zero()
    wp.launch(
        SRC.k_source_volume_vector_component,
        dim=grid.voxel_capacity,
        inputs=[
            vector_volume.id, 2, grid.leaf_coords, grid.num_leaves,
            grid.origin, grid.voxel_size, 1, 1.0, 0.5, component_dst.data,
        ],
        device=device,
    )

    fog_type = ScalarField(grid, dtype=wp.int32, name="volume_fog_type")
    fog_u = ScalarField(grid, name="volume_fog_u")
    fog_v = ScalarField(grid, name="volume_fog_v")
    fog_w = ScalarField(grid, name="volume_fog_w")
    fog_temp = ScalarField(grid, name="volume_fog_temp")
    for field in (fog_type, fog_u, fog_v, fog_w, fog_temp):
        field.zero()
    wp.launch(
        SRC.k_source_volume_type_fog_with_temperature,
        dim=grid.voxel_capacity,
        inputs=[
            fog_volume.id, temp_volume.id, grid.leaf_coords, grid.num_leaves,
            grid.origin, grid.voxel_size, 0.5, wp.vec3(3.0, 4.0, 5.0),
            fog_type.data, fog_u.data, fog_v.data, fog_w.data, fog_temp.data,
        ],
        device=device,
    )

    fog_plain_type = ScalarField(grid, dtype=wp.int32, name="volume_fog_plain_type")
    fog_plain_u = ScalarField(grid, name="volume_fog_plain_u")
    fog_plain_v = ScalarField(grid, name="volume_fog_plain_v")
    fog_plain_w = ScalarField(grid, name="volume_fog_plain_w")
    for field in (fog_plain_type, fog_plain_u, fog_plain_v, fog_plain_w):
        field.zero()
    wp.launch(
        SRC.k_source_volume_type_fog,
        dim=grid.voxel_capacity,
        inputs=[
            fog_volume.id, grid.leaf_coords, grid.num_leaves,
            grid.origin, grid.voxel_size, 0.5, wp.vec3(7.0, 8.0, 9.0),
            fog_plain_type.data,
            fog_plain_u.data, fog_plain_v.data, fog_plain_w.data,
        ],
        device=device,
    )

    mask_type = ScalarField(grid, dtype=wp.int32, name="volume_mask_type")
    mask_type.zero()
    wp.launch(
        SRC.k_source_volume_mask_fog,
        dim=grid.voxel_capacity,
        inputs=[
            fog_volume.id, grid.leaf_coords, grid.num_leaves, grid.origin,
            grid.voxel_size, 0.5, VOXEL_SINK, mask_type.data,
        ],
        device=device,
    )

    sdf_type = ScalarField(grid, dtype=wp.int32, name="volume_sdf_type")
    sdf_u = ScalarField(grid, name="volume_sdf_u")
    sdf_v = ScalarField(grid, name="volume_sdf_v")
    sdf_w = ScalarField(grid, name="volume_sdf_w")
    sdf_temp = ScalarField(grid, name="volume_sdf_temp")
    for field in (sdf_type, sdf_u, sdf_v, sdf_w, sdf_temp):
        field.zero()
    wp.launch(
        SRC.k_source_volume_type_sdf_with_temperature,
        dim=grid.voxel_capacity,
        inputs=[
            sdf_volume.id, temp_volume.id, grid.leaf_coords, grid.num_leaves,
            grid.origin, grid.voxel_size, wp.vec3(-2.0, 0.0, 1.0),
            sdf_type.data, sdf_u.data, sdf_v.data, sdf_w.data, sdf_temp.data,
        ],
        device=device,
    )

    sdf_plain_type = ScalarField(grid, dtype=wp.int32, name="volume_sdf_plain_type")
    sdf_plain_u = ScalarField(grid, name="volume_sdf_plain_u")
    sdf_plain_v = ScalarField(grid, name="volume_sdf_plain_v")
    sdf_plain_w = ScalarField(grid, name="volume_sdf_plain_w")
    for field in (sdf_plain_type, sdf_plain_u, sdf_plain_v, sdf_plain_w):
        field.zero()
    wp.launch(
        SRC.k_source_volume_type_sdf,
        dim=grid.voxel_capacity,
        inputs=[
            sdf_volume.id, grid.leaf_coords, grid.num_leaves,
            grid.origin, grid.voxel_size, wp.vec3(0.25, 0.5, 0.75),
            sdf_plain_type.data,
            sdf_plain_u.data, sdf_plain_v.data, sdf_plain_w.data,
        ],
        device=device,
    )

    sdf_mask_type = ScalarField(grid, dtype=wp.int32, name="volume_sdf_mask")
    sdf_mask_type.zero()
    wp.launch(
        SRC.k_source_volume_mask_sdf,
        dim=grid.voxel_capacity,
        inputs=[
            sdf_volume.id, grid.leaf_coords, grid.num_leaves, grid.origin,
            grid.voxel_size, VOXEL_ACTIVE, sdf_mask_type.data,
        ],
        device=device,
    )

    scalar_ok = bool(np.allclose(scalar_dst.data.numpy()[interior], 6.0))
    vector_ok = bool(
        np.allclose(vec_u.data.numpy()[interior], 2.5)
        and np.allclose(vec_v.data.numpy()[interior], -5.0)
        and np.allclose(vec_w.data.numpy()[interior], 8.0)
        and np.allclose(component_dst.data.numpy()[interior], 2.0)
    )

    fog_types = fog_type.data.numpy()
    fog_marked = (fog_types & VOXEL_FLUID) != 0
    fog_plain_types = fog_plain_type.data.numpy()
    fog_plain_marked = (fog_plain_types & VOXEL_FLUID) != 0
    fog_ok = bool(
        fog_marked.any()
        and np.all((fog_types[fog_marked] & VOXEL_SOURCE_REGION_TRACKING) != 0)
        and np.allclose(fog_u.data.numpy()[fog_marked], 3.0)
        and np.allclose(fog_v.data.numpy()[fog_marked], 4.0)
        and np.allclose(fog_w.data.numpy()[fog_marked], 5.0)
        and np.allclose(fog_temp.data.numpy()[fog_marked], 37.5)
        and np.array_equal(fog_plain_marked, fog_marked)
        and np.allclose(fog_plain_u.data.numpy()[fog_plain_marked], 7.0)
        and np.allclose(fog_plain_v.data.numpy()[fog_plain_marked], 8.0)
        and np.allclose(fog_plain_w.data.numpy()[fog_plain_marked], 9.0)
    )
    fog_mask_count = int(((mask_type.data.numpy() & VOXEL_SINK) != 0).sum())

    sdf_types = sdf_type.data.numpy()
    sdf_marked = (sdf_types & VOXEL_FLUID) != 0
    sdf_plain_types = sdf_plain_type.data.numpy()
    sdf_plain_marked = (sdf_plain_types & VOXEL_FLUID) != 0
    sdf_ok = bool(
        sdf_marked.any()
        and np.all((sdf_types[sdf_marked] & VOXEL_SOURCE_REGION_TRACKING) != 0)
        and np.allclose(sdf_u.data.numpy()[sdf_marked], -2.0)
        and np.allclose(sdf_w.data.numpy()[sdf_marked], 1.0)
        and np.allclose(sdf_temp.data.numpy()[sdf_marked], 37.5)
        and np.array_equal(sdf_plain_marked, sdf_marked)
        and np.allclose(sdf_plain_u.data.numpy()[sdf_plain_marked], 0.25)
        and np.allclose(sdf_plain_v.data.numpy()[sdf_plain_marked], 0.5)
        and np.allclose(sdf_plain_w.data.numpy()[sdf_plain_marked], 0.75)
    )
    sdf_mask_count = int(((sdf_mask_type.data.numpy() & VOXEL_ACTIVE) != 0).sum())

    alias_ok = (
        lookup("coreSourceOpenVDBFloat").fn is lookup("coreSourceNanoVDBFloat").fn
        and lookup("coreSourceOpenVDBType").fn is lookup("coreSourceNanoVDBTypeFog").fn
        and lookup("coreSourceOpenVDBTypeWithTemperature").fn
        is lookup("coreSourceNanoVDBTypeFogWithTemperature").fn
        and lookup("coreSourceOpenVDBMask").fn is lookup("coreSourceNanoVDBMaskFog").fn
    )
    ok = (
        scalar_ok and vector_ok and fog_ok and fog_mask_count > 0
        and sdf_ok and sdf_mask_count > 0 and alias_ok
    )
    return ok, (
        f"scalar={scalar_ok}, vector={vector_ok}, fog fluid={int(fog_marked.sum())}, "
        f"fog masks={fog_mask_count}, sdf fluid={int(sdf_marked.sum())}, "
        f"sdf masks={sdf_mask_count}, openvdb aliases={alias_ok}"
    )


# ---------------------------------------------------------------------------
# 9. collision/sink voxel bit behavior
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[8])
def test_collision_sink_bits(device):
    from .enums import SourceShapeShape, SourceShapeSourceMode, SourceShapeType
    from .solver import FluidSolver, SolverSettings, SourceShapeDesc

    settings = SolverSettings()
    settings.domain_min = (-0.6, -0.6, 0.0)
    settings.domain_max = (0.6, 0.6, 1.2)
    settings.voxel_size = 0.05
    settings.max_particles = 200_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.device = device

    box = SourceShapeDesc()
    box.shape = SourceShapeShape.BOX
    box.stype = SourceShapeType.COLLISION
    box.position = (0.0, 0.0, 0.15)
    box.half_extents = (0.2, 0.2, 0.15)

    sink = SourceShapeDesc()
    sink.shape = SourceShapeShape.SPHERE
    sink.stype = SourceShapeType.SINK
    sink.position = (0.4, 0.4, 0.9)
    sink.radius = 0.12

    solver = FluidSolver(settings)
    solver.set_sources([box, sink])
    solver.step_frame()

    types = solver.voxel_type.data.numpy()
    n_col = int(((types & VOXEL_COLLISION) != 0).sum())
    n_sink = int(((types & VOXEL_SINK) != 0).sum())
    overlap = int((((types & VOXEL_COLLISION) != 0) & ((types & VOXEL_SINK) != 0)).sum())
    exp_col = (0.4 * 0.4 * 0.3) / settings.voxel_size**3
    exp_sink = (4.0 / 3.0 * np.pi * 0.12**3) / settings.voxel_size**3
    col_ok = 0.5 * exp_col < n_col < 2.0 * exp_col
    sink_ok = 0.5 * exp_sink < n_sink < 2.0 * exp_sink
    ok = col_ok and sink_ok and overlap == 0
    return ok, (
        f"collision voxels {n_col} (expect ~{exp_col:.0f}), "
        f"sink voxels {n_sink} (expect ~{exp_sink:.0f}), overlap {overlap}"
    )


# ---------------------------------------------------------------------------
# 10. pressure projection divergence reduction
# ---------------------------------------------------------------------------


def _divergence_norm(solver):
    solver.pressure_solver.build_rhs(
        solver.vel, solver.voxel_type.data, solver.collision_vel,
        solver.voxel_counts.data, rest_count=8.0, count_stiffness=0.0,
    )
    div = solver.pressure_solver.divergence.numpy()
    return float(np.abs(div).max())


@_test(VALIDATION_TESTS[9])
def test_divergence_reduction(device):
    from .enums import SourceShapeShape, SourceShapeSourceMode, SourceShapeType
    from .solver import FluidSolver, SolverSettings, SourceShapeDesc

    settings = SolverSettings()
    settings.domain_min = (-0.6, -0.6, 0.0)
    settings.domain_max = (0.6, 0.6, 1.2)
    settings.voxel_size = 0.05
    settings.max_particles = 500_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.device = device

    src = SourceShapeDesc()
    src.shape = SourceShapeShape.BOX
    src.stype = SourceShapeType.SOURCE
    src.mode = SourceShapeSourceMode.PARTICLES
    src.position = (0.0, 0.0, 0.35)
    src.half_extents = (0.5, 0.5, 0.3)
    src.emit_rate = 8.0

    solver = FluidSolver(settings)
    solver.set_sources([src])
    solver.step_frame()
    solver.set_sources([])
    for _ in range(2):
        solver.step_frame()

    # Measure divergence of the CURRENT velocity field, project, re-measure.
    div_before = _divergence_norm(solver)
    solver.pressure_solver.build_preconditioner(solver.voxel_type.data)
    solver.pressure_solver.solve_pcg(
        solver.voxel_type.data, max_iterations=400, tolerance=1.0e-7
    )
    solver.pressure_solver.project(
        solver.vel, solver.voxel_type.data, solver.collision_vel
    )
    div_after = _divergence_norm(solver)
    reduced = div_after < div_before * 0.05 or div_after < 1.0e-4
    return reduced, (
        f"max |divergence| {div_before:.3e} -> {div_after:.3e} "
        f"({solver.pressure_solver.last_iterations} PCG iterations)"
    )


# ---------------------------------------------------------------------------
# 11. Jacobi and PCG residual convergence
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[10])
def test_solver_convergence(device):
    grid = _make_grid(device, size=0.8)
    psys, n = _seed_particles(
        grid, device, lo=(-0.3, -0.3, 0.05), hi=(0.3, 0.3, 0.5)
    )
    voxel_counts = ScalarField(grid, dtype=wp.int32)
    voxel_type = ScalarField(grid, dtype=wp.int32)
    voxel_counts.zero()
    voxel_type.zero()
    wp.launch(
        KF.k_particle_to_count, dim=psys.capacity,
        inputs=[psys.position, psys.count, grid.leaf_table, grid.voxels_dim,
                grid.leaves_dim, grid.origin, grid.voxel_size],
        outputs=[voxel_counts.data], device=device,
    )
    wp.launch(
        KF.k_particle_to_grid_active, dim=grid.voxel_capacity,
        inputs=[voxel_counts.data, grid.num_leaves],
        outputs=[voxel_type.data], device=device,
    )
    vel = MacVelocityField(grid)
    vel.zero()
    # Divergent field: u = x (du/dx = 1 everywhere).
    u_np = np.zeros(grid.voxel_capacity, dtype=np.float32)
    coords = grid.leaf_coords.numpy()[: grid.active_leaf_count]
    for slot, lc in enumerate(coords):
        for lz in range(LEAF_DIM):
            for ly in range(LEAF_DIM):
                base = slot * LEAF_VOLUME + (lz * LEAF_DIM + ly) * LEAF_DIM
                gx0 = lc[0] * LEAF_DIM
                for lx in range(LEAF_DIM):
                    u_np[base + lx] = (gx0 + lx) * grid.voxel_size
    wp.copy(vel.u, wp.array(u_np, dtype=wp.float32, device=device))
    for arr in (vel.wu, vel.wv, vel.ww):
        arr.fill_(1.0)

    collision_vel = wp.zeros(grid.voxel_capacity, dtype=wp.vec3, device=device)
    ps = P.PressureSolver(grid, device)

    ps.build_rhs(vel, voxel_type.data, collision_vel, voxel_counts.data)
    ps.build_preconditioner(voxel_type.data)
    ps.solve_jacobi(voxel_type.data, iterations=400)
    res_jacobi = ps.measure_residual(voxel_type.data)

    ps.build_rhs(vel, voxel_type.data, collision_vel, voxel_counts.data)
    ps.solve_pcg(voxel_type.data, max_iterations=400, tolerance=1.0e-8)
    res_pcg = ps.last_residual

    rhs = ps.rhs.numpy()
    rhs_norm = float(np.linalg.norm(rhs))
    ok = res_jacobi < rhs_norm * 0.05 and res_pcg < rhs_norm * 1.0e-3
    return ok, (
        f"|rhs|={rhs_norm:.3e}, Jacobi residual {res_jacobi:.3e}, "
        f"PCG residual {res_pcg:.3e} ({ps.last_iterations} iters)"
    )


# ---------------------------------------------------------------------------
# 12. surface SDF build/filter parity
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[11])
def test_surface_sdf(device):
    grid = _make_grid(device)
    # Ball of particles at center.
    psys, n = _seed_particles(
        grid, device, spacing_frac=0.5,
        lo=(-0.25, -0.25, 0.35), hi=(0.25, 0.25, 0.85),
    )
    sdf = ScalarField(grid)
    tmp = ScalarField(grid)
    wp.launch(
        SURF.k_surface_sdf_reset, dim=grid.voxel_capacity,
        inputs=[sdf.data, grid.num_leaves, 4.0 * grid.voxel_size],
        device=device,
    )
    wp.launch(
        SURF.k_build_surface_sdf, dim=psys.capacity,
        inputs=[psys.position, psys.count, grid.leaf_table, grid.voxels_dim,
                grid.leaves_dim, grid.origin, grid.voxel_size,
                0.5 * grid.voxel_size * 1.2, 2],
        outputs=[sdf.data], device=device,
    )
    raw = sdf.data.numpy().copy()
    inside = int((raw < 0.0).sum())
    if inside == 0:
        return False, "no interior voxels after SDF build"
    # Box filter parity: filtered field must stay signed and keep interior.
    wp.launch(
        SURF.k_smooth_surface_sdf_box, dim=grid.voxel_capacity,
        inputs=[sdf.data, grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                grid.leaves_dim, grid.num_leaves, grid.voxel_size, 0.7],
        outputs=[tmp.data], device=device,
    )
    filtered = tmp.data.numpy()
    inside_f = int((filtered < 0.0).sum())
    # Interior voxel count should not change drastically under one box pass.
    ratio = inside_f / inside
    ok = 0.5 < ratio < 1.5 and np.isfinite(filtered).all()
    box_vol = 0.5 * 0.5 * 0.5
    expect_inside = box_vol / grid.voxel_size**3
    vol_ok = 0.4 * expect_inside < inside < 1.6 * expect_inside
    return ok and vol_ok, (
        f"interior voxels raw={inside} (expect ~{expect_inside:.0f}), "
        f"after box filter={inside_f} (ratio {ratio:.2f})"
    )


# ---------------------------------------------------------------------------
# 13. viscosity solve stability
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[12])
def test_viscosity_stability(device):
    from .viscosity import ViscositySolver

    grid = _make_grid(device, size=0.8)
    psys, n = _seed_particles(
        grid, device, lo=(-0.3, -0.3, 0.05), hi=(0.3, 0.3, 0.5)
    )
    voxel_counts = ScalarField(grid, dtype=wp.int32)
    voxel_type = ScalarField(grid, dtype=wp.int32)
    voxel_counts.zero()
    voxel_type.zero()
    wp.launch(
        KF.k_particle_to_count, dim=psys.capacity,
        inputs=[psys.position, psys.count, grid.leaf_table, grid.voxels_dim,
                grid.leaves_dim, grid.origin, grid.voxel_size],
        outputs=[voxel_counts.data], device=device,
    )
    wp.launch(
        KF.k_particle_to_grid_active, dim=grid.voxel_capacity,
        inputs=[voxel_counts.data, grid.num_leaves],
        outputs=[voxel_type.data], device=device,
    )
    vel = MacVelocityField(grid)
    vel.zero()
    rng = np.random.default_rng(3)
    noisy = rng.uniform(-2.0, 2.0, grid.voxel_capacity).astype(np.float32)
    wp.copy(vel.u, wp.array(noisy, dtype=wp.float32, device=device))
    for arr in (vel.wu, vel.wv, vel.ww):
        arr.fill_(1.0)
    max_before = float(np.abs(noisy).max())

    visc_field = ScalarField(grid)
    visc_field.data.fill_(5.0)  # strong viscosity
    vsolver = ViscositySolver(grid, device)
    vsolver.solve(
        vel, visc_field.data, voxel_type.data, dt=1.0 / 24.0,
        use_pcg=True, iterations=100, tolerance=1.0e-6,
    )
    u_after = vel.u.numpy()
    max_after = float(np.abs(u_after).max())
    finite = bool(np.isfinite(u_after).all())
    # Implicit viscosity must damp, never amplify; strong viscosity should
    # smooth the random field significantly.
    ok = finite and max_after <= max_before * 1.001 and max_after < max_before * 0.9
    return ok, (
        f"max |u| {max_before:.3f} -> {max_after:.3f}, finite={finite}"
    )


# ---------------------------------------------------------------------------
# 14. whitewater churn/emission/advection
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[13])
def test_whitewater(device):
    from .enums import SourceShapeShape, SourceShapeSourceMode, SourceShapeType
    from .solver import FluidSolver, SolverSettings, SourceShapeDesc

    settings = SolverSettings()
    settings.domain_min = (-0.8, -0.8, 0.0)
    settings.domain_max = (0.8, 0.8, 1.6)
    settings.voxel_size = 0.05
    settings.max_particles = 500_000
    settings.max_whitewater = 200_000
    settings.whitewater_enabled = True
    settings.surface_enabled = True
    settings.ww_churn_threshold = 0.02
    settings.ww_emission_rate = 200.0
    settings.device = device

    # Fast dam-break generates churn at impact.
    src = SourceShapeDesc()
    src.shape = SourceShapeShape.BOX
    src.stype = SourceShapeType.SOURCE
    src.mode = SourceShapeSourceMode.PARTICLES
    src.position = (-0.4, 0.0, 0.9)
    src.half_extents = (0.35, 0.7, 0.6)
    src.emit_rate = 8.0
    src.velocity = (2.5, 0.0, -1.5)

    solver = FluidSolver(settings)
    solver.set_sources([src])
    for _ in range(3):
        solver.step_frame()
    solver.set_sources([])

    churn_max = 0.0
    ww_count = 0
    for _ in range(6):
        solver.step_frame()
        churn_max = max(churn_max, float(solver.churn.data.numpy().max()))
        ww_count = max(ww_count, solver.whitewater.refresh_host_count())

    if churn_max <= 0.0:
        return False, "no churn accumulated"
    if ww_count == 0:
        return False, f"churn {churn_max:.4f} but no whitewater emitted"

    # Advection sanity: whitewater particles must remain inside the domain.
    pos = solver.whitewater_positions()
    if len(pos):
        in_dom = (
            (pos >= np.array(settings.domain_min) - 1.0e-3)
            & (pos <= np.array(settings.domain_max) + 1.0e-3)
        ).all()
    else:
        in_dom = True
    return bool(in_dom), (
        f"peak churn {churn_max:.4f}, peak whitewater {ww_count}, "
        f"all inside domain={bool(in_dom)}"
    )


# ---------------------------------------------------------------------------
# 15. benchmark profile schema compatibility
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[14])
def test_benchmark_schema(device):
    from ..diagnostics.benchmark import (
        BENCHMARK_FIELDS,
        compare_benchmark,
        run_benchmark,
    )
    from ..diagnostics.profiler import PROFILE_FRAME_FIELDS, Profiler
    from .solver import FluidSolver, SolverSettings

    # Tiny benchmark run (schema check, not performance).
    from ..diagnostics.benchmark import default_benchmark_settings

    settings = default_benchmark_settings()
    settings.voxel_size = 0.08
    settings.max_particles = 400_000
    settings.device = device
    result = run_benchmark(frames=4, emit_frames=2, settings=settings)
    data = result.as_dict()
    missing = [f for f in BENCHMARK_FIELDS if f not in data]
    if missing:
        return False, f"missing benchmark fields: {missing}"
    cmp = compare_benchmark(data, data)
    if cmp["result"] != "pass":
        return False, f"self-comparison failed: {cmp['result']}"

    # Frame stats schema.
    profiler = Profiler(device=device)
    settings2 = SolverSettings()
    settings2.domain_min = (-0.4, -0.4, 0.0)
    settings2.domain_max = (0.4, 0.4, 0.8)
    settings2.voxel_size = 0.08
    settings2.max_particles = 100_000
    settings2.device = device
    solver = FluidSolver(settings2, profiler=profiler)
    stats = solver.step_frame()
    frame_dict = stats.as_dict()
    missing2 = [f for f in PROFILE_FRAME_FIELDS if f not in frame_dict]
    if missing2:
        return False, f"missing frame fields: {missing2}"
    return True, (
        f"benchmark fields ok ({len(BENCHMARK_FIELDS)}), "
        f"frame fields ok ({len(PROFILE_FRAME_FIELDS)}), self-compare pass"
    )




# ---------------------------------------------------------------------------
# 16. spatiotemporal FLIP jitter/phase/projection smoke test
# ---------------------------------------------------------------------------


def _spacetime_probe_settings(device, jitter):
    """Settings for the ST-FLIP smoke test; ``jitter`` is the sample-time gamma."""
    from .solver import SolverSettings

    settings = SolverSettings()
    settings.domain_min = (-0.7, -0.4, 0.0)
    settings.domain_max = (0.7, 0.4, 0.8)
    settings.voxel_size = 0.08
    settings.leaf_capacity = 512
    settings.max_particles = 80_000
    settings.frame_rate = 24.0
    settings.time_scale = 1.0
    settings.min_substeps = 1
    settings.max_substeps = 1
    settings.cfl = 16.0
    settings.gravity = (0.0, 0.0, 0.0)
    settings.relax_enabled = False
    settings.surface_enabled = False
    settings.whitewater_enabled = False
    settings.pressure_iterations = 80
    settings.pressure_tolerance = 1.0e-5
    settings.spacetime_enabled = True
    settings.spacetime_jitter_strength = jitter
    settings.spacetime_adaptive_jitter = False
    settings.spacetime_variable_projection = True
    settings.spacetime_phase_eta = 0.5
    settings.spacetime_phase_threshold = 0.2
    settings.spacetime_local_cfl = 1.0
    settings.spacetime_max_local_substeps = 32
    # The beta probe below reads the post-frame face weights, which is after
    # Algorithm 1 line 21 has run.  At cfl = 16 the auto band is 8 rings, which
    # floods this small domain end to end and leaves no untouched face to read
    # a free-surface beta from.  Pin one ring: enough to exercise the pass,
    # few enough that empty faces survive.  Band growth itself is covered by
    # test_spacetime_velocity_extrapolation, which needs no solver.
    settings.spacetime_extrapolation_iterations = 1
    settings.device = device
    return settings


def _spacetime_probe_solver(settings, frames):
    """Seed a block of rightward-moving particles and run ``frames`` frames."""
    from .solver import FluidSolver

    solver = FluidSolver(settings)
    xs = np.linspace(-0.35, -0.05, 7, dtype=np.float32)
    ys = np.linspace(-0.18, 0.18, 5, dtype=np.float32)
    zs = np.linspace(0.18, 0.58, 6, dtype=np.float32)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    vels = np.tile(np.array((0.35, 0.0, 0.0), dtype=np.float32), (len(pts), 1))
    solver.inject_particles(pts, velocities=vels)
    for _ in range(frames):
        solver.step_frame()
    return solver


def _spacetime_max_residual(solver):
    """Largest |delta_t| carried by any live particle, or None if none survived."""
    n = solver.particles.refresh_host_count()
    if n <= 0:
        return None
    offsets = solver.particles.time_offset.numpy()[:n]
    return float(np.max(np.abs(offsets))) if len(offsets) else 0.0


@_test(VALIDATION_TESTS[15])
def test_spatiotemporal_flip(device):
    settings = _spacetime_probe_settings(device, 1.0)

    solver = _spacetime_probe_solver(settings, 4)
    ref_mass = float(solver._st_reference_mass())
    ref_expected = float(settings.particles_per_voxel)
    ref_ok = bool(
        np.isfinite(ref_mass)
        and ref_mass > 0.0
        and 0.5 * ref_expected <= ref_mass <= 1.5 * ref_expected
    )
    n = solver.particles.refresh_host_count()
    max_abs = _spacetime_max_residual(solver)
    if max_abs is None:
        return False, "all particles were lost"
    frame_dt = settings.time_scale / settings.frame_rate
    bound_ok = max_abs <= 0.5 * frame_dt + 2.0e-5

    # Lemma A.1's |delta_t| <= dt/2 holds only for gamma in [0, 1]: the residual
    # update is delta_t' = -gamma*xi*dt with xi in [-1/2, 1/2], so gamma = 2 would
    # reach |delta_t| = dt unless the solver clamps gamma at the launch site.
    # Headless callers set SolverSettings directly and bypass the property max,
    # so drive that path here.
    hi_settings = _spacetime_probe_settings(device, 2.0)
    hi_solver = _spacetime_probe_solver(hi_settings, 2)
    hi_abs = _spacetime_max_residual(hi_solver)
    clamp_ok = hi_abs is not None and hi_abs <= 0.5 * frame_dt + 2.0e-5

    pos = solver.particle_positions()
    finite_pos = bool(np.isfinite(pos).all())
    weights = solver.vel.wu.numpy()[: solver._nvox]
    phase = solver.density.data.numpy()[: solver._nvox]
    weights_ok = bool(np.isfinite(weights).all() and float(weights.max()) > 0.0)
    phase_ok = bool(
        np.isfinite(phase).all()
        and float(phase.max()) > 0.0
        and float(phase.min()) >= -1.0e-6
        and float(phase.max()) <= 1.0001
    )
    status = f"{solver.pressure_solver.last_pressure_status}"
    status_ok = bool(np.isfinite(solver.pressure_solver.last_residual))

    # Eq. (15): eps_rho is a floor, not a cap.  Reproduce st_face_beta on the
    # allocated u-faces; free-surface faces carry rho(phi_f) -> 0, so beta_f must
    # be free to reach ~1/eps_rho.  The old 0.05 capped it at 20 and stiffened
    # exactly the surface Eq. (17)'s rho_gas -> 0 limit exists to free.  The
    # projection must still converge under the stiffer operator.
    eps_rho = max(float(settings.spacetime_min_density), 1.0e-8)
    denom = max(ref_mass * max(float(settings.spacetime_phase_eta), 1.0e-6), 1.0e-8)
    phi_u = np.minimum(np.sqrt(np.maximum(weights, 0.0) / denom), 1.0)
    rho_u = (float(settings.spacetime_liquid_density) * phi_u
             + float(settings.spacetime_gas_density) * (1.0 - phi_u))
    # The bar is absolute, not relative to eps_rho: beta_max is 1/eps_rho for any
    # eps_rho, so a relative bound would pass at the old cap too.  5000 encodes
    # "eps_rho <= 2e-4 relative to rho_liquid = 1", four orders below liquid.
    beta_max = float((1.0 / np.maximum(rho_u, eps_rho)).max())
    beta_floor = 5000.0 / max(float(settings.spacetime_liquid_density), 1.0e-6)
    beta_ok = beta_max >= beta_floor and status == "converged"

    # Algorithm 1 line 21 actually ran in the substep loop.  The flood stamps
    # the 1e-6 extrapolated marker weight, a value P2G cannot produce, so its
    # presence in the post-frame face weights is the wiring check.  Kernel
    # behaviour itself is test_spacetime_velocity_extrapolation.
    extrap_faces = int(np.isclose(weights, 1.0e-6, rtol=0.0, atol=1.0e-12).sum())
    extrap_ok = extrap_faces > 0

    ok = (ref_ok and bound_ok and clamp_ok and finite_pos and weights_ok
          and phase_ok and status_ok and beta_ok and extrap_ok)
    return ok, (
        f"particles={n}, m0={ref_mass:.4f} (target ppc={ref_expected:.4f}), "
        f"max |delta_t|={max_abs:.6f} <= {0.5 * frame_dt:.6f}, "
        f"max |delta_t| at gamma=2 (clamped)="
        f"{'n/a' if hi_abs is None else format(hi_abs, '.6f')}, "
        f"max weight={float(weights.max()):.4f}, max phase={float(phase.max()):.4f}, "
        f"max free-surface beta={beta_max:.1f} >= {beta_floor:.1f} "
        f"(eps_rho={eps_rho:g}), "
        f"extrapolated faces={extrap_faces}, "
        f"pressure={status}"
    )


# ---------------------------------------------------------------------------
# 17. spatiotemporal velocity extrapolation
# ---------------------------------------------------------------------------


def _expand_flood(grid, device, seed_weight, iterations):
    """Seed one interior voxel and run the expand flood ``iterations`` times.

    Returns (values, weights) as numpy arrays over the whole voxel capacity.
    """
    cap = grid.voxel_capacity
    src = wp.zeros(cap, dtype=float, device=device)
    src_w = wp.zeros(cap, dtype=float, device=device)
    dst = wp.zeros(cap, dtype=float, device=device)
    dst_w = wp.zeros(cap, dtype=float, device=device)

    # Centre voxel of the 3x3x3-leaf box, addressed the way
    # voxel_coord_of_flat does: flat = slot*LEAF_VOLUME + (lz*8 + ly)*8 + lx.
    ld = grid.leaves_dim
    leaf_table = grid.leaf_table.numpy()
    lc = (1, 1, 1)
    slot = int(leaf_table[(lc[2] * int(ld[1]) + lc[1]) * int(ld[0]) + lc[0]])
    local = (4, 4, 4)
    flat = slot * LEAF_VOLUME + (local[2] * LEAF_DIM + local[1]) * LEAF_DIM + local[0]

    v_np = np.zeros(cap, dtype=np.float32)
    w_np = np.zeros(cap, dtype=np.float32)
    v_np[flat] = 3.0
    w_np[flat] = seed_weight
    wp.copy(src, wp.array(v_np, dtype=float, device=device))
    wp.copy(src_w, wp.array(w_np, dtype=float, device=device))

    for _ in range(iterations):
        wp.launch(
            KC.k_grid_expand_scalar, dim=cap,
            inputs=[src, src_w, ST_EPS_MASS, grid.leaf_table, grid.leaf_coords,
                    grid.voxels_dim, grid.leaves_dim, grid.num_leaves],
            outputs=[dst, dst_w], device=device,
        )
        src, dst = dst, src
        src_w, dst_w = dst_w, src_w
    return src.numpy(), src_w.numpy()


@_test(VALIDATION_TESTS[16])
def test_spacetime_velocity_extrapolation(device):
    # ST-FLIP Algorithm 1 line 21.  Kernel-level: no solver, no pressure solve.
    grid = _make_grid(device)
    band = 3
    counts = []
    values_ok = True
    for k in range(1, band + 1):
        vals, weights = _expand_flood(grid, device, 1.0, k)
        valid = weights > ST_EPS_MASS
        counts.append(int(valid.sum()))
        # A constant field must survive neighbour averaging exactly -- any
        # accumulator off-by-one shows up here as a value that is not 3.0.
        if not np.allclose(vals[valid], 3.0, atol=1.0e-6):
            values_ok = False
    # One 26-neighbour ring per iteration: the valid set is the Chebyshev ball
    # of radius k, so (2k+1)^3 voxels.  The seed sits at global voxel (12,12,12)
    # of a 24^3 domain, so the ball stays inside the allocated region.
    expected = [(2 * k + 1) ** 3 for k in range(1, band + 1)]
    growth_ok = counts == expected

    # Eq. (9): a face carrying m_f <= eps_m is under-sampled and its
    # u* = P_f/m_f is amplified noise.  Flooding that outwards is worse than
    # flooding nothing, so the seed must not count as valid at all.
    _, tiny_w = _expand_flood(grid, device, 1.0e-12, 2)
    gate_ok = int((tiny_w > ST_EPS_MASS).sum()) == 0

    ok = growth_ok and values_ok and gate_ok
    return ok, (
        f"band counts={counts} (expected {expected}), "
        f"constant field preserved={values_ok}, "
        f"eps_m gate rejects w=1e-12: {gate_ok} (eps_m={ST_EPS_MASS:g})"
    )


@_test(VALIDATION_TESTS[17])
def test_spacetime_solid_apertures(device):
    # ST-FLIP Eq. (14)/(15) alpha_f.  Kernel-level: no solver, no pressure solve.
    grid = _make_grid(device)
    dx = float(grid.voxel_size)
    centers = _voxel_centers(grid)
    nvox = grid.active_leaf_count * LEAF_VOLUME

    # Half-space collider filling z < z0, with the interface deliberately off
    # grid: 0.37 dx above a cell centre, so the two cells straddling it produce
    # an aperture strictly between 0 and 1 that a binary mask cannot represent.
    z_centers = np.unique(np.round(centers[:nvox, 2] / dx - 0.5).astype(np.int64))
    k_ref = int(z_centers[len(z_centers) // 2])
    z0 = float(grid.origin[2]) + (k_ref + 0.5) * dx + 0.37 * dx
    sdf_np = (centers[:, 2] - z0).astype(np.float32)

    sdf = wp.array(sdf_np, dtype=float, device=device)
    alpha_u = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
    alpha_v = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
    alpha_w = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
    wp.launch(
        P.k_compute_face_apertures, dim=grid.voxel_capacity,
        inputs=[sdf, grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                grid.leaves_dim, grid.num_leaves, 0.0],
        outputs=[alpha_u, alpha_v, alpha_w], device=device,
    )
    au = alpha_u.numpy()[:nvox]
    av = alpha_v.numpy()[:nvox]
    aw = alpha_w.numpy()[:nvox]

    finite = bool(np.isfinite(au).all() and np.isfinite(av).all()
                  and np.isfinite(aw).all())
    in_range = bool((au >= 0.0).all() and (au <= 1.0).all()
                    and (av >= 0.0).all() and (av <= 1.0).all()
                    and (aw >= 0.0).all() and (aw <= 1.0).all())

    # phi varies only along z, so the two cells sharing an x or y face read the
    # same value: those apertures must be exactly binary.  Anything in between
    # means the wrong pair of samples reached solid_face_fraction.
    lateral_binary = bool(
        np.isin(au, (0.0, 1.0)).all() and np.isin(av, (0.0, 1.0)).all())

    # Analytic expectation for the z faces.  b is this cell, a the -z neighbour
    # one voxel below; the bottom row's neighbour is outside the domain, which
    # the kernel treats as solid.
    b = centers[:nvox, 2] - z0
    a = b - dx
    k = np.round((centers[:nvox, 2] - float(grid.origin[2])) / dx - 0.5).astype(np.int64)
    expect = np.where(
        (a >= 0.0) & (b >= 0.0), 1.0,
        np.where((a < 0.0) & (b < 0.0), 0.0,
                 np.where(a >= 0.0, a / (a - b), b / (b - a))))
    expect = np.where(k == 0, 0.0, expect).astype(np.float32)
    exact_ok = bool(np.allclose(aw, expect, atol=1.0e-6))

    partial = aw[(aw > 1.0e-6) & (aw < 1.0 - 1.0e-6)]
    # b/(b-a) with b = 0.63 dx and b - a = dx.
    partial_ok = bool(len(partial) > 0
                      and np.allclose(partial, 0.63, atol=2.0e-3))

    ok = finite and in_range and lateral_binary and exact_ok and partial_ok
    return ok, (
        f"finite={finite}, in [0,1]={in_range}, lateral faces binary="
        f"{lateral_binary}, matches 1D fraction_inside={exact_ok}, "
        f"partial faces={len(partial)} at alpha={float(partial[0]) if len(partial) else float('nan'):.4f} "
        f"(expected 0.63 for a 0.37 dx offset)"
    )


@_test(VALIDATION_TESTS[18])
def test_spacetime_aperture_operator(device):
    """Apertures actually reaching Eq. (14)'s operator, RHS, and Eq. (16).

    Fixture is a closed box with a flat floor whose interface sits 0.37 dx above
    a cell centre, air on top so the system is non-singular, and a uniform
    downward velocity w = -1.  A uniform field is divergence free everywhere
    *except* across the floor, so the whole right side is where the wall is --
    which is exactly what alpha_f moves.  Binary masking blocks a full cell of
    flux; the aperture blocks the true 0.37, and the two rows straddling the
    interface then have closed-form divergences the test pins directly.
    """
    grid = _make_grid(device, size=0.8)
    dx = float(grid.voxel_size)
    nvox = grid.active_leaf_count * LEAF_VOLUME
    centers = _voxel_centers(grid)
    k = np.round((centers[:, 2] - float(grid.origin[2])) / dx - 0.5).astype(np.int64)

    k_solid = 7        # topmost solid row
    k_air = 13         # first air row
    z0 = float(grid.origin[2]) + (k_solid + 0.5) * dx + 0.37 * dx
    sdf_np = (centers[:, 2] - z0).astype(np.float32)

    vt_np = np.zeros(grid.voxel_capacity, dtype=np.int32)
    vt_np[(k >= 0) & (k < k_air)] = VOXEL_FLUID
    vt_np[(k >= 0) & (k <= k_solid)] |= VOXEL_COLLISION
    vt_np[nvox:] = 0
    voxel_type = wp.array(vt_np, dtype=wp.int32, device=device)
    voxel_counts = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=device)
    collision_vel = wp.zeros(grid.voxel_capacity, dtype=wp.vec3, device=device)

    sdf = wp.array(sdf_np, dtype=float, device=device)
    alpha = tuple(wp.zeros(grid.voxel_capacity, dtype=float, device=device)
                  for _ in range(3))
    wp.launch(
        P.k_compute_face_apertures, dim=grid.voxel_capacity,
        inputs=[sdf, grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                grid.leaves_dim, grid.num_leaves, 0.0],
        outputs=list(alpha), device=device,
    )

    st_args = dict(reference_mass=8.0, eta_phi=0.5, rho_liquid=1.0,
                   rho_gas=0.0, eps_rho=ST_EPS_RHO)

    def fresh_velocity():
        vel = MacVelocityField(grid)
        vel.zero()
        vel.w.fill_(-1.0)
        # Face mass == reference_mass puts phi at 1 everywhere, so beta is a
        # constant 1/rho_liquid and every effect below is purely alpha's.
        for arr in (vel.wu, vel.wv, vel.ww):
            arr.fill_(8.0)
        return vel

    ps = P.PressureSolver(grid, device)

    def run(apertures):
        vel = fresh_velocity()
        ps.build_preconditioner_st(voxel_type, vel, apertures=apertures, **st_args)
        inv_diag = ps.inv_diag.numpy()[:nvox].copy()
        ps.build_rhs(vel, voxel_type, collision_vel, voxel_counts,
                     apertures=apertures)
        div_before = ps.divergence.numpy()[:nvox].copy()
        ps.solve_pcg_st(voxel_type, 400, 1.0e-8, vel, apertures=apertures,
                        **st_args)
        residual, iters = ps.last_residual, ps.last_iterations
        ps.project_st(vel, voxel_type, collision_vel, apertures=apertures,
                      **st_args)
        ps.build_rhs(vel, voxel_type, collision_vel, voxel_counts,
                     apertures=apertures)
        div_after = float(np.abs(ps.divergence.numpy()[:nvox]).max())
        return inv_diag, div_before, div_after, residual, iters, vel

    inv_off, div_off, off_after, off_res, off_iters, _ = run(None)
    inv_on, div_on, on_after, on_res, on_iters, vel_on = run(alpha)
    off_before = float(np.abs(div_off).max())
    on_before = float(np.abs(div_on).max())

    ks = k[:nvox]
    solid = ks <= k_solid
    interface = ks == k_solid
    deep = (ks >= 0) & (ks < k_solid)

    # (1) aperture-off keeps the legacy unknown set exactly: no solid row is an
    # unknown, every fluid row is.
    legacy_ok = bool((inv_off[solid] == 0.0).all()
                     and (inv_off[(ks > k_solid) & (ks < k_air)] != 0.0).all()
                     and (inv_off[ks >= k_air] == 0.0).all())

    # (2) apertures on admit exactly the partially open solid row.
    unknown_ok = bool((inv_on[interface] != 0.0).all()
                      and (inv_on[deep] == 0.0).all()
                      and (inv_on[ks >= k_air] == 0.0).all())

    # (3) operator symmetry -- PCG is only valid if A stays symmetric once
    # alpha weights the off-diagonals.
    rng = np.random.default_rng(0)
    xs = wp.array(rng.standard_normal(grid.voxel_capacity).astype(np.float32),
                  dtype=float, device=device)
    ys = wp.array(rng.standard_normal(grid.voxel_capacity).astype(np.float32),
                  dtype=float, device=device)
    ax = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
    ay = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
    # Rebuild the face coefficients from vel_on explicitly rather than relying on
    # what run(alpha) happened to leave in ps: this is the operator under test.
    wp.launch(
        P.k_compute_face_coeffs_st, dim=grid.voxel_capacity,
        inputs=[voxel_type, vel_on.wu, vel_on.wv, vel_on.ww,
                grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                grid.leaves_dim, grid.num_leaves, 8.0, 0.5, 1.0, 0.0,
                ST_EPS_RHO, alpha[0], alpha[1], alpha[2], 1,
                ps.st_coeff_u, ps.st_coeff_v, ps.st_coeff_w],
        device=device,
    )
    for vec, out in ((xs, ax), (ys, ay)):
        wp.launch(
            P.k_apply_matrix_st, dim=grid.voxel_capacity,
            inputs=[vec, voxel_type, ps.inv_diag,
                    ps.st_coeff_u, ps.st_coeff_v, ps.st_coeff_w,
                    grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                    grid.leaves_dim, grid.num_leaves, 1, out],
            device=device,
        )
    xv, yv = xs.numpy()[:nvox], ys.numpy()[:nvox]
    x_ay = float(np.dot(xv, ay.numpy()[:nvox]))
    y_ax = float(np.dot(yv, ax.numpy()[:nvox]))
    scale = max(abs(x_ay), abs(y_ax), 1.0e-12)
    sym_err = abs(x_ay - y_ax) / scale
    sym_ok = bool(sym_err < 1.0e-4)

    # (4) alpha really reaches Eq. (14)'s right side.  Closed form for w = -1
    # with the interface 0.37 dx up from the k_solid centre, so alpha = 0.63 on
    # the face stored at k_solid + 1 and 0 on the one at k_solid:
    #   apertures on : row k_solid   div = (-0.63 -     0)/dx
    #                  row k_solid+1 div = (-1.00 - -0.63)/dx
    #   apertures off: row k_solid   not an unknown, div = 0
    #                  row k_solid+1 div = (-1.00 -     0)/dx   (a full cell of
    #                                flux blocked instead of 0.37 of one)
    a_face = 0.63
    first_fluid = ks == k_solid + 1
    rhs_ok = bool(
        np.allclose(div_on[interface], -a_face / dx, atol=1.0e-3)
        and np.allclose(div_on[first_fluid], -(1.0 - a_face) / dx, atol=1.0e-3)
        and np.allclose(div_off[interface], 0.0, atol=1.0e-6)
        and np.allclose(div_off[first_fluid], -1.0 / dx, atol=1.0e-3))
    solve_ok = bool(on_after < on_before * 0.01 or on_after < 1.0e-4)

    ok = legacy_ok and unknown_ok and sym_ok and rhs_ok and solve_ok
    return ok, (
        f"legacy unknown set unchanged={legacy_ok}, partial-solid row admitted="
        f"{unknown_ok}, symmetry error={sym_err:.2e}, "
        f"RHS matches closed form={rhs_ok} (aperture floor rows "
        f"{float(div_on[interface].mean()):.2f}/"
        f"{float(div_on[first_fluid].mean()):.2f} vs binary "
        f"{float(div_off[interface].mean()):.2f}/"
        f"{float(div_off[first_fluid].mean()):.2f}), "
        f"aperture max |div| {on_before:.3e} -> {on_after:.3e} "
        f"({on_iters} PCG iters, residual {on_res:.2e}), "
        f"aperture-off max |div| {off_before:.3e} -> {off_after:.3e} "
        f"({off_iters} iters, residual {off_res:.2e})"
    )


@_test(VALIDATION_TESTS[19])
def test_spacetime_surface_tension(device):
    """Curvature magnitude, force sign, and that a sphere does not blow up.

    Parts (1) and (2) run on an analytic phase field rather than a simulated
    one: a particle-derived phase field is noisy enough that a wrong stencil
    and a right one both land inside any tolerance loose enough to pass, which
    would make the check decorative.  A linear ramp through the interface has a
    known curvature 2/R and a known inward normal, so both the |grad phi|^-1
    normalisation and the sign are falsifiable to a few percent.  Part (3) then
    runs the real solver, which is the only thing that catches a force that is
    correct per-face but unstable in the loop.
    """
    grid = _make_grid(device, size=0.8)
    dx = float(grid.voxel_size)
    nvox = grid.active_leaf_count * LEAF_VOLUME
    centers = _voxel_centers(grid)[:nvox]
    origin = np.asarray(grid.origin, dtype=np.float64)
    center = origin + 0.4
    rel = centers.astype(np.float64) - center
    r = np.linalg.norm(rel, axis=1)

    radius = 5.0 * dx
    # Ramp half-width 2.5 dx, not the solver's 1.5 dx: the sampled band below
    # needs a one-cell margin on both sides so no stencil arm reaches a clamped
    # cell, and 1.5 dx leaves no such margin.
    eps = 2.5 * dx
    # phi = 1 inside, 0 outside, linear across a 2*eps band: |grad phi| =
    # 1/(2 eps) and n = grad phi/|grad phi| = -r_hat, so kappa = -div(n) = 2/r.
    phi_np = np.clip(0.5 - (r - radius) / (2.0 * eps), 0.0, 1.0)
    phi_full = np.zeros(grid.voxel_capacity, dtype=np.float32)
    phi_full[:nvox] = phi_np
    phase = wp.array(phi_full, dtype=float, device=device)
    kappa = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
    wp.launch(
        KF.k_phase_curvature, dim=grid.voxel_capacity,
        inputs=[phase, grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                grid.leaves_dim, grid.num_leaves, dx],
        outputs=[kappa], device=device,
    )
    kap = kappa.numpy()[:nvox]

    # Sample strictly inside the ramp so no stencil arm reaches a clamped cell.
    band = np.abs(r - radius) < (eps - dx)
    expected = 2.0 / radius
    kappa_med = float(np.median(kap[band]))
    kappa_err = abs(kappa_med - expected) / expected
    curvature_ok = bool(kappa_err < 0.25 and np.isfinite(kap).all())

    # Flat interior and the outside must stay at exactly zero -- that is the
    # |grad phi| floor doing its job rather than dividing noise by noise.
    flat = r < radius - eps - dx
    flat_ok = bool(flat.any() and np.all(kap[flat] == 0.0))

    vel = MacVelocityField(grid)
    vel.zero()
    for arr in (vel.wu, vel.wv, vel.ww):
        arr.fill_(1.0)
    # rho_gas == rho_liquid makes the density uniform, so the only thing left
    # in the acceleration is sigma*kappa*grad(phi) -- the sign under test.
    wp.launch(
        KF.k_surface_tension_csf, dim=grid.voxel_capacity,
        inputs=[phase, kappa, grid.leaf_table, grid.leaf_coords,
                grid.voxels_dim, grid.leaves_dim, grid.num_leaves, dx,
                1.0, 1.0, 1.0, ST_EPS_RHO, 1.0,
                vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww],
        device=device,
    )
    acc = np.stack([vel.u.numpy()[:nvox], vel.v.numpy()[:nvox],
                    vel.w.numpy()[:nvox]], axis=1).astype(np.float64)
    # The three faces stored at a voxel sit half a cell apart, but for a sign
    # test the cell-centred radial direction is more than accurate enough.
    rhat = rel / np.maximum(r, 1.0e-12)[:, None]
    radial = np.sum(acc * rhat, axis=1)[band]
    inward = float(radial.mean())
    sign_ok = bool(inward < 0.0 and (radial < 0.0).mean() > 0.9)

    # (3) a real sphere, zero gravity, nothing but surface tension acting.
    from .solver import FluidSolver, SolverSettings

    settings = SolverSettings()
    settings.domain_min = (-0.4, -0.4, 0.0)
    settings.domain_max = (0.4, 0.4, 0.8)
    settings.voxel_size = 0.05
    settings.leaf_capacity = 512
    settings.max_particles = 40_000
    settings.frame_rate = 24.0
    settings.min_substeps = 1
    settings.max_substeps = 16
    settings.cfl = 2.0
    settings.gravity = (0.0, 0.0, 0.0)
    settings.relax_enabled = False
    settings.surface_enabled = False
    settings.whitewater_enabled = False
    settings.pressure_iterations = 120
    settings.pressure_tolerance = 1.0e-5
    settings.spacetime_enabled = True
    settings.spacetime_adaptive_jitter = False
    settings.spacetime_extrapolation_iterations = 1
    settings.spacetime_surface_tension = 0.05
    settings.spacetime_surface_tension_smoothing = 2
    settings.device = device
    solver = FluidSolver(settings)

    r0 = 0.2
    step = 0.5 * settings.voxel_size
    axis = np.arange(-r0, r0 + 0.5 * step, step, dtype=np.float32)
    lat = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    lat = lat[np.linalg.norm(lat, axis=1) <= r0]
    pts = lat + np.array((0.0, 0.0, 0.4), dtype=np.float32)
    solver.inject_particles(pts)

    def gyration():
        n = solver.particles.refresh_host_count()
        if n <= 0:
            return None
        p = solver.particles.position.numpy()[:n].astype(np.float64)
        return float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))

    gyr0 = gyration()
    substeps = solver._compute_substeps(1.0 / settings.frame_rate)
    for _ in range(4):
        solver.step_frame()
    gyr1 = gyration()
    stable_ok = bool(gyr0 is not None and gyr1 is not None
                     and np.isfinite(gyr1) and gyr1 <= gyr0 * 1.05)

    ok = curvature_ok and flat_ok and sign_ok and stable_ok
    return ok, (
        f"kappa median={kappa_med:.4f} vs 2/R={expected:.4f} "
        f"(error {kappa_err * 100.0:.1f}%, {int(band.sum())} band cells), "
        f"flat interior zero={flat_ok}, mean radial acceleration="
        f"{inward:.4e} (inward fraction {float((radial < 0.0).mean()):.2f}), "
        f"gyration {gyr0:.4f} -> {gyr1:.4f} over 4 frames at sigma=0.05 "
        f"({substeps[0]} substeps, capillary clamp={substeps[1]})"
    )


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda:0", progress=None) -> list[ValidationResult]:
    results = []
    for fn in _TESTS:
        name = fn.validation_name
        t0 = time.perf_counter()
        try:
            passed, detail = fn(device)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
            passed, detail = False, f"exception: {exc!r}"
        elapsed = time.perf_counter() - t0
        result = ValidationResult(name, passed, detail, elapsed)
        results.append(result)
        if progress is not None:
            progress(result)
    return results
