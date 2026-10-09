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
from . import surface_tension as ST
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
    "spatiotemporal FLIP jitter, phase, and residual boundedness",
    "surface-tension static droplet boundedness and volume proxy",
    "surface-tension droplet oscillation frequency trend",
    "surface-tension zero-coefficient solver equivalence",
    "surface-tension capillary timestep constraint",
    "surface-tension finite-energy and diagnostic guard",
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


# 1. sparse tiled leaf-grid coordinate/index roundtrip


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


# 2. leaf-local addressing / start vector reconstruction


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


# 3. field precision parity 16/32-bit


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


# 4. particle-to-grid count/average/active masks


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


# 5. grid-to-particle sampling


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


# 6. source shape coverage + particle attribute transfer


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
    try:
        solver.set_source_batch([src])
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
    finally:
        solver.release()


# 7. height-field sourcing flat/profile behavior


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
    # Validation stamps the height field without particle emission, so the
    # transaction admission guard is explicitly disabled.
    emission_state = wp.zeros(4, dtype=wp.int32, device=device)

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
            wp.vec3(1.0, 2.0, 3.0), 0, emission_state,
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
            wp.vec3(-1.0, 0.5, 4.0), 0, emission_state,
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


# 8. NanoVDB/OpenVDB volume sourcing kernels


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


# 9. collision/sink voxel bit behavior


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
    try:
        solver.set_source_batch([box, sink])
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
    finally:
        solver.release()


# 10. pressure projection divergence reduction


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
    try:
        solver.set_source_batch([src])
        solver.step_frame()
        solver.set_source_batch()
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
        result = reduced, (
            f"max |divergence| {div_before:.3e} -> {div_after:.3e} "
            f"({solver.pressure_solver.last_iterations} PCG iterations)"
        )
    finally:
        solver.release()
    return result


# 11. Jacobi and PCG residual convergence


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
    try:
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
        result = ok, (
            f"|rhs|={rhs_norm:.3e}, Jacobi residual {res_jacobi:.3e}, "
            f"PCG residual {res_pcg:.3e} ({ps.last_iterations} iters)"
        )
    finally:
        ps.release()
    return result


# 12. surface SDF build/filter parity


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
                0.5 * grid.voxel_size * 1.2, 2, 4.0 * grid.voxel_size],
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


# 13. viscosity solve stability


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
    try:
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
    finally:
        vsolver.release()


# 14. whitewater churn/emission/advection


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
    try:
        solver.set_source_batch([src])
        for _ in range(3):
            solver.step_frame()
        solver.set_source_batch()

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
    finally:
        solver.release()


# 15. spatiotemporal FLIP jitter/phase/projection smoke test


@_test(VALIDATION_TESTS[14])
def test_spatiotemporal_flip(device):
    from .solver import FluidSolver, SolverSettings

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
    settings.spacetime_jitter_strength = 1.0
    settings.spacetime_adaptive_jitter = False
    settings.spacetime_variable_projection = True
    settings.spacetime_phase_eta = 0.5
    settings.spacetime_phase_threshold = 0.2
    settings.spacetime_min_density = 0.05
    settings.spacetime_local_cfl = 1.0
    settings.spacetime_max_local_substeps = 32
    settings.device = device

    solver = FluidSolver(settings)
    try:
        ref_mass = float(solver.spacetime.reference_mass())
        ref_expected = float(settings.particles_per_voxel)
        ref_ok = bool(
            np.isfinite(ref_mass)
            and ref_mass > 0.0
            and 0.5 * ref_expected <= ref_mass <= 1.5 * ref_expected
        )
        xs = np.linspace(-0.35, -0.05, 7, dtype=np.float32)
        ys = np.linspace(-0.18, 0.18, 5, dtype=np.float32)
        zs = np.linspace(0.18, 0.58, 6, dtype=np.float32)
        pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        vels = np.tile(np.array((0.35, 0.0, 0.0), dtype=np.float32), (len(pts), 1))
        solver.inject_particles(pts, velocities=vels)

        for _ in range(4):
            solver.step_frame()

        n = solver.particles.refresh_host_count()
        if n <= 0:
            return False, "all particles were lost"
        offsets = solver.particles.time_offset.numpy()[:n]
        frame_dt = settings.time_scale / settings.frame_rate
        max_abs = float(np.max(np.abs(offsets))) if len(offsets) else 0.0
        bound_ok = max_abs <= 0.5 * frame_dt + 2.0e-5
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
        status_ok = bool(np.isfinite(solver.pressure_solver.last_residual))
        ok = ref_ok and bound_ok and finite_pos and weights_ok and phase_ok and status_ok
        return ok, (
            f"particles={n}, m0={ref_mass:.4f} (target ppc={ref_expected:.4f}), "
            f"max |delta_t|={max_abs:.6f} <= {0.5 * frame_dt:.6f}, "
            f"max weight={float(weights.max()):.4f}, max phase={float(phase.max()):.4f}, "
            f"pressure={solver.pressure_solver.last_pressure_status}"
        )
    finally:
        solver.release()


# 17-21. surface-tension physics and capillary stability


def _surface_tension_settings(device, *, coefficient=0.072, enabled=True):
    from .solver import SolverSettings

    settings = SolverSettings()
    settings.domain_min = (-0.5, -0.5, 0.0)
    settings.domain_max = (0.5, 0.5, 1.0)
    settings.voxel_size = 0.1
    settings.leaf_capacity = 64
    settings.max_particles = 20_000
    settings.auto_grow_particles = False
    settings.frame_rate = 24.0
    settings.time_scale = 1.0
    settings.min_substeps = 1
    settings.max_substeps = 8
    settings.cfl = 0.8
    settings.gravity = (0.0, 0.0, 0.0)
    settings.relax_enabled = False
    settings.surface_enabled = False
    settings.surface_filter_iterations = 0
    settings.surface_dilate_iterations = 0
    settings.surface_erode_iterations = 0
    settings.surface_smooth_iterations = 0
    settings.surface_tension_enabled = bool(enabled)
    settings.surface_tension_coefficient = float(coefficient)
    settings.surface_tension_density = 1000.0
    settings.surface_tension_band_voxels = 1.5
    settings.surface_tension_safety = 0.5
    settings.surface_tension_smoothing_iterations = 1
    settings.whitewater_enabled = False
    settings.pressure_iterations = 120
    settings.pressure_tolerance = 1.0e-5
    settings.device = device
    return settings


def _droplet_points(center=(0.0, 0.0, 0.5), radius=0.22, spacing=0.07):
    center = np.asarray(center, dtype=np.float32)
    axis = np.arange(-radius, radius + spacing * 0.5, spacing, dtype=np.float32)
    offsets = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = offsets[np.linalg.norm(offsets, axis=1) <= radius]
    return np.ascontiguousarray(offsets + center, dtype=np.float32)


def _run_droplet(device, *, coefficient=0.072, enabled=True, frames=3):
    from .solver import FluidSolver

    settings = _surface_tension_settings(
        device, coefficient=coefficient, enabled=enabled)
    solver = FluidSolver(settings)
    try:
        points = _droplet_points()
        solver.inject_particles(points, velocity=(0.0, 0.0, 0.0))
        initial = solver.particle_positions().copy()
        stats = None
        for _ in range(int(frames)):
            stats = solver.step_frame()
        positions = solver.particle_positions()
        velocities = solver.particle_velocities()
        return solver, stats, initial, positions, velocities
    except Exception:
        solver.release()
        raise


@_test(VALIDATION_TESTS[15])
def test_surface_tension_static_droplet(device):
    solver, stats, initial, positions, velocities = _run_droplet(device, frames=4)
    try:
        if len(positions) != len(initial) or len(positions) == 0:
            return False, f"particle count changed: {len(initial)} -> {len(positions)}"
        center_initial = initial.mean(axis=0)
        center_final = positions.mean(axis=0)
        radius_initial = float(np.mean(np.linalg.norm(initial - center_initial, axis=1)))
        radius_final = float(np.mean(np.linalg.norm(positions - center_final, axis=1)))
        drift = abs(radius_final - radius_initial) / max(radius_initial, 1.0e-9)
        max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
        active = int(getattr(stats, "surface_tension_active_voxels", 0))
        ok = finite and drift < 0.12 and max_speed < 5.0 and active > 0
        return ok, (
            f"particles={len(positions)}, radius drift={drift:.5f}, "
            f"max speed={max_speed:.5f}, active force voxels={active}"
        )
    finally:
        solver.release()


@_test(VALIDATION_TESTS[16])
def test_surface_tension_oscillation_trend(_device):
    baseline = ST.rayleigh_lamb_frequency(1000.0, 0.1, 0.072)
    stronger = ST.rayleigh_lamb_frequency(1000.0, 0.1, 0.144)
    smaller = ST.rayleigh_lamb_frequency(1000.0, 0.05, 0.072)
    expected_ratio = np.sqrt(2.0)
    ratio = stronger / baseline
    ok = (
        stronger > baseline
        and smaller > baseline
        and abs(ratio - expected_ratio) < 1.0e-12
    )
    return ok, (
        f"omega={baseline:.6f}, strong/base={ratio:.6f}, "
        f"small/base={smaller / baseline:.6f}"
    )


@_test(VALIDATION_TESTS[17])
def test_surface_tension_zero_equivalence(device):
    disabled = _run_droplet(
        device, coefficient=0.0, enabled=False, frames=2)
    zero = None
    try:
        zero = _run_droplet(
            device, coefficient=0.0, enabled=True, frames=2)
        disabled_positions = disabled[3]
        zero_positions = zero[3]
        disabled_velocities = disabled[4]
        zero_velocities = zero[4]
        if len(disabled_positions) != len(zero_positions):
            return False, (
                "particle count differs: "
                f"{len(disabled_positions)} != {len(zero_positions)}"
            )
        # GPU compaction is free to return the same particles in a different
        # order.  Equivalence here is the particle state, not atomic ordering.
        disabled_order = np.lexsort(tuple(disabled_positions[:, axis]
                                          for axis in (2, 1, 0)))
        zero_order = np.lexsort(tuple(zero_positions[:, axis]
                                      for axis in (2, 1, 0)))
        position_error = float(np.max(np.abs(
            disabled_positions[disabled_order] - zero_positions[zero_order]
        )))
        velocity_error = float(np.max(np.abs(
            disabled_velocities[disabled_order] - zero_velocities[zero_order]
        )))
        ok = position_error <= 1.0e-7 and velocity_error <= 1.0e-7
        return ok, (
            f"max position error={position_error:.3e}, "
            f"max velocity error={velocity_error:.3e}"
        )
    finally:
        disabled[0].release()
        if zero is not None:
            zero[0].release()


@_test(VALIDATION_TESTS[18])
def test_surface_tension_capillary_timestep(_device):
    coarse = ST.capillary_dt(1000.0, 0.1, 0.072, 0.5)
    fine = ST.capillary_dt(1000.0, 0.05, 0.072, 0.5)
    strong = ST.capillary_dt(1000.0, 0.1, 0.144, 0.5)
    disabled = ST.capillary_dt(1000.0, 0.1, 0.0, 0.5)
    frame_dt = 1.0 / 24.0
    required = ST.required_capillary_substeps(frame_dt, fine)
    ok = fine < coarse and strong < coarse and np.isinf(disabled) and required >= 1
    return ok, (
        f"coarse={coarse:.6g}s, fine={fine:.6g}s, "
        f"strong={strong:.6g}s, fine substeps={required}"
    )


@_test(VALIDATION_TESTS[19])
def test_surface_tension_finite_energy(device):
    solver, stats, _initial, positions, velocities = _run_droplet(
        device, coefficient=0.25, enabled=True, frames=5)
    try:
        energy = 0.5 * float(np.sum(np.square(velocities, dtype=np.float64)))
        finite = bool(
            np.isfinite(positions).all()
            and np.isfinite(velocities).all()
            and np.isfinite(energy)
        )
        max_acceleration = float(
            getattr(stats, "surface_tension_max_acceleration", 0.0))
        curvature = float(getattr(stats, "surface_tension_max_curvature", 0.0))
        curvature_limit = 2.0 / solver.settings.voxel_size
        ok = (
            finite
            and energy < 10_000.0
            and max_acceleration > 0.0
            and 0.0 < curvature <= curvature_limit + 1.0e-4
            and not bool(getattr(stats, "capillary_substeps_clamped", False))
        )
        return ok, (
            f"energy proxy={energy:.6g}, max acceleration={max_acceleration:.6g}, "
            f"max |curvature|={curvature:.6g}/{curvature_limit:.6g}, "
            f"limiter={getattr(stats, 'substep_limiter', 'unknown')}"
        )
    finally:
        solver.release()


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
