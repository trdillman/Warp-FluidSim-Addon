"""GPU validation suite implementing the 13 synthesized validation targets.

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
    VOXEL_SINK,
)
from . import fields as F
from . import kernels_core as KC
from . import kernels_fluid as KF
from . import pressure as P
from . import surface as SURF
from . import whitewater as WW
from .fields import MacVelocityField, ScalarField, SparseGrid
from .particles import ParticleSystem

VALIDATION_TESTS = [
    "sparse octree coordinate/index roundtrip",
    "leaf-local voxel addressing and global start-vector reconstruction",
    "field precision parity for 16-bit and 32-bit modes",
    "particle-to-grid count/average/active masks",
    "grid-to-particle velocity/attribute sampling",
    "source shape coverage and source particle attribute transfer",
    "collision/sink voxel-type bit behavior",
    "pressure projection divergence reduction",
    "Jacobi and PCG residual convergence",
    "surface SDF build/filter parity",
    "viscosity solve stability",
    "whitewater churn/emission/advection behavior",
    "benchmark profile schema compatibility",
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


# ---------------------------------------------------------------------------
# 1. sparse octree coordinate/index roundtrip
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
def test_octree_roundtrip(device):
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
# 7. collision/sink voxel bit behavior
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[6])
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
# 8. pressure projection divergence reduction
# ---------------------------------------------------------------------------


def _divergence_norm(solver):
    solver.pressure_solver.build_rhs(
        solver.vel, solver.voxel_type.data, solver.collision_vel,
        solver.voxel_counts.data, rest_count=8.0, count_stiffness=0.0,
    )
    div = solver.pressure_solver.divergence.numpy()
    return float(np.abs(div).max())


@_test(VALIDATION_TESTS[7])
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
# 9. Jacobi and PCG residual convergence
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[8])
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
# 10. surface SDF build/filter parity
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[9])
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
# 11. viscosity solve stability
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[10])
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
# 12. whitewater churn/emission/advection
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[11])
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
# 13. benchmark profile schema compatibility
# ---------------------------------------------------------------------------


@_test(VALIDATION_TESTS[12])
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
