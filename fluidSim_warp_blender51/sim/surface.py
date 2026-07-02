"""Liquid surface SDF construction, smoothing and SDF->collision conversion.

Ports of the fluid_surface_collision kernel group:
  fluidSimBuildSurfaceSdf, fluidSimSmoothSurfaceSdfDialate,
  fluidSimSmoothSurfaceSdfErode, fluidSimSmoothSurfaceSdfBox,
  fluidSimSmoothSurfaceSdfMeanCurvatureFlow, fluidSimSdfToAxisCollision,
  fluidSimExpandAndShiftAxisVelocityInColliderAxis
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .fields import (
    leaf_linear_index,
    leaf_coord_of_voxel,
    leaf_local_of_voxel,
    voxel_in_domain,
    voxel_flat_index,
    voxel_coord_of_flat,
    voxel_center_world,
    world_to_voxel,
    field_read_f,
    field_read_i,
)
from .registry import register_fn


# ---------------------------------------------------------------------------
# Surface SDF build (union of particle spheres, splatted into the grid)
# ---------------------------------------------------------------------------


@wp.kernel
def k_surface_sdf_reset(
    sdf: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    default_value: float,
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    sdf[tid] = default_value


@wp.kernel
def k_build_surface_sdf(
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
    """Union-of-spheres distance splat. sdf holds distance-to-surface
    (negative inside), quantized to int via atomic_min on scaled values.

    We store float distances directly using atomic_min emulation:
    warp supports wp.atomic_min on float arrays, so use it directly.
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


register_fn(k_build_surface_sdf, "fluidSimBuildSurfaceSdf",
            notes="Union-of-spheres particle splat with atomic min distance.")


# ---------------------------------------------------------------------------
# Surface smoothing filters
# ---------------------------------------------------------------------------


@wp.func
def _sdf_read(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    fallback: float,
):
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return fallback
    return src[flat]


@wp.kernel
def k_smooth_surface_sdf_dilate(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Morphological dilate of the liquid region: take min over 6-neighborhood."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = src[tid]
    m = center
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = wp.vec3i(ijk[0], ijk[1], ijk[2])
            n[axis] = n[axis] + step
            m = wp.min(m, _sdf_read(src, leaf_table, voxels_dim, leaves_dim, n, center))
    dst[tid] = center + (m - center) * wp.clamp(strength, 0.0, 1.0)


register_fn(k_smooth_surface_sdf_dilate, "fluidSimSmoothSurfaceSdfDialate",
            notes="Min-filter dilation; source name keeps original spelling.")


@wp.kernel
def k_smooth_surface_sdf_erode(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Morphological erode of the liquid region: take max over 6-neighborhood."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = src[tid]
    m = center
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = wp.vec3i(ijk[0], ijk[1], ijk[2])
            n[axis] = n[axis] + step
            m = wp.max(m, _sdf_read(src, leaf_table, voxels_dim, leaves_dim, n, center))
    dst[tid] = center + (m - center) * wp.clamp(strength, 0.0, 1.0)


register_fn(k_smooth_surface_sdf_erode, "fluidSimSmoothSurfaceSdfErode",
            notes="Max-filter erosion.")


@wp.kernel
def k_smooth_surface_sdf_box(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Box (mean) filter over the 7-point stencil, blended by strength."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = src[tid]
    acc = center
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = wp.vec3i(ijk[0], ijk[1], ijk[2])
            n[axis] = n[axis] + step
            acc += _sdf_read(src, leaf_table, voxels_dim, leaves_dim, n, center)
    mean = acc / 7.0
    dst[tid] = center + (mean - center) * wp.clamp(strength, 0.0, 1.0)


register_fn(k_smooth_surface_sdf_box, "fluidSimSmoothSurfaceSdfBox",
            notes="7-point box smoothing.")


@wp.kernel
def k_smooth_surface_sdf_mean_curvature(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Mean curvature flow smoothing step: phi += dt * kappa * |grad phi|.

    kappa*|grad| is approximated with the standard second-order stencil.
    Only voxels near the interface (|phi| < 3 voxels) are updated.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = src[tid]
    h = voxel_size
    if wp.abs(c) > 3.0 * h:
        dst[tid] = c
        return
    xp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2]), c)
    xm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1], ijk[2]), c)
    yp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2]), c)
    ym = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] - 1, ijk[2]), c)
    zp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1], ijk[2] + 1), c)
    zm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1], ijk[2] - 1), c)

    gx = (xp - xm) / (2.0 * h)
    gy = (yp - ym) / (2.0 * h)
    gz = (zp - zm) / (2.0 * h)
    gxx = (xp - 2.0 * c + xm) / (h * h)
    gyy = (yp - 2.0 * c + ym) / (h * h)
    gzz = (zp - 2.0 * c + zm) / (h * h)

    # Mixed derivatives.
    xpyp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1] + 1, ijk[2]), c)
    xpym = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1] - 1, ijk[2]), c)
    xmyp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1] + 1, ijk[2]), c)
    xmym = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1] - 1, ijk[2]), c)
    xpzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2] + 1), c)
    xpzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2] - 1), c)
    xmzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1], ijk[2] + 1), c)
    xmzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1], ijk[2] - 1), c)
    ypzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2] + 1), c)
    ypzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2] - 1), c)
    ymzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] - 1, ijk[2] + 1), c)
    ymzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] - 1, ijk[2] - 1), c)

    gxy = (xpyp - xpym - xmyp + xmym) / (4.0 * h * h)
    gxz = (xpzp - xpzm - xmzp + xmzm) / (4.0 * h * h)
    gyz = (ypzp - ypzm - ymzp + ymzm) / (4.0 * h * h)

    g2 = gx * gx + gy * gy + gz * gz
    if g2 < 1.0e-12:
        dst[tid] = c
        return

    # kappa |grad phi| = (gxx(gy^2+gz^2) + gyy(gx^2+gz^2) + gzz(gx^2+gy^2)
    #                     - 2(gx gy gxy + gx gz gxz + gy gz gyz)) / g2
    num = (
        gxx * (gy * gy + gz * gz)
        + gyy * (gx * gx + gz * gz)
        + gzz * (gx * gx + gy * gy)
        - 2.0 * (gx * gy * gxy + gx * gz * gxz + gy * gz * gyz)
    )
    kappa_grad = num / g2
    # Stable explicit step: dt <= h^2/6, scaled by user strength.
    dt = wp.clamp(strength, 0.0, 1.0) * (h * h) / 6.0
    dst[tid] = c + dt * kappa_grad


register_fn(k_smooth_surface_sdf_mean_curvature, "fluidSimSmoothSurfaceSdfMeanCurvatureFlow",
            notes="Explicit mean curvature flow with stable step size.")


# ---------------------------------------------------------------------------
# SDF -> axis collision classification (per-face solid fractions/bits)
# ---------------------------------------------------------------------------


@wp.kernel
def k_sdf_to_axis_collision(
    sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    threshold: float,
    voxel_type: wp.array(dtype=wp.int32),
):
    """Classify collision voxels + per-axis face collision bits from an SDF.

    A voxel is solid when sdf < threshold. The face bits CollisionX/Y/Z mark
    faces whose lower-axis neighbor OR the voxel itself is solid; these are
    consumed by pressure projection and velocity masking.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = sdf[tid]

    vt = voxel_type[tid]
    vt &= ~(C.VOXEL_COLLISION | C.VOXEL_COLLISION_X | C.VOXEL_COLLISION_Y | C.VOXEL_COLLISION_Z)

    solid_c = c < threshold
    if solid_c:
        vt |= C.VOXEL_COLLISION
        vt &= ~C.VOXEL_FLUID

    for axis in range(3):
        n = wp.vec3i(ijk[0], ijk[1], ijk[2])
        n[axis] = n[axis] - 1
        flat_n = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
        solid_n = True
        if flat_n >= 0:
            solid_n = sdf[flat_n] < threshold
        elif voxel_in_domain(n, voxels_dim):
            solid_n = False
        if solid_c or solid_n:
            if axis == 0:
                vt |= C.VOXEL_COLLISION_X
            elif axis == 1:
                vt |= C.VOXEL_COLLISION_Y
            else:
                vt |= C.VOXEL_COLLISION_Z
    voxel_type[tid] = vt


register_fn(k_sdf_to_axis_collision, "fluidSimSdfToAxisCollision",
            notes="Voxel + per-axis-face collision bits from collision SDF.")


@wp.kernel
def k_expand_shift_axis_velocity_in_collider(
    u_in: wp.array(dtype=float),
    v_in: wp.array(dtype=float),
    w_in: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    collision_vel: wp.array(dtype=wp.vec3),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    u_out: wp.array(dtype=float),
    v_out: wp.array(dtype=float),
    w_out: wp.array(dtype=float),
):
    """Replace face velocities on collision faces with the collider velocity
    and extrapolate fluid velocity one ring into the collider so that
    interpolation near walls sees sane values.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    vt = voxel_type[tid]

    cu = u_in[tid]
    cv = v_in[tid]
    cw = w_in[tid]

    if (vt & C.VOXEL_COLLISION_X) != 0:
        cu = collision_vel[tid][0]
    if (vt & C.VOXEL_COLLISION_Y) != 0:
        cv = collision_vel[tid][1]
    if (vt & C.VOXEL_COLLISION_Z) != 0:
        cw = collision_vel[tid][2]

    if (vt & C.VOXEL_COLLISION) != 0:
        # Average neighboring non-collision voxel velocities into the collider
        # cell (simple one-ring extrapolation), keeping the normal component
        # from the collider so separation is preserved.
        acc = wp.vec3(0.0)
        n_count = float(0.0)
        for axis in range(3):
            for step in range(-1, 2, 2):
                n = wp.vec3i(ijk[0], ijk[1], ijk[2])
                n[axis] = n[axis] + step
                flat_n = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
                if flat_n < 0:
                    continue
                if (voxel_type[flat_n] & C.VOXEL_COLLISION) != 0:
                    continue
                acc += wp.vec3(u_in[flat_n], v_in[flat_n], w_in[flat_n])
                n_count += 1.0
        if n_count > 0.0:
            avg = acc / n_count
            if (vt & C.VOXEL_COLLISION_X) == 0:
                cu = avg[0]
            if (vt & C.VOXEL_COLLISION_Y) == 0:
                cv = avg[1]
            if (vt & C.VOXEL_COLLISION_Z) == 0:
                cw = avg[2]

    u_out[tid] = cu
    v_out[tid] = cv
    w_out[tid] = cw


register_fn(k_expand_shift_axis_velocity_in_collider,
            "fluidSimExpandAndShiftAxisVelocityInColliderAxis",
            notes="Collider face velocity override + one-ring extrapolation.")


# ---------------------------------------------------------------------------
# Marching cubes surface extraction (host orchestrated, GPU compute)
# ---------------------------------------------------------------------------


class SurfaceExtractor:
    """Extracts a triangle mesh from the liquid surface SDF using
    warp's built-in marching cubes on a dense sub-grid covering the
    allocated leaves. Dense extraction is bounded by max_dense_voxels."""

    def __init__(self, grid, device, max_dense_dim=256):
        self.grid = grid
        self.device = device
        self.max_dense_dim = int(max_dense_dim)
        self._mc = None
        self._dense = None
        self._dense_dim = None

    def _ensure_dense(self, dims):
        nx, ny, nz = dims
        if self._dense is not None and self._dense_dim == (nx, ny, nz):
            return
        self._dense = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self._dense_dim = (nx, ny, nz)
        self._mc = wp.MarchingCubes(
            nx=nx, ny=ny, nz=nz,
            max_verts=nx * ny * nz,
            max_tris=2 * nx * ny * nz,
            device=self.device,
        )

    def extract(self, sdf_field, iso=0.0):
        g = self.grid
        nx = min(int(g.voxels_dim[0]), self.max_dense_dim)
        ny = min(int(g.voxels_dim[1]), self.max_dense_dim)
        nz = min(int(g.voxels_dim[2]), self.max_dense_dim)
        self._ensure_dense((nx, ny, nz))
        wp.launch(
            k_sparse_to_dense_sdf,
            dim=nx * ny * nz,
            inputs=[
                sdf_field.data, g.leaf_table, g.voxels_dim, g.leaves_dim,
                wp.vec3i(nx, ny, nz), 4.0 * g.voxel_size,
            ],
            outputs=[self._dense],
            device=self.device,
        )
        self._mc.surface(self._dense, iso)
        verts = self._mc.verts.numpy()
        tris = self._mc.indices.numpy().reshape(-1, 3)
        # Vertices are in dense-voxel index space; convert to world.
        origin = [float(g.origin[0]), float(g.origin[1]), float(g.origin[2])]
        vs = g.voxel_size
        verts = verts * vs + [origin[0] + 0.5 * vs, origin[1] + 0.5 * vs, origin[2] + 0.5 * vs]
        return verts, tris


@wp.kernel
def k_sparse_to_dense_sdf(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    dense_dim: wp.vec3i,
    background: float,
    dense: wp.array3d(dtype=float),
):
    tid = wp.tid()
    nx = dense_dim[0]
    ny = dense_dim[1]
    nz = dense_dim[2]
    if tid >= nx * ny * nz:
        return
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    ijk = wp.vec3i(i, j, k)
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    val = background
    if flat >= 0:
        val = src[flat]
    dense[i, j, k] = val
