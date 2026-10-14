"""Core buffer/field/SDF kernels (``core*`` family).

Implements the core kernel groups from the manifest:
- core_buffer_field_ops: clear / sync / reduce / sort(gather) / convert /
  clamp / remap
- core_misc: max-active-value, scale-field reset/ramp
- core_sdf_collision: SDF smoothing, fast-sweep renormalize, gradient,
  collision bits, axis SDF creation
- core_particle_grid_transfer: generic particle<->grid attribute transfer

Flat Warp kernels replace simple NDRange OpenCL kernels; reductions use
Warp's device-side primitives instead of workgroup-local memory trees.
No bpy imports allowed here.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import (
    LEAF_VOLUME,
    VOXEL_COLLISION,
)
from .fields import (
    field_read_f,
    leaf_linear_index,
    sample_scalar_trilinear,
    voxel_coord_of_flat,
    voxel_flat_index,
    voxel_in_domain,
    world_to_voxel,
    voxel_floor,
)
from .registry import register, register_fn

# Clear kernels (coreClearScalar*/Vector*)


def _make_clear_kernel(dtype):
    def _clear(data, value):
        data[wp.tid()] = value

    # explicit annotation objects (avoids lazy-eval closure issues on 3.14)
    _clear.__annotations__ = {"data": wp.array(dtype=dtype), "value": dtype}
    _clear.__name__ = f"k_clear_{getattr(dtype, '__name__', str(dtype))}"
    return wp.kernel(_clear)


k_clear_i32 = _make_clear_kernel(wp.int32)
k_clear_i16 = _make_clear_kernel(wp.int16)
k_clear_i8 = _make_clear_kernel(wp.int8)
k_clear_f32 = _make_clear_kernel(wp.float32)
k_clear_f16 = _make_clear_kernel(wp.float16)
k_clear_v3 = _make_clear_kernel(wp.vec3)
k_clear_c2 = _make_clear_kernel(wp.vec2i)  # "couple" = int pair

register_fn(k_clear_i32, "coreClearScalarInt32", "coreClearScalarVoxelType")
register_fn(k_clear_c2, "coreClearCoupleInt32")
register_fn(k_clear_i16, "coreClearScalarInt16")
register_fn(k_clear_i8, "coreClearScalarInt8")
register_fn(k_clear_f32, "coreClearScalarFloat32", "coreClearScalarField")
register_fn(k_clear_v3, "coreClearVectorField")


# Sync kernels (device buffer copies; the original synchronized host/device
# mirrors — under Warp the canonical implementation is a device-device copy)


def sync_copy(dst: wp.array, src: wp.array, count: int | None = None):
    wp.copy(dst, src, count=count)


for _n in (
    "coreSyncScalarInt32", "coreSyncScalarInt16", "coreSyncScalarVoxelType",
    "coreSyncScalarFloat32", "coreSyncScalarFloat16", "coreSyncScalarField",
    "coreSyncCoupleInt32", "coreSyncVectorFloat32", "coreSyncCoupleField",
    "coreSyncVectorField",
):
    register_fn(sync_copy, _n, strategy="host_orchestrated",
                notes="device-resident copy; host mirrors eliminated")


# Reductions (coreReduceSum*, coreReduce*DotPartialLinear*)
# Warp device primitives replace workgroup-size-specialized OpenCL trees.


def reduce_sum(src: wp.array, out: wp.array):
    wp.utils.array_sum(src, out=out)


def reduce_dot(a: wp.array, b: wp.array, out: wp.array):
    wp.utils.array_inner(a, b, out=out)


for _sz in (128, 256, 512, 1024):
    register_fn(reduce_sum, f"coreReduceSum{_sz}", strategy="tiled",
                notes="workgroup size variants collapse to warp array_sum")
    register_fn(reduce_dot, f"coreReduceScalarFieldDotPartialLinear{_sz}",
                strategy="tiled")
    register_fn(reduce_dot, f"coreReduceVectorFieldDotPartialLinear{_sz}",
                strategy="tiled")


# Conversions / clamp / remap


@wp.kernel
def k_convert_f32_to_f16(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float16)):
    tid = wp.tid()
    dst[tid] = wp.float16(src[tid])


@wp.kernel
def k_convert_f16_to_f32(src: wp.array(dtype=wp.float16), dst: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    dst[tid] = wp.float32(src[tid])


@wp.kernel
def k_convert_i32_to_i16(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int16)):
    tid = wp.tid()
    dst[tid] = wp.int16(src[tid])


@wp.kernel
def k_convert_i16_to_i32(src: wp.array(dtype=wp.int16), dst: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    dst[tid] = wp.int32(src[tid])


register_fn(k_convert_f32_to_f16, "coreConvertFloat32ToFloat16")
register_fn(k_convert_f16_to_f32, "coreConvertFloat16ToFloat32")
register_fn(k_convert_i32_to_i16, "coreConvertInt32ToInt16")
register_fn(k_convert_i16_to_i32, "coreConvertInt16ToInt32")


@wp.kernel
def k_clamp_scalar_field(data: wp.array(dtype=float), lo: float, hi: float):
    tid = wp.tid()
    data[tid] = wp.clamp(data[tid], lo, hi)


@wp.kernel
def k_remap_scalar_field(
    data: wp.array(dtype=float),
    in_lo: float, in_hi: float, out_lo: float, out_hi: float,
):
    tid = wp.tid()
    t = (data[tid] - in_lo) / wp.max(1.0e-20, in_hi - in_lo)
    data[tid] = out_lo + wp.clamp(t, 0.0, 1.0) * (out_hi - out_lo)


register_fn(k_clamp_scalar_field, "coreClampScalarField")
register_fn(k_remap_scalar_field, "coreRemapScalarField")


# Vector magnitude conversions


@wp.kernel
def k_vector_to_magnitude(src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=float)):
    tid = wp.tid()
    dst[tid] = wp.length(src[tid])


@wp.kernel
def k_staggered_to_magnitude(
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=float),
):
    """Voxel-centred magnitude of a staggered vector field.

    Audit note from the synthesis: the original Y/Z component reads and a Z
    half-scale assignment were suspect. This port deliberately averages each
    axis pair (lower face + upper face) symmetrically for X, Y and Z.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    ux0 = u[tid]
    uy0 = v[tid]
    uz0 = w[tid]
    xp = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2]))
    yp = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2]))
    zp = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1], ijk[2] + 1))
    # Branches, not wp.where: both wp.where operands are evaluated, which
    # read index -1 at a positive edge.
    ux1 = ux0
    uy1 = uy0
    uz1 = uz0
    if xp >= 0:
        ux1 = u[xp]
    if yp >= 0:
        uy1 = v[yp]
    if zp >= 0:
        uz1 = w[zp]
    c = wp.vec3(0.5 * (ux0 + ux1), 0.5 * (uy0 + uy1), 0.5 * (uz0 + uz1))
    dst[tid] = wp.length(c)


register_fn(k_vector_to_magnitude, "coreConvertFieldVectorToMagnitude")
register_fn(k_staggered_to_magnitude, "coreConvertFieldStaggeredVectorToMagnitude",
            notes="audit item resolved: symmetric axis-pair averaging")


# Max active value (core_misc) — masked absolute max over active voxels


@wp.kernel
def k_max_active_value(
    data: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    out_max: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    wp.atomic_max(out_max, 0, wp.abs(data[tid]))


@wp.kernel
def k_max_vec3_magnitude(
    data: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    out_max: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    wp.atomic_max(out_max, 0, wp.length(data[tid]))


register_fn(k_max_active_value, "coreComputeMaxActiveValue")


# Scale field (control ramps)


@wp.kernel
def k_reset_scale_field(data: wp.array(dtype=float), value: float):
    data[wp.tid()] = value


@wp.kernel
def k_ramp_to_scale_field_volume(
    src: wp.array(dtype=float),
    ramp_lo: float,
    ramp_hi: float,
    out_lo: float,
    out_hi: float,
    combine: wp.int32,  # RampToScaleFieldCombine
    dst: wp.array(dtype=float),
):
    tid = wp.tid()
    t = wp.clamp((src[tid] - ramp_lo) / wp.max(1.0e-20, ramp_hi - ramp_lo), 0.0, 1.0)
    s = out_lo + t * (out_hi - out_lo)
    if combine == 0:
        dst[tid] = dst[tid] * s
    elif combine == 1:
        dst[tid] = wp.max(dst[tid], s)
    else:
        dst[tid] = wp.min(dst[tid], s)


@wp.kernel
def k_ramp_to_scale_field_particle(
    particle_attr: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    ramp_lo: float,
    ramp_hi: float,
    out_lo: float,
    out_hi: float,
    combine: wp.int32,
    dst: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    t = wp.clamp((particle_attr[tid] - ramp_lo) / wp.max(1.0e-20, ramp_hi - ramp_lo), 0.0, 1.0)
    s = out_lo + t * (out_hi - out_lo)
    if combine == 0:
        dst[tid] = dst[tid] * s
    elif combine == 1:
        dst[tid] = wp.max(dst[tid], s)
    else:
        dst[tid] = wp.min(dst[tid], s)


register_fn(k_reset_scale_field, "coreResetScaleField")
register_fn(k_ramp_to_scale_field_volume, "coreRampToScaleFieldVolume")
register_fn(k_ramp_to_scale_field_particle, "coreRampToScaleFieldParticle")


# Sort/gather kernels (coreSort*) — apply a permutation to attribute buffers


def _make_gather_kernel(dtype):
    def _gather(order, count, src, dst):
        tid = wp.tid()
        if tid >= count[0]:
            return
        dst[tid] = src[order[tid]]

    _gather.__annotations__ = {
        "order": wp.array(dtype=wp.int32),
        "count": wp.array(dtype=wp.int32),
        "src": wp.array(dtype=dtype),
        "dst": wp.array(dtype=dtype),
    }
    _gather.__name__ = f"k_gather_{getattr(dtype, '__name__', str(dtype))}"
    return wp.kernel(_gather)


k_gather_i32 = _make_gather_kernel(wp.int32)
k_gather_i16 = _make_gather_kernel(wp.int16)
k_gather_f16 = _make_gather_kernel(wp.float16)
k_gather_f32 = _make_gather_kernel(wp.float32)
k_gather_v3 = _make_gather_kernel(wp.vec3)
k_gather_c2 = _make_gather_kernel(wp.vec2i)

register_fn(k_gather_i32, "coreSortScalarInt32", "coreSortScalarVoxelType")
register_fn(k_gather_i16, "coreSortScalarInt16")
register_fn(k_gather_f16, "coreSortScalarFloat16")
register_fn(k_gather_f32, "coreSortScalarFloat32", "coreSortScalarField")
register_fn(k_gather_v3, "coreSortVectorFloat32", "coreSortVectorField")
register_fn(k_gather_c2, "coreSortCoupleField")


# SDF utilities (core_sdf_collision)


@wp.kernel
def k_create_sdf_from_mesh_axis(
    mesh_id: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    max_dist: float,
    sign_band: float,
    sdf: wp.array(dtype=float),
):
    """coreCreateSDFAxisX/Y/Z analogue.

    The original built SDFs by per-axis distance sweeps; with Warp we query
    the BVH-backed mesh directly per voxel (all three axes in one pass),
    which preserves the result contract (signed distance in a narrow band).
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    p = origin + wp.vec3(
        (float(ijk[0]) + 0.5) * voxel_size,
        (float(ijk[1]) + 0.5) * voxel_size,
        (float(ijk[2]) + 0.5) * voxel_size,
    )
    q = wp.mesh_query_point_sign_normal(mesh_id, p, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        d = wp.length(p - cp) * q.sign
        sdf[tid] = wp.min(sdf[tid], d)
    else:
        sdf[tid] = wp.min(sdf[tid], sign_band)


@wp.kernel
def k_smooth_sdf(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    strength: float,
    dst: wp.array(dtype=float),
):
    """coreSmoothSDF: 6-neighbour box relaxation blended by strength."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = src[tid]
    acc = float(0.0)
    n = float(0.0)
    for a in range(3):
        for s in range(-1, 2, 2):
            o = wp.vec3i(ijk[0], ijk[1], ijk[2])
            if a == 0:
                o = wp.vec3i(ijk[0] + s, ijk[1], ijk[2])
            elif a == 1:
                o = wp.vec3i(ijk[0], ijk[1] + s, ijk[2])
            else:
                o = wp.vec3i(ijk[0], ijk[1], ijk[2] + s)
            flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
            if flat >= 0:
                acc += src[flat]
                n += 1.0
    if n > 0.0:
        dst[tid] = c + strength * (acc / n - c)
    else:
        dst[tid] = c


@wp.kernel
def k_renormalize_sdf_sweep(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    dst: wp.array(dtype=float),
):
    """coreRenormalizeSdfFastSweep: parallel Eikonal relaxation step.

    |grad phi| = 1 is enforced by an upwind Godunov update; several
    iterations approximate the sequential fast sweep of the original.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    phi = src[tid]
    s = wp.sign(phi)
    if wp.abs(phi) <= voxel_size:
        dst[tid] = phi  # keep interface voxels
        return
    # min |phi| of upwind neighbours per axis
    h = voxel_size
    m = wp.vec3(1.0e9, 1.0e9, 1.0e9)
    for a in range(3):
        best = float(1.0e9)
        for sgn in range(-1, 2, 2):
            o = wp.vec3i(ijk[0], ijk[1], ijk[2])
            if a == 0:
                o = wp.vec3i(ijk[0] + sgn, ijk[1], ijk[2])
            elif a == 1:
                o = wp.vec3i(ijk[0], ijk[1] + sgn, ijk[2])
            else:
                o = wp.vec3i(ijk[0], ijk[1], ijk[2] + sgn)
            flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
            if flat >= 0:
                best = wp.min(best, s * src[flat])
        if a == 0:
            m = wp.vec3(best, m[1], m[2])
        elif a == 1:
            m = wp.vec3(m[0], best, m[2])
        else:
            m = wp.vec3(m[0], m[1], best)
    # sort ascending
    a0 = wp.min(m[0], wp.min(m[1], m[2]))
    a2 = wp.max(m[0], wp.max(m[1], m[2]))
    a1 = m[0] + m[1] + m[2] - a0 - a2
    d = a0 + h
    if d > a1:
        d = 0.5 * (a0 + a1 + wp.sqrt(wp.max(0.0, 2.0 * h * h - (a1 - a0) * (a1 - a0))))
        if d > a2:
            ss = a0 + a1 + a2
            q = ss * ss - 3.0 * (a0 * a0 + a1 * a1 + a2 * a2 - h * h)
            d = (ss + wp.sqrt(wp.max(0.0, q))) / 3.0
    if a0 > 1.0e8:
        dst[tid] = phi
    else:
        dst[tid] = s * wp.min(s * phi, d)


@wp.kernel
def k_recompute_collision_bit_from_sdf(
    sdf: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    threshold: float,
    voxel_type: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if sdf[tid] < threshold:
        voxel_type[tid] = voxel_type[tid] | VOXEL_COLLISION
    else:
        voxel_type[tid] = voxel_type[tid] & (~VOXEL_COLLISION)


@wp.kernel
def k_sdf_to_gradient(
    sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    grad: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = sdf[tid]
    g = wp.vec3(0.0, 0.0, 0.0)
    inv2h = 1.0 / (2.0 * voxel_size)
    for a in range(3):
        lo = wp.vec3i(ijk[0], ijk[1], ijk[2])
        hi = wp.vec3i(ijk[0], ijk[1], ijk[2])
        if a == 0:
            lo = wp.vec3i(ijk[0] - 1, ijk[1], ijk[2])
            hi = wp.vec3i(ijk[0] + 1, ijk[1], ijk[2])
        elif a == 1:
            lo = wp.vec3i(ijk[0], ijk[1] - 1, ijk[2])
            hi = wp.vec3i(ijk[0], ijk[1] + 1, ijk[2])
        else:
            lo = wp.vec3i(ijk[0], ijk[1], ijk[2] - 1)
            hi = wp.vec3i(ijk[0], ijk[1], ijk[2] + 1)
        flo = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, lo)
        fhi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, hi)
        vlo = c
        vhi = c
        if flo >= 0:
            vlo = sdf[flo]
        if fhi >= 0:
            vhi = sdf[fhi]
        d = (vhi - vlo) * inv2h
        if a == 0:
            g = wp.vec3(d, g[1], g[2])
        elif a == 1:
            g = wp.vec3(g[0], d, g[2])
        else:
            g = wp.vec3(g[0], g[1], d)
    grad[tid] = g


register_fn(k_create_sdf_from_mesh_axis, "coreCreateSDFAxisX", "coreCreateSDFAxisY",
            "coreCreateSDFAxisZ",
            notes="axis sweeps replaced by BVH mesh sign/normal queries")
register_fn(k_smooth_sdf, "coreSmoothSDF")
register_fn(k_renormalize_sdf_sweep, "coreRenormalizeSdfFastSweep")
register_fn(k_recompute_collision_bit_from_sdf, "coreRecomputeCollisionBitFromSdf")
register_fn(k_sdf_to_gradient, "coreSDFToGradient")


# Generic particle <-> grid transfer (core_particle_grid_transfer)


@wp.kernel
def k_grid_to_particle_field_attribute(
    field_data: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    blend: float,  # 0=replace .. 1=keep
    attr: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    val = sample_scalar_trilinear(field_data, leaf_table, voxels_dim, leaves_dim, vp, 0.0)
    attr[tid] = blend * attr[tid] + (1.0 - blend) * val


@wp.kernel
def k_particle_to_grid_collect(
    positions: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    accum: wp.array(dtype=float),
    weight: wp.array(dtype=float),
):
    """Trilinear scatter of a particle scalar into voxel centres."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    q = vp - wp.vec3(0.5, 0.5, 0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    val = values[tid]
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                ijk = wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                if flat >= 0:
                    wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                    wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                    wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                    w = wx * wy * wz
                    wp.atomic_add(accum, flat, w * val)
                    wp.atomic_add(weight, flat, w)


@wp.kernel
def k_particle_to_grid_average(
    accum: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    w = weight[tid]
    if w > 1.0e-8:
        accum[tid] = accum[tid] / w


register_fn(k_grid_to_particle_field_attribute, "coreGridToParticleFieldAttribute")
register_fn(k_particle_to_grid_collect, "coreParticleToGridCollect")
register_fn(k_particle_to_grid_average, "coreParticleToGridAverage")


@wp.kernel
def k_grid_expand_scalar(
    src: wp.array(dtype=float),
    src_weight: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=float),
    dst_weight: wp.array(dtype=float),
):
    """coreGridExpand: flood one voxel ring of valid data into empty voxels."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if src_weight[tid] > 0.0:
        dst[tid] = src[tid]
        dst_weight[tid] = src_weight[tid]
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    acc = float(0.0)
    n = float(0.0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                o = wp.vec3i(ijk[0] + dx, ijk[1] + dy, ijk[2] + dz)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
                if flat >= 0 and src_weight[flat] > 0.0:
                    acc += src[flat]
                    n += 1.0
    if n > 0.0:
        dst[tid] = acc / n
        dst_weight[tid] = 1.0e-6  # extrapolated marker weight
    else:
        dst[tid] = src[tid]
        dst_weight[tid] = 0.0


register_fn(k_grid_expand_scalar, "coreGridExpand")
