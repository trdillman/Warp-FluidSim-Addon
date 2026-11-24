"""Guided shell upres: steer a fine narrow-band solve from a coarse liquid.

Nielsen and Bridson, "Guide Shapes for High Resolution Naturalistic Liquid
Simulation" (SIGGRAPH 2011): a finalised low-resolution liquid is re-run at
high resolution only in a thin outer shell, and the coarse motion is imposed
below the shell. The shell then forms its own fine sheets, splashes and
ripples while the large motion stays the approved one.

Here the fine solve is an ordinary ``FluidSolver`` in narrow-band mode, which
already keeps particles only near the surface and carries the deep liquid as
a level set ``phi`` plus a stored MAC velocity (``nb_u/v/w``). The guide
replaces that deep state with the coarse one each substep:

* ``k_guide_interior_velocity``: below the shell, the stored velocity becomes
  the coarse MAC velocity sampled at the fine faces, blended in time between
  two coarse frames. The fine pressure solve then sees the coarse motion as
  the liquid it must move with, which is the guide boundary condition.
* ``k_guide_phi``: the fine level set always contains the coarse liquid
  eroded by the shell depth, and deep fine liquid may not drift more than one
  more shell outside the coarse surface. The coarse level set is first made
  a true distance inside the liquid (``GuideFrame.reconstruct_depth``): a
  particle-splat SDF is only a union of balls and is never deeper than a
  particle radius.
* ``k_guide_seed_*``: fine particles fill the coarse shell on the first frame,
  then wherever the coarse liquid advances. The narrow band's own reseed
  cannot start a solve from nothing (it only replaces particles it killed).
* ``k_guide_mark_leaves``: fine leaves are allocated wherever the coarse
  liquid is, so the union above has somewhere to land.

A ``GuideFrame`` is a self-contained copy of one coarse frame's level set and
MAC velocity together with the coarse leaf table, so the coarse solver may
rebuild its topology or step on while the fine solver still samples it.
"""

from __future__ import annotations

import copy
import math

import warp as wp

from . import constants as C
from .fastlaunch import launch as fast_launch
from .fields import (
    leaf_coord_of_voxel,
    leaf_linear_index,
    sample_mac_component,
    sample_scalar_trilinear,
    voxel_center_world,
    voxel_coord_of_flat,
    voxel_flat_index,
    voxel_floor,
    voxel_in_domain,
    world_to_voxel,
)
from .kernels_narrow_band import k_nb_eikonal_inward, k_nb_reseed_reserve
from .pressure import neighbor_flat
from .registry import register_fn

wp.set_module_options({"enable_backward": False})

#: Value a guide level-set sample takes outside the coarse active leaves.
GUIDE_FAR = 1.0e6


@wp.kernel
def k_guide_deepen_init(
    sdf: wp.array(dtype=float),
    collision: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    depth: float,
    out: wp.array(dtype=float),
):
    """Keep liquid cells that touch air; set the rest of the liquid to ``-depth``.

    A particle-splat SDF is only a union of balls: inside the liquid it never
    goes below minus the particle radius. The inward eikonal sweeps that
    follow rebuild the true distance from the surface cells kept here.
    A solid neighbour (``collision < 0``) is a wall, not air, so liquid
    resting on a collider is deepened too.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    p = sdf[tid]
    out[tid] = p
    if p >= 0.0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    touches_air = wp.int32(0)
    for axis in range(3):
        for sign in range(2):
            n = ijk
            n[axis] = n[axis] + 2 * sign - 1
            flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
            if flat >= 0:
                if sdf[flat] >= 0.0 and collision[flat] >= 0.0:
                    touches_air = 1
    if touches_air == 0:
        out[tid] = -depth


register_fn(k_guide_deepen_init, "fluidSimGuideUpresDeepenInit",
            notes="Guided upres: seed the coarse interior for depth reconstruction.")


@wp.kernel
def k_guide_face_max(
    vel: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    slot: int,
    out: wp.array(dtype=float),
):
    """Largest |face velocity| over the valid (weighted) MAC faces."""
    tid = wp.tid()
    if tid < num_leaves[0] * C.LEAF_VOLUME and weight[tid] > 0.0 and wp.isfinite(vel[tid]):
        wp.atomic_max(out, slot, wp.abs(vel[tid]))


register_fn(k_guide_face_max, "fluidSimGuideUpresFaceMax",
            notes="Guided upres: guide speed bound for the fine CFL schedule.")


class GuideFrame:
    """One coarse frame, copied so it outlives the coarse solver's buffers.

    With ``depth``, the copied level set is made a true distance inside the
    liquid down to ``-depth`` (``reconstruct_depth``), so shell erosion works
    on a particle-splat SDF too.
    """

    def __init__(self, grid, sdf, vel, time, *, depth=None, collision=None):
        self.time = float(time)
        self.origin = grid.origin
        self.voxel_size = float(grid.voxel_size)
        self.voxels_dim = grid.voxels_dim
        self.leaves_dim = grid.leaves_dim
        self.leaf_table = wp.clone(grid.leaf_table)
        self.leaf_coords = wp.clone(grid.leaf_coords)
        self.num_leaves = wp.clone(grid.num_leaves)
        self.active_voxels = max(1, int(grid.active_voxel_count))
        self.sdf = wp.clone(sdf.data)
        self.u = wp.clone(vel.u)
        self.v = wp.clone(vel.v)
        self.w = wp.clone(vel.w)
        self.wu = wp.clone(vel.wu)
        self.wv = wp.clone(vel.wv)
        self.ww = wp.clone(vel.ww)
        self._face_speed_limits = None
        # Coarse one-shot sources this frame had already consumed.
        self.fired = frozenset()
        if depth is not None:
            self.reconstruct_depth(float(depth), collision)

    def face_speed_limits(self):
        """Maximum absolute valid face velocity along each axis (cached)."""
        if self._face_speed_limits is None:
            out = wp.zeros(3, dtype=float, device=self.u.device)
            for axis, (vel, weight) in enumerate(
                    ((self.u, self.wu), (self.v, self.wv), (self.w, self.ww))):
                n = min(self.active_voxels, int(vel.shape[0]), int(weight.shape[0]))
                if n > 0:
                    fast_launch(k_guide_face_max, dim=n,
                                inputs=[vel, weight, self.num_leaves, axis],
                                outputs=[out], device=vel.device)
            self._face_speed_limits = tuple(float(component) for component in out.numpy())
        return self._face_speed_limits

    def reconstruct_depth(self, depth, collision=None):
        """Rebuild the inward distance down to ``-depth`` (eikonal sweeps).

        Solid cells (``collision < 0``) and unallocated cells are not a free
        surface, so liquid resting on a wall stays deep.
        """
        dev = self.sdf.device
        n = int(self.sdf.shape[0])
        if collision is None:
            collision = wp.full(n, 1.0, dtype=float, device=dev)
        tmp = wp.empty_like(self.sdf)
        dim = self.active_voxels
        fast_launch(
            k_guide_deepen_init, dim=dim,
            inputs=[self.sdf, collision, self.leaf_table, self.leaf_coords,
                    self.num_leaves, self.voxels_dim, self.leaves_dim, depth],
            outputs=[tmp], device=dev)
        phi, out = tmp, self.sdf
        sweeps = int(math.ceil(depth / self.voxel_size)) + 2
        for _ in range(sweeps):
            fast_launch(
                k_nb_eikonal_inward, dim=dim,
                inputs=[phi, collision, self.leaf_table, self.leaf_coords,
                        self.num_leaves, self.voxels_dim, self.leaves_dim,
                        self.voxel_size, depth],
                outputs=[out], device=dev)
            phi, out = out, phi
        self.sdf = phi

    @classmethod
    def from_solver(cls, solver, time, depth=None):
        """Snapshot a coarse solver after its frame (its SDF made current)."""
        if not solver.surface.ensure_solver_sdf_current():
            raise RuntimeError(solver.surface.solver_sdf_error)
        sdf = solver.surface.sdf
        narrow_band = getattr(solver, "narrow_band", None)
        if narrow_band is not None:
            narrow_band.union_into_solver_sdf(sdf)
        return cls(solver.grid, sdf, solver.vel, time, depth=depth,
                   collision=solver.collision_sdf.data)


@wp.func
def _guide_phi_at(
    sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    h: float,
    x: wp.vec3,
) -> float:
    return sample_scalar_trilinear(
        sdf, leaf_table, voxels_dim, leaves_dim,
        world_to_voxel(x, origin, h), GUIDE_FAR)


@wp.func
def _guide_blend(ga: float, gb: float, alpha: float) -> float:
    """Blend two guide samples; ``GUIDE_FAR`` if frame ``b`` has none.

    A frame ``a`` sample outside its leaves (an empty first frame, or liquid
    the coarse solve has just added) is not a constraint, so frame ``b``
    alone is used there.
    """
    if gb >= GUIDE_FAR * 0.5:
        return GUIDE_FAR
    if ga >= GUIDE_FAR * 0.5:
        return gb
    return ga + (gb - ga) * alpha


@wp.kernel
def k_guide_phi(
    phi: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    h: float,
    shell: float,
    limit: float,
    a_sdf: wp.array(dtype=float),
    a_table: wp.array(dtype=wp.int32),
    b_sdf: wp.array(dtype=float),
    b_table: wp.array(dtype=wp.int32),
    g_voxels_dim: wp.vec3i,
    g_leaves_dim: wp.vec3i,
    g_origin: wp.vec3,
    g_h: float,
    alpha: float,
):
    """Keep the fine level set within the coarse liquid's shell.

    Fine solid (collision SDF or collision bit) and sink cells stay air, as
    the narrow band's own union
    leaves them: the coarse level set does not resolve the finer walls.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    # Native collision regions stamp only the voxel bit, not the SDF.
    if collision_sdf[tid] < 0.0 or (
            voxel_type[tid] & (C.VOXEL_COLLISION | C.VOXEL_SINK)) != 0:
        phi[tid] = limit
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                             float(ijk[2]) + 0.5)
    ga = _guide_phi_at(a_sdf, a_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    gb = _guide_phi_at(b_sdf, b_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    g = _guide_blend(ga, gb, alpha)
    if g >= GUIDE_FAR * 0.5:
        return
    p = phi[tid]
    # Coarse liquid deeper than the shell is always liquid here too.
    p = wp.min(p, g + shell)
    # Deep fine liquid may not sit more than a further shell outside the
    # coarse surface; the band near the surface stays free.
    if p < -shell:
        p = wp.max(p, g - 2.0 * shell)
    phi[tid] = wp.clamp(p, -limit, limit)


register_fn(k_guide_phi, "fluidSimGuideUpresPhi",
            notes="Guided upres: coarse level set bounds the fine one.")


@wp.func
def _mac_sample(
    data: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    vp: wp.vec3,
    axis: wp.int32,
) -> wp.vec2:
    """``sample_mac_component`` that also says whether any face was valid:
    returns (value, 1) or (0, 0)."""
    v = sample_mac_component(data, weight, leaf_table, voxels_dim, leaves_dim, vp, axis)
    off = wp.vec3(0.5, 0.5, 0.5)
    if axis == 0:
        off = wp.vec3(0.0, 0.5, 0.5)
    elif axis == 1:
        off = wp.vec3(0.5, 0.0, 0.5)
    else:
        off = wp.vec3(0.5, 0.5, 0.0)
    q = vp - off
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wsum = wp.float32(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                ijk = wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                w = (wp.where(dx == 1, f[0], 1.0 - f[0])
                     * wp.where(dy == 1, f[1], 1.0 - f[1])
                     * wp.where(dz == 1, f[2], 1.0 - f[2]))
                if flat >= 0 and weight[flat] > 0.0:
                    wsum += w
    if wsum > 1.0e-8:
        return wp.vec2(v, 1.0)
    return wp.vec2(0.0, 0.0)


@wp.func
def _guide_face(
    a_u: wp.array(dtype=float), a_wu: wp.array(dtype=float),
    a_table: wp.array(dtype=wp.int32),
    b_u: wp.array(dtype=float), b_wu: wp.array(dtype=float),
    b_table: wp.array(dtype=wp.int32),
    g_voxels_dim: wp.vec3i,
    g_leaves_dim: wp.vec3i,
    g_origin: wp.vec3,
    g_h: float,
    x: wp.vec3,
    axis: int,
    alpha: float,
) -> float:
    vp = world_to_voxel(x, g_origin, g_h)
    sa = _mac_sample(a_u, a_wu, a_table, g_voxels_dim, g_leaves_dim, vp, axis)
    sb = _mac_sample(b_u, b_wu, b_table, g_voxels_dim, g_leaves_dim, vp, axis)
    # A frame with no valid face here has no velocity (not a velocity of 0):
    # use the other frame alone, as ``_guide_blend`` does for phi.
    if sa[1] > 0.0 and sb[1] <= 0.0:
        return sa[0]
    if sb[1] > 0.0 and sa[1] <= 0.0:
        return sb[0]
    return sa[0] + (sb[0] - sa[0]) * alpha


@wp.kernel
def k_guide_interior_velocity(
    phi: wp.array(dtype=float),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    h: float,
    shell: float,
    a_u: wp.array(dtype=float), a_v: wp.array(dtype=float), a_w: wp.array(dtype=float),
    a_wu: wp.array(dtype=float), a_wv: wp.array(dtype=float), a_ww: wp.array(dtype=float),
    a_table: wp.array(dtype=wp.int32),
    b_u: wp.array(dtype=float), b_v: wp.array(dtype=float), b_w: wp.array(dtype=float),
    b_wu: wp.array(dtype=float), b_wv: wp.array(dtype=float), b_ww: wp.array(dtype=float),
    b_table: wp.array(dtype=wp.int32),
    g_voxels_dim: wp.vec3i,
    g_leaves_dim: wp.vec3i,
    g_origin: wp.vec3,
    g_h: float,
    alpha: float,
    nb_u: wp.array(dtype=float),
    nb_v: wp.array(dtype=float),
    nb_w: wp.array(dtype=float),
):
    """Below the shell, the stored interior velocity is the coarse one."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    if phi[tid] >= -shell:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    i = float(ijk[0])
    j = float(ijk[1])
    k = float(ijk[2])
    # Lower faces of this voxel, in world space.
    xu = origin + h * wp.vec3(i, j + 0.5, k + 0.5)
    xv = origin + h * wp.vec3(i + 0.5, j, k + 0.5)
    xw = origin + h * wp.vec3(i + 0.5, j + 0.5, k)
    nb_u[tid] = _guide_face(a_u, a_wu, a_table, b_u, b_wu, b_table,
                            g_voxels_dim, g_leaves_dim, g_origin, g_h, xu, 0, alpha)
    nb_v[tid] = _guide_face(a_v, a_wv, a_table, b_v, b_wv, b_table,
                            g_voxels_dim, g_leaves_dim, g_origin, g_h, xv, 1, alpha)
    nb_w[tid] = _guide_face(a_w, a_ww, a_table, b_w, b_ww, b_table,
                            g_voxels_dim, g_leaves_dim, g_origin, g_h, xw, 2, alpha)


register_fn(k_guide_interior_velocity, "fluidSimGuideUpresVelocity",
            notes="Guided upres: coarse MAC velocity below the fine shell.")


@wp.kernel
def k_guide_stamp_core(
    voxel_type: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    h: float,
    depth: float,
    a_sdf: wp.array(dtype=float),
    a_table: wp.array(dtype=wp.int32),
    b_sdf: wp.array(dtype=float),
    b_table: wp.array(dtype=wp.int32),
    g_voxels_dim: wp.vec3i,
    g_leaves_dim: wp.vec3i,
    g_origin: wp.vec3,
    g_h: float,
    alpha: float,
):
    """Exclude deep guide liquid from pressure unknowns without tagging real solids."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & C.VOXEL_FLUID) == 0 or (vt & (C.VOXEL_COLLISION | C.VOXEL_SINK)) != 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = voxel_center_world(ijk, origin, h)
    ga = _guide_phi_at(a_sdf, a_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    gb = _guide_phi_at(b_sdf, b_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    g = _guide_blend(ga, gb, alpha)
    if g < -depth:
        voxel_type[tid] = vt | C.VOXEL_COLLISION | C.VOXEL_GUIDED_CORE


register_fn(k_guide_stamp_core, "fluidSimGuideUpresStampCore",
            notes="Guided upres: deep core is a coarse-driven solid for pressure.")


@wp.kernel
def k_guide_core_faces(
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    h: float,
    a_u: wp.array(dtype=float), a_v: wp.array(dtype=float), a_w: wp.array(dtype=float),
    a_wu: wp.array(dtype=float), a_wv: wp.array(dtype=float), a_ww: wp.array(dtype=float),
    a_table: wp.array(dtype=wp.int32),
    b_u: wp.array(dtype=float), b_v: wp.array(dtype=float), b_w: wp.array(dtype=float),
    b_wu: wp.array(dtype=float), b_wv: wp.array(dtype=float), b_ww: wp.array(dtype=float),
    b_table: wp.array(dtype=wp.int32),
    g_voxels_dim: wp.vec3i,
    g_leaves_dim: wp.vec3i,
    g_origin: wp.vec3,
    g_h: float,
    alpha: float,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
):
    """Prescribe guide flux at the MAC faces. Physical walls take precedence."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    solid_mask = C.VOXEL_COLLISION | C.VOXEL_GUIDED_CORE
    here = voxel_type[tid]
    if (here & solid_mask) == C.VOXEL_COLLISION:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = voxel_center_world(ijk, origin, h)
    for axis in range(3):
        n = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, -1)
        if n < 0:
            continue
        other = voxel_type[n]
        if (other & solid_mask) == C.VOXEL_COLLISION:
            continue
        if ((here | other) & C.VOXEL_GUIDED_CORE) == 0:
            continue
        x = center
        x[axis] = x[axis] - 0.5 * h
        if axis == 0:
            u[tid] = _guide_face(a_u, a_wu, a_table, b_u, b_wu, b_table,
                                 g_voxels_dim, g_leaves_dim, g_origin, g_h, x, 0, alpha)
            wu[tid] = wp.max(wu[tid], 1.0e-6)
        elif axis == 1:
            v[tid] = _guide_face(a_v, a_wv, a_table, b_v, b_wv, b_table,
                                 g_voxels_dim, g_leaves_dim, g_origin, g_h, x, 1, alpha)
            wv[tid] = wp.max(wv[tid], 1.0e-6)
        else:
            w[tid] = _guide_face(a_w, a_ww, a_table, b_w, b_ww, b_table,
                                 g_voxels_dim, g_leaves_dim, g_origin, g_h, x, 2, alpha)
            ww[tid] = wp.max(ww[tid], 1.0e-6)


register_fn(k_guide_core_faces, "fluidSimGuideUpresCoreFaces",
            notes="Guided upres: prescribe coarse velocity on staggered core faces.")


@wp.kernel
def k_guide_unstamp_core(
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
):
    """Undo ``k_guide_stamp_core`` after the pressure solve."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & C.VOXEL_GUIDED_CORE) != 0:
        voxel_type[tid] = vt & (~(C.VOXEL_COLLISION | C.VOXEL_GUIDED_CORE))


register_fn(k_guide_unstamp_core, "fluidSimGuideUpresUnstampCore",
            notes="Guided upres: restore the core after the pressure solve.")


@wp.kernel
def k_guide_mark_leaves(
    g_sdf: wp.array(dtype=float),
    g_leaf_coords: wp.array(dtype=wp.vec3i),
    g_num_leaves: wp.array(dtype=wp.int32),
    g_origin: wp.vec3,
    g_h: float,
    reach: float,
    origin: wp.vec3,
    h: float,
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    leaf_flags: wp.array(dtype=wp.int32),
):
    """Allocate fine leaves under every coarse voxel within ``reach`` of liquid."""
    tid = wp.tid()
    if tid >= g_num_leaves[0] * C.LEAF_VOLUME:
        return
    if g_sdf[tid] > reach:
        return
    c = voxel_coord_of_flat(g_leaf_coords, tid)
    for corner in range(8):
        x = g_origin + g_h * wp.vec3(float(c[0] + (corner & 1)),
                                     float(c[1] + ((corner >> 1) & 1)),
                                     float(c[2] + ((corner >> 2) & 1)))
        ijk = voxel_floor(world_to_voxel(x, origin, h))
        if voxel_in_domain(ijk, voxels_dim):
            leaf_flags[leaf_linear_index(leaf_coord_of_voxel(ijk), leaves_dim)] = 1


@wp.kernel
def k_guide_seed_demand(
    phi: wp.array(dtype=float),
    cls: wp.array(dtype=wp.int32),
    collision_sdf: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    voxel_counts: wp.array(dtype=wp.int32),
    band_demand: wp.array(dtype=wp.int32),
    band_offsets: wp.array(dtype=wp.int32),
    band_admitted: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    h: float,
    shell: float,
    a_sdf: wp.array(dtype=float),
    a_table: wp.array(dtype=wp.int32),
    b_sdf: wp.array(dtype=float),
    b_table: wp.array(dtype=wp.int32),
    g_voxels_dim: wp.vec3i,
    g_leaves_dim: wp.vec3i,
    g_origin: wp.vec3,
    g_h: float,
    alpha: float,
    bootstrap: int,
    fill: int,
    demand: wp.array(dtype=wp.int32),
):
    """Fine particles owed to empty shell cells of the coarse liquid.

    Only cells in the coarse shell (between half a fine voxel and the shell
    depth plus one voxel below the coarse surface) that hold no particle are
    filled. After the first frame (``bootstrap`` 0) only liquid the coarse
    solve has just added is filled (no liquid there in frame ``a``), so the
    fine shell keeps its own shape where it already lives.
    """
    tid = wp.tid()
    demand[tid] = 0
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    if cls[tid] == 2 or collision_sdf[tid] < 0.0:
        return
    # Same blocked cells as the band's own reseed: native collision regions
    # stamp only the voxel bit, so the SDF test above misses them.
    if (voxel_type[tid] & (C.VOXEL_COLLISION | C.VOXEL_SINK)) != 0:
        return
    if voxel_counts[tid] > 0:
        return
    # The band's own reseed already filled this cell (its block start fits
    # in what the band admitted; a cell it asked for but could not get is
    # still the guide's to fill).
    if band_demand[tid] > 0 and band_offsets[tid] < band_admitted[0]:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = voxel_center_world(ijk, origin, h)
    ga = _guide_phi_at(a_sdf, a_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    gb = _guide_phi_at(b_sdf, b_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    g = _guide_blend(ga, gb, alpha)
    if g >= GUIDE_FAR * 0.5:
        return
    if g > -0.5 * h or g < -(shell + h):
        return
    # A missing frame-a sample (``GUIDE_FAR``) counts as new liquid.
    if bootstrap == 0 and ga < 0.0:
        return
    demand[tid] = fill


register_fn(k_guide_seed_demand, "fluidSimGuideUpresSeedDemand",
            notes="Guided upres: fine particles owed to the coarse shell.")


@wp.kernel
def k_guide_seed_commit(
    demand: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    h: float,
    seed: int,
    slot_base: wp.array(dtype=wp.int32),
    slot_limit: wp.array(dtype=wp.int32),
    a_u: wp.array(dtype=float), a_v: wp.array(dtype=float), a_w: wp.array(dtype=float),
    a_wu: wp.array(dtype=float), a_wv: wp.array(dtype=float), a_ww: wp.array(dtype=float),
    a_table: wp.array(dtype=wp.int32),
    b_u: wp.array(dtype=float), b_v: wp.array(dtype=float), b_w: wp.array(dtype=float),
    b_wu: wp.array(dtype=float), b_wv: wp.array(dtype=float), b_ww: wp.array(dtype=float),
    b_table: wp.array(dtype=wp.int32),
    g_voxels_dim: wp.vec3i,
    g_leaves_dim: wp.vec3i,
    g_origin: wp.vec3,
    g_h: float,
    alpha: float,
    position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    time_offset: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
):
    """Emit the shell particles with the coarse velocity at their position."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    emit_n = demand[tid]
    if emit_n <= 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = voxel_center_world(ijk, origin, h)
    local = offsets[tid]
    for k in range(emit_n):
        if local + k >= slot_limit[0]:
            return
        slot = slot_base[0] + local + k
        state = wp.rand_init(seed, tid * C.MAX_PARTICLES_PER_VOXEL + k)
        x = center + wp.vec3((wp.randf(state) - 0.5) * h,
                             (wp.randf(state) - 0.5) * h,
                             (wp.randf(state) - 0.5) * h)
        position[slot] = x
        velocity[slot] = wp.vec3(
            _guide_face(a_u, a_wu, a_table, b_u, b_wu, b_table,
                        g_voxels_dim, g_leaves_dim, g_origin, g_h, x, 0, alpha),
            _guide_face(a_v, a_wv, a_table, b_v, b_wv, b_table,
                        g_voxels_dim, g_leaves_dim, g_origin, g_h, x, 1, alpha),
            _guide_face(a_w, a_ww, a_table, b_w, b_ww, b_table,
                        g_voxels_dim, g_leaves_dim, g_origin, g_h, x, 2, alpha))
        age[slot] = 0.0
        life[slot] = 1.0e30
        temperature[slot] = 0.0
        viscosity[slot] = 0.0
        density[slot] = 1.0
        time_offset[slot] = 0.0
        alive[slot] = 1


register_fn(k_guide_seed_commit, "fluidSimGuideUpresSeedCommit",
            notes="Guided upres: emit shell particles at the coarse velocity.")

# Distinct from the narrow band's reseed seeds.
_SEED_SALT = 0x5F3759DF


class ShellGuide:
    """Applies two coarse ``GuideFrame``s to a fine narrow-band solver.

    The fine solver's ``NarrowBand`` calls ``apply_velocity`` after it
    advects its stored velocity and ``apply_phi`` after it rebuilds its level
    set; its topology rebuild calls ``mark_leaves``. ``alpha`` is the time of
    the current fine substep between the two frames (0 at ``a``, 1 at ``b``).
    """

    def __init__(self, shell_voxels, core_shells=2.0):
        self.shell_voxels = max(1.0, float(shell_voxels))
        # Fine liquid deeper than this many shells below the coarse surface
        # is a coarse-driven solid in the pressure solve (0 turns it off).
        core_shells = float(core_shells or 0.0)
        if not math.isfinite(core_shells):
            raise ValueError("core_shells must be finite")
        self.core_shells = max(0.0, core_shells)
        self.a = None
        self.b = None
        self.alpha = 0.0
        # Fill the whole coarse shell with fine particles until a frame ran.
        self.bootstrap = True
        self._demand = None
        self._budget = None
        self._stamp_live = False

    def set_frames(self, a, b):
        self.a, self.b = a, (b if b is not None else a)
        self.alpha = 0.0

    def set_time(self, t):
        if self.a is None:
            return
        span = self.b.time - self.a.time
        self.alpha = 0.0 if span <= 0.0 else min(1.0, max(0.0, (t - self.a.time) / span))

    @property
    def ready(self):
        return self.a is not None

    def max_speed(self):
        """Conservative speed limit over the current frame pair.

        Each component can fall back to a different frame when a face is missing.
        """
        if self.a is None:
            return 0.0
        a = self.a.face_speed_limits()
        b = self.b.face_speed_limits()
        return math.hypot(*(max(before, after) for before, after in zip(a, b, strict=True)))

    def _guide_inputs(self):
        a, b = self.a, self.b
        return a, b, [a.voxels_dim, a.leaves_dim, a.origin, a.voxel_size]

    def apply_phi(self, grid, phi, nvox, limit, device,
                  collision_sdf=None, voxel_type=None):
        """Bound the fine level set by the coarse one; solid and sink stay air."""
        if not self.ready:
            return
        a, b, common = self._guide_inputs()
        h = float(grid.voxel_size)
        n = int(phi.shape[0])
        if collision_sdf is None:
            collision_sdf = wp.full(n, 1.0, dtype=float, device=device)
        if voxel_type is None:
            voxel_type = wp.zeros(n, dtype=wp.int32, device=device)
        fast_launch(
            k_guide_phi, dim=max(1, int(nvox)),
            inputs=[phi, collision_sdf, voxel_type, grid.leaf_coords, grid.num_leaves, grid.origin, h,
                    self.shell_voxels * h, float(limit),
                    a.sdf, a.leaf_table, b.sdf, b.leaf_table, *common,
                    float(self.alpha)],
            device=device)

    def apply_velocity(self, grid, phi, nb_u, nb_v, nb_w, nvox, device):
        if not self.ready:
            return
        a, b, common = self._guide_inputs()
        h = float(grid.voxel_size)
        fast_launch(
            k_guide_interior_velocity, dim=max(1, int(nvox)),
            inputs=[phi, grid.leaf_coords, grid.num_leaves, grid.origin, h,
                    self.shell_voxels * h,
                    a.u, a.v, a.w, a.wu, a.wv, a.ww, a.leaf_table,
                    b.u, b.v, b.w, b.wu, b.wv, b.ww, b.leaf_table,
                    *common, float(self.alpha)],
            outputs=[nb_u, nb_v, nb_w], device=device)

    def seed(self, grid, nvox, phi, cls, collision_sdf, voxel_type, voxel_counts,
             band_demand, offsets, slot_base, slot_limit, overflow,
             particles, fill, seed, device):
        """Fill empty coarse-shell cells with fine particles; returns nothing.

        The narrow band's own reseed only tops up cells a voxel below its
        surface and only replaces particles it killed, so it cannot start a
        fine solve from nothing. This is the guide's emitter: the whole shell
        on the first frame, then only where the coarse liquid advanced.
        ``offsets`` and the slot arrays are the narrow band's scratch.
        """
        if not self.ready:
            return
        a, b, common = self._guide_inputs()
        h = float(grid.voxel_size)
        nv = max(1, int(nvox))
        if self._demand is None or self._demand.shape[0] < nv:
            self._demand = wp.zeros(max(nv, int(offsets.shape[0])),
                                    dtype=wp.int32, device=device)
        if self._budget is None:
            self._budget = wp.zeros(1, dtype=wp.int32, device=device)
        fast_launch(
            k_guide_seed_demand, dim=nv,
            inputs=[phi, cls, collision_sdf, voxel_type, voxel_counts, band_demand,
                    offsets, slot_limit,
                    grid.leaf_coords, grid.num_leaves, grid.origin, h,
                    self.shell_voxels * h, a.sdf, a.leaf_table, b.sdf,
                    b.leaf_table, *common, float(self.alpha),
                    1 if self.bootstrap else 0, int(fill)],
            outputs=[self._demand], device=device)
        wp.utils.array_scan(self._demand[:nv], offsets[:nv], inclusive=False)
        # Seeding adds liquid the coarse solve owns; only the buffer limits it.
        self._budget.fill_(int(particles.capacity))
        fast_launch(
            k_nb_reseed_reserve, dim=1,
            inputs=[self._demand, offsets, nv, int(particles.capacity),
                    self._budget, particles.count],
            outputs=[slot_base, slot_limit, overflow], device=device)
        fast_launch(
            k_guide_seed_commit, dim=nv,
            inputs=[self._demand, offsets, grid.leaf_coords, grid.num_leaves,
                    grid.origin, h, int((int(seed) ^ _SEED_SALT) & 0x7FFFFFFF),
                    slot_base, slot_limit,
                    a.u, a.v, a.w, a.wu, a.wv, a.ww, a.leaf_table,
                    b.u, b.v, b.w, b.wu, b.wv, b.ww, b.leaf_table,
                    *common, float(self.alpha),
                    particles.position, particles.velocity,
                    particles.attr("age"), particles.attr("life"),
                    particles.attr("temperature"), particles.attr("viscosity"),
                    particles.attr("density"), particles.attr("time_offset"),
                    particles.alive],
            device=device)
        # The caller (NarrowBand.reseed) refreshes the host particle count.

    def stamp_core(self, grid, voxel_type, velocity, nvox, device):
        """Before the fine pressure solve: the deep core becomes a solid that
        moves with the coarse velocity, so the solve only covers the shell."""
        self._stamp_live = False
        if not self.ready or self.core_shells <= 0.0:
            return
        a, b, common = self._guide_inputs()
        h = float(grid.voxel_size)
        nv = max(1, int(nvox))
        fast_launch(
            k_guide_stamp_core, dim=nv,
            inputs=[voxel_type, grid.leaf_coords, grid.num_leaves,
                    grid.origin, h, self.core_shells * self.shell_voxels * h,
                    a.sdf, a.leaf_table, b.sdf, b.leaf_table,
                    *common, float(self.alpha)], device=device)
        self._stamp_live = True
        fast_launch(
            k_guide_core_faces, dim=nv,
            inputs=[voxel_type, grid.leaf_table, grid.leaf_coords, grid.num_leaves,
                    grid.voxels_dim, grid.leaves_dim, grid.origin, h,
                    a.u, a.v, a.w, a.wu, a.wv, a.ww, a.leaf_table,
                    b.u, b.v, b.w, b.wu, b.wv, b.ww, b.leaf_table,
                    *common, float(self.alpha), *velocity.arrays()], device=device)

    def unstamp_core(self, grid, voxel_type, nvox, device):
        """After the pressure solve: the core is fine liquid again."""
        if not self._stamp_live:
            return
        self._stamp_live = False
        fast_launch(
            k_guide_unstamp_core, dim=max(1, int(nvox)),
            inputs=[voxel_type, grid.num_leaves],
            device=device)

    def mark_leaves(self, grid, device):
        if not self.ready:
            return
        h = float(grid.voxel_size)
        for frame in (self.a, self.b):
            fast_launch(
                k_guide_mark_leaves, dim=frame.active_voxels,
                inputs=[frame.sdf, frame.leaf_coords, frame.num_leaves,
                        frame.origin, frame.voxel_size,
                        frame.voxel_size + self.shell_voxels * h,
                        grid.origin, h, grid.voxels_dim, grid.leaves_dim,
                        grid.leaf_flags],
                device=device)


def fine_budget_scales(factor):
    """(particle, leaf) budget multipliers for a fine solve ``factor`` finer.

    Both scale with volume (``factor**3``). Particles live only in the shell,
    but liquid thinner than the shell (a sheet spreading over a beach) is all
    shell, and a budget sized for surface area (``factor**2``) then runs out
    and leaves holes in the fine surface. Leaves cover the whole coarse
    liquid volume, deep interior included (the fine level set and guide
    velocity live there).
    """
    factor = int(factor)
    volume = float(factor * factor * factor)
    return volume, volume


def fine_source_batch(batch):
    """The coarse sources the fine solve also needs: everything but liquid.

    Sinks must reach the fine solve (they stamp the sink bit its particle
    lifecycle kills by), as must forces, collision and activation regions.
    Liquid emission is left out: the guide seeds the fine shell itself.
    Returns ``(legacy_descriptors, native_plans)``.
    """
    from .enums import SourceOperation
    from .source_compat import _operation

    liquid = SourceOperation.LIQUID
    descriptors = tuple(d for d in batch.legacy_descriptors
                        if _operation(d) != liquid)
    plans = tuple(p for p in batch.native_plans
                  if SourceOperation(p.operation) != liquid)
    return descriptors, plans


def _moves(col):
    return getattr(col, "mesh", None) is not None and (
        any(abs(v) > 1.0e-9 for v in getattr(col, "linear_velocity", (0.0,)))
        or any(abs(v) > 1.0e-9 for v in getattr(col, "angular_velocity", (0.0,))))


def _private_mesh(src, cache):
    """A fine-owned copy of ``src`` with its current points (cached by source)."""
    key = id(src)
    entry = cache.get(key) if cache is not None else None
    if entry is not None and entry[0] is src and entry[1].points.shape == src.points.shape:
        mesh = entry[1]
        wp.copy(mesh.points, src.points)
        mesh.refit()
        return mesh
    mesh = wp.Mesh(points=wp.clone(src.points), indices=src.indices)
    if cache is not None:
        cache[key] = (src, mesh)
    return mesh


def fine_colliders(colliders, mesh_cache=None):
    """Private copies of the coarse colliders for the fine solve.

    A moving collider also gets its own mesh. ``mesh_cache`` keeps the
    copies for moving colliders that are still present in the current frame.
    The fine solver can re-pose these copies without touching coarse meshes.

    ``FluidSolver.set_colliders`` writes into the descriptors (moving-mesh
    base points, Dynamic body state), so sharing them would let the fine solve
    change the coarse one. Dynamic (two-way) colliders are refused: the fine
    solve would integrate its own bodies and drift off the coarse motion, and
    replaying the coarse body poses is not built yet.
    """
    out = []
    active_meshes = set()
    for col in colliders:
        if getattr(col, "dynamic", False) and getattr(col, "mesh", None) is not None:
            raise NotImplementedError(
                "guided upres does not support Dynamic colliders yet; "
                "the fine solve cannot follow the coarse rigid bodies")
        c = copy.copy(col)
        if _moves(col):
            c.mesh = _private_mesh(col.mesh, mesh_cache)
            active_meshes.add(id(col.mesh))
            c.reposed = False
        out.append(c)
    if mesh_cache is not None:
        for key in mesh_cache.keys() - active_meshes:
            del mesh_cache[key]
    return out


class GuidedUpres:
    """A fine narrow-band solver that follows a coarse ``FluidSolver``.

    Live use, once per display frame::

        upres = GuidedUpres(coarse, factor=2, shell_voxels=4)   # after coarse init
        coarse.step_frame()
        upres.step_frame()       # fine solve over the same frame
        verts, tris = upres.fine.surface_mesh()

    A baked post-pass (driving it from restored coarse checkpoints) is not
    built yet.

    The fine solver reuses every coarse setting except resolution, particle
    and leaf budgets, narrow-band mode (forced on, band = shell depth) and
    whitewater (off: emit it from the fine solve later if wanted).
    """

    def __init__(self, coarse, factor=2, shell_voxels=4.0, *,
                 particle_budget_scale=None, leaf_budget_scale=None,
                 core_shells=2.0, profiler=None):
        import dataclasses

        from .enums import SurfaceResolutionMode
        from .solver import FluidSolver

        factor = int(factor)
        if factor < 2:
            raise ValueError("guided upres needs a factor of at least 2")
        if factor > C.LEAF_DIM:
            # Leaf marking samples each coarse voxel's corners, which covers
            # every fine leaf it touches only while it spans at most one leaf.
            raise ValueError(f"guided upres supports factors up to {C.LEAF_DIM}")
        self.coarse = coarse
        self.factor = factor
        cs = coarse.settings
        p_scale, l_scale = fine_budget_scales(factor)
        p_scale = particle_budget_scale or p_scale
        l_scale = leaf_budget_scale or l_scale
        fine_settings = dataclasses.replace(
            cs,
            voxel_size=float(cs.voxel_size) / factor,
            max_particles=int(cs.max_particles * p_scale),
            leaf_capacity=int(cs.leaf_capacity * l_scale),
            narrow_band_enabled=True,
            narrow_band_voxels=float(shell_voxels),
            whitewater_enabled=False,
            surface_build_lowres_sdf=False,
            # The fine grid already is the high resolution: mesh at it.
            surface_resolution_mode=int(SurfaceResolutionMode.SIMULATION),
            # Reseeded narrow-band particles carry no per-particle product
            # channels (colour, source id, uid); the fine solve exports none.
            export_attributes=False,
        )
        # The fine solve is a narrow-band solve: refuse what narrow-band
        # reseeding cannot carry (species colour, particle-derived viscosity,
        # scalar field sources) before allocating it, instead of running a
        # fine shell that silently lacks it.
        from .enums import SourceOperation
        from .narrow_band import blocking_reasons

        plans = getattr(getattr(getattr(coarse, "sourcing", None), "batch", None),
                        "native_plans", ())
        reasons = blocking_reasons(fine_settings, scalar_field_source=any(
            SourceOperation(p.operation) == SourceOperation.SCALAR_FIELD for p in plans))
        if reasons:
            raise NotImplementedError(
                "Guided upres cannot run with this setup: " + "; ".join(reasons)
                + ". Turn the conflicting feature off for the upres.")
        self._meshes = {}
        colliders = fine_colliders(getattr(coarse, "colliders", ()),
                                   self._meshes)  # refuses Dynamic early
        self.guide = ShellGuide(shell_voxels, core_shells)
        self.fine = FluidSolver(fine_settings, profiler=profiler)
        self.fine.set_colliders(colliders)
        self.fine.narrow_band.guide = self.guide
        self._sources = None
        self._force_fields = None
        self._sync_scene()
        # Keep the original velocity band and reconstruct past the configured
        # pressure boundary so even its interpolated samples can be deeper.
        shell_depth = self.guide.shell_voxels * float(fine_settings.voxel_size)
        self.guide_depth = (max(3.0, self.guide.core_shells) * shell_depth
                            + 2.0 * float(cs.voxel_size))
        self.particle_budget_hit = False
        self._warned_budget = False
        self._last = self._snapshot()
        # One-shot sources the coarse solve already consumed must not fire
        # again in the fine one (it replays from this frame, not frame 0).
        self.fine.sourcing._initial_fired = set(self._last.fired)
        self.guide.set_frames(self._last, None)

    def _sync_scene(self):
        """Give the fine solve the coarse non-liquid sources and proxy force
        fields when they change (both are replaced whole on the coarse side)."""
        batch = self.coarse.sourcing.batch
        if batch is not self._sources:
            self._sources = batch
            descriptors, plans = fine_source_batch(batch)
            self.fine.set_source_batch(descriptors, plans)
        fields = getattr(self.coarse, "force_fields", None)
        if fields is not None and fields is not self._force_fields:
            self._force_fields = fields
            self.fine.set_force_fields(fields)

    def _snapshot(self):
        frame = GuideFrame.from_solver(self.coarse, float(self.coarse.sim_time),
                                       depth=self.guide_depth)
        # The coarse frame counter at this snapshot: frame-windowed sources
        # (sinks, forces) must see the same frame in the fine solve.
        frame.frame = int(getattr(self.coarse, "frame", 0))
        frame.fired = frozenset(
            getattr(getattr(self.coarse, "sourcing", None), "_initial_fired", ()))
        return frame

    def step_frame(self):
        """Advance the fine solve over the coarse frame that just ran."""
        current = self._snapshot()
        self.guide.set_frames(self._last, current)
        # The fine clock and frame counter start where the coarse ones did
        # for this frame.
        self.fine.sim_time = float(self._last.time)
        self.fine.frame = int(getattr(self._last, "frame", 0))
        self.fine.set_colliders(fine_colliders(getattr(self.coarse, "colliders", ()),
                                               self._meshes))
        self._sync_scene()
        stats = self.fine.step_frame()
        # A full particle budget means the shell could not be filled: the
        # fine surface has holes. Say so instead of meshing them silently.
        count = self.fine.particles.refresh_host_count()
        if count >= int(self.fine.settings.max_particles) * 0.99:
            self.particle_budget_hit = True
            if not self._warned_budget:
                self._warned_budget = True
                print("[fluidsim] Guided upres ran out of fine particles "
                      f"({count:,} of {int(self.fine.settings.max_particles):,}); "
                      "the upres surface will have holes. Raise "
                      "particle_budget_scale or the coarse Max Particles.")
        # Keep filling the whole shell until the fine solve holds liquid (a
        # coarse solve that has not emitted yet gives nothing to fill).
        if count > 0:
            self.guide.bootstrap = False
        self._last = current
        return stats

    def release(self):
        self.guide.set_frames(None, None)
        self._last = None
        self._meshes = {}
        self.fine.release()
