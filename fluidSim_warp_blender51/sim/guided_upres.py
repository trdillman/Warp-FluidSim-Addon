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
        if depth is not None:
            self.reconstruct_depth(float(depth), collision)

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

    Fine solid and sink cells stay air, as the narrow band's own union
    leaves them: the coarse level set does not resolve the finer walls.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    if collision_sdf[tid] < 0.0 or (voxel_type[tid] & C.VOXEL_SINK) != 0:
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
    ua = sample_mac_component(a_u, a_wu, a_table, g_voxels_dim, g_leaves_dim, vp, axis)
    ub = sample_mac_component(b_u, b_wu, b_table, g_voxels_dim, g_leaves_dim, vp, axis)
    return ua + (ub - ua) * alpha


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
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    demand[tid] = 0
    if cls[tid] == 2 or collision_sdf[tid] < 0.0:
        return
    if (voxel_type[tid] & C.VOXEL_SINK) != 0:
        return
    if voxel_counts[tid] > 0 or band_demand[tid] > 0:
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

    def __init__(self, shell_voxels):
        self.shell_voxels = max(1.0, float(shell_voxels))
        self.a = None
        self.b = None
        self.alpha = 0.0
        # Fill the whole coarse shell with fine particles until a frame ran.
        self.bootstrap = True
        self._demand = None
        self._budget = None

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
        particles.refresh_host_count()

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

    Particles live only in the shell, so they scale with surface area
    (``factor**2``). Leaves cover the whole coarse liquid volume, deep
    interior included (the fine level set and guide velocity live there), so
    they scale with volume (``factor**3``).
    """
    factor = int(factor)
    return float(factor * factor), float(factor * factor * factor)


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


def fine_colliders(colliders):
    """Private copies of the coarse colliders for the fine solve.

    ``FluidSolver.set_colliders`` writes into the descriptors (moving-mesh
    base points, Dynamic body state), so sharing them would let the fine solve
    change the coarse one. Dynamic (two-way) colliders are refused: the fine
    solve would integrate its own bodies and drift off the coarse motion, and
    replaying the coarse body poses is not built yet.
    """
    out = []
    for col in colliders:
        if getattr(col, "dynamic", False) and getattr(col, "mesh", None) is not None:
            raise NotImplementedError(
                "guided upres does not support Dynamic colliders yet; "
                "the fine solve cannot follow the coarse rigid bodies")
        out.append(copy.copy(col))
    return out


class GuidedUpres:
    """A fine narrow-band solver that follows a coarse ``FluidSolver``.

    Live use, once per display frame::

        upres = GuidedUpres(coarse, factor=2, shell_voxels=4)   # after coarse init
        coarse.step_frame()
        upres.step_frame()       # fine solve over the same frame
        verts, tris = upres.fine.surface_mesh()

    Baked use feeds ``step_to(frame)`` from restored coarse checkpoints in
    order instead of a live coarse solver; the guide only needs the coarse
    grid, level set and MAC velocity at each frame.

    The fine solver reuses every coarse setting except resolution, particle
    and leaf budgets, narrow-band mode (forced on, band = shell depth) and
    whitewater (off: emit it from the fine solve later if wanted).
    """

    def __init__(self, coarse, factor=2, shell_voxels=4.0, *,
                 particle_budget_scale=None, leaf_budget_scale=None):
        import dataclasses

        from .enums import SurfaceResolutionMode
        from .solver import FluidSolver

        factor = int(factor)
        if factor < 2:
            raise ValueError("guided upres needs a factor of at least 2")
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
        )
        fine_colliders(getattr(coarse, "colliders", ()))  # refuse early
        self.fine = FluidSolver(fine_settings)
        self.fine.set_colliders(fine_colliders(getattr(coarse, "colliders", ())))
        self.guide = ShellGuide(shell_voxels)
        self.fine.narrow_band.guide = self.guide
        self._sources = None
        self._force_fields = None
        self._sync_scene()
        # The guide must reach below the deepest test it makes (2 shells,
        # plus the velocity band and a coarse voxel of slack).
        self.guide_depth = (3.0 * float(shell_voxels) * float(fine_settings.voxel_size)
                            + 2.0 * float(cs.voxel_size))
        self._last = self._snapshot()
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
        return GuideFrame.from_solver(self.coarse, float(self.coarse.sim_time),
                                      depth=self.guide_depth)

    def step_frame(self):
        """Advance the fine solve over the coarse frame that just ran."""
        current = self._snapshot()
        self.guide.set_frames(self._last, current)
        # The fine clock starts where the coarse one did for this frame.
        self.fine.sim_time = float(self._last.time)
        self.fine.set_colliders(fine_colliders(getattr(self.coarse, "colliders", ())))
        self._sync_scene()
        stats = self.fine.step_frame()
        # Keep filling the whole shell until the fine solve holds liquid (a
        # coarse solve that has not emitted yet gives nothing to fill).
        if self.fine.particles.refresh_host_count() > 0:
            self.guide.bootstrap = False
        self._last = current
        return stats

    def release(self):
        self.guide.set_frames(None, None)
        self._last = None
        self.fine.release()
