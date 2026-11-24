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
  eroded by the shell depth (new liquid appears where the coarse sim
  advances; the narrow-band reseed fills its shell with fine particles), and
  deep fine liquid may not drift more than one more shell outside the coarse
  surface.
* ``k_guide_mark_leaves``: fine leaves are allocated wherever the coarse
  liquid is, so the union above has somewhere to land.

A ``GuideFrame`` is a self-contained copy of one coarse frame's level set and
MAC velocity together with the coarse leaf table, so the coarse solver may
rebuild its topology or step on while the fine solver still samples it.
"""

from __future__ import annotations

import warp as wp

from . import constants as C
from .fastlaunch import launch as fast_launch
from .fields import (
    leaf_coord_of_voxel,
    leaf_linear_index,
    sample_mac_component,
    sample_scalar_trilinear,
    voxel_coord_of_flat,
    voxel_floor,
    voxel_in_domain,
    world_to_voxel,
)
from .registry import register_fn

wp.set_module_options({"enable_backward": False})

#: Value a guide level-set sample takes outside the coarse active leaves.
GUIDE_FAR = 1.0e6


class GuideFrame:
    """One coarse frame, copied so it outlives the coarse solver's buffers."""

    def __init__(self, grid, sdf, vel, time):
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

    @classmethod
    def from_solver(cls, solver, time):
        """Snapshot a coarse solver after its frame (its SDF made current)."""
        if not solver.surface.ensure_solver_sdf_current():
            raise RuntimeError(solver.surface.solver_sdf_error)
        sdf = solver.surface.sdf
        narrow_band = getattr(solver, "narrow_band", None)
        if narrow_band is not None:
            narrow_band.union_into_solver_sdf(sdf)
        return cls(solver.grid, sdf, solver.vel, time)


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


@wp.kernel
def k_guide_phi(
    phi: wp.array(dtype=float),
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
    """Keep the fine level set within the coarse liquid's shell."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                             float(ijk[2]) + 0.5)
    ga = _guide_phi_at(a_sdf, a_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    gb = _guide_phi_at(b_sdf, b_table, g_voxels_dim, g_leaves_dim, g_origin, g_h, x)
    if ga >= GUIDE_FAR * 0.5 or gb >= GUIDE_FAR * 0.5:
        return
    g = ga + (gb - ga) * alpha
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

    def apply_phi(self, grid, phi, nvox, limit, device):
        if not self.ready:
            return
        a, b, common = self._guide_inputs()
        h = float(grid.voxel_size)
        fast_launch(
            k_guide_phi, dim=max(1, int(nvox)),
            inputs=[phi, grid.leaf_coords, grid.num_leaves, grid.origin, h,
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
        # The shell holds particles over an area ~factor^2 larger at the fine
        # spacing, in a band ``shell_voxels`` deep: budgets scale with area.
        p_scale = particle_budget_scale or float(factor * factor)
        l_scale = leaf_budget_scale or float(factor * factor)
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
        self.fine = FluidSolver(fine_settings)
        self.fine.set_colliders(list(getattr(coarse, "colliders", ())))
        self.guide = ShellGuide(shell_voxels)
        self.fine.narrow_band.guide = self.guide
        self._last = GuideFrame.from_solver(coarse, float(coarse.sim_time))
        self.guide.set_frames(self._last, None)

    def step_frame(self):
        """Advance the fine solve over the coarse frame that just ran."""
        current = GuideFrame.from_solver(self.coarse, float(self.coarse.sim_time))
        self.guide.set_frames(self._last, current)
        # The fine clock starts where the coarse one did for this frame.
        self.fine.sim_time = float(self._last.time)
        self.fine.set_colliders(list(getattr(self.coarse, "colliders", ())))
        stats = self.fine.step_frame()
        self._last = current
        return stats

    def release(self):
        self.guide.set_frames(None, None)
        self._last = None
        self.fine.release()
