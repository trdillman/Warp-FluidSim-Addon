"""Sheet Detail: fine particles where the coarse liquid is a thin sheet.

Why: on Dam Break the full-resolution run differs from the coarse one mostly
in surface *structure* (torn, lace-like sheets with many small holes and
detached fragments) while the liquid sits in nearly the same place. Surface
tricks on the coarse particles cannot make that structure; sheets tear
because there are enough particles inside them to pull apart.

How (render-only, after particle upsampling for splashing liquids: Roy,
Paquette and Poulin, "Particle upsampling as a flexible post-processing
approach to increase details in animations of splashing liquids", Computers &
Graphics 88, 2020; thin-sheet detection by neighbour density as in Extended
Narrow Band FLIP, Sato et al. 2018):

1. A coarse particle is *thin* when it has few neighbours within two
   simulation voxels compared with full liquid (a sheet or a drop).
2. Every thin particle that has no fine particle near it spawns ``split``
   fine particles in a small ball around it. Each fine particle has the
   volume of ``1/split`` of a coarse one.
3. Fine particles persist. Each frame they move with the coarse grid velocity
   where there is liquid and fall ballistically where there is none, so a
   sheet that the coarse flow stretches pulls them apart and opens holes the
   coarse particles are too big to show.
4. A fine particle dies when it is back inside full liquid (no longer a
   sheet), when no coarse particle is near it any more, or when it is older
   than ``max_age``.
5. The render surface splats non-thin coarse particles as usual, then the
   fine particles at their smaller radius, before the surface filters.

Nothing here changes the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .fields import sample_mac_velocity, voxel_flat_index, world_to_voxel

HASH_DIM = 128


@dataclass
class SheetDetailSettings:
    split: int = 8                # fine particles per thin coarse particle
    thin_fraction: float = 0.35   # thin below this share of full-liquid neighbours
    max_age: float = 2.0          # seconds a fine particle may live
    max_particles: int = 4_000_000
    seed: int = 1


def fine_radius(coarse_radius, split):
    """Radius of a fine particle with 1/split of a coarse one's volume."""
    return float(coarse_radius) * max(1, int(split)) ** (-1.0 / 3.0)


def full_neighbour_count(particles_per_voxel, radius_voxels):
    """Neighbours a particle has inside full liquid, within the given radius."""
    return float(particles_per_voxel) * (4.0 / 3.0) * math.pi * float(radius_voxels) ** 3


# --- kernels ------------------------------------------------------------------


@wp.kernel
def k_sd_density(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    radius: float,
    out: wp.array(dtype=float),
):
    i = wp.tid()
    if i >= count[0]:
        return
    x = pos[i]
    n = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    query = wp.hash_grid_query(grid, x, radius)
    j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
    while wp.hash_grid_query_next(query, j):
        if j < count[0] and j != i and wp.length(pos[j] - x) < radius:
            n += 1.0
    out[i] = n


@wp.kernel
def k_sd_thin(
    density: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    limit: float,
    thin: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= count[0]:
        thin[i] = 0
        return
    if density[i] < limit:
        thin[i] = 1
    else:
        thin[i] = 0


@wp.kernel
def k_sd_advect(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    origin: wp.vec3,
    voxel_size: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    gravity: wp.vec3,
    dt: float,
    substeps: int,
):
    i = wp.tid()
    x = pos[i]
    vi = vel[i]
    h = dt / float(substeps)
    for _s in range(substeps):
        vp = world_to_voxel(x, origin, voxel_size)
        cell = wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2])))
        if voxel_flat_index(leaf_table, voxels_dim, leaves_dim, cell) >= 0:
            # Inside the coarse liquid's grid: follow its flow (PIC).
            vi = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table,
                                     voxels_dim, leaves_dim, vp)
        else:
            vi = vi + h * gravity
        x = x + h * vi
    pos[i] = x
    vel[i] = vi
    age[i] = age[i] + dt


@wp.kernel
def k_sd_keep(
    grid: wp.uint64,
    ppos: wp.array(dtype=wp.vec3),
    pcount: wp.array(dtype=wp.int32),
    cpos: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    radius: float,
    full_limit: float,
    lost_distance: float,
    max_age: float,
    keep: wp.array(dtype=wp.int32),
):
    c = wp.tid()
    x = cpos[c]
    keep[c] = 0
    if age[c] > max_age:
        return
    n = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    nearest = float(1.0e30)  # noqa: UP018 - Warp needs a dynamic variable here
    query = wp.hash_grid_query(grid, x, radius)
    j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
    while wp.hash_grid_query_next(query, j):
        if j < pcount[0]:
            d = wp.length(ppos[j] - x)
            if d < radius:
                n += 1.0
            nearest = wp.min(nearest, d)
    if nearest > lost_distance:
        return  # the coarse liquid left it behind
    if n >= full_limit:
        return  # back inside full liquid: the coarse surface covers it
    keep[c] = 1


@wp.kernel
def k_sd_spawn_demand(
    cgrid: wp.uint64,
    cpos: wp.array(dtype=wp.vec3),
    ccount: int,
    ppos: wp.array(dtype=wp.vec3),
    thin: wp.array(dtype=wp.int32),
    pcount: wp.array(dtype=wp.int32),
    cover: float,
    split: int,
    demand: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    demand[i] = 0
    if i >= pcount[0] or thin[i] == 0:
        return
    x = ppos[i]
    if ccount > 0:
        query = wp.hash_grid_query(cgrid, x, cover)
        j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
        while wp.hash_grid_query_next(query, j):
            if j < ccount and wp.length(cpos[j] - x) < cover:
                return  # already detailed
    demand[i] = split


@wp.kernel
def k_sd_spawn(
    ppos: wp.array(dtype=wp.vec3),
    pvel: wp.array(dtype=wp.vec3),
    demand: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    base: int,
    limit: int,
    jitter: float,
    seed: int,
    cpos: wp.array(dtype=wp.vec3),
    cvel: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
):
    i = wp.tid()
    n = demand[i]
    if n == 0:
        return
    state = wp.rand_init(seed, i)
    for k in range(n):
        slot = base + offsets[i] + k
        if slot >= limit:
            return
        # Uniform in a ball of radius ``jitter`` around the parent.
        d = wp.vec3(wp.randf(state) * 2.0 - 1.0, wp.randf(state) * 2.0 - 1.0,
                    wp.randf(state) * 2.0 - 1.0)
        while wp.dot(d, d) > 1.0:
            d = wp.vec3(wp.randf(state) * 2.0 - 1.0, wp.randf(state) * 2.0 - 1.0,
                        wp.randf(state) * 2.0 - 1.0)
        cpos[slot] = ppos[i] + jitter * d
        cvel[slot] = pvel[i]
        age[slot] = 0.0


@wp.kernel
def k_sd_gather_children(
    keep: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
    age_out: wp.array(dtype=float),
):
    i = wp.tid()
    if keep[i] != 0:
        slot = offsets[i]
        pos_out[slot] = pos[i]
        vel_out[slot] = vel[i]
        age_out[slot] = age[i]


@wp.kernel
def k_sd_gather_coarse(
    thin: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i < count[0] and thin[i] == 0:
        out[offsets[i]] = pos[i]


@wp.kernel
def k_sd_render_keep(thin: wp.array(dtype=wp.int32), uncovered: wp.array(dtype=wp.int32),
                     count: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    # A thin coarse particle is left out of the render splat only where fine
    # particles stand in for it; one with none (budget reached) still splats.
    i = wp.tid()
    if i < count[0] and (thin[i] == 0 or uncovered[i] != 0):
        out[i] = 1
    else:
        out[i] = 0


def _scan(flags, device):
    """Exclusive prefix sum of an int32 array; returns (offsets, total)."""
    offsets = wp.empty_like(flags)
    if flags.shape[0] == 0:
        return offsets, 0
    wp.utils.array_scan(flags, offsets, inclusive=False)
    total = int(offsets[-1:].numpy()[0] + flags[-1:].numpy()[0])
    return offsets, total


class SheetDetail:
    """Fine particles carried across frames in the thin parts of a liquid.

    ``step`` once per frame with the coarse particles and the coarse MAC
    velocity; ``render_inputs`` then gives the coarse particles to splat (the
    thin ones removed) and the fine particles with their radius.
    """

    def __init__(self, settings: SheetDetailSettings | None = None, *, device=None):
        self.settings = settings or SheetDetailSettings()
        self.device = wp.get_device(device)
        self.frame = 0
        self.stats = {}
        self._pgrid = wp.HashGrid(HASH_DIM, HASH_DIM, HASH_DIM, device=self.device)
        self._cgrid = wp.HashGrid(HASH_DIM, HASH_DIM, HASH_DIM, device=self.device)
        self.thin = None
        self.coarse_count = 0
        self.voxel_size = 0.0
        self._empty()

    def _empty(self):
        dev = self.device
        self.pos = wp.empty(0, dtype=wp.vec3, device=dev)
        self.vel = wp.empty(0, dtype=wp.vec3, device=dev)
        self.age = wp.empty(0, dtype=float, device=dev)
        self.count = 0

    def reset(self):
        self.frame = 0
        self.stats = {}
        self.thin = None
        self.coarse_count = 0
        self._empty()

    # --- per frame -------------------------------------------------------

    def classify(self, ppos, pcount, np_dim, *, voxel_size, particles_per_voxel):
        """Flag thin coarse particles; returns the device ``thin`` array."""
        s, dev = self.settings, self.device
        n = max(1, int(np_dim))
        h = float(voxel_size)
        self.voxel_size = h
        radius = 2.0 * h
        self._pgrid.build(ppos[:n], radius)
        density = wp.zeros(n, dtype=float, device=dev)
        wp.launch(k_sd_density, dim=n, inputs=[self._pgrid.id, ppos, pcount, radius, density],
                  device=dev)
        full = full_neighbour_count(particles_per_voxel, 2.0)
        self.thin = wp.zeros(n, dtype=wp.int32, device=dev)
        wp.launch(k_sd_thin, dim=n,
                  inputs=[density, pcount, float(s.thin_fraction) * full, self.thin],
                  device=dev)
        return self.thin

    def step(self, ppos, pvel, pcount, np_dim, *, grid, vel, voxel_size,
             particles_per_voxel, gravity, dt):
        s, dev = self.settings, self.device
        n = max(1, int(np_dim))
        h = float(voxel_size)
        radius = 2.0 * h
        full = full_neighbour_count(particles_per_voxel, 2.0)
        self.classify(ppos, pcount, n, voxel_size=h,
                      particles_per_voxel=particles_per_voxel)
        seed = int(s.seed) * 1000003 + self.frame

        # 1. Move the fine particles, then drop the ones that are done.
        m = self.count
        kept = 0
        if m:
            substeps = 2
            wp.launch(k_sd_advect, dim=m,
                      inputs=[self.pos, self.vel, self.age, grid.origin, h,
                              vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww,
                              grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                              wp.vec3(*[float(g) for g in gravity]), float(dt),
                              substeps], device=dev)
            keep = wp.zeros(m, dtype=wp.int32, device=dev)
            # Hysteresis: a fine particle lives on until the liquid round it
            # is clearly full again, a bit above the thin limit.
            wp.launch(k_sd_keep, dim=m,
                      inputs=[self._pgrid.id, ppos, pcount, self.pos, self.age, radius,
                              min(1.0, 1.5 * float(s.thin_fraction)) * full, 2.0 * h,
                              float(s.max_age), keep], device=dev)
            offsets, kept = _scan(keep, dev)
            pos = wp.empty(kept, dtype=wp.vec3, device=dev)
            vel_out = wp.empty(kept, dtype=wp.vec3, device=dev)
            age = wp.empty(kept, dtype=float, device=dev)
            if kept:
                wp.launch(k_sd_gather_children, dim=m,
                          inputs=[keep, offsets, self.pos, self.vel, self.age,
                                  pos, vel_out, age], device=dev)
            self.pos, self.vel, self.age, self.count = pos, vel_out, age, kept

        # 2. Thin coarse particles with no fine particle near them spawn some.
        cover = 0.75 * h
        if self.count:
            self._cgrid.build(self.pos, cover)
        demand = wp.zeros(n, dtype=wp.int32, device=dev)
        wp.launch(k_sd_spawn_demand, dim=n,
                  inputs=[self._cgrid.id, self.pos, int(self.count), ppos, self.thin,
                          pcount, cover, max(1, int(s.split)), demand], device=dev)
        offsets, want = _scan(demand, dev)
        limit = max(0, int(s.max_particles))
        add = max(0, min(want, limit - self.count))
        if add:
            total = self.count + add
            pos = wp.empty(total, dtype=wp.vec3, device=dev)
            vel_out = wp.empty(total, dtype=wp.vec3, device=dev)
            age = wp.empty(total, dtype=float, device=dev)
            if self.count:
                wp.copy(pos, self.pos, count=self.count)
                wp.copy(vel_out, self.vel, count=self.count)
                wp.copy(age, self.age, count=self.count)
            wp.launch(k_sd_spawn, dim=n,
                      inputs=[ppos, pvel, demand, offsets, self.count, total, 0.5 * h,
                              seed, pos, vel_out, age], device=dev)
            self.pos, self.vel, self.age, self.count = pos, vel_out, age, total
        self.frame += 1
        self.stats = {"fine_particles": int(self.count), "kept": int(kept),
                      "added": int(add), "budget_hit": bool(want > add)}
        return self.count

    def render_inputs(self, ppos, pcount, np_dim):
        """Coarse particles to splat and their count array.

        Thin particles that fine particles stand in for are removed.
        """
        dev = self.device
        n = max(1, int(np_dim))
        if self.thin is None or self.thin.shape[0] < n or self.voxel_size <= 0.0:
            self.coarse_count = -1
            return ppos, pcount, n
        cover = 0.75 * self.voxel_size
        if self.count:
            self._cgrid.build(self.pos, cover)
        uncovered = wp.zeros(n, dtype=wp.int32, device=dev)
        wp.launch(k_sd_spawn_demand, dim=n,
                  inputs=[self._cgrid.id, self.pos, int(self.count), ppos, self.thin,
                          pcount, cover, 1, uncovered], device=dev)
        flags = wp.zeros(n, dtype=wp.int32, device=dev)
        wp.launch(k_sd_render_keep, dim=n, inputs=[self.thin, uncovered, pcount, flags],
                  device=dev)
        offsets, total = _scan(flags, dev)
        out = wp.empty(max(1, total), dtype=wp.vec3, device=dev)
        if total:
            wp.launch(k_sd_gather_coarse, dim=n,
                      inputs=[self.thin, pcount, offsets, ppos, out], device=dev)
        count = wp.array(np.array([total], np.int32), dtype=wp.int32, device=dev)
        self.coarse_count = total
        return out, count, max(1, total)

    # --- checkpoints -----------------------------------------------------

    def export_state(self):
        """Host copy of the fine particles: ``(arrays, scalars)``."""
        m = int(self.count)
        arrays = {
            "pos": np.ascontiguousarray(self.pos.numpy()[:m].astype(np.float32).reshape(m, 3)),
            "vel": np.ascontiguousarray(self.vel.numpy()[:m].astype(np.float32).reshape(m, 3)),
            "age": np.ascontiguousarray(self.age.numpy()[:m].astype(np.float32).reshape(m)),
        }
        return arrays, {"frame": int(self.frame)}

    def import_state(self, arrays, scalars):
        pos = np.asarray(arrays["pos"], dtype=np.float32)
        m = int(pos.shape[0]) if pos.ndim == 2 else 0
        vel = np.asarray(arrays["vel"], dtype=np.float32)
        age = np.asarray(arrays["age"], dtype=np.float32)
        if pos.shape != (m, 3) or vel.shape != (m, 3) or age.shape != (m,):
            raise ValueError("sheet detail state has the wrong shape")
        dev = self.device
        self.pos = wp.array(pos, dtype=wp.vec3, device=dev)
        self.vel = wp.array(vel, dtype=wp.vec3, device=dev)
        self.age = wp.array(age, dtype=float, device=dev)
        self.count = m
        self.frame = int(scalars.get("frame", 0))
        self.thin = None
        self.stats = {}

    def fine_inputs(self):
        """Fine particle positions, a device count and the launch size."""
        dev = self.device
        count = wp.array(np.array([self.count], np.int32), dtype=wp.int32, device=dev)
        return self.pos, count, int(self.count)
