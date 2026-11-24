"""Surface waves: fine wave detail on a coarse liquid surface, no second sim.

After Mercier, Beauchemin, Thuerey, Kim and Nowrouzezahrai, "Surface
Turbulence for Particle-Based Liquid Simulations" (SIGGRAPH Asia 2015), in
the line of Kim, Thuerey and Tessendorf, "Closest Point Turbulence for Liquid
Surfaces" (TOG 2013). Each frame:

1. Wave points, an even layer of points finer than the coarse surface mesh,
   move with the coarse surface velocity and are projected back onto the new
   coarse mesh. Points that lose the surface die.
2. The layer is regularized: points closer than half the spacing to an
   older point die, and triangles left too sparse get new points.
3. Waves are seeded where the coarse surface curves more sharply than its
   grid resolves, with a per-point phase so they oscillate instead of
   repeating the coarse shape.
4. A damped wave equation on the points (Laplacian from tangent-plane
   neighbours) carries the waves across the surface.
5. The render mesh is displaced along its normals by the wave height.

The cost follows the surface area, not the liquid volume, and the input is a
plain triangle mesh with per-vertex velocity, so any surfacing backend
(sparse, dense, Surface Nets) and any scene works. All state lives on the
device; the host only sees the displaced vertices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

GRAVITY = 9.81
WATER_SURFACE_TENSION = 0.072  # N/m
WATER_DENSITY = 1000.0


@dataclass
class SurfaceWaveSettings:
    """Authored controls; lengths in metres, times in seconds."""

    spacing: float = 0.01          # wave point spacing (sets finest wavelength)
    strength: float = 1.0          # seeding gain
    damping: float = 1.5           # 1/s amplitude decay
    max_amplitude: float = 0.5     # cap, in multiples of ``coarse_voxel``
    coarse_voxel: float = 0.02     # resolution of the surface being detailed
    max_points: int = 4_000_000
    seed: int = 1


def wave_speed(wavelength):
    """Phase speed of capillary-gravity water waves of ``wavelength`` (deep water)."""
    k = 2.0 * math.pi / max(float(wavelength), 1e-6)
    return math.sqrt(GRAVITY / k + WATER_SURFACE_TENSION * k / WATER_DENSITY)


# Thinning keeps points at least half a spacing apart, and the Laplacian
# clamps tangent distances to that, so one neighbour at that distance is the
# stiffest case: two points swinging against each other have squared
# frequency 2 * 4 * c^2 / (0.5 * spacing)^2 = 32 c^2 / spacing^2.
MIN_SEPARATION = 0.5
STIFFEST_MODE = math.sqrt(8.0) / MIN_SEPARATION


def substep_count(dt, spacing, speed, damping=0.0, cfl=1.0):
    """Explicit wave substeps for one frame (at least one).

    Keeps ``omega_max * sub_dt <= cfl`` for the stiffest mode the point
    Laplacian can produce (symplectic Euler is stable below 2, so 1 leaves
    room for damping), and ``damping * sub_dt <= 0.5``.
    """
    if dt <= 0.0:
        return 0
    omega_max = STIFFEST_MODE * float(speed) / float(spacing)
    waves = math.ceil(float(dt) * omega_max / cfl)
    damp = math.ceil(float(dt) * max(float(damping), 0.0) / 0.5)
    return max(1, int(waves), int(damp))


# Mesh helpers


@wp.kernel
def k_wave_vertex_normals(
    verts: wp.array(dtype=wp.vec3),
    tris: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    a = tris[3 * t]
    b = tris[3 * t + 1]
    c = tris[3 * t + 2]
    n = wp.cross(verts[b] - verts[a], verts[c] - verts[a])  # area weighted
    wp.atomic_add(normals, a, n)
    wp.atomic_add(normals, b, n)
    wp.atomic_add(normals, c, n)


@wp.kernel
def k_wave_normalize(normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    n = normals[i]
    length = wp.length(n)
    if length > 1.0e-20:
        normals[i] = n / length
    else:
        normals[i] = wp.vec3(0.0, 0.0, 1.0)


@wp.func
def _bary(values: wp.array(dtype=wp.vec3), tris: wp.array(dtype=wp.int32),
          face: int, u: float, v: float):
    return (u * values[tris[3 * face]] + v * values[tris[3 * face + 1]]
            + (1.0 - u - v) * values[tris[3 * face + 2]])


# 1. Advect and project


@wp.kernel
def k_wave_advect_project(
    mesh: wp.uint64,
    tris: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
    max_dist: float,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    p = pos[i] + dt * vel[i]
    q = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    if not q.result:
        alive[i] = 0
        return
    x = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    pos[i] = x
    n = _bary(normals, tris, q.face, q.u, q.v)
    length = wp.length(n)
    if length > 1.0e-20:
        nrm[i] = n / length
    vel[i] = _bary(velocities, tris, q.face, q.u, q.v)
    alive[i] = 1


# 2. Regularize


@wp.kernel
def k_wave_thin(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    min_dist: float,
    alive_in: wp.array(dtype=wp.int32),
    alive_out: wp.array(dtype=wp.int32),
):
    """A point dies when an older (lower index) live point is too close."""
    i = wp.tid()
    alive_out[i] = alive_in[i]
    if alive_in[i] == 0:
        return
    x = pos[i]
    query = wp.hash_grid_query(grid, x, min_dist)
    j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
    while wp.hash_grid_query_next(query, j):
        if j < i and alive_in[j] != 0 and wp.length(pos[j] - x) < min_dist:
            alive_out[i] = 0
            return


@wp.kernel
def k_wave_triangle_demand(
    verts: wp.array(dtype=wp.vec3),
    tris: wp.array(dtype=wp.int32),
    spacing: float,
    seed: int,
    demand: wp.array(dtype=wp.int32),
):
    """Candidate points per triangle: its area in spacing squares, dithered."""
    t = wp.tid()
    a = verts[tris[3 * t]]
    area = 0.5 * wp.length(wp.cross(verts[tris[3 * t + 1]] - a,
                                    verts[tris[3 * t + 2]] - a))
    want = area / (spacing * spacing)
    n = int(wp.floor(want))
    state = wp.rand_init(seed, t)
    if wp.randf(state) < want - float(n):
        n = n + 1
    demand[t] = n


@wp.kernel
def k_wave_candidates(
    verts: wp.array(dtype=wp.vec3),
    tris: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    demand: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    seed: int,
    cand_pos: wp.array(dtype=wp.vec3),
    cand_vel: wp.array(dtype=wp.vec3),
    cand_nrm: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    state = wp.rand_init(seed, t + 7919)
    for k in range(demand[t]):
        r1 = wp.sqrt(wp.randf(state))
        r2 = wp.randf(state)
        u = 1.0 - r1
        v = r1 * (1.0 - r2)
        slot = offsets[t] + k
        cand_pos[slot] = _bary(verts, tris, t, u, v)
        cand_vel[slot] = _bary(velocities, tris, t, u, v)
        n = _bary(normals, tris, t, u, v)
        length = wp.length(n)
        if length > 1.0e-20:
            n = n / length
        cand_nrm[slot] = n


@wp.kernel
def k_wave_accept_candidates(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=wp.int32),
    cand_pos: wp.array(dtype=wp.vec3),
    spacing: float,
    accept: wp.array(dtype=wp.int32),
):
    """Keep a candidate only where no live point is within the spacing."""
    c = wp.tid()
    x = cand_pos[c]
    accept[c] = 1
    query = wp.hash_grid_query(grid, x, spacing)
    j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
    while wp.hash_grid_query_next(query, j):
        if alive[j] != 0 and wp.length(pos[j] - x) < spacing:
            accept[c] = 0
            return


# 3-4. Curvature, seeding, wave equation


# Neighbours whose normals differ by more than about 60 degrees belong to
# another sheet (the far side of a thin sheet or a drop): they never count as
# the same local surface.
SAME_SHEET = 0.5


@wp.func
def _same_sheet(a: wp.vec3, b: wp.vec3):
    return wp.dot(a, b) > float(SAME_SHEET)


@wp.func
def _kernel_weight(r: float, radius: float):
    q = 1.0 - r / radius
    if q <= 0.0:
        return 0.0
    return q * q


@wp.kernel
def k_wave_curvature(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    radius: float,
    curvature: wp.array(dtype=float),
):
    """Mean curvature from neighbour heights over the tangent plane."""
    i = wp.tid()
    x = pos[i]
    n = nrm[i]
    total = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    weight = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    query = wp.hash_grid_query(grid, x, radius)
    j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
    while wp.hash_grid_query_next(query, j):
        if j != i and _same_sheet(n, nrm[j]):
            d = pos[j] - x
            r2 = wp.dot(d, d)
            if r2 > 1.0e-20:
                w = _kernel_weight(wp.sqrt(r2), radius)
                total += w * 2.0 * wp.dot(n, d) / r2
                weight += w
    if weight > 0.0:
        curvature[i] = total / weight
    else:
        curvature[i] = 0.0


@wp.kernel
def k_wave_step(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    curvature: wp.array(dtype=float),
    phase: wp.array(dtype=float),
    h_in: wp.array(dtype=float),
    hv_in: wp.array(dtype=float),
    radius: float,
    c2: float,
    damping: float,
    seed_gain: float,
    curvature_min: float,
    omega: float,
    time: float,
    dt: float,
    max_amp: float,
    min_r2: float,
    h_out: wp.array(dtype=float),
    hv_out: wp.array(dtype=float),
):
    i = wp.tid()
    x = pos[i]
    n = nrm[i]
    hi = h_in[i]
    acc = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    weight = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    query = wp.hash_grid_query(grid, x, radius)
    j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
    while wp.hash_grid_query_next(query, j):
        if j != i and _same_sheet(n, nrm[j]):
            d = pos[j] - x
            d = d - wp.dot(d, n) * n  # tangent plane distance
            # On a bend the tangent distance can be shorter than the true
            # one; clamp it to the thinning distance so the stiffest mode
            # stays the one substep_count plans for.
            r2 = wp.max(wp.dot(d, d), min_r2)
            w = _kernel_weight(wp.sqrt(r2), radius)
            if w > 0.0:
                acc += w * (h_in[j] - hi) / r2
                weight += w
    lap = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    if weight > 0.0:
        lap = 4.0 * acc / weight  # 2 * dimension, dimension 2
    # Waves start where the coarse surface bends more sharply than its grid
    # resolves; the per-point phase makes them oscillate, not copy the shape.
    excess = wp.max(wp.abs(curvature[i]) - curvature_min, 0.0)
    force = seed_gain * excess * wp.sin(omega * time + phase[i])
    hv = hv_in[i] + dt * (c2 * lap - damping * hv_in[i] + force)
    h = wp.clamp(hi + dt * hv, -max_amp, max_amp)
    h_out[i] = h
    hv_out[i] = hv


# 5. Displace the render mesh


@wp.kernel
def k_wave_displace(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    nrm: wp.array(dtype=wp.vec3),
    h: wp.array(dtype=float),
    verts: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    radius: float,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x = verts[i]
    n = normals[i]
    total = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    weight = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    query = wp.hash_grid_query(grid, x, radius)
    j = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
    while wp.hash_grid_query_next(query, j):
        if _same_sheet(n, nrm[j]):
            w = _kernel_weight(wp.length(pos[j] - x), radius)
            total += w * h[j]
            weight += w
    height = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    if weight > 0.0:
        height = total / weight
    out[i] = x + height * normals[i]


@wp.kernel
def k_wave_gather(
    keep: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    src_pos: wp.array(dtype=wp.vec3),
    src_vel: wp.array(dtype=wp.vec3),
    src_nrm: wp.array(dtype=wp.vec3),
    src_h: wp.array(dtype=float),
    src_hv: wp.array(dtype=float),
    src_phase: wp.array(dtype=float),
    limit: int,
    dst_pos: wp.array(dtype=wp.vec3),
    dst_vel: wp.array(dtype=wp.vec3),
    dst_nrm: wp.array(dtype=wp.vec3),
    dst_h: wp.array(dtype=float),
    dst_hv: wp.array(dtype=float),
    dst_phase: wp.array(dtype=float),
):
    i = wp.tid()
    if keep[i] == 0:
        return
    k = offsets[i]
    if k >= limit:
        return
    dst_pos[k] = src_pos[i]
    dst_vel[k] = src_vel[i]
    dst_nrm[k] = src_nrm[i]
    dst_h[k] = src_h[i]
    dst_hv[k] = src_hv[i]
    dst_phase[k] = src_phase[i]


@wp.kernel
def k_wave_noise_phase(seed: int, pos: wp.array(dtype=wp.vec3), inv_scale: float,
                       phase: wp.array(dtype=float)):
    """Phase from smooth noise of the birth position: neighbouring points
    oscillate together in patches about a wavelength wide, so the seeding
    makes wave fronts rather than per-point fizz."""
    i = wp.tid()
    state = wp.rand_init(seed)
    phase[i] = 6.283185307 * wp.noise(state, pos[i] * inv_scale)


def _scan(flags, device):
    """Exclusive prefix sum of an int32 array; returns (offsets, total)."""
    offsets = wp.empty_like(flags)
    if flags.shape[0] == 0:
        return offsets, 0
    wp.utils.array_scan(flags, offsets, inclusive=False)
    total = int(offsets[-1:].numpy()[0] + flags[-1:].numpy()[0])
    return offsets, total


class SurfaceWaves:
    """Wave detail carried across frames on one coarse liquid surface.

    Call :meth:`step` once per frame with that frame's coarse surface mesh,
    its per-vertex velocity and the frame time; it returns the displaced
    vertices (same count and order, so the triangles are unchanged).
    """

    def __init__(self, settings: SurfaceWaveSettings | None = None, *, device=None):
        self.settings = settings or SurfaceWaveSettings()
        self.device = wp.get_device(device)
        self.time = 0.0
        self.frame = 0
        self.count = 0
        self.stats = {}
        self._empty()

    def _empty(self):
        dev = self.device
        self.pos = wp.empty(0, dtype=wp.vec3, device=dev)
        self.vel = wp.empty(0, dtype=wp.vec3, device=dev)
        self.nrm = wp.empty(0, dtype=wp.vec3, device=dev)
        self.h = wp.empty(0, dtype=float, device=dev)
        self.hv = wp.empty(0, dtype=float, device=dev)
        self.phase = wp.empty(0, dtype=float, device=dev)
        self.count = 0

    def reset(self):
        self.time = 0.0
        self.frame = 0
        self.stats = {}
        self._empty()

    # Derived parameters

    @property
    def spacing(self):
        return max(float(self.settings.spacing), 1e-5)

    @property
    def radius(self):
        return 2.0 * self.spacing

    @property
    def speed(self):
        # The finest wave the layer holds is about four spacings long.
        return wave_speed(4.0 * self.spacing)

    def _grid(self, pos, count):
        cell = self.radius
        grid = wp.HashGrid(128, 128, 128, device=self.device)
        if count:
            grid.build(pos[:count], cell)
        return grid

    def _gather(self, keep, offsets, total):
        dev = self.device
        n = self.count
        pos = wp.empty(total, dtype=wp.vec3, device=dev)
        vel = wp.empty(total, dtype=wp.vec3, device=dev)
        nrm = wp.empty(total, dtype=wp.vec3, device=dev)
        h = wp.empty(total, dtype=float, device=dev)
        hv = wp.empty(total, dtype=float, device=dev)
        phase = wp.empty(total, dtype=float, device=dev)
        if n and total:
            wp.launch(k_wave_gather, dim=n,
                      inputs=[keep, offsets, self.pos, self.vel, self.nrm, self.h,
                              self.hv, self.phase, total,
                              pos, vel, nrm, h, hv, phase], device=dev)
        self.pos, self.vel, self.nrm = pos, vel, nrm
        self.h, self.hv, self.phase = h, hv, phase
        self.count = total

    # Frame step

    def step(self, verts, tris, velocities, dt):
        s, dev = self.settings, self.device
        verts = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
        tris = np.ascontiguousarray(tris, dtype=np.int32).reshape(-1)
        velocities = np.ascontiguousarray(velocities, dtype=np.float32).reshape(-1, 3)
        nv, nt = len(verts), len(tris) // 3
        if nv == 0 or nt == 0:
            self._empty()
            self.time += float(dt)
            self.frame += 1
            return verts.copy()

        v_dev = wp.array(verts, dtype=wp.vec3, device=dev)
        t_dev = wp.array(tris, dtype=wp.int32, device=dev)
        vel_dev = wp.array(velocities, dtype=wp.vec3, device=dev)
        normals = wp.zeros(nv, dtype=wp.vec3, device=dev)
        wp.launch(k_wave_vertex_normals, dim=nt, inputs=[v_dev, t_dev, normals],
                  device=dev)
        wp.launch(k_wave_normalize, dim=nv, inputs=[normals], device=dev)
        mesh = wp.Mesh(points=v_dev, indices=t_dev)
        seed = int(s.seed) * 1000003 + self.frame

        # 1. Advect the existing points and put them back on the surface.
        n = self.count
        alive = wp.zeros(n, dtype=wp.int32, device=dev)
        if n:
            max_dist = 2.0 * float(s.coarse_voxel) + self.radius
            wp.launch(k_wave_advect_project, dim=n,
                      inputs=[mesh.id, t_dev, normals, vel_dev, float(dt), max_dist,
                              self.pos, self.vel, self.nrm, alive], device=dev)
            grid = self._grid(self.pos, n)
            thinned = wp.zeros(n, dtype=wp.int32, device=dev)
            wp.launch(k_wave_thin, dim=n,
                      inputs=[grid.id, self.pos, MIN_SEPARATION * self.spacing, alive, thinned],
                      device=dev)
            alive = thinned
        else:
            grid = self._grid(self.pos, 0)

        # 2. Fill sparse triangles with new points.
        # The candidates never exceed the point budget: on a surface too big
        # for it, they are drawn sparser (the budget caps the points anyway).
        demand = wp.zeros(nt, dtype=wp.int32, device=dev)
        cap = max(int(s.max_points), 1)
        draw = self.spacing
        # Size the draw from the area first (float64 on the host), so the
        # int32 demand scan can never overflow on a huge surface.
        tri = verts[tris.reshape(-1, 3)].astype(np.float64)
        area = 0.5 * float(np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        if area / draw ** 2 > 0.9 * cap:
            draw = math.sqrt(area / (0.9 * cap))
        while True:
            wp.launch(k_wave_triangle_demand, dim=nt,
                      inputs=[v_dev, t_dev, draw, seed, demand], device=dev)
            offsets, nc = _scan(demand, dev)
            if nc <= cap:
                break
            draw *= 1.02 * math.sqrt(nc / cap)
        cand_pos = wp.empty(nc, dtype=wp.vec3, device=dev)
        cand_vel = wp.empty(nc, dtype=wp.vec3, device=dev)
        cand_nrm = wp.empty(nc, dtype=wp.vec3, device=dev)
        accept = wp.zeros(nc, dtype=wp.int32, device=dev)
        if nc:
            wp.launch(k_wave_candidates, dim=nt,
                      inputs=[v_dev, t_dev, normals, vel_dev, demand, offsets, seed,
                              cand_pos, cand_vel, cand_nrm], device=dev)
            wp.launch(k_wave_accept_candidates, dim=nc,
                      inputs=[grid.id, self.pos, alive, cand_pos, self.spacing, accept],
                      device=dev)

        # Compact survivors then accepted candidates into fresh arrays.
        keep_off, n_keep = _scan(alive, dev) if n else (alive, 0)
        acc_off, n_acc = _scan(accept, dev) if nc else (accept, 0)
        limit = int(s.max_points)
        total = min(n_keep + n_acc, limit)
        pos = wp.empty(total, dtype=wp.vec3, device=dev)
        vel = wp.empty(total, dtype=wp.vec3, device=dev)
        nrm = wp.empty(total, dtype=wp.vec3, device=dev)
        h = wp.zeros(total, dtype=float, device=dev)
        hv = wp.zeros(total, dtype=float, device=dev)
        phase = wp.zeros(total, dtype=float, device=dev)
        if n_keep and total:
            wp.launch(k_wave_gather, dim=n,
                      inputs=[alive, keep_off, self.pos, self.vel, self.nrm, self.h,
                              self.hv, self.phase, total,
                              pos, vel, nrm, h, hv, phase], device=dev)
        kept = min(n_keep, total)
        added = total - kept
        if added:
            zeros = wp.zeros(nc, dtype=float, device=dev)
            wp.launch(k_wave_gather, dim=nc,
                      inputs=[accept, acc_off, cand_pos, cand_vel, cand_nrm, zeros,
                              zeros, zeros, added,
                              pos[kept:], vel[kept:], nrm[kept:], h[kept:], hv[kept:],
                              phase[kept:]], device=dev)
            wp.launch(k_wave_noise_phase, dim=added,
                      inputs=[int(s.seed), pos[kept:], 1.0 / (8.0 * self.spacing),
                              phase[kept:]], device=dev)
        self.pos, self.vel, self.nrm = pos, vel, nrm
        self.h, self.hv, self.phase = h, hv, phase
        self.count = total
        if added > 1:
            # Candidates were only checked against carried points; thin them
            # against each other now (older wins), or next frame's thinning
            # would discard the clumps and look like lost points.
            grid = self._grid(self.pos, total)
            ones = wp.full(total, 1, dtype=wp.int32, device=dev)
            keep = wp.zeros(total, dtype=wp.int32, device=dev)
            wp.launch(k_wave_thin, dim=total,
                      inputs=[grid.id, self.pos, MIN_SEPARATION * self.spacing, ones, keep],
                      device=dev)
            keep_off, total = _scan(keep, dev)
            added = total - kept
            self._gather(keep, keep_off, total)

        # 3-4. Seed and propagate waves.
        steps = substep_count(dt, self.spacing, self.speed, float(s.damping))
        out = wp.empty_like(v_dev)
        if total:
            grid = self._grid(self.pos, total)
            curvature = wp.zeros(total, dtype=float, device=dev)
            wp.launch(k_wave_curvature, dim=total,
                      inputs=[grid.id, self.pos, self.nrm, self.radius, curvature],
                      device=dev)
            coarse = max(float(s.coarse_voxel), self.spacing)
            # The coarse grid resolves bends down to about two voxels' radius.
            curvature_min = 1.0 / (2.0 * coarse)
            wavelength = 4.0 * self.spacing
            omega = 2.0 * math.pi * self.speed / wavelength
            max_amp = float(s.max_amplitude) * coarse
            # Force per unit excess curvature, scaled so a bend one coarse
            # voxel sharper than resolvable drives about a quarter of a voxel.
            seed_gain = float(s.strength) * 0.25 * coarse * omega * coarse
            c2 = self.speed * self.speed
            h_tmp = wp.empty_like(self.h)
            hv_tmp = wp.empty_like(self.hv)
            sub_dt = float(dt) / max(steps, 1)
            for k in range(steps):
                wp.launch(k_wave_step, dim=total,
                          inputs=[grid.id, self.pos, self.nrm, curvature, self.phase,
                                  self.h, self.hv, self.radius, c2, float(s.damping),
                                  seed_gain, curvature_min, omega,
                                  self.time + k * sub_dt, sub_dt, max_amp,
                                  (MIN_SEPARATION * self.spacing) ** 2,
                                  h_tmp, hv_tmp], device=dev)
                self.h, h_tmp = h_tmp, self.h
                self.hv, hv_tmp = hv_tmp, self.hv

            # 5. Displace the render mesh.
            wp.launch(k_wave_displace, dim=nv,
                      inputs=[grid.id, self.pos, self.nrm, self.h, v_dev, normals,
                              self.radius,
                              out], device=dev)
        else:
            wp.copy(out, v_dev)
        self.time += float(dt)
        self.frame += 1
        self.stats = {"wave_points": total, "kept": kept, "added": added,
                      "substeps": steps}
        return out.numpy()

    def heights(self):
        return self.h.numpy()[: self.count]

    def displace(self, verts, tris):
        """Apply the current waves to a mesh without advancing them.

        For a surface re-extracted between frames (a render setting changed
        while paused): the waves stay where they are and the new mesh shows
        them.
        """
        dev = self.device
        verts = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
        tris = np.ascontiguousarray(tris, dtype=np.int32).reshape(-1)
        nv, nt = len(verts), len(tris) // 3
        if nv == 0 or nt == 0 or self.count == 0:
            return verts.copy()
        v_dev = wp.array(verts, dtype=wp.vec3, device=dev)
        t_dev = wp.array(tris, dtype=wp.int32, device=dev)
        normals = wp.zeros(nv, dtype=wp.vec3, device=dev)
        wp.launch(k_wave_vertex_normals, dim=nt, inputs=[v_dev, t_dev, normals],
                  device=dev)
        wp.launch(k_wave_normalize, dim=nv, inputs=[normals], device=dev)
        grid = self._grid(self.pos, self.count)
        out = wp.empty_like(v_dev)
        wp.launch(k_wave_displace, dim=nv,
                  inputs=[grid.id, self.pos, self.nrm, self.h, v_dev, normals,
                          self.radius, out], device=dev)
        return out.numpy()

    # Checkpoints: the layer is state, not scratch, so a resumed bake must
    # carry it on rather than start new waves.

    STATE_FIELDS = (("pos", wp.vec3, 3), ("vel", wp.vec3, 3), ("nrm", wp.vec3, 3),
                    ("h", float, 0), ("hv", float, 0), ("phase", float, 0))

    def export_state(self):
        """Host copy of the wave layer: ``(arrays, scalars)``."""
        arrays = {}
        for name, _dtype, width in self.STATE_FIELDS:
            value = getattr(self, name).numpy()[: self.count].astype(np.float32)
            arrays[name] = np.ascontiguousarray(
                value.reshape(self.count, width) if width else value.reshape(self.count))
        return arrays, {"time": float(self.time), "frame": int(self.frame)}

    def import_state(self, arrays, scalars):
        count = int(np.asarray(arrays["h"]).shape[0])
        for name, dtype, width in self.STATE_FIELDS:
            value = np.asarray(arrays[name], dtype=np.float32)
            if value.shape != ((count, width) if width else (count,)):
                raise ValueError(f"surface wave state {name!r} has the wrong shape")
            setattr(self, name, wp.array(value, dtype=dtype, device=self.device))
        self.count = count
        self.time = float(scalars["time"])
        self.frame = int(scalars["frame"])
        self.stats = {}
