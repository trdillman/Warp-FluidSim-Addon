"""Anisotropic particle surfacing for the render surface (surface detail).

A coarse liquid has few particles per voxel, and the default splat draws each
one as a round ball. Flat water then shows ball bumps, thin sheets break into
strings of beads, and edges round off. Following Yu and Turk, "Reconstructing
Surfaces of Particle-Based Fluids Using Anisotropic Kernels" (ACM TOG 2013),
each particle is instead drawn as an ellipsoid shaped by how its neighbours
spread out:

* The weighted covariance of the neighbours gives the ellipsoid's axes. A
  particle in a sheet gets a flat disc; one in a jet gets a needle; one in
  bulk liquid stays round.
* Each centre is pulled toward its neighbours' weighted mean ("flatten"),
  which is what removes ball bumps from calm surfaces.
* The stretch is capped and the ellipsoid keeps the ball's volume, so the
  liquid neither swells nor thins overall. A particle with too few neighbours
  (an isolated drop) stays a round ball.

Only the render surface uses this. The solver SDF that surface tension,
whitewater and effects read is never built from ellipsoids, so turning the
feature on cannot change the simulation.

The module also holds the splash-preserving smooth: a per-voxel weight from
local particle density that lowers the smoothing strength where particles are
sparse (splashes, sheets, drops), in the spirit of Loeschner et al., "Weighted
Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids"
(VMV 2023). Calm, dense liquid is smoothed as before.
"""

from __future__ import annotations

import math

import warp as wp

from . import constants as C
from .fastlaunch import launch as fast_launch
from .fields import (
    voxel_coord_of_flat,
    voxel_flat_index,
    voxel_center_world,
    world_to_voxel,
)
from .registry import register_fn

wp.set_module_options({"enable_backward": False})

#: Neighbour count below which a particle keeps the round ball. Isolated drops
#: and sparse spray have too few neighbours for a stable covariance.
MIN_NEIGHBORS = 6

def ellipsoid_axis_scales(sd, max_stretch):
    """Host mirror of the kernel's axis rule, for tests.

    ``sd`` are the neighbour spread (standard deviations) along each principal
    axis. Returns per-axis radius multipliers of the ball radius: the stretch
    is capped at ``max_stretch`` and the product is 1, so volume is kept.
    """
    smax = max(max(float(v) for v in sd), 1.0e-12)
    floor = smax / max(1.0, float(max_stretch))
    clamped = [max(float(v), floor) for v in sd]
    norm = (clamped[0] * clamped[1] * clamped[2]) ** (1.0 / 3.0)
    return [v / norm for v in clamped]


def max_axis_multiplier(max_stretch):
    """Largest radius multiplier any ellipsoid axis can reach."""
    return max(1.0, float(max_stretch)) ** (2.0 / 3.0)


@wp.kernel
def k_aniso_particle_shape(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    neighbor_radius: float,
    max_stretch: float,
    flatten: float,
    min_neighbors: int,
    particle_radius: float,
    out_center: wp.array(dtype=wp.vec3),
    out_shape: wp.array(dtype=wp.mat33),
    out_rmax: wp.array(dtype=float),
):
    """Per-particle ellipsoid: centre, inverse-axis matrix and largest radius.

    ``out_shape`` is ``G = Q diag(1/a) Q^T`` with ``a`` the ellipsoid radii,
    so ``|G (x - c)| = 1`` on the ellipsoid surface.
    """
    tid = wp.tid()
    n_live = count[0]
    if tid >= n_live:
        return
    p = positions[tid]
    r = neighbor_radius

    # Pass 1: weighted mean of neighbours (weight 1 - (d/r)^3, Yu and Turk).
    sw = float(0.0)
    mean = wp.vec3(0.0, 0.0, 0.0)
    n = int(0)
    index = int(0)
    query = wp.hash_grid_query(grid, p, r)
    while wp.hash_grid_query_next(query, index):
        if index >= n_live:
            continue
        q = positions[index]
        d = wp.length(q - p)
        if d < r:
            t = d / r
            w = 1.0 - t * t * t
            sw += w
            mean += w * q
            n += 1

    if n < min_neighbors or sw <= 0.0:
        out_center[tid] = p
        inv = 1.0 / particle_radius
        out_shape[tid] = wp.mat33(inv, 0.0, 0.0, 0.0, inv, 0.0, 0.0, 0.0, inv)
        out_rmax[tid] = particle_radius
        return
    mean = mean / sw

    # Pass 2: weighted covariance about that mean.
    cov = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    query = wp.hash_grid_query(grid, p, r)
    while wp.hash_grid_query_next(query, index):
        if index >= n_live:
            continue
        q = positions[index]
        d = wp.length(q - p)
        if d < r:
            t = d / r
            w = 1.0 - t * t * t
            e = q - mean
            cov += w * wp.outer(e, e)
    cov = cov / sw

    Q, ev = wp.eig3(cov)
    s0 = wp.sqrt(wp.max(ev[0], 0.0))
    s1 = wp.sqrt(wp.max(ev[1], 0.0))
    s2 = wp.sqrt(wp.max(ev[2], 0.0))
    smax = wp.max(s0, wp.max(s1, s2))
    if smax <= 1.0e-12:
        out_center[tid] = p
        inv = 1.0 / particle_radius
        out_shape[tid] = wp.mat33(inv, 0.0, 0.0, 0.0, inv, 0.0, 0.0, 0.0, inv)
        out_rmax[tid] = particle_radius
        return
    floor = smax / wp.max(max_stretch, 1.0)
    s0 = wp.max(s0, floor)
    s1 = wp.max(s1, floor)
    s2 = wp.max(s2, floor)
    norm = wp.pow(s0 * s1 * s2, 1.0 / 3.0)
    a0 = particle_radius * s0 / norm
    a1 = particle_radius * s1 / norm
    a2 = particle_radius * s2 / norm

    inv_a = wp.mat33(1.0 / a0, 0.0, 0.0, 0.0, 1.0 / a1, 0.0, 0.0, 0.0, 1.0 / a2)
    out_shape[tid] = Q * inv_a * wp.transpose(Q)
    out_rmax[tid] = wp.max(a0, wp.max(a1, a2))
    lam = wp.clamp(flatten, 0.0, 1.0)
    out_center[tid] = p + lam * (mean - p)


register_fn(k_aniso_particle_shape, "fluidSimAnisoParticleShape",
            notes="Neighbour-covariance ellipsoid per particle (Yu-Turk).")


@wp.kernel
def k_build_surface_sdf_aniso(
    centers: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=wp.mat33),
    rmax: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    splat_radius_voxels: int,
    max_distance: float,
    sdf: wp.array(dtype=float),
):
    """Union-of-ellipsoids splat with atomic min.

    The distance to an ellipsoid has no closed form; this uses the distance
    along the ray from the centre, ``|v| (1 - 1/|G v|)``. It is exact for a
    ball, has the right sign everywhere, and is zero exactly on the
    ellipsoid, so the zero level set (the mesh) is the true union.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    c = centers[tid]
    G = shapes[tid]
    a = rmax[tid]
    vp = world_to_voxel(c, origin, voxel_size)
    ci = int(wp.floor(vp[0]))
    cj = int(wp.floor(vp[1]))
    ck = int(wp.floor(vp[2]))
    r = wp.min(splat_radius_voxels,
               int(wp.ceil((a + max_distance) / voxel_size)) + 1)
    reach = a + max_distance
    for dk in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                ijk = wp.vec3i(ci + di, cj + dj, ck + dk)
                cw = voxel_center_world(ijk, origin, voxel_size)
                v = cw - c
                dist = wp.length(v)
                if dist >= reach:
                    continue
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                if flat < 0:
                    continue
                L = wp.length(G * v)
                d = float(0.0)
                if L > 1.0e-6:
                    d = dist - dist / L
                else:
                    d = -a
                wp.atomic_min(sdf, flat, d)


register_fn(k_build_surface_sdf_aniso, "fluidSimBuildSurfaceSdfAniso",
            notes="Union-of-ellipsoids particle splat with atomic min distance.")


@wp.kernel
def k_density_reset(
    num_leaves: wp.array(dtype=wp.int32),
    density: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    density[tid] = 0.0


@wp.kernel
def k_density_count(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    density: wp.array(dtype=float),
):
    """Count particles in their simulation voxel."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    ijk = wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2])))
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat >= 0:
        wp.atomic_add(density, flat, 1.0)


@wp.kernel
def k_density_to_smooth_weight(
    density: wp.array(dtype=float),
    sim_table: wp.array(dtype=wp.int32),
    sim_voxels_dim: wp.vec3i,
    sim_leaves_dim: wp.vec3i,
    sim_origin: wp.vec3,
    sim_voxel_size: float,
    render_coords: wp.array(dtype=wp.vec3i),
    render_num_leaves: wp.array(dtype=wp.int32),
    render_origin: wp.vec3,
    render_voxel_size: float,
    dense_count: float,
    preserve: float,
    weight: wp.array(dtype=float),
):
    """Smoothing weight from the same 27 simulation voxels at any render resolution.

    ``dense_count`` is the count of calm liquid at the surface (about
    half full). At or above it the weight is 1 (smooth as before); with no
    particles nearby it falls to ``1 - preserve``.
    """
    tid = wp.tid()
    if tid >= render_num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(render_coords, tid)
    point = voxel_center_world(ijk, render_origin, render_voxel_size)
    vp = world_to_voxel(point, sim_origin, sim_voxel_size)
    cell = wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2])))
    total = float(0.0)
    for dk in range(-1, 2):
        for dj in range(-1, 2):
            for di in range(-1, 2):
                n = cell + wp.vec3i(di, dj, dk)
                flat = voxel_flat_index(sim_table, sim_voxels_dim, sim_leaves_dim, n)
                if flat >= 0:
                    total += density[flat]
    t = wp.clamp(total / wp.max(dense_count, 1.0e-6), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    weight[tid] = 1.0 - wp.clamp(preserve, 0.0, 1.0) * (1.0 - t)


@wp.kernel
def k_blend_by_weight(
    before: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    after: wp.array(dtype=float),
):
    """``after = before + (after - before) * weight`` in place on ``after``."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    b = before[tid]
    after[tid] = b + (after[tid] - b) * weight[tid]


class AnisotropicSurfacer:
    """Owns the per-particle ellipsoid buffers and the hash grid.

    Buffers are sized lazily to the particle capacity and kept for the
    session. ``release`` drops them when the feature is switched off.
    """

    def __init__(self, device):
        self.device = device
        self._capacity = 0
        self._grid = None
        self.centers = None
        self.shapes = None
        self.rmax = None
        self.density = None
        self.weight = None
        self.last_stats = {}

    def release(self):
        self.release_shape_buffers()
        self.release_smoothing_buffers()

    def release_shape_buffers(self):
        self._capacity = 0
        self._grid = None
        self.centers = self.shapes = self.rmax = None

    def release_smoothing_buffers(self):
        self.density = self.weight = None

    def _ensure(self, capacity):
        capacity = max(1, int(capacity))
        if self._capacity >= capacity and self._grid is not None:
            return
        dev = self.device
        self.centers = wp.zeros(capacity, dtype=wp.vec3, device=dev)
        self.shapes = wp.zeros(capacity, dtype=wp.mat33, device=dev)
        self.rmax = wp.zeros(capacity, dtype=float, device=dev)
        if self._grid is None:
            dim = C.SHARP_SURFACE_HASH_GRID_DIM
            self._grid = wp.HashGrid(dim, dim, dim, device=dev)
        self._capacity = capacity

    def splat(self, grid, sdf, positions, count, np_dim, *, voxel_size,
              sim_voxel_size, particle_radius, neighbor_radius_voxels,
              max_stretch, flatten, max_distance):
        """Write the union of particle ellipsoids into ``sdf`` (min only).

        The caller resets ``sdf`` first, exactly as for the ball splat.
        ``neighbor_radius_voxels`` is in simulation voxels so the shape of a
        particle does not change with the render resolution.
        """
        dev = self.device
        h = float(voxel_size)
        np_dim = max(1, int(np_dim))
        self._ensure(positions.shape[0])
        nr = max(1.0e-6, float(neighbor_radius_voxels) * float(sim_voxel_size))
        self._grid.build(positions[:np_dim], nr)
        fast_launch(
            k_aniso_particle_shape, dim=np_dim,
            inputs=[self._grid.id, positions, count, nr, float(max_stretch),
                    float(flatten), int(MIN_NEIGHBORS), float(particle_radius)],
            outputs=[self.centers, self.shapes, self.rmax], device=dev)
        reach = particle_radius * max_axis_multiplier(max_stretch)
        splat = max(1, int(math.ceil((reach + max_distance) / h)) + 1)
        fast_launch(
            k_build_surface_sdf_aniso, dim=np_dim,
            inputs=[self.centers, self.shapes, self.rmax, count,
                    grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                    grid.origin, h, int(splat), float(max_distance)],
            outputs=[sdf.data], device=dev)
        self.last_stats = {
            "neighborRadius": nr,
            "maxStretch": float(max_stretch),
            "flatten": float(flatten),
            "splatRadiusVoxels": int(splat),
        }

    def smoothing_weight(self, grid, sim_grid, nvox, positions, count, np_dim, *,
                         particles_per_voxel, preserve):
        """Per-voxel smoothing weight; 1 where dense, lower where sparse."""
        dev = self.device
        size = max(1, int(nvox))
        sim_size = max(1, int(sim_grid.active_voxel_count))
        if self.density is None or self.density.shape[0] < sim_size:
            self.density = wp.zeros(sim_size, dtype=float, device=dev)
        if self.weight is None or self.weight.shape[0] < size:
            self.weight = wp.zeros(size, dtype=float, device=dev)
        fast_launch(k_density_reset, dim=sim_size,
                    inputs=[sim_grid.num_leaves], outputs=[self.density],
                    device=dev)
        fast_launch(k_density_count, dim=max(1, int(np_dim)),
                    inputs=[positions, count, sim_grid.leaf_table,
                            sim_grid.voxels_dim, sim_grid.leaves_dim, sim_grid.origin,
                            float(sim_grid.voxel_size)],
                    outputs=[self.density], device=dev)
        # Calm surface: 27 simulation voxels, about half of them liquid.
        dense = 13.5 * float(particles_per_voxel)
        fast_launch(k_density_to_smooth_weight, dim=size,
                    inputs=[self.density, sim_grid.leaf_table, sim_grid.voxels_dim,
                            sim_grid.leaves_dim, sim_grid.origin, float(sim_grid.voxel_size),
                            grid.leaf_coords, grid.num_leaves, grid.origin, float(grid.voxel_size),
                            dense, float(preserve)],
                    outputs=[self.weight], device=dev)
        return self.weight
