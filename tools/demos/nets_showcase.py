"""Nets Showcase: a high-resolution triple splash for Surface Nets (numpy only, no bpy).

Three drops of different sizes fall into a long shallow marble trough one after
the other, left to right. The first makes a crown, the second lands in its
ripples and the third in the wreckage of both, with the crown sheets meeting in
the middle. It is built to show the Surface Nets render surface
(``FLUIDSIM_SURFACE_BACKEND=nets``): thin crown sheets, fingers and drops are
where an even triangle mesh reads best.

Resolution is the point of the scene. ``HERO`` runs the simulation and the
render surface at 1 mm voxels, 600 cells along the long axis of the domain; the
highest-resolution demo scene before it, Upres Cove, has 280 cells along its
longest axis at 20 mm (``BASELINE``). ``DRAFT`` is the same scene at 2 mm, a
cheap way to check the build and the render before the hero run.

Layout rules that come from the Surface Nets review (see
``docs/NETS_SHOWCASE.md``):

* No liquid touches a domain wall. The mesh stops half a voxel short of the
  extraction box edge, which would open a visible gap. The pool sits on a
  collider slab, and trough walls taller than the splash keep the water in,
  well away from the domain sides.
* Every drop is at least ``MIN_DROP_VOXELS`` voxels across at the hero
  resolution, so the "small droplets shrink" effect of Surface Nets is small
  next to the drop itself. Spray finer than that will shrink a little.
* Narrow Band is off and the pool is a thin film (8 mm). A source's startup
  demand is its whole volume, so a deep pool would be booked at tens of
  millions of particles before any narrow-band culling.
* Every collider books the leaves of its bounding box at startup (leaf capacity),
  so the colliders are separate thin walls on a small slab, not rings or a
  floor-sized slab.
* Analytic sources are booked as a cube of their longest side, so the pool is
  a grid of small box sources (``pool_tiles``).
* The domain is a long box, not a cube. The Surface Nets extractor works in a
  dense box, 40 bytes per voxel in the add-on's own memory estimate, so the two
  short axes are kept short to fit a 16 GB card.

The builder (``build_demo_scenes.py -- nets_showcase``), the render script
(``render_nets_showcase.py``) and the unit tests share these numbers.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

NAME = "nets_showcase"
BACKEND_ENV = "FLUIDSIM_SURFACE_BACKEND"
BACKEND = "nets"
PRESET_ENV = "NETS_SHOWCASE_PRESET"

FPS = 24
# Slow motion: one film frame is this fraction of 1/24 s of liquid time.
TIME_SCALE = 0.30
FRAMES = 144
MAX_SUBSTEPS = 32
PARTICLES_PER_VOXEL = 8.0

G = 9.81

# ---------------------------------------------------------------------------
# Presets

HERO = "hero"
DRAFT = "draft"


@dataclass(frozen=True)
class Preset:
    name: str
    voxel: float
    leaf_capacity: int
    max_particles: int
    max_dense_dim: int


PRESETS = {
    HERO: Preset(HERO, voxel=0.001, leaf_capacity=45056,
                 max_particles=8_000_000, max_dense_dim=640),
    DRAFT: Preset(DRAFT, voxel=0.002, leaf_capacity=16384,
                  max_particles=1_300_000, max_dense_dim=320),
}


def preset(name=None):
    """The named preset, else ``$NETS_SHOWCASE_PRESET``, else ``hero``."""
    name = (name or os.environ.get(PRESET_ENV) or HERO).strip().lower()
    if name not in PRESETS:
        raise ValueError(f"unknown nets_showcase preset {name!r}; use one of {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# Layout (metres). x is the long axis; the trough is centred in the domain.

DOMAIN_HX, DOMAIN_HY = 0.30, 0.14
DOMAIN_LO = (-DOMAIN_HX, -DOMAIN_HY, 0.0)
DOMAIN_HI = (DOMAIN_HX, DOMAIN_HY, 0.40)

SLAB_TOP = 0.016         # the floor the pool sits on (hero: 16 voxels above the domain floor)
SLAB_BELOW = 0.05        # collider reaches below the domain floor
SLAB_OVERHANG = 0.02     # past the outer face of the trough walls
POOL_HX, POOL_HY = 0.22, 0.10   # inner half sizes of the trough
POOL_DEPTH = 0.008      # a thin film: every pool voxel holds particles, see expected_particles
POOL_TOP = SLAB_TOP + POOL_DEPTH
WALL_THICKNESS = 0.008
# Trough wall height above the pool. The walls are taller than any spray the
# three drops throw (the first GPU runs saw 0.28 m with the walls at 0.05 m, and
# the water ran out over the slab to the domain walls). The camera-side wall is
# left out of the render (see HIDDEN_IN_RENDER): it holds the water like a pane
# of clear glass, so the camera sees the splash without a wall in the way.
TROUGH_HEIGHT = 0.30
TROUGH_TOP_FRONT = POOL_TOP + TROUGH_HEIGHT
TROUGH_TOP_BACK = POOL_TOP + TROUGH_HEIGHT
TROUGH_TOP_END = POOL_TOP + TROUGH_HEIGHT
WALL_TOP = TROUGH_TOP_BACK
HIDDEN_IN_RENDER = ("Nets Trough Front",)
MIN_DROP_VOXELS = 18


@dataclass(frozen=True)
class Drop:
    name: str
    center_xy: tuple
    radius: float
    speed: float        # downward launch speed, m/s
    impact_time: float  # seconds of liquid time until the bottom meets the pool

    @property
    def gap(self):
        """Distance from the drop's bottom to the pool surface at frame 1."""
        return self.speed * self.impact_time + 0.5 * G * self.impact_time ** 2

    @property
    def center(self):
        return (self.center_xy[0], self.center_xy[1], POOL_TOP + self.gap + self.radius)

    @property
    def top(self):
        return self.center[2] + self.radius


# Biggest first, left to right. Dropped nearly from rest (the first has a small
# push) so the impacts are gentle: 2.0, 1.9 and 2.35 m/s. A harder splash threw
# spray into the domain walls on the first GPU run. Impacts at about frames 9,
# 16 and 20 (liquid time per frame is TIME_SCALE / FPS = 12.5 ms).
DROPS = (
    Drop("Nets Drop A", (-0.130, 0.000), 0.028, 1.0, 0.102),
    Drop("Nets Drop B", (0.000, 0.015), 0.020, 0.0, 0.190),
    Drop("Nets Drop C", (0.130, -0.020), 0.016, 0.0, 0.240),
)

# Camera for a 16:9 frame: at the top of the trough walls, looking along its side.
CAMERA_LOCATION = (0.0, -1.00, 0.32)
CAMERA_TARGET = (0.0, 0.0, 0.17)
CAMERA_LENS = 45.0

WATER_COLOR = (0.55, 0.80, 0.95, 1.0)

MATERIAL_COLORS = {
    "Nets Marble": (0.86, 0.84, 0.80, 1.0),
    "Nets Slab": (0.045, 0.05, 0.06, 1.0),
}


# ---------------------------------------------------------------------------
# Numbers


def frame_time():
    """Seconds of liquid time per film frame."""
    return TIME_SCALE / FPS


def drop_fall_frame(drop):
    return 1.0 + drop.impact_time / frame_time()


def divisions(p=None):
    """Simulation (and render surface) cells along each domain axis."""
    p = p or preset()
    return tuple(round((hi - lo) / p.voxel)
                 for lo, hi in zip(DOMAIN_LO, DOMAIN_HI, strict=True))


def pool_area():
    return 4.0 * POOL_HX * POOL_HY


def pool_volume():
    return pool_area() * POOL_DEPTH


def expected_particles(p=None):
    """Particles the Initial Fill emits (pool plus drops), at 8 per voxel.

    Narrow Band is off. A source's startup demand is its full volume (the
    admission planner books it before any culling), so a deep pool would be
    budgeted at tens of millions of particles. The pool is a thin film instead,
    a few voxels deep, and every voxel of it holds particles anyway.
    """
    p = p or preset()
    return full_fill_particles(p)


def full_fill_particles(p=None):
    p = p or preset()
    volume = pool_volume() + sum(4.0 / 3.0 * math.pi * d.radius ** 3 for d in DROPS)
    return volume / p.voxel ** 3 * PARTICLES_PER_VOXEL


# The admission planner bounds every analytic source by a cube of its LARGEST
# size and compiles all of that cube's leaves. A pool 0.44 m long therefore
# books a 0.44 m cube (11,200 leaves and 46M particles at 2 mm), is refused for
# leaf capacity, and the planner falls back to the whole cube as demand. So the
# pool is cut into tiles no longer than this, each with a small cube.
MAX_TILE = 0.10


def _split(cells, longest):
    """Cell counts of the smallest equal-ish split whose pieces fit ``longest``."""
    n = max(1, math.ceil(cells / longest))
    base, extra = divmod(cells, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def pool_tiles(p=None):
    """((lo, hi), ...) corner pairs of the pool's Initial Fill box sources.

    The pool is inset by one voxel from the trough walls. Tiles meet on voxel
    faces (never on a voxel centre), so no voxel belongs to two tiles.
    """
    p = p or preset()
    v = p.voxel
    per_tile = max(1, int(MAX_TILE / v))
    x_cells = round((2.0 * POOL_HX - 2.0 * v) / v)
    y_cells = round((2.0 * POOL_HY - 2.0 * v) / v)
    x0, y0 = -POOL_HX + v, -POOL_HY + v
    tiles = []
    ix = 0
    for nx in _split(x_cells, per_tile):
        iy = 0
        for ny in _split(y_cells, per_tile):
            tiles.append(((x0 + ix * v, y0 + iy * v, SLAB_TOP),
                          (x0 + (ix + nx) * v, y0 + (iy + ny) * v, POOL_TOP)))
            iy += ny
        ix += nx
    return tuple(tiles)


def drop_voxels(drop, p=None):
    p = p or preset()
    return 2.0 * drop.radius / p.voxel


# Upres Cove (opt-in, tools/demos/upres_cove.py): the finest demo scene so far.
# The shipped demos are all coarser (dam_break 187 cells, crown_splash 80).
BASELINE = {
    "scene": "upres_cove",
    "voxel": 0.020,
    "longest_axis_cells": 280,
    "domain_cells": 280 * 100 * 80,
    "particles": 864_000,
}


def fidelity(p=None):
    """The scene's resolution against ``BASELINE``, as plain multipliers."""
    p = p or preset()
    cells = divisions(p)
    return {
        "preset": p.name,
        "voxel_mm": p.voxel * 1000.0,
        "baseline_voxel_mm": BASELINE["voxel"] * 1000.0,
        "longest_axis_cells": max(cells),
        "baseline_longest_axis_cells": BASELINE["longest_axis_cells"],
        "cells_per_axis_x": max(cells) / BASELINE["longest_axis_cells"],
        "domain_cells": int(np.prod(cells)),
        "domain_cells_x": float(np.prod(cells)) / BASELINE["domain_cells"],
        "expected_particles": expected_particles(p),
        "particles_x": expected_particles(p) / BASELINE["particles"],
    }


# ---------------------------------------------------------------------------
# Meshes. Closed, outward-facing, at least three hero voxels thick. Boxes inside
# one mesh never overlap; the end walls sink into the side walls by OVERLAP in a
# separate object instead of just touching (separate objects may overlap).

OVERLAP = 0.004


@dataclass(frozen=True)
class MeshData:
    """A closed polygon mesh; faces wind counter-clockwise seen from outside."""

    name: str
    verts: np.ndarray          # (N, 3) float64
    faces: tuple
    material: str
    friction: float


def _box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return verts, faces


def _boxes(name, boxes, material):
    verts, faces = [], []
    for lo, hi in boxes:
        v, f = _box(lo, hi)
        faces += [tuple(i + len(verts) for i in face) for face in f]
        verts += v
    return MeshData(name, np.array(verts, dtype=np.float64), tuple(faces), material, 0.0)


def slab_mesh():
    # Every collider books the leaves of its bounding box at startup, and a slab
    # over the whole domain floor was 13,000 leaves at hero resolution for no
    # purpose: the trough walls hold the water in.
    hx = POOL_HX + WALL_THICKNESS + SLAB_OVERHANG
    hy = POOL_HY + WALL_THICKNESS + SLAB_OVERHANG
    return _boxes("Nets Slab", [((-hx, -hy, -SLAB_BELOW), (hx, hy, SLAB_TOP))], "Nets Slab")


def _ring_boxes(hx, hy, thickness, z0, top_front, top_back, top_end):
    """(sides, ends) of a rectangular ring with inner half sizes hx, hy.

    The end boxes sink into the side boxes by OVERLAP, so they go in their own
    mesh.
    """
    ox, oy = hx + thickness, hy + thickness
    sides = [((-ox, -oy, z0), (ox, -hy, top_front)),
             ((-ox, hy, z0), (ox, oy, top_back))]
    ends = [((-ox, -hy - OVERLAP, z0), (-hx, hy + OVERLAP, top_end)),
            ((hx, -hy - OVERLAP, z0), (ox, hy + OVERLAP, top_end))]
    return sides, ends


def wall_boxes():
    """(side walls, end walls) of the trough."""
    return _ring_boxes(POOL_HX, POOL_HY, WALL_THICKNESS, SLAB_TOP - OVERLAP,
                       TROUGH_TOP_FRONT, TROUGH_TOP_BACK, TROUGH_TOP_END)


def _wall_meshes(prefix, boxes, material):
    """One mesh per wall: each collider books the leaves of its own bounding box,
    so a mesh holding both ends of a ring would book the whole ring's interior."""
    (front, back), (left, right) = boxes
    return [_boxes(f"{prefix} {name}", [box], material)
            for name, box in zip(("Front", "Back", "Left", "Right"),
                                 (front, back, left, right), strict=True)]


def trough_wall_meshes():
    return _wall_meshes("Nets Trough", wall_boxes(), "Nets Marble")


def collider_meshes():
    return [slab_mesh(), *trough_wall_meshes()]


def signed_volume(mesh):
    """Positive when the faces wind counter-clockwise seen from outside."""
    v = mesh.verts
    vol = 0.0
    for face in mesh.faces:
        for k in range(1, len(face) - 1):
            a, b, c = v[face[0]], v[face[k]], v[face[k + 1]]
            vol += float(np.dot(a, np.cross(b, c))) / 6.0
    return vol


# ---------------------------------------------------------------------------
# Checks on particle positions (for the build and the render run)


def free_surface_height(positions, quantile=0.5):
    """Median height of the particles over the middle of the pool."""
    pos = np.asarray(positions, dtype=np.float64)
    inside = (np.abs(pos[:, 0]) < POOL_HX * 0.8) & (np.abs(pos[:, 1]) < POOL_HY * 0.8)
    if not inside.any():
        return 0.0
    return float(np.quantile(pos[inside, 2], quantile))


def wall_contact_fraction(positions, distance):
    """Share of particles within ``distance`` of a domain side wall or the ceiling."""
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0:
        return 0.0
    lo = np.array(DOMAIN_LO)
    hi = np.array(DOMAIN_HI)
    near = ((pos[:, :2] - lo[:2] < distance) | (hi[:2] - pos[:, :2] < distance)).any(axis=1)
    near |= hi[2] - pos[:, 2] < distance
    return float(np.mean(near))


# Stray spray will reach a wall in any hard splash and costs a speck of mesh.
# What must not happen is the body of the liquid meeting a wall.
MAX_WALL_CONTACT_FRACTION = 1.0e-3


def wall_clearance(positions):
    """Smallest distance (m) from any particle to a domain wall (floor excluded)."""
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0:
        return float("inf")
    lo = np.array(DOMAIN_LO)
    hi = np.array(DOMAIN_HI)
    sides = np.minimum(pos[:, :2] - lo[:2], hi[:2] - pos[:, :2]).min()
    top = (hi[2] - pos[:, 2]).min()
    return float(min(sides, top))


def inside_wall_fraction(positions):
    """Share of particles inside the trough walls (should be zero)."""
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0:
        return 0.0
    margin = 0.002
    ox, oy = POOL_HX + WALL_THICKNESS, POOL_HY + WALL_THICKNESS
    in_outer = (np.abs(pos[:, 0]) < ox - margin) & (np.abs(pos[:, 1]) < oy - margin)
    in_inner = (np.abs(pos[:, 0]) < POOL_HX + margin) & (np.abs(pos[:, 1]) < POOL_HY + margin)
    top = np.where(pos[:, 1] < 0.0, TROUGH_TOP_FRONT, TROUGH_TOP_BACK)
    in_height = (pos[:, 2] < top - margin) & (pos[:, 2] > SLAB_TOP + margin)
    inside = in_outer & ~in_inner & in_height
    return float(np.mean(inside))
