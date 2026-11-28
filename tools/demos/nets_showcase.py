"""Nets Showcase: a high-resolution triple splash for Surface Nets (numpy only, no bpy).

Three drops of different sizes fall into a long shallow marble trough one after
the other, left to right. The first makes a crown, the second lands in its
ripples and the third in the wreckage of both, with the crown sheets meeting in
the middle. It is built to show the Surface Nets render surface
(``FLUIDSIM_SURFACE_BACKEND=nets``): thin crown sheets, fingers and drops are
where an even triangle mesh reads best.

Resolution is the point of the scene. ``HERO`` runs the simulation and the
render surface at 1 mm voxels, 560 cells along the long axis of the domain; the
highest-resolution demo scene before it, Upres Cove, has 280 cells along its
longest axis at 20 mm (``BASELINE``). ``DRAFT`` is the same scene at 2 mm, a
cheap way to check the build and the render before the hero run.

Layout rules that come from the Surface Nets review (see
``docs/NETS_SHOWCASE.md``):

* No liquid touches a domain wall. The mesh stops half a voxel short of the
  extraction box edge, which would open a visible gap. The pool sits on a
  collider slab that covers the whole floor of the domain, and the trough wall
  keeps the pool away from the sides.
* Every drop is at least ``MIN_DROP_VOXELS`` voxels across at the hero
  resolution, so the "small droplets shrink" effect of Surface Nets is small
  next to the drop itself. Spray finer than that will shrink a little.
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
NARROW_BAND_VOXELS = 3.0

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
    HERO: Preset(HERO, voxel=0.001, leaf_capacity=32768,
                 max_particles=4_500_000, max_dense_dim=576),
    DRAFT: Preset(DRAFT, voxel=0.002, leaf_capacity=8192,
                  max_particles=1_200_000, max_dense_dim=288),
}


def preset(name=None):
    """The named preset, else ``$NETS_SHOWCASE_PRESET``, else ``hero``."""
    name = (name or os.environ.get(PRESET_ENV) or HERO).strip().lower()
    if name not in PRESETS:
        raise ValueError(f"unknown nets_showcase preset {name!r}; use one of {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# Layout (metres). x is the long axis; the trough is centred in the domain.

DOMAIN_HX, DOMAIN_HY = 0.28, 0.17
DOMAIN_LO = (-DOMAIN_HX, -DOMAIN_HY, 0.0)
DOMAIN_HI = (DOMAIN_HX, DOMAIN_HY, 0.44)

SLAB_TOP = 0.03          # the floor the pool sits on, well above the domain floor
SLAB_BELOW = 0.05        # collider reaches below the domain floor
SLAB_OVERHANG = 0.02     # and past the sides, so no liquid can reach the walls' edge
POOL_HX, POOL_HY = 0.22, 0.10   # inner half sizes of the trough
POOL_DEPTH = 0.05
POOL_TOP = SLAB_TOP + POOL_DEPTH
WALL_THICKNESS = 0.014
WALL_TOP = POOL_TOP + 0.03

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


# Biggest first, left to right. Impacts at about frames 5, 9 and 14 (liquid
# time per frame is TIME_SCALE / FPS = 12.5 ms).
DROPS = (
    Drop("Nets Drop A", (-0.130, 0.000), 0.028, 3.0, 0.045),
    Drop("Nets Drop B", (0.000, 0.015), 0.020, 2.0, 0.100),
    Drop("Nets Drop C", (0.130, -0.020), 0.016, 1.0, 0.160),
)

# Camera for a 16:9 frame: a little above the trough rim, looking along its side.
CAMERA_LOCATION = (0.0, -1.05, 0.36)
CAMERA_TARGET = (0.0, 0.0, 0.20)
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
    """Initial-fill particles with Narrow Band on (a shell estimate).

    The pool keeps ``NARROW_BAND_VOXELS`` of particles under its top face and
    each drop keeps the same under its skin. The builder reports the real
    count; this is for sizing the reserve and for the tests.
    """
    p = p or preset()
    band = NARROW_BAND_VOXELS * p.voxel
    per_voxel = PARTICLES_PER_VOXEL / p.voxel ** 3
    pool = pool_area() * band
    drops = sum(4.0 * math.pi * d.radius ** 2 * band for d in DROPS)
    return (pool + drops) * per_voxel


def full_fill_particles(p=None):
    """What the same fill would cost without Narrow Band."""
    p = p or preset()
    volume = pool_volume() + sum(4.0 / 3.0 * math.pi * d.radius ** 3 for d in DROPS)
    return volume / p.voxel ** 3 * PARTICLES_PER_VOXEL


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
    hx, hy = DOMAIN_HX + SLAB_OVERHANG, DOMAIN_HY + SLAB_OVERHANG
    return _boxes("Nets Slab", [((-hx, -hy, -SLAB_BELOW), (hx, hy, SLAB_TOP))], "Nets Slab")


def wall_boxes():
    """(side walls, end walls) of the trough."""
    t = WALL_THICKNESS
    z0, z1 = SLAB_TOP - OVERLAP, WALL_TOP
    ox, oy = POOL_HX + t, POOL_HY + t
    sides = [((-ox, -oy, z0), (ox, -POOL_HY, z1)),
             ((-ox, POOL_HY, z0), (ox, oy, z1))]
    ends = [((-ox, -POOL_HY - OVERLAP, z0), (-POOL_HX, POOL_HY + OVERLAP, z1)),
            ((POOL_HX, -POOL_HY - OVERLAP, z0), (ox, POOL_HY + OVERLAP, z1))]
    return sides, ends


def trough_wall_meshes():
    sides, ends = wall_boxes()
    return [_boxes("Nets Trough Sides", sides, "Nets Marble"),
            _boxes("Nets Trough Ends", ends, "Nets Marble")]


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
    """Share of particles inside the trough wall's footprint (should be zero)."""
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0:
        return 0.0
    margin = 0.002
    ox, oy = POOL_HX + WALL_THICKNESS, POOL_HY + WALL_THICKNESS
    in_outer = (np.abs(pos[:, 0]) < ox - margin) & (np.abs(pos[:, 1]) < oy - margin)
    in_inner = (np.abs(pos[:, 0]) < POOL_HX + margin) & (np.abs(pos[:, 1]) < POOL_HY + margin)
    in_height = (pos[:, 2] < WALL_TOP - margin) & (pos[:, 2] > SLAB_TOP + margin)
    return float(np.mean(in_outer & ~in_inner & in_height))
