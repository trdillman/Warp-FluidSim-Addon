"""Tsunami-through-a-Parthenon geometry for the tsunami_parthenon demo (numpy only, no bpy).

The builder (tools/demos/build_demo_scenes.py) and the unit tests share this one
definition of the scene's solids and water, so the checks measure against exactly
what the solver collides with.

x runs downstream (the wave comes from x = 0). A tall slab of water stands at the
upstream end with a shallow foreshore in front of it; when the demo starts the
slab collapses into a tsunami bore. It runs up a ramp onto a terrace and into a
Parthenon-style temple at roughly 1:15 scale: three steps, a Doric colonnade of
8 x 17 columns around the long axis, a hollow cella with a door facing the wave,
an entablature ring and a low gabled roof.

Every collider is a closed, outward-facing mesh at least three voxels thick.
Shells inside one mesh never overlap (the solver's sign query cannot resolve
that); separate collider objects may, because their SDFs combine by min. Parts
that rest on each other (columns on the steps, the roof on the entablature) are
therefore separate objects that overlap by a hair instead of just touching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VOXEL = 0.05
MIN_THICKNESS = 3 * VOXEL
BASE_Z = -0.2   # every collider reaches below the floor
OVERLAP = 0.01  # resting parts sink this far into what they rest on

DOMAIN_LO = (0.0, -1.55, 0.0)
DOMAIN_HI = (9.7, 1.55, 3.4)

# The wave: a slab at the upstream end plus a shallow foreshore in front of it.
WAVE_LO = (0.0, -1.5, 0.0)
WAVE_HI = (1.0, 1.5, 2.4)
FORESHORE_LO = (1.0, -1.5, 0.0)
FORESHORE_HI = (2.4, 1.5, 0.2)
PARTICLES_PER_VOXEL = 8.0
MAX_PARTICLES = 800_000

# Ground: a ramp from the sea floor up to a flat terrace that carries the temple.
RAMP_START_X = 2.4
RAMP_END_X = 3.2
TERRACE_Z = 0.30

# Temple, laid out about its plan centre. The long axis runs along x.
TEMPLE_CENTER = (6.45, 0.0)
COLUMNS_X = 17           # along the wave's direction of travel
COLUMNS_Y = 8            # across it: the 8-column Parthenon front
COLUMN_SPACING = 0.32
COLUMN_SIDES = 12
COLUMN_R_BOTTOM = 0.092
COLUMN_R_TOP = 0.078     # across flats 0.1507 m, just over three voxels
COLUMN_HEIGHT = 1.00

STEP_RISE = 0.06
STEP_MARGIN = 0.10       # tread beyond the colonnade, then beyond each step
STEP_HALF_X = (2.75, 2.85, 2.95)   # top step first
STEP_HALF_Y = (1.31, 1.41, 1.55)   # the lowest step spans the domain

ENTABLATURE_BAND = 0.26
ENTABLATURE_HEIGHT = 0.24
ROOF_EAVE = 0.15
ROOF_PITCH_DEG = 13.0
ROOF_OVERHANG = 0.06

CELLA_HALF = (1.9, 0.62)
CELLA_WALL = MIN_THICKNESS
CELLA_DOOR_HALF = 0.20   # door in the wave-facing (low x) wall

FRAMES = 96
FPS = 24


@dataclass(frozen=True)
class MeshData:
    """A closed polygon mesh; faces wind counter-clockwise seen from outside."""

    name: str
    verts: np.ndarray          # (N, 3) float64
    faces: tuple               # tuple of vertex-index tuples
    material: str
    friction: float


# ---------------------------------------------------------------------------
# Derived heights


def step_tops():
    """Top height of each step, highest (innermost) first."""
    low = TERRACE_Z + STEP_RISE
    return (low + 2 * STEP_RISE, low + STEP_RISE, low)


STYLOBATE_Z = step_tops()[0]
COLUMN_BASE_Z = STYLOBATE_Z - OVERLAP
COLUMN_TOP_Z = COLUMN_BASE_Z + COLUMN_HEIGHT
ENTABLATURE_Z0 = COLUMN_TOP_Z - OVERLAP
ENTABLATURE_Z1 = ENTABLATURE_Z0 + ENTABLATURE_HEIGHT
ROOF_Z0 = ENTABLATURE_Z1 - OVERLAP
# The roof overhangs the entablature by the same amount again.
ROOF_HALF_Y = (COLUMNS_Y - 1) * COLUMN_SPACING / 2 + COLUMN_R_BOTTOM \
    + 2 * ROOF_OVERHANG
ROOF_HALF_X = (COLUMNS_X - 1) * COLUMN_SPACING / 2 + COLUMN_R_BOTTOM \
    + 2 * ROOF_OVERHANG
ROOF_RIDGE_Z = ROOF_Z0 + ROOF_EAVE + ROOF_HALF_Y * math.tan(math.radians(ROOF_PITCH_DEG))
TEMPLE_TOP_Z = ROOF_RIDGE_Z
TEMPLE_FRONT_X = TEMPLE_CENTER[0] - STEP_HALF_X[2]
TEMPLE_REAR_X = TEMPLE_CENTER[0] + STEP_HALF_X[2]


def column_centers():
    """Peristyle: the ring of columns around the cella, as an (N, 2) array."""
    points = []
    for i in range(COLUMNS_X):
        for j in range(COLUMNS_Y):
            if i in (0, COLUMNS_X - 1) or j in (0, COLUMNS_Y - 1):
                points.append((
                    TEMPLE_CENTER[0] + (i - (COLUMNS_X - 1) / 2) * COLUMN_SPACING,
                    TEMPLE_CENTER[1] + (j - (COLUMNS_Y - 1) / 2) * COLUMN_SPACING))
    return np.array(points, dtype=np.float64)


def expected_particles():
    """Particles the two source boxes will emit (Initial Fill)."""
    volume = sum(float(np.prod(np.subtract(hi, lo)))
                 for lo, hi in ((WAVE_LO, WAVE_HI), (FORESHORE_LO, FORESHORE_HI)))
    return volume / VOXEL ** 3 * PARTICLES_PER_VOXEL


# ---------------------------------------------------------------------------
# Mesh building


def _signed_volume(verts, faces):
    vol = 0.0
    for face in faces:
        a = verts[face[0]]
        for k in range(1, len(face) - 1):
            vol += np.dot(a, np.cross(verts[face[k]], verts[face[k + 1]])) / 6.0
    return vol


def _outward(verts, faces):
    """Flip every face when the mesh came out inside-out."""
    if _signed_volume(verts, faces) < 0.0:
        faces = [tuple(reversed(face)) for face in faces]
    return tuple(tuple(int(v) for v in face) for face in faces)


def _triangulate(poly):
    """Ear-clip a simple counter-clockwise polygon into index triangles."""
    idx = list(range(len(poly)))
    tris = []

    def cross(o, a, b):
        return ((poly[a][0] - poly[o][0]) * (poly[b][1] - poly[o][1])
                - (poly[a][1] - poly[o][1]) * (poly[b][0] - poly[o][0]))

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    while len(idx) > 3:
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(inside(p, a, b, c) for p in idx if p not in (a, b, c)):
                continue
            tris.append((a, b, c))
            del idx[k]
            break
        else:
            raise ValueError("polygon is not simple and counter-clockwise")
    tris.append(tuple(idx))
    return tris


def extrude(name, poly, lo, hi, to_3d, material, friction):
    """Prism of a counter-clockwise 2D polygon between ``lo`` and ``hi``.

    ``to_3d(u, v, w)`` places a plane point at depth ``w`` along the extrusion.
    """
    n = len(poly)
    verts = np.array([to_3d(u, v, w) for w in (lo, hi) for u, v in poly],
                     dtype=np.float64)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    for a, b, c in _triangulate(poly):
        faces.append((a, c, b))                 # low cap
        faces.append((n + a, n + b, n + c))     # high cap
    return MeshData(name, verts, _outward(verts, faces), material, friction)


def box_mesh(name, lo, hi, material, friction=0.6):
    poly = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    return extrude(name, poly, lo[2], hi[2], lambda u, v, w: (u, v, w),
                   material, friction)


def _frustum_shell(center, z0, z1, r0, r1, sides):
    cx, cy = center
    angles = [2.0 * math.pi * s / sides for s in range(sides)]
    ring0 = [(cx + r0 * math.cos(a), cy + r0 * math.sin(a), z0) for a in angles]
    ring1 = [(cx + r1 * math.cos(a), cy + r1 * math.sin(a), z1) for a in angles]
    return ring0 + ring1


def columns_mesh():
    """Every peristyle column as one mesh of separate, non-touching shells."""
    verts, faces = [], []
    n = COLUMN_SIDES
    for center in column_centers():
        base = len(verts)
        verts.extend(_frustum_shell(
            center, COLUMN_BASE_Z, COLUMN_TOP_Z, COLUMN_R_BOTTOM, COLUMN_R_TOP, n))
        for s in range(n):
            t = (s + 1) % n
            faces.append((base + s, base + t, base + n + t, base + n + s))
        faces.append(tuple(base + s for s in reversed(range(n))))
        faces.append(tuple(base + n + s for s in range(n)))
    verts = np.array(verts, dtype=np.float64)
    # Faces were written for one orientation; check each shell on its own.
    fixed = []
    per = 2 * n
    for k in range(len(column_centers())):
        shell = [f for f in faces if f[0] // per == k]
        if _signed_volume(verts, shell) < 0.0:
            shell = [tuple(reversed(f)) for f in shell]
        fixed.extend(shell)
    return MeshData("Temple Columns", verts,
                    tuple(tuple(int(v) for v in f) for f in fixed), "marble", 0.5)


def terrace_mesh():
    """Ramp from the sea floor up to the terrace, extruded across the domain."""
    profile = [
        (RAMP_START_X, BASE_Z), (DOMAIN_HI[0], BASE_Z),
        (DOMAIN_HI[0], TERRACE_Z), (RAMP_END_X, TERRACE_Z),
        (RAMP_START_X, 0.0),
    ]
    return extrude("Tsunami Terrace", profile, DOMAIN_LO[1], DOMAIN_HI[1],
                   lambda x, z, y: (x, y, z), "stone", 0.7)


def step_meshes():
    cx, cy = TEMPLE_CENTER
    meshes = []
    for tier, (top, hx, hy) in enumerate(
            zip(step_tops(), STEP_HALF_X, STEP_HALF_Y), start=1):
        lo = (cx - hx, max(cy - hy, DOMAIN_LO[1]), BASE_Z)
        hi = (cx + hx, min(cy + hy, DOMAIN_HI[1]), top)
        meshes.append(box_mesh(f"Temple Step {tier}", lo, hi, "marble", 0.6))
    return meshes


def entablature_mesh():
    """Rectangular ring of beams resting on top of the colonnade."""
    cx, cy = TEMPLE_CENTER
    hx = (COLUMNS_X - 1) * COLUMN_SPACING / 2 + COLUMN_R_BOTTOM + ROOF_OVERHANG
    hy = (COLUMNS_Y - 1) * COLUMN_SPACING / 2 + COLUMN_R_BOTTOM + ROOF_OVERHANG
    outer = [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
    ix, iy = hx - ENTABLATURE_BAND, hy - ENTABLATURE_BAND
    inner = [(cx - ix, cy - iy), (cx + ix, cy - iy), (cx + ix, cy + iy), (cx - ix, cy + iy)]
    z0, z1 = ENTABLATURE_Z0, ENTABLATURE_Z1
    verts = np.array([(x, y, z) for z in (z0, z1) for x, y in outer + inner],
                     dtype=np.float64)
    o_lo, i_lo, o_hi, i_hi = 0, 4, 8, 12
    faces = []
    for k in range(4):
        m = (k + 1) % 4
        faces.append((o_lo + k, o_lo + m, o_hi + m, o_hi + k))      # outer wall
        faces.append((i_lo + m, i_lo + k, i_hi + k, i_hi + m))      # inner wall
        faces.append((o_hi + k, o_hi + m, i_hi + m, i_hi + k))      # top
        faces.append((o_lo + m, o_lo + k, i_lo + k, i_lo + m))      # bottom
    return MeshData("Temple Entablature", verts, _outward(verts, faces), "marble", 0.5)


def roof_mesh():
    """Low gabled roof, ridge along x. The eaves are a flat three voxels thick."""
    cx, cy = TEMPLE_CENTER
    profile = [
        (cy - ROOF_HALF_Y, ROOF_Z0), (cy + ROOF_HALF_Y, ROOF_Z0),
        (cy + ROOF_HALF_Y, ROOF_Z0 + ROOF_EAVE), (cy, ROOF_RIDGE_Z),
        (cy - ROOF_HALF_Y, ROOF_Z0 + ROOF_EAVE),
    ]
    return extrude("Temple Roof", profile, cx - ROOF_HALF_X, cx + ROOF_HALF_X,
                   lambda y, z, x: (x, y, z), "terracotta", 0.6)


def cella_mesh():
    """Hollow inner sanctuary: one C-shaped wall plan with a door facing the wave."""
    cx, cy = TEMPLE_CENTER
    hx, hy = CELLA_HALF
    t, d = CELLA_WALL, CELLA_DOOR_HALF
    plan = [
        (-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, d), (-hx + t, d),
        (-hx + t, hy - t), (hx - t, hy - t), (hx - t, -hy + t),
        (-hx + t, -hy + t), (-hx + t, -d), (-hx, -d),
    ]
    poly = [(cx + x, cy + y) for x, y in plan]
    return extrude("Temple Cella", poly, STYLOBATE_Z - OVERLAP, ENTABLATURE_Z1,
                   lambda u, v, w: (u, v, w), "marble", 0.5)


def collider_meshes():
    return [terrace_mesh(), *step_meshes(), columns_mesh(), cella_mesh(),
            entablature_mesh(), roof_mesh()]


MATERIAL_COLORS = {
    "marble": (0.88, 0.85, 0.78, 1.0),
    "stone": (0.52, 0.49, 0.44, 1.0),
    "terracotta": (0.62, 0.36, 0.25, 1.0),
}


# ---------------------------------------------------------------------------
# Analysis helpers (what the builder checks while it runs the scene)


def front_x(points, quantile=0.995):
    """How far downstream the wave has reached (high percentile of particle x)."""
    p = np.asarray(points)
    return float(np.quantile(p[:, 0], quantile)) if len(p) else 0.0


def fraction_past(points, x_limit):
    """Fraction of particles downstream of ``x_limit``."""
    p = np.asarray(points)
    return float((p[:, 0] > x_limit).mean()) if len(p) else 0.0


def colonnade_fill(points):
    """Particles inside the temple footprint below the roofline (water in the temple)."""
    p = np.asarray(points, dtype=np.float64)
    if not len(p):
        return 0
    cx, cy = TEMPLE_CENTER
    inside = ((np.abs(p[:, 0] - cx) < STEP_HALF_X[0])
              & (np.abs(p[:, 1] - cy) < STEP_HALF_Y[0])
              & (p[:, 2] > STYLOBATE_Z) & (p[:, 2] < COLUMN_TOP_Z))
    return int(inside.sum())
