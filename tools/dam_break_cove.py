"""Rocky-cove geometry for the dam_break demo (numpy only, no bpy).

The builder (tools/build_demo_scenes.py), the analyzer (tools/analyze_demos.py)
and the unit tests share this one definition of the scene's solids, so the
penetration check measures against exactly what the solver collides with.

x runs downstream. A reservoir between two dam abutments releases across a
flat apron into a cove: a sand beach between two rock headlands, a sea stack
in the gap and a timber jetty on the far beach. Every collider is a closed,
outward-facing mesh at least three voxels thick. Shells inside one mesh never
overlap (the solver's sign query cannot resolve that); separate collider
objects may, because their SDFs combine by min.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

VOXEL = 0.03
MIN_THICKNESS = 3 * VOXEL
SPACING = 0.015

DOMAIN_LO = (0.0, -1.00146496, 0.0)
DOMAIN_HI = (5.6, 1.00146496, 1.6)
HALF_WIDTH = DOMAIN_HI[1]
BASE_Z = -0.1  # every collider reaches below the floor

RESERVOIR_LO = (0.0, -0.6, 0.0)
RESERVOIR_HI = (0.6, 0.6, 1.2)
# The abutments stand two voxels clear of the column. Grid velocities inside
# a collider are pinned to the collider's, so fluid that starts within a
# voxel of a face is dragged to rest and hangs on it as a film.
DAM_HALF_OPENING = RESERVOIR_HI[1] + 2 * VOXEL
DAM_X = 0.75
DAM_TOP = 1.4
APRON_END_X = 2.4
APRON_THICKNESS = 0.04
WALL_THICKNESS = 0.15

# Cove heightfield.
GRID = 0.04
BEACH_TOE_X = 2.9
BEACH_TOE_CURVE = 0.4       # toe x = BEACH_TOE_X - BEACH_TOE_CURVE * y**2
BEACH_SLOPE = 0.2           # 1:5
DUNE_HEIGHT = 0.5
DUNE_BLEND = 0.1
GAP_HALF_WIDTH = 0.5
CLIFF_WIDTH = 0.12
HEADLAND_HEIGHT = 0.9
HEADLAND_FRONT_X = (2.45, 2.62)
HEADLAND_END_X = {1: 3.6, -1: 3.9}   # rear end per side (sign of y)
HEADLAND_REAR_SLOPE = 0.45
HEADLAND_WOBBLE = (0.10, 0.06, 0.20)  # front x, gap edge y, rear x (m)
ROCK_NOISE = 0.04
ROCK_BLEND = 0.15

NEAR_WALL_TOP = HEADLAND_HEIGHT
WALL_END_X = HEADLAND_FRONT_X[1] + HEADLAND_WOBBLE[0]  # headlands at full height

STACK_CENTER = (3.87, 0.13)
STACK_AXES = (0.225, 0.20)
STACK_TOP = 0.82
STACK_SIDES = 14
STACK_RING_STEP = 0.1
STACK_TOP_TAPER = 0.8
STACK_NOISE = 0.08

JETTY_X = (4.2, 5.6)
JETTY_Y = (-0.675, -0.425)
JETTY_DECK_TOP = 0.75
JETTY_DECK_THICKNESS = MIN_THICKNESS
JETTY_POST = 0.09
JETTY_POST_X = (4.3, 4.7, 5.1, 5.5)
JETTY_POST_Y = (-0.47, -0.63)
JETTY_POST_GAP = 0.003

SAND, ROCK = 0, 1


@dataclass(frozen=True)
class MeshData:
    """A closed polygon mesh; faces wind counter-clockwise seen from outside."""

    name: str
    verts: np.ndarray          # (N, 3) float64
    faces: tuple               # tuple of vertex-index tuples
    face_materials: np.ndarray  # (F,) int32 index into ``materials``
    materials: tuple
    friction: float


# ---------------------------------------------------------------------------
# Noise and height functions


def _hash(i, j, seed):
    """Deterministic integer hash of lattice coordinates -> [-1, 1]."""
    h = (np.asarray(i, dtype=np.int64) * 374761393
         + np.asarray(j, dtype=np.int64) * 668265263
         + seed * 144665) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h.astype(np.float64) / 4294967295.0) * 2.0 - 1.0


def _smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise(x, y, cell, seed):
    """Smooth 2D value noise in [-1, 1] on a square lattice of ``cell``."""
    u, v = np.asarray(x, dtype=np.float64) / cell, np.asarray(y, dtype=np.float64) / cell
    i, j = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = _smoothstep(0.0, 1.0, u - i), _smoothstep(0.0, 1.0, v - j)
    a, b = _hash(i, j, seed), _hash(i + 1, j, seed)
    c, d = _hash(i, j + 1, seed), _hash(i + 1, j + 1, seed)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def sand_height(x, y):
    """1:5 beach from a curved toe line, softly capped at the dune top."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    s = np.maximum((x - (BEACH_TOE_X - BEACH_TOE_CURVE * y * y)) * BEACH_SLOPE, 0.0)
    c, k = DUNE_HEIGHT, DUNE_BLEND
    blend = c - (c + k - s) ** 2 / (4.0 * k)
    return np.where(s <= c - k, s, np.where(s >= c + k, c, blend))


def _wobble(t, cell, seed):
    """Low-frequency noise in [0, 1] that roughens a headland outline."""
    t = np.asarray(t, dtype=np.float64)
    return 0.5 * (value_noise(t, np.zeros_like(t), cell, seed) + 1.0)


def headland_height(x, y):
    """Two rock headlands, a steep cliff down to the 1.0 m central gap.

    The outlines wobble outward only, so the gap never narrows below
    2 * GAP_HALF_WIDTH and the apron (x < HEADLAND_FRONT_X[0]) stays flat.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    side = np.where(y >= 0.0, 1, -1)
    edge_y = GAP_HALF_WIDTH + HEADLAND_WOBBLE[1] * _wobble(x + 7.0 * side, 0.25, 5)
    fy = _smoothstep(edge_y, edge_y + CLIFF_WIDTH, np.abs(y))
    front = HEADLAND_FRONT_X[0] + HEADLAND_WOBBLE[0] * _wobble(y, 0.2, 7)
    x_end = (np.where(side > 0, HEADLAND_END_X[1], HEADLAND_END_X[-1])
             + HEADLAND_WOBBLE[2] * (_wobble(y, 0.3, 9) - 0.5))
    fx = (_smoothstep(front, front + HEADLAND_FRONT_X[1] - HEADLAND_FRONT_X[0], x)
          * (1.0 - _smoothstep(x_end - HEADLAND_REAR_SLOPE, x_end, x)))
    return HEADLAND_HEIGHT * fx * fy


def rock_weight(x, y):
    """0 on sand, 1 where the headland stands clear of the beach."""
    return np.clip((headland_height(x, y) - sand_height(x, y)) / ROCK_BLEND, 0.0, 1.0)


def cove_height(x, y):
    """Top of the cove solid: beach and headlands, rock roughened by noise."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    noise = value_noise(x, y, 0.12, 11) + 0.5 * value_noise(x, y, 0.06, 23)
    rock = headland_height(x, y) + ROCK_NOISE / 1.5 * rock_weight(x, y) * noise
    return np.maximum(sand_height(x, y), rock)


# ---------------------------------------------------------------------------
# Meshes


def _box(lo, hi, offset=0):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return verts, [tuple(v + offset for v in f) for f in faces]


def _boxes(name, boxes, materials, friction):
    verts, faces = [], []
    for lo, hi in boxes:
        v, f = _box(lo, hi, len(verts))
        verts += v
        faces += f
    return MeshData(name, np.array(verts, dtype=np.float64), tuple(faces),
                    np.zeros(len(faces), dtype=np.int32), materials, friction)


def dam_boxes():
    """The two abutments, flush with the domain side walls."""
    return [((0.0, side_lo, BASE_Z), (DAM_X, side_hi, DAM_TOP))
            for side_lo, side_hi in ((-HALF_WIDTH, -DAM_HALF_OPENING),
                                     (DAM_HALF_OPENING, HALF_WIDTH))]


def jetty_boxes():
    deck_lo = JETTY_DECK_TOP - JETTY_DECK_THICKNESS
    boxes = [((JETTY_X[0], JETTY_Y[0], deck_lo), (JETTY_X[1], JETTY_Y[1], JETTY_DECK_TOP))]
    half = JETTY_POST / 2
    for px in JETTY_POST_X:
        for py in JETTY_POST_Y:
            boxes.append(((px - half, py - half, BASE_Z),
                          (px + half, py + half, deck_lo - JETTY_POST_GAP)))
    return boxes


def dam_mesh():
    return _boxes("DamBreak Dam", dam_boxes(), ("DamBreak Concrete",), 0.2)


def jetty_mesh():
    return _boxes("DamBreak Jetty", jetty_boxes(), ("DamBreak Timber",), 0.3)


def apron_mesh():
    """Visual concrete slab under the reservoir and apron; not a collider."""
    return _boxes("DamBreak Apron",
                  [((0.0, -HALF_WIDTH, -APRON_THICKNESS), (APRON_END_X, HALF_WIDTH, 0.0))],
                  ("DamBreak Concrete",), 0.0)


def wall_boxes():
    """Visual training walls just outside the domain; not colliders.

    The domain boundary already holds the water, so these only show it. They
    run past the apron to where the headlands stand at full height. The near
    (camera, -y) wall drops to headland height along the apron so the camera
    sees over it; water there peaks near 0.8 m, while on the far side it
    climbs about 1.5 m where the flow turns off the headland.
    """
    t, z0 = WALL_THICKNESS, -APRON_THICKNESS
    w, end = HALF_WIDTH, WALL_END_X
    return [
        ((-t, -w, z0), (0.0, w, DAM_TOP)),                                  # back
        ((-t, w, z0), (end, w + t, DAM_TOP)),                               # far
        ((-t, -w - t, z0), (DAM_X, -w, DAM_TOP)),                           # near, dam
        ((DAM_X, -w - t, z0), (end, -w, NEAR_WALL_TOP)),                    # near, apron
    ]


def wall_mesh():
    return _boxes("DamBreak Walls", wall_boxes(), ("DamBreak Concrete",), 0.0)


def cove_grid():
    nx = int(round((DOMAIN_HI[0] - APRON_END_X) / GRID)) + 1
    ny = int(round(2 * HALF_WIDTH / GRID)) + 1
    return (np.linspace(APRON_END_X, DOMAIN_HI[0], nx),
            np.linspace(-HALF_WIDTH, HALF_WIDTH, ny))


def cove_mesh():
    """Closed heightfield solid: triangulated top, flat bottom, four sides."""
    xs, ys = cove_grid()
    nx, ny = len(xs), len(ys)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    top = cove_height(gx, gy)
    n = nx * ny
    verts = np.concatenate([
        np.stack([gx.ravel(), gy.ravel(), top.ravel()], axis=1),
        np.stack([gx.ravel(), gy.ravel(), np.full(n, BASE_Z)], axis=1),
    ])

    def t(i, j):
        return i * ny + j

    def b(i, j):
        return n + i * ny + j

    faces, mats = [], []
    weight = rock_weight(gx, gy)

    def mat(*ids):
        return ROCK if np.mean([weight.flat[k % n] for k in ids]) > 0.5 else SAND

    for i in range(nx - 1):
        for j in range(ny - 1):
            q = (t(i, j), t(i + 1, j), t(i + 1, j + 1), t(i, j + 1))
            m = mat(*q)
            faces += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            mats += [m, m]
            faces.append((b(i, j), b(i, j + 1), b(i + 1, j + 1), b(i + 1, j)))
            mats.append(SAND)
    # Side walls: each quad drops a top boundary edge straight to the bottom.
    ring = ([(i, 0) for i in range(nx - 1)] + [(nx - 1, j) for j in range(ny - 1)]
            + [(i, ny - 1) for i in range(nx - 1, 0, -1)]
            + [(0, j) for j in range(ny - 1, 0, -1)])
    for k, (i0, j0) in enumerate(ring):
        i1, j1 = ring[(k + 1) % len(ring)]
        faces.append((t(i0, j0), b(i0, j0), b(i1, j1), t(i1, j1)))
        mats.append(mat(t(i0, j0), t(i1, j1)))
    return MeshData("DamBreak Cove", verts, tuple(faces),
                    np.array(mats, dtype=np.int32),
                    ("DamBreak Sand", "DamBreak Rock"), 0.4)


def stack_rings():
    """Ring heights and per-vertex radial scale of the sea stack."""
    zs = list(np.arange(BASE_Z, STACK_TOP - STACK_RING_STEP / 2, STACK_RING_STEP)) + [STACK_TOP]
    zs = np.array(zs, dtype=np.float64)
    taper = 1.0 - (1.0 - STACK_TOP_TAPER) * np.clip(zs / STACK_TOP, 0.0, 1.0)
    k = np.arange(len(zs))[:, None]
    s = np.arange(STACK_SIDES)[None, :]
    scale = taper[:, None] * (1.0 + STACK_NOISE * _hash(k, s, 37))
    return zs, scale


def stack_mesh():
    """Noisy tapered elliptical prism with flat caps."""
    zs, scale = stack_rings()
    theta = 2 * np.pi * np.arange(STACK_SIDES) / STACK_SIDES
    cx, cy = STACK_CENTER
    ax, ay = STACK_AXES
    verts = []
    for r, z in enumerate(zs):
        for s in range(STACK_SIDES):
            verts.append((cx + ax * scale[r, s] * np.cos(theta[s]),
                          cy + ay * scale[r, s] * np.sin(theta[s]), z))
    nr, ns = len(zs), STACK_SIDES
    bottom, top = len(verts), len(verts) + 1
    verts += [(cx, cy, zs[0]), (cx, cy, zs[-1])]
    faces = []
    for r in range(nr - 1):
        for s in range(ns):
            s1 = (s + 1) % ns
            faces.append((r * ns + s, r * ns + s1, (r + 1) * ns + s1, (r + 1) * ns + s))
    for s in range(ns):
        s1 = (s + 1) % ns
        faces.append((bottom, s1, s))
        faces.append((top, (nr - 1) * ns + s, (nr - 1) * ns + s1))
    return MeshData("DamBreak Sea Stack", np.array(verts, dtype=np.float64),
                    tuple(faces), np.zeros(len(faces), dtype=np.int32),
                    ("DamBreak Rock",), 0.5)


def collider_meshes():
    return [dam_mesh(), cove_mesh(), stack_mesh(), jetty_mesh()]


# ---------------------------------------------------------------------------
# Depth inside the solids (positive inside), for the penetration check


def _box_depth(p, lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    return np.minimum(p - lo, hi - p).min(axis=1)


@lru_cache(maxsize=1)
def _cove_top():
    xs, ys = cove_grid()
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return xs, ys, cove_height(gx, gy)


def cove_surface(x, y):
    """Height and slope of the meshed cove top, triangle for triangle.

    The solver collides with the triangulated grid from cove_mesh(), so this
    interpolates the same triangles instead of the analytic height.
    """
    xs, ys, top = _cove_top()
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    u = np.clip((x - xs[i]) / dx, 0.0, 1.0)
    v = np.clip((y - ys[j]) / dy, 0.0, 1.0)
    h00, h10 = top[i, j], top[i + 1, j]
    h01, h11 = top[i, j + 1], top[i + 1, j + 1]
    # cove_mesh() splits each quad along the (i, j)-(i+1, j+1) diagonal.
    lower = u >= v
    hx = np.where(lower, h10 - h00, h11 - h01) / dx
    hy = np.where(lower, h11 - h10, h01 - h00) / dy
    h = h00 + np.where(lower, u * (h10 - h00) + v * (h11 - h10),
                       v * (h01 - h00) + u * (h11 - h01))
    return h, hx, hy


def _cove_depth(p):
    depth = np.full(len(p), -np.inf)
    over = p[:, 0] >= APRON_END_X
    x, y, z = p[over, 0], p[over, 1], p[over, 2]
    h, hx, hy = cove_surface(x, y)
    # First-order distance to a heightfield: vertical depth times cos(slope).
    vertical = (h - z) / np.sqrt(1.0 + hx * hx + hy * hy)
    depth[over] = np.minimum(vertical, np.minimum(z - BASE_Z, x - APRON_END_X))
    return depth


def _stack_depth(p):
    zs, scale = stack_rings()
    cx, cy = STACK_CENTER
    ax, ay = STACK_AXES
    dx, dy, z = (p[:, 0] - cx) / ax, (p[:, 1] - cy) / ay, p[:, 2]
    rho = np.hypot(dx, dy)
    ang = np.mod(np.arctan2(dy, dx), 2 * np.pi) * STACK_SIDES / (2 * np.pi)
    s0 = np.floor(ang).astype(np.int64) % STACK_SIDES
    s1, fs = (s0 + 1) % STACK_SIDES, ang - np.floor(ang)
    r0 = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
    fz = np.clip((z - zs[r0]) / (zs[r0 + 1] - zs[r0]), 0.0, 1.0)

    def ring(r):
        return scale[r, s0] * (1 - fs) + scale[r, s1] * fs

    boundary = ring(r0) * (1 - fz) + ring(r0 + 1) * fz
    radial = (boundary - rho) * min(ax, ay)
    return np.minimum(radial, np.minimum(z - BASE_Z, STACK_TOP - z))


def solid_depth(points):
    """Deepest penetration of each point into any collider (m, >0 inside)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    depth = np.maximum(_cove_depth(p), _stack_depth(p))
    for lo, hi in dam_boxes() + jetty_boxes():
        depth = np.maximum(depth, _box_depth(p, lo, hi))
    return depth


# ---------------------------------------------------------------------------
# Analysis regions


def downstream_fraction(points, x_limit=HEADLAND_END_X[1]):
    """Fraction of particles that made it past the headlands."""
    p = np.asarray(points)
    return float((p[:, 0] > x_limit).mean()) if len(p) else 0.0


def beach_runup(points, *, max_height=2 * VOXEL):
    """Highest sand elevation wetted by a particle on the open beach.

    Only particles within ``max_height`` of the sand, clear of the headlands,
    the sea stack and the jetty, count as run-up.
    """
    p = np.asarray(points, dtype=np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    sand = sand_height(x, y)
    cx, cy = STACK_CENTER
    clear = ((x > APRON_END_X) & (rock_weight(x, y) == 0.0)
             & (np.hypot((x - cx) / (STACK_AXES[0] + 0.1),
                         (y - cy) / (STACK_AXES[1] + 0.1)) > 1.0)
             & ~((y > JETTY_Y[0] - 0.05) & (y < JETTY_Y[1] + 0.05)
                 & (x > JETTY_X[0] - 0.05)))
    wet = clear & (z - sand < max_height) & (sand > 0.0)
    return float(sand[wet].max()) if wet.any() else 0.0
