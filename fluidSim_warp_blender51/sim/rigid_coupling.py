"""Two-way rigid-body coupling for Dynamic colliders (FluidSim 4.0).

docs/specs/4.0/SPEC-rigid-coupling.md. FluidSim owns the 6-DoF state of every
Dynamic collider (Blender's rigid-body world cannot take per-frame forces):

* On first sight a body captures its authored world mesh. Body-frame points
  are those points relative to the centre of mass, with the authored
  orientation as identity; mass and inertia come from density x mesh volume
  (exact polyhedral integrals for a closed mesh, a winding-number lattice
  otherwise). The state survives ``set_colliders`` and is cleared by Reset.
* Every substep, before collisions are rasterized, ``pose`` writes the body's
  current pose into the collider mesh and its velocity into the descriptor,
  so the projection sees it as a moving solid (Neumann faces).
* After the projection and before G2P, ``step`` gathers the pressure impulse
  on the body's solid faces plus quadratic drag at its boundary samples
  (``kernels_rigid``: segmented gathers, no atomics) and advances the body
  with semi-implicit Euler in float64 on the host.

No bpy imports allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import warp as wp

from . import rigid_scene
from .fastlaunch import launch as fast_launch
from .kernels_rigid import (
    BODY_BOX_DIM,
    BODY_BOX_LO,
    BODY_COEFF,
    BODY_COM,
    BODY_OMEGA,
    BODY_RANK,
    BODY_ROT,
    BODY_VEL,
    BODY_WIDTH,
    k_rigid_drag_chunks,
    k_rigid_face_chunks,
    k_rigid_pose_points,
    k_rigid_reduce,
)

LIQUID_DENSITY = 1000.0      # kg/m^3; collider_density is relative to this
MAX_BODIES = rigid_scene.MAX_BODIES
SAMPLE_SPACING_VOXELS = 0.5  # boundary sample spacing, in voxels
MAX_SAMPLES_PER_BODY = 16384
DRAG_CHUNK = 8               # boundary samples per drag partial
FACE_VOXELS_PER_PARTIAL = 64
MAX_FACE_PARTIALS = 4096
BOX_MARGIN_VOXELS = 2        # face-gather box margin beyond the bounding sphere
STATE_WIDTH = 13             # pos 3, quat 4 (w x y z), lin_vel 3, ang_vel 3
SURFACE_OFFSET_VOXELS = 1.0  # air-centre p = 0 above the particle interface


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _edges_closed(indices):
    """True when every undirected edge is shared by exactly two triangles."""
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if len(tris) == 0:
        return False
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def polyhedron_mass_properties(points, indices):
    """Volume, centre and inertia (unit density, about the centre) of a closed
    triangle mesh by exact polyhedral integrals (Eberly). Orientation-agnostic:
    an inward-wound mesh gives the same result."""
    p = np.asarray(points, dtype=np.float64)
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    p0, p1, p2 = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    d = np.cross(p1 - p0, p2 - p0)

    def sub(w0, w1, w2):
        temp0 = w0 + w1
        f1 = temp0 + w2
        temp1 = w0 * w0
        temp2 = temp1 + w1 * temp0
        f2 = temp2 + w2 * f1
        f3 = w0 * temp1 + w1 * temp2 + w2 * f2
        g0 = f2 + w0 * (f1 + w0)
        g1 = f2 + w1 * (f1 + w1)
        g2 = f2 + w2 * (f1 + w2)
        return f1, f2, f3, g0, g1, g2

    f1x, f2x, f3x, g0x, g1x, g2x = sub(p0[:, 0], p1[:, 0], p2[:, 0])
    _, f2y, f3y, g0y, g1y, g2y = sub(p0[:, 1], p1[:, 1], p2[:, 1])
    _, f2z, f3z, g0z, g1z, g2z = sub(p0[:, 2], p1[:, 2], p2[:, 2])
    integral = np.array([
        np.sum(d[:, 0] * f1x) / 6.0,
        np.sum(d[:, 0] * f2x) / 24.0,
        np.sum(d[:, 1] * f2y) / 24.0,
        np.sum(d[:, 2] * f2z) / 24.0,
        np.sum(d[:, 0] * f3x) / 60.0,
        np.sum(d[:, 1] * f3y) / 60.0,
        np.sum(d[:, 2] * f3z) / 60.0,
        np.sum(d[:, 0] * (p0[:, 1] * g0x + p1[:, 1] * g1x + p2[:, 1] * g2x)) / 120.0,
        np.sum(d[:, 1] * (p0[:, 2] * g0y + p1[:, 2] * g1y + p2[:, 2] * g2y)) / 120.0,
        np.sum(d[:, 2] * (p0[:, 0] * g0z + p1[:, 0] * g1z + p2[:, 0] * g2z)) / 120.0,
    ])
    if integral[0] < 0.0:
        integral = -integral
    volume = float(integral[0])
    if not volume > 0.0:
        raise ValueError("Dynamic collider mesh encloses no volume")
    c = integral[1:4] / volume
    xx = integral[5] + integral[6] - volume * (c[1] ** 2 + c[2] ** 2)
    yy = integral[4] + integral[6] - volume * (c[2] ** 2 + c[0] ** 2)
    zz = integral[4] + integral[5] - volume * (c[0] ** 2 + c[1] ** 2)
    xy = -(integral[7] - volume * c[0] * c[1])
    yz = -(integral[8] - volume * c[1] * c[2])
    xz = -(integral[9] - volume * c[2] * c[0])
    inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return volume, c, inertia


@wp.kernel
def _k_winding_inside(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3),
                      max_dist: float, inside: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh_id, points[tid], max_dist)
    inside[tid] = 0
    if q.result and q.sign < 0.0:
        inside[tid] = 1


def sampled_mass_properties(mesh, points, spacing, device):
    """Winding-number lattice estimate for an open (non-watertight) mesh."""
    p = np.asarray(points, dtype=np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    extent = np.maximum(hi - lo, 1.0e-9)
    step = max(float(spacing), float(np.max(extent)) / 64.0)
    axes = [np.arange(lo[i] + 0.5 * step, hi[i], step) for i in range(3)]
    if any(len(a) == 0 for a in axes):
        raise ValueError("Dynamic collider mesh encloses no volume")
    grid = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    inside = wp.zeros(len(grid), dtype=wp.int32, device=device)
    wp.launch(_k_winding_inside, dim=len(grid),
              inputs=[mesh.id, wp.array(grid.astype(np.float32), dtype=wp.vec3,
                                        device=device),
                      float(np.linalg.norm(extent)) * 2.0],
              outputs=[inside], device=device)
    cells = grid[inside.numpy().astype(bool)]
    if len(cells) == 0:
        raise ValueError("Dynamic collider mesh encloses no volume")
    cell_volume = step ** 3
    volume = len(cells) * cell_volume
    c = cells.mean(axis=0)
    r = cells - c
    inertia = cell_volume * (np.eye(3) * np.sum(r * r) - r.T @ r)
    return volume, c, inertia


def boundary_samples(points, indices, spacing, max_samples=MAX_SAMPLES_PER_BODY):
    """Sub-triangle centroids at about ``spacing`` with outward area normals.

    Returns ``(positions (N, 3), area_normals (N, 3))``: each normal has the
    length of the patch area it stands for. Deterministic (pure numpy, fixed
    triangle and sub-triangle order).
    """
    p = np.asarray(points, dtype=np.float64)
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(area.sum())
    if total_area <= 0.0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    spacing = max(float(spacing), math.sqrt(total_area / max_samples))
    longest = np.max(np.stack([np.linalg.norm(b - a, axis=1),
                               np.linalg.norm(c - b, axis=1),
                               np.linalg.norm(a - c, axis=1)]), axis=0)
    splits = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
    # Never exceed the cap even for slivers: shrink every split together.
    while int(np.sum(splits * splits)) > max_samples and np.any(splits > 1):
        splits = np.maximum(1, splits // 2)
    unit = np.zeros_like(cross)
    nonzero = area > 0.0
    unit[nonzero] = cross[nonzero] / (2.0 * area[nonzero, None])
    positions, normals = [], []
    for t in range(len(tris)):
        if not nonzero[t]:
            continue
        n = int(splits[t])
        sub_area = area[t] / (n * n)
        e1 = (b[t] - a[t]) / n
        e2 = (c[t] - a[t]) / n
        for i in range(n):
            for j in range(n - i):
                positions.append(a[t] + (i + 1.0 / 3.0) * e1 + (j + 1.0 / 3.0) * e2)
                normals.append(unit[t] * sub_area)
                if i + j < n - 1:
                    positions.append(a[t] + (i + 2.0 / 3.0) * e1 + (j + 2.0 / 3.0) * e2)
                    normals.append(unit[t] * sub_area)
    if not positions:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(positions), np.asarray(normals)


@dataclass
class RigidBody:
    """Host-owned 6-DoF state and mass properties of one Dynamic collider."""

    uid: str
    point_count: int
    body_points: np.ndarray          # (P, 3) float64, relative to rest_com
    rest_com: np.ndarray             # authored world centre of mass
    volume: float
    unit_inertia: np.ndarray         # body-frame inertia per unit density
    samples: np.ndarray              # (S, 3) body-frame boundary samples
    sample_normals: np.ndarray       # (S, 3) outward normal x patch area
    radius: float
    density: float = LIQUID_DENSITY
    drag: float = 0.0
    lock: tuple = (False,) * 6
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_points_dev: object = None

    @property
    def mass(self):
        return self.density * self.volume

    def rotation(self):
        return _quat_to_matrix(self.quat)

    def state_row(self):
        return np.concatenate([self.pos, self.quat, self.lin_vel, self.ang_vel])

    def set_state_row(self, row):
        row = np.asarray(row, dtype=np.float64)
        self.pos = row[0:3].copy()
        quat = row[3:7].copy()
        self.quat = quat / max(np.linalg.norm(quat), 1.0e-300)
        self.lin_vel = row[7:10].copy()
        self.ang_vel = row[10:13].copy()

    def record(self):
        """The cache v3.1 per-frame record of this body."""
        return {
            "object_uid": self.uid,
            "pos": tuple(float(v) for v in self.pos),
            "quat": tuple(float(v) for v in self.quat),
            "lin_vel": tuple(float(v) for v in self.lin_vel),
            "ang_vel": tuple(float(v) for v in self.ang_vel),
            "rest_com": tuple(float(v) for v in self.rest_com),
        }


def _host_mesh(mesh):
    points = np.asarray(mesh.points.numpy(), dtype=np.float64)
    indices = np.asarray(mesh.indices.numpy(), dtype=np.int64)
    return points, indices


def _new_body(uid, desc, voxel_size, device):
    points, indices = _host_mesh(desc.mesh)
    if _edges_closed(indices):
        volume, com, inertia = polyhedron_mass_properties(points, indices)
    else:
        volume, com, inertia = sampled_mass_properties(
            desc.mesh, points, SAMPLE_SPACING_VOXELS * voxel_size, device)
    local = points - com
    samples, normals = boundary_samples(
        local, indices, SAMPLE_SPACING_VOXELS * voxel_size)
    body = RigidBody(
        uid=uid, point_count=len(points), body_points=local, rest_com=com,
        volume=float(volume), unit_inertia=inertia, samples=samples,
        sample_normals=normals,
        radius=float(np.max(np.linalg.norm(local, axis=1))) if len(local) else 0.0,
    )
    body.pos = com.copy()
    body.lin_vel = np.asarray(getattr(desc, "initial_velocity", (0.0,) * 3),
                              dtype=np.float64).copy()
    body.body_points_dev = wp.array(local, dtype=wp.vec3d, device=device)
    return body


def collider_uid(desc, rank):
    uid = str(getattr(desc, "object_uid", "") or "")
    return uid or f"collider_{rank}"


class RigidCoupling:
    """Owns every Dynamic collider's state for one solver."""

    def __init__(self, device):
        self.device = device
        self.bodies = {}            # uid -> RigidBody, insertion ordered
        self.attached = []          # [(rank, desc, body)] for this frame
        self._pending_states = {}   # uid -> state row from a checkpoint
        self._sample_key = None
        self._sample_buffers = None
        self._gather_key = None
        self._gather_buffers = None
        self.last_impulses = None

    @property
    def active(self):
        return bool(self.attached)

    def reset(self):
        self.bodies.clear()
        self.attached = []
        self._pending_states.clear()
        self._sample_key = None
        self._sample_buffers = None
        self._gather_key = None
        self._gather_buffers = None
        self.last_impulses = None

    # Attachment

    def attach(self, colliders, voxel_size):
        """Bind this frame's Dynamic colliders to their persistent bodies."""
        dynamic = [(rank, desc) for rank, desc in enumerate(colliders)
                   if getattr(desc, "dynamic", False) and desc.mesh is not None]
        if len(dynamic) > MAX_BODIES:
            raise ValueError(
                f"{len(dynamic)} Dynamic colliders exceed the limit of {MAX_BODIES}")
        attached = []
        seen = set()
        for rank, desc in dynamic:
            uid = collider_uid(desc, rank)
            if uid in seen:
                raise ValueError(f"Dynamic collider id {uid!r} is used twice")
            seen.add(uid)
            body = self.bodies.get(uid)
            if body is None or body.point_count != int(desc.mesh.points.shape[0]):
                body = _new_body(uid, desc, voxel_size, self.device)
                self.bodies[uid] = body
                pending = self._pending_states.pop(uid, None)
                if pending is not None:
                    body.set_state_row(pending)
            body.density = max(float(getattr(desc, "density", LIQUID_DENSITY)), 1.0e-6)
            body.drag = max(float(getattr(desc, "drag", 0.0)), 0.0)
            lock = tuple(bool(v) for v in getattr(desc, "lock_axes", (False,) * 6))
            body.lock = (lock + (False,) * 6)[:6]
            self._describe(desc, body)
            attached.append((rank, desc, body))
        self.attached = attached

    def _describe(self, desc, body):
        """Copy the body's state into its collider descriptor."""
        desc.linear_velocity = tuple(float(v) for v in body.lin_vel)
        desc.angular_velocity = tuple(float(v) for v in body.ang_vel)
        desc.center = tuple(float(v) for v in body.pos)
        r = body.radius
        desc.aabb_min = tuple(float(v - r) for v in body.pos)
        desc.aabb_max = tuple(float(v + r) for v in body.pos)

    def pose(self):
        """Write each attached body's current pose into its collider mesh."""
        for _, desc, body in self.attached:
            rot = body.rotation()
            fast_launch(
                k_rigid_pose_points, dim=body.point_count,
                inputs=[body.body_points_dev, wp.vec3d(*body.pos),
                        wp.mat33d(*rot.ravel())],
                outputs=[desc.mesh.points], device=self.device)
            desc.mesh.refit()
            desc.reposed = True
            self._describe(desc, body)
            desc.pose_center = tuple(float(v) for v in body.pos)

    # Force gather

    def _samples(self):
        key = tuple((body.uid, len(body.samples)) for _, _, body in self.attached)
        if key == self._sample_key:
            return self._sample_buffers
        points, normals, valid, chunk_body = [], [], [], []
        first, count = [], []
        for index, (_, _, body) in enumerate(self.attached):
            n = len(body.samples)
            chunks = (n + DRAG_CHUNK - 1) // DRAG_CHUNK
            padded = chunks * DRAG_CHUNK
            first.append(len(chunk_body))
            count.append(chunks)
            chunk_body.extend([index] * chunks)
            pts = np.zeros((padded, 3))
            nrm = np.zeros((padded, 3))
            ok = np.zeros(padded, dtype=np.int32)
            pts[:n] = body.samples
            nrm[:n] = body.sample_normals
            ok[:n] = 1
            points.append(pts)
            normals.append(nrm)
            valid.append(ok)
        dev = self.device
        if chunk_body:
            buffers = (
                wp.array(np.concatenate(points), dtype=wp.vec3d, device=dev),
                wp.array(np.concatenate(normals), dtype=wp.vec3d, device=dev),
                wp.array(np.concatenate(valid), dtype=wp.int32, device=dev),
                wp.array(np.asarray(chunk_body, dtype=np.int32), dtype=wp.int32, device=dev),
            )
        else:
            buffers = None
        self._sample_buffers = (
            buffers, len(chunk_body),
            wp.array(np.asarray(first, dtype=np.int32), dtype=wp.int32, device=dev),
            wp.array(np.asarray(count, dtype=np.int32), dtype=wp.int32, device=dev),
        )
        self._sample_key = key
        return self._sample_buffers

    def _buffers(self, n, partials, drag_chunks):
        """Device scratch for one gather, reused while its sizes hold.

        Per-substep ``wp.array`` construction and allocation dominated the
        coupling's host cost; every buffer here is fully overwritten by the
        kernel that writes it, so reuse cannot leak state between substeps.
        """
        key = (n, partials, drag_chunks)
        if key == self._gather_key:
            return self._gather_buffers
        dev = self.device
        pinned = wp.get_device(dev).is_cuda
        host = wp.zeros((n, BODY_WIDTH), dtype=wp.float64, device="cpu",
                        pinned=pinned)
        readback = wp.empty((n, 14), dtype=wp.float64, device="cpu",
                            pinned=pinned)
        drags = max(drag_chunks, 1)
        self._gather_buffers = {
            "readback": readback,
            "readback_np": readback.numpy(),
            "host": host,
            "host_np": host.numpy(),
            "state": wp.zeros((n, BODY_WIDTH), dtype=wp.float64, device=dev),
            "face_impulse": wp.zeros(n * partials, dtype=wp.vec3d, device=dev),
            "face_angular": wp.zeros(n * partials, dtype=wp.vec3d, device=dev),
            "face_volume": wp.zeros(n * partials, dtype=wp.float64, device=dev),
            "drag_impulse": wp.zeros(drags, dtype=wp.vec3d, device=dev),
            "drag_angular": wp.zeros(drags, dtype=wp.vec3d, device=dev),
            "drag_stiffness": wp.zeros(drags, dtype=wp.float64, device=dev),
            "no_drag": wp.zeros(n, dtype=wp.int32, device=dev),
            "out": wp.zeros((n, 14), dtype=wp.float64, device=dev),
        }
        self._gather_key = key
        return self._gather_buffers

    def gather(self, solver, dt, pressure_valid=True):
        """Per-body fluid impulses over one substep, as a (bodies, 14) array.

        Columns: pressure impulse (3), pressure angular impulse (3), drag
        impulse (3), drag angular impulse (3), rasterized solid volume, drag
        stiffness (see ``k_rigid_reduce``).
        """
        dev = self.device
        g = solver.grid
        n = len(self.attached)
        h = float(g.voxel_size)
        dims = np.asarray([int(g.voxels_dim[i]) for i in range(3)], dtype=np.int64)
        origin = np.asarray([float(g.origin[i]) for i in range(3)], dtype=np.float64)
        # Each body's face gather walks a voxel box around its bounding
        # sphere. The box extent depends on the radius alone, so the partial
        # count (and every buffer size) holds while the bodies do.
        extents = []
        for _, _, body in self.attached:
            span = int(math.ceil(2.0 * body.radius / h)) + 2 + 2 * BOX_MARGIN_VOXELS
            extents.append(np.minimum(np.full(3, span, dtype=np.int64), dims))
        cells = max(int(np.prod(e)) for e in extents)
        partials = max(1, min(MAX_FACE_PARTIALS,
                              -(-cells // FACE_VOXELS_PER_PARTIAL)))
        samples, drag_chunks, drag_first, drag_count = self._samples()
        if samples is None:
            drag_chunks = 0
        buf = self._buffers(n, partials, drag_chunks)
        # One packed host -> device upload of every per-body input.
        state = buf["host_np"]
        for index, (rank, _, body) in enumerate(self.attached):
            # -2 matches no voxel: an unprojected substep has no pressure.
            state[index, BODY_RANK] = float(rank) if pressure_valid else -2.0
            state[index, BODY_COM:BODY_COM + 3] = body.pos
            state[index, BODY_ROT:BODY_ROT + 9] = body.rotation().ravel()
            state[index, BODY_VEL:BODY_VEL + 3] = body.lin_vel
            state[index, BODY_OMEGA:BODY_OMEGA + 3] = body.ang_vel
            state[index, BODY_COEFF] = 0.5 * body.drag * LIQUID_DENSITY
            extent = extents[index]
            low = np.floor((body.pos - body.radius - origin) / h).astype(np.int64)
            low = np.clip(low - BOX_MARGIN_VOXELS, 0, dims - extent)
            state[index, BODY_BOX_LO:BODY_BOX_LO + 3] = low
            state[index, BODY_BOX_DIM:BODY_BOX_DIM + 3] = extent
        wp.copy(buf["state"], buf["host"])
        gravity = float(np.linalg.norm(np.asarray(solver.settings.gravity,
                                                  dtype=np.float64)))
        # The projection's p = 0 sits about one voxel above the particle
        # interface (first air-voxel centre); see k_rigid_face_chunks.
        q_offset = gravity * float(dt) * float(g.voxel_size) * SURFACE_OFFSET_VOXELS
        face_impulse = buf["face_impulse"]
        face_angular = buf["face_angular"]
        face_volume = buf["face_volume"]
        fast_launch(
            k_rigid_face_chunks, dim=n * partials,
            inputs=[solver.pressure_solver.pressure, solver.voxel_type.data,
                    solver._collision_owner, g.leaf_table,
                    g.voxels_dim, g.leaves_dim, g.origin,
                    float(g.voxel_size), LIQUID_DENSITY, q_offset,
                    buf["state"], int(partials)],
            outputs=[face_impulse, face_angular, face_volume], device=dev)

        drag_impulse = buf["drag_impulse"]
        drag_angular = buf["drag_angular"]
        drag_stiffness = buf["drag_stiffness"]
        if drag_chunks:
            vel = solver.vel
            fast_launch(
                k_rigid_drag_chunks, dim=drag_chunks,
                inputs=[*samples, buf["state"],
                        vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww,
                        solver.voxel_type.data, g.leaf_table, g.voxels_dim,
                        g.leaves_dim, g.origin, float(g.voxel_size), float(dt),
                        DRAG_CHUNK],
                outputs=[drag_impulse, drag_angular, drag_stiffness], device=dev)
        else:
            drag_first = drag_count = buf["no_drag"]
        out = buf["out"]
        fast_launch(
            k_rigid_reduce, dim=n,
            inputs=[face_impulse, face_angular, face_volume, int(partials),
                    drag_impulse, drag_angular, drag_stiffness, drag_first,
                    drag_count],
            outputs=[out], device=dev)
        return self._read_impulses(buf)

    def _read_impulses(self, buf):
        # Match array.numpy()'s null-stream ordering, but reuse the pinned
        # transfer buffer instead of allocating a CPU Warp array per substep.
        dev = wp.get_device(self.device)
        with wp.ScopedStream(dev.null_stream):
            wp.copy(buf["readback"], buf["out"])
        if dev.is_cuda:
            wp.synchronize_stream(dev.null_stream)
        # Callers may retain last_impulses; the next gather must not mutate it.
        return buf["readback_np"].copy()

    # Integration

    def step(self, solver, dt, pressure_valid=True):
        """Gather fluid impulses and advance every attached body by ``dt``."""
        if not self.attached:
            return
        impulses = self.gather(solver, dt, pressure_valid)
        self.last_impulses = impulses
        settings = solver.settings
        gravity = np.asarray(settings.gravity, dtype=np.float64)
        dv_max = float(solver._cfl_speed_limit(dt))
        bounds = (np.asarray(settings.domain_min, dtype=np.float64),
                  np.asarray(settings.domain_max, dtype=np.float64))
        for row, (_, desc, body) in zip(impulses, self.attached):
            self._integrate(body, row, gravity, dt, dv_max)
            self._contain(body, *bounds)
            self._describe(desc, body)

    @staticmethod
    def _contain(body, lo, hi):
        """Inelastic contact with the domain walls.

        Body-body and body-collider contact are not modelled in 4.0; the
        domain box is, so a sinking body rests on the floor instead of
        leaving the grid.
        """
        if not len(body.body_points):
            return
        world = body.body_points @ body.rotation().T
        low = body.pos + world.min(axis=0)
        high = body.pos + world.max(axis=0)
        for axis in range(3):
            if low[axis] < lo[axis]:
                body.pos[axis] += lo[axis] - low[axis]
                body.lin_vel[axis] = max(body.lin_vel[axis], 0.0)
            elif high[axis] > hi[axis]:
                body.pos[axis] -= high[axis] - hi[axis]
                body.lin_vel[axis] = min(body.lin_vel[axis], 0.0)

    @staticmethod
    def _integrate(body, row, gravity, dt, dv_max):
        row = np.asarray(row, dtype=np.float64)
        if not np.all(np.isfinite(row)):
            row = np.zeros_like(row)
        mass = body.mass
        # Pressure: rescale by true / rasterized volume so the voxelized
        # solid displaces exactly the mesh volume (Archimedes on the mesh).
        rasterized = float(row[12])
        scale = body.volume / rasterized if rasterized > 0.0 else 1.0
        # A body half outside the grid rasterizes only in part; bound it.
        scale = min(max(scale, 0.5), 2.0)
        impulse = row[0:3] * scale
        angular = row[3:6] * scale
        # Drag, applied implicitly: dv = J / (m + K) for stiffness K.
        stiffness = max(float(row[13]), 0.0)
        damping = mass / (mass + stiffness)
        impulse = impulse + row[6:9] * damping
        angular = angular + row[9:12] * damping
        # Trap 4: clamp the net per-substep impulse (fluid + weight) to
        # m * dv_max. Clamping the fluid part alone would cap buoyancy near
        # the body's weight at one substep per frame.
        dv = impulse / mass + gravity * dt
        size = float(np.linalg.norm(dv))
        if size > dv_max:
            dv *= dv_max / size
        lin = body.lin_vel + dv
        rot = body.rotation()
        inertia = body.density * (rot @ body.unit_inertia @ rot.T)
        omega = body.ang_vel
        torque = angular - dt * np.cross(omega, inertia @ omega)
        try:
            omega = omega + np.linalg.solve(inertia, torque)
        except np.linalg.LinAlgError:
            omega = omega.copy()
        for axis in range(3):
            if body.lock[axis]:
                lin[axis] = 0.0
            if body.lock[3 + axis]:
                omega[axis] = 0.0
        speed = float(np.linalg.norm(lin))
        if speed > dv_max:
            lin *= dv_max / speed
        spin = float(np.linalg.norm(omega)) * max(body.radius, 1.0e-6)
        if spin > dv_max:
            omega *= dv_max / spin
        body.lin_vel = lin
        body.ang_vel = omega
        body.pos = body.pos + lin * dt
        if np.any(omega != 0.0):
            dq = _quat_mul(np.concatenate([[0.0], omega]), body.quat)
            quat = body.quat + 0.5 * dt * dq
            body.quat = quat / np.linalg.norm(quat)

    # Records and checkpoints

    def states(self):
        return [body.record() for body in self.bodies.values()]

    def export_state(self):
        uids = list(self.bodies)
        rows = np.asarray([self.bodies[uid].state_row() for uid in uids],
                          dtype=np.float64).reshape(len(uids), STATE_WIDTH)
        return uids, rows

    def import_state(self, uids, rows):
        """Restore body state; bodies not attached yet take it on attach."""
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, STATE_WIDTH)
        if len(uids) != len(rows):
            raise ValueError("rigid body checkpoint ids and states disagree")
        self._pending_states = {}
        for uid, row in zip(uids, rows):
            body = self.bodies.get(str(uid))
            if body is None:
                self._pending_states[str(uid)] = row.copy()
            else:
                body.set_state_row(row)
        for _, desc, body in self.attached:
            self._describe(desc, body)
