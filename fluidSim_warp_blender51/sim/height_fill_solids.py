"""Keep-out mask that stops the height fill from spawning inside collider meshes.

The height fill emits particles per voxel. Before it does, every voxel it could
reach is tested against each collider mesh whose "fill inside" override is off,
and flagged when it is inside the mesh, near its surface, or inside a mesh that
has holes. The emission kernels then skip flagged voxels, so the flag is one
shared answer for admission counting, slot reservation and emission.

Why this does not read the solver's ``collision_sdf``: that field merges every
collider into one distance value, so it cannot honour a per-mesh override, and
its sign comes from a closest-face vote that is only reliable on watertight
meshes, the opposite of the "treat open meshes as solid" rule. The collider
mesh geometry itself is reused (one query-mesh per collider, cached for static
colliders), and a voxel is only tested when it is below the fill level and
inside the mesh's padded bounding box, so the cost is a few comparisons for
nearly every voxel and one mesh query for the rest.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .fastlaunch import launch as fast_launch
from .source_contracts import (
    HEIGHT_FILL_SOLID_PADDING_VOXELS,
    HEIGHT_FILL_WINDING_THRESHOLD,
)
from .source_kernels import k_height_nofill_mask, k_points_bounds


class _SolidQuery:
    """Winding-number query mesh, device bounds and band for one collider.

    ``points`` is the query mesh's own point buffer, so a moving collider is
    followed with a device copy and a refit, never a rebuild. ``body`` is the
    rigid body for a Dynamic collider, whose mesh stays in the body frame.
    """

    __slots__ = ("source", "mesh", "points", "bounds", "band", "body", "followed")

    def __init__(self, source, mesh, points, bounds, band, body=None):
        self.source = source
        self.mesh = mesh
        self.points = points
        self.bounds = bounds
        self.band = band
        self.body = body
        # True once the copy has been re-posed to follow a moving collider.
        self.followed = False


def _is_moving(collider) -> bool:
    return collider.base_points is not None


def eligible_solids(colliders) -> list:
    """Colliders the height fill must stay out of (override off, mesh present)."""
    return [
        col for col in colliders or ()
        if getattr(col, "mesh", None) is not None
        and not getattr(col, "allow_height_fill", False)
    ]


_UNBOUNDED = np.array([1.0e30] * 3 + [-1.0e30] * 3, dtype=np.float32)


class HeightFillSolids:
    """Own the keep-out mask buffer and the per-collider query meshes."""

    def __init__(self, device):
        self.device = device
        self._queries: dict[int, _SolidQuery] = {}
        self._mask = None
        self._unused = wp.zeros(1, dtype=wp.int32, device=device)
        self._no_pose_array = None
        self._unbounded_array = None

    @property
    def _no_pose(self):
        if self._no_pose_array is None:
            self._no_pose_array = wp.zeros((1, 1), dtype=wp.float64, device=self.device)
        return self._no_pose_array

    @property
    def _unbounded(self):
        if self._unbounded_array is None:
            self._unbounded_array = wp.array(_UNBOUNDED, dtype=float, device=self.device)
        return self._unbounded_array

    @property
    def unused_mask(self):
        """Placeholder passed to kernels when no mask applies (never indexed)."""
        return self._unused

    def _winding_copy(self, source, pad, body=None) -> _SolidQuery | None:
        """One-off host read: a winding-capable copy, oriented outward."""
        points = source.points.numpy().astype(np.float64)
        triangles = source.indices.numpy().reshape(-1, 3)
        if not len(points) or not len(triangles):
            return None
        # Winding number reads +1 inside a mesh with outward normals. A closed
        # mesh whose normals point inward has negative volume; reverse it so
        # it still counts as solid. Open meshes are left as authored.
        centred = points - points.mean(axis=0)
        a, b, c = (centred[triangles[:, i]] for i in range(3))
        if float(np.einsum("ij,ij->", a, np.cross(b, c))) < 0.0:
            triangles = triangles[:, ::-1]
        buffer = wp.array(points.astype(np.float32), dtype=wp.vec3, device=self.device)
        mesh = wp.Mesh(
            points=buffer,
            indices=wp.array(
                np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1),
                dtype=wp.int32, device=self.device),
            support_winding_number=True,
        )
        extent = points.max(axis=0) - points.min(axis=0)
        bounds = wp.array(_UNBOUNDED, dtype=float, device=self.device)
        query = _SolidQuery(
            source, mesh, buffer, bounds,
            float(np.linalg.norm(extent)) + 4.0 * float(pad), body)
        self._update_bounds(query, pad)
        return query

    def _update_bounds(self, query, pad):
        wp.copy(query.bounds, self._unbounded)
        wp.launch(
            k_points_bounds, dim=len(query.points),
            inputs=[query.points, float(pad)], outputs=[query.bounds],
            device=self.device)

    def _query_for(self, collider, pad, pose) -> _SolidQuery | None:
        # A Dynamic body is queried in its own frame (built once, never moves);
        # any other collider's mesh already holds its current world pose.
        source = pose[2].local_mesh if pose is not None else collider.mesh
        key = id(source)
        query = self._queries.get(key)
        if query is not None and query.source is source:
            moving = pose is None and _is_moving(collider)
            # A collider that has just stopped is restored to its authored mesh
            # by the solver, so the copy must be refreshed once more.
            if moving or (pose is None and query.followed):
                wp.copy(query.points, source.points)
                query.mesh.refit()
                self._update_bounds(query, pad)
                query.followed = moving
            return query
        query = self._winding_copy(source, pad, pose[2] if pose is not None else None)
        if query is not None:
            self._queries[key] = query
        return query

    def build(self, grid, nvox, voxel_size, solids, *, base_height, back_mode,
              profile, profile_res, domain_min, domain_max, pose_binding=None):
        """Return the int32 keep-out mask for the first ``nvox`` voxel slots.

        ``pose_binding(collider)`` returns ``(state, index, body)`` for a
        Dynamic collider's device-resident pose, or ``None``.
        """
        pad = HEIGHT_FILL_SOLID_PADDING_VOXELS * float(voxel_size)
        entries, live = [], set()
        for col in solids:
            pose = pose_binding(col) if pose_binding and col.dynamic else None
            query = self._query_for(col, pad, pose)
            live.add(id(pose[2].local_mesh if pose is not None else col.mesh))
            if query is not None:
                entries.append((query, pose))
        for key in [key for key in self._queries if key not in live]:
            del self._queries[key]
        if not entries:
            return None
        nvox = max(1, int(nvox))
        if self._mask is None or self._mask.shape[0] < nvox:
            self._mask = wp.zeros(nvox, dtype=wp.int32, device=self.device)
        else:
            self._mask[:nvox].zero_()
        for query, pose in entries:
            state, index, body = pose if pose is not None else (self._no_pose, 0, None)
            fast_launch(
                k_height_nofill_mask,
                dim=nvox,
                inputs=[
                    query.mesh.id, query.bounds, state, int(index),
                    float(body.radius) if body is not None else 0.0,
                    1 if pose is not None else 0,
                    pad, query.band, HEIGHT_FILL_WINDING_THRESHOLD,
                    grid.leaf_coords, grid.num_leaves, grid.origin, float(voxel_size),
                    float(base_height), int(back_mode), profile, profile_res,
                    domain_min, domain_max,
                ],
                outputs=[self._mask],
                device=self.device,
            )
        return self._mask
