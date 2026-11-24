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
from .source_kernels import k_height_nofill_mask


class _SolidQuery:
    """Winding-number query mesh, padded bounds and query band for one collider."""

    __slots__ = ("source", "mesh", "box_min", "box_max", "band")

    def __init__(self, source, mesh, box_min, box_max, band):
        self.source = source
        self.mesh = mesh
        self.box_min = box_min
        self.box_max = box_max
        self.band = band


def _is_moving(collider) -> bool:
    return collider.base_points is not None or bool(getattr(collider, "dynamic", False))


def eligible_solids(colliders) -> list:
    """Colliders the height fill must stay out of (override off, mesh present)."""
    return [
        col for col in colliders or ()
        if getattr(col, "mesh", None) is not None
        and not getattr(col, "allow_height_fill", False)
    ]


class HeightFillSolids:
    """Own the keep-out mask buffer and the per-collider query meshes."""

    def __init__(self, device):
        self.device = device
        self._queries: dict[int, _SolidQuery] = {}
        self._mask = None
        self._unused = wp.zeros(1, dtype=wp.int32, device=device)

    @property
    def unused_mask(self):
        """Placeholder passed to kernels when no mask applies (never indexed)."""
        return self._unused

    def _query_for(self, collider, pad) -> _SolidQuery | None:
        key = id(collider.mesh)
        cached = self._queries.get(key)
        if (cached is not None and cached.source is collider.mesh
                and not _is_moving(collider)):
            return cached
        points = collider.mesh.points.numpy().astype(np.float64)
        triangles = collider.mesh.indices.numpy().reshape(-1, 3)
        if not len(points) or not len(triangles):
            return None
        # Winding number reads +1 inside a mesh with outward normals. A closed
        # mesh whose normals point inward has negative volume; reverse it so
        # it still counts as solid. Open meshes are left as authored.
        centred = points - points.mean(axis=0)
        a, b, c = (centred[triangles[:, i]] for i in range(3))
        if float(np.einsum("ij,ij->", a, np.cross(b, c))) < 0.0:
            triangles = triangles[:, ::-1]
        lo = points.min(axis=0) - pad
        hi = points.max(axis=0) + pad
        mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=self.device),
            indices=wp.array(
                np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1),
                dtype=wp.int32, device=self.device),
            support_winding_number=True,
        )
        query = _SolidQuery(
            collider.mesh, mesh, wp.vec3(*lo), wp.vec3(*hi),
            float(np.linalg.norm(hi - lo)) + 4.0 * float(pad))
        self._queries[key] = query
        return query

    def build(self, grid, nvox, voxel_size, solids, *, base_height, back_mode,
              profile, profile_res, domain_min, domain_max):
        """Return the int32 keep-out mask for the first ``nvox`` voxel slots."""
        pad = HEIGHT_FILL_SOLID_PADDING_VOXELS * float(voxel_size)
        queries = [q for q in (self._query_for(col, pad) for col in solids) if q]
        live = {id(col.mesh) for col in solids}
        for key in [key for key in self._queries if key not in live]:
            del self._queries[key]
        if not queries:
            return None
        nvox = max(1, int(nvox))
        if self._mask is None or self._mask.shape[0] < nvox:
            self._mask = wp.zeros(nvox, dtype=wp.int32, device=self.device)
        else:
            self._mask[:nvox].zero_()
        for query in queries:
            fast_launch(
                k_height_nofill_mask,
                dim=nvox,
                inputs=[
                    query.mesh.id, query.box_min, query.box_max, pad, query.band,
                    HEIGHT_FILL_WINDING_THRESHOLD,
                    grid.leaf_coords, grid.num_leaves, grid.origin, float(voxel_size),
                    float(base_height), int(back_mode), profile, profile_res,
                    domain_min, domain_max,
                ],
                outputs=[self._mask],
                device=self.device,
            )
        return self._mask
