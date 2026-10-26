"""Short-lived evaluated Blender geometry and explicit coordinate policies.

Consumers own their copied arrays, never the temporary Blender mesh. Point
inputs deliberately use the original transform; region inputs use the evaluated
transform, except Dynamic colliders which supply their authored rest matrix.
Cache identity reads local coordinates and fingerprints its own authored matrix.
"""

from contextlib import contextmanager
from enum import Enum

import numpy as np


class TransformPolicy(Enum):
    ORIGINAL = "original"
    EVALUATED = "evaluated"


class EvaluatedGeometry:
    def __init__(self, obj, depsgraph):
        self.original = obj
        self.evaluated = obj.evaluated_get(depsgraph) if depsgraph is not None else obj

    def world_matrix(self, policy, *, override=None):
        if override is not None:
            matrix = override
        elif policy is TransformPolicy.ORIGINAL:
            matrix = self.original.matrix_world
        elif policy is TransformPolicy.EVALUATED:
            matrix = self.evaluated.matrix_world
        else:
            raise ValueError(f"Unknown geometry transform policy: {policy!r}")
        return np.array(matrix, dtype=np.float32, copy=True)

    @contextmanager
    def mesh(self, *, data_fallback=False, tolerate_release_error=False):
        """Acquire once and release on every exit after successful acquisition.

        Identity also supports non-Blender data-only adapters and historically
        tolerates an already-invalidated mesh at release. Simulation consumers
        retain strict release errors. Neither policy changes captured bytes.
        """
        evaluated = self.evaluated
        temporary = hasattr(evaluated, "to_mesh")
        if not temporary and not data_fallback:
            raise AttributeError("Evaluated object has no to_mesh")
        mesh = evaluated.to_mesh() if temporary else evaluated.data
        try:
            yield mesh
        finally:
            if mesh is not None and temporary and hasattr(evaluated, "to_mesh_clear"):
                try:
                    evaluated.to_mesh_clear()
                except RuntimeError:
                    if not tolerate_release_error:
                        raise
