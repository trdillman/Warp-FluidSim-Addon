"""Shared capture lifetime and explicit transform policy; no Blender runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender.evaluated_geometry import (
    EvaluatedGeometry, TransformPolicy,
)
from fluidSim_warp_blender51.blender.cache_identity import _geometry_record


def test_transform_policies_and_authored_override_copy_the_selected_matrix():
    original, evaluated, authored = [np.eye(4) for _ in range(3)]
    original[0, 3], evaluated[0, 3], authored[0, 3] = 1, 2, 3
    obj = SimpleNamespace(
        matrix_world=original,
        evaluated_get=lambda _: SimpleNamespace(matrix_world=evaluated))
    capture = EvaluatedGeometry(obj, object())
    assert capture.world_matrix(TransformPolicy.ORIGINAL)[0, 3] == 1
    assert capture.world_matrix(TransformPolicy.EVALUATED)[0, 3] == 2
    selected = capture.world_matrix(TransformPolicy.EVALUATED, override=authored)
    authored[0, 3] = 100
    assert selected[0, 3] == 3


def test_mesh_release_on_failure_and_data_fallback_does_not_release_borrowed_data():
    released = []
    obj = SimpleNamespace(
        to_mesh=lambda: object(), to_mesh_clear=lambda: released.append(True))
    with pytest.raises(RuntimeError, match="projection failed"):
        with EvaluatedGeometry(obj, None).mesh():
            raise RuntimeError("projection failed")
    assert released == [True]
    borrowed = SimpleNamespace(data=object(), to_mesh_clear=lambda: released.append(False))
    with EvaluatedGeometry(borrowed, None).mesh(data_fallback=True) as mesh:
        assert mesh is borrowed.data
    assert released == [True]


def test_identity_keeps_local_vertex_bytes_and_existing_topology_projection():
    class Vertices:
        def __len__(self):
            return 3

        def foreach_get(self, name, out):
            assert name == "co"
            out[:] = np.arange(9)

    mesh = SimpleNamespace(vertices=Vertices(), polygons=[object()])
    released = []
    evaluated = SimpleNamespace(
        type="MESH", to_mesh=lambda: mesh,
        to_mesh_clear=lambda: released.append(True))
    obj = SimpleNamespace(evaluated_get=lambda _: evaluated)
    record = _geometry_record(obj, object())
    assert record == {
        "kind": "MESH", "vertex_count": 3,
        "vertex_bytes": np.arange(9, dtype=np.float32).tobytes(),
        "polygon_count": 1,
    }
    # Cache compatibility: equal-count topology edits are intentionally not
    # newly fingerprinted by this lifetime-only refactor.
    mesh.polygons = [object()]
    assert _geometry_record(obj, object()) == record
    assert released == [True, True]
