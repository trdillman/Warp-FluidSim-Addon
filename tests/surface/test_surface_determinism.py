"""4.0 lane W: surface meshes are byte-identical across repeated extractions.

Derived marching-cubes surfaces were never pinned for repeatability.  The
sparse extractor visits active leaves in slot order and emits candidate cells
in a fixed per-leaf order, so a fresh extractor on the same SDF must produce
the same vertex and index bytes -- that is what makes two bakes of the same
scene give the same surface cache.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import surface as SURF  # noqa: E402

from test_surface_extractor_parity import CASES, _sphere_field  # noqa: E402


@pytest.mark.parametrize("backend", ["sparse", "dense"])
def test_two_extractions_give_identical_mesh_bytes(cuda_device, backend):
    if backend == "sparse" and not SURF.SPARSE_MARCHING_CUBES_AVAILABLE:
        pytest.skip("sparse backend needs Warp >= 1.18")
    meshes = []
    for _ in range(2):
        grid, field = _sphere_field(cuda_device, *CASES["interior"])
        ext = SURF.SurfaceExtractor(grid, cuda_device, backend=backend)
        meshes.append(ext.extract(field))
        # Reuse within a session must not change the answer either.
        meshes.append(ext.extract(field))
    first_v, first_t = meshes[0]
    assert len(first_t) > 0
    for verts, tris in meshes[1:]:
        assert verts.tobytes() == first_v.tobytes()
        assert tris.tobytes() == first_t.tobytes()
