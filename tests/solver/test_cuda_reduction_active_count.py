"""Device-counted dot product: a graph-capturable reduction whose length is
read from ``num_leaves`` at launch instead of being baked into the launch.

The viscosity conditional graphs used to be keyed by the active voxel count
because the captured reductions took it as an immediate. Moving the count on
device lets one captured graph serve every solve in a launch bucket, which is
only acceptable if the reduction stays bitwise identical to the host-counted
one it replaces -- these tests pin that.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME
from fluidSim_warp_blender51.sim.cuda_reduction import CudaDotProduct


def _launch_dims(dot, n, capacity):
    """The exact count, the viscosity bucket, and a fully saturated grid."""
    dims = {n, min(capacity, n + LEAF_VOLUME * 3), capacity,
            256 * dot._max_blocks}
    return sorted(d for d in dims if n <= d <= capacity)


def test_active_count_dot_matches_the_host_counted_dot_bitwise(cuda_device):
    dot = CudaDotProduct(cuda_device)
    try:
        max_leaves = (256 * dot._max_blocks) // LEAF_VOLUME + 8
        capacity = max_leaves * LEAF_VOLUME
        rng = np.random.default_rng(20260929)
        # Stale, non-zero data past the active prefix: an over-long launch
        # that summed it would be caught immediately.
        a = wp.array(rng.standard_normal(capacity).astype(np.float32),
                     device=cuda_device)
        b = wp.array(rng.standard_normal(capacity).astype(np.float32),
                     device=cuda_device)
        num_leaves = wp.zeros(1, dtype=wp.int32, device=cuda_device)
        reference = wp.zeros(1, dtype=float, device=cuda_device)
        active = wp.zeros(1, dtype=float, device=cuda_device)
        # small, around the one-element-per-thread boundary, and saturated
        leaf_counts = sorted({1, 2, 3, 7, 50, max_leaves - 9, max_leaves - 8,
                              max_leaves - 1, max_leaves})
        checked = 0
        for leaves in leaf_counts:
            n = leaves * LEAF_VOLUME
            num_leaves.fill_(leaves)
            reference.zero_()
            dot.launch(a, b, reference, n)
            expected = reference.numpy()[0]
            for dim in _launch_dims(dot, n, capacity):
                active.zero_()
                dot.launch_active(a, b, active, num_leaves, dim)
                got = active.numpy()[0]
                assert got.tobytes() == expected.tobytes(), (
                    f"leaves={leaves} dim={dim}: {got!r} != {expected!r}")
                checked += 1
        assert checked > len(leaf_counts)
    finally:
        dot.release()


def test_active_count_dot_accumulates_like_the_host_counted_dot(cuda_device):
    """Both keep the ``*output += total`` contract callers rely on."""
    dot = CudaDotProduct(cuda_device)
    try:
        a = wp.full(LEAF_VOLUME * 2, 1.0, dtype=float, device=cuda_device)
        num_leaves = wp.full(1, 1, dtype=wp.int32, device=cuda_device)
        out = wp.full(1, 5.0, dtype=float, device=cuda_device)
        dot.launch_active(a, a, out, num_leaves, LEAF_VOLUME * 2)
        assert out.numpy()[0] == 5.0 + LEAF_VOLUME
    finally:
        dot.release()
