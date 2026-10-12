"""B36: APIC affine rows past the live count are zero after any compaction.

``ParticleSystem.affine`` is double buffered for compaction. Emission and
narrow-band reseeding append at ``count`` and rely on zero affine there. A
shrinking sequence of hinted compactions (40 -> 4 -> 2 live particles) used
to leave the spare buffer's old rows (from when it was the primary at a
larger bound) past the new, smaller hint, so newly appended APIC particles
inherited a stale affine matrix before their first P2G.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.particles import ParticleSystem  # noqa: E402

CAPACITY = 64


def _tail_rows(p):
    count = int(p.count.numpy()[0])
    rows = np.any(p.affine.numpy()[count:].reshape(-1, 9) != 0.0, axis=1)
    return count, np.flatnonzero(rows) + count


def test_shrinking_hinted_compaction_leaves_no_stale_affine(cuda_device):
    p = ParticleSystem(CAPACITY, cuda_device)
    p.ensure_affine()
    n0 = 40
    wp.copy(p.count, wp.array([n0], dtype=wp.int32, device=cuda_device))
    alive = np.zeros(CAPACITY, np.int32)
    alive[:n0] = 1
    p.alive.assign(alive)
    affine = np.zeros((CAPACITY, 3, 3), np.float32)
    affine[:n0] = np.eye(3) * 7.0
    p.affine.assign(affine)

    alive[4:n0] = 0
    p.alive.assign(alive)
    p.compact(count_hint=n0)
    count, stale = _tail_rows(p)
    assert count == 4 and stale.size == 0

    alive = p.alive.numpy().copy()
    alive[2:count] = 0
    p.alive.assign(alive)
    p.compact(count_hint=count)
    count, stale = _tail_rows(p)
    assert count == 2
    assert stale.size == 0, f"stale affine rows at slots {stale.tolist()}"
    # the survivors keep their matrices
    np.testing.assert_array_equal(p.affine.numpy()[:2], np.stack([np.eye(3) * 7.0] * 2))
