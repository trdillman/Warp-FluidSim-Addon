"""Rigid impulse readback must reuse transfer storage and return owned snapshots."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")


def test_impulse_readback_reuses_transfer_storage_and_keeps_old_result(monkeypatch):
    from fluidSim_warp_blender51.sim.rigid_coupling import RigidCoupling
    rigid = RigidCoupling("cpu")
    buf = rigid._buffers(2, 1, 0)
    first_values = np.arange(28, dtype=np.float64).reshape(2, 14)
    buf["out"].assign(first_values)
    first = rigid._read_impulses(buf)
    def no_allocation(*args, **kwargs):
        raise AssertionError("readback allocated Warp transfer storage")
    monkeypatch.setattr(wp, "empty", no_allocation)
    monkeypatch.setattr(wp, "zeros", no_allocation)
    buf["out"].assign(first_values + 100)
    second = rigid._read_impulses(buf)
    np.testing.assert_array_equal(first, first_values)
    np.testing.assert_array_equal(second, first_values + 100)
    assert not np.shares_memory(first, second)


def test_impulse_readback_waits_for_cuda_work(cuda_device):
    from fluidSim_warp_blender51.sim.rigid_coupling import RigidCoupling
    if not wp.get_device(cuda_device).is_cuda:
        pytest.skip("requires CUDA stream ordering")
    rigid = RigidCoupling(cuda_device)
    buf = rigid._buffers(1, 1, 0)
    # Match numpy()'s null-stream ordering, including a non-default producer.
    with wp.ScopedStream(wp.Stream(cuda_device)):
        buf["out"].fill_(7.0)
    result = rigid._read_impulses(buf)
    np.testing.assert_array_equal(result, np.full((1, 14), 7.0))
