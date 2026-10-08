"""REVIEW-038: ``CudaDotProduct`` checks array lengths before it launches.

``launch`` and ``launch_active`` hand raw device pointers to a kernel that
indexes them as ``a[i]`` / ``b[i]`` for ``i < count``. A short array was an
out-of-bounds GPU read (or, for ``output``, write) that nothing caught. Both
methods now compare ``array.size`` with what the kernel will touch and raise a
``ValueError`` naming the array, the expected length and the actual length.

No CUDA device is needed: the length check runs before the native launch, so
``_launch`` is replaced by a recorder. The recorder is also how these tests
see a bad call reach the kernel on the old code. They deliberately never run
the real kernel with a short array.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.cuda_reduction import CudaDotProduct


class _Array:
    """An array double that fails the test on any read of its data.

    ``launch`` runs on every pressure iteration, so the check may read
    ``size`` and nothing else: no host copy, no synchronise.
    """

    def __init__(self, size, ptr=0x1000):
        self.size = size
        self.shape = (size,)
        self.ptr = ptr

    def numpy(self):
        raise AssertionError("length check copied the array to the host")

    def __getitem__(self, _index):
        raise AssertionError("length check read the array contents")

    def __len__(self):
        raise AssertionError("length check must use size/shape, not len()")


@pytest.fixture
def dot(monkeypatch):
    """A ``CudaDotProduct`` without a device whose native launch is recorded."""
    instance = CudaDotProduct.__new__(CudaDotProduct)
    instance._kernel = 3
    instance._active_kernel = 4
    instance.launches = []
    instance._launch = lambda kernel, dim, values: instance.launches.append(
        (kernel, dim, len(values))
    )
    return instance


def test_launch_rejects_a_short_first_array_by_name_and_length(dot):
    with pytest.raises(ValueError, match=r"array 'a' has 31 elements but at least 32 are required"):
        dot.launch(_Array(31), _Array(32), _Array(1), 32)
    assert dot.launches == []


def test_launch_rejects_a_short_second_array_by_name_and_length(dot):
    with pytest.raises(ValueError, match=r"array 'b' has 5 elements but at least 32 are required"):
        dot.launch(_Array(32), _Array(5), _Array(1), 32)
    assert dot.launches == []


def test_launch_rejects_an_output_that_cannot_hold_the_result(dot):
    with pytest.raises(ValueError, match=r"array 'output' has 0 elements but at least 1 are required"):
        dot.launch(_Array(32), _Array(32), _Array(0), 32)
    assert dot.launches == []


def test_launch_accepts_exact_and_longer_arrays_and_forwards_the_launch(dot):
    dot.launch(_Array(32), _Array(32), _Array(1), 32)
    dot.launch(_Array(1000), _Array(64), _Array(4), 32)

    assert dot.launches == [(3, 32, 4), (3, 32, 4)]


def test_launch_accepts_a_zero_count_on_empty_arrays(dot):
    dot.launch(_Array(0), _Array(0), _Array(1), 0)

    assert dot.launches == [(3, 0, 4)]


def test_launch_active_rejects_arrays_shorter_than_its_launch_dim(dot):
    # The active count is read on device and is at most ``dim``; ``dim`` is
    # the largest range the kernel can touch, so it is what the host can check.
    with pytest.raises(ValueError, match=r"array 'a' has 1023 elements but at least 1024 are required"):
        dot.launch_active(_Array(1023), _Array(1024), _Array(1), _Array(1), 1024)
    with pytest.raises(ValueError, match=r"array 'b' has 7 elements but at least 1024 are required"):
        dot.launch_active(_Array(1024), _Array(7), _Array(1), _Array(1), 1024)
    assert dot.launches == []


def test_launch_active_rejects_missing_count_and_output_storage(dot):
    with pytest.raises(ValueError, match=r"array 'num_leaves' has 0 elements but at least 1 are required"):
        dot.launch_active(_Array(1024), _Array(1024), _Array(1), _Array(0), 1024)
    with pytest.raises(ValueError, match=r"array 'output' has 0 elements but at least 1 are required"):
        dot.launch_active(_Array(1024), _Array(1024), _Array(0), _Array(1), 1024)
    assert dot.launches == []


def test_launch_active_accepts_exact_arrays_and_forwards_the_launch(dot):
    dot.launch_active(_Array(1024), _Array(1024), _Array(1), _Array(1), 1024)

    assert dot.launches == [(4, 1024, 5)]


def test_errors_name_the_method_so_the_failing_call_site_is_findable(dot):
    with pytest.raises(ValueError, match=r"^CudaDotProduct\.launch: "):
        dot.launch(_Array(1), _Array(32), _Array(1), 32)
    with pytest.raises(ValueError, match=r"^CudaDotProduct\.launch_active: "):
        dot.launch_active(_Array(1), _Array(1024), _Array(1), _Array(1), 1024)


def test_check_uses_real_warp_array_size_for_one_and_multi_dimensional_arrays(dot):
    flat = wp.zeros(32, dtype=float, device="cpu")
    grid = wp.zeros((4, 8), dtype=float, device="cpu")  # 32 elements, linear layout
    short = wp.zeros(31, dtype=float, device="cpu")
    out = wp.zeros(1, dtype=float, device="cpu")

    dot.launch(flat, grid, out, 32)
    assert dot.launches == [(3, 32, 4)]
    with pytest.raises(ValueError, match=r"array 'a' has 31 elements but at least 32 are required"):
        dot.launch(short, flat, out, 32)


def test_the_message_states_the_count_it_was_checked_against(dot):
    with pytest.raises(ValueError) as caught:
        dot.launch(_Array(31), _Array(32), _Array(1), 32)

    assert "count=32" in str(caught.value)
    with pytest.raises(ValueError) as caught:
        dot.launch_active(_Array(1023), _Array(1024), _Array(1), _Array(1), 1024)
    assert "dim=1024" in str(caught.value)
