import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim import cuda_reduction


def test_build_cuda_supplies_required_warp_115_pch(monkeypatch):
    seen = {}

    def build_cuda(cu_path, arch, output_path, *, pch_dir):
        seen["args"] = (cu_path, arch, output_path, pch_dir)

    monkeypatch.setattr(cuda_reduction.build, "build_cuda", build_cuda)
    monkeypatch.setattr(
        cuda_reduction.context,
        "runtime",
        SimpleNamespace(get_nvrtc_pch_dir=lambda: "warp-pch"),
    )

    cuda_reduction._build_cuda("kernel.cu", 89, "kernel.cubin")

    assert seen["args"] == ("kernel.cu", 89, "kernel.cubin", "warp-pch")


def test_build_cuda_omits_removed_warp_116_pch(monkeypatch):
    seen = {}

    def build_cuda(cu_path, arch, output_path):
        seen["args"] = (cu_path, arch, output_path)

    monkeypatch.setattr(cuda_reduction.build, "build_cuda", build_cuda)

    cuda_reduction._build_cuda("kernel.cu", 89, "kernel.cubin")

    assert seen["args"] == ("kernel.cu", 89, "kernel.cubin")


@pytest.mark.parametrize("arity", [8, 11])
def test_launch_supports_warp_115_and_116_cuda_abis(monkeypatch, arity):
    calls = []

    def launch(*args):
        if len(args) != arity:
            raise TypeError(f"expected {arity} arguments, got {len(args)}")
        calls.append(args)
        return 0

    launch.argtypes = [None] * arity
    monkeypatch.setattr(
        cuda_reduction.context,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(wp_cuda_launch_kernel=launch)),
    )

    dot = cuda_reduction.CudaDotProduct.__new__(
        cuda_reduction.CudaDotProduct)
    dot.device = SimpleNamespace(
        context=1,
        stream=SimpleNamespace(cuda_stream=2),
    )
    dot._kernel = 3
    dot._max_blocks = 4
    array = SimpleNamespace(ptr=16)

    dot.launch(array, array, array, 32)

    assert len(calls) == 1
    assert len(calls[0]) == arity


@pytest.mark.parametrize("arity", [8, 11])
def test_species_reduction_launch_passes_authoritative_device_count_pointer(
        monkeypatch, arity):
    seen = {}

    def launch(*args):
        if len(args) != arity:
            raise TypeError(f"expected {arity} arguments, got {len(args)}")
        params = args[6] if arity == 8 else args[8]

        def parameter(index, ctype):
            return ctypes.cast(
                params[index], ctypes.POINTER(ctype)).contents.value

        seen.update(
            dim=args[2],
            volume=parameter(0, ctypes.c_void_p),
            concentration=parameter(1, ctypes.c_void_p),
            count=parameter(2, ctypes.c_void_p),
            receipt=parameter(3, ctypes.c_void_p),
            after=parameter(4, ctypes.c_int),
            minimum=parameter(5, ctypes.c_int),
            max_loading=parameter(6, ctypes.c_int),
        )
        return 0

    launch.argtypes = [None] * arity
    monkeypatch.setattr(
        cuda_reduction.context,
        "runtime",
        SimpleNamespace(core=SimpleNamespace(wp_cuda_launch_kernel=launch)),
    )

    reducer = cuda_reduction.CudaSpeciesBoundsReduction.__new__(
        cuda_reduction.CudaSpeciesBoundsReduction)
    reducer.device = SimpleNamespace(
        context=1,
        stream=SimpleNamespace(cuda_stream=2),
    )
    reducer._kernel = 3
    reducer._max_blocks = 4
    volume = SimpleNamespace(ptr=16)
    concentration = SimpleNamespace(ptr=32)
    count = SimpleNamespace(ptr=48)
    receipt = SimpleNamespace(ptr=64)

    reducer.launch(
        volume, concentration, count, receipt,
        launch_bound=200_000, after_offset=6, minimum_offset=8,
        max_loading_offset=9)

    assert seen == {
        "dim": 200_000,
        "volume": 16,
        "concentration": 32,
        "count": 48,
        "receipt": 64,
        "after": 6,
        "minimum": 8,
        "max_loading": 9,
    }


def test_species_reduction_cuda_source_uses_hierarchy_and_float_cas():
    source = cuda_reduction._CUDA_SOURCE

    assert "fluid_species_postcompact_reduce" in source
    assert "const int *count" in source
    assert "atomicCAS" in source
    assert "__shfl_down_sync" in source


def _cuda_device_or_skip():
    wp.init()
    try:
        return wp.get_device("cuda:0")
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")


def _receipt_seed(device):
    values = np.zeros(11, dtype=np.float32)
    values[8] = np.finfo(np.float32).max
    return wp.array(values, dtype=wp.float32, device=device)


def test_native_species_reduction_uses_device_count_and_ignores_poisoned_tail():
    device = _cuda_device_or_skip()
    reducer = cuda_reduction.CudaSpeciesBoundsReduction(device)
    volume = wp.array(
        (0.25, 0.75, 1000.0, 1000.0), dtype=wp.float32, device=device)
    concentration = wp.array(
        ((0.1, 0.2, 0.0, 0.0),
         (0.5, 0.25, 0.0, 0.0),
         (100.0, 100.0, 0.0, 0.0),
         (100.0, 100.0, 0.0, 0.0)), dtype=wp.vec4, device=device)
    count = wp.array((2,), dtype=wp.int32, device=device)
    receipt = _receipt_seed(device)

    reducer.launch(
        volume, concentration, count, receipt, launch_bound=4,
        after_offset=6, minimum_offset=8, max_loading_offset=9)
    values = receipt.numpy()

    np.testing.assert_allclose(values[6:8], (0.4, 0.2375), rtol=1.0e-6)
    assert values[8] == pytest.approx(0.1)
    assert values[9] == pytest.approx(0.75)


def test_native_species_reduction_zero_count_preserves_empty_bounds_sentinel():
    device = _cuda_device_or_skip()
    reducer = cuda_reduction.CudaSpeciesBoundsReduction(device)
    volume = wp.array((1000.0,), dtype=wp.float32, device=device)
    concentration = wp.array(
        ((100.0, 100.0, 0.0, 0.0),), dtype=wp.vec4, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    receipt = _receipt_seed(device)

    reducer.launch(
        volume, concentration, count, receipt, launch_bound=1,
        after_offset=6, minimum_offset=8, max_loading_offset=9)
    values = receipt.numpy()

    np.testing.assert_array_equal(values[6:8], (0.0, 0.0))
    assert values[8] == np.finfo(np.float32).max
    assert values[9] == 0.0


def test_native_species_reduction_is_graph_capturable_with_persistent_arrays():
    device = _cuda_device_or_skip()
    reducer = cuda_reduction.CudaSpeciesBoundsReduction(device)
    volume = wp.array((0.25, 0.75), dtype=wp.float32, device=device)
    concentration = wp.array(
        ((0.1, 0.2, 0.0, 0.0), (0.5, 0.25, 0.0, 0.0)),
        dtype=wp.vec4, device=device)
    count = wp.array((2,), dtype=wp.int32, device=device)
    receipt = _receipt_seed(device)

    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as capture:
        reducer.launch(
            volume, concentration, count, receipt, launch_bound=2,
            after_offset=6, minimum_offset=8, max_loading_offset=9)
    wp.capture_launch(capture.graph)
    values = receipt.numpy()

    np.testing.assert_allclose(values[6:8], (0.4, 0.2375), rtol=1.0e-6)
    assert values[8] == pytest.approx(0.1)
    assert values[9] == pytest.approx(0.75)
