from types import SimpleNamespace

import pytest

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
