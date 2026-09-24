from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.sim import cuda_reduction
from fluidSim_warp_blender51.sim.pressure import PressureSolver


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
    dot._partials = SimpleNamespace(ptr=48)
    dot._ticket = SimpleNamespace(ptr=64)
    array = SimpleNamespace(ptr=16)

    dot.launch(array, array, array, 32)

    assert len(calls) == 1
    assert len(calls[0]) == arity
    # a, b, count, output, then the ordered-combine partials and ticket.
    params = calls[0][6 if arity == 8 else 8]
    assert len(params) == 6


def test_module_cache_uses_runtime_context_identity_and_reference_leases(monkeypatch):
    unloads = []
    synchronizes = []
    runtime = SimpleNamespace(
        core=SimpleNamespace(
            wp_cuda_unload_module=lambda context, module: unloads.append(
                (context, module)
            )
        )
    )
    device_a = SimpleNamespace(context=11, arch=89)
    device_b = SimpleNamespace(context=11, arch=89)
    entry = cuda_reduction._CudaModuleEntry(
        runtime=runtime,
        device=device_a,
        module=22,
        kernel=33,
        references=2,
    )
    monkeypatch.setattr(cuda_reduction.context, "runtime", runtime)
    monkeypatch.setattr(
        cuda_reduction.wp,
        "synchronize_device",
        lambda device: synchronizes.append(device),
    )
    owner = cuda_reduction._CudaModuleCache()
    key_a = cuda_reduction._module_key(device_a)
    key_b = cuda_reduction._module_key(device_b)
    owner.entries[key_a] = entry

    assert key_a == key_b
    owner.release(key_a, entry)
    assert entry.references == 1
    assert unloads == []

    owner.release(key_b, entry)
    owner.release(key_b, entry)

    assert synchronizes == [device_a]
    assert unloads == [(11, 22)]
    assert owner.entries == {}


def test_cuda_dot_product_release_is_idempotent(monkeypatch):
    releases = []
    entry = object()
    dot = cuda_reduction.CudaDotProduct.__new__(cuda_reduction.CudaDotProduct)
    dot._cache_key = (1, 2, 3)
    dot._cache_owner = object()
    dot._entry = entry
    dot._module = 4
    dot._kernel = 5
    monkeypatch.setattr(
        cuda_reduction,
        "_release_kernel",
        lambda key, owned, owner: releases.append((key, owned, owner)),
    )

    dot.release()
    dot.release()

    assert releases == [((1, 2, 3), entry, dot._cache_owner)]
    assert dot._module is None
    assert dot._kernel is None


def test_reset_cuda_module_cache_rejects_active_leases_and_is_idempotent(
    monkeypatch,
):
    owner = cuda_reduction._CudaModuleCache()
    monkeypatch.setattr(cuda_reduction, "_MODULE_CACHE", owner)
    entry = cuda_reduction._CudaModuleEntry(
        runtime=SimpleNamespace(),
        device=SimpleNamespace(),
        module=1,
        kernel=2,
        references=1,
    )
    owner.entries[(1, 2, 3)] = entry

    with pytest.raises(RuntimeError, match="1 active lease"):
        cuda_reduction.reset_cuda_module_cache()

    entry.references = 0
    owner.entries.clear()
    cuda_reduction.reset_cuda_module_cache()
    cuda_reduction.reset_cuda_module_cache()


def test_lease_releases_through_its_original_cache_after_global_replacement(
    monkeypatch,
):
    unloads = []
    synchronizes = []
    owner = cuda_reduction._CudaModuleCache()
    device = SimpleNamespace(context=7, arch=89)
    runtime = SimpleNamespace(
        core=SimpleNamespace(
            wp_cuda_unload_module=lambda context, module: unloads.append(
                (context, module)
            )
        )
    )
    entry = cuda_reduction._CudaModuleEntry(
        runtime=runtime,
        device=device,
        module=8,
        kernel=9,
        references=1,
    )
    key = (id(runtime), 7, 89)
    owner.entries[key] = entry
    dot = cuda_reduction.CudaDotProduct.__new__(cuda_reduction.CudaDotProduct)
    dot._cache_owner = owner
    dot._cache_key = key
    dot._entry = entry
    dot._module = 8
    dot._kernel = 9
    monkeypatch.setattr(
        cuda_reduction.wp,
        "synchronize_device",
        lambda owned: synchronizes.append(owned),
    )
    monkeypatch.setattr(
        cuda_reduction,
        "_MODULE_CACHE",
        cuda_reduction._CudaModuleCache(),
    )

    dot.release()

    assert owner.entries == {}
    assert synchronizes == [device]
    assert unloads == [(7, 8)]


def test_pressure_release_drops_graphs_before_native_module_lease():
    events = []

    class Graphs(dict):
        def clear(self):
            events.append("graphs")
            super().clear()

    solver = PressureSolver.__new__(PressureSolver)
    solver._pcg_dot = SimpleNamespace(
        release=lambda: events.append("native lease"))
    solver._pcg_graphs = Graphs(captured=object())
    solver._pcg_conditional_graphs = Graphs(captured=object())
    solver._pcg_voxel_type = object()
    solver._pcg_apply_cmd = object()
    solver._pcg_update_cmd = object()
    solver._pcg_update_p_cmd = object()
    solver._pcg_alpha_cmds = object()
    solver._pcg_beta_cmds = object()
    solver._pcg_pack_cmds = object()

    solver.release()

    assert events == ["graphs", "graphs", "native lease"]
    assert solver._pcg_dot is None
    assert solver._pcg_apply_cmd is None


def test_release_failure_keeps_cache_entry_and_lease_retryable(monkeypatch):
    unloads = []
    owner = cuda_reduction._CudaModuleCache()
    device = SimpleNamespace(context=12, arch=89)
    runtime = SimpleNamespace(
        core=SimpleNamespace(
            wp_cuda_unload_module=lambda context, module: unloads.append(
                (context, module)
            )
        )
    )
    entry = cuda_reduction._CudaModuleEntry(
        runtime=runtime,
        device=device,
        module=13,
        kernel=14,
        references=1,
    )
    key = (id(runtime), 12, 89)
    owner.entries[key] = entry
    dot = cuda_reduction.CudaDotProduct.__new__(cuda_reduction.CudaDotProduct)
    dot._cache_owner = owner
    dot._cache_key = key
    dot._entry = entry
    dot._module = 13
    dot._kernel = 14
    monkeypatch.setattr(
        cuda_reduction.wp,
        "synchronize_device",
        lambda _device: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        dot.release()

    assert owner.entries[key] is entry
    assert dot._entry is entry
    assert unloads == []

    monkeypatch.setattr(cuda_reduction.wp, "synchronize_device", lambda _device: None)
    dot.release()
    assert owner.entries == {}
    assert unloads == [(12, 13)]


def test_kernel_lookup_failure_unloads_new_module_under_context_guard(
    monkeypatch,
):
    events = []

    class Guard:
        def __enter__(self):
            events.append("guard enter")

        def __exit__(self, *_args):
            events.append("guard exit")

    owner = cuda_reduction._CudaModuleCache()
    device = SimpleNamespace(context=21, arch=89, context_guard=Guard())
    runtime = SimpleNamespace(
        toolkit_version=12080,
        core=SimpleNamespace(
            wp_cuda_get_kernel=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("lookup failed")
            ),
            wp_cuda_unload_module=lambda context, module: events.append(
                ("unload", context, module)
            ),
        ),
    )
    monkeypatch.setattr(cuda_reduction, "_MODULE_CACHE", owner)
    monkeypatch.setattr(cuda_reduction.context, "runtime", runtime)
    monkeypatch.setattr(cuda_reduction.Path, "exists", lambda _path: True)
    monkeypatch.setattr(cuda_reduction.build, "load_cuda", lambda *_args: 22)

    with pytest.raises(RuntimeError, match="lookup failed"):
        cuda_reduction._acquire_kernel(device)

    assert events == ["guard enter", ("unload", 21, 22), "guard exit"]
    assert owner.entries == {}
