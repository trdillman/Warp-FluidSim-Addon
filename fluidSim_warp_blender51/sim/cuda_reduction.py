"""Allocation-free CUDA reductions used by captured solver graphs."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import os
import tempfile
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import warp as wp
from warp._src import build, context

_CUDA_SOURCE = r"""
extern "C" __global__ void fluid_dot_reduce(
    const float *a, const float *b, int count, float *output)
{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  float sum = 0.0f;
  for (int i = index; i < count; i += stride) {
    sum += a[i] * b[i];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  __shared__ float warp_sums[32];
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
      atomicAdd(output, sum);
    }
  }
}
"""

@dataclass
class _CudaModuleEntry:
    runtime: object
    device: object
    module: object
    kernel: object
    references: int = 0


class _CudaModuleCache:
    """Own native modules independently of this module's reloadable globals."""

    def __init__(self):
        self.entries = {}
        self.lock = threading.RLock()

    def release(self, key, entry):
        with self.lock:
            cached = self.entries.get(key)
            if cached is not entry:
                return
            if cached.references > 1:
                cached.references -= 1
                return
            wp.synchronize_device(cached.device)
            _unload_native_module(
                cached.runtime, cached.device, cached.module)
            del self.entries[key]
            cached.references = 0

    def reset(self):
        with self.lock:
            active = sum(entry.references for entry in self.entries.values())
            if active:
                raise RuntimeError(
                    "Cannot reset native CUDA module cache with "
                    f"{active} active lease(s)")
            for key, entry in list(self.entries.items()):
                wp.synchronize_device(entry.device)
                _unload_native_module(entry.runtime, entry.device, entry.module)
                del self.entries[key]


_MODULE_CACHE = _CudaModuleCache()


def _build_cuda(cu_path, arch, output_path):
    kwargs = {}
    if "pch_dir" in inspect.signature(build.build_cuda).parameters:
        kwargs["pch_dir"] = context.runtime.get_nvrtc_pch_dir()
    build.build_cuda(cu_path, arch, output_path, **kwargs)


def _module_key(device):
    """Identify the native runtime/context, independent of device wrappers."""
    return (id(context.runtime), int(device.context), int(device.arch))


def _unload_native_module(runtime, device, module):
    guard = getattr(device, "context_guard", None)
    with guard if guard is not None else nullcontext():
        runtime.core.wp_cuda_unload_module(device.context, module)


def _acquire_kernel(device):
    key = _module_key(device)
    owner = _MODULE_CACHE
    with owner.lock:
        cached = owner.entries.get(key)
        if cached is not None:
            cached.references += 1
            return owner, key, cached

    identity = (
        _CUDA_SOURCE,
        wp.__version__,
        str(context.runtime.toolkit_version),
    )
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
    cache_dir = Path(wp.config.kernel_cache_dir) / "fluidSim_cuda_reduction"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = cache_dir / f"dot_{digest}.cu"
    cubin_path = cache_dir / f"dot_{digest}_sm{device.arch}.cubin"

    def compile_kernel():
        source_path.write_text(_CUDA_SOURCE, encoding="utf-8")
        handle, temporary = tempfile.mkstemp(
            prefix=f"dot_{digest}_", suffix=".cubin", dir=cache_dir)
        os.close(handle)
        try:
            _build_cuda(str(source_path), device.arch, temporary)
            os.replace(temporary, cubin_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    with owner.lock:
        cached = owner.entries.get(key)
        if cached is not None:
            cached.references += 1
            return owner, key, cached
        for attempt in range(2):
            if not cubin_path.exists():
                compile_kernel()
            module = build.load_cuda(str(cubin_path), device)
            kernel = None
            try:
                if module:
                    kernel = context.runtime.core.wp_cuda_get_kernel(
                        device.context, module, b"fluid_dot_reduce")
            except Exception:
                if module:
                    _unload_native_module(context.runtime, device, module)
                raise
            if kernel:
                entry = _CudaModuleEntry(
                    runtime=context.runtime,
                    device=device,
                    module=module,
                    kernel=kernel,
                    references=1,
                )
                owner.entries[key] = entry
                return owner, key, entry
            if module:
                _unload_native_module(context.runtime, device, module)
            if attempt == 0:
                cubin_path.unlink(missing_ok=True)
    raise RuntimeError("Failed to load the native CUDA dot-product kernel")


def _release_kernel(key, entry, owner=None):
    (owner or _MODULE_CACHE).release(key, entry)


def reset_cuda_module_cache():
    """Unload orphaned native modules after Solver Session teardown.

    Active leases make reset unsafe because captured CUDA graphs may still
    reference their kernels. Callers must release the Solver Session first.
    """
    _MODULE_CACHE.reset()


class CudaDotProduct:
    """Launch one persistent, graph-capturable dot-product reduction."""

    def __init__(self, device):
        device = wp.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("CudaDotProduct requires a CUDA device")
        self.device = device
        self._cache_owner, self._cache_key, self._entry = _acquire_kernel(device)
        self._module = self._entry.module
        self._kernel = self._entry.kernel
        self._max_blocks = max(1, int(device.sm_count) * 2)

    def release(self):
        """Release this reduction's native-module lease exactly once."""
        entry = getattr(self, "_entry", None)
        if entry is None:
            return
        _release_kernel(self._cache_key, entry, self._cache_owner)
        self._entry = None
        self._module = None
        self._kernel = None

    def launch(self, a, b, output, count):
        a_ptr = ctypes.c_void_p(a.ptr)
        b_ptr = ctypes.c_void_p(b.ptr)
        count_value = ctypes.c_int(count)
        output_ptr = ctypes.c_void_p(output.ptr)
        params = (ctypes.c_void_p * 4)(
            ctypes.addressof(a_ptr),
            ctypes.addressof(b_ptr),
            ctypes.addressof(count_value),
            ctypes.addressof(output_ptr),
        )
        launch = context.runtime.core.wp_cuda_launch_kernel
        args = (
            self.device.context,
            self._kernel,
            count,
            self._max_blocks,
            256,
            0,
            params,
            self.device.stream.cuda_stream,
        )
        if len(getattr(launch, "argtypes", ()) or ()) >= 11:
            args = args[:5] + (1, 0, 0) + args[6:] + (None,)
        result = launch(*args)
        if result != 0:
            raise RuntimeError(f"Native CUDA dot-product launch failed: {result}")
