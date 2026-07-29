"""Allocation-free CUDA reductions used by captured solver graphs."""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
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

_MODULES = {}


def _load_kernel(device):
    key = (id(device), int(device.context), device.arch)
    cached = _MODULES.get(key)
    if cached is not None:
        return cached[1:]

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
            build.build_cuda(
                str(source_path),
                device.arch,
                temporary,
            )
            os.replace(temporary, cubin_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    for attempt in range(2):
        if not cubin_path.exists():
            compile_kernel()
        module = build.load_cuda(str(cubin_path), device)
        kernel = None
        if module:
            kernel = context.runtime.core.wp_cuda_get_kernel(
                device.context, module, b"fluid_dot_reduce")
        if kernel:
            _MODULES[key] = (device, module, kernel)
            return module, kernel
        if module:
            context.runtime.core.wp_cuda_unload_module(device.context, module)
        if attempt == 0:
            cubin_path.unlink(missing_ok=True)
    raise RuntimeError("Failed to load the native CUDA dot-product kernel")


class CudaDotProduct:
    """Launch one persistent, graph-capturable dot-product reduction."""

    def __init__(self, device):
        device = wp.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("CudaDotProduct requires a CUDA device")
        self.device = device
        self._module, self._kernel = _load_kernel(device)
        self._max_blocks = max(1, int(device.sm_count) * 2)

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
        result = context.runtime.core.wp_cuda_launch_kernel(
            self.device.context,
            self._kernel,
            count,
            self._max_blocks,
            256,
            0,
            params,
            self.device.stream.cuda_stream,
        )
        if result != 0:
            raise RuntimeError(f"Native CUDA dot-product launch failed: {result}")
