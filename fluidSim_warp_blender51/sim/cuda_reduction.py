"""Allocation-free CUDA reductions used by captured solver graphs."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
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

__device__ float fluid_atomic_min_float(float *address, float value)
{
  int *bits = reinterpret_cast<int *>(address);
  int observed = *bits;
  while (value < __int_as_float(observed)) {
    const int assumed = observed;
    observed = atomicCAS(bits, assumed, __float_as_int(value));
    if (observed == assumed) {
      break;
    }
  }
  return __int_as_float(observed);
}

__device__ float fluid_atomic_max_float(float *address, float value)
{
  int *bits = reinterpret_cast<int *>(address);
  int observed = *bits;
  while (value > __int_as_float(observed)) {
    const int assumed = observed;
    observed = atomicCAS(bits, assumed, __float_as_int(value));
    if (observed == assumed) {
      break;
    }
  }
  return __int_as_float(observed);
}

extern "C" __global__ void fluid_species_postcompact_reduce(
    const float *volume, const float4 *concentration, const int *count,
    float *receipt, int after_offset, int minimum_offset,
    int max_loading_offset)
{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int n = max(0, *count);
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float minimum = 3.402823466e38f;
  float maximum = -3.402823466e38f;
  int seen = 0;
  for (int i = index; i < n; i += stride) {
    const float4 value = concentration[i];
    sum0 += volume[i] * value.x;
    sum1 += volume[i] * value.y;
    minimum = fminf(minimum, fminf(value.x, value.y));
    maximum = fmaxf(maximum, value.x + value.y);
    seen = 1;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum0 += __shfl_down_sync(0xffffffff, sum0, offset);
    sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
    minimum = fminf(
        minimum, __shfl_down_sync(0xffffffff, minimum, offset));
    maximum = fmaxf(
        maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    seen += __shfl_down_sync(0xffffffff, seen, offset);
  }
  __shared__ float warp_sum0[32];
  __shared__ float warp_sum1[32];
  __shared__ float warp_minimum[32];
  __shared__ float warp_maximum[32];
  __shared__ int warp_seen[32];
  if (lane == 0) {
    warp_sum0[warp] = sum0;
    warp_sum1[warp] = sum1;
    warp_minimum[warp] = minimum;
    warp_maximum[warp] = maximum;
    warp_seen[warp] = seen;
  }
  __syncthreads();
  if (warp == 0) {
    const int warp_count = blockDim.x >> 5;
    sum0 = lane < warp_count ? warp_sum0[lane] : 0.0f;
    sum1 = lane < warp_count ? warp_sum1[lane] : 0.0f;
    minimum = lane < warp_count
        ? warp_minimum[lane] : 3.402823466e38f;
    maximum = lane < warp_count
        ? warp_maximum[lane] : -3.402823466e38f;
    seen = lane < warp_count ? warp_seen[lane] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum0 += __shfl_down_sync(0xffffffff, sum0, offset);
      sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
      minimum = fminf(
          minimum, __shfl_down_sync(0xffffffff, minimum, offset));
      maximum = fmaxf(
          maximum, __shfl_down_sync(0xffffffff, maximum, offset));
      seen += __shfl_down_sync(0xffffffff, seen, offset);
    }
    if (lane == 0 && seen > 0) {
      atomicAdd(receipt + after_offset, sum0);
      atomicAdd(receipt + after_offset + 1, sum1);
      fluid_atomic_min_float(receipt + minimum_offset, minimum);
      fluid_atomic_max_float(receipt + max_loading_offset, maximum);
    }
  }
}
"""

_MODULES = {}


def _build_cuda(cu_path, arch, output_path):
    kwargs = {}
    if "pch_dir" in inspect.signature(build.build_cuda).parameters:
        kwargs["pch_dir"] = context.runtime.get_nvrtc_pch_dir()
    build.build_cuda(cu_path, arch, output_path, **kwargs)


def _load_kernels(device):
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
            _build_cuda(str(source_path), device.arch, temporary)
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
        dot_kernel = None
        species_kernel = None
        if module:
            dot_kernel = context.runtime.core.wp_cuda_get_kernel(
                device.context, module, b"fluid_dot_reduce")
            species_kernel = context.runtime.core.wp_cuda_get_kernel(
                device.context, module, b"fluid_species_postcompact_reduce")
        if dot_kernel and species_kernel:
            _MODULES[key] = (device, module, dot_kernel, species_kernel)
            return module, dot_kernel, species_kernel
        if module:
            context.runtime.core.wp_cuda_unload_module(device.context, module)
        if attempt == 0:
            cubin_path.unlink(missing_ok=True)
    raise RuntimeError("Failed to load the native CUDA reduction kernels")


class CudaDotProduct:
    """Launch one persistent, graph-capturable dot-product reduction."""

    def __init__(self, device):
        device = wp.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("CudaDotProduct requires a CUDA device")
        self.device = device
        self._module, self._kernel, _species_kernel = _load_kernels(device)
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


class CudaSpeciesBoundsReduction:
    """Reduce actual compacted species state without allocating scratch."""

    backend = "native_cuda"

    def __init__(self, device):
        device = wp.get_device(device)
        if not device.is_cuda:
            raise RuntimeError(
                "CudaSpeciesBoundsReduction requires a CUDA device")
        self.device = device
        self._module, _dot_kernel, self._kernel = _load_kernels(device)
        self._max_blocks = max(1, int(device.sm_count) * 2)

    def launch(
            self, volume, concentration, count, receipt, *, launch_bound,
            after_offset, minimum_offset, max_loading_offset):
        volume_ptr = ctypes.c_void_p(volume.ptr)
        concentration_ptr = ctypes.c_void_p(concentration.ptr)
        count_ptr = ctypes.c_void_p(count.ptr)
        receipt_ptr = ctypes.c_void_p(receipt.ptr)
        after_value = ctypes.c_int(after_offset)
        minimum_value = ctypes.c_int(minimum_offset)
        max_loading_value = ctypes.c_int(max_loading_offset)
        params = (ctypes.c_void_p * 7)(
            ctypes.addressof(volume_ptr),
            ctypes.addressof(concentration_ptr),
            ctypes.addressof(count_ptr),
            ctypes.addressof(receipt_ptr),
            ctypes.addressof(after_value),
            ctypes.addressof(minimum_value),
            ctypes.addressof(max_loading_value),
        )
        launch = context.runtime.core.wp_cuda_launch_kernel
        args = (
            self.device.context,
            self._kernel,
            max(1, int(launch_bound)),
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
            raise RuntimeError(
                f"Native CUDA species reduction launch failed: {result}")
