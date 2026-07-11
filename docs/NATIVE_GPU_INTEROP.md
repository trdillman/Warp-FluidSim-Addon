# Native GPU Interop

This branch adds an opt-in particle presentation path that keeps live Warp/CUDA positions on the GPU.
Blender creates an exportable Vulkan VBO; CUDA imports and maps it once; each frame uses a device-to-device
`wp.copy` and two directional timeline semaphores; Blender draws the same allocation with `GPUBatch`.

## Supported lane

- Branch: `codex/native-gpu-interop`
- Windows, Vulkan, NVIDIA CUDA/Warp
- Fixed-capacity liquid particles, host-known active count
- Full display fraction and no particle attribute visualization

The Surface, whitewater, fractional display, attribute visualization, cache, export, and render-engine
paths keep their existing CPU behavior. If native initialization or a capability check fails, the add-on
records `session().native_interop_error` and automatically rebuilds the normal point-cloud datablock.

## Use

Open `demos/native_gpu_interop_particles.blend` in an interop-enabled Blender build started with the
Vulkan backend. In Fluid Simulation, click **Initialize Solver**, then enable **Live Preview** or play the
timeline. `Native GPU Interop` is already enabled in the demo.

The stock Blender 5.1 binary does not contain the private `GPUVertBuf` export methods used by this branch;
on stock or non-Vulkan builds the automatic fallback remains available.

## Proven results

Target: NVIDIA GeForce RTX 4080 SUPER, CUDA Toolkit 12.9, Warp 1.12.1, Vulkan.

- Direct proof: 4,225 CUDA-written positions produced 4,225 blue framebuffer pixels.
- Real solver: two frames, 26,511 particles, three one-time imports, one persistent map, zero bulk host
  position bytes.
- Forced failure: native initialization error restored a populated Blender `PointCloud`.
- Benchmark at 24,840 particles: CPU datablock median 1.578 ms; native enqueue/draw median 0.385 ms;
  4.10x median speedup; 5,961,600 CPU position bytes avoided over 20 samples.
- Demo gate: saved scene reopened, initialized 500,000-particle capacity, advanced, and published 67,584
  particles with zero bulk host-position bytes.

Evidence:

- `docs/evidence/native_interop/acceptance.json`
- `docs/evidence/native_interop/benchmark.json`
- `docs/evidence/native_interop/warp_cuda_vulkan_points.png`
- `demos/native_gpu_interop_particles.blend`

## Reproduce

Run these scripts with the interop-enabled Blender executable, `--gpu-backend vulkan`, system Python
environment enabled, this repository on `PYTHONPATH`, and the Blender 5.1 Python 3.13 NumPy plus Warp
packages available:

```text
tests/blender_native_interop_acceptance.py
tests/blender_native_interop_solver_gate.py
tests/blender_native_interop_benchmark.py
tests/blender_native_interop_demo_gate.py
```

The benchmark measures steady-state presentation, not the fluid solve. Native samples enqueue the CUDA
copy, timeline handoff, and Vulkan draw; one final framebuffer read drains the queue after all samples.
