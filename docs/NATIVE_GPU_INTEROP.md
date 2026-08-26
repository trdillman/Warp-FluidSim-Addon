# Native GPU Interop

## Status

This branch contains the Windows/NVIDIA consumer for Blender's native PointCloud interop path. The
implementation is release-candidate code until every gate in `GPU_BUFFER_INTEROP_ACCEPTANCE.md` has
been rerun against one unchanged packaged build.

Product authority, active sequencing, and release gates live in the Blender source tree:

- `GOALS.md`
- `GPU_BUFFER_INTEROP_NATIVE_FEATURE_PLAN.md`
- `GPU_BUFFER_INTEROP_ACCEPTANCE.md`

Blender creates exportable Vulkan VBOs and grants each one managed `GPUVertBufExternalLease`. CUDA
imports each generation once and maps it persistently. Directional timeline semaphores transfer queue
ownership each frame, and Blender's ordinary PointCloud/Draw Manager path draws the same allocations.

The supported Python boundary is:

```python
lease = vbo.external_lease_acquire()
descriptor = lease.descriptor_get()
lease.release(value)
# CUDA waits, writes, and signals completion.
lease.acquire(value)
lease.submit(value)
# After CUDA destroys its imported memory and semaphores:
lease.close()
```

One VBO grants one terminal lease. The lease strongly owns the VBO, carries a unique generation, and
rejects duplicate acquisition, stale or post-close use, timeline regression, unmatched submission, and
double close. Blender owns the exported Win32 handles; the consumer never calls `CloseHandle`.

## Supported lane

- Branch: `codex/native-gpu-interop`
- Windows, Vulkan, NVIDIA CUDA/Warp
- Two persistent presentation slots with safe capacity-generation replacement
- GPU-written active count, draw command, bounds, position/radius, velocity, and supported color modes

Surface, whitewater, cache, export, and render-engine ingestion keep their existing CPU behavior. If
native initialization, descriptor validation, device matching, import, or publication fails, the add-on
records the persistent reason and byte cost and automatically rebuilds the normal PointCloud datablock.

## Use

Open `demos/native_gpu_interop_particles.blend` in an interop-enabled Blender build started with the
Vulkan backend. In Fluid Simulation, click **Initialize Solver**, then enable **Live Preview** or play the
timeline. `Native GPU Interop` is already enabled in the demo.

Stock Blender does not contain this custom lease API; stock, OpenGL, and unsupported configurations use
the automatic CPU fallback.

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

Run these scripts with the packaged interop-enabled Blender executable and `--gpu-backend vulkan`. The
release build supplies pinned private Warp and NumPy packages; runtime must not borrow another Blender
installation, a source worktree, user site-packages, or a separately installed CUDA Toolkit:

```text
tests/blender_native_interop_acceptance.py
tests/blender_native_interop_solver_gate.py
tests/blender_native_interop_benchmark.py
tests/blender_native_interop_demo_gate.py
```

The benchmark measures steady-state presentation, not the fluid solve. Native samples enqueue the CUDA
copy, timeline handoff, and Vulkan draw; one final framebuffer read drains the queue after all samples.
