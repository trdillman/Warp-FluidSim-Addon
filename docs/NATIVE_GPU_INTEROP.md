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
- A three-run rebaseline against one unchanged executable measured presentation-only speedups of
  1.30x-1.51x at 25,000 particles, 1.69x-1.72x at 250,000, and 2.23x-2.28x at
  1,000,000. These include normal View3D redraw but not the fluid solve. The 250,000-particle result
  does not satisfy the historical 2x A-19 threshold.
- The matching three-run 600,000-particle Dam Break baseline has a median 1.139x whole-frame speedup,
  a 12.2% frame-time reduction, and a 27.9% process-CPU reduction. A final post-change functional run
  on the same source measured 1.109x and a 29.3% process-CPU reduction; it validates the instrumented
  code path but does not replace the three-run performance baseline.
- Three fresh-process, 250,000-particle idealized no-repack trials measured 1.094x, 0.985x, and
  1.018x relative to the current packed path. Cross-process spread was 3.60% packed and 4.30%
  no-repack. The CUDA position-pack stage had a 0.01024 ms cross-run median and a 0.01127 ms maximum,
  less than 0.1% of the measured 12.9-14.3 ms viewport frame time. Removing that stage cannot
  recover the missing 2x result, so the current vec3-to-vec4 publication remains the supported path.
  Viewport drawing/redraw work, not the packing kernel, dominates this end-to-end measurement.
- The 50,000-particle solver-plus-presentation regression, excluding View3D redraw, is not a stable
  speedup result. Recorded runs measured native/CPU ratios of 0.772 and 1.100, and another observed run
  reached 1.120; it remains a regression detector, not performance evidence.
- A synchronized standalone solver profile averages 25.457 ms at 58,575 active particles; pressure is
  the largest measured phase at 10.982 ms. It is a separate solver workload, not the Dam Break A/B.
- Demo gate: saved scene reopened, initialized 500,000-particle capacity, advanced, and published 67,584
  particles with zero bulk host-position bytes.

Evidence:

- `docs/evidence/native_interop/acceptance.json`
- `docs/evidence/native_interop/benchmark.json`
- `docs/evidence/native_interop/dam-break-profile-main.json`
- `docs/evidence/native_interop/gpu-solver-profile-main.json`
- `docs/evidence/native_interop/rebaseline-20260715-db2bc63c/report.md`
- `docs/evidence/native_interop/stage-profile-20260716-db2bc63c/report.md`
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
tests/blender_native_interop_dam_break_profile.py
tests/blender_native_interop_stage_profile.py
tests/blender_native_interop_demo_gate.py
```

The scale benchmark measures steady-state presentation, not the fluid solve. The Dam Break profiler is
the authoritative whole-frame comparison and includes solver, presentation, and View3D redraw. Its
individual phase timings are host boundaries and may overlap GPU work; only whole-frame time is used for
the end-to-end speedup claim.

`blender_native_interop_stage_profile.py` is a benchmark-only diagnostic. Profiling is disabled in
normal operation. When explicitly enabled, CUDA events measure external wait, position packing,
metadata publication, and total publication through the completion signal. Event results are collected
only after `lease.submit()`, so diagnostics cannot leave a lease transaction pending. The no-repack mode
skips only the counted position-pack kernel; it retains the external wait, GPU-authored count, bounds,
draw command, completion signal, visible framebuffer proof, and ordinary PointCloud draw path.
