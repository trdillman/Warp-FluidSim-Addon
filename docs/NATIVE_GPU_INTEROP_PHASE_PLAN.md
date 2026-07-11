# Native GPU Interop Consumer Phase

## Outcome

FluidSim Warp presents live particles from a Blender-owned Vulkan vertex buffer that Warp/CUDA
writes without copying positions through NumPy or rebuilding a Blender point-cloud datablock.

This phase starts from consumer commit `1956742` on branch `codex/native-gpu-interop`. The original
`codex/performance-hotpaths` checkout remains untouched because it contains later material-response
work.

## Confirmed data path

Current live presentation performs:

```text
Warp particle position array
  -> wp.array.numpy()
  -> NumPy host array
  -> PointCloud/Mesh foreach_set()
  -> Blender draw-cache upload
```

The target path is:

```text
Warp particle position array
  -> CUDA device-to-device publication copy
  -> Blender-owned exportable Vulkan VBO
  -> gpu.types.GPUBatch draw
```

The solver remains authoritative. The shared VBO is a stable presentation allocation, not new
simulation storage.

## Scope

- Windows, Blender Vulkan backend, NVIDIA CUDA, Warp 1.12.
- Particle positions only, fixed allocation capacity, host-known active draw count.
- Two directional Vulkan timeline semaphores.
- Existing point-cloud/datablock presentation remains the automatic fallback.
- Cache, export, surface meshing, render-engine ingestion, and attribute materialization retain their
  existing CPU paths.

Deferred: indirect drawing, GPU bounds, shared velocities/attributes, whitewater, surface topology,
multiple presentation slots, Linux, public provider ABI, and `GPUResidentGeometry`.

## Implementation sequence

| ID | Task | Status | Exit evidence |
| --- | --- | --- | --- |
| C-API-1 | Expose the proven Vulkan VBO export and directional-timeline operations through the existing Python `GPUVertBuf` binding | Complete | Python runtime contract and non-unity Blender build |
| C-CUDA-1 | Import the Win32 allocation and timelines in the add-on, persistently map once, and wrap the pointer as a Warp array | Complete | Runtime import on matching RTX 4080 Super UUID |
| C-PUB-1 | Publish active particle positions with one Warp device copy and no `.numpy()` call | Complete | Two solver frames and transfer counters |
| C-DRAW-1 | Draw the shared VBO from a 3D-view draw handler using `GPUBatch.draw_range` | Complete | 4,225 verified framebuffer pixels |
| C-FALLBACK-1 | Select native interop only when every capability succeeds; retain current datablock path otherwise | Complete | Forced-unavailable populated `PointCloud` gate |
| C-BENCH-1 | Compare CPU presentation and native presentation with identical simulation state and particle counts | Complete | Tracked median/p95/bytes JSON report |
| C-DEMO-1 | Save a demo scene configured for native particle presentation | Complete | Saved scene reopened and published 67,584 particles |

## Runtime protocol

For publication value `N`:

1. Blender records a buffer release to `VK_QUEUE_FAMILY_EXTERNAL`, submits it, and signals release
   timeline `N`.
2. CUDA stream waits for release `N`.
3. Warp copies/transforms active positions into the mapped `float4` VBO.
4. CUDA stream signals completion timeline `N`.
5. The draw callback records the external-to-graphics acquire, draws `active_count` vertices, and
   submits with a completion-timeline wait for `N`.

No steady-state `GPU_finish`, Vulkan queue/device idle, CUDA context/device synchronization, position
readback, per-frame import, or per-frame mapping is allowed. A bounded teardown wait is allowed after
the final signal.

## TDD order

1. Red Python API contract: exportable VBO creation and interop descriptor are unavailable.
2. Green the smallest existing-binding extension.
3. Red consumer contract: native presenter cannot initialize from the descriptor.
4. Green CUDA import, persistent mapping, and Warp pointer wrapping.
5. Red presentation contract: live update still calls `particle_positions()`.
6. Green native publication and draw, keeping fallback unchanged.
7. Add runtime pixel proof, benchmark, and demo-scene contract.

## Acceptance gates

- Add-on runs from branch `codex/native-gpu-interop` in a Vulkan Blender build containing the proven
  interop implementation.
- RTX 4080 Super UUID matches between Blender Vulkan and CUDA.
- Allocation and both timelines import once and remain mapped for the presenter lifetime.
- Live native particle presentation executes zero bulk position readbacks and zero datablock rebuilds.
- Draw uses the same Blender-owned VBO written by Warp.
- Correct pixels are captured from a real Blender viewport or offscreen GPU path.
- Baseline and native benchmarks use the same scene, frames, particle counts, and solver settings.
- Existing Python suite and production gate remain green; forced fallback remains functional.
- Demo `.blend`, preview/capture, benchmark JSON, and environment record are committed on the branch.

## Stop conditions

Stop and record exact evidence rather than substituting another mechanism if:

- the Blender Python GPU API cannot keep acquire, draw, wait, and submission in one ordered graph;
- target Vulkan external-memory/timeline capabilities regress;
- CUDA/Vulkan UUIDs do not match;
- validation reports queue ownership, visibility, handle lifetime, or synchronization errors.

## Completed evidence

Target run: RTX 4080 Super, Vulkan, CUDA Toolkit 12.9, Warp 1.12.1. The real solver published 26,511
particles across two frames with three imports, one persistent mapping, and zero bulk host-position
bytes. The final 24,840-particle presentation run measured a 1.578 ms CPU median and 0.385 ms native
median (4.10x). See `docs/evidence/native_interop/` and `demos/native_gpu_interop_particles.blend`.
