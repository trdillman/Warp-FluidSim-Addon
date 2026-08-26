# Native GPU Interop Viewport Attribution

Status: `awaiting_user_review`

## Decision

Retain the current Blender interop publication and PointCloud draw implementation. Do not rebuild
Blender or implement deferred acquire submission from this evidence.

The interop path is no longer the dominant Dam Break cost. The next performance work, if approved as
a separate scope, belongs in FluidSim solver scheduling and kernel-launch reduction.

## Frozen candidate

- Executable: `F:\blender-gpu-interop-official-build\bin\blender.exe`
- SHA-256: `DB2BC63CD7234F59251F88E7BA9C3D3F9C056DB8681FA0C54DD1E05B86B48B82`
- Blender: 5.2.0 Alpha, Vulkan backend
- Add-on source commit: `c182d3a39be9e9fce548df0a61ef2b1951974cf3`
- Add-on branch: `codex/native-gpu-interop-main-20260715`
- Workload: canonical `demos/dam_break.blend`, 600,000 particles, CUDA/Warp
- Runtime limits: BelowNormal priority and affinity `0xFFFF`

## Qualified functional proof

The live review session proves:

- `presentation_mode == "Native GPU"`;
- 600,000 particles and 4,800,000 procedural PointCloud draw vertices;
- six persistent imports and two persistent maps;
- zero bulk host-position bytes and zero presentation bytes;
- one GPU publication per actual frame-change event (38/38 over 2.01 seconds);
- visible live playback in the Vulkan viewport;
- unchanged candidate hash and the configured process limits.

The review process is PID 44368 on MCP port 9878. It uses a non-persistent temporary add-on copy under
`C:\Users\Tyler\AppData\Local\Temp\codex-fluidsim-review-20260716T2005\addons`. The normal Blender 5.2
`userpref.blend` hash remained
`1CFC6818E403DBDC278339A474B9B9A8C9E05F10BF07F55061CE0F6B62177A13` after launch.

## Performance attribution

The diagnostic synchronized the solver only to establish a GPU-wall attribution boundary. Across the
12-frame Nsight capture:

- synchronized solver wall median: 83.32 ms/frame;
- interop CUDA publication median: 0.88 ms/frame;
- CUDA kernel instances: 6,836;
- CUDA kernel execution: 274.53 ms total, 22.88 ms/frame;
- CUDA memcpy execution: 0.31 ms/frame;
- CUDA memset execution: 0.32 ms/frame;
- total CUDA GPU activity: 23.51 ms/frame;
- CUDA API calls: 34,659 and 447.04 ms total, 37.25 ms/frame;
- `cuLaunchKernel`: 4,998 calls;
- `cudaLaunchKernel`: 2,288 calls;
- `cudaGraphLaunch`: 324 calls.

The publication kernels themselves are small: position/radius publication is about 0.014 ms/frame,
draw-count publication about 0.001 ms/frame, and bounds reduction/encoding about 0.52 ms/frame. The
remaining publication time is synchronization and metadata handling.

This makes a Blender-side publication micro-optimization incapable of a material Dam Break speedup.
Even removing the entire measured publication cost would save only about one percent of synchronized
solver wall time. The CUDA trace instead points to high solver scheduling and launch volume.

## Rejected redraw evidence

Two three-process visible-versus-hidden wall-clock sets are retained but rejected. The third process in
each set exceeded the predeclared 10% stability gate because `wm.redraw_timer` includes Windows
compositor/present pacing. One set produced provisional medians of:

- visible redraw: 7.48 ms;
- hidden redraw: 6.64 ms;
- paired PointCloud delta: 0.80 ms;
- interop publication: 0.87 ms;
- solver GPU wall: 89.95 ms.

These values support the qualitative conclusion that PointCloud drawing is subdominant, but they are
not promoted as a qualified GPU draw measurement. Nsight Systems 2025.3.2 captured CUDA successfully
but did not intercept Blender Vulkan API/debug-marker data on this Windows installation. Nsight
Graphics is not installed, so hardware-stage Vulkan attribution remains unavailable.

## Verification

- `python -m pytest -q tests/test_native_interop_stage_profile_aggregate.py`: 4 passed.
- Python compilation passed for the draw profiler, aggregator, and live review launcher.
- `git diff --check` passed in both dirty repositories; existing line-ending warnings remain.
- `.pin` was unchanged; Blender `.pin` SHA-256 is
  `AC916A58F2FF2E55A0ABCCED0B262C8B394C82617DB4CDBF86FF60E75B048959`.
- No Blender or add-on production source was modified in this loop.
- No build, commit, push, preference save, or startup-file save occurred.

## Key artifact hashes

- `nsight-vulkan-draw.nsys-rep`:
  `310666EBCD3BD056AE1900816F3665DAED896BE46F1555A47CDAEB2B507C92C6`
- `nsight-vulkan-draw.sqlite`:
  `2D265DE0F3C26E31DFEBA14AACE866D20815CB3BA24000E9B054D36E3116DFB0`
- `nsight-run.json`:
  `7DD3AD144F0728E36B1B38A84B6D1676F70873E84A488D7A735CD070235D632F`
- `live-review.json`:
  `D1A018BCB48CD94B801188EF204DF799D1B55C37E4FC02E354299461D646D0CF`
- `live-publication-rate.json`:
  `5F428A403A29CBA5647743813A6C2CF9D8DC27BCBC5C1CA1744D8990AA8C1F2F`

## Next action

Tyler reviews the open live simulation. When that review is finished, close only the owned candidate
window. Then verify the Blender 5.2 profile hash is unchanged and remove the temporary add-on copy.

If further performance work is approved, start a separate solver-performance TDD loop. First attribute
the 34,659 CUDA API calls and graph boundaries to FluidSim solver stages, then target fewer graph
launches, fewer uncaptured kernels, or safe kernel fusion. Do not broaden the Blender interop API based
on the present evidence.
