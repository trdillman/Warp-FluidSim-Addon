# GPU Interop Publication Stage Profile

## Decision

Retain the current packed publication path. The idealized direct-write/no-repack path did not approach
the historical 2x target at 250,000 visible points, and the measured pack kernel is too small to explain
the gap.

This report is authoritative only for the publication-stage question. It does not qualify the broader
A-01-A-22 release matrix.

## Frozen candidate

- Executable: `F:\blender-gpu-interop-official-build\bin\blender.exe`
- SHA-256 before and after every run:
  `DB2BC63CD7234F59251F88E7BA9C3D3F9C056DB8681FA0C54DD1E05B86B48B82`
- Add-on source:
  `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop_main_20260715`
- Backend/device: Vulkan, NVIDIA GeForce RTX 4080 SUPER, CUDA/Warp
- Process controls: BelowNormal priority, affinity mask `0xFFFF`
- Candidate rebuilds during this evaluation: none

## Method

Each stage run used a fresh Blender process, two persistent native slots, 250 warm-up frames per mode,
and four measured batches of 500 redraw frames per mode. The benchmark retained every raw frame sample
and used the median batch mean for the paired speedup. Batch-mean coefficient of variation had to remain
at or below 10% in both modes.

The no-repack diagnostic skips only the counted CUDA position-pack kernel. It retains the external wait,
GPU-authored active count, bounds, indirect draw command, timeline completion signal, lease transaction,
and ordinary PointCloud draw. Visible-versus-hidden framebuffer differences prove that points contribute
to the image; packed-versus-no-repack framebuffer mean absolute difference had to remain at or below
0.5 of 255. It also bounded the 99th-percentile difference at 3, the fraction above 4 at 0.5%, and the
fraction above 16 at 0.1%, preventing a localized mismatch from hiding inside the mean. The GPU indirect
draw count had to resolve to exactly 250,000 points.

CUDA stage profiling was enabled explicitly for five short diagnostic samples and remained disabled in
normal runs. Events bound external wait, position pack, metadata work, and total publication through the
completion signal. Timings were collected only after `lease.submit()`.

## Results

| Run | First mode | Packed ms | No-repack ms | Speedup | Packed CV | No-repack CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | packed | 14.2990 | 13.0689 | 1.0941x | 5.00% | 4.60% |
| 2 | no-repack | 13.9780 | 14.1940 | 0.9848x | 4.40% | 5.20% |
| 3 | packed | 13.1216 | 12.8939 | 1.0177x | 2.60% | 5.40% |

All three runs passed the execution, provenance, visibility, draw-count, zero-host-position-byte,
fixed-import/map, and stability gates. Cross-process coefficient of variation was 3.60% packed and
4.30% no-repack. The paired speedup median was 1.0177x. No-repack was faster twice and slower once, but
the effect was small and inconsistent.

| Run | External wait ms | Position pack ms | Metadata ms | Total CUDA publication ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0061 | 0.01024 | 0.6943 | 0.9421 |
| 2 | 0.3174 | 0.01126 | 0.9544 | 1.2268 |
| 3 | 0.0061 | 0.01024 | 0.6308 | 1.0199 |

The position pack was below 1 ms, below 10% of publication, and below 0.1% of end-to-end viewport frame
time in every run. Even eliminating it completely cannot yield a 2x end-to-end result. The useful next
optimization boundary is the viewport draw/redraw pipeline, not another solver-output copy removal.

## Same-source Dam Break check

The final functional run used 600,000 particles in `native,cpu,cpu,native` order:

- Native whole-frame median: 82.9920 ms
- CPU whole-frame median: 92.0348 ms
- Whole-frame speedup: 1.1090x
- Process-CPU reduction: 29.27%
- Native imports/maps: 6/2
- Native bulk host position bytes: 0

This confirms the instrumented add-on still uses native interop in the canonical workload. It is a
single post-change validation run, not a replacement for the three-run rebaseline.

## Artifacts

- `run1.json`, `run2.json`, `run3.json`: authoritative stage measurements
- `aggregate.json`: authoritative cross-process stability gate and decision
- `dam-break-final.json`: same-source functional validation
- Matching `.stdout.txt` and `.stderr.txt`: process logs
- Prefixed `invalid-*`, `failed-*`, `noisy-*`, `unstable-*`, and `cv100-*` files: preserved diagnostic
  history; they are not release measurements

The failed and invalid attempts were retained to preserve provenance. Only the unprefixed final files
support this decision.
