# Native GPU Interop Same-Executable Rebaseline

Date: 2026-07-15
Status: measurement complete; the existing 250,000-particle `>= 2x` scale gate fails

## Pinned candidate

- Executable: `F:\blender-gpu-interop-official-build\bin\blender.exe`
- SHA-256 before and after all runs:
  `DB2BC63CD7234F59251F88E7BA9C3D3F9C056DB8681FA0C54DD1E05B86B48B82`
- Size: 95,957,504 bytes
- Blender: 5.2.0 Alpha
- Build hash: `f51c2552fb3e`
- Build branch: `codex/gpu-buffer-interop (modified)`
- Add-on source:
  `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop_main_20260715`
- Add-on branch/base: `codex/native-gpu-interop-main-20260715` at
  `c182d3a39be9e9fce548df0a61ef2b1951974cf3`
- Blender source/base: `C:\Users\Tyler\blender-5.1-source\blender` at
  `f51c2552fb3ee3841dc6bad24947bfd51cc8df5f`
- Dam Break file SHA-256:
  `90C472A6E1956FE61C7DE5114F97CBA2707A1BE6AAEBC4E81C3E0ED03EF9E77B`

Every measurement used this one executable. The executable was not rebuilt, replaced, or modified between
runs. Runs were serial, `BelowNormal`, limited to logical processors 0-15 (`0xFFFF`), and started with factory
settings and the Vulkan backend. User-site packages were disabled and `PYTHONPATH` was removed.

The candidate's private runtime was verified as NumPy 2.3.4 and Warp 1.12.1 under
`F:\blender-gpu-interop-official-build\bin\5.2\python\Lib\site-packages`.

## Scale benchmark

Two measurement runs were completed. Each workload used 20 iterations. The reported range is the range of the
two per-run medians.

| Particles | CPU median range | Native median range | Speedup range | Existing gate |
|---:|---:|---:|---:|---|
| 25,000 | 28.45-29.77 ms | 19.75-21.95 ms | 1.30-1.51x | Pass: native within 10% of CPU |
| 250,000 | 33.05-33.52 ms | 19.20-19.80 ms | 1.69-1.72x | **Fail: below 2x** |
| 1,000,000 | 44.46-46.66 ms | 19.93-20.50 ms | 2.23-2.28x | Pass: at least 2x |

All native cases reported zero bulk host-position bytes, six imports, and two maps. The fresh results supersede
the previous broad scale claim of approximately 1.80x / 2.92x / 4.33x for this release candidate.

The original gating script was run first and failed at its 250,000-particle assertion before it could write a
result. Its shutdown timer is registered only after all assertions, so the failed process remained open. After a
bounded wait, only that owned candidate process was terminated. `scale-benchmark.stdout.log` and
`scale-benchmark.stderr.log` retain the failure. The canonical pre-existing `benchmark.json` was restored
byte-for-byte with SHA-256
`CBEF2273943DB0BFA3CB99AFAD5A7A150E8BDEAE68CEAFE5F2C82E5ED1162B0D`.

`blender_native_interop_rebaseline.py` reuses the original benchmark implementation but records threshold
outcomes instead of aborting before evidence is written. It does not change or weaken the release gate.

## Canonical Dam Break benchmark

Three independent Blender processes ran the canonical 600,000-particle Dam Break scene. Each process used three
warm-up frames and twelve measured frames for each case, in the order native / CPU / CPU / native.

| Run | Native whole frame | CPU whole frame | CPU/native | Frame-time reduction | Process-CPU reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 107.69 ms | 136.40 ms | 1.267x | 21.0% | 27.9% |
| 2 | 109.96 ms | 116.31 ms | 1.058x | 5.5% | 31.0% |
| 3 | 90.44 ms | 103.01 ms | 1.139x | 12.2% | 15.0% |
| Median | 107.69 ms | 116.31 ms | **1.139x** | **12.2%** | **27.9%** |

All three structured results passed their correctness checks. All four cases in every run retained 600,000
particles. Native cases reported `presentation_bytes_last_frame = 0`, zero bulk host-position bytes, six imports,
two maps, and 15 published frames.

The supported performance statement from this rebaseline is:

> On the unchanged candidate, native interop reduced canonical 600,000-particle Dam Break whole-frame time by
> 5.5%-21.0% across three runs, with a 12.2% median reduction (1.139x median speedup). Process CPU time fell by
> 15.0%-31.0%, with a 27.9% median reduction.

The direction was consistent, but the magnitude varied materially. The evidence does not support describing the
current Dam Break speedup as dramatic or fixed.

## Verification and artifact map

- Candidate SHA-256 after runs: unchanged.
- Blender `.pin` SHA-256 after runs:
  `AC916A58F2FF2E55A0ABCCED0B262C8B394C82617DB4CDBF86FF60E75B048959`.
- Original benchmark source SHA-256:
  `6565C2199460832424D2EB91B0879603FE6DCAC25154E87C87365D68AA44B16A`.
- Measurement results: `scale-benchmark.json`, `scale-benchmark-run2.json`.
- Canonical results: `dam-break-profile-run1.json`, `dam-break-profile-run2.json`,
  `dam-break-profile-run3.json`.
- Process output: corresponding `.stdout.log` and `.stderr.log` files.
- Runtime provenance: `runtime-paths.stdout.log`.
- Pre-run benchmark backup: `benchmark-before.json`.

Structured-result SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `scale-benchmark.json` | `3CEC79C0BA7CB95D46F491643DB02768628F564DA3AD2CF1A1BBDC7BFF240681` |
| `scale-benchmark-run2.json` | `2A2DBE373D2FE53FB133B0EA6D1E6029ABFADF917A4EB9BADA2D42A22F782010` |
| `dam-break-profile-run1.json` | `3121212E67059F7F673D4A6A17521A2E746E501B084C02C21FC3FDAF63ED5F90` |
| `dam-break-profile-run2.json` | `D7535D9F9BB9C8D747A74C3B875B6CB98D450DEFABC2EA944F23B03067198149` |
| `dam-break-profile-run3.json` | `3E9B203E6FE9727DA47FA3732D1F4E6F360B8BF15C80D8099F1D9872EAA8895B` |

The first measurement-only scale run wrote a valid result but did not exit after a direct quit request; its owned
process was stopped after the result flushed. The runner was changed to schedule shutdown on a Blender timer, and
the second scale run then exited cleanly. The same timer repair was applied to the Dam Break profiler; all three
Dam Break processes exited cleanly. These harness-only changes do not alter simulation, interop, or presentation
behavior.
