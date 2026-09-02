# Real-World Add-on Performance Results — 2026-08-27

## Recommended course

Retain the clean integrated branch `codex/real-world-performance-final-20260827` at `e699ef0a08ad74aff5f26126330c61144258146d` as the implementation candidate. It combines four reviewed production optimizations and the measurement infrastructure needed to keep future work honest:

1. GPU nearest-particle surface attributes for opt-in surface channels.
2. Live-prefix device readbacks for viewport particle/whitewater channels.
3. Liquid compaction count publication during scatter.
4. Removal of the duplicate whitewater type-count reduction.

Do not integrate the cache validation cursor yet. Its correctness is approved, but its current performance campaign was blocked by unrelated GPU clients and remains provisional.

After final production proof, move directly to default-playback sparse-topology attribution and compact-frontier work. Do not spend another cycle on the rejected effects/source/scene-fingerprint/count-only micro-candidates. The optimized all-attribute surface path is now about the same cost as the default path, so further surface-attribute work has sharply lower ROI.

## Scope and authority

- Add-on code only; archived experiment branches were excluded.
- Original master was preserved at `8c6dbd20f07512195782f953dce3cd3a4a7ccbe7`.
- Trusted profiler baseline was master plus measurement-only commits at `f2b82dfc1fb969ffb8ca301fb9ad666d7497946e`.
- GPU surface candidate was `d9bc30fde9175036ee2dc38f4cf3b649a11570c8`.
- Final cumulative branch is `e699ef0a08ad74aff5f26126330c61144258146d`.
- Historical reports that predate this campaign are labeled historical; they are not presented as new results.

## ROI-ranked implemented results

| Rank | Optimization | Current evidence | Decision |
|---:|---|---|---|
| 1 | GPU surface nearest-attribute transfer | Seven balanced fresh-process ABBA/BAAB pairs: whole-frame trial median `175.52 → 20.71 ms` (`-88.20%`), viewport `165.32 → 10.19 ms` (`-93.83%`), attribute adaptation `154.27 → 1.73 ms` (`-98.88%`); candidate won `7/7` pairs for all three metrics | Retain; large, tangible win when nearest-particle surface channels are enabled |
| 2 | Live-prefix HostFrame readbacks | 28 fresh Blender processes: warm handler median `-1.5019%`, p95 `-10.4531%`, median and p95 wins in `5/7` pairs | Retain; modest median and material tail-latency improvement |
| 3 | Compaction count fusion | Removed `2,520` redundant count-kernel launches; exact survivor/channel digests; `61/84` scenario wins; aggregate pair-observation median `-9.59%`; typical absolute saving about `0.05–0.12 ms` | Retain as exact low-risk launch removal, not a headline user-path gain |
| 4 | Duplicate whitewater type reduction | Exact launch count reduced from two to one; `13/21` pair-observation wins; aggregate median `-7.68%`, but live-count bands crossed parity and absolute savings were tiny | Retain as exact redundant-work removal, not a headline gain |
| Provisional | Cache validation cursor | Correctness and comparator independently approved; current paired timing could not run under its strict contention classifier | Keep isolated until a naturally uncontended window; do not weaken the classifier or close user applications |

The earlier fixed `0.5 ms` materiality floor was agent-authored, not a user or project requirement. It was removed. Current retention decisions use correctness, repeatability, absolute milliseconds, whole-frame effect, tail behavior, and activation scope.

## GPU surface campaign detail

Workload per fresh process:

- Blender 5.1 / Warp 1.16 / RTX 4080 SUPER.
- Six measured frames, voxel size `0.08`, maximum particles `200,000`, collider subdivision `2`.
- All persistent surface attributes enabled.
- Stable live facts across the campaign: `24,704` mean particles and `59,392` mean voxels; final surface vertex count was `4,046–4,056` because independent CUDA simulation processes are not bitwise deterministic.
- 28 total raw reports: 14 baseline and 14 candidate, ordered as seven balanced ABBA/BAAB pairs.

Trial-median results:

| Metric | Baseline median | Candidate median | Absolute change | Relative change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| Complete frame | 175.52 ms | 20.71 ms | -154.81 ms | -88.20% | 7/7 pairs |
| Viewport phase | 165.32 ms | 10.19 ms | -155.13 ms | -93.83% | 7/7 pairs |
| Surface attribute adaptation | 154.27 ms | 1.73 ms | -152.54 ms | -98.88% | 7/7 pairs |
| Surface extraction | 1.11 ms | 1.22 ms | +0.11 ms | noisy/non-dominant | 1/7 pairs |
| Blender publication | 5.78 ms | 5.20 ms | -0.58 ms | -10.02% | 7/7 pairs |

Paired median-ratio bootstrap intervals (20,000 deterministic resamples of seven pair ratios):

- Complete frame: `-88.34%`, 95% interval `[-89.21%, -87.90%]`.
- Viewport: `-93.73%`, 95% interval `[-94.06%, -93.59%]`.
- Attribute adaptation: `-98.84%`, 95% interval `[-98.93%, -98.81%]`.

These comparative reports remain explicitly raw/non-approval evidence because `nvidia-smi` cannot reliably enumerate Windows WDDM graphics clients. The effect is far larger than observed machine noise and repeats in every pair, but the contention limitation must remain attached to the numbers.

Final integrated Blender validation at `e699ef0`:

- All attributes: `19.06 ms` total, `9.80 ms` viewport, `32/32` expected attribute readbacks, complete surface-attribute evidence, zero added synchronization, `USER-PATH-PROFILE-PASS`.
- Default attributes: `18.71 ms` total, `7.67 ms` viewport, and exactly zero surface HashGrid/sample/lookup/fallback/gather/readback operations, `USER-PATH-PROFILE-PASS`.

Raw evidence directory:

`C:\tmp\liquid-warp-real-performance-20260827\surface\paired_d9bc30f`

## Correctness proof

The GPU path does not change solver state, surface geometry generation, topology, triangle order, cadence, cache schema, or default velocity/motion behavior. It replaces host nearest-index construction plus full particle-channel readbacks with one publication-time GPU index and surface-sized channel readbacks.

- Exact CPU/CUDA nearest parity passed negative coordinates, cell boundaries, duplicate/equidistant ties, lowest-original-index tie resolution, global fallback, intentional HashGrid collisions, repeated execution, and one-ULP near ties.
- Stable sampling-cap boundaries passed at `119,999`, `120,000`, `120,001`, and `240,001` particles on CPU and CUDA.
- Float, int32, vec3, and vec4 gathers passed on CUDA.
- Independent review added 432 randomized CPU/CUDA parity cases; all matched `nearest_sample_indices` byte-for-byte.
- Warp 1.16 zero-target materialization was independently reproduced and fixed with an owned zero-length device allocation.
- Full final-branch Python suite: `1,298 passed`; combined focused suite: `175 passed`; compileall and `git diff --check` passed.
- Final integrated worktree was clean before this report was added.

Cross-process surface hashes are not required to be bitwise identical because CUDA atomics and marching-cubes ordering are not globally deterministic. Correctness is bound by exact transfer fixtures, unchanged simulation ownership, stable live counts/workload, complete per-run attribute inventory/dtype/count/byte fingerprints, and live Blender output validation.

## Rejected work preserved as negative evidence

| Candidate | Historical measured result | Decision |
|---|---|---|
| Custom Effects active-bound dispatch | About `0.012 ms` nominal; paired median `1.026x` slower, p95 `1.133x` | Do not port |
| Same-frame source-batch reuse | About `0.0423 ms` / `2.33%`; confidence interval crossed parity, p95 `1.195x` | Do not port |
| Full scene-input transaction fingerprint | `1.586x` slower static; `1.164x` slower moving collider | Reject; any future attempt must use an upstream Blender invalidation signal |
| Count-only particle occupancy | No scenario cleared materiality; p95 `1.25–1.29x`; particle equivalence incomplete | Do not port; revisit only as a new production-scale investigation |

The historically reported `74.5%` solver and `87.7%` pressure improvements were already present before this campaign. They are not new work and were not re-credited here.

## Next implementation plan, highest ROI first

1. **Run final production proof on the clean cumulative branch.** Execute the default headless production gate. Run GUI-inclusive acceptance only when foreground Blender ownership is available. Preserve generated demo artifacts instead of silently resetting them.
2. **Measure default-playback sparse topology.** Extend the trusted host-observed profiler with topology rebuild/cutover timing without added synchronization. Use `dam_break`, `crown_splash`, and `adhesive_calender`; include topology-heavy effect/snippet mode separately.
3. **Implement compact marked-leaf membership if topology dominates.** Replace full-domain candidate/flag clears and repeated full-domain 27-neighbor scans with compact membership and frontier-only expansion. Preserve the exact canonical leaf set, coordinates, cutover order, overflow behavior, churn remap, and sparse-field reset contract. Benchmark seven fresh-process pairs with whole-frame and peak-memory evidence.
4. **Defer additional surface work unless a larger real mesh reactivates it.** Eliminating the remaining one host surface-position upload and unconditional empty-fallback kernel can only recover a fraction of the current `~1.7 ms` attribute stage in this workload.
5. **Re-run the cache cursor only in a naturally uncontended window.** Require the already implemented frame/payload/cache identity comparator. Integrate only after current paired evidence.
6. **Profile remaining regular pressure only after topology.** Existing pressure work is already highly optimized. Pursue producer/reduction fusion or exact warm start only if current whole-frame attribution shows a material remaining pressure segment; do not revive rejected micro-slices.

