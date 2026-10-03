# FluidSim Warp 2.1: handoff from 2.0

This is the starting point for 2.1. It lists what 2.0 left open on purpose, the
evidence behind each item, and a suggested order. The 2.0 record is
`docs/RELEASE_2_0_PROGRESS.md`; the plan it executed is
`docs/NEXT_MAJOR_RELEASE_PLAN.md`.

## Baseline 2.0 hands over

| Item | State at handoff |
|---|---|
| Platform | Blender 5.2 LTS, bundled Warp 1.17.0 (cu12), Windows 11 x64 certified on one RTX 4080 SUPER; Linux x64 packaged, gate 52/53 GUI on WSL2, uncertified |
| Demo reference hash | `e49d23ef…e1ae` (demo bakes are bitwise reproducible on the same GPU, driver and Warp build). Since the collider wall-film fix: `ffd50faa…2873`. Since collider grid friction (`75f9265`): `cfd0bf04…ec52`. Since the rocky-cove dam_break rebuild: `2cb4f781…3a73` (see `demos/ANALYSIS.md`) |
| Tests | see the final row of the progress record |
| Performance | vs 1.0.255: crown_splash −2.7%, dam_break +0.4%, paddle_stir +4.0% (absolute medians; paired median +5.5%). chromatic_vortex vs 1.1.0: −2.2%. 2.0 adds no measurable solver cost over 1.1.0; paddle_stir's gap is 1.1.0's deterministic P2G gather (section 3) |
| Open P0/P1 | none |

## 1. Carry-over work (must do)

1. **Certify Linux.** 2.0 ships Linux x64 packaged but uncertified. On WSL2
   (Ubuntu 24.04) the package build, clean install, smoke and installed GPU
   regression pass, and the full gate passes every headless lane and 52/53
   GUI checks. The one failure is `viewport screenshot ... 0/4000 visible
   samples`: Blender's `screen.screenshot_area` reads back black under Mesa
   (WSLg or Xvfb, d3d12 or llvmpipe), with or without the add-on. Options:
   (a) run the gate on native Linux with the NVIDIA driver, where readback
   should work; (b) with owner approval, prove viewport visibility with an
   offscreen `gpu.offscreen` view3d draw instead of a front-buffer read.
   Environment traps found on the way (both in the user guide): WSLg's
   Wayland EGL path crashes Blender's GUI (Mesa zink, `EGL_BAD_MATCH`; use
   `WAYLAND_DISPLAY=` and `GALLIUM_DRIVER=d3d12`), and Ubuntu's
   `libnvidia-compute-*` (pulled in by `nvidia-cuda-toolkit`) shadows the WSL
   `libcuda.so` so Warp sees no GPU (`LD_LIBRARY_PATH=/usr/lib/wsl/lib`).
   Consider having the add-on name that fix instead of "no CUDA device". Also
   check why the Linux `crown_splash`/`dam_break` `.blend` bytes differed
   between two gate runs (progress record, "Release").
2. **Cache frames should stamp source animation.** A cache frame records the
   settings signature only (REVIEW-022). Changing a keyframed source or
   collider after a bake does not invalidate the cache. Add a per-frame
   source stamp taken at the frame's own time, and compare it on playback.
3. **Stamp the source schema.** Source objects carry no schema version. Add
   one before the next source-settings change, so a future migration can
   tell old from new.
4. **Resolve the remaining RNA / solver default divergences** found while
   unifying defaults. RNA (`properties.py`) and `SolverSettings`
   (`sim/solver.py`) disagree on `viscosity_constant` (0.05 vs 0.0),
   `ww_bubble_drag` (4.0 vs 8.0) and `ww_spray_drag` (0.2 vs 0.5). Blender
   scenes always pass the RNA value, so only headless callers that build
   `SolverSettings` directly see the other one. 2.0 left them alone because
   changing either side changes someone's results. Pick one
   value each, rebind the demo hash, and add them to the default-agreement
   test.
5. **Defect #2 (compounding slowdown per reset).** Still not reproduced. On
   2.0, dam_break with viscosity and surface tension on, reset ten times
   through the Reset operator and stepped 24 frames each time, ran flat
   (median 106 → 96 ms per frame), with GPU memory fixed at 408 MB and no
   datablock or particle growth. Every headless probe so far has been flat,
   so the next step needs the owner's GUI repro: live playback in the
   viewport, which headless runs do not exercise, and the exact settings.
   Probe: `defect2_probe.py` in the 2.0 perf evidence folder.

## 2. Open review cards (medium / low)

None block a release. Each has a card in `backlog/tasks/`.

| Card | Severity | Summary | Suggested fix |
|---|---|---|---|
| REVIEW-013 | medium | `operators.py` / `panels.py` unregister stops at the first failure | Wrap each class in `try` and continue; test with a class that fails to unregister |
| REVIEW-024 | medium | `test_memory_safety_recovery.py` checks source text for substrings instead of behaviour | Replace the substring asserts with a registration or poll check |
| REVIEW-025 | medium | `_phase2_source_acceptance_common` deletes every add-on module from `sys.modules` at import time | Move the purge into an explicit setup call |
| REVIEW-027 | medium | No tests for the solver's error paths (bad device string, non-CUDA device, reduction launch failure, sourcing count mismatch) | One test per path, asserting the actionable message |
| REVIEW-038 | low | `CudaDotProduct.launch()` passes raw pointers with no check that the arrays hold `count` elements | Check lengths before the launch |

Closed in 2.0: REVIEW-041 (fixed), REVIEW-031 (won't fix, undo policy) and
REVIEW-042 (no production caller after REVIEW-022).

Closed in 2.1: REVIEW-015 (fixed; the cache key no longer holds the
tolerance, and a miss evicts one entry instead of all eight).

## 3. Performance (plan D7)

2.0 did no frontier work. Its budget was ≤5% against 1.0.255.

**What 2.0 measured.** Demo `step_ms` was measured on one RTX 4080 SUPER in
paired runs with alternating order (ABBA), median of frames 2–48. On this
shared machine one run varies by 5–30% from the next, so only paired
medians mean anything.

| Demo | Reference | Result |
|---|---|---|
| crown_splash | 1.0.255 | paired median 0.973 (2.0 faster in 5 of 6 pairs) |
| dam_break | 1.0.255 | 1.004 |
| paddle_stir | 1.0.255 | +4.0% on absolute medians (190.0 → 197.6 ms); paired median +5.5% (n=8), +10% in an earlier n=6 run |
| chromatic_vortex | 1.1.0 (the re-saved file; 1.0.255 cannot open it) | −2.2% on absolute medians; paired median +2.8% (n=8) |

The paddle_stir gap is not a 2.0 change. The GPU-synced stage profiler puts
it in P2G:

| Version | P2G | Solver total |
|---|---|---|
| 1.0.255 | 1.78 ms/frame | 40.4 ms |
| 1.1.0 | 3.42 ms | 47.0 ms |
| 2.0 | 3.50 ms | 43.1 ms |

1.1.0 replaced the atomic P2G scatter with a sorted gather (`d2a6b6d`) to make
bakes bitwise reproducible. Warp 1.17 against 1.16 is neutral on paddle_stir
and about 4% faster on chromatic_vortex. Evidence:
`C:\Users\Tyler\fluidsim-gate-logs\perf-2.0-20260927\`.

**Solver-only follow-up (2026-09-29).**
`docs/PERFORMANCE_2_0_BENCHMARK_20260929.md` runs one dam_break workload on the
pre-optimization code, 1.1.0, 2.0 and master, all on Warp 1.17. Against the
pre-optimization code, 2.0 is:
- about 5.6× faster with viscosity
- about 4.5× faster with viscosity and surface tension
- 6% faster with surface tension
- 2.4% slower per substep with neither (the P2G gather)

Against 1.1.0 it shows no regression.

**Work, in order of expected return** (see `docs/` performance results):

1. **Faster deterministic P2G.** It is the only measured regression against
   1.0.255. Keep the gather, and cut its cost with fewer passes over the bins
   or per-leaf shared-memory accumulation. Accept a change only if the
   GUI-gate hash is unchanged.
2. Topology frontier (6 tasks).
3. Pressure follow-on (5 tasks). The REVIEW-015 cache is done.
4. Device-resident surface work and cache-cursor integration.
5. Viscosity SDF reuse.
6. **Viscosity graph-key churn — DONE** (branch `perf/visc-graph-key-churn`).
   The key held the solve size, `dt` and the tolerance, so on the dam_break
   viscosity benchmark 435 solves made 213 captures. Now `dt` and the
   tolerance live in device arrays that one kernel stages each solve, and the
   captured launches and dot products read the active voxel count from
   `num_leaves` on device (`CudaDotProduct.launch_active`, bitwise equal to the
   host-counted dot). Launches are rounded up to 4 buckets per octave. The key
   holds only shape and array pointers.
   - Captures 213 → 30, capture time 380 → 100 ms (visc; visc_st 362 → 107 ms)
     over 40 frames. The floor of 30 is 5 buckets × 6 component buffers, since
     the velocity components alternate between two buffer sets. 2 buckets per
     octave gives 18 captures and 8 gives 54, with no measurable host-time
     difference.
   - Per-frame particle and velocity hashes are bitwise identical to REVIEW-015
     on visc and visc_st, and the GUI gate hash is unchanged (`cfd0bf04…`, master's reference after
     collider grid friction; re-verified on the rebased branch).
   - `tools/benchmark_versions.py`, 5 paired trials against REVIEW-015:
     visc 0.81× (range 0.77–0.88, 48.8 → 38.9 ms/frame), visc_st 0.86×
     (0.82–1.32, 63.6 → 54.5 ms/frame). Evidence:
     `C:\Users\Tyler\wt\bench_viscgraph\`, probe in `C:\Users\Tyler\wt\probe\`.

Add the 2.0 comparison to the gate, so the budget is enforced on every run
rather than measured once. Use ABBA pairs, at least 8 per demo, median
excluding frame 1, and the stage-profiler breakdown. Whole-frame time alone
is too noisy on a shared workstation to settle a 5% question.

## 4. Features (2.1+ candidates, need owner approval)

- ST-FLIP promotion out of Experimental once its reference and conservation
  gates run.
- Species v2 (foam / spray / bubble), a fresh port.
- Alembic / VDB export.
- Narrow-band particles; resolution divisions instead of voxel size.
- Pressure-field sourcing; multigrid.

## 5. Process notes for whoever picks this up

- The GUI gate (`tools/run_production_gate.py --include-gui`) is the only
  lane that writes `artifacts/production_gate_manifest.json`. Run it detached,
  don't touch GUI windows during the run, and expect the demo hash to change
  only when results change.
- The demo builder (`tools/build_demo_scenes.py`) rewrites the tracked
  `demos/*` files. Back them up first.
- Keep Blender-loaded worktrees under short paths; Blender's Python hits the
  260-character path limit.
- 1.x needs Warp 1.16 and 2.0 needs Warp 1.17. Isolate them per run with
  `PYTHONNOUSERSITE` and `PYTHONPATH` rather than uninstalling either.
