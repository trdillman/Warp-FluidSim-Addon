# Independent Review — `reviews/WORKTREE_ROI_INTEGRATION_PLAN_2026-07-29.md`

Date: 2026-07-29
Reviewer: independent review subagent (read-only; no repository modifications)
Authority: `AGENTS.md` (branched-experiment exclusions), verified live git state

## Verdict

**REVISE** — the plan's structure, ordering, and experiment handling are sound and no
recommendation is dangerous on its face, but it has one real data-loss gap (untracked
authored files), one classification-ambiguity gap (the PCG delta is byte-identical to an
excluded experiment), and under-specified validation lanes relative to `AGENTS.md`.
No blocker to the Phase-4-first ordering.

## Verification performed

All 16 worktree rows were re-checked against live `git worktree list`, per-worktree
`rev-list --left-right --count master...HEAD`, `status --porcelain`, and `diff --stat`.
Kernel and pressure deltas were read directly.

| # | Row claim | Verified? | Note |
|---:|---|---|---|
| 1 | master `556bce2`, dirty = `.gitignore`, demo blends/previews, `.agents/`, no dirty source | ✅ exact | |
| 2 | `codex/fix-cone-pyramid-distances`, 34 behind, 0 ahead, dirty | ✅ | substantive diff is exactly `sim/sourcing.py` (+85) and `tests/test_sourcing_pipeline.py` (+89); the other 8 "modified" source files are pure CRLF churn |
| 3 | `codex/deepen-four-modules`, 38 behind, 0 ahead | ✅ counts | **incomplete**: also modifies `demos/dam_break/meta.json`, not just the blend + preview |
| 4 | 61 behind, clean | ✅ exact | |
| 5 | `e9bf` detached, 43 behind, clean | ✅ exact | |
| 6 | 48 behind, clean | ✅ exact | |
| 7 | `orbit-gpu-pcg-baseline`, 46 behind, clean | ✅ exact | at `c182d3a` |
| 8 | 46 behind, "3 substantive modified files plus new `cuda_reduction.py`" | ✅ **correct** | `git status` shows 14 entries but `git diff --stat` is only `sim/pressure.py` (+336/-41-ish), `tests/test_pressure_projection_regressions.py` (+216), `tests/test_performance_hotpath_contracts.py` (+22) = 533 insertions; the remaining 10 are EOL-only |
| 9 | phase4 branch `4d1a895`, clean, 3 behind / 8 ahead | ✅ exact | |
| 10 | `phase4/live-property-interaction`, 19 behind, 130 dirty | ✅ exact (32 tracked + 98 untracked) | superseded-status claim independently confirmed, see below |
| 11 | 4 behind / 1 ahead, dirty kernels + demo outputs; two kernel fixes | ✅ **exact, including the technical description** | verified `tau = wp.clamp(..., -0.5, 0.5)` and `phi = accum / nfaces` (existing-face count) in `sim/kernels_fluid.py`; the single commit is `2fc5cb4 fix(policy-drift): repair nested GUI evidence` |
| 12 | 21 behind / 2 ahead, dirty — EXPERIMENT | ✅ exact (5 dirty) | |
| 13 | based on current master, 16 dirty — EXPERIMENT | ✅ exact (`556bce2`, 16 dirty) | worktree is **`locked`**; plan does not say so |
| 14 | 107 behind, deletes `CLAUDE.md` and `sim/__init__.py`, `review_2635080.blend` | ✅ exact | |
| 15 | `codex/native-gpu-interop`, 73 behind / 1 ahead, heavily dirty — EXPERIMENT | ✅ exact (55 dirty) | |
| 16 | `codex/native-gpu-interop-main-20260715`, 46 behind, staged+unstaged — EXPERIMENT | ✅ exact | branch tip is `c182d3a`, i.e. the branch has **no unique commits**; all experiment content is index/worktree-only |

Independent confirmation of the #10 claim ("its intended Phase-4 payload was integrated into #9"):
file-level comparison of the dirty worktree against `phase4/live-interaction-merge-20260728`
shows `tools/build_demo_scenes.py` and `tools/run_production_gate.py` byte-identical
(modulo EOL), and `sim/sourcing.py`, `sim/solver.py`, `properties.py` differing **only by
additions present in #9 and absent in #10** (accumulating `emitted_any`/`count_dirty`,
`cfl_substeps_clamped`, `getattr(self, "surface", None)`, `max_frame_catchup`,
`stat_cfl_clamped`). #9 is a strict superset for tracked source. The plan's 0/5 is justified.

## Findings, ordered by severity

### 1. HIGH — Cleanup can destroy untracked authored work; no preservation gate exists
`## Proposed execution plan → 3. Cleanup after preservation` says "Remove clean fully-contained
architecture and baseline worktrees", "Archive and remove the superseded dirty Phase-4
development checkout", and §2.2 says "Exclude stale reports, backlog duplication, `.agents/`".
Two of the prune/exclude targets contain **untracked files that do not exist on master**:

- `C:/Users/Tyler/.codex/worktrees/34f4/liquid_warp_blender51_port`: `backlog/tasks/TASK-POLICY-DRIFT-004-SAFE-SETTINGS-TYPE.md`, `-005-STALE-AUTHORITY.md`, `-006-EXACT-SOURCE-SHAPES.md`, `-007-FRONT-RANGE-CONTRACT.md`, `-008-WHITEWATER-ACTIVE-WORK.md`, `-009-PRESSURE-ACTIVE-SCALING.md`, `-010-FRONT-SPEED-DOC.md`, `-011-VALIDATION-TABLE-COUNTS.md`, plus `.agents/` and `demos/backlog/`. I checked `backlog/tasks/` on master: `004-SAFE-SETTINGS-TYPE`, `006-EXACT-SOURCE-SHAPES`, `011-VALIDATION-TABLE-COUNTS` are **absent from master**. Calling these "backlog duplication" is factually wrong for at least three of them.
- `liquid_warp_blender51_phase4_interaction`: 98 untracked entries including `backlog/tasks/TASK-REVIEW-001-WATERTIGHT-HOTPATH-DEFERRAL.md`, `TASK-REVIEW-002-RENDER-INIT-PERSISTENT-HANDLER.md`, `TASK-POLICY-DRIFT-004/007/010-*.md`, `demos/cache_export_playback_review.blend`, `demos/height_fill.blend`, `demos/liquid_sink.blend`, `demos/mesh_shell_source.blend`, `demos/mesh_solid_source.blend`. The "final payload comparison" in the plan is worded as a tracked-source comparison only and would not catch these.

`git worktree remove --force` deletes untracked files irrecoverably (they are in no object database).

**Exact recommended edit** — replace the first three bullets of §3 with:

```
- Before any removal, run `git -C <worktree> status --porcelain` and archive every `??`
  entry that is not reproducible output. Record the archive path in this plan.
  Known non-duplicate untracked files: `34f4` backlog tasks
  TASK-POLICY-DRIFT-004/006/011 (absent from master) and the `phase4_interaction`
  TASK-REVIEW-001/002 and TASK-POLICY-DRIFT-004/007/010 tasks.
- Remove clean, fully contained worktrees (#4, #5, #6, #7) with plain
  `git worktree remove` (no `--force`); a failure means the tree was not clean and
  must be re-inventoried, not forced.
- Archive and remove the superseded Phase-4 development checkout (#10) only after
  (a) a tracked-file comparison against `phase4/live-interaction-merge-20260728`
  and (b) an untracked-file archive. Note already verified: #9 is a strict superset
  of #10 for tracked source.
```

### 2. HIGH — The ROI-4 PCG delta is byte-identical to an excluded experiment's delta; the plan does not say so
`§2.3` says only "Do not source duplicate files from the excluded native-interop experiments."
That understates the situation. I diffed the two working-tree deltas:

`git diff -- fluidSim_warp_blender51/sim/pressure.py` in
`.codex/worktrees/orbit-gpu-pcg-conditional-loop-addon` is **identical** to
`git diff HEAD -- fluidSim_warp_blender51/sim/pressure.py` in
`liquid_warp_blender51_port_native_interop_main_20260715` (excluded experiment #16). Both
worktrees sit on `c182d3a`, and the experiment carries the same `+336` pressure change and
the same `+216` / `+22` test deltas alongside its 1204-line `native_interop.py`.

This is not an `AGENTS.md` violation — `orbit-gpu-pcg-conditional-loop-addon` is **not** one
of the four named experiments, and it is the non-experiment home of that work (it also
uniquely holds `sim/cuda_reduction.py`). But leaving it unstated invites a later reviewer to
either (a) flag the extraction as smuggling experiment code, or (b) "helpfully" pull the
missing pieces from #16. Make the boundary explicit.

**Exact recommended edit** — in row 8's Recommendation column and in §2.3, add:

```
Note: the `pressure.py` and pressure/hotpath test deltas in this worktree are
byte-identical to the same deltas carried in the excluded native-interop experiment
(#16), which shares the base commit `c182d3a`. The orbit worktree is the
non-experiment source of record for this work; extract exclusively from it. Do not
open, read, or copy anything from #15/#16 during the extraction, and do not treat
the identical delta as evidence that the experiment is a merge candidate.
```

### 3. MEDIUM — Extraction is exposed to CRLF/LF whole-file rewrites; the plan never warns about it
Rows 2, 8, and 10 all show far more `git status` entries than `git diff --stat` files: 10 of
14 (row 8) and 8 of 10 source files (row 2) are pure line-ending churn. `AGENTS.md`
(repository orientation) requires preserving a file's existing endings. A naive
`git add -A` / `git stash` / patch-apply during extraction will commit thousands of
no-op EOL lines and destroy the reviewability of the real 533- and 174-line deltas.

**Exact recommended edit** — add to §2 preamble:

```
Extraction hygiene: several candidate worktrees show EOL-only churn
(`git status` lists ~10 more files than `git diff --stat` for #2 and #8). Extract with
explicit per-file `git diff -- <path>` patches for only the files that appear in
`git diff --stat`, and verify the resulting branch's `git diff --stat` against master
matches the expected file count and insertion count (#2: 2 files / 174 insertions;
#8: 3 files / 533 insertions; plus untracked `sim/cuda_reduction.py`).
```

### 4. MEDIUM — Validation lanes are named too loosely relative to `AGENTS.md`
`AGENTS.md` prescribes exact commands; the plan paraphrases them, which weakens the gate.

- §2.1 (ST kernel corrections) changes `k_spacetime_weights_to_phase_active`, i.e. sparse
  leaf-grid fringe behaviour and active-cell/Dirichlet classification. `AGENTS.md` mandates
  the lane `python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_snippet_library.py tests/test_production_gate_truthfulness.py tests/test_sparse_leaf_grid_contract.py tests/test_viscosity_active_work_regressions.py`
  for sparse leaf-grid / active-viscosity changes. The plan says only "focused ST-FLIP
  validation plus the production gate". Add the sparse-leaf-grid lane by name, plus
  `blender --background --factory-startup --python tests/headless_spatiotemporal_validation.py -- --device cuda:0`
  and `tests/headless_stflip_benchmark.py`.
- §2.2 (sourcing) says "Run sourcing regressions and Blender source acceptance lanes".
  `AGENTS.md` names five capability scripts (`blender_source_lifecycle_acceptance.py`,
  `blender_source_region_acceptance.py`, `blender_animated_source_acceptance.py`,
  `blender_point_source_acceptance.py`, `blender_volume_source_acceptance.py`), each with
  `-- --json-out artifacts/source_acceptance/<capability>.json` and
  `python tools/validate_source_acceptance_report.py`, expecting
  `*-SOURCE-ACCEPTANCE-PASS` + `SOURCE-ACCEPTANCE-REPORT-PASS`. Also add
  `tests/blender_mesh_source_acceptance.py` (`MESH-SOURCE-ACCEPTANCE-PASS`) and
  `python -m pytest -q tests/test_sourcing_pipeline.py tests/test_mesh_source_pipeline.py`.
- §2.3 says "measured GPU-profile improvement" without a command. Name
  `"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/profile_gpu_solver.py --device cuda:0 --frames 8 --warmup 2 --json-out artifacts/gpu_solver_profile.json`
  and `blender --background --factory-startup --python tests/blender_memory_safety_gate.py`,
  and require a **baseline** profile on pre-extraction master for comparison (currently the
  plan asks for "improvement" with no baseline artifact).
- §1.3 lists "Phase-4 CUDA acceptance", "handler persistence gate", "viewport and
  cache-export gates" informally. Replace with the literal `AGENTS.md` commands
  (`tests/blender_handler_persistence_gate.py`, `tests/blender_viewport_2frame_gate.py`,
  `tests/blender_cache_export_pipeline.py`, `tests/blender_memory_safety_gate.py`).

### 5. MEDIUM — Step 1.1 is the riskiest step and is the least specified
"Preserve or explicitly resolve main/master's dirty `.gitignore` and generated demo files."
Master's dirty `.gitignore` is plausibly the file that would ignore `.agents/`,
`.pi-subagents/`, and demo previews; leaving it unresolved means the demo-binary churn
question in §3 recurs at every subsequent step. Recommend making it explicit and first:

```
1. Inspect `git diff -- .gitignore` on master. If it covers `.agents/`,
   `.pi-subagents/`, and generated demo artifacts, commit it on master before the
   Phase-4 merge. Then decide the authoritative `demos/*.blend` / `*_preview.png`
   by regenerating with `blender --background --python tools/build_demo_scenes.py`
   and verifying with `tools/check_demo_blends.py` + `python tools/analyze_demos.py`
   (expect `DEMO-ANALYSIS-PASS (18/18)`) rather than adopting any worktree's binaries.
```

Also: `merge-tree` was reported clean at assessment time. Master can move; add
"re-run `git merge-tree --write-tree phase4/live-interaction-merge-20260728 master`
immediately before step 1.2 and abort if it reports a conflict."

### 6. LOW — Small factual gaps
- Row 3: dirty set also includes `demos/dam_break/meta.json`. Row 11's dirty set likewise
  includes `demos/crown_splash/meta.json` and `demos/dam_break/meta.json` (plan says only
  "kernels and demo outputs" — acceptable but less precise than rows 1/2).
- Row 13: the `stflip-implementation` worktree is **`locked`**. Any accidental
  `git worktree prune`/`remove` will refuse — which is good, and worth stating as an
  existing safeguard consistent with the `AGENTS.md` exclusion.
- Row 16: worth stating that `codex/native-gpu-interop-main-20260715` has **zero unique
  commits** (tip `c182d3a` is an ancestor of master); the entire experiment lives in the
  index/worktree. This matters for cleanup: the branch itself is disposable, the checkout is not.

### 7. LOW — Scope note, not a defect
The plan covers only the 16 registered worktrees. The repository carries ~43 branches,
including several with no worktree that are not obviously contained in master
(`codex/deepen-pressure-module`, `codex/deepen-snippets-module`, `codex/deepen-sourcing-module`,
`codex/viewport-rendering`, `codex/orbit-st-flip-free-surface-completion`,
`feature/snippet-library`, `perf-next-optimizations-20260704`). The "Known limitations"
section should say a branch-level residual sweep is out of scope and still open, so the
plan is not later read as a complete residual-work inventory.

## What is already correct (evidence)

- **Experiment handling is fully compliant with `AGENTS.md`.** All four named experiments
  (#12, #13, #15, #16) are scored 0/5, labelled `EXPERIMENT—exclude`, excluded from the
  execution plan, and §3 explicitly says to leave them untouched "unless explicitly
  reactivated by the user". No step rebases, merges, cherry-picks, or extracts from them.
  The plan also correctly does **not** frame their overlap as future merge work.
- **The scoring rubric is stated and applied consistently.** The stated basis ("explicitly
  classified experiments receive 0/5 because they are excluded from integration—not because
  their ideas have no technical value") matches every 0/5 assignment. The three non-zero
  non-target scores (3/3/4) track substantive-delta size and risk: #2 = 174 insertions in
  2 files, #11 = 13 insertions in 1 kernel file, #8 = 533 insertions in 3 files plus a new
  module and CUDA/private-API risk. That ordering is defensible.
- **Ordering minimizes risk correctly.** Phase-4 first is right: #9 is the only clean,
  ahead-of-master candidate (3/8) and the only one with a completed validation story;
  landing it first means the three extractions rebase onto a settled tree exactly once
  instead of three times. Extract-onto-fresh-branch (never merge dirty worktrees) is the
  correct pattern for #2/#8/#11, all of which are 34–46 commits behind.
- **No genuine work appears misclassified.** I independently confirmed #10's tracked work is
  a strict subset of #9 (the 0/5 is earned, not assumed), #3/#4/#5/#6/#7 have zero unique
  commits, and #14 is a 107-behind snapshot whose only tracked deltas are two deletions plus
  an operators edit. The only work at risk of loss is untracked (finding 1).
- **Row 11's technical description is verifiably accurate**, including the physics rationale
  (tau clamped to the `[-1/2, 1/2]` W_T support to avoid dropping particles; averaging phi
  over existing faces rather than a fixed 6 to avoid biasing fringe cells toward Dirichlet air).
  That is the strongest-evidenced row in the document.
- **"Known limitations" is honest** about untested extraction ROI and about `merge-tree`
  predicting textual, not semantic, integration.

## Summary of required edits

1. Rewrite §3 bullets 1–3 with an untracked-file archive gate (finding 1). **Required.**
2. Add the "byte-identical to excluded experiment #16" note to row 8 and §2.3 (finding 2). **Required.**
3. Add the EOL-hygiene / expected-diffstat paragraph to §2 (finding 3). **Required.**
4. Replace paraphrased gates in §1.3, §2.1, §2.2, §2.3 with the literal `AGENTS.md`
   commands and add a pre-extraction GPU baseline profile (finding 4). **Required.**
5. Expand §1.1 and add the merge-tree re-run before §1.2 (finding 5). **Recommended.**
6. Correct row 3's dirty set; note the `stflip-implementation` lock and #16's zero unique
   commits (finding 6). **Recommended.**
7. Add a branch-level-sweep scope caveat to "Known limitations" (finding 7). **Optional.**

With edits 1–4 applied, this plan is safe to execute as written.

---
Read-only review: no repository files were created, modified, or deleted, and nothing was staged.
