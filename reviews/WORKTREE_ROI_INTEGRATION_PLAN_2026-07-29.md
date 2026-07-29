# Worktree ROI and Integration Plan — 2026-07-29

## Scope and scoring

This is a current-state integration plan for the 16 registered worktrees. ROI measures the value of integrating or carefully extracting a worktree's changes into current `master`, balanced against staleness, conflict risk, and validation cost. Explicitly classified experiments receive `0/5` because they are excluded from integration—not because their ideas have no technical value.

Current reference points:

- `master`: `556bce2`
- Phase-4 landing worktree: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_phase4_commit`
- Phase-4 branch: `phase4/live-interaction-merge-20260728` at `4d1a895`
- Phase-4 divergence when assessed: 3 commits behind / 8 commits ahead of master
- `git merge-tree --write-tree phase4/live-interaction-merge-20260728 master` completed without reporting a conflict.

## Worktree assessment

| # | Worktree / branch | High-level purpose and current changes | State vs master | ROI | Recommendation |
|---:|---|---|---|---:|---|
| 1 | Main `liquid_warp_blender51_port` / `master` | Canonical baseline. Recent commits enforce report authority and add a real Domain Height Fill emission lifecycle. Dirty files are `.gitignore`, generated demo `.blend`/preview files, and `.agents/`; no dirty source code. | Current at `556bce2` | **5/5** | Integration target. Resolve or preserve generated/config changes before landing Phase 4. |
| 2 | `34f4/...` / `codex/fix-cone-pyramid-distances` | Historical exact cone/pyramid SDF attempt. Current master already has deeper exact `_cone_sdf`/`_pyramid_sdf` implementations, degenerate-dimension handling, world-transform coverage, and stronger edge/resolution tests. | 34 behind, no unique commits, dirty | **0/5** | Do not extract code. Archive its non-duplicate authored backlog files before eventual removal. |
| 3 | `54d6/...` / `codex/deepen-four-modules` | No remaining module implementation; only modified `dam_break.blend`, preview, and `meta.json` output. | 38 behind, no unique commits | **0/5** | Prune after confirming the demo files are not authoritative. |
| 4 | `arch-deepen/...` / `codex/deepen-fluidsim-architecture-seams` | Historical architecture/deep-module work already absorbed elsewhere. | Clean, 61 behind, no unique commits | **0/5** | Prune. |
| 5 | `e9bf/...` / detached | Historical clean snapshot with no unique work. | Clean, 43 behind | **0/5** | Prune. |
| 6 | `merge-architecture-main/...` / `codex/integrate-master-architecture-20260713` | Historical architecture integration branch already contained in master. | Clean, 48 behind, no unique commits | **0/5** | Prune. |
| 7 | `orbit-gpu-pcg-baseline` / detached | Baseline checkout for the PCG experiment; no changes of its own. | Clean, 46 behind | **0/5** | Prune after the conditional-loop work is safely assessed. |
| 8 | `orbit-gpu-pcg-conditional-loop-addon` | Upgrades pressure PCG to a GPU-resident conditional graph loop. Adds an allocation-free compiled CUDA dot-product reduction and tests intended to eliminate repeated host synchronization/readback. | 46 behind, uncommitted; 3 substantive modified files plus new `cuda_reduction.py` | **4/5** | High potential, high risk. This worktree is the non-experiment source of record even though its pressure/test deltas are byte-identical to experiment #16 at shared base `c182d3a`. Extract exclusively from #8; never read or copy #15/#16 during extraction. Audit Warp private-API/build portability, then require correctness and GPU benchmark proof. |
| 9 | `liquid_warp_blender51_phase4_commit` / `phase4/live-interaction-merge-20260728` | Completed Phase 4: atomic live-setting reconciliation, pending/reinitialize workflow, UI feedback, cache signatures, ST-FLIP live calibration, and acceptance lanes. | Clean; 3 behind / 8 ahead | **5/5** | Highest priority. Merge current master into it, rerun full validation, then land it. |
| 10 | `liquid_warp_blender51_phase4_interaction` / `phase4/live-property-interaction` | Original dirty development checkout containing Phase 1–4 history, source work, tests, demos, and reviews. Its intended Phase-4 payload was integrated into #9. | 19 behind; 130 dirty entries | **0/5** | Do not merge. Preserve until a final comparison confirms #9 captured the intended payload, then archive/remove. |
| 11 | `.claude/worktrees/addon-st-flip-review-717052` | Its committed policy-evidence repair is substantially superseded by master’s authority commit. Its dirty kernel delta contains two potentially useful fixes: clamp temporal samples to ST support and average phase over existing sparse-grid faces instead of always dividing by six. | 4 behind / 1 ahead; dirty kernel plus demo, preview, and metadata outputs | **3/5** | Do not merge the branch. Extract the two kernel changes into a new branch with focused regression tests; discard superseded policy and generated-demo changes. |
| 12 | `.claude/worktrees/particle-surface-rendering-apis-f62356` | Rendering-APIs branched experiment with nested validation/report and demo evidence. | 21 behind / 2 ahead; dirty | **0/5** | **EXPERIMENT—exclude from analysis and integration.** |
| 13 | `.claude/worktrees/stflip-implementation` | Broad ST-FLIP implementation/benchmark experiment touching solver, pressure, kernels, UI, validation, and benchmark scripts. | Based on current master; 16 dirty entries; worktree is locked | **0/5** | **EXPERIMENT—exclude from analysis and integration.** Keep the worktree lock as a deletion safeguard. |
| 14 | `liquid_warp_blender51_port_cherry_pick_merge` / detached | Abandoned cherry-pick snapshot. Deletes `CLAUDE.md` and `sim/__init__.py`, modifies operators, and contains a review `.blend`. | 107 behind | **0/5** | Preserve the review file only if needed, then prune. Never merge the deletions. |
| 15 | `liquid_warp_blender51_port_native_interop` / `codex/native-gpu-interop` | Older native CUDA/Vulkan viewport interop experiment with implementation, tests, generated evidence, demos, and binaries. | 73 behind / 1 ahead; heavily dirty | **0/5** | **EXPERIMENT—exclude from analysis and integration.** |
| 16 | `liquid_warp_blender51_port_native_interop_main_20260715` / `codex/native-gpu-interop-main-20260715` | Larger uncommitted version of the native interop experiment, including staged implementation, pressure changes, acceptance tests, profiling, and evidence. | 46 behind; zero unique commits; all experiment content is staged/unstaged/untracked | **0/5** | **EXPERIMENT—exclude from analysis and integration.** The branch ref is disposable; the checkout contents are not. |

## Proposed execution plan

### 1. Land Phase 4 first

1. Inspect `git diff -- .gitignore` on master. If it correctly excludes `.agents/`, `.pi-subagents/`, and generated artifacts, commit it separately before Phase 4. Decide authoritative demo binaries by regenerating with `blender --background --python tools/build_demo_scenes.py`, checking with `blender --background --python tools/check_demo_blends.py`, and running `python tools/analyze_demos.py` (expect `DEMO-ANALYSIS-PASS (18/18)`), rather than adopting another worktree's binaries.
2. Immediately before integration, rerun `git merge-tree --write-tree phase4/live-interaction-merge-20260728 master`; stop and resolve deliberately if it reports conflicts. Merge current master into `phase4/live-interaction-merge-20260728`.
3. Rerun these literal gates:
   - `python -m pytest -q`
   - `blender --background --factory-startup --python tests/blender_phase4_live_interaction_acceptance.py`
   - `blender --background --factory-startup --python tests/blender_handler_persistence_gate.py`
   - `blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py`
   - `blender --background --factory-startup --python tests/blender_cache_export_pipeline.py`
   - `blender --background --factory-startup --python tests/blender_memory_safety_gate.py`
   - `python tools/run_production_gate.py --include-gui`
4. Inspect the resulting commit range and staged tree, then merge the validated Phase-4 branch into master.

### 2. Extract worthwhile residual fixes independently

After Phase 4 lands, create fresh branches from the new master. Do not directly merge old dirty worktrees.

Extraction hygiene: candidate worktrees contain EOL-only churn. Extract explicit per-file patches only for paths appearing in `git diff --stat`, preserve each destination file's current line endings, and verify expected scope before commit: #2 is 2 files / 174 insertions; #8 is 3 tracked files / 533 insertions plus untracked `sim/cuda_reduction.py`. Never use `git add -A` in a source worktree.

#### 2.1 ST kernel corrections from `addon-st-flip-review` — ROI 3

- Extract temporal-support clamping and sparse-fringe face-count normalization only.
- Add focused regression tests for both behaviors.
- Exclude the superseded policy-evidence commit and generated demo files.
- Run `python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_snippet_library.py tests/test_production_gate_truthfulness.py tests/test_sparse_leaf_grid_contract.py tests/test_viscosity_active_work_regressions.py`.
- Run `blender --background --factory-startup --python tests/headless_spatiotemporal_validation.py -- --device cuda:0` and `blender --background --factory-startup --python tests/headless_stflip_benchmark.py -- --device cuda:0 --frames 6`.
- Run `python tools/run_production_gate.py --include-gui` before landing.

#### 2.2 Exact cone/pyramid source distances — superseded during execution

A current-master extraction was attempted and produced conflicts because master already contains deeper exact-distance helpers from `f5a9d12`, including finite degenerate handling and stronger exact surface, resolution-offset, and world-transform tests. The candidate's two source-only tests duplicate existing coverage at lower generality. No code was extracted. Preserve only its non-duplicate untracked authored backlog files during cleanup.

#### 2.3 GPU conditional PCG — ROI 4

- Port `pressure.py`, `cuda_reduction.py`, and focused tests into a fresh post-Phase-4 branch.
- The `pressure.py` and pressure/hotpath test deltas are byte-identical to experiment #16 because both share base `c182d3a`; #8 is the non-experiment source of record. Extract exclusively from #8 and do not open, read, or copy #15/#16.
- Audit CUDA compilation/cache lifecycle, Warp private APIs, device teardown, graph capture, numerical stop reasons, and fallback behavior.
- Before extraction, capture a baseline with `"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/profile_gpu_solver.py --device cuda:0 --frames 8 --warmup 2 --json-out artifacts/gpu_solver_profile_baseline.json`.
- After extraction, run the same command with `--json-out artifacts/gpu_solver_profile.json`, plus pressure regressions, `blender --background --factory-startup --python tests/blender_memory_safety_gate.py`, and `python tools/run_production_gate.py --include-gui`. Require numerical correctness, no memory-safety regression, and a measured improvement before integration.

### 3. Cleanup after preservation

- Before any removal, run `git -C <worktree> status --porcelain` and archive every `??` entry that is not reproducible output, recording the archive path. Known non-duplicate authored files include `34f4` policy tasks 004/006/011 and `phase4_interaction` review tasks 001/002 plus policy tasks 004/007/010.
- Remove clean, fully contained worktrees #4/#5/#6/#7 with plain `git worktree remove` and never `--force`; any failure requires re-inventory.
- Archive and remove #10 only after both a tracked-file comparison against the Phase-4 landing branch and an untracked-file archive. Independent review verified #9 is a strict superset of #10 for tracked source.
- Remove #14 only after archiving `review_2635080.blend`; never merge its deletions.
- Decide which generated demo files are authoritative rather than merging binary churn from multiple worktrees.
- Keep the rendering-APIs, ST-FLIP implementation, and both native-interop experiments untouched and excluded unless explicitly reactivated by the user.

## Known limitations of this assessment

- Git state and substantive-path inventories were refreshed against current master.
- No candidate dirty worktree was validated by running its test suite during this assessment.
- ROI for the cone/pyramid, ST kernel, and PCG candidates is therefore provisional until each extraction is tested on current post-Phase-4 master.
- A conflict-free `merge-tree` result predicts textual integration, not semantic correctness; all Phase-4 gates must still be rerun after merging current master.
- This plan covers the 16 registered worktrees only. A branch-level residual sweep is out of scope and remains open for branches without registered worktrees.
