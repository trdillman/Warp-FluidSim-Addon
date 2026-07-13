# Unmerged Worktree Report

Generated: 2026-07-07

Repository: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`

Base branch: `master`

Base commit: `87971a7 fix(validation): harden policy drift gates`

## Master Preparation

- Reviewed the dirty `master` remediation set covering cache/live playback ownership, sparse tiled leaf-grid wording, active-work bounds, production-gate truthfulness, policy-drift docs, and focused regressions.
- Ran focused validation before commit:
  `python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_snippet_library.py tests/test_production_gate_truthfulness.py tests/test_sparse_leaf_grid_contract.py tests/test_viscosity_active_work_regressions.py`
- Result: `43 passed in 3.61s`.
- Committed master remediation as `87971a7 fix(validation): harden policy drift gates`.
- Confirmed clean `master` before worktree comparison.

## Summary

| Category | Count | Worktrees |
| --- | ---: | --- |
| Clean base | 1 | main `master` checkout |
| Unmerged branch work | 2 | `codex/audit-corrections`, `codex/diagnose-simulation-slowdown` |
| Dirty-only local work | 1 | detached `liquid_warp_blender51_port_cherry_pick_merge` |
| Merged or superseded clean worktrees | 4 | `29a2`, `49e3`, `d073`, `dbfc`, `eed4` |
| Missing/prunable metadata | 2 | `/mnt/c/...first_checkout`, `/mnt/c/...viewport_viz` |

## Unmerged Branch Work

### `codex/audit-corrections`

- Path: `C:\Users\Tyler\.codex\worktrees\b88b\liquid_warp_blender51_port`
- Branch: `codex/audit-corrections`
- Head: `a0b6496 fix(solver): mitigate live whitewater crash`
- Unique commits vs clean `master`: 2
- Behind clean `master`: 2
- Dirty: yes
- Category: unmerged bug-fix / validation lane with dirty generated artifacts

Unique commits:

- `a0b6496 fix(solver): mitigate live whitewater crash`
- `4d6a2c9 fix(solver): close audit remediation gates`

Touched surfaces:

- Validation and reporting: `POLICY_DRIFT_REPORT.md`, `tools/run_production_gate.py`, `tests/blender_viewport_2frame_gate.py`, production-gate tests.
- Solver/runtime: scene bridge, viewport, custom kernels, particles, pressure, solver, validation, viscosity.
- Docs and policy drift: sparse-grid wording, porting brief, validation plan, user/technical docs.
- Demo assets: `demos/*.blend` and preview PNGs.

Dirty state:

- Modified generated demo assets: `demos/crown_splash.blend`, `demos/crown_splash_preview.png`, `demos/dam_break.blend`, `demos/dam_break_preview.png`.
- Untracked directories: `.claude/`, `audit/`, `bench/`.

Recommended next action:

- Rebase or cherry-pick the two commits onto `87971a7`, then resolve overlap with master gate hardening.
- Treat the dirty demo assets as generated evidence; inspect before staging, do not merge them blindly.

### `codex/diagnose-simulation-slowdown`

- Path: `C:\Users\Tyler\.codex\worktrees\ed32\liquid_warp_blender51_port`
- Branch: `codex/diagnose-simulation-slowdown`
- Head: `a08df0f fix: bound particle relaxation slowdown`
- Unique commits vs clean `master`: 1
- Behind clean `master`: 10
- Dirty: yes
- Category: unmerged performance / bug-fix lane with dirty follow-up edits

Unique commit:

- `a08df0f fix: bound particle relaxation slowdown`

Touched surfaces:

- Runtime and profiling: diagnostics profiler, operators, properties, constants, fluid kernels, solver, surface.
- Tooling: demo build script, monitored quick setup, user-path profiler.
- Tests: slowdown telemetry regression.

Dirty state:

- Modified `fluidSim_warp_blender51/sim/particles.py`.
- Modified `fluidSim_warp_blender51/sim/whitewater.py`.
- Modified `tests/test_slowdown_telemetry.py`.

Recommended next action:

- Preserve this lane as a candidate performance merge.
- First review the three dirty follow-up files, then rebase the committed slowdown fix onto `87971a7`; expect overlap in properties/solver behavior.

## Dirty-Only Local Work

### Detached cherry-pick merge worktree

- Path: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_cherry_pick_merge`
- Branch: detached
- Head: `2635080 merge: preserved master cherry-picks`
- Unique commits vs clean `master`: 0
- Behind clean `master`: 20
- Dirty: yes
- Category: dirty-only local work; committed head appears merged or superseded

Dirty state:

- Modified `fluidSim_warp_blender51/operators.py`.
- Untracked `review_2635080.blend`.

Recommended next action:

- Diff `operators.py` directly against `87971a7` and either port the local edit or discard only after review.
- Treat `review_2635080.blend` as a local review artifact, not a merge candidate unless its provenance is confirmed.

## Merged Or Superseded Clean Worktrees

### Detached `29a2`

- Path: `C:\Users\Tyler\.codex\worktrees\29a2\liquid_warp_blender51_port`
- Head: `1fc54e2 merge: promote realtime surfacing controls`
- Unique commits vs clean `master`: 0
- Behind clean `master`: 2
- Dirty: no
- Category: merged or superseded
- Recommended next action: no merge needed; prune later if no longer useful.

### `codex/plan-to-allow-realtime-updates`

- Path: `C:\Users\Tyler\.codex\worktrees\49e3\liquid_warp_blender51_port`
- Head: `f249ddd feat: add realtime surfacing display controls`
- Unique commits vs clean `master`: 0
- Behind clean `master`: 5
- Dirty: no
- Category: merged or superseded
- Recommended next action: no merge needed; branch can be retired after confirmation.

### Detached `d073`

- Path: `C:\Users\Tyler\.codex\worktrees\d073\liquid_warp_blender51_port`
- Head: `cf1b00a chore: clean artifacts and add production gate`
- Unique commits vs clean `master`: 0
- Behind clean `master`: 11
- Dirty: no
- Category: merged or superseded
- Recommended next action: no merge needed; prune later if no longer useful.

### Detached `dbfc`

- Path: `C:\Users\Tyler\.codex\worktrees\dbfc\liquid_warp_blender51_port`
- Head: `27b3024 perf(solver): bound compact active ranges`
- Unique commits vs clean `master`: 0
- Behind clean `master`: 6
- Dirty: no
- Category: merged or superseded
- Recommended next action: no merge needed; prune later if no longer useful.

### `codex/update-cache-export-support`

- Path: `C:\Users\Tyler\.codex\worktrees\eed4\liquid_warp_blender51_port`
- Head: `f43e276 feat(cache): add export playback pipeline`
- Unique commits vs clean `master`: 0
- Behind clean `master`: 5
- Dirty: no
- Category: merged or superseded
- Recommended next action: no merge needed; branch can be retired after confirmation.

## Missing Or Prunable Metadata

### `/mnt/c/...first_checkout`

- Path: `/mnt/c/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port_first_checkout`
- Status: missing path; `git worktree list` reports prunable metadata.
- Last recorded head: `9311d5d`
- Category: stale worktree metadata, not a merge candidate
- Recommended next action: prune only if Tyler wants cleanup; no code merge action.

### `/mnt/c/...viewport_viz`

- Path: `/mnt/c/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port_viewport_viz`
- Status: missing path; `git worktree list` reports prunable metadata.
- Last recorded head: `a04b2c9`
- Category: stale worktree metadata, not a merge candidate
- Recommended next action: prune only if Tyler wants cleanup; no code merge action.

## Next Merge Order

1. Review and either preserve or discard dirty-only `operators.py` edit in `liquid_warp_blender51_port_cherry_pick_merge`.
2. Port `codex/diagnose-simulation-slowdown` if the slowdown fix is still desired; it is the narrowest unmerged feature/optimization lane.
3. Port `codex/audit-corrections` after checking overlap with the newly committed production-gate and policy-drift fixes.
4. Retire clean superseded worktrees after confirming no local artifact needs preservation.
