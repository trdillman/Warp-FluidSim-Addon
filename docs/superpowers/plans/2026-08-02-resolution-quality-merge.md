# Resolution Quality Merge Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revalidated:** 2026-08-05 against live repository state.

**Goal:** Integrate `feature/resolution-quality-standardization` into current `master` without losing the committed source-capacity corrections on master, disturbing the user's four regenerated demo/preview assets, duplicating sibling slices, or importing any archived experiment.

**Architecture:** Treat `5d5e468` as the observed repository baseline. Let the active evidence-only rebind finish and land separately, commit this updated plan alone, then capture that bounded descendant as `BASE_HEAD` and work only in a disposable integration worktree created from it. Start one no-commit merge, resolve all conflict groups and run focused checks while that merge is pending, then create exactly one merge commit. Run HEAD-sensitive full regression, production, GUI, playback, and policy-report gates only after that merge commit exists.

**Tech Stack:** Git worktrees, PowerShell, Python 3, Blender 5.1 bundled Python, Warp/CUDA, pytest.

## Global Constraints

- Preserve the original master worktree. Do not reset, clean, stash, checkout over, overwrite, or broadly stage its four modified generated files.
- The live field contract remains fixed 8x8x8 sparse leaf tiles addressed through a flat leaf table.
- Production simulation remains GPU-resident; CPU preview cannot substitute for CUDA acceptance.
- Preserve transactional behavior: failed admission, growth, solver replacement, source application, topology commit, or frame publication leaves prior authoritative state unchanged.
- Merge the umbrella branch once. Do not merge `agent/res-quality-t01-source-truth`, `agent/res-quality-t02-session-capacity`, or `agent/res-quality-t03-whitewater-truth` afterward.
- The four AGENTS.md experiments remain `EXPERIMENT — NOT A MERGE CANDIDATE`: `worktree-stflip-implementation`, `claude/particle-surface-rendering-apis-f62356`, `codex/native-gpu-interop`, and `codex/native-gpu-interop-main-20260715`.
- Do not resolve semantic conflicts with whole-file `ours` or `theirs`. Preserve both sides' proven corrections at the shared seam.
- Do not make task-level commits while the merge is pending. All 23 conflict resolutions belong to the single merge commit.
- The policy-report-authority correction landed separately as `5d5e468`. Before execution, its focused test must pass and the five current policy-report/document refreshes must land in a separate evidence-only commit. Do not absorb or rewrite that work here.
- Commit this plan alone afterward without staging the four user-owned generated files. No source, test, runtime, or other documentation change may enter that plan-only commit.
- Run Blender/OpenVDB workflows with Blender's bundled Python or the documented Blender command, not system Python.
- A special alias, partial lane, stale report, missing marker, or mismatched HEAD/hash is not production proof.

---

## 0. Live authority and scope

State verified on 2026-08-05:

| Worktree / branch | Relation to current master | State | Treatment |
|---|---:|---|---|
| `master / 5d5e468` | observed target baseline | four modified generated demo/preview files, this plan edit, and five policy-report/document refreshes; no untracked files | Block until evidence edits land separately; then commit plan alone |
| `feature/resolution-quality-standardization / c82988f` | 21 ahead, 14 behind | clean; 67-file umbrella change | Sole merge source |
| `agent/res-quality-t01-source-truth / 2a55a62` | 3 ahead, 13 behind | clean | Reference only; patch-represented in umbrella |
| `agent/res-quality-t02-session-capacity / e36d84d` | 6 ahead, 13 behind | clean | Reference only; range-diff adapted in umbrella |
| `agent/res-quality-t03-whitewater-truth / 365db1f` | 3 ahead, 13 behind | clean | Reference only; patch-represented in umbrella |
| `phase4/live-interaction-merge-20260728 / 701490ed` | 21 behind, no unique commits | clean | Do not merge |
| archived experiments | irrelevant to this merge | clean or dirty experimental records | Explicitly excluded |

The prior dirty source/capacity work is now committed on master:

- `62d013a feat(sim): fail closed on source capacity with transactional emission`
- `7977e0d test(gate): require capacity-planner acceptance in the headless gate`
- `92d60af fix(sim): gate direct particle injection on the device counter`
- `1c6381a chore(git): ignore generated caches and fix dangling doc reference`
- `c2bae4c docs(evidence): rebind policy-drift reports to the current full gate`
- `c2df030 docs(plan): revalidate the resolution-quality merge plan against live state`
- `5d5e468 fix(tools): let policy-drift validation survive its own commit`

`git cherry` shows one T02 commit (`7ffdb5c`) is not patch-identical to the umbrella. `git range-diff` maps it to adapted commit `08a78a8`, where source binding uses the umbrella's `set_source_batch` seam. Treat that as semantic adaptation, not proof of byte-equivalent commits.

## 1. Current conflict inventory

A read-only `git merge-tree --write-tree master feature/resolution-quality-standardization` from merge-base `f1e9b1e` predicts 23 conflicts:

**Documents**

- `docs/industry_standard_resolution_quality_implementation_plan.md`
- `docs/parallel_resolution_quality_subagent_orchestration_plan.md`
- `fluidSim_warp_blender51/docs/user_guide.md`

**Planning, runtime, and UI**

- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/blender/sources.py`
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/panels.py`
- `fluidSim_warp_blender51/properties.py`
- `fluidSim_warp_blender51/sim/capacity_planner.py`
- `fluidSim_warp_blender51/sim/particle_capacity.py`
- `fluidSim_warp_blender51/sim/solver.py`
- `fluidSim_warp_blender51/sim/source_regions.py`
- `fluidSim_warp_blender51/sim/sourcing.py`
- `fluidSim_warp_blender51/sim/surface.py`

**Tests and acceptance**

- `tests/blender_capacity_planner_acceptance.py`
- `tests/test_capacity_planner.py`
- `tests/test_operator_panel_behavior.py`
- `tests/test_particle_capacity_growth.py`
- `tests/test_phase4_pending_interaction.py`
- `tests/test_scene_bridge_capacity_initialization.py`
- `tests/test_source_emission_lifecycle.py`
- `tests/test_source_region_compilation.py`
- `tests/test_sourcing_pipeline.py`

If the actual unmerged-path list differs, stop and classify the difference before resolving anything.

## 2. Merge tasks

### Task 1: Freeze the original master evidence

**Files:**
- Preserve outside the repository: `C:\tmp\fluidsim-master-generated-20260805\`
- Record outside the repository: `C:\tmp\fluidsim-master-status-20260805.txt`
- Record outside the repository: `C:\tmp\fluidsim-master-generated-20260805.sha256`
- Record outside the repository: `C:\tmp\fluidsim-resolution-quality-base-20260805.txt`

- [ ] Confirm `5d5e468` changes only `tools/validate_policy_drift_reports.py` and `tests/test_policy_drift_report_authority.py`. If any policy-report/document path below is still dirty, stop until its owner finishes the evidence lane:

~~~text
POLICY_DRIFT_REPORT.md
demos/ANALYSIS.md
demos/POLICY_DRIFT_REPORT.md
demos/analysis_report.json
fluidSim_warp_blender51/docs/technical_report.md
~~~

- [ ] Require `python -m pytest -q tests/test_policy_drift_report_authority.py` to pass against `5d5e468` or its bounded descendant.
- [ ] Require a following evidence-only commit containing only the five report/document files above; it must not include code, tests, binaries, or this plan.
- [ ] Commit this updated plan by itself afterward. Do not stage any generated demo/evidence file.
- [ ] Verify `5d5e468` is an ancestor of current `master`.
- [ ] Inspect `git diff --name-only 5d5e468..master`; permit only this plan and the five policy-report/document paths listed above. Stop on any other path.
- [ ] Record current `master` as `BASE_HEAD` in `C:\tmp\fluidsim-resolution-quality-base-20260805.txt`.
- [ ] Run `git status --short --untracked-files=all`; after the plan-only commit, require exactly these four paths and no others:

~~~text
demos/crown_splash.blend
demos/crown_splash_preview.png
demos/dam_break.blend
demos/dam_break_preview.png
~~~

- [ ] Copy those four files, preserving names, into `C:\tmp\fluidsim-master-generated-20260805\`.
- [ ] Record SHA-256 hashes for the four originals and copies; require exact equality.
- [ ] Save the exact status output.
- [ ] Re-run status and hashes; stop if either changed.

**Current reference hashes; refresh rather than trust them if execution starts later:**

~~~text
5B03D31667CC7FBC4F1B38E9E622AF49B3A16B116C69AB19A24BD60594777C23  demos/crown_splash.blend
EEDA28780C90C724B5CC847C798E55545428D0F04B3126544742C3F76CC3211B  demos/crown_splash_preview.png
48C6A874F9DEB52CC503246A1E9F400C98E9BA409FC014FBA903D3877DB561EB  demos/dam_break.blend
BCE47069FF458D2498C91030C34B847F5E12C4C1BDCAE2441A7A89D49F8BCF5A  demos/dam_break_preview.png
~~~

### Task 2: Create the isolated integration worktree

**Files:**
- Worktree: `C:\tmp\fluidsim-resolution-quality-merge-20260805`
- Branch: `codex/merge-resolution-quality-20260805`

- [ ] Stop if either path or branch already exists; inspect it instead of deleting or reusing it.
- [ ] Run:

~~~powershell
$baseHead = Get-Content -Raw C:\tmp\fluidsim-resolution-quality-base-20260805.txt
git worktree add -b codex/merge-resolution-quality-20260805 C:\tmp\fluidsim-resolution-quality-merge-20260805 $baseHead.Trim()
~~~

- [ ] In the new worktree, verify `HEAD=BASE_HEAD`, branch identity, and a clean status.
- [ ] Do not copy the four generated demo/preview files into this worktree.

### Task 3: Start the one umbrella merge

**Inputs:**
- Repository baseline: `5d5e468`
- Execution base: recorded `BASE_HEAD`, differing from `5d5e468` only by the evidence-only rebind commit and the plan-only commit
- Merge source: `feature/resolution-quality-standardization=c82988f`

- [ ] Reconfirm `BASE_HEAD`, umbrella identity `c82988f`, and merge-base `f1e9b1e`.
- [ ] Load `$baseHead` from the receipt, run `git rev-list --left-right --count "$baseHead...feature/resolution-quality-standardization"`, require 21 umbrella-side commits, and inspect every base-side commit after `5d5e468`; permit only the two bounded commits from Task 1.
- [ ] Run:

~~~powershell
git merge --no-ff --no-commit feature/resolution-quality-standardization
~~~

- [ ] Capture `git diff --name-only --diff-filter=U`; expect the 23 paths in Section 1.
- [ ] Do not commit until Tasks 4–7 have resolved every conflict and focused checks pass.
- [ ] Do not merge any sibling slice or archived experiment.

### Task 4: Reconcile planning and runtime authority

**Files:**
- `fluidSim_warp_blender51/blender/scene_bridge.py`: master planning `159-419`; umbrella planning/replacement `264-687`
- `fluidSim_warp_blender51/blender/sources.py`: master prepared seeds `12-28, 151-262`; umbrella prepared seeds `12-28, 153-180`
- `fluidSim_warp_blender51/sim/capacity_planner.py`: master `276-526`; umbrella `290-573`
- `fluidSim_warp_blender51/sim/particle_capacity.py`: master `43-97`; umbrella `47-103`
- `fluidSim_warp_blender51/sim/solver.py`: master source/step `1058-1729`; umbrella exact injection/step `1098-1803`
- `fluidSim_warp_blender51/sim/source_regions.py`: master compiler `580-622`; umbrella compiler `615-657`
- `fluidSim_warp_blender51/sim/sourcing.py`: master transaction/direct injection `2355-2454, 2901-2967`; umbrella transaction/exact injection `2430-2536, 3063-3125`

- [ ] Keep umbrella `quality_contract.py` and `capacity_service.py` as the authored-to-resolved quality and public assessment authority.
- [ ] Keep master `PreparedParticleSeed`, `prepare_auto_seeds`, and atomic `commit_prepared_seeds`; automatic seeds must be included in initial admission before solver replacement.
- [ ] Keep umbrella candidate-solver replacement, scene-specific plan signatures, topology transaction, and exact packet injection.
- [ ] Preserve `92d60af`: direct injection admission is gated by the authoritative device counter, not a stale host count.
- [ ] Keep `set_source_batch(descriptors, extra_plans)`; do not reintroduce separate partial source setters.
- [ ] Keep source lifecycle advancement after whole-batch admission and application only.
- [ ] Keep cache playback isolated from solver initialization and live stepping.

- [ ] Run:

~~~text
python -m pytest -q tests/test_capacity_planner.py tests/test_capacity_service.py tests/test_quality_contract.py tests/test_particle_capacity_growth.py tests/test_scene_bridge_capacity_initialization.py tests/test_session_capacity_transactions.py tests/test_source_emission_lifecycle.py tests/test_source_region_compilation.py tests/test_sourcing_pipeline.py tests/test_topology_transaction.py tests/test_scene_bridge_cache_playback.py tests/test_frame_change_catchup.py tests/test_performance_hotpath_contracts.py
~~~

### Task 5: Reconcile surface, viewport, properties, panels, and operators

**Files:**
- `fluidSim_warp_blender51/sim/surface.py`: master pipeline `34-259`; umbrella pipeline `35-305`
- `fluidSim_warp_blender51/blender/viewport.py`
- `fluidSim_warp_blender51/properties.py`: callback `166-175`
- `fluidSim_warp_blender51/panels.py`: master capacity UI `115-182`; umbrella `123-190`
- `fluidSim_warp_blender51/operators.py`: master actions `155-543`; umbrella `186-542`
- `fluidSim_synth/ui_contract.py:18-20`
- `tests/test_frame_change_catchup.py:107-139`
- `tests/blender_phase4_live_interaction_acceptance.py`

- [ ] Preserve decoupled render-surface resolution, solver-SDF freshness, and `after_substep` / `after_frame` cadence without mutating authored settings.
- [ ] Preserve the master viewport point-cloud pre-warm correction from `0cde9cd`.
- [ ] Keep property edits plan-only: no callback may create or replace a solver.
- [ ] Keep capacity/safe-setting application as explicit, signature-revalidated operators.
- [ ] Preserve pending reinitialize/reset truth; do not report a setting applied before candidate replacement succeeds.
- [ ] Preserve `SIMULATION_ADVANCED_PLAYBACK = ["max_frame_catchup"]` and the actionable catch-up error naming **Maximum Playback Recovery Frames** under **Advanced Playback**.
- [ ] Preserve source role separation: Force/Field objects do not become liquid sources.

- [ ] Run:

~~~text
python -m pytest -q tests/test_source_sampling_migration.py tests/test_surface_resolution_feature_contracts.py tests/test_whitewater_truth_contract.py tests/test_height_source_pipeline.py tests/test_surface_stack.py tests/test_operator_panel_behavior.py tests/test_phase4_pending_interaction.py tests/test_frame_change_catchup.py
blender --background --factory-startup --python tests/blender_phase3_source_ui_acceptance.py -- --json-out artifacts/phase3_source_ui_acceptance.json
blender --background --factory-startup --python tests/blender_phase4_live_interaction_acceptance.py
~~~

Expected markers: `PHASE3-SOURCE-UI-ACCEPTANCE-PASS` and `PHASE4-LIVE-INTERACTION-PASS`.

### Task 6: Reconcile tests, gate wiring, and documents

**Files:**
- Every conflicted test in Section 1
- `tools/run_production_gate.py:182-365`
- `tests/test_production_gate_truthfulness.py`
- `tools/validate_policy_drift_reports.py`
- `tests/test_policy_drift_report_authority.py`
- `POLICY_DRIFT_REPORT.md`
- `demos/ANALYSIS.md`
- `demos/POLICY_DRIFT_REPORT.md`
- `demos/analysis_report.json`
- `fluidSim_warp_blender51/docs/technical_report.md`
- The three conflicted documents in Section 1

- [ ] Preserve master regression corrections and umbrella feature coverage in every conflicted test; do not choose a whole test file from one side.
- [ ] Verify the canonical headless gate invokes capacity-planner acceptance exactly once with `CAPACITY-PLANNER-ACCEPTANCE-PASS`.
- [ ] Verify Phase 4 live interaction remains in the canonical gate with `PHASE4-LIVE-INTERACTION-PASS`.
- [ ] Keep traceback, missing-marker, stale-manifest, and zero-exit false-green rejection.
- [ ] Preserve the separately committed policy-authority rule: an evidence-only commit after the proof HEAD is allowed, while any non-evidence path after the proof HEAD is drift.
- [ ] Resolve the two add/add plan documents against final code behavior. Remove stale pre-merge claims; keep historical orchestration records labelled historical rather than current proof.
- [ ] Update the user guide only with behavior preserved by the resolved implementation and focused tests.

- [ ] Run:

~~~text
python -m pytest -q tests/test_production_gate_truthfulness.py tests/test_policy_drift_report_authority.py tests/test_operator_panel_behavior.py tests/test_scene_bridge_cache_playback.py tests/test_source_region_compilation.py tests/test_performance_hotpath_contracts.py
~~~

### Task 7: Close the pending merge and create one merge commit

- [ ] Require `git diff --name-only --diff-filter=U` to print nothing.
- [ ] Run `git diff --check`.
- [ ] Inspect `git status --short --untracked-files=all`; classify every path.
- [ ] Re-run the focused commands from Tasks 4–6 after the final conflict edit.
- [ ] Review the merge index, then stage each resolved conflict path. Leave `artifacts/` reports and all other generated output unstaged.
- [ ] Create exactly one merge commit:

~~~text
merge: integrate resolution-quality umbrella
~~~

- [ ] Record the merge commit SHA and both parents.

### Task 8: Run post-commit production proof

HEAD-sensitive gates run only now, against the merge commit.

- [ ] Run the full Python regression:

~~~text
python -m pytest -q
~~~

- [ ] Run the canonical headless production gate:

~~~text
python tools/run_production_gate.py
~~~

- [ ] After headless passes, run the GUI-inclusive gate and validators:

~~~text
python tools/run_production_gate.py --include-gui
python tools/validate_policy_drift_reports.py
python tools/validate_gui_playback_report.py artifacts/gui_playback_probe.json
~~~

- [ ] Require current merge HEAD, gate verdict, manifest metadata, acceptance markers, checked-in demo JSON, GUI report, and artifact hashes to agree.
- [ ] If a gate fails, do not amend the merge commit. Make the smallest evidence-backed correction in a follow-up commit, rerun its focused check, then rerun every HEAD-sensitive gate because HEAD changed.
- [ ] Do not commit generated evidence unless the canonical workflow requires it. If an evidence commit changes HEAD, rerun production and report validators on that new HEAD.

### Task 9: Handoff and original-worktree proof

- [ ] Capture `BASE_HEAD`, integration HEAD, both merge parents, merge-base, branch status, focused test outputs, full pytest output, production-gate receipt/manifest, GUI report, and hashes.
- [ ] Recheck the original master worktree. Require its four paths and SHA-256 hashes to match Task 1 exactly.
- [ ] Do not copy, restore, stage, or otherwise modify those original generated assets.
- [ ] Do not promote the integration branch if any marker/report/hash is stale or any required lane is partial.

## 3. Acceptance invariants

- Unknown particle or leaf demand fails closed.
- Initial automatic seeds and frame-one source emissions share one admission decision.
- Particle growth charges particle, sparse topology, transaction scratch, and whitewater candidate workspace residency.
- Failed growth/replacement preserves the live solver, source lifecycle, leaf table, and published output.
- Whole-batch source lifecycle advances only after admission and successful application.
- Cache playback never initializes or steps a live solver and rejects settings/scene mismatches.
- Surface resolution remains decoupled from simulation resolution without rewriting authored settings.
- Whitewater never consumes an unavailable solver SDF and reports deterministic availability/drops.
- UI suggestions remain scene-owned, signature-safe, explicit, and reversible.
- Direct injection uses the authoritative device counter.
- Current HEAD, manifests, reports, hashes, and gate markers agree.

## 4. Stop and rollback conditions

- Stop before editing if the evidence edits remain uncommitted, the policy-authority focused test fails, the evidence or plan commits mix scopes, umbrella HEAD changes, repository history since `5d5e468` contains any other path, worktree status drifts, the conflict list changes, or original generated-file hashes differ from this plan.
- Stop if resolving a conflict requires choosing between incompatible planner, transaction, or publication authorities without a focused regression proving the choice.
- Stop if candidate failure releases/mutates the live solver, advances source lifecycle, commits sparse topology, or publishes partial output.
- Stop if any sibling slice or archived experiment appears in the merge ancestry beyond its already adapted umbrella representation.
- Stop if a production claim depends on an alias, partial lane, stale report, missing marker, mismatched HEAD, or mismatched artifact hash.
- Rollback means retain the disposable branch/worktree and logs as blocked evidence, or abandon it only with explicit approval. Never reset or clean the original master checkout.

## 5. Plan self-review

- **Live identity:** target and umbrella SHAs, divergence, dirty paths, hashes, and predicted conflicts are explicit.
- **Executable ordering:** one pending merge, one merge commit, then HEAD-sensitive gates.
- **Dirty-work preservation:** after the plan-only commit, only the four current generated assets remain dirty; they are copied and hash-checked outside the repository and never imported.
- **Coverage:** prepared seeds, source compiler, frame catch-up/UI contract, Phase 4 acceptance, device-counter injection, surface/whitewater, gate truth, policy-report authority, and documentation each have a task and check.
- **Experiment exclusion:** all four archived experiments remain outside the merge regardless of cleanliness or unique commits.
- **No placeholders:** every stop condition, command, marker, and handoff artifact is named.
