# Phase-4 Parallel Worktree Merge-Readiness Investigation (READ-ONLY)

- Repo: `liquid_warp_blender51_port` (common git dir shared by 14 worktrees)
- `master` tip: `0df90d4` — 14 commits after `695fad9`
- Investigation date basis: worktree state as observed; **nothing was modified, staged, checked out, merged, or rebased**
- Method: `git worktree list`, per-worktree `status --porcelain`, `rev-list --left-right --count`, `merge-tree --write-tree`, and offline 3-way `git merge-file --diff3` simulations into `$TMPDIR` (CRLF-normalized, since this repo has mixed line endings and raw diffs otherwise report whole-file churn).

> **Post-investigation update:** The empty landing branch described below was subsequently populated and committed as `phase4/live-interaction-merge-20260728`. Master commits `239f80a` and `af4136e` were merged into it, all integration gates passed, and the branch finished 0 commits behind master.
>
> **Classification correction:** `…_port_native_interop_main_20260715`, `…_port_native_interop`, `.claude/worktrees/particle-surface-rendering-apis-f62356`, and the later-created `.claude/worktrees/stflip-implementation` are archived **branched experiments, not genuine worktree candidates**. They must be excluded from merge-readiness recommendations. Conflict data below is retained only as historical forensic evidence and must not be interpreted as an integration plan. The ST-FLIP experiment was not present in this original snapshot and must be ignored in future refreshes.

---

## 1. Headline findings

1. **Only two branches in the entire worktree set have any commit that master does not already have.** `git merge-tree --write-tree master <branch>` returns the *identical* tree `5bcef09…` (i.e. a no-op merge) for `orbit-gpu-pcg-conditional-loop`, `codex/deepen-four-modules`, `codex/fix-cone-pyramid-distances`, `codex/deepen-fluidsim-architecture-seams`, `codex/integrate-master-architecture-20260713`, and `codex/native-gpu-interop-main-20260715`. Their only remaining value is **uncommitted working-tree state**.
2. **The Phase-4 payload is not committed anywhere.** It exists solely as uncommitted working-tree state in `…/liquid_warp_blender51_phase4_interaction` on `phase4/live-property-interaction @ 695fad9` (14 behind master, 130 dirty entries, **0 staged**). The intended landing branch `phase4/live-interaction-merge-20260728` is clean and already equals master (`0df90d4`) — it is an *empty* integration branch.
3. **Master already absorbed a large slice of the Phase-4 tree** via `246fa90 "chore: track test suite, demo evidence, and backlog cards"`. 81 of the Phase-4 worktree's untracked files are byte-identical (modulo CRLF) to master's tracked versions. What remains genuinely un-merged is the *source* rewrite plus 11 divergent test files and 37 new-only files.
4. **`master`'s own worktree is dirty with un-committed, Phase-4-adjacent solver-session work** (`ensure_session_before_frames` / `_on_playback_pre` persistent handlers). This is the single highest data-loss risk in the whole set and must be committed before any Phase-4 merge.
5. Real cross-worktree collision surface is much smaller than the dirty-file counts suggest: most "modified" entries in the codex worktrees are **CRLF-only noise** (local delta = 0 lines after normalization).

---

## 2. Per-worktree state

| # | Worktree | Branch / HEAD | vs master (ahead/behind) | Working tree | Unique commits | Merge-soon? |
|---|---|---|---|---|---|---|
| 1 | `…/liquid_warp_blender51_port` (main) | `master` `0df90d4` | 0 / 0 | **dirty**: 10 (incl. `scene_bridge.py +55`, `render.py +7`, `test_scene_bridge_cache_playback.py +111`, `blender_handler_persistence_gate.py +1`, `demos/*.blend`), 0 staged, `?? .agents/` | — | **Commit first** (§4 step 0) |
| 2 | `…/liquid_warp_blender51_phase4_interaction` | `phase4/live-property-interaction` `695fad9` | 0 / 14 | **dirty**: 130 entries (32 tracked-modified, 98 untracked), **0 staged** | none (payload uncommitted) | **Primary integration target** |
| 3 | `…/liquid_warp_blender51_phase4_commit` | `phase4/live-interaction-merge-20260728` `0df90d4` | 0 / 0 | clean | none — equals master | Empty landing branch; ready to receive |
| 4 | `…_port_native_interop_main_20260715` | `codex/native-gpu-interop-main-20260715` `c182d3a` | 0 / 41 | **20 STAGED + 24 unstaged + 7 untracked** | none committed | **EXPERIMENT — NOT A MERGE CANDIDATE** |
| 5 | `…_port_native_interop` | `codex/native-gpu-interop` `3135f8b` | 1 / 68 | dirty 55 | `3135f8b feat(viewport): add native CUDA Vulkan particle interop` | **EXPERIMENT — NOT A MERGE CANDIDATE** |
| 6 | `.codex/worktrees/orbit-gpu-pcg-conditional-loop-addon` | `orbit-gpu-pcg-conditional-loop` `c182d3a` | 0 / 41 | dirty 14 — only real delta is `sim/pressure.py +336` and new `sim/cuda_reduction.py`; the other 12 are CRLF-only | none | Small, extractable |
| 7 | `.codex/worktrees/orbit-gpu-pcg-baseline` | detached `c182d3a` | 0 / 41 | clean | none | Prune |
| 8 | `.codex/worktrees/34f4/…` | `codex/fix-cone-pyramid-distances` `01a5437` | 0 / 29 | dirty 27 — only real delta is `sim/sourcing.py +85` + 9 new policy-drift backlog cards; rest CRLF-only | none | Small, extractable |
| 9 | `.codex/worktrees/54d6/…` | `codex/deepen-four-modules` `83cb49a` | 0 / 33 | dirty 3 (`demos/*.blend`, `meta.json` only) | none | Prune |
| 10 | `.codex/worktrees/arch-deepen/…` | `codex/deepen-fluidsim-architecture-seams` `57afa59` | 0 / 56 | clean | none | **Prune** — fully in master |
| 11 | `.codex/worktrees/e9bf/…` | detached `6bc8916` | 0 / 38 | clean | none | Prune |
| 12 | `.codex/worktrees/merge-architecture-main/…` | `codex/integrate-master-architecture-20260713` `e80ea01` | 0 / 43 | clean | none | **Prune** — architecture integration already in master |
| 13 | `…_port/.claude/worktrees/particle-surface-rendering-apis-f62356` | `claude/particle-surface-rendering-apis-f62356` `697fe6e` | 1 / 16 | dirty 5 (docs + `?? backlogs/`) | `697fe6e docs(policy-drift): re-audit the nested worktree` (docs only) | **EXPERIMENT — NOT A MERGE CANDIDATE** |
| 14 | `…_port_cherry_pick_merge` | detached `2635080` | 0 / 102 | dirty 4 (`D CLAUDE.md`, `D sim/__init__.py`, `?? review_2635080.blend`) | none | **Prune** — 102 behind, deletions look accidental |

Historical `git merge-tree` preview for the two branches with unique commits. Both results are forensic only; neither experiment is a merge candidate:

```
codex/native-gpu-interop            -> CONFLICT AGENTS.md
                                       CONFLICT (modify/delete) PROJECT_MANIFEST.md (deleted in master)
                                       CONFLICT fluidSim_warp_blender51/blender/scene_bridge.py
                                       CONFLICT fluidSim_warp_blender51/blender/viewport.py
                                       (panels.py, properties.py auto-merge clean)
claude/particle-surface-…-f62356    -> CONFLICT POLICY_DRIFT_REPORT.md
                                       CONFLICT demos/POLICY_DRIFT_REPORT.md
```

---

## 3. Phase-4 ↔ master conflict analysis (the core question)

### 3.1 Tracked-file conflicts, simulated 3-way (ours = Phase-4 worktree, base = `695fad9`, theirs = `master`)

| File | Conflict hunks | Phase-4 local delta (lines) | Master delta (lines) |
|---|---|---|---|
| `fluidSim_warp_blender51/blender/scene_bridge.py` | **14** | 778 | 277 |
| `fluidSim_warp_blender51/blender/viewport.py` | **8** | 335 | 524 |
| `fluidSim_warp_blender51/io/cache.py` | **4** | 97 | 95 |
| `fluidSim_warp_blender51/operators.py` | **3** | 270 | 269 |
| `tools/run_production_gate.py` | **3** | 89 | 86 |
| `fluidSim_warp_blender51/preferences.py` | **2** | 45 | 65 |
| `fluidSim_warp_blender51/properties.py` | 1 | 343 | 254 |
| `fluidSim_warp_blender51/sim/sourcing.py` | 1 | 956 | 975 |
| `tests/test_operator_panel_behavior.py` | 1 | 586 | 268 |
| `tests/test_frame_output.py` | 1 | 2 | 2 |
| `fluidSim_warp_blender51/panels.py` | 0 (textually) | 496 | 424 |
| `fluidSim_warp_blender51/docs/user_guide.md` | 0 | 174 | 128 |
| `fluidSim_warp_blender51/sim/solver.py` | 0 | 259 | 293 |
| `fluidSim_warp_blender51/sim/pressure.py` | 0 | 76 | 109 |
| `sim/memory_safety.py`, `sim/frame_output.py`, `warp_discovery.py` | 0 | 0 (untouched by Phase 4) | 39 / 16 / 29 |

### 3.2 Which of master's 9 flagged commits are *missing* from the Phase-4 tree

Marker probe over the Phase-4 working tree (`grep -rl` for each commit's distinctive new symbol):

| Commit | Marker | Present in Phase-4 tree? | Consequence |
|---|---|---|---|
| `3a497a7` re-entrancy guard | `_IN_FRAME_HANDLER`, `ensure_outputs` | **NO** | Largest contributor to the 14 `scene_bridge` + 8 `viewport` hunks. Phase-4 rewrote `handle_frame_change`; master's re-entrancy guard and ID-allocation split must be **re-applied on top**, not auto-merged. |
| `451f483` single device owner | `preferences.configured_device` | Partially — Phase-4 references `configured_device` in `scene_bridge.py`/`operators.py`/`test_operator_panel_behavior.py`, but `preferences.py` is at the pre-`451f483` shape | 2 hunks in `preferences.py`. Converge on **master's** single owner in `preferences.py`; drop any Phase-4-local duplicate lookup. |
| `241b981` dropped-frame catch-up | `unregister_handlers`, `timeline_frame` | Names present, semantics differ (Phase-4 has its own `timeline_frame` plumbing) | Semantic (not just textual) conflict inside the 14 `scene_bridge` hunks. Must preserve master's whole-gap stepping, `catchup` scene property, and handler-teardown release. `sim/memory_safety.py` (+39) is untouched by Phase 4 → take master verbatim. |
| `a2f089c` panel/poll guards | `_NO_POLL_REQUIRED`, `_requires_live_solver` | **NO** | 3 `operators.py` hunks. `panels.py` shows 0 textual hunks only because Phase-4's +496-line rewrite lands elsewhere in the file — master's `draw` guard must be re-applied **manually**. Master-only `tests/test_operator_poll_and_guards.py` imports operator symbols Phase-4 renamed → expect import/attr failures until re-pointed. |
| `895c938` strided readback | `_display_stride` | **NO** | 0 conflicts in `frame_output.py`/`warp_discovery.py` (Phase 4 untouched) but the `viewport.py` half lands inside the 8 conflicting hunks. |
| `f2cbd84` PCG zeroing / LRU | `_PCG_GRAPH_CAPACITY` | **NO** | `pressure.py` 0 conflicts → take master cleanly. |
| `81eb577` cache scene-key guard | `_CACHE_SCENE_KEY_MESSAGE`, `SCENE_KEY_VERSION` | **NO** | **Highest functional risk.** Phase-4 rewrote `io/cache.py` (+97) without the scene-key digest; 4 hunks. If master's guard is lost, cache playback silently accepts a cache baked with different settings. |
| `1ce25b4` sourcing emission faults | (test-only markers) | n/a | `sourcing.py` 1 hunk against Phase-4's +956; verify master's three `apply_compiled` fixes survive. |
| `0df90d4` point-cloud padding | `POINT_COUNT_TAG`, `_POINT_CAPACITY_HEADROOM` | **NO** | 8 `viewport.py` hunks. Phase-4's viewport rewrite must absorb master's capacity padding / realloc logic or live playback regresses to per-frame datablock swapping. |
| `e71bd9b` (also relevant) | `blender_manifest.toml` | **absent** from Phase-4 tree | `git diff master` shows it as a deletion. Do **not** let the merge drop the 5.1 extension manifest. |
| `a37ef0e` (also relevant) | CFL clamp reporting | absent | `solver.py`/`panels.py`/`properties.py` sides show 0–1 hunks; verify the CFL status string survives Phase-4's panels rewrite. |

### 3.3 Untracked-file hazard (this will hard-abort a naive merge)

The Phase-4 worktree is at `695fad9`, *before* `246fa90` tracked the test suite. So many files are **untracked locally but tracked on master**. Any `git merge master` / `git checkout master` from that worktree aborts with *"untracked working tree files would be overwritten"*.

- **81 untracked files are byte-identical to master's blobs** → harmless once staged or removed.
- **11 untracked files differ from master** (must be handled explicitly). All 11 merge **cleanly (0 conflicts)** in a 3-way with base `246fa90`:

| File | Phase-4-only delta | Master-only delta |
|---|---|---|
| `tests/test_scene_bridge_cache_playback.py` | 120 | 180 |
| `tests/test_cache_io.py` | 42 | 0 |
| `tests/blender_gui_acceptance.py` | 18 | 0 |
| `tests/blender_handler_persistence_gate.py` | 1 | 0 |
| `tests/blender_viewport_2frame_gate.py` | 0 | 136 |
| `tests/test_source_emission_lifecycle.py` | 0 | 62 |
| `tests/blender_cache_export_pipeline.py` | 0 | 23 |
| `tests/headless_stflip_timeline_viewport.py` | 0 | 13 |
| `tests/blender_gui_playback_probe.py` | 0 | 11 |
| `tests/test_pressure_projection_regressions.py` | 0 | 10 |
| `tests/test_memory_safety_recovery.py` | 0 | 5 |

- **37 new-only paths** (exist nowhere in master): `reviews/PHASE_4_LIVE_INTERACTION_PLAN_2026-07-27.md`, `reviews/PHASE_4_INITIAL_INDEPENDENT_REVIEW_2026-07-27.md`, `reviews/PHASE_4_INITIAL_REVIEW_FOLLOWUP_GATE_2026-07-27.md`, `reviews/PHASE_4_FINAL_INTERACTION_REVIEW_2026-07-27.md`, `tests/blender_phase4_live_interaction_acceptance.py`, `tests/test_phase4_pending_interaction.py`, plus 31 `.pi-subagents/artifacts/*` scratch files. The `.pi-subagents/` tree is agent scratch and should stay **untracked + gitignored**, not committed.

### 3.4 Phase-4 gate evidence (as claimed in the worktree)

`reviews/PHASE_4_FINAL_INTERACTION_REVIEW_2026-07-27.md` records a **PASS** at `git_head: 695fad96072e`, `git_dirty: True`: `python -m pytest -q` → 377 passed; `tools/run_production_gate.py` → HEADLESS-GATE-PASS; `--include-gui` → FULL-PRODUCTION-GATE-PASS; `PHASE4-LIVE-INTERACTION-PASS (14/14)`.
**Caveat:** that evidence was produced against a tree *without* master's 14 commits. It is **not** valid for the merged result. Every gate must be re-run after integration.

---

## 4. Recommended merge order

**Step 0 — De-risk `master`'s own dirty worktree (do this first).**
`…/liquid_warp_blender51_port` has uncommitted `scene_bridge.ensure_session_before_frames` + `_on_playback_pre` / `render._on_render_init` persistent handlers (+55/+7) with a matching `tests/test_scene_bridge_cache_playback.py` (+111) and a `blender_handler_persistence_gate.py` entry (+1). All 4 files merge into master with **0 conflicts** (they are ahead-of-master edits). This is live-session-ownership code that directly collides with the Phase-4 `scene_bridge` rewrite. Commit it to master now, or it will be silently clobbered. Also run `blender --background --factory-startup --python tests/blender_handler_persistence_gate.py` per AGENTS.md, since a new handler list entry was added.

**Step 1 — Commit Phase 4 on its own branch.** In `…_phase4_interaction`, stage the 32 tracked modifications + the 11 divergent untracked files + the 37 new-only files *except* `.pi-subagents/`, and commit on `phase4/live-property-interaction`. Do **not** attempt `git merge master` before this commit — it will abort on the untracked-overwrite hazard (§3.3).

**Step 2 — Merge `master` into `phase4/live-property-interaction`** (merge master → feature, not the reverse), resolving §5. Re-run the full gate.

**Step 3 — Fast-forward `phase4/live-interaction-merge-20260728` to the resolved Phase-4 tip, then merge into `master`.** That branch is currently identical to master and is the natural landing branch.

**Step 4 — `codex/fix-cone-pyramid-distances` worktree**: cherry-pick the isolated `sim/sourcing.py +85` (2 conflicts vs master's `1ce25b4`) and the 9 new policy-drift backlog cards. Everything else there is CRLF noise.

**Step 5 — Pressure/PCG work (dedupe first)**: `sim/pressure.py +336` + new `sim/cuda_reduction.py` also appears in the archived native-interop experiment, but that does not authorize sourcing it there. If the separate `orbit-gpu-pcg-conditional-loop-addon` work is independently approved, evaluate and land it from that non-interop worktree only.

**Step 6 — Exclude native GPU interop.** Both `…_native_interop_main_20260715` and `codex/native-gpu-interop` are archived branched experiments, not integration candidates. Do not rebase, merge, cherry-pick, or extract from them without an explicit user request to reactivate the experiment under a new implementation and validation plan. Historical overlap in `viewport.py`, `scene_bridge.py`, and `pressure.py` is collision evidence only—not future work.

**Step 7 — Exclude rendering APIs.** `claude/particle-surface-rendering-apis-f62356` is an archived branched experiment, not a docs-only merge candidate. Do not merge its commit, policy reports, or backlog files without an explicit user request to reactivate the experiment. It is also a nested checkout inside master at `.claude/worktrees/…`, so recursive tooling must avoid traversing it.

**Step 8 — Prune** `arch-deepen`, `merge-architecture-main`, `54d6`, `e9bf`, `orbit-gpu-pcg-baseline`, `cherry_pick_merge` (all fully contained in master, 0 unique commits, 29–102 behind).

---

## 5. Concrete conflict-resolution requirements

**`blender/scene_bridge.py` (14 hunks — hardest file).** Resolution is a re-application, not a text merge:
1. Keep master's `_IN_FRAME_HANDLER` re-entrancy guard (`3a497a7`) wrapping Phase-4's rewritten `handle_frame_change`, including the reset-on-exception path.
2. Keep master's `ensure_outputs` / ID-allocation split so no datablock is created inside `frame_change_post`.
3. Keep master's whole-gap catch-up loop and `catchup` scene property (`241b981`), reconciled with Phase-4's `timeline_frame` argument on `Session.step(..., write_cache=, timeline_frame=)`.
4. Keep master's `_remove_handlers` / `unregister_handlers` teardown release.
5. Keep master's `_check_cache_scene_key` call site (`81eb577`).
6. Keep master's device lookup routed through `preferences.configured_device` (`451f483`) — exactly one owner.
7. Re-attach master-worktree step-0 work (`ensure_session_before_frames`, `_on_playback_pre`) after the Phase-4 rewrite; it must remain `@persistent` and registered in every handler list tracked by `tests/blender_handler_persistence_gate.py`.

**`blender/viewport.py` (8 hunks).** Phase-4's viewport rewrite must retain master's `POINT_COUNT_TAG` / `_POINT_CAPACITY_HEADROOM` / `_POINT_CAPACITY_PAD` / `_needs_realloc` padding path (`0df90d4`) *and* `_display_stride` device-side readback striding (`895c938`). Validate with `blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py` (master's version, +136 lines, is authoritative) and `tests/test_scene_bridge_cache_playback.py`.

**`io/cache.py` (4 hunks).** Non-negotiable: keep `SCENE_KEY_VERSION`, `_SCENE_KEY_IGNORED_SETTINGS`, `_digest_rna`, `scene_key()` from `81eb577`. Then reconcile with Phase-4's `+97`. Take **master's** `tests/blender_cache_export_pipeline.py` (+23) and merge Phase-4's `tests/test_cache_io.py` (+42) on top.

**`operators.py` (3 hunks) + `panels.py`.** Re-apply master's `_NO_POLL_REQUIRED` / `_KNOWN_SESSION_OPERATORS` / `_requires_live_solver` poll registry and the defensive domain-panel `draw` guard (`a2f089c`) onto Phase-4's rewritten operator/panel set, and extend the registry with every operator Phase 4 adds. Master-only `tests/test_operator_poll_and_guards.py` must be re-pointed at any renamed operator IDs.

**`preferences.py` (2 hunks).** Take master's `configured_device()` wholesale; delete Phase-4's local device lookup. `tests/test_device_preference_routing.py` (master-only) asserts *nothing else* reaches into add-on preferences — Phase-4 code in `scene_bridge.py`/`operators.py` must satisfy that.

**`properties.py` (1) / `solver.py` (0) / `panels.py`.** Verify master's `241b981` catch-up property, `a37ef0e` CFL-clamp reporting property + panel row, and `a2f089c` `sim/fields.py` guard all survive Phase-4's +343/+259/+496 rewrites — textual absence of conflict markers here is *not* evidence of semantic survival.

**Take-master-verbatim (0 conflicts, Phase 4 untouched):** `sim/memory_safety.py`, `sim/frame_output.py`, `warp_discovery.py`, `sim/pressure.py`, `fluidSim_warp_blender51/blender_manifest.toml`, `tests/test_frame_change_catchup.py`, `tests/test_readback_stride_and_discovery.py`, `tests/test_pressure_st_and_graph_cache.py`, `tests/test_cfl_clamp_reporting.py`, `tests/test_device_preference_routing.py`, `tests/test_operator_poll_and_guards.py`.

**Tests/gates.** `tests/test_scene_bridge_cache_playback.py` needs a genuine 3-way merge (Phase-4 +120 / master +180, 0 textual conflicts but both sides add live-playback cases). Phase-4-only `tests/test_phase4_pending_interaction.py` and `tests/blender_phase4_live_interaction_acceptance.py` must be wired into `tools/run_production_gate.py` — that tool has **3 conflict hunks** (Phase-4 +89 vs master +86) and both sides' lane registrations must survive.

**Post-merge validation (per AGENTS.md, all must be re-run — the recorded Phase-4 PASS predates master's 14 commits):**
`python -m pytest -q` · `python tools/run_production_gate.py --include-gui` · `blender --background --factory-startup --python tests/blender_handler_persistence_gate.py` · `… tests/blender_viewport_2frame_gate.py` · `… tests/blender_cache_export_pipeline.py` · `… tests/blender_memory_safety_gate.py` · `blender --factory-startup --python tests/blender_gui_acceptance.py` · `python tools/validate_gui_playback_report.py artifacts/gui_playback_probe.json`.

---

## 6. Additional notes / risks

- **Hygiene:** `.agents/` (main worktree), `.pi-subagents/` (Phase-4 worktree), `dist/` + `docs/evidence/native_interop/*` (interop worktree), `review_2635080.blend` (cherry-pick worktree), `backlogs/swarm-review-*.md` (nested claude worktree) are untracked agent/scratch/evidence output. They should be covered by `.gitignore` and left untracked — not committed with Phase 4.
- **CRLF trap:** the repo has mixed line endings. Raw `diff`/`git merge-file` without `tr -d '\r'` reports whole-file conflicts on nearly every file. Any scripted merge tooling must normalize, and resolution must preserve each file's existing endings.
- **Blend-file churn:** `demos/*.blend` and `*_preview.png` are modified in 5 different worktrees with no textual merge possible. Decide a single authoritative demo-rebake worktree (Phase-4's, which also carries the `demos/*/meta.json` evidence) and take theirs wholesale; then re-run `blender --background --python tools/check_demo_blends.py` and `python tools/analyze_demos.py` (expect `DEMO-ANALYSIS-PASS (18/18)`).
- **`codex/native-gpu-interop-main-20260715` has 20 staged files**, but the checkout is an archived branched experiment. Preserve it as experimental evidence; do not treat its index as queued integration work.
- **`…_port_cherry_pick_merge`** carries `D CLAUDE.md` and `D fluidSim_warp_blender51/sim/__init__.py` in a 102-behind detached checkout. Those deletions must never be merged forward.
- I did not run any test lane; all conflict figures come from offline 3-way simulations in a temp dir. Actual `git merge` output may differ slightly in hunk granularity but the missing-marker findings in §3.2 are direct content evidence and are not merge-strategy dependent.

---

## Review

- **Correct:** Master's 14 post-`695fad9` commits are internally coherent and all nine flagged fixes are self-contained with dedicated tests (`test_frame_change_catchup.py`, `test_device_preference_routing.py`, `test_operator_poll_and_guards.py`, `test_readback_stride_and_discovery.py`, `test_pressure_st_and_graph_cache.py`, `test_cfl_clamp_reporting.py`). Nine of the fourteen worktrees are fully contained in master and carry zero merge risk.
- **Blocker:** Phase-4 is uncommitted and its worktree is 14 commits behind; **none** of `3a497a7`, `a2f089c`, `895c938`, `f2cbd84`, `81eb577`, `0df90d4` are present in the Phase-4 tree. A naive `git merge master` from that worktree aborts on 11 untracked-overwrite collisions before it even reaches the 14 `scene_bridge` / 8 `viewport` / 4 `io/cache` content conflicts. Merging in the wrong direction (Phase-4 tree wins) silently reverts the cache scene-key guard, the handler re-entrancy guard, and the point-cloud padding fix.
- **Blocker:** `master`'s own worktree holds uncommitted `ensure_session_before_frames` / `_on_playback_pre` session-ownership code (`scene_bridge.py +55`, `render.py +7`, plus tests) that overlaps Phase-4's `scene_bridge` rewrite. Commit it before step 1 or it is lost.
- **Note:** The Phase-4 gate PASS (377 tests, FULL-PRODUCTION-GATE-PASS) was produced at `695fad9` + dirty tree and does not certify the merged result.
- **Note:** `sim/pressure.py +336` and `sim/cuda_reduction.py` are duplicated verbatim across two worktrees — land once, not twice.
