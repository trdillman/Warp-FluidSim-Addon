# Release 3.0 status

The task record for executing `docs/RELEASE_3_0_PLAN.md`. The orchestrator updates it at
each dispatch, merge and gate. Newest state wins; history lives in git.

## Bases and baselines

| Ref | Commit | Suite | Demo artifact sha256 |
|---|---|---|---|
| master | `cc36403` | 2079 passed, 3 skipped | `2cb4f781…3a73` (GUI gate reference) |
| release/3.0 base: master + merge of release/2.1.0 | `94c79eb` | 2104 passed, 3 skipped | `2cb4f781…3a73` (headless, equals master) |
| `archive/st-flip-final` tag (local) | `94c79eb` | | |
| release/3.0 after lane D merge | `6ce6b33` | 2053 passed, 3 skipped (on D head `34528aa`; merge adds docs only) | `2cb4f781…3a73` (headless, equals master) |

- Demo artifact = `artifacts/demo_analysis/analysis_report.json`, the file the production
  gate hashes. The orchestrator recomputes it headless after every merge that touches
  `sim/` or `properties.py`.
- Refactor merges must keep it identical.
- The headless recompute is `C:\Users\Tyler\wt\demo_hash.ps1` in the throwaway worktree
  `C:\Users\Tyler\wt\r30-hash`. On `94c79eb` it reproduced the GUI gate reference exactly,
  so the headless hash stands in for the gate between milestones.

## Deviations from the plan

1. **The branch base is not a shipped 2.1.1.**
   - The plan branches `release/3.0` off master once 2.1.1 ships.
   - Shipping is an external action, and the owner's 2.1.1 version bump is still
     uncommitted in the main checkout.
   - So `release/3.0` is master plus a merge of `release/2.1.0` at `cd39e78`, which
     carries the plan, the Demo Gallery and the kernel prewarm.
   - The 2.1.1 bump will merge in when it is committed.
2. **Lane worktrees are created by hand** from `release/3.0` under `C:\Users\Tyler\wt\`,
   not with the Agent tool's worktree isolation.
   - That isolation would base each lane on the main checkout's branch, `release/2.1.0`.
3. **Wave 0 runs in two steps.**
   - D (ST-FLIP removal) shares `properties.py`, `pending_settings.py`, `ui_scene.py`,
     `solver_settings.py` and `pressure.py` with A and C.
   - So D, B and E go first. A and C start once D is merged.
4. **Lane E's Linux offscreen check** goes into
   `tests/acceptance/blender_gui_acceptance.py` and a new helper module, not
   `tools/run_production_gate.py`.
   - That script is what the gate runs for the viewport check.
5. **Lane E takes REVIEW-024 and REVIEW-025.** The plan's lane table gives them to E;
   one bullet suggests doing them with the split instead. E lands before A starts, so
   the split rebases on top.
6. **`SPEC-settings-split-completion` is partly stale.** Checked on `94c79eb`:
   - `FLUIDSIM_PG_scene` (76 fields) and `FLUIDSIM_PG_domain` (150) already share no
     field. 2.0 (`625b937`) narrowed the scene group and dropped the pre-1.1 migration.
   - So Phase 4 step 2 is done, and step 1 (a 3.0 migration operator) has nothing to
     migrate. Lane A does not build it.
   - Physics-tab domain panels are generated from the Scene-tab panel classes, so
     "delete the Scene panels" becomes "stop registering them on the Scene tab".
   - Phase 5 has one remaining wrong-group read, in an acceptance script.
7. **`SPEC-multigrid-pressure`'s hook point is stale.** Production PCG fuses Jacobi into
   three update kernels; `k_apply_preconditioner` was used only by the ST-FLIP solver.
   Lane C's brief says so. The preconditioner setting (RNA, `SolverSettings`, migration)
   is wired by the orchestrator after A and C merge, not by C.

## Lanes

| Lane | Branch | Worktree | Head | Tests | Blocker | Next action |
|---|---|---|---|---|---|---|
| D st-flip-removal | `3.0/lane-d-stflip-removal` | `C:\Users\Tyler\wt\r30-d` | `34528aa` | 2053 passed, 3 skipped | | **merged** as `6ce6b33`; demo hash unchanged |
| B cache-v3 | `3.0/lane-b-cache-v3` | `C:\Users\Tyler\wt\r30-b` | base `94c79eb` | | | dispatched; rebases onto D at merge |
| E hygiene | `3.0/lane-e-hygiene` | `C:\Users\Tyler\wt\r30-e` | base `94c79eb` | | | dispatched; rebases onto D at merge |
| A split + defaults | `3.0/lane-a-split-defaults` | `C:\Users\Tyler\wt\r30-a` | base `6ce6b33` | | | dispatched |
| C multigrid | `3.0/lane-c-multigrid` | `C:\Users\Tyler\wt\r30-c` | base `6ce6b33` | | | dispatched, stop 1 (tests); orchestrator reviews before kernels |

### Open items from lane D

- Decide what the `TIME_OFFSET` color mode shows now that ST-FLIP is gone
  (`output_publication.py:840`).
- Run the Blender scripts D edited but could not run headless.
- Verify the count quoted at `technical_report.md:481`.
- Regenerate the ownership map after all Wave 0 merges and re-check
  `EXPECTED_FIELD_COUNT = 212`.
- CHANGELOG entry for the ST-FLIP removal (written by the orchestrator at alpha.1).

Lane briefs are under `C:\Users\Tyler\wt\r30_briefs\`, outside the repository. Each is
`docs/specs/3.0/WORKER_BRIEF_TEMPLATE.md` filled in, with a shared environment section.

## Milestones

| Milestone | Contents | Gate | State |
|---|---|---|---|
| alpha.1 | Wave 0 (A–E) | 3× GUI gate, ABBA perf | in progress |
| alpha.2 | Wave 1 (F–I) | | not started |
| beta | Wave 2 (J, K), multigrid re-baseline | | not started |
