# I01 Wave-1 integration conflict and decision log

Integration branch: `feature/resolution-quality-standardization`
Base: `53564ac3d8a3ea44a1f867b02425f323800a8751`

## Integrated reviewed commits (task order)

1. T01: `e2fb9ff8034dd785c7d3521b52e151aa7b061e70`, `2a55a625e8144e35e169937d6043616ebd83ad99`
2. T02: `7ffdb5c80ea62ddf79c66e20c44013a8ce19bf25`, `06f0011c0649d7f57cb6a8bc4dde2340137f6e7e`, `1680c351bb9d49c83c31bec0e2b0119deb14714e`, `c9ebcd1843775421c3c4f075795a28c48b7cfbd9`, `e36d84da0a4b7238b31d8b886a9bfc26a8ed1d71`
3. T03: `0a15f766840416c3dc696c26a0c5b9d397364a5e`, `365db1f82547578c9dbe1aba1421b3cc0301eee5`

Task commits remain distinct cherry-picks; none were rewritten or squashed.

## Conflicts and decisions

Cherry-picking T02 `7ffdb5c` after T01 produced two content conflicts.

- `fluidSim_warp_blender51/blender/scene_bridge.py`, `Session._apply_scene_snapshot`:
  retained T02's transactional candidate arguments (`solver`, `commit_inputs`) and lifecycle helpers, while replacing T02's older sequential `set_sources` / append-`set_source_plans` calls with T01's atomic `solver.set_source_batch(snapshot.descriptors, snapshot.extra_plans)`. This preserves both the T02 prepare-before-commit session transaction and T01's single-owner, complete-source-snapshot contract.
- `tests/test_operator_panel_behavior.py`:
  retained the union of T01 direct-injection capacity/empty-batch tests and T02 scene-owned capacity suggestion/signature recheck tests. No assertions were removed or weakened.

All later task commits applied without textual conflicts. Post-integration inspection confirmed:

- initial snapshot application and live refresh both use `set_source_batch`;
- capacity plans retain an explicit operation binding and include live-solver residency for owned replacement operations;
- solver SDF availability and whitewater diagnostic paths remain present.

No conflict required an unapproved semantic change. Archived experiments were not inspected or used. The original master worktree was not modified.

## Preserved timed-out T02 reviewer evidence

The timed-out independent reviewer run `477fcabc-aab0-450f-b895-c734429009c4` was not recreated or overwritten. Its original probe, Windows-path-only adaptation, session/events/logs, hashes, stdout, and result are preserved under:

`C:/Users/Tyler/AppData/Local/Temp/fluidsim-resolution-quality-orchestration/recovered/t02-final-review-timeout/`

`probe-result.md` records exit 0 and `T02-PROBE-PASS`. The recovered probe demonstrated:

- blocked and refreshed `REINITIALIZE` used authored capacity 1024 plus identical live residency 310,095,324 bytes;
- the scene-owned signature remained stable with `operation=REINITIALIZE`;
- `preserve_simulation` recheck finished and applied `max_particles=40000`;
- `RESET` refresh used runtime capacity 65,536 with the same residency;
- first initialization used authored capacity 1024 with zero residency;
- cross-scene, stale-operation, authored-change, invalidation-clear, and operation-routing guards passed.

This evidence is preserved as reviewer evidence only; no T02 implementation was recreated during I01.

## Preserved timed-out IR01 reviewer evidence and accepted fix

The timed-out cross-lane reviewer run `ccd6825c-ea8a-4e79-b6c5-b1c2fde74c19` was also preserved rather than restarted. Its original 9,957-byte probe, session/events/logs, and hashes are stored under:

`C:/Users/Tyler/AppData/Local/Temp/fluidsim-resolution-quality-orchestration/recovered/ir01-timeout/`

The recovered probe initially reported **1 failed, 5 passed** and exposed a real cross-scene peak-memory defect: scene-B initialization passed zero transaction residency while scene A's solver remained allocated until candidate commit. The accepted shared-seam fix makes `Session.initialize()` account for any existing solver, not only a solver owned by the target scene; genuine first initialization still passes zero. The checked-in regression `test_cross_scene_initialize_charges_existing_solver_residency` preserves that finding.

Post-fix evidence:

- recovered IR01 probe plus session transaction tests: **23 passed**;
- full Python suite: **514 passed**;
- real Blender/CUDA capacity acceptance: `CAPACITY-PLANNER-ACCEPTANCE-PASS`.

## Validation performed on the integrated tree

- Focused union Python lane: 214 passed.
- Full Python suite: 513 passed.
- Blender 5.1 / CUDA (`cuda:0`, RTX 4080 SUPER):
  - `MIXED-SOURCE-BATCH-ACCEPTANCE-PASS`
  - `CAPACITY-PLANNER-ACCEPTANCE-PASS`
  - `MEMORY-SAFETY-GATE-PASS`
  - `WHITEWATER-TRUTH-ACCEPTANCE-PASS`
  - `SOURCE-LIFECYCLE-ACCEPTANCE-PASS`
  - `SOURCE-ACCEPTANCE-REPORT-PASS`
  - `CACHE-EXPORT-PIPELINE-PASS`
- Default production gate: `HEADLESS-GATE-PASS`; GUI acceptance was not included.

The production gate rebuilt tracked demo outputs as part of validation; those generated changes were restored afterward so the integration remains limited to reviewed task changes plus this decision log.
