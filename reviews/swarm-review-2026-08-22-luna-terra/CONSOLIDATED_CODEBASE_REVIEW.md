# Consolidated Codebase Review Report

**Target**: `fluidSim_warp_blender51` (35,763 source/config LOC, 78 files)
**Supporting surface**: `tests` (21,296 Python LOC, 113 files) and `tools` (4,845 Python LOC, 14 files)
**Working tree**: `codex/fluidsim-1.0.248-merge-prep` at `0d395d78582af1f8876539f0f927110c3eff872d`, including pre-existing modified and untracked work
**Date**: 2026-08-22
**Mode**: Read-only seven-lane swarm review; Trust & Safety/security lane intentionally skipped
**Routing**: Luna-high for Architecture & Style, Type System & Contracts, UI/UX & Operators, and Test Coverage; Terra-high for Execution Engine, Particle Pipeline, and Serialization & IO
**Concurrency**: Maximum six subagents; the Test Coverage lane launched only after Serialization & IO completed

## Verdict

The review found **18 actionable issues: 0 critical, 4 high, 10 medium, and 4 low**. The full system-Python regression suite remains green at **655 passed in 28.02 seconds**, so these are gaps and activation-specific defects not exercised by that suite. No add-on source was changed.

The highest-risk cluster is multi-scene ownership: a non-owning scene can release, mutate, export, or disturb the singleton live solver. The second cluster is persistence integrity: cache utilities can relocate an active production bake, and frame export publishes directly to final files without a bundle transaction.

## Critical Findings (0)

None. `UI-001` arrived as critical but was recalibrated to high: state loss is certain after its multi-scene activation condition, but that condition is not generally certain.

## High Findings (4)

| ID | Location | Dimension | Finding | Impact | Recommended fix |
|---|---|---|---|---|---|
| ARCH-001 | `fluidSim_warp_blender51/blender/scene_bridge.py:181` | Architecture & Style | `Session.release(scene)` clears the singleton runtime without proving that `scene` owns it. Shared callers include playback, operators, production-bake restoration, and teardown. | A non-owner scene can destroy another scene's in-memory solver and GPU state. | Make scene-scoped release owner-checked; add a separate explicit forced/global teardown API and two-scene tests. Cross-reference UI-001 and the Release facet of UI-002. |
| IO-001 | `fluidSim_warp_blender51/blender/cache_manager.py:171` | Serialization & IO | Free, Move, and Rename can mutate the configured cache while `ProductionBakeBackend` retains and continues writing its original `base_dir`; the UI/operators have no active-bake exclusion. | Committed frames can be deleted or moved, after which later ticks recreate a split, non-resumable cache. | Hold a durable per-cache bake lease checked by every mutator and disable/refuse cache mutations until bake finish/cancel/failure cleanup. |
| UI-001 | `fluidSim_warp_blender51/blender/playback.py:173` | UI/UX & Operators | Enabling cache playback unconditionally releases the shared session even when another scene owns it. | Switching scene B to cache playback discards scene A's live simulation. | Reject the transition when another scene owns the session, restore the RNA flag, and release only an owned session. Cross-reference ARCH-001. |
| UI-002 | `fluidSim_warp_blender51/operators.py:100` | UI/UX & Operators | The shared session-operator poll checks only runtime/bake/activity, not `owns_scene(context.scene)`. Release, Snippet Apply, and Export Frame can therefore act on another scene's solver; Reset is the contrasting guarded path. | The UI can release or mutate the wrong solver, or export one scene's state under another scene's action. | Require owner-scene checks in poll and execute for non-switching operators, with explicit poll messages and cross-scene tests. The Release facet cross-references ARCH-001. |

## Medium Findings (10)

| ID | Location | Dimension | Finding | Recommended fix |
|---|---|---|---|---|
| ARCH-002 | `fluidSim_warp_blender51/io/cache.py:105` | Architecture & Style | Low-level `io.cache` lazily imports Blender solver-settings and scene-input adapters for cache identity, reversing the required adapters-to-`sim`/`io` dependency direction. | Move scene identity collection to a Blender-owned adapter and pass plain records/signatures into `io.cache`. |
| TYPE-001 | `fluidSim_warp_blender51/sim/source_regions.py:67` | Type System & Contracts | `_owned_array` flattens arbitrary leading dimensions; a `(1, 3, 2)` point input is silently accepted as `(2, 3)`. | Require exact rank/tail shapes before dtype conversion and ownership; add malformed-rank tests for point and mesh plans. |
| TYPE-002 | `fluidSim_warp_blender51/properties.py:316` | Type System & Contracts | The synthesized `FieldPermission` contract is absent from Blender snippet arguments and the runtime validator/code generator, making every field effectively mutable. | Persist permission through the adapter and enforce read/write constraints during validation/code generation. |
| TYPE-003 | `fluidSim_warp_blender51/sim/custom.py:793` | Type System & Contracts | The synthesized FUNCTION snippet variant is not represented or dispatched; all snippets compile as kernels and function definitions are rejected later. | Implement the typed function path or model it as explicitly unsupported and fail closed at adapter conversion with a durable diagnostic. |
| IO-002 | `fluidSim_warp_blender51/io/export.py:25` | Serialization & IO | Surface/particle/whitewater files are written sequentially and directly to final paths. A late failure can truncate prior files or leave a mixed snapshot while the operator returns cancelled. | Validate first, stage all artifacts in same-directory temporaries, flush/fsync, and publish the complete bundle only after every write succeeds. |
| UI-003 | `fluidSim_warp_blender51/operators.py:101` | UI/UX & Operators | Release inherits runtime probing and is disabled when runtime/package/device probing fails even though releasing the existing session needs no new runtime selection. | Give Release an owned-session/bake-idle poll independent of runtime readiness and expose it as a recovery action. |
| UI-004 | `fluidSim_warp_blender51/blender/playback.py:286` | UI/UX & Operators | Cache playback writes its scene frame into singleton `sess.last_frame`, contaminating the live owner scene's duplicate/jump logic. | Store cache playback cursors per scene and reserve `sess.last_frame` for the owning live scene. |
| TEST-001 | `fluidSim_warp_blender51/io/cache.py:522` | Test Coverage | `write_sequence_metadata` normalizes mapping keys in a dict comprehension before duplicate detection; `{1: ..., "1": ...}` silently keeps the latter record. | Use one collision-detecting normalization helper in the production path and test mixed key types through `write_sequence_metadata`. |
| TEST-002 | `fluidSim_warp_blender51/diagnostics/benchmark.py:146` | Test Coverage | Malformed numeric benchmark fields raise `TypeError`/`ValueError`; the operator only handles `FileNotFoundError`. | Validate numeric fields into a structured fail-closed report and add null/string/corrupt-baseline tests. |
| TEST-003 | `fluidSim_warp_blender51/io/export.py:47` | Test Coverage | PLY export accepts NaN/Infinity and writes a successful file; current tests cover shape only. | Reject non-finite positions/velocities before opening the output and test that rejection leaves no file. |

## Low Findings (4)

| ID | Location | Dimension | Finding | Recommended fix |
|---|---|---|---|---|
| UI-005 | `fluidSim_warp_blender51/panels.py:54` | UI/UX & Operators | Reset-tier changes are labelled as applying after restart although the distinct action is Reset Now. | Render reset/reinitialize-specific wording and assert both presentations. |
| UI-006 | `fluidSim_warp_blender51/properties.py:110` | UI/UX & Operators | Turning off both surface and particle visibility leaves `liquid_view_mode` displaying its previous value. | Add an explicit hidden state or prevent the contradictory toggle combination; test save/reload presentation. |
| UI-007 | `fluidSim_warp_blender51/operators.py:996` | UI/UX & Operators | Empty snippet-stack remove/add-argument actions return `FINISHED` without mutation or feedback. | Poll-disable invalid actions or return warning/cancelled and add empty-stack operator tests. |
| TEST-004 | `tests/test_addon_core_regressions.py:77` | Test Coverage | The duplicate-registration test removes its registry entry but leaves its generated `SOURCE_ALIASES` entry, contaminating later tests in the same process. | Snapshot/restore both registries or remove the alias in `finally`, then assert clean teardown. |

## Summary by Lane

| Dimension | Model | Files reviewed | Critical | High | Medium | Low | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| Architecture & Style | Luna high | 25 | 0 | 1 | 1 | 0 | 2 |
| Execution Engine | Terra high | 10 | 0 | 0 | 0 | 0 | 0 |
| Type System & Contracts | Luna high | 10 | 0 | 0 | 3 | 0 | 3 |
| Particle Pipeline | Terra high | 13 | 0 | 0 | 0 | 0 | 0 |
| Serialization & IO | Terra high | 14 | 0 | 1 | 1 | 0 | 2 |
| UI/UX & Operators | Luna high | 20 | 0 | 2 | 2 | 3 | 7 |
| Test Coverage | Luna high | 20 | 0 | 0 | 3 | 1 | 4 |
| **Total** |  |  | **0** | **4** | **10** | **4** | **18** |

## Validation and Deduplication

- All seven lane JSON files parsed against the required schema; every retained file and line exists in the live checkout after correcting one artifact-only path typo.
- `ARCH-001`, `UI-001`, and the Release facet of `UI-002` share the same singleton ownership root. They remain separate, cross-referenced findings because they occur at different production locations and the swarm deduplication contract requires same-issue/different-location findings to be retained.
- Main-agent source tracing checked the deepest owner, direct production callers, activation conditions, and contrasting paths for every finding.
- Pure-Python probes directly reproduced: malformed source arrays reshaped to a different cardinality; duplicate-normalized metadata keys silently overwrote; malformed benchmark values raised; NaN PLY data wrote successfully; and registry-only cleanup left its generated alias behind.
- Full regression suite: `python -m pytest -q` -> **655 passed in 28.02s**.
- Reviewer-focused checks also passed: Architecture 72, Execution 73, Type/Contracts 54, Particle 99, and Test Coverage 122 tests. These sets overlap and are not summed.
- No Blender GUI/CUDA production gate was run. This was a read-only code review, and the default production gate can refresh runtime artifacts; the retained findings are source/reproduction-backed rather than claims of GUI runtime acceptance.

## Top Priority Actions

1. Establish one explicit scene-ownership policy across session release, cache playback, operators, and playback cursors; add two-scene Blender regressions.
2. Add exclusive production-cache ownership and transactional multi-artifact export publication.
3. Restore synthesized custom-kernel permissions and FUNCTION variant fidelity, or reject unsupported variants at the adapter boundary.
4. Fail closed at input/file boundaries: source array rank, benchmark numeric fields, non-finite export data, and normalized metadata keys.
5. Move Blender cache identity adaptation out of the low-level IO module.
6. Correct pending/view/snippet UI state reporting and isolate registry tests.

## Independent Verification and Correction Outcome

All 18 findings were independently rechecked against live source and retained.
The recalibrated severity was 4 high, 8 medium, and 6 low; the differences from
the original swarm labels were prioritization changes, not rejected claims.

Tasks 1-9 in the correction plan are implemented. The corrections include
owner-scoped release plus explicit forced teardown, per-scene playback cursors,
resolved production-cache path reservation from the initialization window,
stage-first export with rollback, exact source-array ranks, structured metadata
and benchmark validation, snippet permission/type persistence with fail-closed
FUNCTION admission, Blender-owned cache identity adaptation, and truthful UI
state/actions. The final independent code re-review returned **PASS**.

Current proof after the last code fix:

- Focused correction integration: **328 passed**.
- Full Python suite: **738 passed**.
- Blender cache/export pipeline: `CACHE-EXPORT-PIPELINE-PASS`.
- Blender handler persistence: `HANDLER-PERSISTENCE-GATE-PASS`.
- Blender background acceptance: `BACKGROUND-ACCEPTANCE-PASS`.
- Default aggregate production gate (before the final cache-reservation patch):
  `HEADLESS-GATE-PASS`; the post-patch targeted live cache/export gate passed.

One non-code preservation blocker remains. The aggregate gate explicitly
rebuilt already-dirty `crown_splash` and `dam_break` `.blend`/preview artifacts.
Their captured original hashes are recorded in the autonomous checkpoint, but
the originals were not in the scoped ZIP and were not recoverable from repo
copies, Blender `.blend1` intermediates, or the available non-elevated Windows
history checks. The rebuilt files were left untouched pending exact recovery
from an external backup/history or explicit owner acceptance.

## Artifacts

- `reviews/swarm-review-2026-08-22-luna-terra/CONSOLIDATED_CODEBASE_REVIEW.md` — retained consolidated report.
- Intermediate `findings/*.json` files were generated for all seven lanes, schema-validated, deduplicated into this report, and removed according to the swarm-review cleanup protocol.
