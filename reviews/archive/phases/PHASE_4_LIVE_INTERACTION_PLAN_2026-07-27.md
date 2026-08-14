# Phase 4 — Live Interaction and Pending-Reinitialize Implementation Plan

## Status

**In progress in an isolated worktree. Not accepted.**

- Worktree: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_phase4_interaction`
- Branch: `phase4/live-property-interaction`
- Approved product policy: **Hybrid pending + auto-on-rewind**
- Initial independent review: `reviews/archive/phases/PHASE_4_INITIAL_INDEPENDENT_REVIEW_2026-07-27.md`
- Initial review verdict: no blocker/high, but correctness and coverage follow-ups remain.

This phase repairs an interaction-truth defect discovered after the three-phase source-system program: many live physics edits previously reset simulation state without reapplying the edited `SolverSettings` value.

## Truth contract

### Core behavior that must be real

1. A setting labeled or classified **Live** reaches the active solver before the next applicable frame without clearing particles, whitewater, or frame history.
2. Live values are reconciled consistently for interactive playback, manual stepping, cache export, keyframes, drivers, undo/redo, and scripted edits.
3. Restart/reallocation settings never silently destroy an advanced simulation.
4. Restart/reallocation settings remain visibly pending until successfully applied.
5. Pending settings apply automatically on timeline rewind/reset and through an explicit `Apply Pending & Reinitialize` action.
6. Reset and reinitialize paths rebuild their authoritative state from current Blender RNA values in the correct order.
7. Every exposed setting belongs to exactly one mechanically checked interaction tier.
8. Errors during live application are contained, recorded, visible, and do not silently kill playback when the previous valid setting can continue.
9. Pending state and deferred actions are scene-specific.
10. Cache output records enough settings-signature/change information to avoid presenting a mixed-parameter sequence as homogeneous.
11. Standard Blender UI constraints remain truthful: there is no claim of CSS-style sticky content inside a scrolling sidebar.

### Allowed experimental surfaces

- The optional clickable 2D viewport gizmo may remain experimental and disabled by default.
- ST-FLIP remains Beta, but any setting classified Live must still update its real runtime state consistently.
- A viewport overlay may summarize pending state, but it is informational and cannot substitute for the real apply/reinitialize path.

### Review/progress artifacts

- Screenshots, GUI reports, overlays, demos, and review scenes prove presentation and user reachability only.
- They do **not** prove solver-state mutation, particle preservation, rewind semantics, or cache truth.
- Runtime and behavioral evidence must pass before visual artifacts can complete a gate.

## Interaction tiers

| Tier | Contract | Examples | Required behavior |
|---|---|---|---|
| `LIVE` | Safe bounded setting mutation | CFL, FLIP ratio, gravity, pressure iterations, viscosity, global force values, most whitewater tuning | Copy into active `SolverSettings`; preserve simulation state; reconcile before every step entry point |
| `LIVE_REFRESH` | Live mutation plus bounded subsystem refresh | Surface settings, whitewater enable, ST-FLIP derived calibration | Apply value and run only the necessary targeted refresh/recalibration |
| `RESET_PENDING` | Meaningful only from simulation start | Initial-fill lifecycle, direct-append latch policy, ST seed | Keep current run; apply on rewind/reset or explicit action |
| `REINIT_PENDING` | Constructor/allocation/domain change | Voxel size, leaf/particle/whitewater capacities, domain bounds, device | Keep current run; reconstruct on rewind or explicit action |
| `BLOCKED` | Unsafe transition in current context | Device transition during active playback | Keep pending and show an actionable reason |

Every user-facing property must be assigned to exactly one tier by a mechanical test.

## Current claim matrix

| Claim | Source | Required evidence | Current implementation | Status | Next action |
|---|---|---|---|---|---|
| Hot settings update without state loss | User + truth contract 1 | Source + behavioral unit + live Blender/CUDA | In-place `_HOT_SETTING_FIELDS` copy; unit preserves particle identity | `partial` | Add live Blender/CUDA mutation evidence |
| Reset uses edited settings | User + truth contract 6 | Source + reset-order test + Blender reset evidence | Settings applied before `solver.reset()`; unit passes | `partial` | Add Blender rewind/reset evidence |
| Live errors are contained | Truth contract 8 | Forced-error behavior test | LIVE callback exceptions can escape; playback reconcile can release session | `nonconformant` | Fix review S1 before file gate |
| ST-FLIP remains internally calibrated | Truth contract 1/6 | Derived-state invariant test | Hot `particles_per_voxel` leaves cached reference mass stale | `nonconformant` | Add recalibration hook or reclassify field |
| Animated values behave identically across step paths | Truth contract 2 | Playback/manual/cache differential test | Playback reconciles; manual step and cache export do not | `nonconformant` | Centralize pre-step reconciliation |
| Every property has one tier | Truth contract 7 | Mechanical registry/RNA test | Registries currently consistent, but coupling is unenforced | `partial` | Add exhaustive property-tier test |
| Gravity source/custom branches are live | Truth contract 1 | Behavioral tests for both branches | Code exists; assertions are missing | `partial` | Add both branch tests |
| Destructive edits stay pending | Approved hybrid policy | Mid-run behavioral test | Current rebuild/reset timer remains destructive | `missing` | Implement per-scene pending model after review file gate |
| Rewind applies pending changes | Approved hybrid policy | Blender timeline test | Not implemented | `missing` | Implement P2 auto-on-rewind |
| Always-reachable apply controls | User interaction request | Real Blender GUI interaction | Not implemented | `missing` | Add header popover and local actions in P3 |
| Cache identifies settings changes | Truth contract 10 | Cache metadata/report validation | Not implemented | `missing` | Add change boundaries/signatures in P4 |

No capability may be promoted to `conformant` from source inspection or screenshots alone.

## Relevant implementation-truth traps

- **Review artifact replacing implementation truth:** UI screenshots and the review file remain subordinate to runtime behavior.
- **Tests proving presence, not behavior:** symbol/regex tests may support registry checks but cannot certify live behavior alone.
- **Tuning around a broken core:** do not hide ST-FLIP inconsistency by changing defaults or thresholds.
- **Mislabeled behavior:** do not label a property Live until every authoritative step path applies it.

No new generalized trap pattern has been proven, so the shared trap catalog does not need modification yet.

# Execution plan

## P1A — Close the initial independent-review findings

This is the immediate critical path. P2 may not start until the review-follow-up file gate passes.

### P1A.1 — Error containment

- Add a single bounded live-application error boundary.
- RNA callback failure must set persistent error/status and redraw without propagating into Blender property machinery.
- Per-frame reconcile must retain the last valid settings and avoid releasing the solver for a recoverable settings-copy failure.
- Add forced bad-enum/translation tests covering callback and playback paths.

Closes: review `S1`.

### P1A.2 — ST-FLIP recalibration truth

- Decide `particles_per_voxel` behavior from runtime semantics, not label convenience.
- Recommended: classify it `LIVE_REFRESH` and recompute `_st_reference_mass_cached` when ST-FLIP is enabled and automatic reference mass is selected.
- Keep explicit `spacetime_reference_mass` restart/reset semantics unless a separately verified hot hook is implemented.
- Add invariant tests proving the cached reference mass changes with the live particle-density control.

Closes: review `S2` and the plan's existing ST-FLIP item.

### P1A.3 — Unify all authoritative step entry points

- Create one pre-step reconciliation seam used by:
  - interactive frame-change playback;
  - manual `Step 1` / `Step 10`;
  - cache export/bake;
  - any direct bridge stepping entry point.
- Evaluate animated/driven RNA at the intended timeline frame before reconciliation.
- Add differential tests proving the same keyframed CFL/gravity/FLIP sequence reaches playback, manual, and export paths.

Closes: review `S3`.

### P1A.4 — Mechanical property-tier gate

Add a test that asserts:

1. every `_request_live` RNA property appears in `_HOT_SETTING_FIELDS` or the explicit gravity allowlist;
2. every hot/reset/surface/whitewater field maps to a real `SolverSettings` target or declared translator;
3. tier sets are pairwise disjoint where required;
4. allocation-bound fields never enter hot/reset mutation sets;
5. every user-facing update callback is assigned to one declared tier;
6. the interaction matrix and runtime registries cannot drift silently.

Replace or subordinate the brittle regex-only callback test.

Closes: review `T1`, `T3`.

### P1A.5 — Gravity and status truth

- Add behavioral tests for scene-gravity enabled and custom-gravity fallback.
- Use truthful inactive status text such as `Live setting stored; applies on initialization`.
- Avoid replacing frame statistics with repetitive slider-tick status unless an error or pending transition occurs.

Closes: review `S4`, `S5`, `T2`.

### P1A.6 — Bounded hot-path cleanup

- Hoist or memoize invariant registry validation so it is not rebuilt each frame.
- Apply hot + reset fields in one bounded pass during reset.
- Profile the reconcile helper separately from the GPU frame step.

Closes: review `T4`, `T5`.

## FILE GATE A — Initial review follow-up

**Hard gate: P2 cannot begin until this exact file exists:**

`reviews/archive/phases/PHASE_4_INITIAL_REVIEW_FOLLOWUP_GATE_2026-07-27.md`

The file must be produced from a **fresh independent post-fix review** and contain all of the following:

1. A finding-by-finding disposition for `S1–S6` and `T1–T6` from `PHASE_4_INITIAL_INDEPENDENT_REVIEW_2026-07-27.md`.
2. Exact `path:line` evidence for every fixed finding.
3. Any deferred finding, its risk, and explicit reason it does not block P2. Correctness findings `S1`, `S2`, `S3`, `T1`, and `T2` may not be deferred.
4. Focused interaction test command and result.
5. Full non-Blender regression command and result.
6. A bounded performance result for pre-step reconciliation.
7. Confirmation that the original Phase 3 worktree has no Phase 4 symbols and neither worktree has staged files.
8. Reviewer verdict with **no blocker/high and no unresolved correctness medium**.
9. Exact terminal marker:

```text
PHASE4-INITIAL-REVIEW-FOLLOWUP-PASS
```

A parent-authored summary, test log, screenshot, or the existing initial review file does not satisfy this gate. The fresh reviewer must inspect the post-fix implementation and write the gate file.

## P2 — Per-scene pending state and auto-on-rewind

Depends on: **FILE GATE A passed**.

### P2.1 — Applied/current signatures

- Define canonical signatures for `RESET_PENDING` and `REINIT_PENDING` values.
- Compare current RNA values against the last successfully applied values; do not use a stale boolean.
- Include domain bounds/object identity and device where relevant.
- Keep runtime ownership per scene/session and expose only cached presentation values to panel draw.

### P2.2 — Non-destructive transition policy

- If no solver exists, store values normally for the next initialization.
- If the solver is at a disposable start state, allow debounced automatic application.
- If the simulation has advanced, preserve the active run and mark the highest required pending action.
- Remove the current silent mid-run destructive rebuild/reset behavior.

### P2.3 — Rewind/reset semantics

- On return to `scene.frame_start` or another authoritative backward-reset transition:
  - apply `REINIT_PENDING` through full initialization;
  - otherwise apply `RESET_PENDING` settings before state reset.
- Clear pending state only after success.
- Preserve pending reasons after a failed apply.

### P2.4 — Explicit operators

- `Apply Pending & Reinitialize`
- `Apply Pending & Reset`
- `Revert Pending Values` using the applied snapshot
- Operators must be context-safe, report failures, and never save Blender preferences.

### P2 verification checkpoint

- Multiple-scene isolation.
- Mid-run voxel-size edit preserves the current run.
- Rewind reconstructs with the new voxel size/capacity.
- Reset-semantic lifecycle changes re-arm exactly once.
- Undo/scripted edit signatures update correctly.

## P3 — Always-reachable Blender interaction surfaces

Depends on: P2 runtime behavior green.

### P3.1 — Fixed header control

- Append a compact control to `VIEW3D_HT_header`.
- Use a popover containing solver state, pending count/reasons, Live toggle, Apply, Reset, Revert, and Release.
- Register/unregister symmetrically and persistently without modifying user preferences.

### P3.2 — Owning-panel feedback

- Mark Live, Restart Required, and Reinitialize Required controls truthfully.
- Show a pending banner and local action row in the owning panel.
- Do not perform compilation, GPU queries, geometry analysis, or RNA writes in `Panel.draw`.

### P3.3 — Viewport overlay

- Reuse the existing POST_PIXEL overlay for informational pending text.
- Keep it non-clickable and subordinate to the real header/operator path.
- Optional 2D gizmo remains preference-gated and default off.

### P3 verification checkpoint

- Real Blender GUI interaction, not source-string presence.
- Header remains accessible while the sidebar is scrolled.
- Pending count/reasons match runtime state.
- Profile-safe registration/unregistration and unchanged user profile hashes.

## P4 — Cache truth, documentation, and runtime acceptance

### P4.1 — Cache metadata

- Record settings signature/change boundaries for each sequence segment or frame.
- Validate that mixed-parameter sequences are reported rather than presented as homogeneous.

### P4.2 — Documentation

- Add the five-tier interaction matrix to the user guide.
- Explain auto-on-rewind and explicit Apply/Revert behavior.
- State Blender's no-sticky-sidebar limitation and the supported header alternative.

### P4.3 — Runtime evidence

Required Blender/CUDA lanes:

1. Hot FLIP ratio/gravity/pressure/viscosity edit while particles remain alive.
2. ST-FLIP derived recalibration.
3. Playback/manual/cache step-path parity.
4. Mid-run reallocation edit stays pending.
5. Timeline rewind applies and reconstructs.
6. Multiple-scene pending isolation.
7. Header popover and overlay GUI acceptance.
8. Save/reopen and profile-safe review.
9. Balanced performance and memory profiles.

## FILE GATE B — Final interaction acceptance

Required file:

`reviews/archive/phases/PHASE_4_FINAL_INTERACTION_REVIEW_2026-07-27.md`

It must contain:

- final claim matrix with every shipped claim `conformant`;
- source, runtime, cache, GUI, and performance evidence;
- fresh independent reviewer verdict with no blocker/high and all correctness mediums resolved;
- separate headless and GUI-inclusive production markers;
- explicit remaining limitations;
- marker `PHASE4-FINAL-INTERACTION-PASS`.

# Dependencies, parallel opportunities, and critical path

## Critical path

```text
P1A review fixes
  -> FILE GATE A independent follow-up
  -> P2 pending/signature runtime
  -> P3 header/panel/overlay UI
  -> P4 cache/docs/Blender evidence
  -> FILE GATE B final independent acceptance
```

## Safe parallel work

- Read-only cache metadata research may run while P1A tests are implemented.
- UI copy/icon exploration may run while P2 runtime work is implemented, but UI code must wait for the pending-state API.
- Independent reviewers may inspect completed slices in parallel only when they do not edit the worktree.

## Single-writer rule

Only one writer may modify the active Phase 4 worktree at a time. Reviewers are read-only. No implementation begins past a failed file gate.

# Current evidence and open state

Initial implementation evidence:

```text
72 focused tests passed
348 full non-Blender tests passed (independent reviewer run)
```

Current truth verdict:

- Real but partial: in-place hot-setting copy, particle preservation unit seam, reset ordering, per-playback reconciliation.
- Nonconformant: live error containment, ST-FLIP derived recalibration, all-step-path parity.
- Missing: pending signatures, auto-on-rewind, fixed controls, cache metadata, Blender/CUDA acceptance.

This plan itself is not evidence that Phase 4 works. Phase 4 remains incomplete until both file gates and their underlying runtime evidence pass.
