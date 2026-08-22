# FluidSim Playback, Cache, and UI Hardening Design

## Status

Approved direction captured from the 2026-08-14 playback review and Tyler's follow-up corrections. Implementation remains gated on review of this written design.

## Problem

The current add-on mixes four concerns in the visible workflow:

- live simulation playback;
- viewport output publication;
- cache playback;
- legacy and production cache recording.

Full production surface attributes are transferred during live viewport updates, making publication substantially slower than the prior implementation. Timeline recovery republishes every missed frame on Blender's synchronous frame handler, amplifying the slowdown into apparent hangs. Production Bake has a transactional backend, but its active controller is owned by the operator adapter, so Live and Cache Playback cannot reliably reject conflicting transitions. The UI exposes separate booleans and legacy controls instead of one comprehensible activity state.

## Goals

1. Restore responsive live playback without weakening solver or cache correctness.
2. Make Production Bake v2 the only primary cache-capture workflow.
3. Give one deep module authoritative ownership of an active production bake.
4. Prevent Live, Cache Playback, and Production Bake from mutating the shared solver session concurrently.
5. Present a small, explicit UI state: idle, live, cache playback, baking, cancelling, complete, or failed.
6. Remove old cache controls from the visible workflow while retaining read compatibility where practical.
7. Promote core simulation settings out of Advanced and leave only implementation, diagnostic, and experimental controls there.

## Non-goals

- Replacing the cache-v2 manifest/checkpoint format.
- Adding background worker threads around Blender RNA or GPU solver ownership.
- Changing simulation catch-up, desynchronization, rewind, or cancellation semantics.
- Removing cache-v1 read compatibility in this change.
- Redesigning unrelated source, collider, force, pressure, or viscosity panels.

## Considered Approaches

### A. Explicit modes over existing deep modules (selected)

Keep `playback.py`, `scene_bridge.py`, `output_publication.py`, and `production_bake.py` as the established owners. Move the active bake registry into `production_bake.py`, expose small query/registration functions, and have playback transitions fail closed while a bake is active. Build a read-only activity snapshot for UI presentation from those authoritative sources.

This is the smallest change that fixes ownership and keeps the existing architecture legible.

### B. Retain independent checkboxes with additional guards

This requires fewer UI changes, but the user still has to reason about invalid combinations of Live, Cache Playback, and legacy recording. It preserves the current source of confusion and is rejected.

### C. Introduce a new workflow-manager subsystem

A new state-machine module could own every transition, but it would duplicate established module responsibilities and increase migration risk. It is rejected unless the existing deep modules prove unable to enforce the required invariants.

## Runtime Design

### Live viewport publication

`output_publication.prepare_live_payload()` will publish only attributes needed for the selected live display mode. It will not transfer the complete production render-attribute inventory during ordinary live preview.

The full production inventory remains captured by explicit Production Bake/export paths. Cache playback may publish attributes already stored in the cache because it does not require live nearest-particle GPU readback.

### Timeline recovery

`playback.handle_frame_change()` will continue to simulate every allowed recovery frame in order. Intermediate recovery steps will use `publish_outputs=False`; only the final recovered frame will publish Blender viewport output. This preserves solver catch-up and desynchronization policy while removing redundant main-thread mesh work.

### Production Bake ownership

`production_bake.py` will own the active controller registry and expose narrow functions to register, query, cancel, and clear the controller for a scene. `operators.py` remains responsible only for modal timer integration and RNA presentation mirrors.

Live enablement, Cache Playback enablement, reset, release, and production-bake start will consult authoritative bake activity and reject conflicting transitions with an actionable status message.

### Explicit cache capture

Production Bake `step_and_commit()` remains the authoritative capture seam:

1. suppress playback handlers;
2. set and synchronize the requested timeline frame;
3. step the solver without viewport publication or implicit cache writing;
4. write the cache-v2 frame atomically;
5. capture and write the restart checkpoint;
6. commit the manifest last.

Implicit cache-v1 writes during live stepping will be removed. The deprecated `cache_enabled` RNA field may remain hidden for old `.blend` compatibility, but it will no longer activate live recording. Cache-v1 readers remain available for existing caches.

## UI Design

### Simulation panel

The primary panel will lead with one compact activity card containing:

- current state and icon;
- current mode detail, such as live frame, cached frame, or bake frame range;
- progress when baking;
- the current actionable error, when present.

Live controls, reset/release, and manual stepping remain nearby, but controls that conflict with Production Bake are disabled from authoritative activity state rather than inferred RNA strings.

The existing Advanced Playback disclosure is removed. Maximum Playback Recovery Frames becomes a normal Simulation setting because it directly controls visible timeline behavior.

### Cache and Bake panel

The panel is renamed to `Cache & Bake` and contains only the primary workflow:

- cache path;
- explicit `Play Cache` / `Stop Cache Playback` action;
- bake range;
- Start, Resume, Cancel, and explicit Restart actions;
- progress, last committed frame, completion, and failure presentation.

The following old controls are removed from the visible UI:

- `Record v1 During Live Playback (Legacy)`;
- the legacy/single-frame export section;
- the raw `cache_playback_enabled` checkbox.

Cache verification, open-folder, orphan cleanup, move, rename, and free operations remain available in a collapsed `Cache Utilities` child panel or disclosure so the primary workflow stays short.

### Settings promoted from Advanced

- `Min Substeps` moves into Quality beside `Max Substeps`.
- `Domain / Sparse Grid` becomes a regular top-level panel. Domain selection, boundaries, and grid summary stay visible; manual leaf-capacity/topology overrides remain under Advanced.
- `Particles` becomes a regular top-level panel because sampling, transfer, growth, and particle limits are core simulation settings. The existing ST-FLIP Beta block is split into its own Advanced child panel rather than being promoted with the core controls.
- `Advanced Field Sources` moves under Inputs and is renamed `Field Sources`.
- Maximum Playback Recovery Frames becomes a regular Simulation setting.

Custom Warp Snippets, Diagnostics/Profiling, Validation/Benchmarks, ST-FLIP Beta, true manual capacity overrides, experimental controls, and low-level surface capacity controls remain Advanced.

## Failure Handling

- Conflicting mode transitions fail before allocating, releasing, or mutating the solver.
- Production Bake cancellation remains safe-boundary cancellation.
- Cache playback reports missing, corrupt, wrong-scene, or out-of-range frames without silently falling back to live simulation.
- A failed bake clears authoritative ownership during cleanup while preserving the final FAILED status and message for UI presentation.
- UI draw functions consume read-only snapshots and must not trigger I/O, solver work, or state transitions.

## Test and Proof Contract

Implementation is complete only when all of the following pass:

1. Focused Python tests covering activity registration, conflict rejection, final-frame-only catch-up publication, explicit cache capture, and panel structure.
2. `tests/test_frame_output.py`, `tests/test_scene_bridge_cache_playback.py`, `tests/test_operator_panel_behavior.py`, `tests/test_production_gate_truthfulness.py`, and relevant production-bake tests.
3. Blender cache export, handler persistence, background acceptance, and viewport two-frame gates.
4. The user-path profiler with expectations derived from configured surface cadence and viewport mean below its declared ceiling.
5. A GUI-opened saved demo showing the simplified state panel and a real transition through Live, Cache Playback or Production Bake as available.
6. Clean diff review and clean worktree except for the intentional implementation.

## Acceptance Criteria

- Ordinary live preview does not transfer unselected production surface attributes.
- Catch-up publishes one viewport frame per Blender timeline event.
- No live frame handler performs an implicit cache-v1 write.
- Production Bake is the sole primary cache-capture workflow.
- Live, Cache Playback, and Production Bake cannot own the shared solver session simultaneously.
- The main UI visibly distinguishes idle, live, cache playback, baking, cancelling, complete, and failed states.
- Legacy cache controls are absent from the primary UI.
- Core Domain, Particle, Field Source, substep, and playback settings are no longer hidden under Advanced.
- Existing cache-v1 read compatibility is preserved.
