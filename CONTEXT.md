# FluidSim Domain Context

This glossary names the domain concepts used by architecture reviews and implementation plans. It does not replace the synthesized FluidSim requirements or the sparse-grid contract in `AGENTS.md`.

## Simulation runtime

- **Solver Session** — the currently owned GPU solver runtime for one Blender scene, including direct scene synchronization, stepping, teardown, errors, and frame statistics.
- **Simulation Admission** — the Blender-owned transaction that turns one authored Blender scene evaluation into a running Solver Session.
- **Admission Snapshot** — one coherent capture of scene identity, frame, depsgraph generation, configured device, domain transform, sources, colliders, forces, height source, seeds, and settings used by one Simulation Admission attempt.
- **Admission Preflight** — every non-destructive gather, compile, memory, capacity, and integrity check performed before the existing Solver Session is released.
- **Replacement Cutover** — the explicit point after successful Admission Preflight where the existing Solver Session is released and replacement GPU allocation begins. Replacement is preflight-atomic, not fully swap-atomic.
- **Active Domain** — the scene's designated domain object (`scene.fluidsim.domain_object`) and the `fluidsim_domain` settings group on it. Domain settings exist only there; `scene.fluidsim` holds scene-level state. No domain object means "no domain", never a cue to read `scene.fluidsim`.
- **Scene identity** — the key that names one Blender scene across Python wrappers (the datablock pointer; `id()` only for dead datablocks and test doubles).

## Blender output

- **Output Preparation** — creation or resizing of persistent Blender output objects, datablocks, materials, point capacity, and attribute schema. Initial and property-triggered preparation occurs outside the accepted frame-publication hot path. Existing structural point-cloud replacement for growth, shrink, or attribute-schema change remains a characterized fallback until separately changed under the red-test rule.
- **Publication Payload** — owned geometry, attribute, visibility, and cadence facts prepared from either a live solver frame or a cached frame. Preparing a live payload may use the existing required device-to-host readback; publishing it must not reacquire solver data.
- **Output Publication** — mutation of persistent Blender output state from one Publication Payload. It owns geometry, attributes, capacity/schema decisions, materials, viewport visibility, and render visibility. Diagnostic overlay drawing is not Output Publication.

## Playback and cache

- **Preview Playback** — non-authoritative viewport feedback driven by Blender timeline events. Its current catch-up, desynchronization, error, and backend-status behavior is preserved by the architecture program.
- **Cache Playback** — visual publication of already cached frames without stepping or initializing a live solver.
- **Production Cache Bake** — control-flow deterministic sequential frame advancement, scene synchronization, solver stepping, cache writing, and metadata recording independent of viewport event delivery and preview/display settings. Control-flow determinism does not by itself mean repeat-bake numerical equality. Since `d2a6b6d` the `dam_break` and `crown_splash` bakes are bitwise repeatable on the same GPU and Warp version, but that is not a general guarantee: mesh and compiled-source emitters, color mixing and some splats still accumulate in arrival order, derived marching-cubes surfaces are unverified, and results change across GPUs and Warp versions. Exact payload hashes, particle/surface counts, and surface topology are diagnostics only. Acceptance instead proves at each real write seam that full non-preview-decimated liquid and non-empty whitewater arrays and attributes equal the immediate solver counts, and proves frame handlers and Output Publication are bypassed. The bake restores Blender frame/live and temporary handler suppression, but intentionally leaves solver state advanced; it is not a solver rollback transaction. Production output comes from this path, not from Preview Playback.

## Settings interaction

- **Pending Settings** — authored scene edits whose effect on an active Solver Session is classified as live, synchronized, output-only, reset, reinitialize, or authored-next-run.
- **Applied Settings Snapshot** — the per-scene record of settings and scene facts currently represented by the Solver Session, used to derive Pending Settings without making UI adapters reconstruct policy.

## Module ownership and dependency direction

- **Simulation Admission** (`blender/simulation_admission.py`) owns the authored-scene-to-running-Solver-Session transaction, coherent Admission Snapshot, preflight, Replacement Cutover, and failure cleanup.
- **Solver Session** (`blender/scene_bridge.py`) owns only the live solver runtime, direct scene synchronization and stepping, frame statistics, errors, teardown, and its established lifecycle compatibility functions. It also owns every write to its own runtime fields: Simulation Admission performs Replacement Cutover through `install_solver`, restarts per-run bookkeeping with `begin_run`, and binds the scene with `bind_scene`; it never assigns Session fields (`tests/test_session_runtime.py` guards this).
- **Output Publication** (`blender/output_publication.py`) owns persistent output objects/datablocks, payload adaptation, preparation, publication, materials, attributes, capacity/schema, and visibility. `viewport_overlay.py` only draws diagnostics.
- **Playback** (`blender/playback.py`) owns Preview/Cache Playback selection, timeline interpretation, catch-up/desynchronization, rewind delegation, render-restore and re-entry suppression, and playback handlers.
- **Production Cache Bake** (`blender/cache_bake.py`) owns authoritative sequential cache generation and restoration of temporary Blender state. It does not call Preview Playback handlers or Output Publication.
- **Pending Settings** (`blender/pending_settings.py`) owns the exhaustive lifecycle registry, applied snapshots, pending state/presentation, apply/revert/debounce policy, and depsgraph/undo/load handlers.
- **Solver Settings** (`blender/solver_settings.py`) owns Blender-authored-value projection only.
- **Scene access** (`blender/scene_access.py`) owns the Active Domain accessors (`active_domain_object`, `active_domain_settings`), Scene identity, and the scene custom-property helpers. It is a leaf (no `bpy`, no FluidSim imports), so every module imports it at the top; nothing re-derives the designated domain or a scene's identity.

Dependency direction is one-way: Blender UI/handler adapters call these deep modules; deep modules call Solver Settings, Output Publication, the Solver Session, and `sim`/`io` owners through narrow seams. UI adapters do not reconstruct lifecycle or playback policy, and Output Publication never reacquires solver data while publishing.

Runtime tests and source-level contracts certify these claims. HTML reviews, screenshots, demos, profiles, and policy reports are supporting evidence only, not substitutes for core implementation proof.

## Storage invariant

- **Sparse Leaf Grid** — the live field-storage contract: fixed `8x8x8` leaf tiles addressed through a flat leaf table, with no hierarchical tree storage.

## Change truth rule

Architecture ownership changes preserve accepted behavior. A targeted behavior change is allowed only when a reproduced user-visible defect or documented contract violation is first captured by a failing regression test. Review reports, demos, screenshots, and policy artifacts are evidence surfaces, not substitutes for source-level and runtime proof.
