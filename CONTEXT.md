# FluidSim Domain Context

This glossary names the domain concepts used by architecture reviews and implementation plans. It does not replace the synthesized FluidSim requirements or the sparse-grid contract in `AGENTS.md`.

## Simulation runtime

- **Solver Session** — the currently owned GPU solver runtime for one Blender scene, including direct scene synchronization, stepping, teardown, errors, and frame statistics.
- **Simulation Admission** — the Blender-owned transaction that turns one authored Blender scene evaluation into a running Solver Session.
- **Admission Snapshot** — one coherent capture of scene identity, frame, depsgraph generation, configured device, domain transform, sources, colliders, forces, height source, seeds, and settings used by one Simulation Admission attempt.
- **Admission Preflight** — every non-destructive gather, compile, memory, capacity, and integrity check performed before the existing Solver Session is released.
- **Replacement Cutover** — the explicit point after successful Admission Preflight where the existing Solver Session is released and replacement GPU allocation begins. Replacement is preflight-atomic, not fully swap-atomic.

## Blender output

- **Output Preparation** — creation or resizing of persistent Blender output objects, datablocks, materials, point capacity, and attribute schema. Initial and property-triggered preparation occurs outside the accepted frame-publication hot path. Existing structural point-cloud replacement for growth, shrink, or attribute-schema change remains a characterized fallback until separately changed under the red-test rule.
- **Publication Payload** — owned geometry, attribute, visibility, and cadence facts prepared from either a live solver frame or a cached frame. Preparing a live payload may use the existing required device-to-host readback; publishing it must not reacquire solver data.
- **Output Publication** — mutation of persistent Blender output state from one Publication Payload. It owns geometry, attributes, capacity/schema decisions, materials, viewport visibility, and render visibility. Diagnostic overlay drawing is not Output Publication.

## Playback and cache

- **Preview Playback** — non-authoritative viewport feedback driven by Blender timeline events. Its current catch-up, desynchronization, error, and backend-status behavior is preserved by the architecture program.
- **Cache Playback** — visual publication of already cached frames without stepping or initializing a live solver.
- **Production Cache Bake** — control-flow deterministic sequential frame advancement, scene synchronization, solver stepping, cache writing, and metadata recording independent of viewport event delivery and preview/display settings. Control-flow determinism does **not** mean repeat-bake numerical equality: CUDA atomic accumulation, compaction/emission ordering, and derived marching-cubes surfaces are nondeterministic, so no bitwise or particle-corresponded match is claimed. Exact payload hashes, particle/surface counts, and surface topology are diagnostics only. Acceptance instead proves at each real write seam that full non-preview-decimated liquid and non-empty whitewater arrays and attributes equal the immediate solver counts, and proves frame handlers and Output Publication are bypassed. The bake restores Blender frame/live and temporary handler suppression, but intentionally leaves solver state advanced; it is not a solver rollback transaction. Production output comes from this path, not from Preview Playback.

## Settings interaction

- **Pending Settings** — authored scene edits whose effect on an active Solver Session is classified as live, synchronized, output-only, reset, reinitialize, or authored-next-run.
- **Applied Settings Snapshot** — the per-scene record of settings and scene facts currently represented by the Solver Session, used to derive Pending Settings without making UI adapters reconstruct policy.

## Storage invariant

- **Sparse Leaf Grid** — the live field-storage contract: fixed `8x8x8` leaf tiles addressed through a flat leaf table, with no hierarchical tree storage.

## Change truth rule

Architecture ownership changes preserve accepted behavior. A targeted behavior change is allowed only when a reproduced user-visible defect or documented contract violation is first captured by a failing regression test. Review reports, demos, screenshots, and policy artifacts are evidence surfaces, not substitutes for source-level and runtime proof.
