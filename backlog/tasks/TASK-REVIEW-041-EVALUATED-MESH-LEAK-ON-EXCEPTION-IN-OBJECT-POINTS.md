# TASK-REVIEW-041: Evaluated mesh leak on exception in object_points_world

## State

completed

## Severity

low

## What to build

In object_points_world, the MESH branch calls eval_obj.to_mesh() at line 118 to allocate an evaluated mesh, but eval_obj.to_mesh_clear() at line 124 is only reached on the success path. If mesh.vertices.foreach_get('co', pts) at line 121, pts.reshape at line 122, or the matrix multiply at line 123 raises an exception, to_mesh_clear() is never called, leaking the evaluated mesh data in Blender's memory for the remainder of the session.

**Location:** `fluidSim_warp_blender51/blender/sources.py:118`
**Dimension:** Serialization & IO

## Impact

Leaked mesh data accumulates in Blender memory. The trigger is low probability since the operations rarely fail on valid meshes, but each occurrence leaks permanently until the .blend file is closed.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/blender/sources.py:118` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Wrap the mesh operations in a try/finally block to guarantee eval_obj.to_mesh_clear() is called regardless of exceptions.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SERIO-002` in `reviews/swarm-2026-08-08/findings/io.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Serialization & IO lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately

## Resolution

Fixed for 2.0 (2026-09-27). `object_points_world` (`fluidSim_warp_blender51/blender/sources.py`) now reads the vertices inside `try` and calls `eval_obj.to_mesh_clear()` in `finally`, so a failed read no longer leaks the evaluated mesh. Regression: `tests/test_sources_object_points_release.py` (world-space result plus release on success, and release when the vertex read raises).
