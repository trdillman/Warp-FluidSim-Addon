# TASK-POLICY-DRIFT-001: Settle sparse-octree contract

## State

completed

## What to build

Resolve the mismatch between the promoted sparse-octree product contract and the live flat sparse leaf-table implementation. Either make the runtime genuinely satisfy a hierarchical octree contract or narrow the product/docs/UI/test wording to the implemented sparse tiled leaf-grid semantics.

## Acceptance criteria

- [x] `POLICY_DRIFT_REPORT.md` is changed to `in-progress` before project files are edited.
- [x] Product, docs, UI descriptions, and validation names no longer overstate "sparse-octree" if the flat leaf-grid design remains.
- [x] "octree" no longer remains promoted, so hierarchical parent/child or adaptive-level validation is not applicable.
- [x] A focused guard prevents the chosen contract from drifting again.
- [x] Relevant pytest and Blender acceptance lanes pass.
- [x] `POLICY_DRIFT_REPORT.md` is updated to `completed` only after verified repair.

## Blocked by

None - can start immediately
