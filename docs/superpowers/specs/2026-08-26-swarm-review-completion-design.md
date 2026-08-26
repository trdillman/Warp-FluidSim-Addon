# FluidSim Swarm Review Completion Design

## Status

Implemented, runtime-verified, and independently reviewed PASS.

## Goal

Close every valid non-security finding from the 2026-08-22 Luna/Terra swarm
review on the current `master` architecture without merging the divergent
species prototype or disturbing unrelated dirty work.

## Verified scope

All 18 findings remain valid. Current master already implemented the session
ownership, playback cursor, cache identity, staged export, metadata collision,
and finite-output corrections. This completion slice addresses the remaining
or partial behavior only:

- exact source-array ranks;
- legacy snippet permission/type persistence and fail-closed FUNCTION handling;
- complete benchmark schema validation and durable structured failures;
- production-cache reservation through failure cleanup;
- truthful pending, hidden-output, and empty-action UI states;
- complete duplicate-registry test cleanup;
- explicit documentation that multi-file export has in-process rollback, not
  crash-atomic filesystem publication.

## Design

### Boundary validation

`sim/source_regions._owned_array` validates the exact `(N,)` or `(N, tail...)`
shape before making an owned immutable copy. It never repairs malformed ranks by
reshaping them into plausible data.

`diagnostics/benchmark.compare_benchmark` validates every numeric field in
`BENCHMARK_FIELDS` before exact or performance comparison. Its report retains a
per-field `contract` section. The Blender operator stores structured failure
reports for invalid values and invalid baseline files so diagnostics remain
available after cancellation.

### Legacy snippet contract

Legacy Blender RNA and JSON retain `CustomKernelFieldPermission` and
`CustomKernelType`, defaulting old data to `READ_WRITE` and `KERNEL`.
Conservative AST effect analysis rejects forbidden reads, writes, atomics,
aliases, and unknown calls before compilation. `FUNCTION` is retained but
rejected at adapter, manager, compiler, and legacy-to-v2 conversion boundaries
until helper linkage has a real implementation.

Legacy permissions map explicitly into Custom Effects v2 `FieldAccess`; v2
field validation remains the final authority for structural/read-only fields.

### Ownership and publication

Production-bake cache ownership remains active while `backend.fail()` performs
cleanup and changes to `FAILED` only afterward. Free, Move, Rename, and orphan
cleanup therefore cannot mutate the reserved cache during failure handling.

Export continues to prevalidate and durably stage every sibling artifact before
replacing final paths, with rollback for caught publication failures. It is
explicitly not described as crash-atomic multi-file publication.

### UI truthfulness

Pending Settings owns Reset and Reinitialize presentation strings and action
identities. Both Scene Properties and the viewport consume that presentation.
`HIDDEN` is a real liquid view state for both visibility toggles off. Empty
legacy-argument and Custom Effects removal actions poll-disable or cancel with
actionable feedback.

### Test isolation

The duplicate kernel identity regression removes both its canonical registry
entry and generated source alias, then asserts clean teardown.

## Proof contract

- claim-focused host suites;
- full `python -m pytest -q` regression suite;
- Blender Phase 4 live-interaction acceptance;
- Blender handler-persistence gate;
- Blender background acceptance;
- CUDA cache/export pipeline;
- independent final code review of the current diff against all 18 claims.
