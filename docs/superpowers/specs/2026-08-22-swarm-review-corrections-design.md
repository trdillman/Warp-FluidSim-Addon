# FluidSim Swarm Review Corrections Design

## Status

Approved for autonomous execution by Tyler's explicit instruction to verify,
plan, and autonomously correct every valid swarm-review claim. This design
extends the approved 2026-08-14 playback/cache/UI architecture; it does not
replace established ownership boundaries.

## Goal

Correct all 18 independently verified findings in
`reviews/swarm-review-2026-08-22-luna-terra/CONSOLIDATED_CODEBASE_REVIEW.md`
with the smallest maintainable behavior, preserve the existing dirty working
tree, and prove every behavior change with observed test-first failure and
focused runtime evidence.

## Verified scope

All 18 claims are valid. They collapse into nine correction batches because
ARCH-001, UI-001, and part of UI-002 share one singleton-ownership root.
Recalibrated severity is four high, eight medium, and six low.

## Considered approaches

### A. Extend the existing deep owners (selected)

Keep solver lifecycle in `scene_bridge.py`, playback policy in `playback.py`,
production-bake ownership in `production_bake.py`, persistent output in
`output_publication.py`, and setting lifecycle in `pending_settings.py`.
Strengthen their narrow APIs and keep operators/panels as adapters. This closes
the verified gaps without a competing workflow subsystem.

### B. Guard only UI/operator entry points

This is smaller but leaves unsafe deep APIs callable by handlers, scripts, and
future adapters. It cannot make scene-scoped release or cache-path ownership
truthful, so it is rejected.

### C. Add a workflow/transaction manager

A central coordinator could own solver modes, cache leases, exports, and UI
state, but it would duplicate established owners and create an unrequested
subsystem. It is rejected as unnecessary architectural expansion.

## Architecture

### 1. Solver-session ownership

`Session.release(scene)` becomes an owner-scoped operation: it releases only an
inactive session or a session owned by `scene`, returns a truthful boolean, and
does not mutate a foreign owner's state. A separate explicit forced teardown
function owns load-pre, unregister, fatal admission cleanup, and other global
lifecycle paths. UI and playback callers use only owner-scoped release.

Session-backed operators must validate ownership in both `poll()` and
`execute()` because Python invocation can bypass UI availability. Release gets
its own poll that requires an active owned session and idle production bake but
does not require a fresh Warp/device readiness probe.

Cache playback refuses activation when another scene owns the live session and
restores the requested RNA flag under the existing transition guard. Cache
playback maintains a per-scene cursor in `playback.py`; `Session.last_frame`
remains exclusive to the live owner. Cursor state is cleared on load and
unregister.

### 2. Production cache path ownership

The existing scene-owned production-bake registry remains the single authority.
It gains a resolved-path query that reports whether an initializing, running,
or cancelling backend owns a cache directory. No second lease registry is
introduced.

Free, Move, Rename, and Remove Orphans check that authority in both operator
poll and execute. The deep cache-manager functions also accept a guard or call
the shared path authority so scripts cannot bypass the safety boundary. An
active backend retains its directory through finish, cancel, or failure cleanup.

### 3. Current-frame export publication

All surface, particle, and whitewater payloads are captured and validated before
any final path is touched. Writers emit to unique same-directory staging files,
flush and fsync where supported, then publish the complete bundle. Existing
finals are backed up before replacement; any publication failure restores them
and removes staging/backup debris.

This is a bounded rollback transaction, not a claim of literal cross-file crash
atomicity. A generation-directory/manifest commit protocol is outside scope.

### 4. Cache dependency direction

Blender scene/settings/source identity collection moves to a new
`blender/cache_identity.py` adapter. It passes plain values into pure hash and
serialization helpers in `io/cache.py`. Admission, playback, and production
bake call the Blender adapter. `io.cache` must not import `fluidSim_warp_blender51.blender`.

### 5. Boundary validation

- Source plans require exact vector shapes `(N, 3)` and scalar shapes `(N,)`;
  they copy to C-contiguous immutable arrays without reshaping malformed input.
- Legacy sequence metadata preserves raw frame keys until collision detection;
  normalized duplicates fail before writing metadata.
- Benchmark comparison accepts finite JSON numbers only (excluding booleans),
  returns a structured fail result for malformed fields, and the operator turns
  malformed baseline files into an actionable `CANCELLED` state.
- OBJ and PLY export reject non-finite geometry/attribute values before opening
  output paths.

### 6. Custom snippet contract

Snippet field permission is persisted through RNA, preset serialization,
adapter conversion, and `CustomArgument`. Conservative AST effect analysis
enforces:

- READ: loads allowed; stores, augmented stores, atomics, aliases, and unknown
  mutation calls rejected.
- WRITE: plain stores allowed; loads and read-modify-write rejected.
- READ_WRITE: both allowed, subject to existing sandbox validation.

Warp array declarations alone are not treated as permission enforcement.
Unmodelled effects fail closed.

Snippet type is also persisted with legacy/default value KERNEL. FUNCTION is
rejected at admission with a durable per-snippet diagnostic and is never
compiled or dispatched. The synthesized contract does not define helper
linkage/signatures, so inventing a function compiler is outside this repair.

### 7. UI and test truthfulness

Pending presentation consumes the authoritative action from
`pending_settings.py`: reset-tier text says reset, reinitialize-tier text says
reinitialize. Panels do not reconstruct lifecycle policy.

Liquid view adds an explicit hidden state mapping to both output toggles false.
Snippet mutation operators poll-disable empty/no-argument operations and retain
execute-time defensive cancellation.

Registry-mutating tests snapshot and restore both `KERNEL_REGISTRY` and
`SOURCE_ALIASES`; no production registry change is required.

## Error handling

Every boundary fails closed before destructive publication or solver mutation.
Errors are preserved as actionable operator reports/status messages. Cleanup
removes only transaction-owned temporaries, backups, cursors, or leases.
Forced solver teardown remains explicit so owner checks cannot suppress global
lifecycle cleanup.

## TDD and proof contract

Each batch follows RED -> observe expected failure -> minimal GREEN -> focused
suite. Tests assert behavior, not source text, except the explicit dependency
direction contract where import-graph structure is the behavior being guarded.

Required final proof:

1. Focused source, cache/export, snippet, ownership/playback, pending/UI, and
   admission suites.
2. Blender background cache-export, handler-persistence, and general add-on
   acceptance.
3. Full `python -m pytest -q`.
4. Default `python tools/run_production_gate.py`; GUI-inclusive gate when the
   owned process and add-on/MCP runtime can satisfy the project gate without
   disturbing unrelated profile state.
5. Live diff/status audit against the captured pre-loop hashes.
6. Independent reviewer PASS over requirements, diff, tests, and runtime proof.

## Acceptance criteria

- A foreign scene cannot release, mutate, export, or change snippets on the
  owning scene's solver.
- Load/unregister/fatal teardown still releases the solver explicitly.
- Cache playback cannot change a foreign live cursor or activate over a foreign
  live owner.
- Runtime probe failure does not hide safe owner-scoped Release.
- An active production bake's resolved cache directory cannot be freed, moved,
  renamed, or orphan-cleaned.
- Failed current-frame export preserves the prior complete bundle and leaves no
  transaction debris.
- `io.cache` has no Blender dependency.
- Malformed shapes, metadata keys, benchmark fields, and non-finite geometry
  fail closed before publication.
- Field permissions survive persistence and are enforced conservatively.
- FUNCTION snippets persist truthfully and fail before code generation or launch.
- Pending, liquid-view, and snippet operator presentation matches real state.
- Registry tests restore every global map they mutate.
- All required gates pass and unrelated pre-existing dirty work is preserved.
