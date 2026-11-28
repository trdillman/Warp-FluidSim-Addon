# FSPT P0 completion and persistent live development

Session: 27 November 2026, 23:21:57 UTC through 28 November, 04:21:57 UTC.
Feature freeze and delivery preparation begin at 04:06:57 UTC. Continue the
existing isolated `experimental/fspt-p1-20261125` worktree and draft PR 89.

## Goal and accepted design

Complete P0, then advance the existing live bridge design without changing
solver ownership or the full compatibility/quality endpoints. The accepted
design is `docs/fspt/P2_LIVE_BRIDGE_SPEC.md` and the linked architecture and
implementation Pages. `docs/fspt/P0_ACCEPTANCE_CONTRACT.md` now freezes the
joint quality/freshness/overhead requirements. The architecture remains
Blender/Python/Warp CUDA with an explicit pinned-host image fallback. Native
OptiX is a separately qualified feasibility probe, not an adopted backend.

## Global constraints

- No renderer-triggered solver steps or authored-quality changes. No ST-FLIP,
  merge, release or relaxed performance/compatibility target.
- One native source writer and one admitted GPU lane: root owns Windows
  synchronization, commits, source freezes, packaging and GPU launches.
- Immutable source/publication/image identities; reset, quarantine, owner or
  cache changes invalidate incompatible work. Final rendering is exact-frame.
- Optional work must be admitted before capture/allocation. Own every buffer
  until copy and last GPU reader retirement are separately proven.
- A retained earlier live image keeps its real identity and cannot count as
  a newly solved/presented state. Superseded presentation is bounded to 250 ms
  from publication completion; that clock is not the contract's authoritative
  solver-completion freshness clock.
- Tests need meaningful failure controls and real runtime acceptance for
  claimed CUDA/Blender behavior. Host, source, installed and performance
  outcomes remain separate.
- Preserve failed attempts and historical P1 evidence. Back up owned work at
  meaningful checkpoints and at least every 30–45 minutes when available.

## Task 1 — P0 reconciliation and feasibility

Freeze source/repair ownership, fixtures, runtime pins, pass/optics/material
requirements and baseline schedule. Execute actual Blender API/color/graphics
and current-ABI OptiX probes. Independently verify reports and terminal exits.

Result: complete at `1995a7548e632566674a3bda08c7201491a57e5c`; normative
evidence is `docs/fspt/P0_RECONCILIATION.md`. No realtime claim follows.

## Task 2 — Asynchronous evaluated live route

Finish the ongoing implementation in `fspt/viewport.py`, `blender/fspt_engine.py`
and `blender/fspt_bridge.py`. Capture only after short admission, submit/poll
outside draw, present only completed owned pixels, serialize final claims,
and retire on history/lifecycle boundaries. Preserve immutable capture-time
geometry audits in a separate async live verifier. Verify static two-view,
resize, final handoff and teardown behavior, then genuine changing fluid.
Wire new acceptance scripts into the existing production inventory.

Owner: live_completion. Independent review: p0_contract. Root owns native runs.

## Task 3 — Persistent per-asset acceleration

Replace the single combined acceleration structure with stable per-asset
world-space meshes and a small nearest-hit dispatch table. Keep flattened
scene geometry as an independent full-rebuild oracle. Asset keys identify an
evaluated object/instance, independent of aggregate offsets, time, transform
and topology. Exact local topology identity controls rebuild versus refit.

Unchanged assets retain their device geometry and mesh across fluid updates.
Camera, world and shading changes do not rebuild geometry. Position-only
changes may refit only after prior readers retire and exact connectivity,
active counts and order match. Equal-count connectivity changes rebuild.
Every asset participates in nearest-hit traversal with correct per-face
material offsets. Empty/removal/addition and instances remain valid.

Accepted resource state must not pretend an old digest remains valid after a
failed in-place refit. Invalidate/quarantine changed resources and force a
safe replacement before another dispatch. Update synchronous and asynchronous
launch boundaries and close/abort together. Do not change optical sampling.

Owned files: `fspt/contracts.py`, `fspt/backend.py`, `fspt/async_backend.py`,
`fspt/kernels.py`, coordinated asset metadata additions in
`blender/fspt_scene.py`, new focused tests and one native acceptance script.
Do not edit the ongoing viewport/bridge work. Root later wires the new script
into the existing gate after Task 2 releases that file.

Acceptance: test the reuse/refit/rebuild decisions, failed replacement/refit,
empty/removal and instance identity; real CUDA must match the flattened
reference pixels across static+fluid nearest hits, materials and topology
changes. Report allocation/build/refit counters. Performance remains a later
matched-budget result even when structural work is reduced.

## Task 4 — Bounded GPU snapshot consumer

Build on the already qualified final producer. Two owned geometry slots must
count active, candidate and retired generations against the same byte budget.
Admit before optional materialization; a full pool drops capture without
blocking the solver. Source ownership retires at copy completion; destination
ownership retires only after every reader. Failed fences retain ownership.
Connect only after the M3 asset interface is stable. Unsupported modifiers,
normals, attributes or routes require explicit evaluated fallback, preserving
native outputs and stateful Waves/Sheet Detail obligations.

Root owns this slice unless delegated separately. A partial implementation
stays explicitly unqualified and disabled until native parity/lifetime tests
and integration pass; do not claim M2 merely from its producer test.

## Task 5 — Qualification, versioning and delivery

Run the existing production gate on the final frozen source, with actual
reviewed distribution and verifier binding. Preserve its performance budget,
storage hooks and automatic GPU routing. Package with the existing builder,
freshly install and test every claimed route. Obtain independent optical and
lifecycle reviews of the exact final candidate, disposition findings, update
the existing Pages and draft PR, and confirm a restorable cloud backup.

Root will choose further work from observed blockers, keeping the last 15
minutes for packaging/delivery and preserving unfinished source explicitly.

## Rulings and progress

- Continue the recovered isolated branch; no duplicate worktree.
- User authorized autonomous development and backups. Existing design answers
  routine implementation choices, so no repeated approval exchange is needed.
- P0 closes contract/reconciliation/feasibility; M6 measured targets stay open.
- Historical cloud-only drafts were lost; snapshot new drafts into owned
  native artifacts between commits so recovery does not depend on one host.
- Task 1 complete, commit 1995a754; three native terminal receipts independently
  verified. Tasks 2–5 continue within their distinct acceptance scopes.
- At 5f6d83ad, the asynchronous evaluated route passed native CPU, source-render,
  edge, handler, CUDA backend and 24-state live checks. GUI lifecycle acceptance
  exposed harness launch/instrumentation defects; its repairs and rerun remain
  separate from the passed live evidence. No realtime qualification follows.
- M3 review found that per-asset traversal and BVH refit change equal-distance
  hit winners in Warp 1.19. The original candidate and failed probes are retained.
  This session first implements a bounded subset: reuse aggregate geometry
  buffers and unchanged upload spans, with a fresh aggregate BVH on every
  geometry change. Per-asset acceleration/refit and the full M3 target stay open.
  No optical tolerance or compatibility target changes.
- The two-slot device surface pool passed independent host review and is ready
  for native lifetime/parity qualification. It is not yet an enabled renderer
  route. Fresh one-shot destination allocations are counted; backing-array reuse
  and total latency improvements are not claimed.
- Add an authoritative completed-solver-state timestamp before optional output
  work, bound to actual solver/epoch/frame/revision/time. This supplies measurement
  provenance; it does not change presentation eligibility or qualify freshness.
- Dropbox confirmed the 41,116,279-byte 00:23 UTC checkpoint at 5f6d83ad with
  matching content hash. It contains P0 closure, native live evidence, the
  committed asynchronous route and explicitly unfinished component drafts.
- The resumed 04:01–09:01 UTC session retains the P0 contract. Native aggregate
  reuse, 50 GUI lifecycle cycles, solver completion-clock provenance, 24 changing
  asynchronous live states, bounded surface-pool controls, ordered GPU consumer,
  and inactive cold shared producer have passed at separately recorded source
  commits. These results do not establish the full normal gate or real-time targets.
- The optional runtime, per-device route broker and checked main-thread Final
  handoff are implemented behind a process opt-in. Failed owners remain strongly
  retained and admission stays blocked until checked retirement. Independent
  reviews reproduced and corrected zero-byte generation, cancelled-callback,
  failed-close ownership, first/pending edit and late-activation faults.
- Coalesced native/manual fluid updates lack an exclusive owner observation
  boundary. Such batches latch evaluated fallback for the view lifetime; later
  publication alone cannot clear it. The live direct-GPU milestone remains open.
  No extra dependency-graph flush is placed on the solver's critical path.
- The 91,015,649-byte checkpoint at 41771aa0 is present in Dropbox. A separate
  05:30 UTC source/review draft checkpoint is confirmed, and ccf14086 was pushed
  without force to the existing experimental branch. Integrated native route and
  Final qualification, full normal gate, final reviews and package installs follow.
