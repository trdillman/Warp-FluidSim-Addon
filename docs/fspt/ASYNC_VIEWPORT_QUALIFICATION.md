# FSPT asynchronous viewport implementation and qualification

This source slice separates Blender capture, CUDA submission, completion polling
and viewport presentation. It implements the evaluated-mesh route and uses the
existing checked asynchronous CUDA backend. Native qualification owns any
runtime pass; host controls alone do not qualify GPU behavior, live throughput,
full liquid appearance or an installed package.

## Qualification ledger

| Candidate / lane | Observed result | Evidence boundary |
| --- | --- | --- |
| Cloud async candidate and gate inventory | 196 render/inventory tests passed before the final FIFO controls; 75 focused controls passed after that correction; full configured Ruff passed all 13 manifest files | Host control logic only. Final two-file FIFO change preserves active GPU leases and withdraws obsolete queued intent. |
| Native `5f6d83ad4894ffc0d673a4a734129048f05d19ee`, first all-lane launch | Controller failed before any test ran because its artifact-script import path omitted the repository root | `artifacts/fspt-p0/bridge-qualification/all-5f6d83ad48-20261128T001015849339Z/status.json`; unchanged source-before/after fingerprints. This is neither a product pass nor a renderer failure. |
| Native `5f6d83ad4894ffc0d673a4a734129048f05d19ee`, V2 CPU/source/handler/backend lanes | Bundled-Python CPU lane: 764 passed, 5 skipped, 3 deselected in 61.35 s; source render, edge render, handler persistence and asynchronous CUDA backend all verified | `artifacts/fspt-p0/bridge-qualification/all-5f6d83ad48-20261128T001143730598Z/`; the controller import correction changed only its artifact wrapper. Actual GUI lifecycle and live-image lanes remain separate results. |
| The same V2 source, asynchronous static GUI launch | Failed before the verifier ran: its `--cycles` option collided with Blender's built-in Cycles parser | `async-viewport/stderr.log` in the V2 evidence directory. The installed `cycles/engine.py` parses all script arguments with abbreviation enabled for `--cycles-print-stats` and `--cycles-device`. The correction uses `--lifecycle-cycles`; this attempt provides no GUI lifecycle proof. |
| The same V2 source, bounded asynchronous live acceptance | Verified: 24 independently audited changing fluid geometries and 24 changing completed images; pause/refinement, frozen output rejection, publication fault controls, reset/reused frame and save/reopen passed | `async-live/acceptance.json` SHA-256 `bab0bd9db0de995ecaca0c4d9551cf0dd3e79462b5d53316006ee87c9df6c7e3`; terminal code 0 and exact success marker. Verifier SHA-256 `e88dedf7923a4b2809df525649ceaaf20e378e161fff242faa93bb736dce7a6a`. Source-before/after fingerprints were identical. |

The 24 accepted live images retained their actual earlier source identities.
Their observed publication-to-draw ages ranged from 146.13 to 247.74 ms, with a
241.98 ms median. The minimum successive mean absolute RGB difference was
0.00125979. The paused image refined to eight samples. These are conditional
observations from the accepted functional samples, with verifier overhead; they
do not establish throughput, the distribution of all producer states, or the
authoritative solver-ready freshness requirement. The 240-state sequence and 600-second soak still require qualification.
The later static GUI lifecycle qualification below supersedes its earlier open status.

## Callback and lifetime boundaries

`view_update` observes the current scene/view layer and bounded camera/settings
metadata. A dirty view obtains device admission before extracting an owned
`SceneSnapshot`. A pending job prevents another capture allocation for that view.
An unsuccessful capture retains no dependency graph or evaluated Blender object.

The application timer polls an owned CUDA completion once without requesting a
wait. It submits at most one sample quantum, using each view's current immutable
snapshot and newest camera request. Per-device admission excludes capture and
submission while final rendering takes its exact snapshot. An already admitted
capture/tick finishes before final admission; final then drains only the selected
view backend, even if the view migrates to another device concurrently.

`view_draw` reads the small camera/settings request, validates an independently
owned completed frame and uploads its texture only when that frame changes. It
does not extract geometry, build a CUDA backend, trace, collect pixels or wait for
CUDA. Display shader and blend state are restored on binding, drawing and
unbinding errors. Camera navigation can generate a draw without `view_update`, so
the bounded ray request is also refreshed there.

Final rendering uses a new backend/history, exact source publication validation
and the same device arbiter. Final cancellation and failure retain checked
backend cleanup. Persistent load/undo/redo callbacks retire view state; weak
scene subscriptions avoid keeping retired Blender scenes alive.

## Current and retained images

Mutable surface capture and final rendering still require exact publication
identity. Completed viewport pixels may retain an earlier LIVE state only when:

- Scene, domain/output owner, epoch, channel, source, output object and mesh
  identity remain the same.
- The new publication advances the actual solver frame, state revision,
  simulation time and source timeline frame.
- Both publications remain visible and nonempty, and the current source passes
  the ordinary owner/readiness checks.
- No more than 250 ms has elapsed since the earlier surface publication
  completed.

Retained pixels keep their original submitted source labels and show a retained
update-pending label. A paused current image remains valid and can refine.
Same-state rebuilds, static/material changes, source switches, reset, quarantine,
pending render-input synchronization and visibility/empty transitions do not
inherit this permission.

`presentation.publication_age_seconds` uses the host monotonic publication
completion clock. It is a lower bound on authoritative solver-state age, because
optional surfacing precedes that timestamp. The authoritative Session completion clock is now carried independently as
`state_ready_at`. A native 24-state verifier measured completion to independent
draw entry, not display presentation. It therefore does not close the P0
completion-to-presentation freshness gate.

Surface advancement and unrelated scene edits use separate invalidation
revisions. The adapter recognizes only matching published surface Object/Mesh
updates; other geometry/material updates remain invalidating. A surface transform
or material-slot change is independently part of the scene input identity.
Expiring an old displayed frame clears that frame without invalidating a newer
pending job.

The device arbiter's queued submission intent is withdrawn when a view reaches a
reduced sample goal or becomes dirty and needs another capture. An active GPU
lease remains owned until collection or checked teardown confirms retirement.

## Executable controls

Host tests cover delayed completion, rapid publication, exact captured source
identity, sample refinement, camera/resolution/device changes, shared admission,
final/capture and final/collection interleavings, device migration during final
retirement, stale queue entries, capture failure, failed retirement retries,
shader state restoration and owned-history age/readiness boundaries.

`tests/acceptance/blender_fspt_async_viewport_acceptance.py` is a real-window
static-scene gate. It audits callback ownership, splits the actual viewport,
compares two camera images on a common normalized grid, reassigns the World,
starts final rendering while a viewport job is pending, and performs 50 pending
job close/reopen cycles. It retains raw images, screenshots, retirement receipts
and bounded callback statistics. Those statistics are functional diagnostics,
not a fixed-quality performance result.

`tests/acceptance/blender_fspt_async_live_acceptance.py` independently reads the
evaluated fluid at the capture boundary and stores a bounded immutable audit
receipt. It joins that receipt to `presented_snapshot` in the actual completed
draw. It inherits the reference verifier's changing-geometry/image, pause/refine,
frozen-output, mutation quarantine, synchronization warning, reset/reused-frame
and save/reopen controls. The ordinary GUI playback path supplies the simulation.
The full gate requests 240 changing states and a 600-second live soak; the initial
bounded qualification can request 24 states without the soak. Soak progress
requires real source advancement and rejects changed geometry with frozen RGB.

The original strict draw-current verifier remains a manual synchronous-baseline
probe. Its immutable native reference evidence is separate from the asynchronous
candidate. Production-gate inventory checks require both new asynchronous
verifiers and their advertised counts.

## Remaining optimized-route work

This slice still captures evaluated host geometry per view, uses the existing
software-BVH backend and copies the completed image through pinned host memory.
Its backend currently copies after each submitted sample quantum. The optional
device components described in `DEVICE_VIEWPORT_QUALIFICATION.md` add a bounded
owned surface pool and shared explicit-geometry accounting. They do not establish
the fully optimized live GPU viewport, a total renderer/solver VRAM budget,
display-cadence-only transfer, graphics interop, the target solver overhead or
presentation freshness, full materials/optics, or the complete P2/P3 gates. Those requirements remain in
`P2_LIVE_BRIDGE_SPEC.md` and `P0_ACCEPTANCE_CONTRACT.md`.


## Continued native qualification, 2026-11-28 UTC

All rows use the retained Blender 5.2 build `fbe6228777e7`, Warp 1.19.0 and
RTX 4080 SUPER. The qualifying controller retained matching source-before and
source-after identities. These are separate source revisions and scopes; they
are not a substitute for the full normal production gate on the final candidate.
Paths below are under `artifacts/fspt-p0/bridge-qualification/`.

| Source / lane | Observed result | Retained report / SHA-256 |
| --- | --- | --- |
| `1da2739f6c3c3bf50f5d3736e953acf7b579c865`, aggregate geometry reuse | Native CUDA passed; unchanged static ranges avoid copies, position-only updates avoid index uploads, and five coincident-hit ordering controls passed. Each new aggregate still builds one fresh BVH. | `geometry-reuse-1da2739f6c-20261128T040553483789Z/geometry-reuse/acceptance.json`; `611077af651f131316f1dafcf6becc86c6fcaf26d354074172c140436681d7cd` |
| `fe9a9627a638baa2ccc7cbee07de951f1f6325c6`, asynchronous static viewport | Actual GUI passed 50 close/reopen cycles with pending work, two views and Final rendering. The harness retains a naturally submitted job until observation; it does not fabricate CUDA completion. | `viewport-fe9a9627a6-20261128T042145324041Z/async-viewport/acceptance.json`; `fadbd5a665b47d2d41417f820755a0b3f2ad129acfcea8307d475fa66d026a6f` |
| `2893409902e2c1e41e134a0ab8a2ed42dbb0b70d`, solver completion clock | Native same-frame solve, failed completion and unowned-session controls passed. The solver pointer and publication UID remain distinct identity namespaces. | `state-clock-2893409902-20261128T042503214331Z/state-clock/acceptance.json`; `607def3c3b550a2550b37e22aaabd7efc97a9862694b79c0f6fc4535eb45d2c8` |
| `feb325a6e3f5c7c01764da4f925ea3aae38758bc`, asynchronous live 24-state sequence | 24 changing geometries and completed images passed. Solver completion to independent draw entry: 147.8225 ms minimum, 255.48215 ms median, 265.0852 ms maximum. This is not a presentation endpoint or a freshness pass. | `live-feb325a6e3-20261128T043303160330Z/async-live/acceptance.json`; `1397e00a1380d3954a97ef82cee58e6d6ac853062f3ab0ef2655f26ddab61211` |
| `a243aa0d164c60aef776628759a024886dd8a011`, earlier static viewport verifier | Camera, material, resize and refinement controls passed at the normal 120-second deadline after explicit area redraw observation. | `static-viewport-a243aa0d16-20261128T043746986477Z/static-viewport/acceptance.json`; `38a92242b81839e8863263bf40531e51cf94d19bb38e56b8c73c1e4ea852cbea` |

The initial static/GUI observation failures and report-wrapper failures remain
in their original attempt folders. Their later corrections do not turn those
failed attempts into passes. P0 completion is recorded separately in
`P0_RECONCILIATION.md`; P2/P3 and the full real-time contract remain open.
