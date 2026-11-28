# FSPT asynchronous viewport implementation and qualification

This source slice separates Blender capture, CUDA submission, completion polling
and viewport presentation. It implements the evaluated-mesh route and uses the
existing checked asynchronous CUDA backend. Native qualification owns any
runtime pass; host controls alone do not qualify GPU behavior, live throughput,
full liquid appearance or an installed package.

## Qualification ledger

### Sampling correction after independent review

The original sampling key added seed and absolute sample number. That made
adjacent seeds share almost every sample stream in a long render: seed 31/sample 1
and seed 32/sample 0 were identical in the retained Warp CPU control. The corrected
key hashes seed/pixel first, then the absolute sample index, with explicit 32-bit
wrapping. An independently written flat-mesh primary-ray oracle uses the same
documented sample-zero mapping without calling the production helper.

Actual Warp CPU tracing now separates the formerly identical pair while keeping
same-seed split and single-dispatch progressive renders bitwise equal. Boundary
controls include pixel 16,777,215 and sample 4,095 against an independent integer
PCG reference. The independent review observed 2,525 of 32,768 glass pixels differ
across the formerly aliased pair; normal-incidence reflection and transmission
weight remained consistent with their analytic controls. A finite 32-bit state
still permits collisions. This fixes exact systematic reuse; it does not certify
statistical independence or the roadmap's final optical confidence requirements.
All image receipts below remain bound to their original source; successor native
pixel and GUI qualification are required after this sampler change.

### Retained native and host receipts

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


## Frozen candidate status, 2026-11-28 UTC

At `5c91c8188e5e148f0218a596ed64de4c087e1cc5`, the independent optical review
found no blocker within the bounded Diffuse, smooth Glass, Emission, constant
world and mesh-emitter subset. The exact-source Warp CPU diagnostic separated
the former adjacent-seed alias in 2,525 of 32,768 pixels and kept same-seed split
and single-dispatch samples bitwise equal. The previous native images keep
their original sampling identities. Broader optical and statistical confidence
remain to be established under the frozen contract.

The same source verified the native broker and the controlled integrated
device-publication/worker-Final route in a real Blender window. That route used
two controlled four-sample frames and a real eight-sample Final. It verifies
the specific source and retirement boundaries described in
[DEVICE_VIEWPORT_QUALIFICATION.md](DEVICE_VIEWPORT_QUALIFICATION.md).
Ordinary rendered-region live uses the conservative evaluated fallback when
Blender coalesces fluid geometry updates. The controlled device fixture supplies
no new 240-state/600-second soak, compositor timing, authoring fidelity or
steady-state performance result.

The normal source gate that followed these component proofs did not qualify.
Two subsequent 0607 production attempts stopped on unattributed exclusive
broker contamination. Historical 50-cycle and 24-state rows retain their
executed source identities. [Exact status and attempts](CURRENT_QUALIFICATION_STATUS.md)

## Exact 0607 standalone native viewport and Live trials

The ordinary evaluated route (no device-surface opt-in) ran on Windows source
`0607daad9354b872c7530277371b1d998443c394`. The static asynchronous viewport
completed all 50 cycles and the pending-work Final controls, exit 0, under a
clean exclusive lease. The standalone Live case recorded the initial 240
distinct audited geometries and completed images, plus fixture controls, then
exited 1 in the 600-second soak. Seven new soak states and six changed soak
images were reported before the 10-second progress guard fired. Its separate
exclusive lease released clean.

All 240 initial rows had RETAINED presentation labels. The initial stage itself
had 22.910 seconds between accepted solver frames 15 and 143, during playback
and before the scripted pause. The soak showed ongoing solver readiness and
expired retained imagery. A later bounded 0607 diagnostic isolated
capture-before-first-draw in 47 observed late cases and informed the
ce6afc9a scheduling change. Both successor lanes stopped before broker admission and produced no
acceptance data or new screenshots. Timing comparison requires an admitted
run. [Diagnostic and successor scope](CURRENT_QUALIFICATION_STATUS.md#0607-failure-and-the-ce6afc9a-change)
Distinct-image count, static lifecycle coverage and sustained Live each keep
their own result. The historical draw-entry timing remains separate from a
completed-image presentation measurement. [Exact report and case hashes](CURRENT_QUALIFICATION_STATUS.md)

### Original 0607 desktop captures

All four original desktop PNGs are 2560 × 1351. Their byte hashes were
rechecked. Reduced inspection views and native-resolution status crops
confirm the visible content below. They are in `async-live/acceptance-originals/` under the native
run directory indexed in [current qualification status](CURRENT_QUALIFICATION_STATUS.md).
The observer records accepted draw pixels before the screenshot operation.
A corresponding NPY and observer row do not certify the later screen contents.

| Original PNG | SHA-256 | Visible content and caption |
| --- | --- | --- |
| `paused-refinement-screenshot.png` | `bcef118ead3e7feaa352d3e0a08f11ced7df49ddc522fa71b8476590455f760f` | **Provisional paused evaluation.** The camera-view Dam Break cove and blue fluid are visible, with coarse, noisy shading. Timeline 146, solver 145, evaluated route, 525 × 298 render buffer at 25%, eight spp, six bounces, seed 31. This is evidence of the paused visible result. |
| `live-240-screenshot.png` | `86c5bfaa67f0e104ce58354b1236b9e7d407eadb574e39611e5d0d94df9a4eaa` | **Live progress diagnostic.** Purple waiting framebuffer displays “FSPT: Retained image publication age exceeded 250 ms”. The desktop shows timeline 761. This captures expiration after the accepted 240th observer row; it does not show that fluid image. |
| `reopen-live-screenshot.png` | `e7d91178a8256449d4afb1e0a9a1ea8b59e3ca3e06645f531b1e4e8e295100f3` | **Reopen diagnostic.** Purple waiting framebuffer and the same retained-age message, timeline 4. The earlier accepted reopen buffer remains a separate NPY. |
| `soak-000-screenshot.png` | `d548781c563d27ecade5acc8599f55bfcd1a5a26fb0c8fdeaf8aac9699c4272a` | **Failed-soak diagnostic.** Purple waiting framebuffer and the same retained-age message, timeline 5. The earlier raw soak sample has separate observer telemetry. The PNG records visible expiration during the failed lane. |

Only the paused capture visibly shows the rendered fluid. The three waiting
captures explain the failure boundary and cannot illustrate successful live
display. Their contents do not invalidate the prior accepted draw observations,
and neither evidence type qualifies sustained progress, presentation timing or
optical quality.
