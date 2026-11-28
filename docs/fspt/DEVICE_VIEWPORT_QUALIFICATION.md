# Optional FSPT device geometry and lifecycle qualification

The default viewport remains the evaluated-mesh asynchronous renderer. The
process opt-in `FLUIDSIM_FSPT_DEVICE_SURFACE=1` enables development components;
it is not a real-time or visual-quality acceptance switch. P0 is complete under
`P0_RECONCILIATION.md`. The full P2/P3 live GPU viewport remains open.

## Implemented boundaries

The demand coordinator records weak viewport demand without creating CUDA
resources. A compatible cold native surface publication may share its already
required sparse extraction with an owned GPU surface copy. Publication requires
both checked copy completion and the exact acknowledged native host receipt.
A recoverable synchronization retry may acknowledge that same pending receipt;
a different source, owner, epoch or mesh cannot do so. Forced Final publication
keeps the existing host route. Warm host geometry, including stateful Waves,
explicitly falls back before a duplicate extraction.

GPU eligibility uses a separate `GPU_FINAL` channel. Its publication generation
is independently minted and paired with the pool key by an immutable selection.
The source carries the authoritative Session completion timestamp and the first
optional preparation timestamp. Repeated polls do not refresh either age or
publication time. Drawing performs only metadata validation of the selection;
it does not poll, extract geometry, build acceleration structures or wait.

The pool and aggregate backend share a ledger for explicit position/index/status
arrays, including retained generations and duplicate aggregate copies. A
zero-byte empty packet still occupies a generation. Route transitions require
both zero bytes and zero generations after all readers retire. The development
ceiling is 256 MiB and at most two geometry generations. This ledger excludes
BVH internals, allocator caching, images, shading data and the solver: it is not
a combined-VRAM measurement or the roadmap's 80 percent VRAM acceptance.

The adapter preserves static object ordering and global material/face ordering
and retains immutable static captures when their caller-owned revision is
unchanged. The consumer builds one aggregate BVH for each changed geometry;
per-asset acceleration and refit are still open. A real render worker may collect
an owned job, but pool/ledger retirement is deferred to the main publication
thread. The device-wide route broker controls all optional per-view delegates
and does not admit evaluated allocation beside retained GPU geometry.

## Evaluated fallback for ambiguous Blender edits

Blender may coalesce native publication and manual fluid edits into one
Object/Mesh dependency-graph update. Object identities, counts and a publication
notification do not prove exclusive writer provenance. Every such ambiguous
fluid update therefore hard-invalidates static/pixel history and latches the
viewport to evaluated geometry for its lifetime. This applies on the first
subscription and while a native receipt is pending or quarantined. A later
publication or a repeated capability refresh cannot clear the latch.

Consequently, these components do not establish a qualified ordinary live GPU
viewport. A separately proved native observation boundary is required before
that route can be admitted for ambiguous update batches. This implementation
does not force an extra dependency-graph flush on the solver's publication path
and does not add a repeated CPU fluid-mesh comparison as an unmeasured substitute.
The evaluated fallback preserves manual edits. Existing material, modifier,
transform, cache, instance and unsupported-channel fallbacks remain explicit.

## Checked Final handoff and failure ownership

Interactive Final rendering runs on a Blender worker. A persistent main-thread
service, independent of viewport sample goals, handles an identity-bound cleanup
request. The Final coordinator first suspends viewport submission and drains its
owned pending jobs, then releases its admission lock before the worker waits.
The main thread detaches every matching viewport backend, retires the runtime
broker/pool, and returns the exact device/request receipt only after observing
zero reserved bytes, zero generations and no old viewport backend.

Only then may Final construct its fresh evaluated backend. The hold lasts through
checked Final close and exit from coordinator suspension. Timeout, cancellation,
a wrong receipt, an independent failure hold or a nonempty ledger prevents Final
allocation. Unstarted timeout callbacks are removed so they cannot retain owners
indefinitely. A cleanup already running retains admission until it returns.

Failed viewport closes remain discoverable through a strong recovery registry,
even after the last external engine reference disappears. Failed Final closes
retain both backend and admission hold, including interruption by BaseException.
A failure from the default-off route receives a separate suppression-only token;
that token cannot certify a zero-owner receipt. Multiple failed owners have
independent tokens. Lifecycle retry releases a hold only after checked cleanup.
Changing the process opt-in retires all previous same-device route owners under
one coordinator admission before new demand can allocate resources.

## Native evidence before the integrated route candidate

All following reports used Blender 5.2 build `fbe6228777e7`, Warp 1.19.0 and an
RTX 4080 SUPER. Their controllers bound matching source-before/source-after
identities. Paths are under `artifacts/fspt-p0/bridge-qualification/`.

| Exact source / scope | Observed result | Report and SHA-256 |
| --- | --- | --- |
| `41771aa0ea9de788f97c8764dc68bda524f47421`, device consumer | Actual CUDA aggregate positions/indices/materials and pixels matched an independent flat-mesh oracle. Real ordinary host output was exercised with and without Waves/Sheet Detail. Checked late reads, pressure and fault quarantine retained ownership. | `device-consumer-41771aa0ea-20261128T045232958046Z/device-consumer/acceptance.json`; `d4282f5be86aa00acf13941c2db19095d477388e6e221d9d6477c58f5ebef404` |
| Same source, surface pool regression | Two owned snapshots, a pressure denial and checked last-read retirement passed. Explicit peak arrays were 57,216 bytes in this fixture. | `surface-pool-41771aa0ea-20261128T045434918745Z/surface-pool/acceptance.json`; `425571af112edad22a8a9fb4bc79833ab0907f6eaeefb18f8ca751284d3313ad` |
| `ccf14086b336dffa6cbb5bab9241ce4234bc88c4`, inactive producer | Actual cold CUDA extraction shared native host geometry; no-demand work stayed absent; copy/native-commit ordering, generation pressure and failed-record quarantine passed. Warm Waves fallback performed no extra extraction. | `device-producer-ccf14086b3-20261128T051451760848Z/device-producer/acceptance.json`; `22fb5e27e033e3ba8020c25ffc8ff4181002f0429044633a758206204d562672` |
| Same source, actual Final thread probe | EXEC used the main thread. INVOKE used a worker and successfully waited for a real main-thread application timer before completing FSPT Final. This proved the rendezvous mechanism, not active device geometry retirement. | `render-thread-v2-ccf14086b3-20261128T051605887330Z/render-thread/acceptance.json`; `6250fee5c8f0ff4cd947f95aa32e32654ae69cf64bf06da9d4ffdcff380fcd9e` |

The first thread probe's product render succeeded but its report wrapper rejected
a flattened runtime record. That failed wrapper attempt remains retained. The
corrected report uses nested runtime identity and a separate actual timer wait.
No earlier failure is relabelled as a pass.

## Integrated qualification status

Host runtime/render regressions, focused adversarial review controls, Ruff and
source manifests are retained with the candidate. Native broker/worker acceptance
is now a required report-producing production-gate lane. It uses real CUDA
packets, independent geometry/image oracles, a real worker, delayed CUDA events,
multiple facades, zero-byte generations, failed-read recovery and guarded actual
Blender static capture. Its supplied selection binding is explicitly a component
fixture, not a claim of ordinary runtime/GUI publication provenance.

At this document's initial source checkpoint, native integrated runtime/Final
qualification and the full normal production gate are pending. Their exact
candidate reports must be added before claiming those scopes. Existing limited
functional proof does not satisfy 24 fresh states/s, simulation/wall ratio,
1080p/1M-triangle throughput, completion-to-presentation age, 1.05 solver overhead,
combined VRAM, full Principled/attributes/optics, all export passes or Linux GPU
runtime qualification. No roadmap target has been reduced.
