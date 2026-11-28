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

The first integrated native broker candidate, `a113b5dd51ee9d60ccd9d82f45aa2ebe5aa46f72`,
failed at the two-facade transition. The retained report is
`viewport-backend-a113b5dd51-20261128T055117396702Z/viewport-backend/acceptance.json`
(SHA-256 `133d08eb2046786fb1b697a9384993cb87acbb3ac4713eeeaad6aaaad50b170c`).
An artifact-only timing probe on unchanged source measured 15.002368 seconds
inside an idle peer's close while another stream was held by the verifier.
Its failed report, SHA-256
`44ee808ea2e8c181b4c56e8bd216ff73c758b21811cba8d2982252b51e7fe6d9`,
is retained under `viewport-diag-a113b5dd51-20261128T055602619084Z/viewport-diag/`.
The native timeout was not extended or reclassified.

The reviewed successor polls the full facade cohort before destructive cleanup
and holds device admission through reclamation. Ordinary view closure only
withdraws intent and leaves the delegate strongly owned. Explicit broker close
drains all pending delegates and refuses success while any delegate remains,
including evaluated backends outside the optional geometry ledger. Native stream
destruction is explicit and retryable: a failed destroy retains its raw handle,
wrapper and failure state, even when other Python references still exist.

Lifecycle cleanup retries orphaned failed views and failed Final owners. A
poisoned pool permits one checked recovery pass; active peers and denied admission
preserve that unused attempt. A typed result distinguishes a discarded job whose
drain succeeded from an unclassified cleanup failure. Only checked bridge close
with zero bytes and zero generations clears the block. Independent host
controls and the successor native receipts below cover these repairs.

Host runtime/render regressions, focused adversarial review controls, Ruff and
source manifests are retained with the candidate. Native broker/worker acceptance
is now a required report-producing production-gate lane. It uses real CUDA
packets, independent geometry/image oracles, a real worker, delayed CUDA events,
multiple facades, zero-byte generations, failed-read recovery and guarded actual
Blender static capture. Its supplied selection binding is explicitly a component
fixture, not a claim of ordinary runtime/GUI publication provenance.

The full production gate also requires the controlled device-publication and
GUI-worker Final verifier. It must retain two advancing native Session GPU
publications, independent GPU/evaluated geometry observations and changing images,
then invoke a real Blender Final while a viewport job is pending. The engine's
own main-thread timer must retire its owned aggregate before Final allocation,
and preserve admission through checked Final close. This fixture does not claim
ordinary rendered-region GPU eligibility, display timing or performance.

The controller grades required cases, source and runtime identities, ordered
ownership receipts and distinct nonempty originals. It rehashes originals at
manifest time. Active root test configurations are part of the verifier
fingerprint, including all pytest 9 discovery names; an ignored untracked config
cannot silently replace the reviewed test options.
The lightweight add-on registration path remains Warp-free. The optional host
snapshot entry checks the public exact solver owner before extraction; it
preserves the existing private read pin and native publication identity.

### Exact successor evidence

The repaired broker ran on `383a6e5e41d8aca6e03a4aabb1966084731b8d2c`.
Its two-facade held-stream, worker, poison recovery, zero-byte generation and
five checked stream destructions passed. The run as a whole failed when the
fixture handed Blender's static `Object.to_mesh` a Python graph guard.
The retained failed report is
`ownership-383a6e5e41-20261128T062919970689Z/broker/acceptance.json`
(SHA-256 `7d8c3e9e265e1c2870a90e996fc1404a4f63d32b7452120558483aa62e6a1247`).
The fixture now passes the native depsgraph to that static RNA call. The
fluid geometry guards and numerical comparisons remain active. Two host
controls reproduced the type error with the forwarding removed.

Both successor native lanes returned **0** and **VERIFIED** at
`5c91c8188e5e148f0218a596ed64de4c087e1cc5`. Their common directory is
`artifacts/fspt-p0/bridge-qualification/ownership-5c91c8188e-20261128T064212158236Z/`.
The environment is the reviewed Windows Blender 5.2 build `fbe6228777e7`,
Warp 1.19.0 and RTX 4080 SUPER on `cuda:0`.

| Lane | Acceptance JSON SHA-256 | Terminal JSON SHA-256 |
| --- | --- | --- |
| `broker/` | `cf8dc504a3f2b484c56496f062b7a221de8f2e973088bfc48ddc7ee8bfbd2cf2` | `a297e8321af05c121aadd7b082ca60c873cda6c4cfcbdfb2de7d14036bf65aa8` |
| `device-viewport/` | `5122cdfce0d34f86564d8b978b814e092ee199fb91981e86b3d5551d376e1d49` | `0407353bbaf7ccc42990402894ef769479242b3e6bcd26b6307293498fb65e79` |

- **Broker.** The fixture observed two generations at peak, five successful
  checked stream destructions, stale work rejection and poison recovery. An
  empty pending packet held one generation until its actual copy fence
  retired, even with zero reserved bytes. All three fluid guard negative
  controls fired, and guarded capture logged zero bulk fluid reads. The
  guarded actual Blender scene used an explicitly supplied selection. An
  independent reopen of both retained NPZs found zero error on the compared
  positions and pixels against the fixture oracles.
- **Runtime and worker Final.** Actual Session steps produced GPU publications
  at timeline frames 2 and 3, solver frames 1 and 2, and state revisions 5 and
  9. The two four-sample 128×96 images changed by mean absolute RGB
  `0.00647503277`. The independently read GPU and evaluated positions and
  indices matched in both originals. The controlled `ViewportState` owned an
  open aggregate and a pending trace when the real `INVOKE_DEFAULT` returned
  `RUNNING_MODAL`. The worker drained the trace. The engine's main-thread
  timer then retired 7,163,956 explicit geometry bytes and one generation,
  returned an identity-matching zero-byte/zero-generation receipt and detached
  the view before Final backend allocation. The admission hold lasted through
  checked stream destruction (result 0), then released. Final completed at
  timeline frame 3 with eight samples, and the reopened 128×96 EXR was finite
  and nonconstant. Cleanup recorded no surviving device owner or hold.

The successor run verifies the named component and controlled integration
boundaries. An independent evidence-hygiene rebind confirmed that exact
`0607daad9354b872c7530277371b1d998443c394` carries the same product and verifier
inputs. The native receipts remain at their executed source, `5c91c818…`.
The ordinary rendered viewport retains evaluated fallback for ambiguous fluid
updates. The 0607 50-cycle evaluated viewport lane passed and its standalone
Live soak failed after the 240-image functional prefix. The current ce6afc9a
scheduling change has its own targeted host result. Both native GUI lanes stopped before
broker admission. The normal full gate remains blocked after two
contamination decisions with unproven PID ownership. Provisional and installed
archive evidence retains its own executed source and runtime scope.

Frozen real-time, optical and compatibility requirements remain in force,
including 24 fresh states/s, simulation/wall ratio, native-1080p throughput,
solver-completion-to-presentation age, 1.05 solver overhead, combined VRAM, full
Principled/attribute/optical semantics, all export passes and Linux GPU runtime.
[Current status and receipt identities](CURRENT_QUALIFICATION_STATUS.md)
