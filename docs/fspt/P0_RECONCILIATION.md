# FSPT P0 reconciliation and bounded feasibility record

**P0 / M0 complete, 27 November 2026 (America/Toronto).** The architectural
lead reviewed the frozen contract, source/repair ownership, actual runtime
capability matrix and baseline schedule, and independently reopened all three
native acceptance reports and terminal receipts below. Each report hash
matched this record and each terminal returned 0. This closes reconciliation,
contract freezing and bounded feasibility. Later optical, compatibility,
installed-package and joint performance gates remain separate requirements.

Evidence in this record is captured through candidate
`e3cb526324a838cfb762ff4f1b157b0e42c7f286`. A later commit does not inherit an
unchanged-source receipt merely by sharing the branch name. The final delivery
manifest must name each tested source and the exact later delta.

## 1. Recovered source and ownership

The active isolated worktree is
`C:/Users/Tyler/projects/fluidsim-fspt-20261125`, branch
`experimental/fspt-p1-20261125`. The renewed session recovered checkpoint
`82f08a9b34c0e2971103acb87ed6d0a4cb63bb89`; its committed work survived. The
restored cloud working copy contained 925 tracked Python/configuration,
documentation and JSON files. Uncommitted cloud-only viewport/device drafts
from the previous session were not recovered and are not claimed as delivered
code. The canonical dirty checkout and historical evidence remain separate.

| Source | Owned result and status |
| --- | --- |
| `49aa02be60a28cc9f3af06c797a4bcf07c42b850` | Prior P1 checkpoint and retained evidence on draft PR 89. Historical qualification remains bound to its original candidate. |
| `2ecebfcf3aec109b949e8192e061dcd1d4d25383` | Separately recovered device-selection/shared-bounce repair in four FSPT files. |
| `d5c9b5fe58acd2fbcab6fb0e9752d039708e5d24` | Integrated repair. Its stable Git patch ID equals the separate repair: `cac475f92919559fc10a90078d576042e338c89f`. It is already present; no second application is needed. |
| `86be693079423a59496db9b17b2ef00fe7c6c6c2` | Committed surface provenance and evaluated FSPT live bridge. |
| `bb354f67b8e086d254b9d28987743e53325b2678` | Fenced device surface copies and asynchronous CUDA backend primitives. Their presence alone does not qualify the full live GPU route. |
| `82f08a9b34c0e2971103acb87ed6d0a4cb63bb89` | Recovered checkpoint; observes live proof at the completed draw boundary. |
| `362780dd775fc3652651093ca4a19f603ec80294` | Excludes the authoritative designated domain from FSPT geometry, with a contrasting ordinary `NONE` object retained. The prior role-only filter admitted an opaque enclosing domain. |
| `efe61c01c3e60750bd0621adc13788974f905d45` | Observed pause transition fix and the verified 24-state live functional lane. |
| `9609ac0fb823e1968f9d25a9c5f4405da085221e` | Frozen P0 acceptance contract, runtime/RTX probe and actual device-surface acceptance. |
| `e3cb526324a838cfb762ff4f1b157b0e42c7f286` | Corrected bound-instance Blender API inspection; complete runtime/graphics/OptiX lane verified. |

The repair comparison was a read-only `git show`/`git patch-id --stable`
comparison of both original commits. The renewed-session commits retain the
existing simulation-admission, Solver Session, Output Publication, Pending
Settings, Preview Playback and Production Bake ownership in `AGENTS.md`.
Renderer callbacks receive owned states; they cannot step the solver, change
authored quality, or derive solved-state identity from a timeline number.
ST-FLIP remains permanently removed.

## 2. Frozen acceptance authority

The normative requirements are [P0_ACCEPTANCE_CONTRACT.md](P0_ACCEPTANCE_CONTRACT.md)
and [fspt_acceptance_contract_v1.json](../../tests/render/fixtures/fspt_acceptance_contract_v1.json),
contract ID `fspt-p0-20261127-v1`.

| Binding | SHA-256 |
| --- | --- |
| Canonical contract JSON, using `tools.gate.fspt_contract.contract_sha256` | `25635c40bec4a34f4acbd947524f13972d58df37ff1317e76677990c5f905c04` |
| Exact contract file bytes | `c818939af5ed435412c7de815e01e2645f909a257eb2eb9bc81dac5cbdd814fd` |
| `demos/dam_break.blend` | `05e1918423e48fdf22ea702ff5ca2db1c8f7464be865619d382ad5d8118b92f8` |
| `demos/crown_splash.blend` | `f3ee4e00dbcbf003fab15733c8494b4f7932bd1e68d21b7ce3dd58af4f6f46d4` |
| `demos/paddle_stir.blend` | `fa0521b10fa6b0e029ab60e0c2ddcb08ff5f8d7cde9c7c536eea8edf7bb022c3` |
| `demos/chromatic_vortex.blend` | `678c3851daa79d3743b5e1cebf050abba81dbf3f20a94b8697f8db40b30eab0c` |

The contract freezes FIXED, ANIMATED_CACHE and LIVE lanes; fixed geometry at
250,000/1,000,000/4,000,000 triangles; and live populations at
100,000/500,000/1,000,000 particles. Native 1920×1080 profiles use separate
1/2/4-sample Live, 64-sample Paused and 1,024-sample Final settings with a total
12-scattering-event limit. Transmission events are included in that total.
Hidden adaptation, clamping, upscaling and unqualified history reuse cannot
make a profile pass.

The authored-24-Hz joint Live gate requires at least 24 distinct fresh solved
states presented per wall second, simulation-time/wall-time at least 1,
p95/worst age at most 100/250 ms, both D/A wall and p95 step-interval ratios at
most 1.05, and combined peak VRAM at most 80% of observed physical VRAM. These
must pass together with the frozen image, optical, temporal and compatibility
criteria. The separate one-million-triangle fixed-surface target is at least
30 new completed images/s and p95 interval at most 33⅓ ms; 60 images/s remains
a separately reported point. **These are acceptance targets, not results of
the bounded probes below.**

The existing FluidSim 4.0.1 comparison, its pinned reference commits,
eight-trial ABBA/BAAB schedule, guarded four-trial early stop, 0.06 margin,
frames 1–48, host-time ratio, residual checks and explicitly owned exceptions
remain intact. FSPT's interference bound cannot borrow an unrelated solver
algorithm allowance.

## 3. Runtime distribution and actual API capability matrix

The actual probe used Blender **5.2.0 LTS**, build **`fbe6228777e7`**, bundled
Python **3.13.13**, locked Warp **1.19.0**, and an **NVIDIA GeForce RTX 4080
SUPER** on `cuda:0`. Warp reported CUDA toolkit/driver integers 13040/13050.
The graphics context independently reported OpenGL 4.6.0, NVIDIA 621.14.
These observations describe this machine and do not qualify another driver,
GPU, Blender binary, Python ABI or operating system.

The retained official ZIP/checksum and its installed executable/Python members
were independently rehashed. The installed bytes match the retained archive.

| Runtime input | SHA-256 |
| --- | --- |
| Official Blender Windows ZIP | `2d184b626c001692c362291911293b6a297179d618d95e9e9192c3a80318adc4` |
| Published checksum file retained with ZIP | `f35709c2eb91fbb58ebbd354285039df62217e6dbda9c3a4713ec113d728057f` |
| Installed/archived `blender.exe` | `0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327` |
| Installed/archived bundled `python.exe` | `7d984ee02e70c22f04d5e26f1328865d75a5da14b677890896a294f35815d37a` |
| Retained locked Windows Warp 1.19 wheel | `bb4865dde4c2ae916d3d2a4e20ed1cd406e1d0a76ae722c3c574a9d43385c4d9` |

Distribution-only review is retained at
`artifacts/fspt-p0/runtime-distribution-review-20261127T234245316001Z.json`,
SHA-256 `5e4896711431ca3bc536481c809f12160ab0bca7de7286d774a49faa00842347`.
The older reviewed record at
`D:/Codex/fluid77-5775f3b2-nov21/production-reviewed/blender-provenance.json`
belongs to its old candidate. Its verifier fingerprint is not approval of this
session's tests. Final production qualification requires an independently
reviewed binding of the final frozen verifier outside the checkout, through
the existing `tools/gate/qualification.py` machinery.

The full probe at `e3cb526324` completed with terminal return code **0**,
status **VERIFIED**, and marker **`FSPT-P0-RUNTIME-PASS`**. Its actual API calls
and originals are retained in the runtime evidence directory in section 6.

| Capability | Observed result | Boundary still unqualified |
| --- | --- | --- |
| RenderEngine result APIs | Real `begin_result`, `update_result`, `end_result`, progress and pass callbacks executed. Ten named pass buffers were added, written and read with maximum error 0. | Actual FSPT ten-pass path decomposition, multilayer export and optical semantics. |
| Timers and draw callbacks | Actual timer, `view_update` and `view_draw` ran on Blender's main thread; counts were 1, 1 and 2. Render and pass callbacks were also main-thread in this run. | General asynchronous scheduling/lifecycle performance; no guarantee that all future render callbacks use this thread. |
| Camera/dependency graph | Bound evaluated-camera projection and dependency-graph APIs executed; region/view and shader APIs inspected. | Final thin-lens and shutter-time rendering. |
| Saved-scene inventory | All four exact `.blend` fixtures appended read-only without activation or solving. Each authoritative `scene.fluidsim.domain_object` had role `NONE`. Lens values were 40/50/50/58 mm respectively. All four saved DOF and motion-blur switches were off. | Their disabled settings do not remove required Final DOF/motion blur. All-demo runtime/effect coverage remains a separate gate. |
| Startup and event APIs | Actual `preferences.view.show_splash` and bound `Window.event_simulate` available; `blender --help` confirmed `--enable-event-simulate`. Private-profile splash disabled for probe drawing. | No regular user profile changes were made. |
| Color/file output | Reopened 32-bit EXR ramp error 0. PNG16 RGBA encoded-sRGB maximum error `8.761882781982422e-6`, below `2/65535`; opaque alpha exact. | Straight/premultiplied conversion, zero alpha, HDR/color compositing and actual scene export. |
| Blender graphics | Actual default backend was **OPENGL**. RGBA32F 8×8 host upload, texture readback, shader draw and offscreen readback maximum error 0; display shader bind/unbind executed. | Shared CUDA/graphics images and viewport color-transform parity. |
| Public native image handles | The actual GPUTexture type exposed none of `bindcode`, `as_pointer`, `native_handle`, `vulkan_image`, `cuda_array`. | An opaque Python object address is not an external image handle. |
| CUDA context/fences | Actual Warp kernel returned 3; pinned copy completed; current context matched nonblocking stream context, flag 1; stream destruction returned 0. | Full renderer/solver concurrency and interference. |
| OptiX on current Python | C ABI triangle GAS and known hit/miss trace succeeded in bundled Python 3.13 using the same Warp CUDA context and stream. Wrong-context control rejected before OptiX work. | Dynamic acceleration, FSPT optical equivalence, package support and performance. |

The exact pass names are Combined, Alpha, Depth, Normal, Albedo, Emission,
Environment, Direct, Indirect and Transmission. Synthetic API transport is
explicitly distinguished from their required product meanings in the frozen
contract. [P0_RUNTIME_FEASIBILITY.md](P0_RUNTIME_FEASIBILITY.md) documents probe
implementation and primary API references.

## 4. RTX build cost and presentation decision

The retained OptiX SDK 9.0.0, CUDA Toolkit 12.9.2 (`nvcc` 12.9.86), MSVC
14.39.33519 and Windows SDK 10.0.26100.0 compiled the isolated probe. No
dependency was downloaded or installed. The build report pins the builder,
C++/CUDA sources, compiler/linker, import libraries, SDK headers, logs and
outputs. NVIDIA's advertised Python-3.12 `warp-optix` wheels do not establish
this Blender Python-3.13 ABI; the executed C ABI provides a bounded alternative
without replacing the certified runtime.

| Built item | Bytes | SHA-256 |
| --- | ---: | --- |
| `fspt_optix_probe.dll` | 25,088 | `d5a429f2f53038fa78b6d6f508634b45bcd77a4ffd78a25b587a7dbd23ad9195` |
| `fspt_optix_probe.ptx` | 2,845 | `2d68d0018f29c32ff242caff80ca703171d9a9e123f274966226d933052ab381` |
| `OPTIX_PROBE_NOTICES.txt` | 2,069 | `13d889c4ffd2cc404471d84f5ccc35f83e205779eb6d8aabd28078c4e978e6a7` |

The three deliverable probe files total **30,002 bytes**. The SDK, CUDA
toolchain and driver are not bundled. The compiled SDK helper implementations
retain their BSD-3-Clause attribution; this does not grant or claim general
SDK redistribution rights. Product adoption needs a reviewed dependency,
license, packaging and support contract.

The trace observed exactly `[[1, 0.25, 0.5, 1], [-1, -1, -1, 0]]`, with GAS
2,304 bytes and scratch 512 bytes. Checked stream retirement and native
cleanup both returned 0. The recorded **5,740.3459 ms** cold wall interval
includes initialization/compilation/build work and is not a qualified
throughput measurement. Driver-module hashes are retained for actual loaded
`nvcuda.dll` and `nvoptix.dll`.

**Backend decision:** keep the Warp CUDA software-BVH path as the reference;
treat native OptiX as a proven small feasibility route awaiting integration
and all required renderer tests. A triangle hit does not justify replacing
the reference backend or claiming RTX performance.

**Presentation decision:** retain completed pinned-host RGBA transfer and
Blender texture upload as the explicit fallback. The observed public OpenGL
API does not supply a supported external resource/semaphore bridge. The
retained private `blender-vulkan-interop-sdk` checkout at
`50289d5fa63f1783960a8dfdda191425ad7fd457` was inspected read-only: its public
API supports a qualified vec3 positions transport under its exact Vulkan
activation profile; its RGBA HOST-01 path is internal and unactivated. No
private entry-point bypass, backend switch or GPU-display claim is made.
Actual RGBA sharing still needs device-UUID agreement, supported handles,
fences, resize/teardown, color parity and installed evidence.

## 5. Functional bridge and producer evidence

The live acceptance at `efe61c01c3` observed **24 changing geometries and 24
changing completed images** through actual live frame handlers. The distinct
solver frames were 1–8 and 10–25; the intervening pause/refinement state is not
misrepresented as a continuous 24-frame movie. Controls covered designated
domain ownership, observed pause and refinement, frozen payload rejection,
failed publication followed by cadence skip/recovery, sync warning, reset and
reopen. The production add-on Python-tree SHA-256 in this proof, the producer
proof and the runtime proof was
`883ebc043e8b36f4b1ec45fd7f15b0416733d18860b12f8df284c47c4a3a0e97`.

The live fixture preserved the real dam-break solver and owned sources but
used an explicitly bounded supported-material surface presentation. It hid
particle/whitewater presentation and Blender light objects, used constant
world/emissive mesh lighting, 25% viewport resolution, 6 bounces, 1 live sample
and 8 paused samples. Whitewater simulation remained enabled. This verifies
changing CUDA-rendered fluid and lifecycle behavior; it does not qualify the
authored Principled material, visible whitewater, native-1080p throughput,
final image quality or the 240-state/50-cycle M5 gate.

The original paused-refinement screenshot was directly inspected. It shows
the FSPT fluid/cove viewport without Quick Setup or a splash modal; the opaque
enclosing domain no longer masks the scene. Visible noise/coarseness limits
the visual disposition to **functional presentation**. The original has
SHA-256 `b5d4103dd426962d9f96d8b7c7a40cc5b14ec5dcf53230715a6b1d5385caffc7`.
The image-review receipt is
`artifacts/fspt-p0/live-original-review-20261127T235035464888Z.json`, SHA-256
`8e977bde4909f1d73a26c53dc87e84fa0587945a1aa1d57699b2c9e9f0fceac7`.
One screenshot alone is not evidence of changing states or performance.

The device-surface acceptance at `9609ac0fb8` passed three actual Blender/CUDA
cases: sheet/waves, substep clock and frame clock. It checked exact topology
and world-space positions within `1e-6 m` against the published evaluated
mesh at translated origin `(3, -2, 1)`. Sheet Detail produced 2,048 fine
children, and applied wave displacement changed the final mesh by
`0.00034306084853596985 m` at maximum. Independent reader-stream completion,
pinned readback, source lifetime, real cadence skip/forced publication,
capacity/stale-host/stale-stamp rejection and read-only solver/wave clocks
were asserted. The owned copied packet remained unchanged through two later
solver steps. Device capture readbacks were limited to small scalar metadata.

That proof does not establish renderer-pool integration, normals, named
attributes, reflected/nonuniform transforms, modifier fallback, dynamic
acceleration, installed behavior or solver-interference performance. These
remain required rows; positions/topology success cannot certify them.

## 6. Exact native receipts and retained failures

Paths below are relative to the native worktree. Every passing lane retains
the actual command, private profile, bounded automatic admission lease,
stdout/stderr, source binding, JSON report and terminal return code. All three
terminal return codes were **0**.

| Lane | Directory beneath `artifacts/fspt-p0/bridge-qualification/` | Acceptance JSON SHA-256 |
| --- | --- | --- |
| Live functional | `live-efe61c01c3-20261127T233929604266Z/live/` | `fad9ec4b19bd70ab4d53344f17a6b05a21f45159c7a55b4f8d39ce397e17a332` |
| Device surface | `p0-runtime-producer-9609ac0fb8-20261127T234444726283Z/device-surface/` | `485eac26585fbcd4e4dd24c66579bf4205726b79f00e7bffecad0380f531fa07` |
| Runtime/API/graphics/OptiX | `p0-runtime-e3cb526324-20261127T234956265827Z/p0-runtime/` | `842f2661ad0fbd348d9debb62d4e99576104bd6b0d7987c2c30681687fd92db7` |

The last lane's terminal JSON SHA-256 is
`4648ce33ffc34e7f6b97d8c79bdb63f0e2cdb32fdefff69b8577772f50523a15`;
it completed at `2026-11-27T23:51:15.421613+00:00`.
The successful build report is
`artifacts/fspt-p0/optix-probe-20261127T233648325382Z/build/build.json`,
SHA-256 `0224d593a50e26d3291aced992a1e925357a38f0326db9b8340d2e20f5ea7474`.

Earlier build failures (vcvars quoting, Windows min/max macro interaction,
and missing Advapi32 import library) remain in their separate timestamped
artifact directories. The first native API run at `9609ac0fb8` failed because
the probe inspected `bpy.types.Depsgraph.update` as a class attribute; Blender
exposes it on the bound runtime instance. Its failure report SHA-256 is
`d25b7709a9fa53566c6c53c4be9663de3c5c417ddb0c7dea92d4eb593060fa4c`.
The corrected `e3cb526324` verifier uses actual bound instances. The failed
run is retained and is not labeled a CUDA/product failure.

## 7. Established commands and scheduled baseline work

Source-side contract controls and trial evaluation are:

```text
python -m pytest -q tests/render/test_fspt_acceptance_contract.py
python tools/gate/fspt_contract.py <retained-trial.json>
```

The evaluator's scope is `contract_evaluation_only`; a passing calculation
cannot authenticate pixels, CUDA execution, installed identity or independent
review. The normal production command remains:

```text
python tools/run_production_gate.py --include-gui --measure-budget
```

It requires the reviewed final verifier/distribution binding, clean committed
inputs, certified bundled Python, locked Warp payload, automatic one-job
admission, private Blender directories and intact storage hooks. Session
focused receipts are not a substitute for its release/installed receipt.

The isolated CPU-only native build is reproducible with:

```text
python tools/fspt/build_optix_probe.py --out-dir <fresh-owned-build-directory>
```

Through the same existing admission route, the full native probe used:

```text
blender --python-use-system-env --factory-startup --enable-event-simulate --python-exit-code 1 --python tests/acceptance/blender_fspt_p0_runtime_acceptance.py -- --json-out <fresh-acceptance.json> --deadline 120 --optix-build <build-directory>/build.json
```

Private absolute `BLENDER_USER_CONFIG`, `BLENDER_USER_EXTENSIONS` and
`BLENDER_USER_SCRIPTS` are part of that launch environment. No graphics backend
is forced. The device-surface script uses the same runtime in background;
the live script uses a real window and `--blend demos/dam_break.blend`.
Their exact observed commands are in the terminal JSONs, including actual
absolute source and report paths.

The baseline schedule is **frozen and not yet measured**: for each immutable
quality/tier configuration, run A-B-C-D, D-C-B-A, A-B-C-D, D-C-B-A; each lane
has 120 warmup and at least 600 measured completed states/images. Add a
600-second same-profile sustained Live soak. A preserves required native
outputs; B adds the recorded optional final-surface demand; C adds admitted
snapshot/translation/copy/acceleration without tracing; D adds selected
tracing and presentation. Match solver, surface, output and optional demand
identities. Retain cold costs, stage events, skipped states, actual completed
times, combined memory and originals. This plan makes the next measurement
reviewable; it supplies no measured A/B/C/D ratio.

## 8. P0 exit and later work

| Requirement | Disposition at this evidence snapshot |
| --- | --- |
| Exact source/repair ownership and runtime pins | Reconciled; native patch identity and distribution bytes observed. |
| Frozen scene bindings, lanes, quality, compatibility, optics, export and baseline schedule | Recorded in versioned contract and executable evaluator. |
| Actual Blender API/graphics feasibility | Verified by complete foreground runtime lane. |
| Current-ABI RTX feasibility | Verified bounded C ABI GAS/trace/readback; package and performance excluded. |
| Actual live surface and producer foundation | Verified within the narrow functional scopes above. |
| P0 lead review and exact feasibility-candidate bindings | Complete: report hashes and terminal return codes independently reopened against the exact candidates above. Contract and source inputs are immutable for each proof. |
| Final production verifier provenance and delivery manifest | Still required for the expanded final candidate; does not inherit an older verifier approval. This later release gate is not a missing P0 feasibility probe. |
| Production, all content, installed package and M6 joint realtime claim | Not established by P0; later gates remain required. |

Required subsequent work includes connecting and qualifying bounded snapshot
pools and asynchronous scheduling, no-consumer inactivity, lifecycle/error
retirement, modifier/normal/attribute parity, broad material/light/world
translation, liquid absorption/scattering/nested media, caustics and thin
features, visible particles/whitewater, required Final DOF and motion blur,
ten-pass semantics, alpha/color/compositing, complete 240-frame fresh-process
sequence and cancel/resume, and all fixed/cache/live quality/performance
trials. RTX and shared-image presentation may be adopted only after their
separate integration and qualification gates pass. Final export must refuse
required unsupported content unless the artist explicitly selected a
documented approximation.

P0/M0 closes the reconciliation, frozen contract and bounded feasibility
questions. It does not lower a later requirement to the scope of the current
smoke fixture or turn unmeasured targets into achieved performance.

## 9. Recovery checkpoint

The owned source/evidence checkpoint
`fspt-p2-20261127T235033208133Z-e3cb526324a8-15ae8ba2.zip` is confirmed in
`Dropbox/ChatGPT Deliveries/FluidSim FSPT Backups`, size 24,242,515 bytes.
Local SHA-256 is
`24ffa8b80c8c64c111d87921eb8887de156bd3a2e85b98d1ac192e0af6b55ae6`;
Dropbox content hash is
`436c04ce865b8afd11aabbd3c0bb0efb2d63b8b6d9dee4e7185caf93f7de1868`.
The cloud hash matched the completed local archive. Native receipt:
`artifacts/fspt-p0/backup-cloud-confirmed-20261127T2350.json`.
The archive includes an incremental verified Git bundle from `49aa02be`,
owned originals/receipts and explicitly unfinished cloud drafts. The later
final runtime receipt and this disposition are included in the next checkpoint;
their absence from an earlier archive is not hidden by the archive name.


## 10. Follow-on qualification snapshot, 2026-11-28 UTC

**P0/M0 remains complete at the original feasibility boundary.** The P0
receipts above retain their original candidates, fixture limits and hashes.
The later independent distribution/verifier review admitted exact source
`5c91c8188e5e148f0218a596ed64de4c087e1cc5` for the normal production gate.
Its source fingerprint is
`ec5d2d0eccb21d71f21232821b0e2b8d7746b01f6f79993d2d508981753200cf`
and its verifier fingerprint is
`9d22b451dadda8e99ef987d57fdf0ee983fea69e302ec4fd1bb51ef700afaf2c`.
The independent flat admission JSON SHA-256 is
`0c041ca0338d700304fc6159760cffa7dfbd040c02c07d3d7bc86f42b9e92a80`.

This later source carries the authoritative Session completion clock, checked
aggregate/pool ownership and stream retirement, and a strict production
evidence grader. Native successor tests verified the CUDA broker and the
controlled live device publication/GUI-worker Final handoff. The current
receipts and retained failures are recorded in
[DEVICE_VIEWPORT_QUALIFICATION.md](DEVICE_VIEWPORT_QUALIFICATION.md).
Evaluated viewport and fixed aggregate reuse history are recorded in
[ASYNC_VIEWPORT_QUALIFICATION.md](ASYNC_VIEWPORT_QUALIFICATION.md).

The current Windows source is `ce6afc9ae8b607a33b5c429e6743a60a86df8b1e`.
It retains the completed P0/M0 feasibility record and adds a presentation
scheduling fix after the 0607 Live soak failure. The historical 0607 full
Windows host regression and 50-cycle viewport passed. Its 600-second Live
soak failed after the initial 240 images. Two normal full gate attempts
stopped on exclusive broker contamination of unproven PID ownership. The
successor native GUI attempt ended without exclusive GPU admission. Full P2/P3,
real-time, optics, exports and Linux runtime remain open.
[Current milestone and receipt status](CURRENT_QUALIFICATION_STATUS.md)
