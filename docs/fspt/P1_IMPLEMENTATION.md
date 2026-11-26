# FSPT P1 Blender/CUDA foundation

**Authorization:** Tyler: “Start building this”, 25 November 2026, referring to
https://chatgpt.com/space/page_e736fb28ec2881918bc688878f62ab03.
**Base:** freshly fetched `df8b72bed7967ff7ce855099335c762577f89b0f`.
**Branch:** `experimental/fspt-p1-20261125`.
**Workspace:** `C:/Users/Tyler/projects/fluidsim-fspt-20261125`.
**Owner:** ChatGPT GPT-6.2 Astra lead, per the roadmap's OpenAI roster.

## Goal and boundary

Implement the first real Blender RenderEngine proof, including an actual CUDA
path-traced image, scene/material/camera changes, progressive refinement and
camera Render Result. This is an experimental P1 slice; P2–P6 and the full
product compatibility/performance contract remain open. No merge or release.

The dirty canonical checkout is preserved. Its committed tree matches the
fresh remote base; its additional merge commits do not add source differences.
Existing worktrees and all local edits remain owned by their prior sessions.
No FSPT commit or worktree was found. The historical rendering experiment is
absent; preserved discovery records identify particle draw and interop SDKs,
not a path tracer. Their future transport reuse remains a separate decision.

## Architecture decision for this proof

Use a Blender adapter with owned immutable NumPy snapshots and a lazy Warp
CUDA backend. This reuses the add-on's existing Warp dependency and version
policy. The renderer explicitly rejects CPU devices; no other renderer is
substituted. Warp's software BVH is a feasibility backend, not an OptiX/RTX
performance claim. `warp-optix` is an alternative for the backend comparison:
its published wheels currently target CPython 3.12, whereas this Blender's
Python is 3.13.13. A native 3.13 build remains unqualified. RTXPT would add its
own graphics integration. No dependency upgrade/download is required here.

GPU images use an explicit host-staging presentation path. Record it as such;
do not claim zero-copy. Read bpy only from Blender callbacks; no Python worker
thread traverses Blender state. Existing solver, publication, playback, cache,
surface waves, sheet detail and simulation outputs are not rewritten.

## P1 compatibility contract

Support evaluated mesh triangles, transforms/instances, perspective and
orthographic cameras, constant Background worlds, Diffuse BSDF, Emission,
and smooth Glass BSDF with Fresnel reflection/refraction/TIR. A deliberately
restricted unlinked Principled subset may be translated only with explicit
diagnostics for unsupported semantics. Emissive mesh lighting and world light
produce an actual multi-bounce estimator. Texture graphs, rough transmission,
general Principled, Blender light objects, volumes, whitewater, caustic
qualification, MIS/NEE, denoising, and final compatibility belong to later
milestones. Unsupported active content must be visible and final rendering
must refuse it; never silently replace complex materials with generic water.

## Tasks and completion proof

- [x] Reconcile base, preserve local work, read roadmap/tracker/owner rules.
- [x] Baseline: 43 registration/runtime host tests passed, CUDA hidden.
- [ ] Implement owned/validated snapshot and frame identity contracts.
- [ ] Implement lazy CUDA path tracing and progressive accumulation.
- [ ] Integrate atomic add-on registration and rendered viewport callbacks.
- [x] Execute real Blender preflight and image/change/negative-control tests.
- [ ] Build through the existing extension packager and inspect package identity.
- [ ] Run independent Sol optical and integration reviews of one candidate.
- [ ] Retain evidence and update roadmap plus canonical FluidSim tracker.

## Tests and evidence

Host contract tests: immutable copied arrays, finite/index validation, optical
input validation, camera/scene/settings invalidation, stale-image rejection,
and registration preservation. Runtime tests must exercise `bpy.ops.render`
with FSPT selected, assert finite nonuniform output, prove camera/material
edits change the image, reject CPU and unsupported material input, and record
the loaded paths, source/package hash, Blender/Python/Warp/GPU identities.
Windowed rendered-viewport tests must additionally prove presentation and
capture an original screenshot/recording. Headless results cannot pass that
claim. Profiling numbers remain diagnostic unless an exclusive clean lease
and matched fixed-quality benchmark satisfy the roadmap contract.

All Blender/GPU jobs use blender-gate automatic routing with an explicit
session and kill time, declared inputs, and private profile paths. No forced
node/local/offload flags. Source, installed-package and GUI results are
separate. `evidence/fspt-p1-20261125/` retains failures and receipts.

## Review focus

Negative scale and instancing; unsupported linked material/world nodes;
camera projection changes at unchanged scene generation; viewport resize and
engine switch; empty geometry and nonfinite input; current-frame identity;
module import/registration without a GPU; package versus source import paths.

## Open acceptance decisions

No live FPS target or complete material/pass list is frozen by this P1 proof.
The roadmap's 1080p/1M-triangle/30-FPS points remain proposed evaluations.
The first small fixture is a feasibility proof, not a production benchmark or
a FluidSim demo. The full P0 production-scene inventory and backend comparison
must close before later qualification. Capture a real FSPT canonical image
after P1 output is verified; concept imagery cannot substitute.

## Review corrections and bounded evidence

Both independent Sol reviewers inspected `7dfa4421159e5e5caad63095b9635f2a68b8e18f`.
The next candidate addresses their material override, muted link, evaluated
geometry, camera clip, final-output routing and GPU cleanup findings. The
original negative-control native crash, failing regressions and first package
proof remain retained separately from the corrected results.

On Blender 5.2.0 `fbe6228777e7`, Python 3.13.13, Warp 1.18.0 and RTX 4080 SUPER,
the corrected source passed 78 focused source/registration/gate-inventory
checks, 13 real Blender edge cases, camera Render Result acceptance, and
windowed progressive presentation. The viewport verifier no longer forces
redraws from its polling loop and captures actual pixels at every edit stage.
Glass at IOR 1, an isolated 70-degree total-internal-reflection interface,
diffuse bounce depth and perspective/orthographic far clipping matched the
small analytic fixtures. These are specific controls, not general production
optical equivalence or performance qualification.

P1 final output is one opaque Combined image from one enabled view layer.
Transparent film, stereo/multiview, view-layer material/world overrides,
anonymous GeometrySet instances, non-mesh evaluated components and unsupported
shader graphs fail visibly. Explicit object material slots are respected.
The production gate now includes the headless FSPT checks, with the viewport
check in its GUI lane. The earlier broad host run timed out at 300 seconds;
its identified gate-inventory failure was corrected and the inventory suite
then passed. The broad suite is not claimed as passing.

The current surface offset is `max(1e-5, 1e-6 * length(hit_position))` in
Blender units. Qualification is limited to the ordinary unit-scale fixtures
near the world origin. Thin interfaces, extreme coordinate scales, nested
media, interpolated shading normals and complete material/pass support remain
unqualified. A component-wise error bound and translated thin-interface
regression are required before extending that claim in P2. The certified
Warp 1.19 production environment and approved runtime provenance remain a
separate release gate; the observed Warp 1.18 prototype result does not pass it.
