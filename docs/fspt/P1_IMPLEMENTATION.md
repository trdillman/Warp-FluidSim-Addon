# FSPT P1 Blender/CUDA foundation

Tyler authorized implementation on 25 November 2026. The experimental branch
is `experimental/fspt-p1-20261125`, based on
`df8b72bed7967ff7ce855099335c762577f89b0f`, in
`C:/Users/Tyler/projects/fluidsim-fspt-20261125`. The Astra lead owns delivery
under the roadmap's OpenAI roster. The canonical worktree and prior sessions'
uncommitted changes were preserved. [Roadmap](https://chatgpt.com/space/page_e736fb28ec2881918bc688878f62ab03)

## Implemented scope

FSPT renders static evaluated triangle meshes through a Blender RenderEngine.
The adapter copies geometry, material assignments and camera data into owned
NumPy snapshots. A lazy Warp CUDA backend traces paths against a software BVH
and accumulates samples under a key derived from scene, camera, resolution and
render settings. The rendered viewport uploads completed pixels through host
memory. Final output is one opaque Combined image from one enabled view layer.

The supported material graphs connect Diffuse BSDF at zero roughness, Emission,
or smooth Glass BSDF directly to Surface with constant inputs. All Principled
BSDFs are rejected. Constant Background worlds and mesh emitters provide
lighting. The adapter respects object-linked material slots. Unsupported
active graphs, layer material/world overrides, transparent film, multiview,
non-mesh evaluated components and anonymous GeometrySet instances trigger
diagnostics. CPU rendering is rejected.

## Implementation and verification status

- [x] Preserve existing work and establish the baseline of 43 registration/runtime checks with CUDA hidden.
- [x] Implement owned snapshots, validation and frame identity.
- [x] Implement CUDA path tracing, progressive accumulation and invalidation.
- [x] Integrate add-on registration, Blender camera Render Result and rendered viewport presentation.
- [x] Run real Blender/CUDA preflight, image, edit and rejection controls.
- [x] Build and inspect the extension package and obtain independent optical and integration reviews.
- [x] Bind final candidate `0ece56afe51b88b3a42f7596b65ba58f750f36b5`, package, runtime receipts and review disposition to the immutable source.
- [x] Update the roadmap and canonical FluidSim tracker with real imagery and bounded results.

The focused host lane passed 79 checks. The broader host run timed out at
300 seconds. Its gate-inventory defect was corrected and that focused suite
passed. The broader suite has no passing result.

The recorded experimental runtime used Blender 5.2.0 build `fbe6228777e7`,
Python 3.13.13, Warp 1.18.0 and an RTX 4080 SUPER. Source and installed-package
acceptance exercise the selected engine via `bpy.ops.render.render`, check saved
images and compare camera/material edits. The windowed verifier observes
progressive draws and captures actual pixels. It checks changes in viewport
render resolution with a fixed window size. Automatic blender-gate routing,
private profiles, declared inputs and bounded leases apply to these lanes.

The 14 analytic and guard cases cover both object-slot overrides, a layer
material override, muted Surface linkage, mixed mesh/point output, anonymous
geometry instances, IOR-1 glass, a 70-degree total-internal-reflection
interface, both camera far-clip modes, diffuse bounce depth, transparent film,
multiview and multiple enabled layers. A fifteenth control issues a real
valid render after refusal, checks saved EXR pixels and rechecks all refused
output directories. All 15 passed in one standalone Blender process with
exit 0. Final-source camera and windowed viewport acceptance also exited 0.
The Windows extension installed and passed its camera-render controls from
the installed module path. All 186 packaged Python files matched the source
and installed payloads. See the [evidence index](../../evidence/fspt-p1-20261125/README.md)
for artifact hashes, terminal receipts and the two final review dispositions.

The two Sol reviews began at `7dfa4421159e5e5caad63095b9635f2a68b8e18f`.
Follow-up work addressed material selection, muted links, evaluated geometry,
camera clipping, final output routing and GPU cleanup. Earlier failures,
including a native crash during an unsupported render, remain part of the
record. The canonical static fixture rendered from `3a062` at 512 × 384,
128 samples, six bounces and seed 31. Its evidence index records the PNG hash.

## Remaining qualification

The complete product endpoint still includes live fluid integration, the P2–P6
roadmap and production compatibility and performance. Current evidence covers
static scenes at ordinary Blender unit scales near the origin. Live simulation,
RTX traversal, zero-copy presentation, MIS/NEE, complete materials and passes,
image sequences, cancellation, and physical viewport resizing remain
unqualified. The proposed 1080p, one-million-triangle, 30 FPS evaluation
requires a fixed-quality benchmark. The P0 production-scene inventory and
backend comparison remain open.

The current surface offset is `max(1e-5, 1e-6 * length(hit_position))` in
Blender units. Thin interfaces, extreme coordinates, nested media and
interpolated shading normals need additional controls. P2 needs a
component-wise error bound and translated thin-interface regression before
extending the supported scale range.

The full production gate invoked from `3a062` was blocked by missing reviewed
`BLENDER_SHA256` provenance. The prototype profile used Warp 1.18; the
certified Warp 1.19 environment was not qualified by this work. The production
gate has no passing result. Merge and release remain outside the authorized
experimental work.
