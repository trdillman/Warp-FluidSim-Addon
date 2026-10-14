# FluidSim Warp 4.0 — release plan

Status: proposal, 2026-10-14. Baseline is master `296fa95` (3.2.1: Blender 5.2 LTS,
Warp 1.17.0 cu12 installed from Preferences, CUDA only, Windows certified, Linux
packaged, 2936+ tests, GUI gate reference hash `8da88683…2105`).

Inputs: the 3.0 handoff (`docs/RELEASE_3_0_HANDOFF.md`), the 3.2 review register
(`.codex/context/FLUIDSIM_REVIEW_SOURCE_OF_TRUTH_20261011.md`), the archived
native-interop and surface-rendering experiments, the Warp 1.16–1.18 changelogs, the
Blender 5.0–5.2 Python API notes, and the 2026-10-14 research dossier (SPH, neural
operators, SSFR). Appendix A reconciles every dossier claim against this codebase:
what is already shipped, what is adopted, what is adapted, and what is declined with
evidence. **Nothing in this plan is adopted because a paper says so; each item has a
measurable acceptance and a named owner module.**

Specs for the items most likely to be built wrong live in `docs/specs/4.0/`.
Implementers read the spec before touching the code.

## Theme: liquid you can see and touch

3.x made the solver trustworthy (deterministic bakes, cache v3, bake safety, ~100
fixes). 4.0 changes what the artist *experiences*: water that looks like water in the
viewport without a mesh build, a direct-manipulation interface drawn in the viewport
instead of a grey panel stack, objects that float and get pushed, and a solver whose
defaults are the right ones. Underneath, it moves to the Warp 1.18 geometry stack and
puts the first learned component into the pressure solve, gated so it can only make
things faster, never different.

## Why a major version

| Break | What changes for the user | Spec |
|---|---|---|
| Multigrid becomes the default preconditioner | Results change (the Jacobi hydrostatic-depth defect goes away). Every cache re-bakes. | `SPEC-defaults-4.md` |
| Warp 1.18 / CUDA 13.4 certification | New NVIDIA driver floor; `wp.MarchingCubes` is deprecated upstream, surface extraction moves to `warp.geometry` | `SPEC-warp-118-surface.md` |
| Cache v3.1 (additive) | Frames gain rigid-body pose and impulse records; v3 caches still play | `SPEC-rigid-coupling.md` |
| Liquid Deck interaction layer | New primary interaction surface in the 3D viewport; panels remain as the parity surface | `SPEC-liquid-deck.md` |

Nothing else in 4.0 breaks a file. The learned pressure guess and Liquid Preview are
additive and switchable.

## Scope

### 1. Liquid Preview — screen-space fluid rendering in the viewport
Today the only liquid look during playback is the marching-cubes mesh (built on a
dense sub-grid with a span cap) or raw points. Add a **Liquid Preview** display mode
that draws particles as depth sprites into a `GPUOffScreen`, filters depth with the
Truong–Yuksel narrow-range bilateral filter (two separable 1D passes plus one
clean-up), reconstructs normals from depth derivatives, accumulates thickness for
Beer–Lambert absorption, and composites with Schlick Fresnel over the viewport colour
in a `POST_PIXEL` handler. Whitewater draws as a second sprite layer (foam/spray/
bubble tinted by the existing type attribute). All of it is `gpu` module code
(`GPUShaderCreateInfo`, `create_from_info`), so it runs on Vulkan and OpenGL alike and
is testable headlessly with Blender 5.2's new `gpu.init()`.

Input is the same host position array Output Publication already reads back for the
PointCloud (the native-interop audit measured that readback at about 1% of solver wall
at 600k particles; see Appendix A, "zero-copy"). No new readback, no change to Output
Publication ownership: Liquid Preview is a *display* of the published payload.

Acceptance: a headless `gpu.init()` offscreen render of the crown_splash frame-30
payload matches a committed reference image within a perceptual tolerance; Liquid
Preview on at 600k particles adds ≤ 4 ms per redraw at 1080p on the gate GPU; with it
off, every existing gate and hash is unchanged. Spec: `SPEC-liquid-preview.md`.

### 2. Liquid Deck — a modern, in-viewport interaction layer
The ask: break out of the grey panel into something modern and not seen before in
Blender. The design that satisfies that *and* the repository's ownership rules is an
interaction layer drawn in the 3D viewport with the `gpu` module and driven by gizmos
and a modal operator, where **every Deck action resolves to an existing operator or
property**. Panels stay as the parity and accessibility surface; the Deck is the fast
surface.

Deck components (each one a lane-sized deliverable, each optional and user-disableable):

- **Deck bar.** A translucent card anchored to the viewport bottom edge: transport
  (play/pause/step/restart), live readout (particles, substeps, ms/frame sparkline),
  pending-change chip ("Apply" / "Reset pending" with the reason), bake progress ring.
  Hover and click are routed by one modal operator; keyboard parity via the existing
  operators.
- **Direct-manipulation gizmos.** `GizmoGroup` per role: an emitter shows a velocity
  arrow you drag, a rate dial you turn, and a shape outline; a collider shows friction
  and "floats" toggles; the domain shows resize handles that preview the padded grid
  and cell count while dragging. Each gizmo writes the same property the panel writes,
  so Pending Settings classification is unchanged.
- **Command palette.** A type-to-search overlay (modal operator, `gpu`-drawn) listing
  FluidSim operators, presets, demos and settings with inline values; Enter executes or
  focuses the property. Replaces hunting across three tabs.
- **Radial wheel.** A custom-drawn radial menu on a hotkey for the six most common
  actions, with the current stage (Setup/Simulate/Look/Bake) as the hub.
- **Visual language.** One token sheet (colours derived from the active Blender theme,
  one accent, radius, blur fallback, motion timing) shared by the Deck, Liquid Preview
  overlays and the stage header; a "Classic panels only" preference turns the whole
  layer off.

Explicitly not in scope: a custom editor Space (not possible from Python), area
pop-outs (declined 2026-10-02), and a custom node editor for effects (stretch, §8).

Acceptance: every Deck action has a test proving it invokes the same operator or sets
the same property as its panel twin; the GUI gate adds one Deck lane (open demo, drag
the emitter velocity gizmo, confirm the pending chip and the applied value); Deck off
leaves the UI operator/panel behaviour tests byte-identical. Spec: `SPEC-liquid-deck.md`.

### 3. Two-way rigid-body coupling — objects that float and get pushed
Deferred from 3.0 ("no file break needed, so 3.1/3.2"), never built. Colliders already
push liquid (one-way, with velocity, friction and grip). Add the other direction: a
collider tagged **Dynamic** accumulates fluid force and torque each substep from the
pressure impulse on its surface samples (Akinci-style boundary samples with the
existing SDF) plus drag, integrates a 6-DoF rigid state owned by FluidSim, and
publishes its pose as keyframes on the object (so the result renders, exports and
survives with no solver). Blender's Bullet world is not driven, because Python cannot
apply per-frame forces to it; the FluidSim integrator is the authority for Dynamic
colliders and Bullet is told to ignore them.

Acceptance: a buoyancy test (density ratio 0.5 cube settles at half submersion ±5%
within 48 frames under Multigrid); a momentum test (fluid and body momentum conserved
to 2% over a dam-break-into-crate); the crown_splash demo gains a floating cork whose
pose track is in the cache stamp and in Alembic export. Spec: `SPEC-rigid-coupling.md`.

### 4. Solver defaults that are right
- **Multigrid preconditioner default.** The 3.0 known limitation "a deep liquid column
  settles about a third lower than it should under Jacobi" is a correctness defect
  artists hit on any tank. Multigrid is symmetric, atomic-free and inside the graph
  already; its cost (1.34× per solve on dam_break) is paid back in iterations at any
  tolerance that matters. Re-baseline the four demos.
- **APIC stays opt-in** (measured: frames 1.26× longer). Revisit when the learned guess
  lands.
- **Narrow band stays opt-in** (slower on splashy scenes).
- Remove the `TIME_OFFSET` leftovers and the unguarded unregister loops (3.0 handoff §2).

### 5. Learned pressure guess (gated neural component)
The first neural component, built where it cannot change answers: a small
Fourier-neural-operator-style network on the coarse leaf grid predicts the pressure
field from divergence, fluid/solid masks and the previous frame's pressure, and the
prediction seeds PCG's initial guess (`solve_pcg(..., use_coarse_guess=True)` already
exists as a reachable seam, today seeded by the electrostatic leaf solve). PCG still
runs to the same tolerance, so the result differs from today only within tolerance.
Inference is pure Warp (`wp.tile_fft` / `wp.tile_ifft`, available since 1.14, plus
small tile matmuls); no PyTorch in the extension. Training is an offline developer
tool (PyTorch, `tools/learned_guess/`) over bakes of the four demos and a randomised
scene generator; weights ship as one `.npz` under 5 MB with a schema version.

Gate to ship on by default: ≥ 2× fewer PCG iterations at the default tolerance across
the four demos and the randomised set, a wall-clock win per solve on the gate GPU, no
convergence regressions (max iterations never hit where it was not hit before). If the
gate fails, it ships off with the tool kept. Spec: `SPEC-learned-pressure-guess.md`.

### 6. Warp 1.18 and the geometry stack
Warp 1.18.0 (2026-10-05) builds its wheels with CUDA 13.4 and deprecates
`wp.MarchingCubes` in favour of `warp.geometry.IsoSurfaceMarchingCubes` and the new
`warp.geometry.sparse_marching_cubes()`. Our surface module runs `wp.MarchingCubes` on a
dense sub-grid with a span cap. 4.0 (a) certifies Warp 1.18 (driver floor documented,
Install Warp downloads the new wheel, 1.17 accepted with the existing "older than
certified" path for one release), (b) moves surface extraction to sparse marching
cubes over active leaves, removing the span cap and the dense allocation, and (c) wires
the VDB writer to the narrow-band `liquid_sdf_*` data that 3.0 left unwired.
Spec: `SPEC-warp-118-surface.md`.

### 7. Platform and trust
- **Certify Linux.** `VALIDATED_RUNTIME_PLATFORMS` is still Windows only. The blocker
  was the black Mesa window grab; the GUI gate's offscreen draw now has `gpu.init()`
  available for the headless lanes. Target: one Linux GUI gate pass with the reference
  hash, recorded in the manifest.
- **USD point export spike.** Alembic writes an empty transform for a PointCloud on 5.2.
  Spike USD export of the points and whitewater; ship if Blender 5.2's exporter carries
  the attributes, otherwise record the result and keep OBJ/PLY.
- **ABBA performance program.** Paired step-time comparison 4.0 vs 3.2.1 on all four
  demos; budget as in §Sequence.

### 8. Stretch (only after rc gates are green)
- **Guide to Target** (art direction without autodiff): a built-in v2 effect that
  attracts liquid toward a target mesh SDF with a keyframed strength, the classical
  control-force approach. Fits the effects pipeline as-is.
- **Effect Graph**: a custom node editor (`bpy.types.NodeTree`) composing v2 effects,
  fields and stages. Large; 4.1 unless a lane frees up.
- **Differentiable parameter fitting**: `wp.Tape` through a coarse, sort-free FLIP
  substep to fit emitter velocity to a frame-60 target. Research only in 4.0.

## Explicitly out of 4.0

- **Zero-copy CUDA↔viewport interop.** Declined with evidence (Appendix A). Stock
  Blender has no exportable-VBO API; the 2026-07 experiment needed a patched Blender,
  and even then measured 1.11–1.14× whole-frame at 600k particles because the readback
  is ~1% of solver wall. Re-open only if Blender ships an external-memory `gpu` API.
- **SPH / IPBF solver, RT-core neighbour search, SoA particle reorder.** This is a
  sparse-grid FLIP/APIC solver with hash-free P2G/G2P; the SPH data-structure work in
  the dossier does not apply (Appendix A).
- **Neural surrogate replacing the solver.** Only the gated initial-guess use (§5).
- **ST-FLIP re-introduction.** Removed in 3.0 for lack of a conservation reference;
  the published method (TOG 45(4), 2026) is a candidate for 4.1 with a reference gate.
- **CPU or macOS simulation.** Playback-only stays.
- **Adaptive / multi-resolution grids.** 4.1+.

## Sequence and acceptance

Every milestone passes the reproducible GUI gate once with a hash matching its
recorded reference (`tools/run_production_gate.py --include-gui`, scoped validation per
`docs/RELEASING.md`; memory `full-gui-gate-run-procedure`), plus a paired ABBA
step-time comparison against 3.2.1 on all four demos (median of frames 2–48, n ≥ 6).

| Milestone | Contents | Hash change attributable to |
|---|---|---|
| 4.0.0-alpha.1 | Warp 1.18 certification, sparse marching cubes, Multigrid default, VDB narrow-band wiring, TIME_OFFSET/unregister cleanup, Linux gate green | Multigrid default and the surface extractor (re-baseline all four demos; rebind `max_crown_height` evidence per `docs/DEMO_SCENE_CONTRACT.md`) |
| alpha.2 | Liquid Preview (display only), Deck bar + command palette | none (display and UI only; prove it with the alpha.1 hash) |
| beta | Rigid coupling (cache v3.1), Deck gizmos + radial wheel, USD spike result | rigid coupling on the crown_splash cork only; other demos unchanged |
| rc | Learned pressure guess (ship on only if its gate passes), docs and user guide rewritten around the Deck, ABBA report | within PCG tolerance only (prove by forcing the guess off: hash must equal beta) |

Performance budget: alpha.1 may cost up to +10% per frame on dam_break Final (Multigrid)
but must cut pressure iterations ≥ 3× at tolerance 1e-5; alpha.2 must add ≤ 4 ms per
redraw with Liquid Preview on and 0 with it off; beta's rigid coupling must cost
≤ 5% on crown_splash with one dynamic body; rc must be net faster than beta per
solve. Anything else over +5% is a stop.

Evidence rules carry over unchanged: a behaviour change lands behind a failing
regression test first; review prose is not evidence; no verification agents in
workflows (memory `no-verification-agents-in-workflows`); one gate pass + hash match.

## Orchestration: parallel lanes

| Lane | Scope | Owns (files) | Conflicts |
|---|---|---|---|
| W warp-118 | §6 a, b | `runtime/warp_runtime.py`, `version.py`, `preferences.py`, `sim/surface.py` (extractor only), `tools/release/*` | lands first; S rebases |
| D defaults | §4 | `sim/solver_settings_model.py`, `properties.py` defaults, demo rebuild | after W |
| V liquid-preview | §1 | new `blender/liquid_preview/` (shaders, passes, handler), `blender/playback.py` hook only | none |
| K deck | §2 | new `blender/deck/` (tokens, bar, palette, wheel, gizmos), `blender/ui_header.py` toggle | shares `viewport_overlay.py` with V: both draw via one registry |
| R rigid | §3 | new `sim/rigid_coupling.py`, `sim/kernels_rigid.py`, `blender/colliders.py` (Dynamic flag), `io/cache` v3.1 | touches `solver.py` step order; one owner for that hunk |
| N learned-guess | §5 | new `sim/learned_guess.py`, `tools/learned_guess/`, `sim/pressure.py` (seed seam only) | after D (training data must come from Multigrid bakes) |
| L linux + usd | §7 | gate scripts, `blender/interchange.py` | none |
| P perf/ABBA | §7 | `tools/benchmark_versions.py`, budget | last |

Each lane: worktree under `C:\Users\Tyler\wt\r40-<lane>`, red test first, focused
suite green, `--quick` gate before merge, full gate at each milestone (orchestrator).
Durable scratch in `C:\Users\Tyler\wt\orch\scratch` (never TEMP).

## Open decisions (owner: Tyler, with a default if unanswered)

1. **Driver floor.** Warp 1.18 wheels need the CUDA 13.4 driver. Default: certify 1.18,
   keep 1.17 accepted with the existing warning path for 4.0 only.
2. **APIC default.** Default: stays opt-in in 4.0; re-measure at rc with the learned guess.
3. **Deck default state.** Default: on for new files, off for files saved before 4.0,
   one preference to override.
4. **Learned guess shipping with weights.** Default: ship weights only if the gate in §5
   passes; otherwise the tool ships and the feature is hidden.

---

## Appendix A — reconciliation of the research dossier against this codebase

Dossier: "Comprehensive Research Dossier: Real-Time SPH, Neural Operators, and GPU
Fluid Rendering" (2026-10-14). Status key: **shipped** (already here), **adopt**
(taken as proposed), **adapt** (taken with a design change), **decline** (with evidence).

| Dossier item | Status | Evidence / where it lands |
|---|---|---|
| §1 Tier 1: SPH / IPBF-VBD real-time core, SoA reorder, Morton/hash neighbour lists (§3.1–3.2) | decline | This solver is a sparse-leaf-grid FLIP/APIC solver (`sim/fields.py`, `sim/solver.py` `_p2g/_g2p_advect`, `sim/pressure.py`). It has no neighbour search to accelerate; P2G/G2P are grid scatters/gathers, already graph-captured. Rewriting to SPH would discard the deterministic-bake, cache-v3 and effects-v2 work. |
| §2.2 Tait EOS / WCSPH | decline | Incompressibility is enforced by projection; WCSPH's 1% density error would be a regression. |
| §2.3 SDF boundaries with normal reflection | shipped | Collider SDFs, friction and grip are in (`sim/source_kernels.py`, 2.1/3.2 fixes). The *two-way* half is new: §3. |
| §3.3 RT-core / OptiX neighbour search | decline | No neighbour search; Warp has no OptiX path; BVH refit was measured and declined 2026-10-12 (P12). |
| §4.1 Screen-space fluid rendering, narrow-range filter, Beer–Lambert, Fresnel | adopt | §1 Liquid Preview, `SPEC-liquid-preview.md`. Blender's `gpu` module provides offscreen targets, create-info shaders and `gpu.init()` for headless tests (5.2). |
| §4.2 AARBF volumetric ray-casting with blue-noise jitter | decline for 4.0 | Needs an RT-core ray-caster; not reachable from Blender's `gpu` module. Blue-noise temporal jitter is reused inside Liquid Preview's thickness pass. |
| §4.3 Whitewater classification by neighbour count (spray/foam/bubble) | shipped | `sim/whitewater.py` classifies by neighbour density with hysteresis; §1 adds the sprite rendering. |
| §5.1 FNO / FC-PINO velocity prior, resolution independence | adapt | Not a velocity surrogate: §5 uses an FNO-style net as PCG initial guess only, where it cannot change the converged answer. Warp-native inference via `wp.tile_fft`. Resolution independence is why training on 32³ leaf grids and inferring on larger domains is credible; the gate measures it. |
| §5.2 diffSPH / CraftSPH inverse art direction, SFBC | adapt / stretch | Autodiff through sort-free coarse FLIP is a research item (§8). The artist-facing goal (hit a target shape at frame N) is served first by the classical Guide-to-Target control force (§8), which the effects-v2 pipeline supports today. |
| §6 Zero-copy `cudaGraphicsGLRegisterBuffer` → `wp.from_ptr` → VBO | decline | Blender 5.x exposes no VBO handle or external-memory API in `gpu`; the Vulkan backend has no GL buffer to register. The 2026-07 experiment (archive tag `archive/codex/native-gpu-interop-main-20260715`) patched Blender to add a `GPUVertBufExternalLease` and measured 1.109–1.139× whole-frame at 600k particles; publication was ~1% of solver wall. The win does not justify a custom Blender. |
| §6.2 invariant 2, "raw draw calls fail without a bound VAO in 5.2" | n/a | Not applicable to the `gpu` module path, which owns its own batches. |
| §6.2 invariant 3, graph-capture boundaries | shipped | The pressure, viscosity and whitewater paths already capture fixed-shape kernels with conditional graphs (`wp.capture_while`, Warp ≥ 1.8); allocation stays outside capture. |
| [WHK17] infinite continuous adaptivity | decline for 4.0 | Grid solver; adaptivity would be a multi-resolution grid programme (4.1+). Narrow band (3.0) covers the particle-count half. |
| [Bra26] Spatiotemporal FLIP | decline for 4.0 | Implemented pre-3.0 and removed in 3.0 for lack of a conservation reference; the TOG paper now exists and a 4.1 re-introduction needs that reference gate first. |
| [BK15] DFSPH, [MM13] PBF, [Aki+12] rigid-fluid coupling | adapt | Only the boundary-sample force/torque accumulation idea from Akinci is reused, on the grid solver (§3). |
| [TY18] narrow-range filter | adopt | §1. |
| [Mor+24], [Wol22] | partial | Blue-noise jitter only (§1 thickness pass). |
| [Li+20], [Dur25] FNO | adapt | §5 (initial guess only; Warp tiles, no PyTorch at runtime). |

## Appendix B — platform facts this plan relies on (verified 2026-10-14)

- Blender 5.2 LTS released 2026-07-14, Python 3.13 (since 5.1); `gpu.init()` added in
  5.2 for `--background` GPU work; `gpu` module exposes `GPUOffScreen`,
  `GPUShaderCreateInfo`, `GPUVertBuf`, compute shaders, `POST_VIEW`/`POST_PIXEL`
  handlers. No external-memory or VBO-handle API.
- Warp 1.18.0 (2026-10-05): `warp.geometry` (sparse marching cubes, swept volumes),
  **wheels built with CUDA 13.4 (breaking)**, `wp.MarchingCubes` deprecated. 1.17.0
  (certified today): CG/CR restart, BVH sphere/capsule queries, graph-capture fixes in
  radix sort. 1.16: rebuildable NanoVDB volumes, grouped `HashGrid`. 1.14: `wp.tile_fft`.
  1.8: `wp.capture_if` / `wp.capture_while` (already used here).
- Gate GPU: NVIDIA GeForce RTX 4080 SUPER (native-interop evidence); Nsight Graphics
  not installed, so Vulkan draw timing is measured via offscreen timers, not hardware
  counters.
