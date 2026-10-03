# FluidSim Warp — User Guide

FluidSim Warp 2.0.0 is a sparse-grid PIC/FLIP liquid simulator for Blender
5.2.0 or newer using NVIDIA Warp / CUDA. It has two distinct workflows:

- **Live Preview** advances from Blender timeline events for interactive setup
  and look development. It is responsive but not an authoritative production
  transaction log.
- **Production Bake** advances every frame sequentially and commits cache-v2
  frame output plus a deterministic restart checkpoint before the manifest
  exposes that frame as playable or resumable.

## Validated support matrix

| Area | Target |
|---|---|
| FluidSim | 2.1.0 |
| Blender | Blender 5.2.0 LTS or newer 5.2.x |
| Operating systems | Windows x64 and Linux x64 |
| GPU-gate certified | Windows 11 x64 on an NVIDIA GeForce RTX 4080 SUPER. Linux x64 is packaged from the same source but not certified |
| Compute | NVIDIA CUDA GPU required for solver execution; CPU and macOS are not certified |
| Warp dependency | NVIDIA Warp 1.17.0 (CUDA 12 build), installed via preferences button |
| Cache compatibility | Cache v2 write/playback/resume; Cache v1 (1.0.x) is refused: bake again |

Install the extension ZIP for your platform. On first run, open **Edit →
Preferences → Add-ons → FluidSim Warp** and press the **Install Warp** button.
This downloads the certified NVIDIA Warp 1.17.0 wheel (~160 MB) from PyPI and
installs it into Blender's Python. The button is safe to press multiple times:
if Warp is already installed at the correct version, it reports success without
reinstalling. An NVIDIA driver that supports CUDA 12 (R525 or newer) is required.

If a newer 1.x Warp is ever loaded instead (for example through a developer
override), FluidSim still runs but marks the runtime status as
"untested Warp version"; Warp older than 1.17.0 or a 2.x Warp is refused.

The extension is packaged for both Windows x64 and Linux x64. Only Windows has
a full recorded GPU gate run. On Linux (Ubuntu 24.04 under WSL2) the package
installs cleanly and passes every headless gate lane and 52 of 53 GUI checks.
Blender's viewport screenshot reads back black under WSLg, which stopped the
gate before its last GUI lane, so treat Linux as uncertified.

On WSL2, CUDA must come from the Windows driver (`/usr/lib/wsl/lib`). If an
NVIDIA Linux driver package (for example `libnvidia-compute-*`, pulled in by
Ubuntu's `nvidia-cuda-toolkit`) is also installed, Warp can load it instead and
report no CUDA device. Put `/usr/lib/wsl/lib` first on `LD_LIBRARY_PATH` or
remove that package. If Blender's window crashes under WSLg, start it with
`WAYLAND_DISPLAY=` and `GALLIUM_DRIVER=d3d12` set.

Upgrading from 1.x? Read the [1.x → 2.0 migration guide](migrating_to_2_0.md)
first: 2.0 opens only files saved by 1.1.0 or later.

---

## 1. Guided first run

After selecting one or more mesh objects, the shortest workflow is:

1. Open **Scene Properties → Setup**, click **Set Up & Preview…**, choose
   **Quick Liquid** and a product preset, then confirm.
2. FluidSim performs runtime and capacity preflight, prepares the solver and
   outputs, and starts Blender timeline playback.

The compact **3D View → FluidSim** status panel exposes Preview, Pause/Resume,
Apply Changes, and Restart shortcuts. Its Live Tuning child contains only
runtime-compatible adjustments and links to the authoritative Scene Properties
workflow. Object roles and per-input controls remain
in **Object Properties** and also appear contextually in **Physics Properties**.

Guided setup creates or reuses one owned collection tree without deleting or
renaming unrelated user data:

```text
FluidSim
├── Inputs
├── Outputs
└── Helpers
```

The selected mesh becomes an Initial Fill emitter, a gravity-aware
voxel-aligned domain is created in `Helpers`, and six owned output shells with
default materials appear in `Outputs`: `FluidSim Surface`, `FluidSim Particles`,
`FluidSim Whitewater`, `FluidSim Foam`, `FluidSim Bubbles`, and
`FluidSim Spray`. Combined whitewater is the backward-compatible default; the
typed shells are ready for **Separate** or **Combined + Separate** publication.
The current bundled Geometry Nodes revision is attached to every owned output
without replacing user-owned modifiers or node groups. Re-running the same
setup is idempotent. Choose **Set Up Only…** when you want the same owned scene
configuration without starting playback.

Other setup modes are **Quick Container**, **Quick Viscous Liquid**,
**Add Collider**, **Add Inflow**, and **Add Outflow**. Quick Container tags the
selected closed mesh as a collider and enables a height fill. Add modes preserve
the existing domain.

### Product presets

| Preset | Purpose | Relative cost |
|---|---|---|
| Preview Water | Coarse interactive blocking; whitewater off | Low |
| Production Water | General final-quality liquid and cache-v2 defaults | Medium/high |
| Splash + Whitewater | Energetic shots with larger whitewater budgets | High |
| Honey | Implicit high-viscosity liquid | High |
| Thick Liquid | Stylized medium/high viscosity | Medium/high |
| Thin Liquid | Lower-viscosity detailed motion | Medium |
| Water Surface Tension | Physical water-like capillarity (0.072 N/m) | High at fine resolution |
| Stylized High Tension | Explicitly nonphysical tight droplets and beads | High |

Built-ins are read-only JSON resources. **Save User Preset** captures only an
explicit allow-list of solver properties and writes atomically to
extension-managed user storage. Unknown fields and arbitrary RNA paths are
rejected before scene mutation. Applying a preset reports its intended scale,
performance cost, and Reset requirement.

---

## 2. Manual Scene Setup

Manual setup remains available for authored pipelines:

1. Assign the Domain object and resolution in **Scene Properties → Domain & Particles**.
2. Select one or more input objects and assign their role in **Object Properties →
   FluidSim Role — Selected Objects** or the contextual **Physics Properties →
   FluidSim** panel.
3. Configure the active object's role-specific Region, Source, Collider, Force,
   Field, or Volume Grid panels. Role actions affect the selected set; detailed
   values describe the active object.
4. Choose a product preset under **Scene Properties → Setup → Product Presets**
   or edit Simulation Physics directly.
5. Review **Capacity & Safety**, then use **Preview** or Production Bake as
   appropriate.

Guided and manual workflows use the same object roles and solver settings; the
setup wizard is a transaction service, not a separate solver architecture.

---

## 3. Blender Properties workflow

FluidSim uses Blender's native Properties editor as the authoritative settings
surface. The attached workflow is organized from setup through production, not
by implementation module. Daily panels open by default; optional physics,
rendering, cache, diagnostics, and code-authoring panels remain collapsed until
needed. Inapplicable values are hidden rather than shown as inert controls.

### Scene Properties

The root panels appear in this fixed order.

#### Session & Status

Shows one derived activity state: **Idle**, **Live Preview**, **Cache Playback**,
or **Production Bake**. It also shows runtime/device readiness and the actions
valid for the current state. Pause, Resume, Restart, Cancel, and Clear Error
remain available when relevant. Manual Initialize, Release, Reset, and stepping
controls are retained in the default-closed **Advanced > Lifecycle** panel.

Children:

- **Pending Settings** appears only when a Reset or Reinitialize action is
  staged. It offers **Apply Changes** and **Revert Pending Values** from the same
  lifecycle service used by execution. Apply Changes selects the least
  destructive valid transition and confirms only when progress will be lost.
- **Runtime Statistics** appears when this scene owns the solver or its cache
  playback is active.
- **Live Timeline Recovery** owns the bounded catch-up frame limit.

#### Setup

**Set Up & Preview…** configures the selected objects and starts real timeline
playback. **Set Up Only…** stops after configuration. **Product Presets** applies
built-in or user presets and reports the active preset identity.

#### Domain & Particles

This is the sole owner of Domain object, Voxel Size, Liquid Sampling, Maximum
Particles, automatic particle growth, topology expansion, and open boundaries.
**Capacity & Safety** contains sparse-leaf and whitewater capacities, growth
failure policy, current peak-memory estimate, and verified safe alternatives.

#### Initial Liquid — Height Fill

Owns only height-fill enablement, base height, flat/profile back face, profile
image and scale, lifecycle, velocity, and particle emission. Liquid Sampling is
not repeated here; it is inherited from Domain & Particles.

#### Simulation Physics

The root summarizes active solver features. Children are:

- **Time & Transfer** — time scale, CFL, substep limits, PIC/FLIP transfer,
  tracing, and count stiffness.
- **Gravity** — scene gravity or an authored vector.
- **Pressure** — solver, iterations, threshold, and PCG-only tolerance.
- **Viscosity** — enablement, constant/particle source, solver, iterations, and
  PCG-only tolerance.
- **Surface Tension** — Phase-5 coefficient, density, force band, safety, normal
  smoothing, stable capillary timestep preview, and runtime curvature/force
  diagnostics.
- **Particle Relaxation**, **Velocity Limits**, **Global Forces**,
  **Dissipation**, **Attraction**, and **Spatiotemporal Processing**.

#### Custom Effects

Custom Effects follows the same panel density and status-first flow as the rest
of Scene Properties. The root contains four groups:

1. **Effect Stack** — Add Effect, ordered list, duplicate, move, mute, solo, and
   remove.
2. **Selected Effect** — artist-facing description, typed parameters, controller
   objects, frame window, and cadence.
3. **Preview / Apply Changes** — prepares eligible draft effects through the
   production admission path without a separate compile step.
4. **Status** — trust, current draft/applied divergence, production admission,
   compile evidence, and the applied SHA-256 identity.

**Effect Engineering** is an Advanced-only, closed child retaining Test Draft,
Compile & Apply, Revert Draft, and Preview One Frame for troubleshooting.
**Effect Packages** is a closed maintenance child for import, comparison,
export, reveal, rename, and removal. Package mutation is shown only for an
actually installed user package; embedded scene definitions are identified as
such. **Warp Code** is a closed developer child visible for any selected effect
with kernels. It uses Blender Text datablocks, prefers an existing Text Editor,
shows Domain/Stage/Cadence and typed field access, maps diagnostics to source
lines, and exposes generated-source and runtime evidence. The prototype Custom
Warp Snippets panels are not registered.

#### Surface Reconstruction

The root owns surface enablement. Enabled children are **Resolution & Cadence**,
**Particle Splat**, and **Filters**. Preview publication cadence remains
separate from Production Bake transactions.

#### Whitewater

The root owns whitewater enablement. Children are:

- **Emission** — rate, churn accumulation, decay, and speed threshold.
- **Classification** — Phase-4 foam band, spray threshold, and hysteresis.
- **Lifetime** — aggregate lifetime range.
- **Dynamics** — foam, bubble, and spray behavior and type-specific lifetime
  multipliers.
- **Admission Status** — required/admitted/dropped values and foam, bubble,
  spray, and invalid counts when runtime data exists.

#### Viewport & Outputs

**Display** owns the combined liquid view mode, display fraction/scale,
attribute-color selectors, whitewater visibility, and Combined/Separate output
mode. **Materials & Collection** owns output collection and surface/particle and
combined/typed whitewater materials. Forced Geometry Nodes repair/update and
the detailed Surface, Particle, and Whitewater attribute inventories are shown
in Advanced mode. Hidden compatibility aliases do not appear.

#### Cache & Production Bake

The root owns the resolved Cache Path. Children are:

- **Playback** — Preview, Pause, or Resume Cache; this starts the actual Blender
  timeline and does not allocate a live solver.
- **Shared Playback / Export Range** — optional playback/export range.
- **Storage** — compression and checkpoint interval.
- **Production Bake** — authored bake range, progress, one mutually exclusive
  Start/Resume/Restart action and Cancel. Advanced mode also exposes Copy
  Headless Bake Command. Resume
  still performs full signature/checksum/checkpoint verification at execution.
- **Manage Cache** — shown only for a recognized FluidSim cache or interrupted
  FluidSim transaction. Move, Rename, orphan removal, and Free dialogs display
  and revalidate the exact path or inventory immediately before mutation. The
  filesystem service independently rejects arbitrary directories.

#### Diagnostics

Contains **Profiling**, **Benchmark**, and **Report & Export**, with the
Validation Suite action on the root. Benchmark baseline saving and file exports
remain explicit operator dialogs.

### Object Properties

Analytic Volume provides the six analytic signed-distance shapes. Cone and
Pyramid preserve their exact Euclidean boundary distance.

**FluidSim Role — Selected Objects** is the role assignment surface. Its action
labels state that they operate on the selected set. Detailed child panels use
the active object only.

Conditional panels are:

- **Active Object Input Status** for current diagnostics.
- **Region Geometry** for Sources, Sinks, Forces, and Field Sources.
- **Liquid Source** and optional **Source Lifespan** for Sources.
- **Collider** for mesh Colliders.
- **Force Region** for Force inputs.
- **Field Blending** and **Field Source** for advanced Field inputs.
- **Volume Grid** only when the canonical Source, Sink, or Field region is a
  Volume Grid. Selecting a volume payload automatically selects Volume Grid
  coverage; returning to analytic coverage restores a Constant payload so no
  configured value becomes unreachable.

### Add-on Preferences

Preferences contains only runtime selection and detection. The external-Warp
developer path appears only when `FLUIDSIM_ALLOW_EXTERNAL_WARP=1`. Device
selection does not appear in scene panels.

### 3D View FluidSim workflow

The **3D View → FluidSim** sidebar tab follows the order you work in. A pinned
strip at the top shows activity, runtime readiness, Preview / Pause / Resume /
Restart and Apply Changes; below it four stages hold the settings:

1. **Setup** — interface level, Quality, presets, Domain, voxel size, sampling.
2. **Simulate** — flip ratio, viscosity, surface tension, scene gravity, and the
   live readout.
3. **Look** — view mode, surface, whitewater, particle display and Materials.
4. **Bake & Export** — cache folder, Preview Cache, Production Bake and Export
   Current Frame, with **Storage & Range** and **Manage Cache** shelves.

Each control has one home, and every stage edits the same `Scene.fluidsim` /
domain-object owners as Scene and Physics Properties.

**FluidSim header popover.** The 3D View header carries a **FluidSim** button.
It opens a floating panel with the live loop — the status line, Preview / Pause /
Resume / Restart, Apply Changes when an edit is waiting, Quality and the
physics you tune while watching (flip ratio, viscosity, surface tension, scene
gravity). It draws the same functions as the sidebar, so the two never differ.
With no domain yet it offers Guided Setup instead. Use it when the sidebar is
closed or you are working in a large viewport.

**FluidSim workspace.** **FluidSim Workspace** in that popover switches to a
workspace named *FluidSim*: a large 3D Viewport with the sidebar open and a
Timeline, copied from the layout you are using. The first press creates it and
later presses only switch back to it. Blender lets an add-on open the sidebar
but not pick its tab, so click the **FluidSim** tab once if another tab is
showing. The workspace is saved with the .blend like any other; delete it from
the workspace tabs (right-click → Delete) if you do not want it. Disabling the
add-on never removes it.

### Physics Properties — contextual FluidSim objects

Selecting a FluidSim object exposes a **FluidSim** panel in Physics Properties:

- the exact `Scene.fluidsim.domain_object` receives domain and physics groups;
- tagged surface, particle, whitewater, foam, bubble, and spray outputs receive
  their relevant scene-owned display, material, attribute, or dynamics groups;
- Source, Sink, Collider, Force, Field, and Volume Grid inputs reuse the
  existing role-aware Object Properties panels.

Domain assignment changes only the scene pointer. Removing an output requires
confirmation and revalidates its stable tag, FluidSim ownership, and exact
object identity before deletion. Object names are labels, never authority.

## 4. Production render attributes and viewport visualization

FluidSim has two deliberately separate attribute paths.

### Persistent production attributes

Enabled channels are written to live Blender outputs, cache-v2 frames, and—when
restart-critical—solver checkpoints. Blender publication validates data type,
domain, length, and finite values before mutating an output datablock. An old
attribute with a conflicting type or domain is recreated rather than silently
reused.

| Output | Available persistent channels |
|---|---|
| **Surface** | velocity, speed, vorticity, age, lifetime, color, source ID, viscosity, density, motion blur |
| **Liquid particles** | velocity, speed, vorticity, age, lifetime, color, source ID, UID, viscosity, density |
| **Whitewater** | velocity, speed, age, lifetime, mandatory type |

Stable Blender names include `fluidsim_velocity`, `fluidsim_speed`,
`fluidsim_vorticity`, `fluidsim_age`, `fluidsim_lifetime`, `fluidsim_color`,
`fluidsim_source_id`, `fluidsim_uid`, `fluidsim_viscosity`, and
`fluidsim_density`, and `fluidsim_whitewater_type`. These use the `POINT` domain. Motion blur is the separate
conventional Blender attribute `velocity`: it contains displacement per scene
frame. `fluidsim_velocity` remains the physical world-units-per-second value.

Speed is derived from velocity. Vorticity is sampled from the curl of the
projected MAC velocity field. Surface channels are transferred from the nearest
liquid particle with a deterministic, memory-bounded lookup. Color, source ID,
and UID are optional persistent liquid channels; disabled channels do not
allocate a full particle-capacity pool. UID values are unique positive int32
values assigned monotonically at committed emission transactions.

New cache frames carry an ordered attribute-inventory schema and transfer-method
identifier. Declared-but-missing, misaligned, wrong-type, non-finite, duplicate
UID, or unsupported data fails closed. Phase-4 frames require one aligned int32
whitewater type value per point. Phase-3 cache-v2 frames remain readable, but whitewater with no recorded type is published as legacy
unclassified (`-1`) only on the combined output; FluidSim does not invent a
physical classification.

### Production whitewater types

Whitewater uses one simulation pool with a persistent type channel. Stable
values are **Foam = 0**, **Bubble = 1**, and **Spray = 2**. Type `3` is reserved
for a future aerosol/dust model and is rejected by current writers/readers.
Initial classification uses liquid-SDF depth, speed, and churn. Hysteretic enter
and exit thresholds update type during advection without rapid threshold
flicker.

Per-type controls are grouped in **Surface & Whitewater**:

- **Foam:** grid-advection strength, preservation, local-density influence, and lifespan multiplier.
- **Bubbles:** fluid drag, buoyancy opposite gravity, and lifespan multiplier.
- **Spray:** air drag, gravity scale, and lifespan multiplier.

**Combined** publishes `FluidSim Whitewater`; **Separate** publishes filtered
`FluidSim Foam`, `FluidSim Bubbles`, and `FluidSim Spray`; **Combined + Separate**
publishes both views. These are Blender-side views of the same GPU pool. The
combined-plus-separate mode therefore increases Blender host/render geometry, not
cache payload or simulation-pool allocation.

### Bundled Geometry Nodes and materials

The packaged P4 asset installs six revision-tagged groups:

- **FluidSim Surface Render [P4 v1]** assigns the selected surface material while preserving mesh attributes.
- **FluidSim Liquid Particles [P4 v1]** applies point radius, lifespan culling, sphere instancing, attribute-preserving realization, and material assignment.
- **FluidSim Whitewater [P4 v1]** renders the backward-compatible combined pool.
- **FluidSim Foam [P4 v1]**, **FluidSim Bubbles [P4 v1]**, and **FluidSim Spray [P4 v1]** provide independent material routing for typed views.

Installation is idempotent. **Update FluidSim Modifiers** repoints only modifiers
previously tagged as FluidSim-owned; identically named user modifiers and
user-edited node groups are skipped. The default liquid and particle materials
read `fluidsim_color`; the liquid material also maps `fluidsim_viscosity` to
roughness. Point groups realize render instances so point-domain named
attributes are visible to shaders in F12 renders. Realization has a geometry and
memory cost, so particle display fraction and output toggles remain relevant for
heavy scenes.

### Viewport debug coloring

The existing non-Material display modes remain a presentation-only diagnostic.
They write a separate `display_color` attribute and use the blue→cyan→green→
yellow→red percentile ramp. They do not overwrite `fluidsim_color` and do not
change persistent simulation channels.

| Output | Debug color modes |
|---|---|
| **Surface** | Material · Height · Speed · Temperature · Viscosity · Density |
| **Particles** | Material · Height · Speed · Temperature · Viscosity · Density · Age · Life |
| **Whitewater** | Material · Height · Speed · Age · Life · Type |

---

## 5. Custom Effects and Warp Code

Custom Effects replaces the prototype snippet editor with two coordinated
levels:

- **Artist mode** exposes effect intent, typed controls, spatial controller
  objects, stack order, execution status, cost, trust, and preview actions.
- **Warp Code** exposes Text Editor source, domain, stage, cadence, frame scope,
  field permissions, generated source, source-mapped errors, and profiling.

### Fast path

1. Click **Add Effect** and select **Vortex Force**.
2. Move the created Vortex controller and adjust Strength, Radius, and Falloff.
3. Click **Preview**. FluidSim prepares the draft effect, solver, and outputs,
   then starts timeline playback.
4. Use **Apply Changes** after later edits; it confirms only when the required
   transition discards current preview progress.

The Advanced-only Test Draft action performs static validation, forces real
Warp JIT compilation, runs
bounded guarded fixtures, synchronizes the selected device, and verifies
canaries and finite output. Compile & Apply swaps the entire candidate stack
atomically. A failed effect never removes the previous working stack and is
never silently omitted.

### Execution status and production identity

Draft and applied state are separate. Any live Text edit, parameter value,
controller transform, order, cadence, frame scope, binding, or capability
change marks the stack different from the applied identity. Production Bake is
blocked until the exact draft is tested and applied.

The effective stack follows enabled/mute/solo rules. Its canonical SHA-256 and
ordered inventory are written to cache-v2 frames, checkpoints, manifests,
reports, and headless output. Resume rejects mismatches with
`EFFECT_SIGNATURE_MISMATCH` before solver mutation.

### Trusted code

Custom Warp source executes in Blender's process and is trusted local code.
Shipped built-ins are trusted only when their exact content matches the bundled
inventory. Imported or embedded custom code is disabled and untrusted until
**Review & Trust** records its exact digest. Editing code revokes that effective
trust. Headless policy defaults to `builtins-only`; `deny` and `trusted` are
also available.

### Packages and library management

Import and export use strict `.fseffect` packages with source hashes. Keep Both
creates a new effect ID; packages cannot shadow a built-in ID. **Compare
Update** classifies a candidate as absent, identical, modified, newer, older,
ambiguous, or a built-in conflict without installing it. **Rename Package**
changes the installed package identity and display name through an atomic
validated rewrite; existing embedded scene instances deliberately keep their
current identity. Malformed packages remain visible as panel diagnostics. Use
the folder action to reveal the extension-managed library or remove an
installed package while retaining an embedded scene copy.

Detailed workflow and API references are in `docs/custom_effects.md` and
`docs/custom_effects_api.md`.

---

## 6. Profiling

1. In **Diagnostics / Profiling**, enable **Enable Profiling**.
2. **Accurate GPU Timing** brackets each phase with a CUDA synchronize for
   exact timings (leave on unless you need lower profiling overhead).
3. Run the sim (live or stepped). The panel shows average per-phase timers;
   the viewport overlay (toggle **Viewport Overlay**) shows live stats.
4. Click **Write Diagnostics Report** to dump profiler, per-frame stats,
   benchmark, and validation results to a JSON file.

The profiler implements the `fluidSim_profiler::1.0` contract: 16 `compute_*`
phase slots plus `dcc`/`input`/`solver`/`blit`/`queue`/`output`/`total`
tracks and device memory metadata.

---

## 7. Validation

**Validation / Benchmarks → Run Validation Suite** runs 21 registered tests
on the selected certified CUDA device:

1. Sparse tiled leaf-grid coordinate/index roundtrip
2. Leaf-local addressing and start-vector reconstruction
3. Field precision parity (16/32-bit)
4. Particle-to-grid count/average/active masks
5. Grid-to-particle velocity/attribute sampling
6. Source shape coverage and particle attribute transfer
7. Height-field sourcing behavior
8. NanoVDB/OpenVDB volume sourcing kernels
9. Collision/sink voxel-type behavior
10. Pressure projection divergence reduction
11. Jacobi and PCG residual convergence
12. Surface SDF build/filter parity
13. Viscosity stability
14. Whitewater churn/emission/advection behavior
15. Benchmark/profile schema compatibility
16. ST-FLIP jitter, phase, and residual bounds
17. Surface-tension static-droplet boundedness and volume proxy
18. Surface-tension droplet-oscillation trend oracle
19. Surface-tension zero-coefficient solver equivalence
20. Surface-tension capillary timestep constraint
21. Surface-tension finite-energy and diagnostic guard

Every result includes pass/fail, detail, and elapsed time. The five P5 entries
are mandatory CUDA release gates. The supplied-package CPU validator is a
separate analytic one-leaf kernel proof; it does not replace the full CUDA
solver suite.

---

## 8. Benchmarking

1. Set **Benchmark Frames** (default 24).
2. Set a **Baseline** path (or leave default `//fluidsim_benchmark.json`).
3. **Run Benchmark** — runs a deterministic dam-break scenario and shows avg
   step time, voxels/s, and active voxels.
4. **Save Baseline** stores the run; later **Run Benchmark** runs compare
   against it: active-voxel fields must match **exactly** (determinism), and
   performance fields must stay within 25 % relative tolerance.

The benchmark implements the synthesized `BENCHMARK_FIELDS` schema
(`totalTime`, `timeStepAverage/Min/Max`, `voxelsPerSecondAverage/Min/Max`,
`activeVoxelsAverage/Min/Max`, `result`).

---

## 9. Production Bake, cache v2, and headless execution

### Transaction authority

Each cache-v2 frame uses this commit order:

1. Advance the solver exactly one frame with Blender frame handlers suppressed.
2. Write and fsync frame output to a temporary NPZ, then atomically replace the
   final frame file.
3. Capture the full restart checkpoint, write/fsync it, and atomically replace
   the checkpoint file.
4. Compute SHA-256 and byte size for both files.
5. Append a `FrameEntry` to a copied manifest and atomically replace
   `fluidsim_cache.json`.

The manifest is authoritative. A frame/checkpoint file without a committed
manifest entry is an orphan and is never playable or a resume point. Missing,
non-contiguous, size-mismatched, checksum-mismatched, schema-invalid, or
incompatible data fails closed. Cancellation is observed only between frame
transactions, so the last visible frame is always fully committed.

Cache layout:

```text
<cache>/
├── fluidsim_cache.json
├── frames/frame_000001.npz
├── checkpoints/checkpoint_000001.npz
└── logs/
```

### Resume and restart

**Resume** restores only the last verified committed checkpoint, then starts at
the next frame. Scene, solver-setting, source/evaluated-geometry, domain,
capacity, and ST-FLIP compatibility signatures are checked before live state is
replaced. Solver-affecting changes are rejected with a structured reason;
render-only display/material changes remain resume-safe. **Restart** explicitly
removes the prior v2 transaction set and begins at the authored start frame.

Checkpointed state includes liquid and whitewater particles, sparse topology
and persistent active fields, source one-shot/accumulation/RNG state, whitewater
RNG, ST-FLIP temporal/RNG/reference-mass state, solver frame, and simulation
time. Transient bins and scratch fields are rebuilt after restore.

### Cache management

- **Integrity Scan** reports committed, missing, corrupt, and orphan data without
  mutating it. Orphan removal is a separate explicit action.
- **Free** first renames the cache to a same-volume trash path before deleting.
- **Move** copies to a temporary destination, verifies the full cache, then
  atomically adopts it before optional source cleanup.
- **Rename** accepts only a leaf directory name.
- **Open Folder** uses path APIs rather than shell-string interpolation.

A cache written by FluidSim 1.0.x (`fluidsim_sequence.json` plus NPZ frames)
is refused rather than played: free it and bake again. **Export Current Frame**
continues to write OBJ/PLY output.

Older cache-v2 frames remain playable under their recorded attribute/type
schema. Production resume is deliberately stricter than playback: a Phase-4
manifest/checkpoint does not acquire a Phase-5 settings signature merely
because surface tension is disabled, and a pre-1.0.251 cache has no authoritative
Custom Effects identity. Rebake with 1.0.251 before relying on restartable
production work in this release.

### Headless command

The installed package includes `release/run_headless_bake.py`:

```bash
blender --background scene.blend \
  --python /installed/fluidsim_warp/release/run_headless_bake.py -- \
  --scene Scene --start 1 --end 250 \
  --cache /absolute/path/cache --device cuda:0 \
  --resume auto --checkpoint-interval 1 \
  --report-json /absolute/path/report.json --save-after-bake
```

One compact JSON object is written to stdout per committed frame. The final
report is atomic. Stable process exit codes are: `0` success, `2` argument
error, `10` Warp import/version failure, `11` requested device unavailable,
`12` no CUDA device, `13` invalid scene, `14` incompatible/missing required
resume, `15` bake failure, and `16` safe cancellation. SIGINT/SIGTERM request
cancellation at the next transaction boundary.

---

## 10. Live interaction and pending changes

FluidSim classifies every exposed simulation setting into one interaction tier:

| Tier | Meaning | What happens while a run is advanced |
|---|---|---|
| **Live** | Safe bounded solver mutation | Applies before the next step without clearing particles or history. |
| **Live / Targeted Refresh** | Live mutation with a bounded subsystem refresh | Applies without resetting; for example a surface or viewport-output refresh. |
| **Restart Required** | Meaningful from simulation start | Remains pending until rewind, explicit Reset, or Revert. |
| **Reinitialize Required** | Allocation, device, or domain construction change | Remains pending until rewind, explicit Reinitialize, or Revert. |
| **Blocked** | Unsafe in the current context | Remains pending with an actionable reason; the active run is preserved. |

Pending state belongs to each scene. Editing simulation resolution, liquid
sampling, leaf/whitewater capacity, domain, or a restart-only lifecycle value
in the middle of a run does **not** silently erase that run. One neutral
**Pending Settings** child appears under **Scene Properties → Session & Status**
instead of injecting action rows into unrelated panels. Rewind to the timeline
start to apply eligible changes automatically, or choose **Apply Changes** or
**Revert Pending Values**. Apply Changes delegates Reset versus Reinitialize to
the shared lifecycle policy and confirms the actual progress loss. A failed explicit action
retains its pending reasons. Revert never changes or saves Blender add-on
preferences; device changes remain explicit in Add-on Preferences.

Particle capacity is different: **Initial Particle Reserve** is a starting GPU
allocation, not a hard runtime ceiling. With **Auto Grow Particle Pool** enabled,
FluidSim allocates a larger particle/binner pool, copies the live prefix, and
swaps only after successful CUDA allocation and synchronization. The authored
reserve is not rewritten and an active pool is never shrunk automatically. If
the live VRAM guard or allocation refuses growth, the selected advanced policy
either continues while deferring the whole source batch, or pauses Live while
preserving the initialized solver. No partial source batch is committed.

The compact **3D View → FluidSim** launcher remains reachable without becoming
a second settings surface. It shows solver activity, runtime readiness, the
valid Preview, Pause/Resume, Apply Changes, or Restart action, pending-change
count, and a shortcut to Scene Properties. The viewport overlay remains
informational only.

---

## 11. Tips

- **Start coarse.** Use a large Voxel Size (e.g. 0.08) while iterating, then
  refine. Resolution scales cubically.
- **Keep timestep limits sane.** CFL and capillary stability both contribute
  to the selected substep count. Fast impacts, fine voxels, or high surface
  tension may require more Max Substeps.
- **Explosions?** If the sim blows up, lower **Max Speed** (Particles panel),
  enable closed-domain walls, or reduce FLIP Ratio toward PIC.
- **No whitewater?** Lower **Churn Threshold** and raise **Emission Rate**;
  whitewater needs turbulent impact regions.
- **Performance.** Watch the overlay timers; Pressure and Surface phases
  dominate at high resolution.

## Surface resolution

Viewport, cache, and export meshes can use a sparse render grid independent
from simulation resolution. The optional 2x multiplier halves surface voxel
size without multiplying pressure, collision, or particle-transfer grid cost.
Simulation resolution remains the default to preserve interactive performance.
Frame cadence builds once after simulation substeps; Manual builds lazily when
the mesh is requested. Keep the simulation surface field enabled when using
whitewater or snippets that consume `surface_sdf`.

## ST-FLIP (Experimental)

Enable **Simulation > Spatiotemporal Processing > Enable ST-FLIP
(Experimental)** to test spatiotemporal sampling at large time steps. The
implementation includes temporal residuals, space-time P2G, phase-field
activation, optional variable pressure, local trajectory substepping, and
render-surface re-synchronization. It is experimental because it has not yet
passed multi-scene reference-error and long-run conservation gates; keep it
off for production scenes. **Variable Projection (Experimental)** is off by
default since 2.0; turn it on only to evaluate density-weighted pressure. With automatic reference mass, the solver
runs the Monte Carlo calibration for the current ST seed. Liquid Sampling is a
Reset-tier rest-density contract, so changing it stages Reset rather than
rescaling the active ST-FLIP reference mass in place.

## Safe Settings recovery

Before allocating solver arrays, FluidSim estimates peak VRAM against the
GPU's current free memory and reserved safety margin. Particle-only shortfalls
auto-grow to a chunked geometric target when that complete solver allocation
fits the conservative budget. Other unsafe initial allocations remain blocked
before replacing an existing solver. The Simulation panel shows one capacity
report, a severity-ranked recommendation list, and one explicit review/confirm
flow; it does not expose an unverified memory bypass. Runtime particle growth
separately guards the
peak where old and replacement pools coexist; a failed grow-copy never swaps or
uninitializes the existing solver. These estimates reduce allocation risk but
do not guarantee immunity from arbitrary CUDA failures.

The recommendation samples current VRAM and GPU utilization through Warp and
the NVIDIA driver, plus Windows CPU and RAM pressure. It stays as close as
possible to the scene configuration while adjusting the largest cost drivers:
leaf capacity, voxel size, particle and whitewater capacities, particles per
voxel, topology expansion, and secondary surface-grid settings.
---

## 12. Validation limits for this build

Release 2.0.0 passed the GUI-inclusive production gate on Windows 11 with
Blender 5.2.0 LTS and the bundled Warp 1.17.0 on an NVIDIA GeForce RTX 4080
SUPER. The gate runs the CUDA pytest regressions, the CUDA self-validations, the GUI
acceptance checks, saved-demo timeline playback, cache-v2 production bake,
export and resume, handler save/reopen, the release validators and the bounded
demo-analysis checks for all four demos. A clean-profile install of the
Windows package passed its smoke test and the installed GPU regression.

The four demos bake to bitwise-identical results on the same GPU, driver and
Warp build; the gate checks this by hashing the demo analysis across repeated
runs. That claim does not extend to a different GPU, driver or Warp version,
where GPU floating-point results can differ. The solver and cache checks
otherwise assert physical, schema, ownership and bounded-error invariants.

Linux x64 is packaged from the same source but has no recorded Linux CUDA run,
and renderer-specific visual certification is not part of the gate.
