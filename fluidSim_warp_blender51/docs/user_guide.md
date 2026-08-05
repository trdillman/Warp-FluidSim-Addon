# FluidSim Warp — User Guide

GPU liquid simulation for Blender 5.1, powered by NVIDIA Warp / CUDA.
The solver is a sparse-grid PIC/FLIP liquid simulator that runs entirely on
the GPU and plays back **interactively in the viewport** — no cache baking is
required. Optional npz caching and OBJ/PLY export are available as supplemental
tools.

> Requirements: **Blender 5.1**, an **NVIDIA GPU with CUDA**, and `warp-lang`
> importable from Blender's Python. Open the **FluidSim** tab in the 3D
> Viewport sidebar (press **N**).

---

## 1. Quick Demo (30 seconds)

1. **FluidSim → Simulation** panel → click **Quick Setup**.
   This creates a domain box, a box particle source, and a sphere collider.
2. Click **Initialize Solver**. The status line shows the CUDA device, leaf
   capacity, and particle allocation.
3. Click **Live: Off** to turn live simulation **on**.
4. Press **Spacebar** (Play). Liquid emits from the source, falls under
   gravity, and splashes off the collider in real time.
5. Use **Step 1** / **Step 10** to advance frames without playback.

Three output objects appear in the **FluidSim Output** collection:

| Object | Type | Contents |
|---|---|---|
| `FluidSim Surface` | Mesh | Marching-cubes liquid surface |
| `FluidSim Particles` | PointCloud | Liquid particles |
| `FluidSim Whitewater` | PointCloud | Foam / spray / bubble particles |

---

## 2. Manual Scene Setup

If you prefer to build the scene yourself instead of using Quick Setup:

1. Create a **domain** object (any mesh) sized to your simulation volume.
   In **Simulation**, assign it to **Domain**.
2. Create an emitter object, select it, and in **Inputs → Liquid Emitters**
   click **Tag Liquid Emitter**. Choose **Initial Fill** or **Continuous Inflow**,
   an **Analytic Volume** or **Mesh Surface Shell**, inherited or multiplied
   liquid sampling, and initial velocity.
3. Create collider or sink objects, select them, and use the corresponding
   collapsed section under **Inputs**.
4. Set **Simulation Resolution** in **Quality**. Manual leaf and particle
   capacities remain available under collapsed **Advanced** sections.
5. Click **Initialize Solver**, then **Live: Off** to start, then Play.

---

## 3. Panel Reference (7 task-oriented top-level groups)

### Simulation
Core controls: **Quick Setup**, **Initialize Solver**, **Live** toggle,
**Reset**, **Release**, **Step 1 / Step 10**. Live status mirrors particles,
active voxels, step time, and substeps. Time Scale and gravity remain here.
Quick Setup is idempotent when its complete domain/source/collider setup already
exists. The collapsed **Advanced Playback** foldout contains **Maximum Playback
Recovery Frames**, an expert safety limit for how many timeline frames live
playback may recover synchronously after Blender drops frames.

### Quality

The default user-facing quality controls are **Simulation Resolution** (cell
size), **Liquid Sampling** (rest occupancy), **Transfer** (PIC/FLIP and FLIP
ratio), **Adaptive CFL**, and **Max Substeps**. The panel reports authoritative
cell size, unpadded and 8-cell-padded dimensions, sparse leaf-cell count, and
nominal particle spacing. Resolution stages **Reinitialize**; sampling stages
**Reset** and never mutates an advanced live solver in place.

### Inputs

Liquid emitters, sinks, colliders, force regions, height fill, and compatibility
point injection are collapsed task sections under this group.

### Advanced → Domain / Sparse Grid
Domain object picker, **Voxel Size**, **Leaf Capacity** (max 8³ leaves),
**Topology Expand** (dilation iterations around active fluid),
**Open Boundaries** (kill particles leaving the domain). Shows the resolved
resolution and active/total leaves.

### Advanced → Particles
**Initial Particle Reserve**, **Auto Grow Particle Pool**, the guarded-growth
failure policy, advanced copies of liquid sampling and **Transfer** (PIC / FLIP)
with **FLIP Ratio**, **Trace** (Euler / single-sample / midpoint / modified-midpoint
/ **RK3**), relaxation (enable / strength / mode), speed limit
(**Max Speed** + Limit Behavior + Percent), and **Count Stiffness**
(anti-clumping).

### Liquid Emitters
Use **Tag Liquid Emitter** to create a role-owned workflow. New Phase 3 emitters
start as **Initial Fill** without changing the saved default of older files.
Initial Fill acts once after Initialize/Reset; **Continuous Inflow** replenishes
while Enabled. New sources inherit global **Liquid Sampling** by default or use
a bounded multiplier. Migrated sources retain a **Legacy Absolute Target** so
old authored occupancy is preserved. Sampling is occupancy, never volume/second
or spatial resolution, and changes require Reset.

Accepted emission regions are:

- **Analytic Volume** — the six analytic signed-distance shapes; Cone and
  Pyramid preserve their exact Euclidean boundary distance.
- **Mesh Solid Volume** — the interior of a validated watertight mesh. Invalid
  solid input stops that emitter and never falls back to a shell.
- **Mesh Surface Shell** — an explicit narrow band around open or closed mesh
  triangles. The effective half-width is at least half a voxel and is reported.
- **Points / Particles** — either **Direct Append Once** (not a volume fill) or a
  **Rasterized Region** that participates in occupancy top-up.
- **Volume Grid** — an explicit fog/mask or SDF grid from a Blender Volume. The
  grid's authored transform is composed with the object transform before values
  are flattened into live 8x8x8 source leaves.

**First Sim Frame** retains source geometry until reset. **Every Sim Frame**
updates animated input, but complex meshes are bounded by the measured
interactive budget and may require First Sim Frame or lower source resolution.
Errors for watertightness, missing grids, capacity, memory, or dynamic work are
shown beside the input that caused them.

Spawn Motion owns Initial Velocity and Add Animated Object Velocity. The
collapsed **Particle Attributes** panel owns Temperature, Viscosity, **Density
Attribute** (not physical mass), and optional Lifespan.

### Liquid Sinks
Sinks have a separate workflow and no emitter lifecycle, occupancy, spawn
motion, or particle payload controls. Regions are Analytic Volume, Mesh Solid
Interior, Mesh Surface Shell, or an accepted Volume Grid mask. Watertight status
appears only for Solid; shell width appears only for Surface Shell. Collision,
sink, and liquid precedence remains collision > sink > liquid.

### Colliders
Colliders are mesh obstacles with per-object Friction, Restitution, and Transfer
Animated Velocity. Collider gathering never performs source/sink watertight
analysis.

### Domain Height Fill
Height Fill supports **Flat** and **Profile Image**. Profile samples a real
grayscale Blender Image across the domain and scales its height offset. It has
an explicit Initial Fill or Continuous lifecycle, initial velocity, and sampling
controls. **Infinite** is unsupported and intentionally absent; an old saved
Infinite value remains unapplied with an actionable compatibility error.

### Point / Particle Injection
**Inject Active Object Points Once** remains a collapsed compatibility utility
for mesh vertices, PointCloud points, or the first particle system. It requires
an initialized solver and never claims volume emission. Reinject on
Initialize/Reset is labeled as compatibility behavior, not an emitter lifecycle.

### Advanced Field Sources
Advanced Field Sources are explicitly non-liquid. Accepted targets are Pressure,
Scalar/Temperature, Vector Velocity, and face-specific Staggered Velocity.
Analytic constant regions and named Volume grids are shown only for combinations
that construct a valid frozen SourcePlan. Blend and amount apply only to the
selected field operation. The legacy `source_mode` is never presented as a
liquid choice; old FIELD sources are classified as pressure-field compatibility
inputs.

### Forces
**Global Grid Forces** provide directional acceleration, a radial force
(positive repels, negative attracts), velocity diffusion, and exponential
dissipation of particle temperature, viscosity, or density. **Attract to
Collision SDF** pulls liquid toward colliders and closed domain walls within a
chosen distance. Use **Tag Force Region** for an independent analytic region;
only FORCE-role objects appear in its editor. Regions support Off / Directional /
Omnidirectional / Spin / Radial; Spin exposes its axis and force regions never
source liquid.
All 13 native Blender force-field object types are read automatically as GPU
proxies, including animated transforms, field shape, strength, flow/noise,
distance limits, and falloff. Type-specific controls are also bridged: Harmonic
rest length/damping, Drag coefficients, Texture mode/size/coordinates, Curve
Guide speed/free/clump controls, Turbulence size/seed, and Fluid Flow source
direction.

The proxy behavior is intentionally fluid-oriented. Magnetic uses the grid
velocity, Harmonic is a damped spring, Lennard-Jones uses the minimum distance
as its fluid interaction scale, and Drag applies linear plus quadratic drag.
Fluid cells do not have Blender particle charge or boid state, so Charge assumes
a unit charge and Boid acts as a goal field. Texture is a deterministic GPU
vector/gradient/curl field seeded by the assigned texture datablock rather than
a CPU evaluation of arbitrary texture nodes. Curve Guide samples up to four
curve controls. Fluid Flow uses the source object's local Z direction as its
target velocity; it does not copy a Mantaflow domain velocity grid.

### Physics

Pressure and viscosity are collapsed solver sections under this group.

#### Pressure Projection
**Solver** (Jacobi / **PCG**), **Iterations**, **Tolerance**, **Threshold**
(relative pressure cutoff).

#### Viscosity
Enable checkbox. **Solver** (Jacobi / PCG), **Viscosity** constant, **From
Particles** (per-particle viscosity attribute), **Iterations**, **Tolerance**.

### Surface & Whitewater
Basic render-surface controls expose generation, surface upres, particle radius,
smoothing, and viewport update cadence while stating that render visibility does
not disable the solver liquid SDF. Whitewater exposes aggregate churn,
emission, speed, validated life range, liquid/air drag, and display controls;
it does not claim unsupported foam/spray/bubble species classification. Raw
surface and whitewater capacities remain collapsed advanced controls.

#### Advanced Surface / Display
The detailed owner for viewport-facing liquid surface and particle display
settings. Surface mesh controls include visibility, generate surface, particle
radius, independent dilate iterations/strength, independent erode
iterations/strength, Box or Mean-Curvature smooth iterations/strength,
attribute color mode, and material. Liquid particle controls include
visibility, display fraction, display scale, attribute color mode, and particle
material. Whitewater particle controls include visibility, enable, display
scale, attribute color mode, and material.

**Surface Mesh** contains normal surfacing workflow: generation, resolution,
particle reconstruction radius, filtering, dilation, erosion, smoothing,
update cadence, surface visibility, attribute display, and material. Resolution
fields are shown only for the selected mode. **Live Surface Every** defaults to
1 so the viewport publishes every frame.

The inline collapsed **Advanced Surface** foldout is nested inside the
**Surface Mesh** subsection, after its viewport appearance controls. It
contains only **Leaf Scale**, **Dense Max**, and **Keep Sim Surface Field**.
Keep Sim Surface Field defaults off.

**Liquid Particles** follows as a peer viewport-visualization subsection with
visibility, display fraction, point scale, attribute color mode, and material.
It is not hidden under Advanced and is not mixed with Whitewater.

When a live solver is active, surface edits rebuild the current frame's surface
SDF and output mesh without advancing the simulation frame. Visibility and
material edits also apply to existing output or cache objects.

### Output & Cache
Output collection name and overlay toggle. **Optional Cache / Export** includes
cache enablement, path and playback range, **Export Cache Sequence**, and
**Export Current Frame**.

### Advanced

Manual capacities, sparse-grid implementation controls, field sources, snippets,
diagnostics, and validation are hidden by default but remain reachable.

#### Custom Warp Snippets
**Snippet Library** dropdown with ten ready-to-run examples plus your saved
snippets, and a **Load** button. Add / remove / apply custom kernels.
Per-snippet: enabled, kernel name, **Order** slot, **Run Over**, frame window
(**Start / End / Step**) for layering, code text block, and field/constant
arguments. Subpanels: **Snippet Editor** (inline source view with per-line
editing, **Validate**, **Auto Fix**, **Test**, **Save to Library**) and
**Shortcuts** (one-click insertion of fields, constants, and common
context/time/math code). See §5.

#### Diagnostics / Profiling
**Enable Profiling**, **Accurate GPU Timing** (CUDA-sync each phase),
report path, **Write Diagnostics Report**. Shows average per-phase timers
(Sourcing, Indexing, P2G, Forces, Viscosity, Pressure, G2P, Collision, Relax,
Surface, Whitewater, Custom, Lifecycle, Topology).

#### Validation / Benchmarks
**Run Validation Suite** (13 GPU tests, ~30 s). Benchmark **Frames** and
**Baseline** path, **Run Benchmark** / **Save Baseline**. Shows avg step time,
voxels/s, active voxels, and pass/regression vs baseline.

---

## 4. Attribute Visualization

Each output can be colored by a simulation attribute. Selecting a non-Material
mode writes a `display_color` color attribute and assigns an **attribute
material** that binds `display_color` into the Principled BSDF base color (via
`ShaderNodeAttribute`), so the coloring is visible in both the viewport and
final renders.

| Output | Available attribute modes |
|---|---|
| **Surface** | Material · Height · Speed · Temperature · Viscosity · Density |
| **Particles** | Material · Height · Speed · Temperature · Viscosity · Density · Age · Life |
| **Whitewater** | Material · Height · Speed · Age · Life |

Colors use a blue→cyan→green→yellow→red ramp, auto-ranged to the 5th–95th
percentile of the visible values.

**Tip:** surface attribute colors are sampled from the nearest particles onto
the mesh vertices (a host-side nearest-neighbour lookup). For very large
particle counts the lookup is subsampled to keep the viewport responsive.

---

## 5. Custom Warp Snippets

Custom snippets let you inject your own Warp kernel body into the solver
pipeline without editing the add-on. This is the Warp port of the original
custom-OpenCL-kernel feature.

1. In **Custom Warp Snippets**, click **Add Snippet**. A text datablock is
   created with an example body.
2. Set **Kernel Name**, **Order** (insertion slot), and **Run Over**:

   | Order slot | When it runs |
   |---|---|
   | `FIRST_STEP` | Start of each substep |
   | `BEFORE_SOURCING` | Before shape/height emission |
   | `BEFORE_FORCES` | After P2G, before gravity/forces |
   | `BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION` | Before pressure solve |
   | `FINAL_STEP` | End of each substep |

   | Run Over | Index variables in scope |
   |---|---|
   | `VOXEL_TILE` / `VOXEL_LINEAR` | `flat_voxel_index`, `leaf_index`, `local_x/y/z`, `global_x/y/z`, `world_position`, `voxel_size` (collision voxels skipped) |
   | `PARTICLES` | `particle_index`, `position`, `velocity` (dead particles skipped) |

3. Add **Arguments** — either bind a **solver field** (e.g.
   `temperature`, `velocity_u`, `pressure`, `surface_sdf`, `particle_age`) or a
   **constant** (float/int). The bound field appears as a `wp.array` under its
   name; constants appear as scalars.
4. Click **Apply Snippets** to compile. Compile errors are shown in the
   snippet's **Last Error** field. Compiled snippets run every substep until
   removed/disabled.

A time context (`frame`, `time`, `real_time_step`, `solver_time_step`,
`time_scale`, `substeps`) is always in scope. Generated kernels are written to
a temp directory and imported (Warp re-reads kernel source for codegen).

### Snippet library

The **Snippet Library** dropdown ships ten examples in four categories —
*Impressive* (Vortex Tornado, Curl Noise Turbulence), *Creative* (Moses
Split, Heartbeat Pulse), *Learning* (commented walkthroughs of grid-field
and particle snippets), and *Utility* (Whirlpool Drain, Wave Maker, Thermal
Buoyancy, Age Dye). **Load** appends the preset to the snippet stack with
its arguments pre-bound, so you can inspect, edit, and layer it like any
hand-written snippet. **Save to Library** writes the active snippet (code,
arguments, order, frame window) as JSON under the user config directory
(`fluidsim_snippets/`); saved snippets appear in the dropdown under
*My Snippets*.

### Validation, Auto Fix, and Test

Snippets can only run in the scene when they pass static validation — both
**Apply Snippets** and solver initialization gate on it, and the sim layer
re-checks before compiling. The **Snippet Editor** subpanel shows the source
of the active snippet (loaded example, saved preset, or hand-written) with
per-line edit/delete, and:

* **Validate** — static analysis: syntax, unknown names (with
  did-you-mean suggestions), fields referenced without a bound argument,
  unsupported constructs (imports, comprehensions, `.x/.y/.z` component
  access), and run-over mismatches. Issues appear in the panel.
* **Auto Fix** — conservatively repairs common mistakes: `;`, `//`
  comments, `&&`/`||`, `true/false`, C float suffixes (`1.0f`),
  `math.*`/`np.*` → `wp.*`, vector `.x` → `[0]`, smart quotes, tabs.
* **Test** — runs validation plus a real Warp module compile without
  attaching the snippet to the running scene.

A snippet's state icon shows valid (check), invalid (error), or edited
since last validation (question mark).

### Layering snippets

The stack runs every enabled snippet, grouped by **Order** slot, in list
order — so multiple snippets combine freely (e.g. Curl Noise + Wave Maker
+ a speed limit at `FINAL_STEP`). Each snippet also has a frame window:
**Start Frame** / **End Frame** (0 = open ended) and **Frame Step** (run
every Nth frame), letting you sequence effects over time — a tornado for
frames 1–100, then a heartbeat pulse from 120, with a dye pass every 4th
frame.

### Shortcuts

The **Shortcuts** subpanel (inspired by Houdini's VEXpression presets)
speeds up snippet writing: **Bind Data** adds a solver-field or named
float-constant argument to the active snippet and inserts a reference
comment; the **Context / Time / Math / Constants** buttons insert ready
snippets of code (world position, speed, dt, sine wave, lerp, clamp,
noise, curl noise, leaf constants ...) filtered to the snippet's run-over
domain.

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

**Validation / Benchmarks → Run Validation Suite** runs 15 GPU tests on
`cuda:0` (~30 s):

1. Sparse tiled leaf-grid coordinate/index roundtrip
2. Leaf-local addressing & start-vector reconstruction
3. Field precision parity (16/32-bit)
4. Particle-to-grid count/average/active masks
5. Grid-to-particle velocity/attribute sampling
6. Source shape coverage & particle attribute transfer
7. Collision/sink voxel-type bit behavior
8. Pressure projection divergence reduction
9. Jacobi & PCG residual convergence
10. Surface SDF build/filter parity
11. Viscosity solve stability
12. Whitewater churn/emission/advection behavior
13. Benchmark profile schema compatibility

Each test reports pass/fail with a detail line. 13/13 is the green bar.

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

## 9. Optional Cache & Export

FluidSim Warp is interactive-first; these are supplemental:

- **Cache** (Viewport / Render Output panel): enable **Enable Cache**, set a
  **Cache Path**, and each played frame is written as a compressed
  `fluidsim_NNNNN.npz` (positions, velocities, whitewater, surface
  verts/tris). Useful for reloading a run later.
- **Export Current Frame** writes the surface as `*_surface.obj` and particles
  / whitewater as binary `*_particles.ply` / `*_whitewater.ply` (particles
  include velocity).
- Every cached frame stores its applied settings signature. Sequence metadata
  records contiguous signature boundaries and marks `mixed_settings: true`
  when a bake contains more than one applied parameter set; it never presents
  that sequence as homogeneous. `settings_baseline` stores the first complete
  setting set and `settings_changes` stores only changed/removed fields for
  later signatures, avoiding repeated full snapshots during animation.
  Non-finite diagnostic floats use an explicit
  `__fluidsim_nonfinite_float__` JSON tag rather than non-standard JSON values
  or ambiguous ordinary strings.

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

Pending state belongs to each scene. Editing simulation resolution,
liquid sampling, leaf/whitewater capacity, domain, or a restart-only lifecycle
value in the middle of a run does
**not** silently erase that run. One neutral summary appears in Simulation and
the fixed header popover instead of injecting red action rows into individual
panels. Rewind to the timeline start to apply staged changes automatically, or
use **Restart Now**, **Reset Now**, or **Discard Changes** from that fixed
summary. A failed explicit restart retains its pending reasons. Discard never
changes or saves Blender add-on preferences; device changes must be selected
explicitly.

Particle capacity is different: **Initial Particle Reserve** is a starting GPU
allocation, not a hard runtime ceiling. With **Auto Grow Particle Pool** enabled,
FluidSim allocates a larger particle/binner pool, copies the live prefix, and
swaps only after successful CUDA allocation and synchronization. The authored
reserve is not rewritten and an active pool is never shrunk automatically. If
the live VRAM guard or allocation refuses growth, the selected advanced policy
either continues while deferring the whole source batch, or pauses Live while
preserving the initialized solver. No partial source batch is committed.

The FluidSim control in the fixed 3D Viewport header remains reachable while
the sidebar is scrolled. Its popover shows solver state, pending reasons, Live,
Restart/Reset/Discard and Release controls. The viewport overlay is informational
only. Blender does not support CSS-style sticky content inside a scrolling
sidebar, so FluidSim does not claim that behavior.

---

## 11. Tips

- **Start coarse.** Use a large Voxel Size (e.g. 0.08) while iterating, then
  refine. Resolution scales cubically.
- **Keep CFL sane.** Default CFL 1.0 with Min/Max Substeps 1/4 handles most
  cases; fast impacts may need more Max Substeps.
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

## ST-FLIP beta

Enable **Particles > ST-FLIP (Beta)** to test spatiotemporal sampling at large
time steps. The implementation includes temporal residuals, space-time P2G,
phase-field activation, optional variable pressure, local trajectory
substepping, and render-surface re-synchronization. It is beta because phase
three still needs multi-scene reference-error and long-run conservation work.
Keep the default feature disabled for production scenes until those results
are acceptable for the scene type. With automatic reference mass, the solver
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
