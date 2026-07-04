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
2. Create a **source** mesh, select it, and in **Sources** click
   **Tag Source**. Adjust its shape, emit rate, and initial velocity.
3. Create collider meshes, select them, and in **Colliders / Sinks** click
   **Tag Collider** (mesh SDF) or **Tag Sink** (removes liquid).
4. Set **Voxel Size** and **Leaf Capacity** in **Domain / Sparse Grid**.
5. Click **Initialize Solver**, then **Live: Off** to start, then Play.

---

## 3. Panel Reference (13 panels)

### Simulation
Core controls: **Quick Setup**, **Initialize Solver**, **Live** toggle,
**Reset**, **Release**, **Step 1 / Step 10**. Live status mirrors (particles,
active voxels, step time, substeps). Time Scale, CFL, Min/Max Substeps,
gravity (or *Use Scene Gravity*).

### Domain / Sparse Grid
Domain object picker, **Voxel Size**, **Leaf Capacity** (max 8³ leaves),
**Topology Expand** (dilation iterations around active fluid),
**Open Boundaries** (kill particles leaving the domain). Shows the resolved
resolution and active/total leaves.

### Particles
**Max Particles**, **Particles / Voxel**, **Transfer** (PIC / FLIP) with
**FLIP Ratio**, **Trace** (Euler / single-sample / midpoint / modified-midpoint
/ **RK3**), relaxation (enable / strength / mode), speed limit
(**Max Speed** + Limit Behavior + Percent), and **Count Stiffness**
(anti-clumping).

### Sources
**Tag Source** / **Clear**, **Inject Object Particles** (inject the active
object's points/vertices as liquid). Per-source settings: enabled, shape,
mode, emit rate, initial velocity, object-velocity inheritance, temperature,
viscosity, lifespan. **Height Fill** (base height, back mode, velocity,
particle emission). Lists all tagged sources/sinks/forces.

### Colliders / Sinks
**Tag Collider** / **Tag Sink**. Per-collider friction, restitution, and
velocity transfer toggle. Global restitution/friction. Lists colliders.

### Forces / Attraction
Per-object force regions (tag a Source as a force). Force type
(Off / Directional / Omnidirectional / Spin / Radial), strength, direction.

### Pressure Projection
**Solver** (Jacobi / **PCG**), **Iterations**, **Tolerance**, **Multigrid**
(Off / V-Cycle / Half-V-Cycle — coarse initial guess), **Threshold**
(relative pressure cutoff).

### Viscosity
Enable checkbox. **Solver** (Jacobi / PCG), **Viscosity** constant, **From
Particles** (per-particle viscosity attribute), **Iterations**, **Tolerance**.

### Surface / Whitewater
**Generate Surface**, particle radius scale, **Filter** (Dilate / Erode /
Box / Mean-Curvature-Flow), filter iterations & strength. **Whitewater**
enable, max whitewater, churn gain/decay/threshold, emission rate, min speed,
life min/max, bubble & spray drag.

### Viewport / Render Output
**Liquid View** (Surface / Particles / Surface + Particles). Surface, particle,
and whitewater visibility toggles. **Display Fraction** for particles.
Attribute modes (see §4). Surface/whitewater material pickers, output
collection name, overlay toggle. **Optional Cache / Export**: enable cache,
cache path, **Export Current Frame**.

### Custom Warp Snippets
**Snippet Library** dropdown with ten ready-to-run examples plus your saved
snippets, and a **Load** button. Add / remove / apply custom kernels.
Per-snippet: enabled, kernel name, **Order** slot, **Run Over**, frame window
(**Start / End / Step**) for layering, code text block, and field/constant
arguments. Subpanels: **Snippet Editor** (inline source view with per-line
editing, **Validate**, **Auto Fix**, **Test**, **Save to Library**) and
**Shortcuts** (one-click insertion of fields, constants, and common
context/time/math code). See §5.

### Diagnostics / Profiling
**Enable Profiling**, **Accurate GPU Timing** (CUDA-sync each phase),
report path, **Write Diagnostics Report**. Shows average per-phase timers
(Sourcing, Indexing, P2G, Forces, Viscosity, Pressure, G2P, Collision, Relax,
Surface, Whitewater, Custom, Lifecycle, Topology).

### Validation / Benchmarks
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

**Validation / Benchmarks → Run Validation Suite** runs 13 GPU tests on
`cuda:0` (~30 s):

1. Sparse octree coordinate/index roundtrip
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

---

## 10. Tips

- **Start coarse.** Use a large Voxel Size (e.g. 0.08) while iterating, then
  refine. Resolution scales cubically.
- **Keep CFL sane.** Default CFL 1.0 with Min/Max Substeps 1/4 handles most
  cases; fast impacts may need more Max Substeps.
- **Explosions?** If the sim blows up, lower **Max Speed** (Particles panel),
  enable closed-domain walls, or reduce FLIP Ratio toward PIC.
- **No whitewater?** Lower **Churn Threshold** and raise **Emission Rate**;
  whitewater needs turbulent impact regions.
- **Performance.** Watch the overlay timers; Pressure and Surface phases
  dominate at high resolution. Multigrid coarse guess can speed pressure.
