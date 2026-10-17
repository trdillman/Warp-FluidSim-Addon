# Changelog

## 4.1.0 — Static colliders, rigid contact and five new demos

### Changes

- **Body Type** on each collider: **Static**, **Animated** (the default,
  same behaviour as before) or **Dynamic**. Files saved before 4.1 keep
  working unchanged. A Static collider never moves: liquid and rigid
  bodies hit it, but nothing pushes it.
- **Rigid-body contact.** Dynamic bodies now land on and slide off Static
  and Animated colliders using each collider's Friction and Bounce, collide
  with each other, and rest stacked on one another. The same bake run twice
  gives identical results. Contact costs about 1 ms per substep with 8
  touching bodies; dense Animated collider meshes add about 0.8 ms per
  substep, only when a Dynamic body is in the scene.
- Three new Demo Gallery scenes, each showing one feature:
  - **Viscosity Race**: three identical columns of water, syrup and tar
    collapse side by side. Each source carries its own Viscosity (Domain >
    Viscosity > From Particles), so the fronts run out in viscosity order.
  - **Deep Plunge**: a block of water drops into a 70 cm deep tank with
    Narrow Band on. It runs on about a ninth of the particles a full-band
    run needs and settles at the same water level.
  - **Zero-G Droplet**: a 6 cm bar of liquid floats with gravity off, and
    surface tension pulls it into a round drop. With Surface Tension off
    it just hangs there as a bar.
- Two new rigid-coupling demos in the Demo Gallery:
  - **Density Ladder**: four blocks (250, 500, 750 and 1500 kg/m³) dropped
    into still water. They float in density order and the heaviest sinks.
  - **Water Wheel**: a six-paddle wheel locked to its axle, turned by an
    inflow stream.
- **Transfer** (PIC, FLIP, APIC) now sits beside **FLIP Ratio** in the
  sidebar's Simulate stage and the viewport-header popover, and the user
  guide explains what each choice and FLIP Ratio value looks like.
- Selecting a collider shows its surface and rigid-body settings in the
  3D View sidebar's Setup stage.

### Fixes

- **Surface tension no longer loses liquid on fine grids.** On voxels
  smaller than 1 cm the pressure solve stopped too early (its tolerance
  did not scale with voxel size), so surface tension squeezed the liquid
  away: a 2 mm bar shrank to a few voxels in half a second. The pressure
  tolerance now scales with voxel area below 1 cm; 1 cm and coarser
  domains, which includes every shipped demo, solve exactly as before.
  Fine-grid bakes may take a few more pressure iterations. Re-bake
  fine-grid caches.
- **Surface tension no longer churns the inside of the liquid.** It now
  measures curvature from a distance field rebuilt from the free surface
  instead of the particle field, whose gaps between particles read as
  tiny bubbles. Drops oscillate cleanly instead of gaining energy. This
  adds 8 bytes per voxel while Surface Tension is on.
- Fixed a possible crash after reopening a saved scene with Dynamic
  colliders and pressing Play or rewinding. Reset freed the pose action
  while Blender was still processing the frame change. It now frees the
  action right after the frame change finishes.
- Play no longer refuses a Dynamic collider whose keyframes you deleted.
  Deleting the last key left an empty action on the object, which still
  counted as animation.

## 4.0.1 — Install Warp from a file, and downloads that survive slow links

- **Install from File** in Add-on Preferences installs the certified Warp
  wheel from a `.whl` you downloaded yourself (pypi.org/project/warp-lang).
  FluidSim checks the file's size and SHA-256 against its lock before
  installing, and refuses any other file. No internet connection needed.
- **Install Warp** now resumes a dropped download from where it stopped
  instead of starting over, and only gives up after eight drops in a row
  with no progress. On a slow link that keeps moving it finishes.
- A Production Bake resumed from a checkpoint now continues Dynamic
  colliders bit-for-bit as if it had never stopped. Restoring no longer
  renormalises an already-unit rotation, which moved its last digit.
## 4.0.0 — Warp 1.18, Multigrid by default, Liquid Deck and two-way rigid coupling

### Upgrade notes

- **Simulation results change.** Bake caches again. New domains use the
  Multigrid pressure preconditioner.
- Files saved before 4.0 keep Jacobi. On load the add-on writes Jacobi into
  each such domain and stamps it with domain schema 4, so existing caches
  still match. The Bake stage shows a notice with **Use Multigrid** and
  **Keep Jacobi**; either choice dismisses it.
- Warp 1.18.0 (CUDA 13 build) is the certified runtime and needs an NVIDIA
  R580-series or newer driver. Warp 1.17 still loads and is reported as older
  than certified. Press **Install Warp** in Add-on Preferences to update.

### Changes

- **Liquid Preview** (Look stage toggle, `L` in the 3D View): a screen-space
  fluid look drawn from the published particles, with refraction-tinted
  thickness, a Fresnel rim and optional whitewater. Viewport only; it adds no
  GPU-to-CPU readback, and nothing is drawn or registered while it is off.
- Added the Liquid Preview pixel references
  (`tests/references/liquid_preview/`, synthetic seeded sphere + sheet).
- **Liquid Deck**: a stage bar, a command palette (`Ctrl+K`), a radial wheel
  (`Alt+Right Mouse`) and on-object handles in the 3D View, drawn over the
  scene instead of in the sidebar. The **Liquid Deck** preference (Auto, On,
  Off) controls it; under Auto it shows in new files and files saved by 4.0 or
  later, and files saved by 3.x open without it. The sidebar panels still work.
- Multigrid is the default pressure preconditioner. Under Jacobi a 2 m column
  settles about 6% below its true height; Multigrid holds it within 2%
  (`tests/solver/test_hydrostatic_column.py`). At a tight tolerance both
  converge to the same pressure (relative L2 at most 1e-4).
- Multigrid checks convergence after every iteration and stops at the first
  V-cycle that meets half the pressure tolerance, instead of running on to the
  next 5th iteration. Its coarsest level uses 4 smoothing sweeps instead of 16.
  A dam_break solve takes a median of 7 iterations instead of 10, and the demo
  runs at about 1.03x 3.2.1's frame time instead of 1.10x. Stopping at the
  full tolerance was faster still, but over a full run it lowered
  chromatic_vortex's liquid level by about 3%; at half the tolerance liquid
  volume and surface smoothness over all four full demo runs match the old
  rule within run-to-run variation.
- On Warp 1.18 the liquid surface is extracted from only the occupied tiles of
  the grid. The surface span cap has no effect with sparse extraction and will
  be removed in 4.1.
- Disabling the add-on keeps unregistering the remaining classes when one
  fails and prints a single summary line listing every class it left behind.
- **Export Liquid as USD** (Bake stage, next to Alembic) writes the baked
  surface, particles and whitewater over the cached range as one animated USD
  stage: points with per-frame positions and widths, plus a `fluidsim_*`
  primvar per exported attribute (speed, age, velocity and the rest). Points
  are the cached particles, not the viewport/render spheres. Alembic still
  exports only the surface; on Blender 5.2 it writes point clouds as empty
  transforms. The zero-radius capacity padding is not written. Render-enabled
  Dynamic colliders are written too, as in Alembic, with their solved poses
  as per-frame transforms.
- Experimental learned pressure initial guess, **hidden and off**. It did not
  meet its ship gate: even a perfect one-value-per-tile guess saves at most
  1.2x PCG iterations on the four demos, against the required 2x. No weights
  ship; turned on (scripting only), it reports "Learned guess: off (weights
  missing)" in the Live Readout. Off, results and cache identity are
  unchanged. Training tools are in `tools/learned_guess/`; numbers are in
  `docs/RELEASE_4_0_STATUS.md`.
- **Dynamic colliders** (two-way rigid coupling). A collider marked
  **Dynamic** floats, sinks or is pushed by the liquid according to its
  **Density**, **Drag**, **Initial Velocity** and **Lock** axes. Its pose is
  published as `matrix_world` keyframes on an add-on-owned action, cleared on
  Reset, saved per frame in the cache (format 3.1, resumable) and carried by
  Alembic and USD export. Forces are gathered in a fixed order without atomics, so
  bakes stay bitwise repeatable. A Dynamic object cannot also be a source or
  sink, and its Blender rigid body is forced to Animated (kinematic). Body to
  body and body to static-collider contact are not modelled in 4.0; bodies
  rest against the domain walls. The crown_splash demo gains a floating cork.

### Known limitations

- 4.0 is certified on Windows only (the GUI production gate). Linux builds
  are not CI-tested or certified, including USD export; Linux certification
  is planned for 4.1.

### Corrections

- An earlier known limitation said a deep column settles about a third low
  under Jacobi. Measured for 4.0, a 2 m column settles about 6% low, and
  Multigrid, now the default, holds it within 2%.

## 3.2.1 — Explicit Warp installation

- **Warp is no longer bundled.** The extension zip does not contain NVIDIA
  Warp wheels. **Install Warp** in Add-on Preferences downloads the locked
  Warp 1.17.0 wheel (CUDA 12) from PyPI, checks its SHA-256, and extracts
  it into this extension's owned user directory. A second press is idempotent.
- Foreign Warp copies are left untouched; offline access is refused before
  downloading. The extension declares `network` permission for the PyPI download.
- **Repair Warp** restores a damaged owned runtime safely after restarting
  Blender, including when version metadata is missing. Installation and repair
  support long Windows paths.
- The local installation helper supports an explicit `--install-warp` step;
  ordinary installation and `--probe-only` keep downloads disabled.
- Updated dependency notices and installation documentation.
- The certified Warp version (1.17.0, CUDA 12 build), solver behavior and cache
  format are unchanged. Existing 3.2.0 caches remain compatible.
- After upgrading from a bundled-Warp release, press **Install Warp**.

## 3.2.0 — Reliability, bake safety and speed

A full review of the add-on: about 80 fixes and 20 speed-ups.

### Upgrade notes

- **Simulation results change.** Bake caches again. Fixes that change
  results: pressure no longer leaves liquid compressible at domain walls and
  topology edges; particle-pool growth no longer drops a substep's grid
  transfer; compiled sources mark exactly their own leaves and jitter
  particles inside each voxel; rotated sources keep their corners; height
  profiles shape emission; viscosity sees mesh colliders and walls; JACOBI
  viscosity and Diffusion no longer slow moving liquid at its surface;
  collider velocity is correct for tilted and translating spinners; and
  several narrow-band, APIC, whitewater and surface-tension fixes.
- **Existing caches still play**: a cache no longer depends on which GPU baked
  it, and caches baked by 3.0/3.1 are still accepted.
- **Point, volume and field sources show Evaluation** (First Sim Frame /
  Every Sim Frame). First Sim Frame now really freezes the input. Files saved
  before 3.2 switch those sources to Every Sim Frame once, so they keep
  animating as they did.

### Production Bake

- A running bake owns the solver. Initialize, Step, Apply Preset, Apply or
  Revert pending changes, Inject, Cache Playback and Load Demo wait until it
  ends, in every scene. Settings and effect edits made during a bake are held
  and applied when it ends; keyframed solver settings do not animate inside a
  bake.
- Only Esc cancels a bake; right-click works normally again.
- Resuming with a later end frame extends the cache. A failed start restores
  the live solver, Live is restored after a bake, the bake starts from its
  start frame wherever the playhead is, and resumed colliders keep their
  velocity history.
- Command-line bakes accept seed-only scenes (crown_splash, paddle_stir), and
  cache errors exit with code 14.

### Fixed

- Ticking Color, Source ID or UID mid-run no longer stops the live session.
- Live Surface Tension and Mixing Strength edits reach the running solver;
  toggling Surface Tension or Species shows "Reset pending".
- Moving a cache into a folder inside itself is refused instead of deleting it.
- A collider outside the domain is ignored instead of stopping the frame.
- Whitewater with Type Hysteresis 0 no longer fails.
- A collider that stops moving no longer keeps an extrapolated pose.
- Revert, presets, capacity options and narrow-band refusals behave as shown.
- Custom Effects: safe mode closes several permission bypasses, Expert access
  needs the acknowledgement, generated names are reserved, effects use the
  timeline frame in preview and bake, and particle velocity effects move to
  Before Emission so PIC/APIC keep them.
- Quick Liquid and Set Selected As no longer tag the domain or outputs; Add >
  Fluid Domain draws as wire and does not render.

### Faster

- Point sources (8,000 points: 69 s to 1 s), CPU mesh admission (seconds to
  under 0.1 s), capacity planning on fine grids (9 s to under 1 s), checkpoint
  export, cache manifests and resume, static colliders, Bake panel and effect
  panel redraws, and display-fraction readback.

## 3.1.0 — Play drives the simulation, VDB export, pressure-solve fix

- **Changed: Play is the preview.** The Preview / Resume Preview button is
  gone. Blender's own Play (spacebar, Timeline, or the panel's Play button)
  starts and resumes the simulation on any scene with a domain. Pausing,
  editing an effect, or applying changes no longer leaves the spacebar
  playing the animation without simulating. Edits that would discard
  progress stay pending until you rewind to the start frame or use Apply
  Changes. Revoking trust in a running effect still stops the solver.
  Preview Cache and Set Up & Preview are now Play Cache and Set Up & Play.
- **Fixed: resting liquid slowly sank under the default pressure solver.**
  The Jacobi-preconditioned pressure solve gave up as "stalled" the first
  time its residual rose between two checks (iteration 25), which is normal
  conjugate-gradient behaviour, and projected an unconverged pressure. A
  resting tank lost about 25% of its height in 30 frames. A solve now
  stalls only after 100 iterations without a new best residual. The bug
  predates 3.0 (it is in 2.0 and 2.1). Simulation results change, so caches
  should be baked again; solves that used to stop early now run to
  convergence and can take longer.
- **Fixed: File > Append of a pre-3.0 domain** now converts its voxel size to
  Resolution Divisions, as opening the file already did.
- **New: Export Liquid Level Set as VDB** (Bake stage) writes a baked
  narrow-band cache as an OpenVDB sequence.

## 3.0.0 — Resolution Divisions, cache format 3, APIC, Multigrid and narrow band

See `docs/migrating_to_3_0.md` for what happens to a 2.x file, cache and
checkpoint.

### Upgrade notes

- **Caches must be baked again.** Cache format 3 is the only format 3.0
  reads. A 2.x cache is refused with "This cache was baked by FluidSim 2.x and
  cannot be played by 3.0. Re-bake, or open it in 2.1.1." and is never
  migrated. Solver checkpoints are format 2, so a 2.x checkpoint is refused as
  well.
- **Voxel Size is replaced by Resolution Divisions** (cells along the longest
  domain axis, default 64). A `.blend` saved before 3.0 is migrated on load:
  its divisions are `round(longest axis / voxel size)`, so the cell size moves
  to the nearest whole division. Resizing the domain now changes the cell size,
  not the cell count. New domains use 64 divisions (0.0625 m on the 4 m default
  box, previously 0.05 m).
- **A cell size that went through float32 no longer pads an extra leaf.** Some
  saved scenes get a smaller grid for the same settings and need a re-bake.
  The crown_splash demo grid changes from 88 × 88 × 56 to 80 × 80 × 56 cells,
  and dam_break's cell size from 0.03 m to 0.02995 m (187 divisions, same
  dimensions). The demo reference hash changes with them.
- **ST-FLIP (Spatiotemporal FLIP, Variable Projection) is removed.** See
  Removed.
- **The settings signature changed** when Narrow-Band Particles was added, so
  caches baked by an earlier 3.0 alpha build must be baked again.
- **The Scene tab shows only scene settings.** Every domain setting is on the
  Physics tab; saved values are unchanged.

### Added

- **Resolution Divisions** replaces the authored Voxel Size in Setup and in the
  Physics tab. The effective cell size and padded grid are shown read-only.
  `voxel_size` stays as a derived property: writing it sets the nearest whole
  divisions.
- **APIC (affine particle-in-cell)** as a third **Transfer** mode beside PIC and
  FLIP. It keeps swirl and rotation that PIC averages away. It uses its own
  transfer kernels and costs 72 bytes per particle only while selected. FLIP
  stays the default.
- **Multigrid pressure preconditioner.** Pressure > Preconditioner (shown for
  the PCG solver) chooses Jacobi or Multigrid. Multigrid is a symmetric,
  atomic-free V-cycle inside the existing CUDA graph and needed 13 PCG
  iterations against 123 for Jacobi on a 32³ obstacle test. Jacobi stays the
  default and is bitwise unchanged.
- **Export Surface as Alembic** in the Bake & Export stage. It writes the liquid
  surface of a verified baked cache as a mesh sequence through Blender's
  exporter and restores your playback state afterwards. It exports the surface
  mesh only, not particles or whitewater.
- **Playback only mode.** With Warp loaded but no CUDA device, the extension
  enables, loads, plays and exports caches. The UI says "Playback only — CUDA
  unavailable", the Simulate stage is disabled and baking is hidden.
- **Cache format 3.** Every frame records a stamp of every source and collider
  as evaluated at that frame. Resume refuses when an animation changed after
  the bake and names the first changed frame; **Re-bake from Frame N** keeps the
  frames before it and bakes the rest. The Bake stage explains a refused 2.x
  cache. The manifest also records the grid, transfer mode, platform and source
  schema, and each frame carries empty sparse `liquid_sdf_*` arrays.
- **Narrow-Band Particles** (Setup stage, off by default). Particles are kept
  only within **Band Width** voxels (default 3) of the liquid surface and the
  deep interior is carried on the grid. A 16 × 16 × 32 test tank needed 8×
  fewer particles (49,152 to 6,144) with the same column height under
  Multigrid. The toggle is greyed with the reason when color mixing,
  viscosity from particles, a scalar field source or attribute export is on,
  and Initialize refuses such a setup. While it is on, baked frames carry the
  liquid surface SDF, checkpoints save the level set, and the Live Readout
  shows Liquid voxels. With it off, results are bitwise unchanged.
- **Checkpoint format 2** reserves the `particle_affine` and
  `narrow_band_seed_state` slots.

### Changed

- Bake manifests record the real transfer mode (PIC, FLIP or APIC) instead of
  always FLIP.
- Switching Transfer between FLIP and APIC mid-simulation is allowed. Importing
  a checkpoint from the other mode drops or zero-fills the affine data without
  error.
- The Resumable Cache format is labelled "Resumable Cache" instead of "v2" (the
  saved identifier is unchanged, so old files still resolve).
- **Open Full Settings** opens the Physics tab when a domain is active.
- `SolverSettings` defaults now equal the Blender property defaults
  (`viscosity_constant` 0.05, `ww_bubble_drag` 4.0, `ww_spray_drag` 0.2,
  `speed_limit_percent` 0.5, `churn_decay` 2.0). Blender scenes always passed
  the property value, so only code that builds `SolverSettings()` directly sees
  a difference.
- Kernel prewarm is skipped when the runtime probe reports no CUDA.
- A Multigrid pressure solve is about 1.85× faster: the start V-cycle and the
  coarse-level setup replay from CUDA graphs, and the finest level visits only
  pressure unknowns. Results are bitwise unchanged.

### Removed

- **ST-FLIP.** The Spatiotemporal FLIP solver, its 14 settings, the
  Spatiotemporal Processing panel, its kernels, its validation gate and its
  checkpoint scalars are gone. A saved scene that had ST-FLIP on opens without
  it and simulates with the ordinary pressure projection; re-bake it.
- **Time Offset colour mode.** It read a channel only ST-FLIP filled, so it
  always showed zero. A scene saved with it opens with Material colouring.

### Fixed

- A float32 cell size no longer pads an extra leaf per axis (for example 64
  divisions padding to 72).
- A domain whose mesh has no size no longer derives a zero voxel size, which drew
  its particles with no radius.
- Capacity Options that coarsen the resolution apply again: they now suggest whole
  Resolution Divisions, never finer than the size they were costed at.
- Demos opened from the Demo Gallery keep their authored resolution instead of
  the default 64 divisions.
- Add-on unregistration finishes past a class that refuses to unregister and
  names it, instead of stopping at the first failure.
- `CudaDotProduct` checks array lengths before a launch.
- A refused bake start restores the Live flag.
- Narrow-Band Particles no longer grows the particle count past the full band
  on spreading liquid (dam_break reached the 600k cap against 256k). Reseeding
  fills only nearly empty band cells, to half the authored density, and never
  adds more particles than the band removed from the interior.
- Opening a baked cache no longer needs Warp to be importable. The cache settings
  signature is unchanged, so existing 3.0 caches stay valid.

### Internal

- `io/vdb_writer.py` is a pure-Python OpenVDB level-set writer that Blender 5.2
  loads as a Volume. It is not connected to bake, cache or the UI.

### Known limitations

- APIC retains 0.939 of a spinning disc's angular momentum over 100 frames
  (PIC 0.596, FLIP 0.940). Most of the loss is in the shared free-surface path,
  not in APIC.
- APIC, the Multigrid preconditioner and Narrow-Band Particles stay opt-in;
  FLIP, Jacobi and the full band remain the defaults. Measured on dam_break:
  - APIC costs about 8% less per substep than FLIP but keeps more energy, so
    splashes move faster and it takes about 35% more substeps (frames about
    1.26× as long).
  - Multigrid needs about 10× fewer iterations, but a solve takes about 1.34×
    as long as Jacobi (2.07 against 1.54 ms), and about 1.9× at twice the
    Resolution Divisions.
  - Narrow band saves particles on deep, calm volumes. On splashy scenes that
    turn into thin sheets it saves few and is slower (dam_break about 1.4–1.6×
    per frame).
- The cache frame stamp covers each source's world matrix and geometry.
  Collider motion driven by rigid-body simulation is not tested.
- Resume checks every baked frame's stamp with `frame_set`, so its start-up
  cost grows with the number of baked frames.
- Points and whitewater are not exported to Alembic: Blender 5.2's exporter
  writes only an empty transform for a point cloud.
- Under the default Jacobi preconditioner a deep liquid column settles about
  a third lower than it should within 30 frames, with or without narrow band.
  Use the Multigrid preconditioner where hydrostatic depth matters.
- If a narrow-band reseed needs more particles than Max Particles allows, the
  band is left under-seeded and the bake prints a notice to the console.
## 2.1.0 — staged sidebar, collider walls that slide and grip, faster viscosity

### Upgrade notes

- **Simulation results change wherever liquid touches a collider.** Fluid no
  longer clings to collider walls as a film, and collider **Friction** now
  acts on the grid, not only on particles. Friction 0 gives free-slip walls
  and 1 gives no-slip walls. Re-bake scenes with colliders; a high-friction
  collider now slows the liquid near it noticeably (the paddle_stir demo's
  swirl is weaker).
- **Caches must be baked again.** The new Export Attributes setting is part
  of the cache settings signature, so 2.0.0 caches no longer match.
- **The Scene tab is slimmer.** Transport, the five Cache panels and the
  3D View Live Tuning shelf moved into the FluidSim sidebar stages (below).

### Added

- **Workflow stages in the 3D View FluidSim sidebar:** 1 · Setup,
  2 · Simulate, 3 · Look and 4 · Bake & Export. Each control writes to the
  group that owns it (scene or active domain). The top panel is now a status,
  runtime and transport strip.
- **FluidSim header popover** in the 3D Viewport header (Play/Reset, Quality,
  live-tunable values) and a **FluidSim workspace** button that creates and
  then reuses a workspace with a large viewport, the sidebar and a timeline.
- **Export Attributes** switch on the domain, off by default. When off, only
  the state the simulation needs is published and cached; the per-attribute
  toggles apply only when it is on. Baking with it off measured about 10%
  faster than 2.0's default attribute set.
- **Custom Effect starter templates.** A new custom effect can start from a
  template instead of an empty definition.
- **Ten example effect packages** (`.fseffect`) and **twelve example scenes**
  under `resources/examples/custom_effects/`. The scenes open Untrusted
  Imported: review and trust the code, then Compile & Apply and resume Live.
- The Status & Runtime panel names the loaded FluidSim version and whether it
  is the installed extension or a repository checkout.
- `tools/release/install_local.py` installs a package into a given Blender and
  checks the FluidSim version, the Warp build that loaded and CUDA.

### Fixed

- Fluid that starts against a collider face slides down it instead of
  hanging on it as a film for seconds.
- Collider friction 0 and 1 no longer produce identical grid velocities.
- A Warp already loaded by another extension (such as Warp Foundry) that is
  older than 1.17.0 or a different major version is rejected before it is
  initialized, with a message naming the version.
- The copied **Headless Bake Command** points at the runner shipped inside the
  installed extension.

### Changed

- The **Dam Break** demo is a rocky cove: a reservoir released from between
  two dam abutments over a concrete apron onto a sand beach between two
  headlands, with a sea stack and a timber jetty as real colliders.

### Performance

- Viscosity solves reuse their captured CUDA graphs across time steps
  (213 → 30 captures on the dam_break viscosity benchmark): about 19% less
  time per frame with viscosity, 14% with viscosity and surface tension.
- Pressure solves reuse their CUDA graphs across tolerances (35 → 3 captures
  over 42 dam_break frames).
- Per-substep kernels replay recorded launches, and grid clears, velocity
  copies and PCG vector clears are fused into fewer kernels. Results are
  bitwise identical.
- Integer attributes are published to Blender without a copy again.

### Known limitations

- Linux x64 remains packaged but uncertified (see 2.0.0).
- A cache frame still records the simulation settings, not source or collider
  animation; bake again after changing an animation.

## 2.0.0 — Blender 5.2 LTS, bundled Warp 1.17 and the end of 1.0.x compatibility

### Upgrade notes

- **Open 1.0.x files in 1.1 first.** 2.0 opens files saved by 1.1.0 or later.
  A file saved by 1.0.x is flagged when it opens (a load message and an alert
  on the Setup and Physics tabs): its domain settings were not carried over,
  and a scene whose Custom Warp Snippets 1.1 never converted to Custom Effects
  will not start a simulation. The stored data is kept, so opening and
  re-saving the file in 1.1.0 recovers it.
- **1.0.x caches must be baked again.** A cache written by 1.0.x
  (`fluidsim_*.npz` frames with `fluidsim_sequence.json`) is recognized and
  refused rather than played; free it and bake again.
- **ST-FLIP is Experimental.** Its controls are labelled Experimental, and
  **Variable Projection** is now off by default. A saved ST-FLIP scene that
  never changed Variable Projection runs with it off; turn it back on in
  **Spatiotemporal Processing** to keep the 1.x result.
- **Scripts that write a setting to the wrong group now fail.** Writing a
  domain setting to `scene.fluidsim` (or a scene setting to
  `obj.fluidsim_domain`) used to create a plain Python attribute that the
  solver never read. It now raises `AttributeError` naming the right group.

See `docs/migrating_to_2_0.md` for the full upgrade path.

### Platform

- Blender 5.2 LTS only; Blender 5.1 is no longer supported.
- NVIDIA Warp 1.17.0 is bundled as checksum-locked wheels and installed with
  the extension. An external Warp install is neither used nor needed.
- Every gate lane (pytest, CUDA regressions, Blender background acceptance,
  release validators, the demo bake and the GUI production gate) runs against
  Blender 5.2 and Warp 1.17.

### Fixed

- A production bake with an animated source or collider (such as the
  paddle_stir demo) played back frame 1 and then refused every later frame as
  baked from different scene settings, and resuming it depended on the
  playhead. Playback now compares only the frame's settings signature, and
  bake and resume signatures are pinned to the bake start frame.
- NaN or infinite face weights no longer reach the pressure solve.
- Whitewater collision normals near the edge of a sparse collider now use the
  same gradient helper as the liquid instead of an out-of-range fallback.
- The ST-FLIP Jacobi solve reports its real residual instead of a constant.
- The particles-per-voxel ceiling (64) is one constant instead of six copies.
- **Apply capacity suggestion** can be undone with Ctrl+Z.
- The capacity dialog and its status lines no longer show garbled characters.
- Live Tuning knobs bind to the group that declares each setting, so changes
  made there reach the solver.
- One default per setting: RNA defaults, solver defaults and fallbacks agree,
  and length and speed settings carry Blender units.
- The height-field *Infinite* back mode, which the solver could not run, is
  removed. A file that saved it reports an actionable error.
- GPU self-validations release their solvers.
- An evaluated mesh used for point sourcing is released even if reading its
  vertices fails.

### Changed

- A failed CUDA-graph capture in the pressure or viscosity solve is shown as
  an error (`stat_native_solver_message`) instead of silently running slower.
- Examples 01–04 were regenerated with 1.1 so 2.0 opens them without flagging
  them as pre-1.1 files.
- Demo bakes are bitwise reproducible on the same GPU, driver and Warp build.
  The whitewater collision fix changes the demo reference hash.

### Known limitations

- Only Windows 11 x64 on a single GPU model (RTX 4080 SUPER) has a recorded
  GPU gate run. Linux x64 is packaged from the same source but not certified:
  on Ubuntu 24.04 under WSL2 it installs cleanly and passes every headless
  gate lane and 52 of 53 GUI checks. The viewport screenshot reads back black
  under WSLg, which stopped the gate before its last GUI lane. On WSL2, an NVIDIA Linux driver package installed alongside the
  WSL driver can hide the GPU from Warp; see the user guide.
- A cache frame records the simulation settings, not source or collider
  animation. Changing an animation after a bake does not invalidate the
  cache; bake again yourself.
- If an older Warp in Blender's user site-packages is loaded ahead of the
  bundled one, FluidSim refuses to run and names the Warp version it found.
  Remove the old Warp from the user site-packages.
- Speed matches 1.1.0. Against 1.0.x, scenes dominated by particle-to-grid
  transfer run about 4% slower (paddle_stir demo), which is the cost of the
  deterministic transfer 1.1.0 introduced. The other demos run at the same
  speed or faster.
- The reported slowdown after repeated resets with viscosity and surface
  tension on has not been reproduced outside the viewport.

### Removed

- The retired 1.x snippet runtime and its library: example and saved snippet
  presets, the snippet validator and auto-fix, shortcuts and the hidden
  library/editor settings.
- The Custom Warp Snippet settings, the `fluidsim.snippet_apply` operator and
  the snippet-to-Custom-Effects conversion. 1.1 already converted saved
  snippets when it opened a file.
- The 1.0.x settings migration and cache format 1 playback (see Upgrade notes).

## 1.1.0 — per-object domains, a Basic/Advanced interface and deterministic bakes

This is the bridge release to 2.0. Open and re-save your 1.0.x files in 1.1.0:
2.0 will only open files saved by 1.1.0 or later.

### Upgrade notes

- Domain settings now live on the Domain object (`obj.fluidsim_domain`) rather
  than on the scene. Files from 1.0.x migrate automatically when they are
  opened; save them afterwards to keep the migrated form.
- Simulation results differ from 1.0.255. Several GPU steps now run in a fixed
  order (see Changed), the viscosity solver now reaches its tolerance, and
  several demo scenes were recalibrated. Re-bake caches you want to compare.
- Blender 5.1, external Warp 1.16.0, Windows x64 and Linux x64 remain the
  supported contract.

### Added

- **Liquid color mixing** on the GPU, shown in the new `chromatic_vortex` demo.
- **Per-object domains.** Each Domain object carries its own settings, the
  **Add → Fluid Domain** menu creates one, and a scene can hold several. The
  **Physics Properties** tab is the home for domain settings; the Scene
  panels edit the active domain.
- **Basic and Advanced modes.** Basic, the default, shows the settings most
  scenes need; Advanced adds the solver panels.
- **Quality tiers** (Draft, Preview, Final) now scale voxel size, substeps,
  pressure iterations and particles per voxel. In 1.0.x the setting had no
  effect.
- **Transport and Lifecycle panels.** Playback controls and destructive actions
  have their own panels; destructive actions are styled as alerts.
- **Viewport HUD toggle** in the Display panel, and a compact **Live Tuning**
  shelf in the 3D View sidebar that replaces the full settings mirror.
- **Diagnostics** gathers the solver telemetry in one place.
- **Whitewater display fraction** thins the whitewater points drawn in the
  viewport without changing the simulation.
- **Custom Effects** parameters marked runtime-bindable update live, without a
  rebuild. Writing Warp code for an effect is now a developer preference
  (**Enable Effect Authoring**), and trust granted while editing lasts for the
  session only.
- **Artist preview.** One preview path for the viewport, and scene actions that
  roll back cleanly if setup fails.
- **Demo scene contract.** The four shipped demos (`dam_break`, `paddle_stir`,
  `crown_splash`, `chromatic_vortex`) must reopen, initialise and play from
  their own sources, and a check proves each one.

### Changed

- **Deterministic bakes.** Four order-dependent GPU steps (the pressure solver's
  dot product, particle-to-grid binning, shape emission and whitewater
  recycling) now run in a fixed order. Baking `dam_break` or `crown_splash`
  twice on the same GPU gives bit-identical frames. Mesh emitters and color
  mixing still accumulate in arrival order, so other scenes can differ
  slightly between bakes.
- The viscosity solver's iteration ceiling rose from 40 to 64 so it can reach
  its tolerance, and its convergence is now reported in the frame statistics.
- Surface-tension band propagation is the default SDF band mode.
- The legacy snippet runtime was removed. Custom Effects use the v2 pipeline.
- Developer tooling moved out of the add-on package into `tools/`.

### Performance

- Viscosity runs its solver loop inside a CUDA conditional graph. With
  viscosity on, a frame costs 1.55x the baseline instead of 7.80x.
- Surface attributes are gathered on the GPU: a surface frame fell from 175.5 ms
  to 20.7 ms in the benchmark scene.
- Live output copies only the used part of each device buffer (p95 −10%).
- Smaller wins: fused particle compaction counts, no duplicate whitewater
  reduction, and less per-substep launch overhead.

### Fixed

- Presets, live settings and quality tiers now apply to the domain after
  migration. Previously they wrote to the old scene settings.
- Edits to domain settings now show as pending and apply on reset. The export
  attribute toggles are visible again.
- Baking from the command line no longer crashes on exit.
- `crown_splash` whitewater now emits foam; the old threshold was unreachable.
- Emitters keep emitting after the particle capacity grows.
- The surface SDF band is wide enough for smoothing and filtering.
- Malformed user snippet presets are skipped with a message instead of failing.
- Errors now always outrank warnings in status displays.

### Security

- Custom Effects modules are written to a private per-user directory and are
  executed only when their digest matches.
- **Restart** refuses to delete a folder that is not a FluidSim cache.
- Closed a legacy path that could bypass Custom Effects trust.

### Release process

- Every pull request now runs hosted CI: lint, tests, CPU kernel validation, and
  a Blender build, install and smoke test. See `docs/RELEASING.md`.
- The GPU production gate validates the release kernels on CUDA and records the
  FluidSim version in its manifest.

## 1.0.255 — contextual Physics Properties and full viewport workflow

### Added

- Added contextual **Physics Properties → FluidSim** panels for the selected
  domain, generated outputs, and input-role objects without adding a second
  settings model.
- Added the complete Scene/Custom Effects workflow to **3D View → FluidSim**
  beneath one collapsed **Full Settings** parent, while retaining the compact
  activity/status and lifecycle panel as the primary viewport surface.
- Added guarded domain assignment/removal and confirmed single-output removal.
  Destructive output actions revalidate stable tags, ownership, and exact object
  identity immediately before deletion.

### Preserved during integration

- Scene/Object RNA, Pending Settings, Simulation Admission, Playback,
  Production Bake, and Output Publication remain the authoritative owners.
- Blender 5.1+, Windows x64, Linux x64, external Warp 1.16.0, and binary-free
  split-platform packaging remain the supported and validated contract.
- Removed legacy cache controls remain removed; the viewport mirrors only the
  current authoritative Scene Properties workflow.

### Fixed

- Multi-name kernel registration now validates the complete identity set before
  mutation, so duplicate or colliding names cannot leave partial registry or
  source-alias state.

## 1.0.253 — restore Windows x64 extension packaging

### Fixed

- Restored the long-standing Windows x64 product platform in the Blender
  extension manifest and split-platform release output.
- Separated supported product platforms from the subset of external runtime
  artifacts available on the current validation host. A Linux-only supplied
  Warp wheel can no longer silently reduce the extension to Linux-only.
- Added release tests that require both `windows-x64` and `linux-x64`, require
  both split archives, and require CI to retain both artifacts.

### Packaging and validation

- The release builder now emits `windows_x64` and `linux_x64` archives from the
  same binary-free Python/assets source. Neither archive bundles Blender or Warp.
- Windows Blender 5.1/Warp 1.16 execution is proven by the GUI-inclusive
  production gate on an RTX 4080 SUPER. Linux remains supported by the
  previously established runtime proof and the same binary-free source archive.

## 1.0.252 — native Properties workflow and Custom Effects UI integration

### Added

- Added a workflow-ordered Scene Properties hierarchy for Session, Setup,
  Domain, Height Fill, Physics, Custom Effects, Surface, Whitewater, Outputs,
  Cache/Production Bake, and Diagnostics.
- Added canonical role-aware Object Properties panels for Sources, Sinks,
  Colliders, Forces, Field Sources, and Volume Grid payloads.
- Added a compact 3D View status/launcher and a Scene Properties navigation
  action without duplicating authored settings.
- Added a mutually exclusive Start/Resume/Restart production-bake presentation
  layer while retaining strict execute-time verification.
- Added supplied-Blender validation for panel metadata, all authored draw paths,
  ten object-role/geometry combinations, Field Source payload coherence,
  save/reopen persistence, and registration cleanup.

### Changed

- Integrated Custom Effects v2 as a Scene Properties root between Physics and
  Surface. Artist flow is Effect Stack → Selected Effect → Workflow → Status;
  package administration and Warp Code are closed child panels.
- Warp Code now prefers an existing Blender Text Editor before repurposing the
  current area.
- Preferences now expose runtime/device selection and only the explicitly
  enabled developer override. The previous verbose toggle remains internal.
- Cache mutation dialogs display the exact resolved path or orphan inventory
  and revalidate it immediately before mutation.
- Field Source payload selection and region coverage stay coherent: Volume Grid
  payload selects Volume Grid coverage, while analytic coverage restores a
  Constant payload.

### Fixed

- Fixed Stop Live Preview, Stop Cache Playback, Cancel Bake, and Release Solver
  disappearing or becoming unusable after runtime readiness degraded.
- Fixed Step One Frame opening the multi-frame dialog and disabled stepping
  during cache playback or production bake.
- Fixed Surface + Particles controls being hidden because the UI tested the
  nonexistent `BOTH` identifier instead of `SURFACE_PARTICLES`.
- Fixed Height Fill errors appearing in a conditional source panel instead of
  the authoritative Session & Status surface.
- Fixed installed-package maintenance actions appearing for embedded-only
  Custom Effects.
- Fixed arbitrary existing directories qualifying for destructive Cache
  management. Filesystem services now fail closed unless the target is a
  recognized FluidSim cache or interrupted FluidSim transaction.
- Fixed cache-management visibility for an empty authored path and added
  byte-for-byte move verification for pre-manifest interrupted transactions.

### Compatibility and validation

- Cache-v1 playback, cache-v2 transactions/resume, Phase-3 attributes, Phase-4
  typed whitewater, Phase-5 surface tension, and Custom Effects v2 identities
  remain compatible. No solver or persistent effect schema was removed.
- External Warp 1.16.0+cu12 remains the runtime dependency. Blender and Warp
  runtimes are not bundled.
- Linux x64 is validated with the supplied Blender 5.2.0 LTS and supplied Warp
  1.16.0 package. The release workflow now requires and uploads a native
  Properties UI report alongside solver, Custom Effects, package, and installed
  lifecycle evidence.
- Updated the Phase-5 Blender regression to verify the native Scene Properties
  surface-tension panel and its Simulation Physics parent after removal of the
  legacy sidebar panel.
- CUDA solver certification and Windows x64 remain external gates.

## 1.0.251 — shippable Custom Effects and Warp Code workflow

### Added

- Replaced the prototype-facing Custom Warp Snippets UI with an artist-facing
  Custom Effects stack and a separate developer-grade Warp Code workspace.
- Added eight typed built-in effects, controller-object workflows, ordered
  mute/solo/duplicate/reorder operations, draft/tested/applied states, live
  dirty detection, profiling, and real one-frame preview actions.
- Added strict Effect schema/API 2 and ABI 1 contracts, an authoritative typed
  field/capability registry, deterministic generated source, source maps, real
  `wp.load_module` compilation, and bounded synchronized sandbox launches.
- Added exact-content SHA-256 trust, imported-code blocking, headless
  `deny|builtins-only|trusted` policy, and process exit code `17` for policy
  rejection before compilation or solver allocation.
- Added strict `.fseffect` packages with per-file hashes, path/symlink/duplicate
  rejection, decompression resource limits, atomic installation, identity-safe
  Keep Both behavior, semantic update comparison, safe package rename,
  malformed-package diagnostics, and library reveal/delete operations.
- Added three generated Blender demos with supplied-Warp numerical fixtures and
  Eevee F12 evidence: Vortex Force, stack-order behavior, and Warp Code
  authoring/trust.

### Changed

- Cache-v2 manifests, frame payloads, checkpoints, production-bake reports,
  headless reports, cache inspection, and support reports now carry the exact
  effective Custom Effects signature and ordered inventory.
- Production admission recomputes current Text datablock, parameter, controller,
  order, cadence, frame, binding, and capability state instead of trusting
  stored UI callback hashes.
- Runtime prerequisites are capability-derived. Particle-only effects no longer
  force sparse topology/SDF rebuilds.
- Legacy snippet operator IDs remain one-release migration adapters. Legacy
  `VOXEL_TILE` maps to Active Voxels with a warning; imported legacy code is
  disabled and untrusted.

### Fixed

- Fixed import-only false-positive validation by forcing Warp JIT compilation
  and a synchronized guarded launch.
- Fixed resume continuing after source/parameter/controller changes by binding
  the effective stack identity to every transactional cache/checkpoint seam.
- Fixed failed Apply removing the prior working stack; candidate replacement is
  atomic and invalid enabled effects fail closed.
- Fixed runtime errors allowing later effects or frames to continue; execution
  now raises a structured error and remains latched failed until a successful
  candidate swap.
- Fixed built-in trust spoofing, package identity shadowing, filename-only Keep
  Both behavior, duplicate instance ambiguity, stale UI status, and diagnostics
  attaching to the wrong instance.
- Fixed effect-package zip-bomb exposure by checking member count, per-member
  size, total uncompressed size, and compression ratio before decompression.

### Compatibility and validation

- Cache v1 remains playback-only. Older cache-v2 frames remain playable, but a
  resume with active Custom Effects requires matching effect identity; rebake
  when identity is absent or differs.
- External Warp 1.16.0+cu12 remains the runtime dependency; no Blender or Warp
  runtime is bundled.
- Supplied Blender 5.2.0 LTS and supplied Warp 1.16.0 CPU proof cover the product
  and integration paths on Linux x64. CUDA numerics/determinism/performance and
  Windows x64 remain external release gates.

## 1.0.250 — stable surface-tension physics and final foundation release

- Added continuum-surface-force capillarity from the simulation liquid SDF.
- Added a Brackbill-style capillary timestep limit, explicit clamp diagnostics,
  curvature/acceleration statistics, and optional 24-byte-per-voxel scratch state.
- Added physical Water Surface Tension and stylized High Tension presets.
- Added supplied-Warp CPU kernel validation and mandatory CUDA release gates.
- Preserved cache-v1 and older cache-v2 frame playback. Surface tension adds
  no persistent stochastic state, but cross-release production resume remains
  fail-closed: rebake under 1.0.250 to obtain its exact solver/settings signature.
- Fixed duplicate Spray UI labeling and stale pre-advection SDF reuse when
  capillarity and whitewater are both active.
- Fixed same-key SDF reuse after continuous source emission, first-frame
  capillary substep bypass, reversible optional-scratch accounting, and
  per-frame maximum capillary diagnostics across multiple substeps.
- Fixed the authored Time Scale = 0 pause path so capillary preview and
  particle sanitation remain finite instead of dividing by zero.
- Fixed direct particle injection invalidation so same-frame physical SDF and
  render-mesh caches cannot omit newly appended particles.
- Kept static/source tests independent of Warp so the release identity and
  packaging gate remains runnable before an external runtime is provisioned.
- Expanded SDF-normal support for smoothing, made optional scratch allocation
  transactional, and closed validation-resource leaks on failure paths.
- Strengthened the supplied-Warp proof to exercise the real solver limiter
  before first-frame emission, and removed checkout-name assumptions from all
  Phase-3/4/5 Warp regression harnesses.
- Expanded the physical SDF and sparse topology halo for every capillary
  stencil, including authored splat radii, and included the resulting topology
  and optional scratch costs in capacity and VRAM-safe recommendations.
- Prevented legacy presets from retaining hidden capillary state after a Phase-5
  preset, and made release JSON/checksum metadata strict atomic replacements.

All notable user-facing changes are recorded here. FluidSim follows a three-part release version and declares cache compatibility explicitly.

## 1.0.249 — deterministic typed whitewater outputs

### Added

- Stable persistent whitewater type values: foam `0`, bubble `1`, and spray `2`; value `3` remains reserved and is rejected by current cache/checkpoint validators.
- Deterministic host/device classification with SDF depth, speed, churn, and hysteretic per-step transitions that prevent threshold flicker.
- Type-specific foam advection/preservation, bubble fluid drag/buoyancy, spray air drag/gravity, and independent lifespan multipliers.
- A mandatory int32 `fluidsim_whitewater_type` point attribute in Phase-4 cache-v2 frames and solver checkpoints, with an explicit whitewater-type checkpoint schema version.
- Separate `FluidSim Foam`, `FluidSim Bubbles`, and `FluidSim Spray` Blender outputs and materials, alongside the backward-compatible `FluidSim Whitewater` combined output.
- A P4 Geometry Nodes asset containing six revisioned groups: surface, liquid particles, combined whitewater, foam, bubbles, and spray.
- Supplied-Warp validation for host/device classification parity, typed admission, type-specific lifetimes, aligned compaction, type counts, checkpoint restoration, and the one-pool invariant.
- Supplied-Blender validation for all six output shells, typed cache publication, legacy-unclassified preservation, user-modifier protection, save/reopen, and an Eevee F12 three-material render.

### Changed

- Whitewater remains one GPU simulation pool. Separate artist outputs are filtered publication views and do not triple solver storage.
- The Splash + Whitewater preset publishes both combined and separate typed views; other workflows retain the combined-only preview default.
- Legacy cache-v1 and Phase-3 cache-v2 whitewater without type state is preserved as `-1` unclassified on the combined output. It is never silently labelled foam, bubble, or spray.
- Whitewater type is mandatory and cannot be disabled in the render-attribute inventory.

### Compatibility

- Cache-v1 and Phase-3 cache-v2 frame playback remains supported. Earlier whitewater is combined/unclassified because no physical type state was recorded.
- Phase-3 solver checkpoints are not resumable as Phase 4 because deterministic type and transition state cannot be reconstructed; rebake to create Phase-4 checkpoints.
- Blender 5.2.0 LTS and Linux x64 remain the validated host/platform. NVIDIA Warp 1.16.0+cu12 remains external and is not bundled.
- CUDA solver determinism, NVIDIA memory/performance behavior, and Windows x64 remain open hardware/platform gates on this host.

## 1.0.248 — artist-ready attributes and render nodes

### Added

- One authoritative Phase-3 attribute schema for surface, liquid-particle, and whitewater outputs, including stable Blender names, domains, data types, cache names, and checkpoint policy.
- Opt-in persistent liquid-particle color, source-ID, and UID channels. UID allocation is monotonic and transactional; source identity is deterministic across Blender sessions.
- Derived speed and projected-grid vorticity output, plus Blender motion data stored as per-scene-frame displacement in the conventional `velocity` attribute while `fluidsim_velocity` remains world units per second.
- Strict cache-v2 attribute inventory metadata and fail-closed shape, dtype, finiteness, alignment, and UID validation. Legacy cache frames without a Phase-3 inventory remain readable.
- A generated, revisioned Geometry Nodes asset for surface, liquid particles, and whitewater, together with idempotent installation and explicit update controls that never claim user-owned modifiers or node groups.
- Attribute-aware default materials for source color and viscosity, host-readback/cache-cost estimates, and a dedicated Render Attributes panel.
- Supplied-Warp validation for optional-channel allocation, source metadata, scan compaction, checkpoint restoration, prepared-packet velocity preservation, and the vorticity kernel.
- Supplied-Blender validation for attribute publication, cache round-trip, save/reopen, typed modifier inputs, evaluated Geometry Nodes, motion data, user-asset protection, and non-empty F12 renders.

### Changed

- Point render groups realize particle instances before material assignment so point-domain simulation attributes survive into shader-visible F12 geometry.
- Guided setup installs the current bundled render-node revision on all three owned output objects.
- Persistent render-attribute inventory changes are Reset-tier and participate in solver/cache compatibility signatures; viewport debug coloring remains a separate presentation path.

### Validation status

- The pure-Python suite, source verification, supplied-Warp CPU execution, supplied-Blender attribute publication, save/reopen, Geometry Nodes evaluation, and Eevee F12 rendering pass for this increment.
- The validation host exposed no NVIDIA CUDA driver/device. CUDA solver numerics, CUDA determinism, GPU memory behavior, GPU performance, and hardware renderer regressions remain explicit release gates.

### Compatibility

- Blender 5.2.0 LTS and Linux x64 remain the validated host/platform.
- NVIDIA Warp 1.16.0+cu12 remains an external dependency and is not bundled.
- Cache v1 playback remains supported. Existing cache-v2 frames without the Phase-3 attribute schema remain readable; newly enabled persistent channels require Reset/rebake and are checkpointed.

## 1.0.247 — resumable production workflow and guided first run

### Added

- Strict cache-v2 manifests, transactional frame/checkpoint commits, SHA-256 and file-size verification, contiguous committed-frame enforcement, and fail-closed corruption/orphan handling.
- Solver checkpoint export/import for liquid, whitewater, sparse topology, source lifecycle/RNG, ST-FLIP temporal/RNG/reference-mass state, solver frame, and simulation time.
- Responsive one-frame production-bake controller with safe cancellation, restart, resume, Blender-frame restoration, and live-session restoration.
- Farm-friendly background bake command with JSONL progress, atomic final reports, signal cancellation, and process exit codes `0`, `2`, and `10`–`16`.
- Six strict product presets, atomic user-preset storage, selected-object guided setup modes, owned FluidSim collection structure, stable domain/output creation, and capacity-preflight reporting.
- Four generated example scenes covering live inflow, moving colliders, viscous liquid, and whitewater cache-v2 bake/resume.

### Changed

- `fluidsim.setup_scene` now implements the artist-facing selected-object workflow while preserving its public operator ID.
- New production bakes write cache v2; cache v1 remains readable for playback.
- The installable extension includes only the installed headless entry point from `release/`; build, smoke, wheel-fetch, and example-generation tooling remains excluded.

### Validation status

- Pure-Python, source verification, supplied-Blender registration, installed-package lifecycle, preset/example save-reopen, and supplied-Warp CPU execution are release evidence for this increment.
- CUDA solver numerics, CUDA resume determinism, NVIDIA memory behavior, and GPU performance remain an explicit hardware release gate; they are not certified by a CPU-only environment.

### Compatibility

- Blender 5.2.0 LTS and Linux x64 remain the validated host/platform in this deliverable.
- NVIDIA Warp 1.16.0+cu12 remains an external dependency and is not bundled.
- Cache v1 playback remains available; resume requires cache v2.

## 1.0.245 — external runtime packaging policy

### Changed

- Removed NVIDIA Warp and all Blender/Warp runtime binaries from FluidSim delivery archives.
- Warp 1.16.0+cu12 remains checksum-locked as an external validation/runtime dependency only.
- Release inspection now rejects bundled `.whl` files, Blender runtime archives, Warp package directories, and Warp distribution metadata.
- Blender smoke tests load Warp from a separately provisioned validation environment and prove it is outside the installed FluidSim extension.

### Compatibility

- Solver numerics and cache-v1 compatibility are unchanged.
- Blender 5.2.0 LTS / Linux x64 remains the certified P0 target.

## 1.0.244 — P0 release foundation

### Added

- Canonical `fluidsim_warp` extension identity and centralized runtime version metadata.
- GPL-3.0-or-later license file and a complete embedded Warp-license inventory.
- Certified P0 target: Blender 5.2.0 LTS on Linux x64 with NVIDIA CUDA.
- Namespace-safe generated custom Warp modules and extension-scoped user snippet storage.
- Checksum-locked, package-contained NVIDIA Warp 1.16.0+cu12 binary.
- Cached runtime/device preflight, enumerated CUDA selection, and privacy-safe support diagnostics.
- Deterministic source verification, platform archive inspection, and SHA-256 artifact manifests.
- Clean-profile install, enable, disable, re-enable, fresh-process reload, bundled-Warp execution, and uninstall smoke coverage.
- Self-hosted NVIDIA gate definition for all 16 GPU validations plus ten preview frames.

### Changed

- Blender extension metadata now comes exclusively from `blender_manifest.toml`.
- Registration no longer searches external Python installations for Warp.
- Initialize, live playback, stepping, and cache baking are blocked with actionable messages until the packaged runtime and selected CUDA device are available.
- Render-handler teardown removes only exact callbacks owned by FluidSim.

### Compatibility

- Existing cache-v1 frame output remains readable. P0 does not change the cache payload format or solver numerics.
- Windows x64 is not shipped in this P0 archive because a matching Warp 1.16.0+cu12 Windows wheel and Windows Blender 5.2 runtime were not supplied for validation.
