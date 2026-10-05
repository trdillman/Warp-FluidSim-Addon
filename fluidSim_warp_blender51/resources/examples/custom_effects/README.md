# FluidSim Custom Effects validation examples

These scenes are generated from real Custom Effects v2 kernel executions using the
certified Warp package on the CPU validation backend.

They are static validation/review artifacts, not playable FluidSim demos. They
contain precomputed before/after geometry and evidence from the generator, but
no live Domain-backed simulation that advances through Blender's frame
handlers. Do not use them to satisfy a request for a Custom Effects demo; a real
demo must meet the repository's `docs/DEMO_SCENE_CONTRACT.md`, including live
effect execution in a non-empty solver.

- **05 — Vortex Force:** compares the initial particle fixture with the validated
  Vortex Force result and trajectory lines.
- **06 — Stack Order:** demonstrates that Radial Pulse → Velocity Damping differs
  from Velocity Damping → Radial Pulse, while independently checking the analytic
  kinetic-energy damping ratio.
- **07 — Warp Code:** includes a developer-authored Lateral Wave Text datablock and
  package. The code was compiled and executed for evidence, but remains disabled and
  untrusted in the distributed `.blend` until the user reviews its exact SHA-256.

CPU validation proves compilation, bounded execution, deterministic fixture results,
and Blender publication. It does not prove live solver playback, full FluidSolver CUDA
numerics, or GPU performance; the NVIDIA CUDA release gate remains required on
compatible hardware.

## Packaged Example Custom Effects (.fseffect)

The following 10 curated custom effect packages are bundled under `resources/examples/custom_effects/` as artist examples and templates for the Custom Effects system (`Scene Properties → Custom Effects → Effect Packages → Import Effect Package`):

| Package | Effect ID | Domain | Stage & Cadence | Category | Description |
|---|---|---|---|---|---|
| `thermal_updraft-1.0.0.fseffect` | `example.thermal_updraft` | `PARTICLES` | `AFTER_PARTICLE_TRANSFER` / substep | Thermal | Convective thermal buoyancy accelerating warm particles upward based on temperature delta. |
| `point_attractor-1.0.0.fseffect` | `example.point_attractor` | `PARTICLES` | `AFTER_PARTICLE_TRANSFER` / substep | Forces | Central spherical attractor or repeller pulling fluid toward an artist Empty controller. |
| `helical_vortex-1.0.0.fseffect` | `example.helical_vortex` | `PARTICLES` | `AFTER_PARTICLE_TRANSFER` / substep | Forces | Spiraling corkscrew vortex combining tangential swirl with axial drive along a cylinder axis. |
| `directional_wind-1.0.0.fseffect` | `example.directional_wind` | `PARTICLES` | `AFTER_PARTICLE_TRANSFER` / substep | Forces | Aerodynamic drag and periodic gusting force within a localized controller bounding sphere. |
| `centrifugal_spinner-1.0.0.fseffect` | `example.centrifugal_spinner` | `PARTICLES` | `AFTER_PARTICLE_TRANSFER` / substep | Forces | Radial $\omega^2 r$ centrifugal ejection perpendicular to a rotating shaft. |
| `surface_tension_bead-1.0.0.fseffect` | `example.surface_tension_bead` | `ACTIVE_VOXELS` | `BEFORE_PRESSURE` / substep | Surface | Damps grid velocity in sparsely filled surface voxels (from the live particle count) so sheets and droplets stay together. |
| `viscous_heating-1.0.0.fseffect` | `example.viscous_heating` | `PARTICLES` | `AFTER_PARTICLE_UPDATE` / substep | Thermal | Friction dissipation converting high particle speeds into temperature increases. |
| `periodic_ripples-1.0.0.fseffect` | `example.periodic_ripples` | `PARTICLES` | `AFTER_PARTICLE_TRANSFER` / substep | Waves | Concentric circular traveling harmonic ripple waves radiating outward from a disturbance controller. |
| `bounding_box_confinement-1.0.0.fseffect` | `example.bounding_box_confinement` | `PARTICLES` | `AFTER_PARTICLE_TRANSFER` / substep | Boundary | Smooth elastic restorative containment barrier keeping particles inside a 3D box. |
| `voxel_viscosity_decay-1.0.0.fseffect` | `example.voxel_viscosity_decay` | `ACTIVE_VOXELS` | `AFTER_PARTICLE_TRANSFER` / substep | Grid | Blends particle-splatted grid viscosity toward a base value by exp(−rate × time), so a thick fluid thins over the shot. Needs viscosity enabled with Viscosity From Particles. |

### How to use an example effect in Blender

1. Open Blender with the FluidSim add-on enabled.
2. Go to **Scene Properties → Custom Effects**.
3. Expand **Effect Packages** and click **Import Effect Package**.
4. Browse to `fluidSim_warp_blender51/resources/examples/custom_effects/` and select any `.fseffect` file.
5. In **Effect Stack**, select the imported effect, inspect or adjust its typed parameters, and assign any scene controller Empty if needed.
6. Click **Review & Trust** to approve local execution, then click **Compile & Apply** to include the effect in simulation playback or cache baking.

## Example Scenes (`scenes/`)

Starter scenes, one per example effect plus two layered stacks, are provided under `fluidSim_warp_blender51/resources/examples/custom_effects/scenes/`. Each contains a designated Domain (`obj.fluidsim_domain`), a configured fluid source, a framed camera and light, and the effect stack with its Warp code embedded.

These are **examples, not FluidSim demos** (the demos are the add-on's own proof scenes): they are not in the `tools/demos/check_demo_blends.py` inventory and carry no per-effect runtime evidence under `docs/DEMO_SCENE_CONTRACT.md`. The scenes ship **Untrusted Imported** on purpose — trust is a local decision stored in your profile and is never shipped inside a `.blend`. After opening one:

1. In **Scene Properties → Custom Effects**, click **Review & Trust** and check the code.
2. Click **Compile & Apply**.
3. Resume **Live** (opening an untrusted stack pauses the preview) and play the timeline.

| Scene File | Primary Effect / Layers | Description & Highlights |
|---|---|---|
| `01_thermal_updraft_example.blend` | Thermal Updraft | Heated liquid column rising rapidly against gravity via buoyancy forces. |
| `02_point_attractor_example.blend` | Point Attractor | Off-center spherical fluid block pulled into orbit around an Empty controller. |
| `03_helical_vortex_example.blend` | Helical Vortex | Liquid cylinder driven in a corkscrew spiral upward and outward around the central axis. |
| `04_directional_wind_example.blend` | Directional Wind | Horizontal aerodynamic drag and periodic gusts sweeping fluid sideways. |
| `05_centrifugal_spinner_example.blend` | Centrifugal Spinner | Fluid column flung outward into a hollow ring/curtain by rotating shaft forces. |
| `06_surface_tension_bead_example.blend` | Surface Tension Bead | Inflow droplet coalescing and forming coherent liquid beads with edge surface tension. |
| `07_viscous_heating_example.blend` | Viscous Heating | Fast-moving falling fluid stream experiencing shear dissipation and kinetic heating. |
| `08_periodic_ripples_example.blend` | Periodic Ripples | Shallow pool displaying harmonic concentric ripples radiating from a center disturbance. |
| `09_bounding_box_confinement_example.blend` | Bounding Box Confinement | Expanding droplet column gently cushioned and bounded by elastic barrier planes. |
| `10_voxel_viscosity_decay_example.blend` | Voxel Viscosity Decay | Thick viscous fluid pool whose active-voxel viscosity decays over time toward water-like flow. |
| **`11_layered_atmospheric_tempest_example.blend`** | **Multi-Layered:** Updraft + Wind + Box Confinement | **Layered example 1:** Dynamic atmospheric simulation layering convective updraft, gusting horizontal wind, and soft bounding box confinement to produce an enclosed turbulent storm column. |
| **`12_layered_hydro_centrifuge_example.blend`** | **Multi-Layered:** Spinner + Helical Vortex + Heating | **Layered example 2:** Multi-physics fluid centrifuge layering radial centrifugal ejection, upward helical swirl, and shear heating dissipation to simulate an industrial separating reactor. |

### Regenerating and verifying

The `.fseffect` packages come from `example_effects.py`; the scenes come from `tools/release/build_custom_effect_example_scenes.py`. The builder trusts effects only in a throwaway store, so your own trust list is never touched and no trust is saved in the `.blend` files.

```powershell
blender --background --python tools/release/build_custom_effect_example_scenes.py -- --all --device cuda:0
blender --background --python tests/acceptance/blender_custom_effect_example_scenes_acceptance.py
```

The acceptance script opens every scene, checks that it ships untrusted, applies the stack once trusted, and checks that particles advance between frames. It runs on `cuda:0` unless `FLUIDSIM_TEST_DEVICE` says otherwise, and prints `CUSTOM-EFFECT-EXAMPLES-ACCEPTANCE-PASS` when all 12 pass. Both scripts need Warp importable from Blender's Python; point `BLENDER_USER_EXTENSIONS` at a private folder rather than adding `--factory-startup`, which can delete the installed Warp wheels from your real profile.

