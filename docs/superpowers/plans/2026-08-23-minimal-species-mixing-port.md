# Minimal Species Mixing Port Implementation Plan

> **Goal:** Move the smallest proven color-mixing implementation from the 1.0.248 species prototype onto a clean worktree based on FluidSim 1.0.255 `master`, while preserving current surface tension and UI behavior.
>
> **Architecture:** Keep species mixing as one optional GPU-resident particle/grid transport attached to the existing solver substep. Sources initialize particle composition; P2G accumulates represented volume and species mass; G2P relaxes particle composition toward the cell mixture; publication derives viewport color. Blender adapters expose only the enable, mixing-strength, palette, and per-source composition controls.
>
> **Tech Stack:** Blender 5.1 Python add-on, Warp/CUDA kernels, pytest, Blender background acceptance scripts.
>
> **Spec:** `docs/superpowers/specs/2026-08-22-dynamic-liquid-species-design.md`
>
> **Constraints:** Base the new worktree on exact commit `b1336ac82cb407fe12811c91bd08f3d811ad12fa`; preserve the prototype worktree unchanged; keep the sparse 8x8x8 leaf-grid contract; add no species-specific substeps; preserve the existing surface-tension force position; do not use archived experiment branches; do not port the prototype's replacement Physics panel, cache/report/profiling changes, or demo-only helpers. This emergency port preserves the currently proven two-active-species red/blue workflow; arbitrary-N authoring is a separate extension.

---

## Critical path and stop rule

The critical path is: clean 1.0.255 worktree -> core GPU transport -> current source/UI adapters -> viewport color -> focused proof. If the transplant requires a cache-format migration, capacity-planner rewrite, or a second transport subsystem, stop at the last passing commit and report the dependency instead of expanding scope.

### Task 1: Create the clean port worktree and isolate the source patch

**Files:**
- Read only: `C:/tmp/liquid_warp_species_v2/fluidSim_warp_blender51/sim/species.py`
- Read only: the species-related diffs in `C:/tmp/liquid_warp_species_v2`
- Create worktree: `C:/tmp/liquid_warp_species_master`
- Create branch: `codex/species-master-port`

**Interfaces:**
- Consumes: prototype worktree `C:/tmp/liquid_warp_species_v2` and base commit `b1336ac82cb407fe12811c91bd08f3d811ad12fa`.
- Produces: a clean 1.0.255 worktree containing no prototype changes yet, plus a bounded list of transplant symbols.

- [ ] **Step 1: Confirm the source and base without modifying either checkout**

```powershell
git -C C:\tmp\liquid_warp_species_v2 status --short
git -C C:\tmp\liquid_warp_species_v2 rev-parse --short HEAD
git rev-parse master
git show master:fluidSim_warp_blender51/__init__.py | Select-String '1, 0, 255'
```

Expected: the prototype remains dirty at `2a05e6d`; `master` resolves to `b1336ac...`; the master add-on reports version 1.0.255.

- [ ] **Step 2: Create the isolated worktree from the exact master commit**

```powershell
git worktree add -b codex/species-master-port C:\tmp\liquid_warp_species_master b1336ac82cb407fe12811c91bd08f3d811ad12fa
git -C C:\tmp\liquid_warp_species_master status --short
```

Expected: the second command prints nothing.

- [ ] **Step 3: Extract only the symbols that implement the working mixer**

Use narrow symbol searches against the prototype; do not copy whole files except a new, trimmed `sim/species.py`:

```powershell
rg -n "SpeciesTransport|species_concentration|represented_volume|mixing_rate|with_species" C:\tmp\liquid_warp_species_v2\fluidSim_warp_blender51\sim C:\tmp\liquid_warp_species_v2\fluidSim_warp_blender51\blender C:\tmp\liquid_warp_species_v2\fluidSim_warp_blender51\properties.py
```

Keep only functions called by the solver, sourcing path, settings adapter, or viewport publication path. Exclude cache serialization, profiling receipts, production-gate wiring, old `panels.py` additions, and demo builder utilities.

- [ ] **Step 4: Copy this plan into the new worktree and commit the clean planning checkpoint**

```powershell
New-Item -ItemType Directory -Force C:\tmp\liquid_warp_species_master\docs\superpowers\plans | Out-Null
Copy-Item docs\superpowers\plans\2026-08-23-minimal-species-mixing-port.md C:\tmp\liquid_warp_species_master\docs\superpowers\plans\2026-08-23-minimal-species-mixing-port.md
git -C C:\tmp\liquid_warp_species_master add docs/superpowers/plans/2026-08-23-minimal-species-mixing-port.md
git -C C:\tmp\liquid_warp_species_master commit -m "docs: plan minimal species mixing port"
```

### Task 2: Port the minimal GPU mixing core

**Files:**
- Create: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/sim/species.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/sim/solver.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/sim/kernels_fluid.py`
- Modify only if required by the existing optional-channel API: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/sim/particles.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/sim/sourcing.py`
- Test: `C:/tmp/liquid_warp_species_master/tests/test_species_transport.py`

**Interfaces:**
- Consumes: existing solver P2G/G2P schedule, `ParticleSet.optional_channels`, source descriptors, and solver `dt`.
- Produces: `SpeciesTransport.mix(dt: float) -> None`, particle channels `represented_volume: float32` and `species_concentration: vec4f`, and `SourceShapeDesc.species_concentration: tuple[float, float, float, float]`.

- [ ] **Step 1: Add the focused failing transport tests**

```python
def test_same_cell_red_and_blue_relax_toward_equal_mix(cuda_solver):
    solver = cuda_solver(species_enabled=True, species_mixing_rate=20.0)
    solver.seed_species_particle((0.5, 0.5, 0.5), (1.0, 0.0, 0.0, 0.0))
    solver.seed_species_particle((0.5, 0.5, 0.5), (0.0, 1.0, 0.0, 0.0))
    solver.species.mix(0.1)
    concentrations = solver.download_species_concentration_for_test()
    assert concentrations[0][0] < 1.0 and concentrations[0][1] > 0.0
    assert concentrations[1][1] < 1.0 and concentrations[1][0] > 0.0
    assert abs(sum(concentrations[0]) - 1.0) < 1.0e-5


def test_disabled_species_uses_no_species_channels(cuda_solver):
    solver = cuda_solver(species_enabled=False)
    assert "species_concentration" not in solver.particles.optional_channels
    assert "represented_volume" not in solver.particles.optional_channels
```

- [ ] **Step 2: Run only the new test and confirm the missing behavior**

```powershell
python -m pytest -q tests/test_species_transport.py
```

Expected: failure because the transport/settings/channels are not present.

- [ ] **Step 3: Add the compact transport API and kernels**

Port the proven equations, retaining four GPU lanes but enabling two authored species in this pass:

```python
SPECIES_CONCENTRATION_CHANNEL = "species_concentration"
REPRESENTED_VOLUME_CHANNEL = "represented_volume"


class SpeciesTransport:
    def __init__(self, solver) -> None:
        self.solver = solver

    def mix(self, dt: float) -> None:
        if not self.solver.settings.species_enabled:
            return
        blend = 1.0 - math.exp(
            -self.solver.settings.species_mixing_rate * max(float(dt), 0.0)
        )
        if blend <= 0.0:
            return
        self._clear_grid_accumulators()
        self._accumulate_volume_and_species_mass()
        self._blend_particles_from_grid(blend)
```

The P2G kernel accumulates `represented_volume` and `represented_volume * concentration`. The G2P kernel computes `cell_mass / max(cell_volume, epsilon)`, linearly blends by `blend`, clamps lanes nonnegative, and normalizes the vec4 sum to one. All allocations and launches stay on the current Warp device.

- [ ] **Step 4: Wire it into the existing solver substep without changing force order**

Add settings with these exact defaults:

```python
species_enabled: bool = False
species_mixing_rate: float = 8.0
species_palette: tuple[tuple[float, float, float, float], ...] = (
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
)
```

Allocate the optional channels only when enabled. Initialize injected particles from `SourceShapeDesc.species_concentration`. Call `self.species.mix(dt)` once in the existing solver substep after grid transfer has valid particle neighborhoods and before the final particle state is published. Leave `self.surface_tension.apply(dt)` and all surrounding force calls in their existing order.

- [ ] **Step 5: Run the core test and the existing sparse-grid/surface-tension contracts**

```powershell
python -m pytest -q tests/test_species_transport.py tests/test_sparse_leaf_grid_contract.py fluidSim_warp_blender51/tests/test_surface_tension_math.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the core independently**

```powershell
git add fluidSim_warp_blender51/sim/species.py fluidSim_warp_blender51/sim/solver.py fluidSim_warp_blender51/sim/kernels_fluid.py fluidSim_warp_blender51/sim/particles.py fluidSim_warp_blender51/sim/sourcing.py tests/test_species_transport.py
git commit -m "feat(sim): port minimal GPU species mixing"
```

### Task 3: Adapt sources, current UI, and viewport color

**Files:**
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/properties.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/blender/solver_settings.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/blender/scene_inputs.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/blender/ui_scene.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/blender/ui_object.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/sim/frame_output.py`
- Modify: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/blender/output_publication.py`
- Modify only where the existing attribute shader requires it: `C:/tmp/liquid_warp_species_master/fluidSim_warp_blender51/blender/materials.py`
- Test: `C:/tmp/liquid_warp_species_master/tests/test_species_ui_contract.py`

**Interfaces:**
- Consumes: Task 2 settings, source composition, and particle concentration channel.
- Produces: scene properties `species_enabled`, `species_mixing_rate`, `species_color_0`, `species_color_1`; source property `species_concentration`; output color attribute `fluidsim_color`.

- [ ] **Step 1: Add the focused UI/property contract test**

```python
from pathlib import Path


def test_species_controls_use_current_scene_and_object_panels():
    scene_ui = Path("fluidSim_warp_blender51/blender/ui_scene.py").read_text()
    object_ui = Path("fluidSim_warp_blender51/blender/ui_object.py").read_text()
    legacy_ui = Path("fluidSim_warp_blender51/panels.py").read_text()
    assert "species_enabled" in scene_ui
    assert "species_mixing_rate" in scene_ui
    assert "species_concentration" in object_ui
    assert "species_concentration" not in legacy_ui
```

- [ ] **Step 2: Run the test and confirm the controls are absent**

```powershell
python -m pytest -q tests/test_species_ui_contract.py
```

Expected: failure on the missing scene/object controls.

- [ ] **Step 3: Port only the active adapters and controls**

Add an artistic section containing Enable Mixing, Mixing Strength, Red color, and Blue color. Put composition on source-capable fluid objects. Keep advanced storage/kernel controls out of the UI. Add the controls to `ui_scene.py` and `ui_object.py`; do not modify `panels.py`, and do not duplicate controls in `ui_physics.py` because it already mirrors `ui_object.OBJECT_PANEL_CLASSES` into Physics Properties.

Map source composition explicitly:

```python
def source_species_concentration(source_props) -> tuple[float, float, float, float]:
    red = min(max(float(source_props.species_concentration), 0.0), 1.0)
    return (red, 1.0 - red, 0.0, 0.0)
```

When mixing is disabled, settings projection omits the species channels and publication supplies the existing uniform liquid color without reading a missing array.

- [ ] **Step 4: Publish the mixed color attribute with one palette dot product**

```python
def concentration_to_rgba(concentration, palette):
    return tuple(
        sum(concentration[i] * palette[i][component] for i in range(4))
        for component in range(4)
    )
```

Use the existing output/publication buffer lifecycle. Do not add a second mesh or shader system. The material reads `fluidsim_color`; no species texture, CPU diffusion pass, or per-frame material rebuild is introduced.

- [ ] **Step 5: Run the UI contract and current panel regression tests**

```powershell
python -m pytest -q tests/test_species_ui_contract.py tests/test_operator_panel_behavior.py tests/test_phase4_pending_interaction.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the adapter/presentation slice**

```powershell
git add fluidSim_warp_blender51/properties.py fluidSim_warp_blender51/blender/solver_settings.py fluidSim_warp_blender51/blender/scene_inputs.py fluidSim_warp_blender51/blender/ui_scene.py fluidSim_warp_blender51/blender/ui_object.py fluidSim_warp_blender51/sim/frame_output.py fluidSim_warp_blender51/blender/output_publication.py fluidSim_warp_blender51/blender/materials.py tests/test_species_ui_contract.py
git commit -m "feat(blender): expose and display liquid color mixing"
```

### Task 4: Port the canonical demo and run the bounded acceptance proof

**Files:**
- Create or rebuild: `C:/tmp/liquid_warp_species_master/demos/chromatic_vortex.blend`
- Create: `C:/tmp/liquid_warp_species_master/tests/blender_species_toggle_acceptance.py`
- Create: `C:/tmp/liquid_warp_species_master/tests/blender_chromatic_color_acceptance.py`
- Modify only if required to register the tracked demo: `C:/tmp/liquid_warp_species_master/tools/check_demo_blends.py`

**Interfaces:**
- Consumes: Task 3 source controls and `fluidsim_color` publication.
- Produces: a live two-source vortex-force demo and executable proof of disable/resume/re-enable plus red/blue-to-purple mixing.

- [ ] **Step 1: Bring over the two focused acceptance scripts and adapt imports only**

The toggle script must exercise this sequence in one solver lifecycle:

```python
enable_species()
advance_frames(2)
disable_species()
advance_frames(2)
enable_species()
advance_frames(2)
assert_no_solver_error()
assert_particle_count() > 0
```

The color script must verify distinct initial source colors and a mixed region:

```python
assert red_source_mean[0] > red_source_mean[2]
assert blue_source_mean[2] > blue_source_mean[0]
assert mixed_mean[0] > 0.15
assert mixed_mean[2] > 0.15
```

- [ ] **Step 2: Rebuild the demo against 1.0.255 rather than copying stale registered data**

Use two Initial Fill source objects, source concentrations `1.0` and `0.0`, and the existing analytic vortex force. Save at `frame_start` with live playback ready and no cache dependency.

- [ ] **Step 3: Run only the focused Blender acceptance scripts**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python tests/blender_species_toggle_acceptance.py
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python tests/blender_chromatic_color_acceptance.py
```

Expected: both scripts print their pass markers; disabling species produces no `(0, 4)` channel-shape error.

- [ ] **Step 4: Run the smallest sibling-lane regression check**

```powershell
python -m pytest -q tests/test_species_transport.py tests/test_species_ui_contract.py fluidSim_warp_blender51/tests/test_surface_tension_math.py tests/test_force_subsystem_contracts.py
```

Expected: all selected tests pass, proving the common solver/UI seams and the force/surface-tension sibling lanes remain intact.

- [ ] **Step 5: Open the rebuilt demo only after background proof passes**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' C:\tmp\liquid_warp_species_master\demos\chromatic_vortex.blend
```

In that same process, verify the enabled add-on version/path and advance live playback long enough to see red and blue enter the vortex and create a purple region. Pause at a representative frame for Owner review.

- [ ] **Step 6: Commit the demo and acceptance slice**

```powershell
git add demos/chromatic_vortex.blend tests/blender_species_toggle_acceptance.py tests/blender_chromatic_color_acceptance.py tools/check_demo_blends.py
git commit -m "test: add canonical chromatic vortex acceptance"
git status --short
```

Expected: the final status is clean. The full production gate is deliberately deferred until this focused port is accepted or prepared for merge.

## Completion evidence

The port is complete when all of the following are directly demonstrated from `C:/tmp/liquid_warp_species_master`:

1. Branch ancestry is `b1336ac...` and add-on version is 1.0.255.
2. Existing surface-tension files and current Scene/Object/Physics panel architecture remain present.
3. Species can be disabled, simulated, and re-enabled without a channel-shape error.
4. Red and blue source regions are initially distinct and the vortex produces a region containing both red and blue components.
5. The prototype checkout remains untouched and no cache/report/profiling subsystem was transplanted.
