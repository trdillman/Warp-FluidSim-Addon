# Dynamic Liquid Species Design

**Status:** Proposed for Owner approval

**Date:** 2026-08-22

## Outcome

FluidSim supports an object-authored, runtime-dynamic number of miscible colored liquid species. A normal user sets **Liquid Color** and **Dye Strength** on any liquid-producing object (analytic or mesh emitter, Point/VDB source, domain fill, initial seed, or injection). The add-on builds the scene registry automatically, transports and mixes concentrations on the GPU, and derives physically plausible Beer-Lambert display color. A red and blue turbine scene produces purple only where authoritative transported concentrations mix.

The feature must not add a separate species substep control. Mixing executes once per existing solver substep and uses that substep's `dt`.

## Truth Contract

1. The canonical state is species concentration, never an animated RGB proxy.
2. Source composition is authored on fluid-producing objects, not in a demo builder.
3. Active species count is runtime-dynamic and limited by memory admission, not a hard-coded count of two or four.
4. GPU work scales with active four-species chunks, never configured capacity or an arbitrary maximum.
5. Red plus blue produces purple through transported concentration and optical absorption.
6. No-species scenes allocate no species resources and launch no species kernels.
7. Turning mixing physics off preserves registry, concentrations, cache identity, and displayed color; it pauses exchange only and can resume without solver replacement.
8. Adding/removing/reidentifying species changes storage stride and therefore requires a controlled solver reinitialization.
9. No separate species substep, maximum-substep, or stability parameter exists in RNA, solver settings, cache settings, or UI.
10. Review scenes and reports are evidence only; they do not substitute for the general object workflow.

## Numerical Model

### Transport

Concentrations are particle-authoritative and move with FLIP particles. Each particle owns a simplex-safe vector `c[active_species_count]`, where every component is finite and nonnegative and `sum(c) <= 1`; the remainder is clear carrier liquid.

### Mixing

Mixing uses one conservative grid-mediated exchange per existing global solver substep. It deposits represented volume and species mass to cell-centred sparse-grid fields, exchanges mass across permeable positive faces, computes one global bound-preserving scale, and gathers the conservative delta to particles.

The artist-facing **Mixing Rate** is mapped to a timestep-invariant bounded exchange fraction:

```python
mix_fraction = 1.0 - math.exp(-max(0.0, mixing_rate) * dt)
```

The operator remains finite, conservative, monotonic, and simplex-safe for every accepted `dt`. It does not create an inner species step loop or increase global solver substeps. This is intentionally an artistic liquid-mixing model calibrated for stable visual convergence; it must not be marketed as an unrestricted high-accuracy molecular-diffusion PDE solver.

### Optics

Each species stores scene-linear absorption coefficients `absorption_rgb`. Display transmittance is derived from concentration:

```python
tau_rgb = sum(c_i * absorption_rgb_i for i in active_species)
rgb = exp(-tau_rgb * optical_path_length)
```

The surface and particle viewport paths use this derived result. No direct source-color blend can satisfy acceptance.

## Data Model

Create `fluidSim_warp_blender51/sim/species_schema.py` as the pure, Blender-independent owner:

```python
@dataclass(frozen=True)
class SpeciesDefinition:
    id: str
    display_name: str
    absorption_rgb: tuple[float, float, float]

@dataclass(frozen=True)
class MixtureComponent:
    species_id: str
    amount: float

@dataclass(frozen=True)
class MixturePayload:
    components: tuple[MixtureComponent, ...]

@dataclass(frozen=True)
class SpeciesRegistry:
    schema_version: int
    definitions: tuple[SpeciesDefinition, ...]
    fingerprint: str

    @property
    def active_count(self) -> int: ...

    @property
    def chunk_count(self) -> int:
        return (self.active_count + 3) // 4

    def pack(self, mixture: MixturePayload) -> np.ndarray: ...

def absorption_from_display_color(
    color: tuple[float, float, float], reference_depth: float
) -> tuple[float, float, float]: ...
```

Definitions are canonically ordered by persistent opaque ID. Equal colors do not imply equal identity. A simple object's implicit species ID derives from its persistent FluidSim source identity; advanced components receive a persistent UUID and can be shared explicitly.

The color picker represents target transmittance at the configured reference depth. Convert it with `absorption = -log(clamp(color, 1e-4, 1.0)) / reference_depth`. Dye Strength is the emitted component amount, so changing strength affects future births without changing registry identity; changing Liquid Color changes the definition and requires reinitialization.

GPU concentration and grid arrays use flattened `wp.vec4` chunks:

```text
particle_chunks[particle_index * chunk_count + chunk_index]
grid_mass[voxel_index * chunk_count + chunk_index]
grid_delta[voxel_index * chunk_count + chunk_index]
```

The last chunk masks lanes beyond `active_count`. Kernel launch count is constant for a given phase; launch dimensions scale by `active_particles * chunk_count` or `active_voxels * chunk_count`, not by one launch per species.

## Authoring

### Basic fluid-object controls

- Liquid Color
- Dye Strength
- Initial Mixture: Single Liquid / Custom Mixture

These controls appear in the existing Fluid object/source context for emitters, mesh sources, Point/VDB sources, domain fill, seed workflows, and injection sources.

### Advanced object controls

- Mixture Components collection
- Component name
- Component color
- Component amount
- Link/reuse existing component identity

### Domain controls

- Species Physics: enabled/paused hot toggle
- Global Mixing multiplier
- Color Intensity / optical path multiplier

Advanced domain controls:

- Mixing Response, an exponent applied to the bounded exchange fraction (`effective_fraction = mix_fraction ** response`, finite and greater than zero)
- Reference Depth, the positive scene-unit distance used only to convert picker colors into registry absorption coefficients; changing it changes the registry fingerprint and requires reinitialization
- Carrier Tint, converted to carrier absorption and weighted by the clear remainder `1 - sum(c)`

No numerical stability or substep control is exposed.

## Source and Admission Flow

`simulation_admission.py` builds one immutable registry from every enabled authored liquid payload in the coherent Admission Snapshot. Registry membership fixes GPU stride for that Solver Session.

Every source family must deliver the same `MixturePayload` through its deepest production seam:

- analytic emitter;
- evaluated mesh emitter;
- Point/VDB direct append;
- Point/VDB rasterized source;
- domain height fill;
- initial/auto seed;
- operator injection.

`scene_bridge.sync_scene()` always replaces the complete source plan set, including an empty tuple. Removing the final source cannot leave a stale source resident.

Changes to amounts, enabled state, or mixing rate for existing registered IDs affect future births and are synchronized live. Adding, removing, or relinking IDs changes the registry fingerprint and requires reinitialization. Existing particle composition never changes merely because a source property changes.

## Runtime Toggle

The structural condition is “the admitted scene contains an active registry.” The hot **Species Physics** toggle controls exchange only:

- On: particles transport concentrations and the bounded mixing exchange runs.
- Off: concentrations remain attached, sources retain composition, display color remains truthful, and exchange kernels do not run.
- On again: exchange resumes from the preserved concentrations without solver replacement.

Removing every dyed source and reclaiming species allocation is a separate reinitialization operation.

## GPU-First Viewport Publication

The live surface path must not read all particle positions/concentrations or invoke CPU nearest-neighbour transfer.

The surface extractor derives a device `float4[surface_vertex_count]` color array from the current post-step concentration field and palette. Live publication transfers only the existing surface mesh plus one surface-sized color array. Blender publishes one POINT-domain `fluidsim_color` attribute, and the Clear Dye material reads only that attribute.

Raw N-component concentrations are advanced/debug/export data and are never implicitly enabled by Clear Dye viewport mode.

## Cache and Resume

Species cache schema version 2 stores:

- canonical registry JSON and fingerprint;
- active species count and chunk count;
- mixing/optical settings;
- particle concentration as host shape `(particle_count, active_species_count)`;
- cumulative source/sink accounting sized to active species count.

Resume rules:

- exact registry fingerprint: restore directly;
- same stable IDs in a different order: remap columns by ID;
- added IDs: initialize their columns to zero after explicit reinitialization;
- removed IDs: reject continuation unless the user explicitly starts a replacement run;
- legacy schema 1: translate only when its two IDs and palette exactly match the admitted registry; otherwise fail closed with a regenerate-cache message;
- mixing toggle changes do not invalidate cache identity.

## Performance Contract

All comparisons use adjacent disabled brackets, identical active particles/voxels, identical global solver substeps, five child processes, 15 warmup frames, and 60 measured frames on CUDA.

| Case | Median overhead | Paired p95 delta |
|---|---:|---:|
| No species authored | exact zero species work/resources | 0 ms species work |
| Registry configured but no dyed source active | <= 1% | <= 0.25 ms |
| Concentration transport, mixing paused | <= 2% | <= 0.50 ms |
| Active mixing, N=1..4 | <= 5% | <= 1.00 ms |
| Active mixing, each additional four-species chunk | <= 4 percentage points | recorded and bounded |

Viewport surface Clear Dye must add no more than 5% median `view_ms` and synchronized end-to-end `total_ms`, with no more than 1 ms paired p95 in either metric, versus the same surface scene without species. It must record zero raw particle-species readbacks, zero CPU nearest calls, one POINT color attribute, and no CORNER/display-color duplication.

The thresholds are release gates, not targets that may be weakened to obtain a pass.

## Acceptance

1. A user can author red and blue emitters without a demo builder.
2. A turbine/blender scene produces a measurable purple region after live solver playback.
3. A four-emitter scene proves four independently authored colors and mixed regions.
4. An N=5 scene proves a second chunk and active-count scaling.
5. Editing an authored object's component membership N=4→N=5 stages reinitialization, completes Replacement Cutover on success, and preserves the old solver on injected failure.
6. Physics off/on preserves solver identity and UID-bound concentrations, suppresses exchange only, permits authored births/advection, and resumes mixing.
7. Save/reopen, rewind, live playback, cache bake/playback, and compatible resume preserve registry identity and concentration.
8. Every source family proves its public RNA/operator path, deepest emitted mixture, and final-source removal.
9. No-species demos remain byte-for-byte behaviorally unaffected in source/runtime contracts.
10. The solver, viewport, cache, and Blender acceptance gates all pass from fresh artifacts.

## Explicit Non-Goals

- Immiscible density interaction, oil/water interfaces, phase transitions, and thermal coupling.
- Exact pigment Kubelka-Munk scattering.
- Direct CUDA-to-Blender mesh interop.
- Reuse of archived native-interoperability or other forbidden experimental branches.
- Bitwise deterministic CUDA accumulation.
