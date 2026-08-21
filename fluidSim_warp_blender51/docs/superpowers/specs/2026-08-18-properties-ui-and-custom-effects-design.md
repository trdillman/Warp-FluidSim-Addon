# FluidSim Properties UI and Custom Effects Integration Design

> **Presentation update (1.0.255):** `docs/contextual_physics_ui.md` supersedes
> this document's compact-viewport-only decision. Scene field ownership remains
> authoritative; the compact status panel remains primary, the full hierarchy
> is mirrored beneath one collapsed parent, and selected FluidSim objects
> receive contextual Physics Properties panels.

## Purpose

FluidSim 1.0.251 has production-capable solver, cache, rendering, whitewater, surface-tension, and Custom Effects systems, but its public controls are split across a dense 3D View sidebar. The result exposes implementation groupings instead of the workflow artists follow and duplicates concepts between the legacy panel tree and Custom Effects v2.

This design relocates the authoritative controls to Blender's native Properties editor, keeps only a compact 3D View status/launcher, and integrates Custom Effects v2 into the same visual hierarchy and interaction model. The attached HTML is a visual-density reference, not literal web UI. The attached Final FluidSim UI Layout Plan is the field-ownership source of truth where it matches current code. Current Phase 4, Phase 5, and Custom Effects v2 behavior takes precedence where the older plan has no placement.

## User stories

### Artist

As an artist, I can move from setup to preview, physics, outputs, and production bake in a predictable top-to-bottom flow, while seeing only controls relevant to the current scene state and active object role.

### Technical artist

As a technical artist, I can inspect capacity, diagnostics, persistent attributes, whitewater classification, surface-tension limits, and Custom Effects without searching through unrelated panels or duplicated controls.

### Warp effect author

As a Warp effect author, I can configure an artist-facing effect stack in Scene Properties, open a subordinate Warp Code workspace, test and apply the exact draft, and understand trust, compilation, sandbox, cache, and runtime state without switching to a parallel legacy snippet workflow.

## Product surfaces

### Scene Properties

The authoritative scene workflow is a set of root panels in this order:

1. Session & Status
2. Setup
3. Domain & Particles
4. Initial Liquid — Height Fill
5. Simulation Physics
6. Custom Effects
7. Surface Reconstruction
8. Whitewater
9. Viewport & Outputs
10. Cache & Production Bake
11. Diagnostics

Root panels use `bl_space_type = "PROPERTIES"`, `bl_region_type = "WINDOW"`, and `bl_context = "scene"`. Open-by-default panels are Session & Status, Setup, Domain & Particles, and Viewport & Outputs. All others are closed by default. `bl_order` fixes the public ordering.

Subpanels follow the field-complete plan and use `bl_parent_id` rather than empty wrapper boxes. A parent that owns only one coherent group draws the fields directly.

### Object Properties

The authoritative input workflow is in Object Properties with `bl_context = "object"`:

1. FluidSim Role — Selected Objects
2. Region Geometry
3. Liquid Source
4. Source Lifespan
5. Field Blending
6. Collider
7. Force Region
8. Field Source
9. Volume Grid

Poll methods use the active object's canonical role and geometry. Role mutation actions explicitly operate on the selected set; detailed values and diagnostics describe the active object only.

### Add-on Preferences

Preferences contain Runtime and, only when `FLUIDSIM_ALLOW_EXTERNAL_WARP=1`, Developer Runtime Override. The unused verbose preference remains internal and is not drawn.

### 3D View sidebar

The sidebar is not a second settings surface. It contains one compact panel with:

- derived activity state;
- runtime readiness;
- Start/Stop Live Preview;
- Initialize/Release Solver;
- Step One Frame;
- a shortcut that switches the current Properties editor to Scene context when possible.

It does not render simulation settings, object roles, cache settings, diagnostics, or effect parameters.

## Derived state and lifecycle ownership

- `blender.playback.activity_snapshot(scene)` is the only public activity truth. Raw `live` and `cache_playback_enabled` are hidden transport state.
- `blender.pending_settings.pending_presentation(scene)` is the only pending-settings truth. The UI does not recreate restart/reset policy.
- `blender.scene_bridge.session().owns_scene(scene)` determines current-scene solver ownership.
- Stop, Cancel, Release, and Clear Error actions remain available when a running/failed state exists even if readiness later degrades.
- Poll and execute paths continue to use the same validators; panels do not invent weaker checks.

## Scene panel ownership

### Session & Status

Draws derived activity, runtime readiness, authored status/error, and context-appropriate actions. Children:

- Pending Settings, visible only when a pending action exists.
- Runtime Statistics, visible when the current scene owns the solver or cache playback is active for this scene.
- Live Timeline Recovery.

### Setup

Draws Guided Setup and setup mode. Product Presets is an open child.

### Domain & Particles

Sole owner of `domain_object`, `voxel_size`, `particles_per_voxel`, `max_particles`, `auto_grow_particles`, `topology_expand`, and `open_boundaries`. Capacity & Safety owns `leaf_capacity`, `max_whitewater`, the growth-failure policy, and safe-option actions.

### Initial Liquid — Height Fill

Owns only the eight height-fill fields. It never draws `particles_per_voxel` or an inherited sampling label.

### Simulation Physics

Children:

- Time & Transfer
- Gravity
- Pressure
- Viscosity
- Surface Tension
- Particle Relaxation
- Velocity Limits
- Global Forces
- Dissipation
- Attraction
- Spatiotemporal Processing

Surface Tension is the Phase 5 placement. It draws the enable switch and authored parameters plus a compact read-only capillary stability/runtime summary. It does not duplicate general substep controls.

### Custom Effects

Custom Effects is a root panel immediately after Simulation Physics. It always presents the artist workflow; the old `artist_mode` toggle becomes internal compatibility state.

The panel has four visual groups:

1. **Effect Stack** — searchable Add Effect, import, stack list, reorder, duplicate, mute, solo, remove.
2. **Selected Effect** — description, typed parameters, controller actions, trust state, and concise cost/prerequisite summary.
3. **Workflow** — Test Draft, Compile & Apply, Revert Draft, Preview One Frame.
4. **Status** — Draft/Tested/Applied/Dirty/Invalid/Untrusted/Runtime Error, applied divergence, production-bake blocking reason, compile target, and digest evidence.

Package maintenance actions are grouped beneath a closed Package & Library child panel instead of competing with the primary workflow.

Warp Code is a closed child of Custom Effects and is visible whenever an effect with at least one kernel is active. It is not gated by `artist_mode`. It contains trust/expert controls, Text datablock/editor actions, domain/stage/cadence, declared field access, source-mapped diagnostics, generated-source inspection, field API reference, and compile/runtime evidence.

Legacy Custom Warp Snippet panels remain unregistered. Legacy operator IDs continue to act only as migration adapters.

### Surface Reconstruction

Root owns `surface_enabled`. Children are Resolution & Cadence, Particle Splat, and Filters, visible only when enabled.

### Whitewater

Root owns `whitewater_enabled`. Children:

- Emission
- Classification
- Lifetime
- Dynamics
- Admission Status

Classification is the Phase 4 placement for foam band, spray threshold, and hysteresis. Dynamics contains type-specific foam, bubble, and spray controls. Admission Status contains aggregate and typed counts.

### Viewport & Outputs

Children:

- Display
- Materials & Collection
- Persistent Attributes, with Surface, Particle, and Whitewater children.

Display owns `liquid_view_mode`, display fractions/scales, attribute selectors, whitewater visibility, and `whitewater_output_mode`. Hidden `show_surface` and `show_particles` aliases are never drawn.

Materials & Collection includes combined and separate whitewater materials plus render-node install/update actions.

### Cache & Production Bake

Root owns `cache_path`. Children are Playback, Shared Playback / Export Range, Storage, Production Bake, and Manage Cache. Playback/export range and bake range remain separate. Destructive actions continue to use their existing invoke/confirmation paths and revalidation.

### Diagnostics

Children are Profiling, Benchmark, and Report & Export. Validation Suite remains on the root. Benchmark baseline-save remains an operator dialog parameter.

## Object panel conditions

- Role root is always visible when Object Properties has a scene and active object context.
- Region Geometry is visible for SOURCE, SINK, FORCE, or FIELD.
- Liquid Source is visible only for SOURCE.
- Source Lifespan is visible only for SOURCE with lifespan enabled.
- Collider is visible only for COLLIDER.
- Force Region is visible only for FORCE.
- Field Blending and Field Source are visible only for FIELD.
- Volume Grid is visible only when canonical region geometry is `VOLUME_GRID` for SOURCE, SINK, or FIELD.
- Constant scalar/vector payload controls are hidden when volume payload is selected.
- Field amount, force strength, force direction, mesh/point/volume details follow actual consumer conditions rather than being disabled dead controls.

## Visual and copy rules

The HTML's useful characteristics are translated to Blender-native layout:

- compact two-column labels and values through `column`, `row`, and `grid_flow`;
- short, status-led root panels;
- open daily-workflow panels and closed expert panels;
- primary actions first, destructive or maintenance actions later;
- status icons and concise labels instead of long explanatory paragraphs;
- conditional hiding rather than disabled irrelevant controls;
- no browser-specific visual constructs or custom colors.

Panel IDs remain stable and descriptive. Operator IDs remain stable unless the old operator is demonstrably obsolete.

## Architecture

### `blender/ui_common.py`

Owns common panel bases, scene/object poll helpers, derived activity access, role normalization, status rows, property existence guards, and compact layout helpers. It contains no mutation policy.

### `blender/ui_scene.py`

Owns Scene Properties panels and their registration list. It consumes the existing runtime, pending-settings, playback, capacity, cache, and solver services.

### `blender/ui_object.py`

Owns Object Properties panels and their registration list. It consumes canonical role/geometry values and existing operators.

### `panels.py`

Becomes the registration aggregator plus the compact View3D launcher and header interaction. The prior dense panel classes are removed from the registered surface.

### `blender/effect_ui.py` and `blender/effect_workspace.py`

Retain operator and list implementations but relocate panel metadata and restructure drawing around the common Scene Properties hierarchy. Package/library maintenance becomes a child panel.

### `preferences.py`

Draws the compact Runtime and conditional Developer sections. It does not expose `verbose`.

## Compatibility

- RNA property names and operator IDs remain unchanged.
- Scene/Object data created by 1.0.251 remains readable.
- `artist_mode` remains serialized for compatibility but does not control visibility.
- Legacy snippet properties and operators remain migration-only and hidden.
- Cache, checkpoint, effect signatures, trust, and solver behavior are not changed by this UI migration.

## Error handling

- Draw methods guard optional properties and missing runtime data so a malformed scene produces a visible diagnostic instead of breaking all Properties panels.
- Invalid embedded effect definitions remain visible in Custom Effects with the existing fail-closed actions.
- Object role migration ambiguity remains explicit; the UI does not guess.
- Missing cache directories hide mutation actions but do not hide the cache path or integrity status.

## Testing

### Pure Python/source tests

- panel metadata, order, parent relationships, default-open state, and absence of duplicate legacy registration;
- one public owner for high-risk duplicate fields;
- no raw transport aliases in public draw methods;
- Custom Effects and Warp Code placement/visibility;
- object role poll matrix;
- Preferences hiding `verbose`;
- compact View3D panel contains actions but no settings duplication.

### Supplied Blender

A background validation script registers the extension and verifies:

- all expected Scene/Object panels register in `PROPERTIES`;
- Custom Effects and Warp Code parent correctly;
- legacy dense View3D panels are absent;
- the compact View3D launcher remains;
- representative Scene and Object draw paths execute with a recording layout;
- conditional role/effect panels poll correctly;
- enable, disable, and re-enable leave no duplicate classes or header callbacks;
- save/reopen preserves active roles and Custom Effects state.

### Supplied Warp

The existing Custom Effects real-JIT and sandbox gate is rerun unchanged to prove that UI relocation did not alter execution behavior.

## Release

The release version becomes `1.0.252`. The Linux x64 extension is built and validated using supplied Blender 5.2.0 LTS and supplied external Warp 1.16.0+cu12. No Blender or Warp runtime is bundled. CUDA and Windows claims remain unresolved unless compatible supplied packages and hardware are executed.
