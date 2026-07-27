# Phase 3 P3.0 — Approved Control Schema

Approved by the user on 2026-07-26 before Phase 3 source edits.

## Panel order

1. `FLUIDSIM_PT_liquid_emitters` — Liquid Emitters.
2. `FLUIDSIM_PT_liquid_emitter_attributes` — Particle Attributes (collapsed child).
3. `FLUIDSIM_PT_liquid_sinks` — Liquid Sinks.
4. `FLUIDSIM_PT_colliders` — Colliders.
5. `FLUIDSIM_PT_forces` — Forces.
6. `FLUIDSIM_PT_domain_height_fill` — Domain Height Fill.
7. `FLUIDSIM_PT_point_particle_injection` — Point / Particle Injection (collapsed).
8. `FLUIDSIM_PT_advanced_field_sources` — Advanced Field Sources (collapsed, non-liquid).

Each saved property has one visible owner. Shared helpers may draw role identity, lists, and cached diagnostics, but must not duplicate editable controls or perform compilation/GPU/mesh probing.

## Approved visible capabilities

- Liquid: Initial Fill and Continuous Inflow; Analytic Volume; Mesh Solid Volume; Mesh Surface Shell; Point/Particle direct append or rasterized region; Volume Grid.
- Sink: Analytic Volume; Mesh Solid Interior; Mesh Surface Shell; accepted Volume mask only after its Blender translator passes.
- Collider: existing mesh collider behavior, friction, restitution, animated velocity transfer.
- Force: global grid forces, explanatory Blender Force Fields section, accepted analytic FORCE role.
- Height: Flat; Profile only through a real grayscale Blender Image input. Infinite is absent.
- Advanced fields: accepted Pressure, Scalar, Vector, and Staggered Velocity operations with only valid region/payload/blend combinations.
- Compatibility utility: Inject Active Object Points Once, never described as volume fill.

## Compatibility/default policy

- Existing RNA identifiers and registered defaults are unchanged.
- Legacy `geometry_mode=MESH` remains Mesh Surface Shell unless a new explicit region selection says Solid; no inference or fallback.
- Legacy `source_mode=FIELD` is classified as an Advanced Field Source, never a liquid emitter.
- Existing saved values remain readable and centrally translated.
- New P3 liquid emitters are initialized to Initial Fill by the role-specific creation operator; the global `emission_mode` RNA default remains unchanged.
- Height Infinite saved values remain readable but unapplied with a compatibility warning; they are never silently mapped to Flat.
- New presentation properties use a non-destructive schema marker where needed and never create two independently editable values for one setting.

## Workflow/control ownership

### Liquid Emitters
Tag/Clear; Enabled; Emission Behavior; Emission Region; region-specific input; Sampling; Spawn Motion; collapsed Particle Attributes; cached diagnostics; emitter list.

- Solid alone shows watertight status.
- Shell alone shows shell half-width and voxel-floor warning.
- Points explicitly distinguish Direct Append from Rasterized Region.
- Volume requires a Blender Volume object and explicit named grids/mapping.
- Capacity, memory, dynamic mesh budget, occupancy clamp, shell floor, and runtime fill status appear beside the owning decision.

### Liquid Sinks
Tag/Clear; Enabled; Removal Region; region input; Solid/Shell validation; diagnostics; list. No lifecycle, occupancy, spawn motion, or particle attributes.

### Colliders
Tag/Clear; Enabled; mesh status; Friction; Restitution; Transfer Animated Velocity. No watertight scan.

### Forces
Global Grid Forces; Blender Force Fields explanation; Analytic Force Regions. FORCE remains on the accepted legacy post-P2G path and never becomes liquid or a compiled FORCE plan.

### Height Fill
Enable; Fill Level; Flat/Profile mode; Image profile input when Profile; Initial Velocity; sampling/lifecycle/status; diagnostics. Infinite absent.

### Point Injection
Inject Active Object Points Once; supported-input explanation; exact Initialize-first disabled reason; optional compatibility reinjection setting. Not a volume workflow.

### Advanced Fields
Tag/Clear; Enabled; Target Field; Region/Input; Payload source; Blend/Amount only for accepted combinations; diagnostics. No liquid controls and no visible legacy `source_mode`.

## Operator policy

Role-specific tag/clear actions have truthful `poll()` messages, `UNDO` for RNA/object mutations, and reject invalid object/input combinations. Compatibility IDs may remain callable but cannot create invalid state. Direct injection retains its tested solver/input polls.

## Diagnostics/status policy

Structured diagnostics are copied into a bounded source-keyed presentation snapshot after sync/compile. Panel draw only reads that snapshot. Errors quarantine the affected source; no shell fallback, missing-grid substitution, dynamic-mesh silent downgrade, or repeated GPU/mesh probing is allowed.

## Required evidence before exposure

Every visible option must have runtime implementation, focused behavior test, Blender acceptance, and matching docs/demo wording. Unsupported matrix rows remain absent rather than greyed-out promises.
