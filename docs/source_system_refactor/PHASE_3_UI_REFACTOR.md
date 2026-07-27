# Phase 3 — Full Source UI Information Architecture

## Goal statement

Replace the mixed Sources/Colliders/Sinks/Forces presentation with a role-first,
capability-driven UI that exposes every accepted Phase 2 source workflow exactly
once, keeps advanced field operations separate from liquid emission, preserves
saved-file compatibility, and provides actionable validation/errors inside the
workflow that owns them.

This is the full UI refactor. It begins only after Phase 2 hands off an accepted
capability matrix and UI schema.

## Entry criteria

- Phase 2 is Accepted. `Accepted with documented deferral` permits Phase 3 only
  when the user explicitly narrows the program goal, the roadmap and capability
  matrix record that scope change, and every deferred capability remains hidden.
  Ordinary implementation deferral cannot bypass mandatory Phase 2V Volume
  acceptance.
- The Phase 2 handoff names every capability allowed in the UI.
- All visible capabilities have behavioral and Blender-runtime evidence.
- Missing capabilities have an explicit hiding/Experimental decision.
- Existing source/sink/collider/force/height/injection saved scenes are available
  as compatibility fixtures.
- A user-review session can be launched profile-safely.

## Exit criteria

The Blender UI has separate, coherent workflows for:

1. Liquid Emitters.
2. Liquid Sinks.
3. Colliders.
4. Forces.
5. Domain Height Fill.
6. Point/Particle Injection utilities.
7. Advanced Field Sources, only if Phase 2 accepted them.

A user can create, configure, validate, animate, reset, and inspect each workflow
without Python scripting or understanding legacy solver enums.

## Information architecture

### Liquid Emitters

#### Activation

- Enabled.
- Emission Behavior:
  - Initial Fill — Once After Initialize/Reset.
  - Continuous Inflow — Refill While Enabled.

#### Emission Region

Show only Phase 2-approved options:

- Analytic Volume.
- Mesh Solid Volume.
- Mesh Surface Shell.
- Volume Grid.
- Points / Particles.

Conditional controls:

- Analytic: shape.
- Mesh Solid: mesh object, evaluation cadence, watertight status.
- Mesh Shell: mesh object, evaluation cadence, shell half-width, optional
  surface sampling controls if that implementation was accepted.
- Volume Grid: object, grid names/mappings, transform/resampling status.
- Points: input object/system, direct append vs rasterized region where supported.

#### Sampling

- Use Domain Particles / Voxel, default on for new sources if Phase 2 supports it.
- Conditional custom Target Particles / Voxel override.
- Capacity estimate/warning when available.

#### Spawn Motion

- Initial Velocity.
- Add Animated Object Velocity.
- Any accepted spawn-velocity pattern controls; never label these persistent
  force regions.

#### Particle Attributes (collapsed)

- Temperature.
- Viscosity.
- Density Attribute.
- Lifespan with Min/Max.
- Accepted imported point/volume attribute mappings.

### Liquid Sinks

- Tag as Liquid Sink.
- Enabled.
- Removal Region:
  - Analytic Volume.
  - Mesh Solid Interior.
  - Mesh Surface Shell.
  - Accepted Volume Grid mask.
- Show watertight validation only for Solid Interior.
- Show shell width only for Surface Shell.
- No emitter lifecycle, occupancy, spawn motion, or particle payload controls.

### Colliders

- Tag as Collider.
- Enabled.
- Collider mesh/input status.
- Friction.
- Restitution.
- Transfer Animated Velocity.
- Keep collision behavior distinct from sink removal and source masks.

### Forces

Separate sections:

1. Global Grid Forces.
2. Blender Force Fields, with explanation that configuration lives in Blender
   Physics properties.
3. Analytic Force Regions:
   - Tag as Force Region;
   - Enabled;
   - analytic shape;
   - force type;
   - acceleration strength;
   - conditional direction/spin axis/falloff controls.

Only FORCE-role objects appear as analytic force regions.

### Domain Height Fill

- Enable Domain Height Fill.
- Fill Level / approved profile input.
- Lifecycle where Phase 2 implemented it.
- Initial Velocity.
- Sampling control or inherited domain occupancy.
- Only distinct, validated Back Mode choices are shown.

### Point/Particle Injection

Collapsed compatibility/import utility:

- Inject Active Object Points Once.
- Supported-input explanation.
- Reinject on Initialize/Reset if accepted.
- Clear “Initialize Solver first” disabled-state reason.
- Never describe vertex/point injection as volume fill.

### Advanced Field Sources

Show only if Phase 2 accepted field sourcing:

- Region/input.
- Target field: pressure, scalar, vector, staggered velocity, voxel type/mask.
- Source payload: constant, particle attribute, volume grid.
- Blend operation and amount only when implemented for that combination.
- Experimental badge/help where evidence is incomplete but explicitly approved.

Do not expose legacy `source_mode` as a liquid-emitter choice.

## UI implementation principles

1. Role first; geometry, lifecycle, and representation are not peer source types.
2. Each property appears in one owning workflow.
3. Conditional visibility removes impossible combinations rather than disabling
   dozens of irrelevant controls.
4. Validation appears beside the invalid selection and tells the user how to fix it.
5. Advanced controls are collapsed, not mixed into the primary workflow.
6. Labels use user concepts; internal enum names remain compatibility details.
7. Panels call shared role-specific draw helpers, not duplicate condition trees.
8. Operators enforce their own `poll()` conditions; panel disabled states are not
   the only protection.
9. Old RNA identifiers/defaults are translated, not renamed destructively.
10. New defaults apply only where compatibility rules explicitly permit them.

## Work packages

### P3.0 — UI schema and migration freeze

**Dependencies:** Phase 2 handoff.

- Convert the Phase 2 capability matrix into a control schema.
- Decide compatibility behavior for every legacy property and role.
- Define panel IDs/order, operator labels, icons, help text, errors, and defaults.
- Create low-cost wireframes or structured draw expectations.
- Obtain user approval before source files are edited.

### P3.1 — Panel/module restructuring

**Dependencies:** P3.0.

- Replace mixed panel ownership with role-specific panels.
- Extract private draw helpers by workflow.
- Keep registration/unregistration atomic and rollback-safe.
- Preserve panel ordering and Blender 5.1 compatibility.

**Likely files:**

- `fluidSim_warp_blender51/panels.py`
- `fluidSim_warp_blender51/__init__.py`

### P3.2 — Role creation and operators

**Dependencies:** P3.0; may proceed with P3.1 under one writer only.

- Implement role-specific tag/clear actions.
- Add `poll()`, disabled-state reasons, undo behavior where appropriate, and
  actionable reports.
- Ensure operators cannot create invalid role/input combinations.

**Likely files:**

- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/blender/sources.py`

### P3.3 — Property presentation and compatibility adapter

**Dependencies:** P3.0.

- Add presentation-only properties only when the backend interface requires them.
- Keep old saved properties readable and translated centrally.
- Add stable migration/version markers if Phase 2 requires them.
- Do not retain two independently editable representations of the same setting.

**Likely files:**

- `fluidSim_warp_blender51/properties.py`
- `fluidSim_warp_blender51/blender/scene_inputs.py`

### P3.4 — Validation, diagnostics, and status

**Dependencies:** P3.1–P3.3.

- Put watertight, missing-grid, unsupported-combination, capacity, and source
  identity errors in their owning workflow.
- Expose source compile/cache status where useful without flooding the UI.
- Ensure runtime failures stop or quarantine the affected source rather than
  silently changing behavior.

### P3.5 — Documentation and demos

**Dependencies:** accepted interactive workflow.

- Rewrite source/sink/force sections of
  `fluidSim_warp_blender51/docs/user_guide.md`.
- Update demo builder and saved demos to use real source workflows.
- Dam break must use a real Initial Fill source/region, not direct injection
  while claiming source-driven playback.
- Add one focused demo each for Solid Volume, Surface Shell, point source,
  volume-grid source, sink, force region, and height fill where accepted.
- Keep compatibility injection demos labeled as such.

### P3.6 — Full acceptance and cleanup

**Dependencies:** P3.5.

- Run focused, full Python, Blender background, GUI, saved-demo playback, and
  production gates.
- Remove temporary UI compatibility branches that are no longer needed, but not
  saved-data translation required for old files.
- Remove debug instrumentation and archive review artifacts.

## Agent workflow policy

In addition to the common policy:

- P3.0 is a decision gate. No writer starts panel restructuring without the
  approved control schema.
- One writer owns `panels.py`, `properties.py`, `operators.py`, and registration
  order. Splitting these files between agents is prohibited.
- Backend behavior is frozen. A backend defect found during UI work is reported
  and fixed in a separately reviewed patch before UI acceptance continues.
- Reviewers must use a fresh temporary add-on copy; no persistent installation.
- Every hidden/conditional control requires a testable visibility rule.
- Documentation and demos are truth surfaces and receive the same review rigor
  as code.

## Required validation

### Focused Python UI/compatibility gate

```bash
python -m pytest -q \
  tests/test_operator_panel_behavior.py \
  tests/test_source_emission_lifecycle.py \
  tests/test_mesh_source_pipeline.py \
  tests/test_source_field_force_contract.py \
  tests/test_force_subsystem_contracts.py \
  tests/test_production_gate_truthfulness.py
```

Add tests for panel ordering, conditional visibility, role-specific operators,
legacy scene translation, and every accepted Phase 2 adapter.

### Blender background gates

```bash
blender --background --factory-startup --python tests/blender_background_acceptance.py
blender --background --factory-startup --python tests/blender_mesh_source_acceptance.py
blender --background --factory-startup --python tests/blender_force_region_acceptance.py
blender --background --factory-startup --python tests/blender_handler_persistence_gate.py
```

Add/extend acceptance probes for source creation without Python scripting and
saved-scene compatibility.

### GUI and saved-demo gate

```bash
blender --factory-startup --python tests/blender_gui_acceptance.py
FLUIDSIM_PLAYBACK_BLEND=demos/dam_break.blend \
FLUIDSIM_PLAYBACK_REPORT=artifacts/gui_playback_probe.json \
blender --factory-startup --python tests/blender_gui_playback_probe.py
python tools/validate_gui_playback_report.py artifacts/gui_playback_probe.json
```

Also perform the profile-safe visible user review using the Phase 1 harness:

```bash
python tools/launch_profile_safe_addon_review.py \
  --blend demos/dam_break.blend \
  --report artifacts/profile_safe_final_source_review.json
```

The final report must prove complete temp-copy identity, one owned package/process,
active blend, successful cleanup, and unchanged profile hashes.

### Full regression and production gate

```bash
python -m pytest -q
python tools/run_production_gate.py --include-gui
```

Report headless and GUI-inclusive markers separately.

### Demo gate

```bash
blender --background --python tools/build_demo_scenes.py
blender --background --python tools/check_demo_blends.py
python tools/analyze_demos.py
```

Expected final marker remains `DEMO-ANALYSIS-PASS (18/18)` unless an approved
change updates the canonical count and its truthful validator.

## Reviewer gates

### R1 — information-architecture review

A fresh reviewer verifies:

- roles are separated;
- control order follows activation, region, sampling, motion, attributes;
- impossible combinations are absent;
- errors appear at the decision that caused them;
- no property is duplicated across panels without a single owner.

### R2 — capability/truth review

For every visible option, the reviewer must point to:

1. runtime implementation;
2. behavioral test;
3. Blender acceptance evidence;
4. documentation/demo wording.

An option lacking this chain is hidden or blocks acceptance.

### R3 — saved-file and registration review

Verify:

- old `.blend` files retain their roles/settings;
- new defaults do not rewrite old data;
- register/unregister and handler persistence remain correct;
- extension/runtime temp-copy loading uses one package identity;
- no profile state is saved.

### R4 — user-path review

A human-visible Blender session must exercise:

- create and run every role from scratch;
- switch valid region/lifecycle modes;
- receive actionable invalid-input errors;
- reset/reinitialize and confirm lifecycle semantics;
- play multiple frames with stable UI/output;
- save/reopen approved compatibility fixtures;
- inspect help text without needing code knowledge.

### R5 — performance/usability review

- UI changes must not synchronously rebuild the solver on every drag tick.
- Heavy compile/rebuild actions must be debounced or explicit.
- Source status must not cause per-draw GPU/mesh probing.
- Dam-break playback remains within the approved Phase 2 baseline threshold.

No blocker/high finding may remain. Medium findings require explicit disposition.

## Phase handoff requirements

The final program handoff must additionally include:

- Final screenshots and role/control map.
- Saved-scene migration/compatibility table.
- Final capability claim matrix with Real/Experimental/Hidden/Missing verdicts.
- Demo-to-capability evidence map.
- Full gate commands, markers, logs, reports, screenshots, and profile hashes.
- User-review notes and unresolved usability feedback.
- Performance and memory comparison to Phase 2 baseline.
- Rollback instructions for UI and compatibility adapters.
- Explicit list of future work not represented as shipped capability.

## Rollback

Role-specific panels may be rolled back while preserving Phase 2 backend and
saved-property translation. The program must never require rolling back backend
correctness to restore an older UI. If the new UI cannot pass user acceptance,
retain Phase 1's truthful UI as the fallback—not the original misleading panels.
