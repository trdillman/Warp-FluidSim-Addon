# Phase 1 Source UI Claim Matrix — 2026-07-26

## Scope and evidence rule

This note reconciles the historical status table in
`docs/SOURCE_EMISSION_PRODUCT_PLAN.md` with the current branch. A claim is
source/focused-test proven only when the production seam and behavioral test
agree. Blender claims additionally require the recorded runtime gates below.
No enum or label alone is treated as proof.

## Structured before/after UI map

| Workflow | Before Phase 1 | Phase 1 surface |
|---|---|---|
| Emitter tagging | `Tag Source` | `Tag Liquid Emitter` (same operator/RNA IDs) |
| Lifecycle | `emission_mode` stored but not drawn | `Initial Fill` / `Continuous Inflow`; Continuous remains default |
| Analytic region | `Geometry: Analytic` | `Region: Analytic Volume` |
| Mesh emitter | `Geometry: Mesh` | `Region: Mesh Surface Shell`; no solid-volume claim |
| Occupancy | `Emit Rate` | `Target Particles / Voxel` |
| Attributes | temperature/viscosity/lifespan inline; density hidden | collapsed `Particle Attributes`, including `Density Attribute` |
| Sink | tag action in Colliders/Sinks, editor in Sources | tag and shared region editor together in Colliders/Sinks |
| Force region | instruction said tag Source; editor accepted SOURCE/FORCE | `Tag Force Region`; editor accepts FORCE only |
| Height fill | ineffective Back Mode visible | only effective base, velocity, and particle toggle visible |
| Point injection | `Inject Mesh Vertices Once` | `Inject Active Object Points Once`, with mesh/PointCloud/first-particle-system wording |
| Compatibility | FIELD hidden | FIELD remains hidden and non-liquid |

The before map was derived before the Phase 1 panel edit from the current
`FLUIDSIM_PT_sources`, `FLUIDSIM_PT_colliders`, and `FLUIDSIM_PT_forces` draw
paths and the pre-edit user guide. The after map was exercised in real Blender
by `tests/blender_phase1_source_ui_acceptance.py`; its 26 structured checks are
stored in `artifacts/profile_safe_source_review.json`.

## Current-branch claim reconciliation

| Claim | Historical plan status | Current source evidence | Focused behavioral evidence | Blender evidence | Phase 1 disposition |
|---|---|---|---|---|---|
| Initial Fill | Missing | `SceneInputs` derives a host-only key from `obj.as_pointer()`; `SourcePipeline` tracks fired keys, disarms after first fill, and resets lifecycle on solver reset | lifecycle and scene-adapter tests prove once-only firing, reset-only rearm, actual-object rename stability, and substep independence | Profile-safe review: actual rename preserved the key; first/later region counts 1037/0 and fired key retained | **Real; exposed** |
| Continuous Inflow | Partial | analytic and mesh particle paths remain armed and replenish occupancy each substep | lifecycle tests prove replenish-after-drain and continuous arming | Profile-safe review: first/later region counts 1037/1246 and no one-shot fired key | **Real; remains default** |
| Analytic Volume emitter | Real/partial boundary caveat | analytic SDF region is production source path | mesh/source pipeline and lifecycle gates cover descriptor execution | Profile-safe structured UI/translation check passed | Expose truthful existing region label |
| Mesh Surface Shell emitter | Conformant | evaluated triangles and bounded shell source path | `tests/test_mesh_source_pipeline.py` | `MESH-SOURCE-ACCEPTANCE-PASS`; profile-safe descriptor/mesh-error check passed | Expose explicitly as shell |
| Mesh Solid Volume emitter | Missing | no emitter interior adapter | no acceptance test | Not applicable | Hidden; Phase 2 only |
| Target Particles / Voxel | Mislabelled | `emit_rate` is occupancy top-up target | lifecycle tests vary occupancy target | Profile-safe panel capture passed | Relabel only; RNA identifier/default unchanged |
| FIELD creates liquid | False premise/corrected | field representation does not create liquid; FORCE applies later in forces phase | `tests/test_source_field_force_contract.py` and force subsystem tests | `FORCE-REGION-ACCEPTANCE-PASS`; hidden-control check passed | Hidden compatibility property |
| Analytic Force Region | Current branch real | FORCE role translates to non-emitting force descriptor | force subsystem tests prove force phase and non-emission | Normal-user FORCE tag/translation passed; CUDA delta-x 0.5 | Surface independent FORCE tagging/editor |
| Density | Implemented attribute | particle payload transfers density | source pipeline coverage | Profile-safe panel capture passed | `Density Attribute` in collapsed attributes |
| Height Back Mode | Partial/ineffective | FLAT/INFINITE currently share behavior; PROFILE has no Blender profile input | no distinct-behavior proof | Not applicable | Hidden |
| One-shot point injection | Conformant compatibility | mesh, PointCloud, first particle system extraction | operator poll/UI behavior tests | Mesh and PointCloud passed in profile-safe review; particle-system case skipped because fixture absent | Truthful compatibility wording |

## Compatibility assertions

- Existing RNA identifiers and enum identifiers are unchanged.
- `emission_mode` default remains `CONTINUOUS_INFLOW`.
- `source_mode=FIELD`, `blend_mode`, `feather`, `height_back_mode`, and
  `auto_inject_seed` remain stored compatibility properties but are not promoted
  as normal liquid-emission controls.
- No saved-data migration is introduced.
- Mesh source shell thickness semantics are unchanged.
- No Phase 2 capability is claimed by this UI.

## Completed parent evidence

1. Focused Python gate: 64 passed.
2. `FORCE-REGION-ACCEPTANCE-PASS` proved normal-user FORCE tagging and no liquid sourcing.
3. `MESH-SOURCE-ACCEPTANCE-PASS` and `GUI-ACCEPTANCE-PASS (35/35)` retained their exact markers.
4. `artifacts/profile_safe_source_review.json` finished with `ok: true`, 45/45
   copied files, exact temporary module identity, 26/26 visible-review checks,
   owned PID 53144 exit, temp-copy removal, unchanged enabled add-ons/profile
   hashes, and no errors.
5. Particle-system injection remains unobserved only because the selected
   `dam_break.blend` fixture has no particle system; its extraction contract is
   covered by focused tests and the runtime report records the skip explicitly.
