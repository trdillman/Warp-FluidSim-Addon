# Phase 3 and Final Program Handoff — Role-First Source Workflows

## Verdict

**Accepted.** Phase 3 and the final source-system program satisfy the approved schema, capability truth, compatibility, visible-user, performance, demo, and full production gates. The fresh consolidated R1–R5 review found no blocker/high; both medium findings were repaired and regression-tested before final acceptance.

## Goal and delivered scope

Phase 3 replaces the mixed Sources/Colliders/Sinks/Forces presentation with separate role-owned workflows and adds real Blender translators for Phase-2-accepted backend capabilities. It preserves legacy RNA/defaults and keeps backend/source truth ahead of labels or demos.

Delivered workflows:

1. Liquid Emitters.
2. Liquid Sinks.
3. Colliders.
4. Forces.
5. Domain Height Fill.
6. Point / Particle Injection compatibility utility.
7. Advanced Field Sources, explicitly non-liquid.

The approved P3.0 schema is `reviews/archive/phases/PHASE_3_CONTROL_SCHEMA_2026-07-26.md`.

## Final role/control map

| Panel | Primary order and ownership |
|---|---|
| Liquid Emitters | Tag/Clear; Enabled; Emission Behavior; Region; region input; Sampling; Spawn Motion; cached diagnostics; scene emitter list |
| Particle Attributes | Temperature; Viscosity; Density Attribute; Lifespan Min/Max; collapsed child of Liquid Emitters |
| Liquid Sinks | Tag/Clear; Enabled; Removal Region; input; Solid/Shell/Volume validation; diagnostics |
| Colliders | Tag/Clear; Enabled; mesh status; Friction; Restitution; Transfer Animated Velocity |
| Forces | Global Grid Forces; Blender Force Fields explanation; Analytic Force Regions |
| Domain Height Fill | Enable; Fill Level; Flat/Profile Image; lifecycle; Initial Velocity; sampling; diagnostics |
| Point / Particle Injection | Inject Active Object Points Once; supported-input help; optional compatibility reinjection |
| Advanced Field Sources | Tag/Clear; Enabled; Target Field; Region/Input; Payload Source; Blend/Amount; diagnostics; non-liquid help |

Panel IDs and ordering are mechanically pinned by `tests/test_phase3_source_ui_schema.py`. No mixed `FLUIDSIM_PT_sources` panel is registered. Panel draw reads cached RNA status only and does not compile, query GPU state, load VDB, or analyze meshes.

## Final capability claim matrix

| Capability | Verdict | User reachability/evidence |
|---|---|---|
| Analytic liquid Initial Fill / Continuous Inflow | Real | Liquid Emitter panel/operator; focused lifecycle tests; Phase 2/3 Blender and GUI acceptance; dam-break demo |
| Mesh Solid Volume emitter | Real | Explicit region; watertight error/no fallback; Phase 2 quantitative region gate; Phase 3 translator gate; focused demo |
| Mesh Surface Shell emitter | Real | Explicit half-width/voxel-floor policy; Phase 2 shell evidence; Phase 3 translator gate; open-shell demo |
| Direct point source | Real | Points region → Direct Append Once; latch/re-arm regression; Phase 2 attribute gate; Phase 3 translator gate |
| Rasterized point source | Real | Points region → Rasterized Region; Phase 2 compiler/runtime gate; Phase 3 translator gate and focused demo |
| Volume liquid source | Real | Blender Volume + fog/SDF grid names; Phase 2V OpenVDB gate; Phase 3 translator gate and focused demo |
| Liquid sink analytic/solid/shell/Volume | Real | Separate sink panel; no emitter controls; Phase 2 precedence/mesh gates; Phase 3 Volume sink translation and demo |
| Collider | Real | Separate panel and mesh-only operator poll; existing background/GUI/runtime gates |
| Global/Blender/analytic FORCE | Real | Separate Forces sections; accepted legacy post-P2G path; force Blender gate |
| Height Flat | Real | Height panel and runtime descriptor |
| Height Profile Image | Real | Real grayscale Blender Image → owned 2-D profile; Phase 3 Blender save/reopen/translation gate; demo |
| Height Infinite | Missing by original semantics | Hidden and rejected with compatibility error; never mapped to Flat |
| Pressure field | Real | Advanced Field role; non-liquid; Phase 2 projected-control evidence; Phase 3 operator/translator/UI gate |
| Scalar/vector/staggered fields | Real | Advanced Field role; constant or accepted Volume payload; Phase 2 field gates; Phase 3 all-operation and Volume-payload translation |
| Compiled FORCE plans | Unsupported by contract | Hidden; accepted analytic FORCE remains on legacy stage |
| Particle-attribute-to-field UI payload | Hidden future work | Not in the final accepted Phase 2 UI capability rows; no placeholder control |
| Non-linear VDB transform | Missing | Rejected actionably; linear authored grid and object transforms are Real |

## Saved-file migration and compatibility

| Saved state | Final behavior |
|---|---|
| Existing RNA identifiers/defaults | Unchanged and readable |
| Legacy SOURCE + Analytic | Remains Analytic; registered lifecycle/default retained |
| Legacy SOURCE + MESH | Remains Mesh Surface Shell; never inferred as Solid |
| Legacy SOURCE + `source_mode=FIELD` | Classified into Advanced Field/Pressure compatibility; never liquid |
| Existing SINK mesh mode | Existing Solid/Surface setting remains authoritative |
| Existing COLLIDER/FORCE | Roles and settings unchanged |
| Legacy Height Flat/Profile | Flat retained; Profile applies only with real image input |
| Legacy Height Infinite | Stored value remains readable but unapplied with actionable warning |
| New P3 emitter | Explicit schema marker; creation operator initializes Initial Fill without mutating global RNA default |
| Rename/duplicate/reset/reopen | Phase 2 runtime identity semantics retained |

`artifacts/phase3_fixtures/phase3_roles.blend` is saved and reopened in the Phase 3 Blender acceptance lane. It proves roles/settings survive and legacy schema/version values are not rewritten.

## Translator/runtime architecture

- `blender/scene_inputs.py` remains the sole Blender-to-frozen-plan translator.
- Legacy analytic/mesh/sink/force descriptors continue through `sim/source_compat.py`.
- Point, Volume sink/liquid, and Advanced Field workflows produce immutable `SourcePlan` values and are appended to the same production plan set.
- Blender values are copied into primitives/owned arrays; no RNA/mathutils/evaluated mesh/OpenVDB object enters persistent solver state.
- Direct point plans are latched once per runtime and re-arm only on reset.
- Structured compiler/runtime diagnostics are copied into bounded RNA presentation snapshots after sync/step; panel draw never triggers work.

## Diagnostics and disabled states

Owning panels surface:

- empty/invalid/non-watertight mesh;
- missing/unreadable Volume/grid and unsupported transform;
- source leaf/memory capacity;
- dynamic mesh work budget and CUDA requirement;
- target occupancy clamp and shell-width floor;
- emitted count and capacity-limited status;
- invalid Height Profile/legacy Infinite state;
- Initialize-first and supported-input reasons for direct injection.

An invalid source is quarantined. There is no Solid→Shell fallback, VDB substitution, point-as-volume relabeling, or silent dynamic-resolution reduction.

## Demo-to-capability evidence map

| Demo | Truthful workflow |
|---|---|
| `demos/dam_break.blend` | Real 0.6×0.6×1.2 Analytic Initial Fill source, 128,000 first-frame particles; no direct-injection stand-in |
| `demos/mesh_solid_source.blend` | Watertight Mesh Solid Volume Initial Fill |
| `demos/mesh_shell_source.blend` | Open Mesh Surface Shell with explicit width |
| `demos/point_source.blend` | Rasterized Point Source, explicitly distinct from direct injection |
| `demos/volume_source.blend` | Blender Volume density grid liquid source |
| `demos/liquid_sink.blend` | Continuous analytic emitter plus Mesh Solid sink |
| `demos/force_fields.blend` | Native Blender force fields and accepted force behavior |
| `demos/height_fill.blend` | Profile Image Initial Fill |
| `demos/crown_splash.blend` | Canonical liquid/whitewater regression |
| `demos/adhesive_calender.blend` | Explicit compatibility point-seed/viscosity workflow; not mislabeled as volume fill |

The exact demo build/check/analyze workflow passes, including `DEMO-ANALYSIS-PASS (18/18)`.

## UI screenshots and visible-review artifacts

- `artifacts/phase3_gui_viewport.png` — real Blender viewport/sidebar acceptance screenshot.
- `artifacts/phase3_gui_render.png` — rendered visible-content check.
- `artifacts/phase3_gui_result.json` — 35/35 GUI checks.
- `artifacts/profile_safe_final_source_review.json` — one temporary package identity, exact `dam_break.blend`, 14 visible role/playback checks, three-frame advance, 128,000 particles, owned-process exit, temp-copy removal, and unchanged profile hashes.
- `artifacts/gui_playback_probe.json` — saved-demo multi-frame playback report.

No user-profile preference/startup/add-on state was saved or changed.

## Performance and memory versus Phase 2

Artifacts:

- Phase 2 baseline copy: `artifacts/phase2_source_pipeline_profile_baseline.json`.
- Final profile: `artifacts/source_pipeline_profile.json`.
- Comparison: `artifacts/phase3_vs_phase2_profile.json`.

The final 120-frame profile passes every pre-approved Phase 2 threshold and reports no hidden compile errors. Absolute dynamic medians/p95 remain within their gates: animated shell, point rasterization, Volume sampling, and the measured near-budget animated mesh all pass. One retained cache generation and zero reset residue remain true. Timing differences versus the saved baseline are microbenchmark-level variation; the final near-budget case measured 11.52 ms median / 12.55 ms p95, well below 16.7/33.3 ms. UI draw performs no compiler/GPU/mesh work.

## Validation evidence

### Focused and full Python

- Phase 3 focused UI/compatibility plus frozen Phase 2 seam: `101 passed`.
- Full suite: `343 passed`.
- Logs: `artifacts/phase3_focused_tests.log`, `artifacts/phase3_full_pytest.log`.

### Blender background

Passed markers include:

- `BACKGROUND-ACCEPTANCE-PASS`
- `MESH-SOURCE-ACCEPTANCE-PASS`
- `FORCE-REGION-ACCEPTANCE-PASS`
- `HANDLER-PERSISTENCE-GATE-PASS`
- all five Phase 2 source capability markers
- `PHASE3-SOURCE-UI-ACCEPTANCE-PASS`

Log: `artifacts/phase3_blender_background_gates.log`.

### GUI, profile-safe, playback

- `GUI-ACCEPTANCE-PASS (35/35)`
- `FLUIDSIM-PROFILE-SAFE-REVIEW-PASS`
- `GUI-PLAYBACK-PROBE-PASS`
- `SAVED-DEMO-GUI-PLAYBACK-PASS`

### Production

- Separate headless lane: `HEADLESS-GATE-PASS` in `artifacts/phase3_headless_production_gate.log`.
- GUI-inclusive lane: `FULL-PRODUCTION-GATE-PASS` in `artifacts/phase3_full_production_gate.log`.
- The GUI-inclusive lane includes Phase 3 source UI acceptance, canonical demo rebuild/analysis, real GUI panel/playback proof, and saved-demo report validation.

### Demo

- `DEMO-BUILD-PASS`
- `DEMO-BLEND-CHECK-PASS`
- `DEMO-ANALYSIS-PASS (18/18)`

## Consolidated reviewer disposition

The only Phase 3 reviewer ran alone, satisfying the user-requested Opus reviewer policy. Its R1–R5 verdict was **no blocker, no high**. It verified the panel order/ownership, full capability evidence chains, saved compatibility, profile-safe package/profile integrity, real user paths, and performance baseline.

Medium findings and disposition:

1. Advanced Field UI drew `volume_input_object` twice when both region and payload used Volume. Fixed by drawing the shared Volume input once; `test_phase3_field_volume_input_is_drawn_once_when_region_and_payload_use_volume` passes.
2. Aggregate scene emission telemetry was broadcast to every source/field object. Fixed by attributing aggregate values only when exactly one liquid source is active; multiple-source panels now remain unassigned rather than displaying false per-object truth. `test_phase3_aggregate_emission_is_not_broadcast_as_per_source_truth` passes.

The reviewer low about an exactly-full capacity result was also repaired: `capacity_limited` now uses the GPU result flag/direct request shortfall instead of treating `count == capacity` alone as truncation. Final Python and both production lanes were rerun after these changes.

## Changed files by Phase 3 ownership

### UI/presentation and translation
- `fluidSim_warp_blender51/panels.py`
- `fluidSim_warp_blender51/properties.py`
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/blender/sources.py`
- `fluidSim_warp_blender51/blender/scene_inputs.py`
- `fluidSim_warp_blender51/blender/scene_bridge.py`

### Focused backend correction
- `fluidSim_warp_blender51/sim/sourcing.py` — Direct Append now truthfully latches once and re-arms on reset; no other backend semantics changed.

### Docs/demos/gates
- `fluidSim_warp_blender51/docs/user_guide.md`
- `tools/build_demo_scenes.py`
- `tools/check_demo_blends.py`
- `tools/run_production_gate.py`
- regenerated canonical and new focused demo `.blend`/cache/report artifacts

### Tests/review
- `tests/test_phase3_source_ui_schema.py`
- `tests/test_operator_panel_behavior.py`
- `tests/test_source_emission_lifecycle.py`
- `tests/blender_phase3_source_ui_acceptance.py`
- `tests/blender_phase3_profile_safe_review.py`
- `tests/blender_background_acceptance.py`
- `tests/blender_gui_acceptance.py`
- `tests/blender_attribute_viz_acceptance.py`
- `reviews/archive/phases/PHASE_3_CONTROL_SCHEMA_2026-07-26.md`
- this handoff

The worktree was already heavily dirty. Ownership is path/hunk-level; aggregate diff is not attributed to Phase 3. No reset, clean, stash, stage, or unrelated rewrite occurred.

## Failed attempts and disposition

- The first source-driven dam-break used the historical display transform, which did not represent the documented 0.6 m column and failed Martin–Moses evidence. The easier option—loosening analysis—was rejected. The source object was corrected to the real 0.6×0.6×1.2 region; it emitted 128,000 particles and restored 18/18 evidence.
- The first full production run exposed a whitewater-attribute fixture that relied on the old Continuous default. The test was made explicit and truthful by configuring an energetic Continuous Inflow; whitewater colors then passed. No product default was reverted.
- A source-profile timing run missed p95 by 0.027 percentage points; the identical fixed 120-frame rerun passed. Thresholds were not changed.

## User-review notes

The user approved the P3.0 schema before source edits. Automated visible review exercised all role-creation operators, Image Profile input, initialization, and three playback frames in an owned temporary Blender process. No unresolved usability feedback is recorded. A future manual design-polish pass may adjust wording/icons without changing ownership or capability truth.

## Rollback

Role-specific panels/operators/presentation RNA may be rolled back to the truthful Phase 1 UI while preserving:

- Phase 2 compiler/pipeline correctness;
- legacy saved-property translation;
- runtime identity/lifecycle semantics;
- source diagnostics and data compatibility.

Rollback must never restore the original mixed/misleading UI or require a `.blend` rewrite.

## Explicit future work not represented as shipped

- Height Infinite semantics.
- Non-linear VDB transforms.
- Unbounded high-complexity Every Sim Frame mesh compilation.
- A general particle-attribute-to-arbitrary-field UI matrix beyond accepted translators.
- Cosmetic panel/icon refinements without new behavior.
