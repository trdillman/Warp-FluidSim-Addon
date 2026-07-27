# Phase 1 — UI Truthfulness and Workflow Repair

## Goal statement

Make every currently visible source-related control describe and invoke behavior
that already exists, while exposing essential implemented controls that are
currently unreachable. This phase must improve current workflows without
claiming Mesh Solid Volume, Volume Grid sourcing, arbitrary field sourcing, or
other Phase 2 capabilities.

## Entry criteria

- `docs/source_system_refactor/README.md` has been read and accepted.
- Current source behavior is baselined against:
  - `docs/SOURCE_EMISSION_PRODUCT_PLAN.md`
  - `docs/source_shape_spec.md`
  - `tests/test_source_emission_lifecycle.py`
  - `tests/test_source_field_force_contract.py`
- Existing dirty worktree state is recorded.
- The historical status claims in `SOURCE_EMISSION_PRODUCT_PLAN.md` are
  reconciled against current source, focused lifecycle tests, and a Blender
  runtime probe. In particular, Initial Fill may be exposed only if P1.1 records
  current behavioral evidence; otherwise its UI work moves to Phase 2.
- The profile-safe launcher contract in the program README is accepted for P1.0.

## Exit criteria

A user can identify and configure the currently implemented workflows without
scripting or false labels:

- Liquid Emitter with Initial Fill or Continuous Inflow.
- Analytic Volume or Mesh Surface Shell region.
- Target Particles / Voxel with truthful units.
- Liquid Sink with analytic, Mesh Solid Interior, or Mesh Surface Shell removal.
- Collider.
- Analytic Force Region with a working tag action.
- Domain Height Fill with only effective choices shown.
- One-shot active-object point injection with truthful supported input wording.

No new backend capability is implied.

## Truth matrix for this phase

| UI claim | Runtime status | Phase 1 action |
|---|---|---|
| Initial Fill | Candidate real in current branch; historical plan status is stale/conflicting | Expose only after P1.1 behavioral and Blender proof |
| Continuous Inflow | Candidate real/default | Expose only after P1.1 proof |
| Analytic Volume emitter | Real | Label it clearly |
| Mesh Surface Shell emitter | Real | Rename plain Mesh |
| Mesh Solid Volume emitter | Missing | Do not expose |
| Target Particles / Voxel | Real | Rename Emit Rate in UI only |
| FIELD liquid source | False/compatibility-only | Keep hidden |
| Analytic Force Region | Real for role FORCE | Add correct tag/workflow |
| Source tagged as force region | Misleading/ineffective | Remove instruction |
| Height Back Mode choices | Partially ineffective | Hide or clearly quarantine |
| Inject mesh vertices once | Misnamed | Rename to active-object points |
| Density Attribute | Real but hidden | Expose under advanced attributes |

## Scope

### Required UI changes

1. In `panels.py`, expose `emission_mode` for SOURCE-role objects only after
   P1.1 proves both lifecycle modes through the current production seam.
2. Present source geometry as:
   - **Analytic Volume**
   - **Mesh Surface Shell**
3. Draw `emit_rate` as **Target Particles / Voxel** without renaming the RNA
   property or breaking saved files.
4. Rename Tag Source to **Tag Liquid Emitter** while retaining operator/RNA
   compatibility.
5. Extend `FLUIDSIM_OT_tag_object` to support role `FORCE`; add **Tag Force
   Region** in the Forces panel.
6. Remove “tag as Source” force-region instructions and accept only FORCE-role
   objects in the force-region editor.
7. Put sink configuration where the user tags a sink. Either move the existing
   draw helper or temporarily share one helper; do not duplicate divergent
   logic.
8. Rename the one-shot action to **Inject Active Object Points Once** and explain
   mesh vertices, PointCloud points, and the first particle system.
9. Expose Density as **Density Attribute** in a collapsed Particle Attributes
   subsection.
10. Hide Height Back Mode until distinct behavior/profile input is implemented,
    or label it Experimental with a visible limitation. Hiding is preferred.
11. Update tooltips and user documentation to match runtime behavior.
12. Add operator `poll()`/disabled-state messages for newly surfaced actions.

### Compatibility constraints

- Keep existing enum identifiers and saved property names.
- Preserve Continuous Inflow as the default for old and new tagged sources in
  this phase.
- Do not change existing source execution semantics.
- Do not reinterpret FIELD as liquid emission.
- Do not change mesh source shell thickness semantics.
- Existing `.blend` files must open without migration scripts or data loss.

### Non-goals

- No Mesh Solid Volume emission.
- No Blender Volume/OpenVDB adapter.
- No point-source rasterizer.
- No source UUID migration.
- No full panel information-architecture rewrite.
- No new field sourcing UI.
- No change to source kernels except a defect required to make an already-real
  exposed control function correctly; such a defect requires escalation.

## Work packages

### P1.0 — Reusable profile-safe visible-review harness

**Dependencies:** none.

Create `tools/launch_profile_safe_addon_review.py` and its focused tests before
manual GUI acceptance is allowed. It must implement the exact command/report
contract in the program README, including:

- normal user profile inheritance;
- pre/post enabled-add-on and preference/startup existence/hash snapshots;
- temporary complete source copy with inventory verification and no stale bytecode;
- one package identity per process and non-persistent enablement;
- owned PID, module path/version, active blend, and error/crash reporting;
- cleanup of only the owned process/temp copy;
- explicit failure on profile drift.

Add `tests/test_profile_safe_addon_review.py`. The JSON report must include
`ok`, process identity, source/copy inventory, module identity, active blend,
pre/post profile snapshots, cleanup result, and any crash-report path.

### P1.1 — Baseline claim and UI test matrix

**Dependencies:** P1.0.

- Add/extend behavioral panel/operator tests for every row in the truth matrix.
- Test role assignment and conditional visibility, not only source strings.
- Add a saved-default compatibility test for Continuous Inflow.
- Resolve every historical/current status discrepancy in a dated claim-matrix
  note. Initial Fill and Continuous Inflow require source-level, focused-test,
  and Blender-runtime evidence before their rows become Real.
- Record current Blender screenshots or a structured UI probe before changes.

**Likely files:**

- `tests/test_operator_panel_behavior.py`
- `tests/test_source_emission_lifecycle.py`
- `tests/test_force_subsystem_contracts.py`
- `tests/blender_gui_acceptance.py`

### P1.2 — Shared role-specific draw helpers

**Dependencies:** P1.1 failing tests.

- Extract private panel draw helpers only where they prevent source/sink/force
  logic from diverging.
- Keep the helper interface role-oriented and small.
- Do not create a general form framework.

**Likely files:**

- `fluidSim_warp_blender51/panels.py`

### P1.3 — Tagging and operator workflow repair

**Dependencies:** P1.1.

- Add FORCE to the tag operator enum and validation.
- Add truthful labels and `poll()` behavior.
- Ensure collider mesh-only restrictions do not leak into Source/Sink/Force.
- Preserve role-change reset/update semantics.

**Likely files:**

- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/blender/sources.py`
- `tests/test_operator_panel_behavior.py`

### P1.4 — Current-capability UI correction

**Dependencies:** P1.2, P1.3.

- Expose lifecycle, occupancy, region, attributes, sink, force, height, and
  injection controls as specified.
- Keep advanced/compatibility state hidden.
- Display mesh errors next to the controls that caused them.

**Likely files:**

- `fluidSim_warp_blender51/panels.py`
- `fluidSim_warp_blender51/properties.py` (labels/descriptions only unless approved)

### P1.5 — Documentation and acceptance probes

**Dependencies:** P1.4.

- Update the user guide and screenshots/text probes.
- Add an acceptance scenario that tags and configures one emitter, sink, and
  force region without Python scripting.
- Verify opening an old source scene retains the same stored behavior.

**Likely files:**

- `fluidSim_warp_blender51/docs/user_guide.md`
- `tests/blender_gui_acceptance.py`
- `tests/blender_force_region_acceptance.py`

## Agent workflow policy

In addition to the common policy:

- The writer may edit only UI/property/operator/docs/tests files unless a
  runtime defect is escalated and approved.
- Reviewers must inspect the runtime branch behind each visible control.
- A UI screenshot is supporting evidence, never proof that the operation works.
- Do not rename RNA identifiers to improve display text.
- Do not add greyed-out Phase 2 controls as teasers.
- One writer owns `panels.py`, `properties.py`, and `operators.py`; do not split
  these files across concurrent agents.

## Required validation

### Focused Python gate

```bash
python -m pytest -q \
  tests/test_profile_safe_addon_review.py \
  tests/test_operator_panel_behavior.py \
  tests/test_source_emission_lifecycle.py \
  tests/test_source_field_force_contract.py \
  tests/test_force_subsystem_contracts.py \
  tests/test_mesh_source_pipeline.py
```

### Blender acceptance gate

```bash
blender --background --factory-startup --python tests/blender_force_region_acceptance.py
blender --background --factory-startup --python tests/blender_mesh_source_acceptance.py
blender --factory-startup --python tests/blender_gui_acceptance.py
```

Expected markers must remain exact and positive. The phase handoff must record
which GUI command was automated versus manually reviewed.

### Visible review gate

Run the Phase 1 launcher and retain its report:

```bash
python tools/launch_profile_safe_addon_review.py \
  --blend demos/dam_break.blend \
  --report artifacts/profile_safe_source_review.json
```

The report must finish with `ok: true`, one owned Blender PID, the exact blend
and temp module identity, complete source inventory, successful cleanup, and
unchanged profile hashes. During that owned session:

- Tag and configure a Liquid Emitter from scratch.
- Select Initial Fill and Continuous Inflow and observe different lifecycle
  behavior over multiple frames.
- Configure an analytic Source and Mesh Surface Shell source.
- Tag and run a Force Region without changing RNA through Python.
- Configure a Solid and Surface mesh sink and observe validation placement.
- Exercise point injection on mesh, point cloud, and particle-system input where
  fixtures are available.
- Confirm profile hashes are unchanged after the owned process closes.

## Reviewer gates

### R1 — UI truth reviewer

Must answer, with file/line and runtime evidence:

- Does every label describe existing behavior?
- Can a normal user reach Initial Fill and Force Region workflows?
- Is Mesh Solid Volume absent?
- Is FIELD hidden and non-liquid?
- Are units and supported input types truthful?

### R2 — compatibility reviewer

Must verify:

- No RNA identifier/default was unintentionally changed.
- Existing `.blend` sources preserve behavior.
- Old Continuous Inflow scenes remain Continuous Inflow.
- No duplicate source/sink/force draw logic can drift.

### R3 — interaction reviewer

Must use live Blender and report control order, conditional visibility, error
placement, disabled states, and playback behavior. Static code review is not
sufficient.

No blocker/high finding may remain.

## Phase handoff requirements

The Phase 1 handoff must additionally include:

- Before/after UI map and screenshots or structured UI artifacts.
- Final list of visible controls by role.
- List of hidden compatibility properties and why each remains hidden.
- Saved-file compatibility evidence.
- Exact statement: “No Phase 2 capability is claimed by this UI.”
- Backend assumptions that Phase 2 may rely on.
- An implementation-ready Phase 2 prompt limited to the accepted backend scope.

## Rollback

Phase 1 must remain revertible by restoring panel/operator presentation without
migrating saved data. If rollback would require editing `.blend` data, the phase
has violated its compatibility constraint.
