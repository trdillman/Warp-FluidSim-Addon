# Phase 1 Handoff — UI Truthfulness and Workflow Repair

## Goal and delivered scope

Phase 1 exposes only source, sink, force, height-fill, and point-injection behavior
that exists on the current runtime branch. It adds no Phase 2 source backend.

Delivered user workflows:

- **Liquid Emitter** tagging with **Initial Fill** and **Continuous Inflow**.
- **Analytic Volume** and **Mesh Surface Shell** emitter regions.
- **Target Particles / Voxel** occupancy control.
- Shared, adjacent Liquid Sink tagging and region configuration.
- Independent **Tag Force Region** workflow for FORCE-role objects.
- Effective Domain Height Fill controls with ineffective Back Mode hidden.
- **Inject Active Object Points Once** with mesh, PointCloud, and first-particle-system wording.
- Collapsed Particle Attributes containing **Density Attribute**.

## Explicit non-goals and deferred work

- No Mesh Solid Volume emission.
- No Volume/OpenVDB adapter, point rasterizer, arbitrary field-source UI, or
  source-identity migration.
- FIELD remains a hidden compatibility/internal representation and never claims
  liquid emission.
- Height Back Mode remains hidden until its modes have distinct backend behavior.
- The profile-safe `dam_break.blend` fixture lacked a particle system, so that
  injection variant was recorded as skipped in Blender; focused extraction tests
  remain its evidence.

**No Phase 2 capability is claimed by this UI.**

## Changed files by ownership

### Product UI and workflow

- `fluidSim_warp_blender51/panels.py`
- `fluidSim_warp_blender51/properties.py` — Phase 1 owns only source-related
  label/description hunks; identifiers/defaults were not changed by Phase 1.
  Pre-existing `domain_sdf_walls=True` and global-force RNA hunks in this dirty
  file belong to earlier reviewed work and are explicitly outside this handoff.
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/blender/sources.py`
- `fluidSim_warp_blender51/blender/scene_inputs.py` — host-only rename-stable runtime source key
- `fluidSim_warp_blender51/docs/user_guide.md`

### Profile-safe review harness

- `tools/launch_profile_safe_addon_review.py`
- `tests/test_profile_safe_addon_review.py`
- `tests/blender_phase1_source_ui_acceptance.py`

### Behavioral and Blender acceptance

- `tests/test_operator_panel_behavior.py`
- `tests/test_scene_bridge_collider_cache.py`
- `tests/blender_gui_acceptance.py`
- `tests/blender_force_region_acceptance.py`

### Evidence

- `reviews/PHASE_1_SOURCE_UI_CLAIM_MATRIX_2026-07-26.md`
- `artifacts/profile_safe_source_review.json`
- `artifacts/profile_safe_source_review.json.stdout.log`
- `artifacts/profile_safe_source_review.json.stderr.log`

## Interface and saved-data compatibility

- Existing RNA property names and enum identifiers are unchanged.
- `emission_mode` still defaults to `CONTINUOUS_INFLOW`.
- Existing `.blend` files require no migration.
- Initial Fill identity is a host-only `obj.as_pointer()`-derived string scoped
  to the Blender object/solver lifetime; renaming no longer re-arms it and no
  persistent UUID or saved-data field was added.
- Mesh shell thickness and source execution semantics are unchanged.
- `source_mode=FIELD`, `blend_mode`, `feather`, `height_back_mode`, and
  `auto_inject_seed` remain stored compatibility properties but are not promoted
  as ordinary liquid-emission controls.
- Phase 1 rollback is presentation/operator restoration only; no `.blend` data
  rewrite is required.

## Claim-matrix delta

Moved to runtime-proven and visible:

- Initial Fill: first/later region counts 1037/0; fired source remains disarmed.
- Continuous Inflow: first/later region counts 1037/1246; source remains armed.
- Analytic Force Region: normal-user FORCE tag translated to a non-emitting force
  descriptor; CUDA acceptance measured `delta_x=0.500000`.
- Mesh Surface Shell: evaluated-mesh shell acceptance passed.

Still missing/hidden: Mesh Solid Volume, Volume grids, arbitrary field sourcing,
and distinct Height Back Mode semantics.

## Validation evidence

### Focused Python gate

```text
python -m pytest -q tests/test_profile_safe_addon_review.py tests/test_operator_panel_behavior.py tests/test_source_emission_lifecycle.py tests/test_source_field_force_contract.py tests/test_force_subsystem_contracts.py tests/test_mesh_source_pipeline.py
64 passed in 4.26s (exit code 0)
```

The launcher-only gate subsequently passed `11 passed in 0.63s` after adding the
optional automated review-script seam.

### Blender 5.1 gates

```text
blender --background --factory-startup --python tests/blender_force_region_acceptance.py
FORCE-REGION-ACCEPTANCE-PASS (exit code 0)

blender --background --factory-startup --python tests/blender_mesh_source_acceptance.py
MESH-SOURCE-ACCEPTANCE-PASS (exit code 0)

blender --factory-startup --python tests/blender_gui_acceptance.py
GUI-ACCEPTANCE-PASS (35/35) (exit code 0)
```

Runtime identity: Blender 5.1.0, Warp 1.12.1, CUDA 12.9, NVIDIA GeForce RTX
4080 SUPER.

### Profile-safe visible review

Automated structured runtime review:

```text
python tools/launch_profile_safe_addon_review.py \
  --blend demos/dam_break.blend \
  --report artifacts/profile_safe_source_review.json \
  --review-script tests/blender_phase1_source_ui_acceptance.py
FLUIDSIM-PROFILE-SAFE-REVIEW-PASS
```

Report evidence:

- `ok: true`, stage `complete`, no errors.
- Owned PID 53144 exited with code 0.
- Exact `demos/dam_break.blend` active.
- One package identity; module resolved under the temporary root.
- Source/copy inventory complete.
- Visible review schema `fluidsim-phase1-source-ui-review-v1`; 26 checks, zero failures.
- Actual Blender-object rename retained the same runtime source key and did not
  re-arm the fired Initial Fill source.
- Temporary copy removed.
- Enabled add-ons and `userpref.blend`/`startup.blend` hashes unchanged before/after.

### Genuine UI-event supplement

`artifacts/phase1_ui_event_evidence.json` records Windows mouse/keyboard events,
not RNA assignment, against an owned profile-safe Blender window. Screenshots
prove a normal user could select an object, click **Tag Liquid Emitter**, open
the lifecycle dropdown, select **Initial Fill**, select **Mesh Surface Shell**,
and edit **Target Particles / Voxel**. FORCE and sink execution remain covered
by their normal-user operators and the independent CUDA/mesh gates; control
order, disabled states, adjacent errors, and playback are recorded by the
structured profile-safe and GUI acceptance artifacts.

The exploratory coordinate session later mis-targeted **Tag Collider** and ended
with `EXCEPTION_STACK_OVERFLOW`; it is explicitly not the accepted runtime run.
Its profile stayed unchanged and its owned process/temp copy were cleaned. The
accepted run is `artifacts/profile_safe_source_review.json`.

## Final visible controls by role

- **SOURCE:** Enabled, lifecycle, analytic/shell region, shape or mesh input,
  target particles/voxel, shell thickness/evaluation controls where applicable,
  initial velocity/object velocity, collapsed particle attributes, nearby mesh error.
- **SINK:** Enabled, analytic/mesh region, analytic shape or mesh input, Solid
  Interior/Surface Shell sink mode, surface thickness, nearby mesh error.
- **FORCE:** Enabled, analytic shape, force type/strength/direction controls.
- **COLLIDER:** Existing collider workflow, separate from source/sink/force.
- **Domain Height Fill:** Enabled, height, initial velocity, particle toggle; Back
  Mode hidden.
- **Point Injection:** Active-object one-shot action with truthful supported-input
  wording and disabled-state message before solver initialization.

## Hidden compatibility properties

- `source_mode=FIELD`: internal force/pressure representation, not liquid emission.
- `blend_mode` and `feather`: not effective for ordinary particle top-up.
- `height_back_mode`: modes are not behaviorally distinct; PROFILE lacks input.
- `auto_inject_seed`: compatibility property, not a separate normal workflow.

## Failed attempts and limitations

- The P1.1–P1.5 implementation worker exceeded its supervisor turn budget after
  completing the changes; the parent independently reran and passed the full
  focused gate, so no duplicate implementation was performed.
- Particle-system injection was skipped in the visible review because the chosen
  fixture contains no particle system. The skip is explicit rather than reported
  as a runtime pass.
- The known unrelated `tests/blender_cache_export_pipeline.py` hang was not rerun.

## Worktree ownership snapshot

`artifacts/phase1_git_status.txt` contains the final `git status --short`.
Ownership is hunk-level because the listed source files were already dirty.
Phase 1 owns only the source UI/operator/identity hunks described in this report,
the new launcher/probes, focused test hunks, two Phase 1 review notes, and named
evidence artifacts. In particular, `properties.py` changes to
`domain_sdf_walls`, directional/radial/dissipation/attraction force properties,
and registration hardening predate Phase 1 and are not claimed here. All other
dirty entries likewise predate Phase 1 or belong to earlier review/fix work;
none were reset, staged, or rewritten wholesale.

## Performance and memory applicability

Phase 1 changes presentation, operators, host identity, and acceptance tooling;
it adds no GPU kernel, allocation, or per-frame hot-path work. Dedicated
performance before/after measurement is therefore not applicable. Existing CUDA
force and mesh acceptance remained green.

## Residual risks and rollback

- Particle-system injection remains unobserved in Blender because the selected
  fixture has no particle system; focused extraction coverage remains green.
- The failed exploratory coordinate session is retained as non-acceptance
  evidence; the clean automated profile-safe run is authoritative.
- Rollback requires restoring only the listed Phase 1 UI/operator/adapter/docs
  hunks and deleting the new launcher/probe/review files. No `.blend`, RNA name,
  enum identifier, default, or migration must be changed.

## Reviewer gate

- Spec/truth review: original run `906a993b-9abe-4a22-88db-605a05c574d3`,
  recheck `5c7e2e61`, `openai-codex/gpt-5.6-sol`, fresh context. Initial rename
  identity and direct-RNA interaction highs were repaired. Final disposition:
  **no blocker/high remains**.
- Standards/module review: run `63b75370-2967-49a0-bec1-2188e72fcf8e`,
  recheck `08052378`, `openai-codex/gpt-5.6-terra`, fresh context. Hunk-level
  dirty-worktree ownership and evidence-count findings were corrected. Final
  disposition: **accepted with documented residual risk; no blocker/high
  remains**.
- The earlier `anthropic/claude-opus-5` standards attempt failed before review
  because provider usage limits were exhausted; it supplied no acceptance and
  was replaced by the fresh-context Terra review above.

The orchestrator owns the final disposition; reviewer output cannot self-accept
Phase 1.

## Phase 2 backend assumptions and implementation prompt

Phase 2 may rely on stable Phase 1 UI meanings, unchanged RNA identifiers, a
FORCE-only force-region workflow, and the profile-safe review harness. It must
not reinterpret Mesh Surface Shell as solid volume or FIELD as liquid.

Implementation prompt:

> Implement only the accepted backend scope in
> `docs/source_system_refactor/PHASE_2_SOURCE_BACKEND.md`. Preserve the Phase 1
> UI/RNA compatibility contract. Freeze and test the compiled source-region
> interface before adapters; keep new adapters unreachable from normal scene
> translation until their focused and Blender acceptance gates pass. Treat Phase
> 2V Volume as mandatory for full Phase 2 acceptance, and block rather than fake
> grid access when Blender cannot provide an approved truthful path.
