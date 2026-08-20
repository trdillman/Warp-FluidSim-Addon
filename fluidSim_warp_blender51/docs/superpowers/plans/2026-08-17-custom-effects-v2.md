# FluidSim Custom Effects v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace prototype Custom Warp Snippets with a shippable Custom Effects workflow, strict Warp Code workspace, deterministic cache/trust/package contracts, and supplied-runtime demo evidence.

**Architecture:** Keep the existing solver hooks and `sim.custom` import surface for compatibility, but move v2 semantics into a focused `sim/effects/` package. Scene PropertyGroups become adapters for immutable effect snapshots. Candidate runtime managers compile and sandbox the entire enabled stack before an atomic swap; canonical fingerprints participate in cache, checkpoint, bake, headless, and diagnostics contracts.

**Tech Stack:** Blender 5.2 Python API, NVIDIA Warp 1.16.0, NumPy, strict JSON/ZIP effect packages, SHA-256, pytest, Blender background integration tests, Eevee F12 evidence renders, and existing FluidSim cache-v2/checkpoint/release tooling.

## Global Constraints

- All Blender validation must execute `/mnt/data/blender-5.2.0-linux-x64/blender`.
- All Warp validation must import from `/mnt/data/warp_lang_1_16_external`, installed from the supplied wheel.
- Do not claim CUDA validation because this host exposes Warp CPU only.
- Preserve cache-v1 playback and existing cache-v2 playback.
- Preserve Phase 1 transactional/resume guarantees and fail closed on incompatible effect identity.
- Preserve existing public snippet operator IDs as compatibility adapters for one release.
- Unknown embedded/imported custom code is disabled until the exact SHA-256 content is trusted locally.
- A failed compile, sandbox test, or Apply must preserve the previously applied runtime stack.
- Invalid enabled effects must never be silently omitted.
- Release artifacts must not bundle Blender or Warp runtimes.
- Target release version is `1.0.251`.

---

### Task 1: Effect schema, field registry, fingerprinting, and trust

**Files:**
- Create: `sim/effects/__init__.py`
- Create: `sim/effects/enums.py`
- Create: `sim/effects/schema.py`
- Create: `sim/effects/fields.py`
- Create: `sim/effects/fingerprint.py`
- Create: `sim/effects/trust.py`
- Create: `tests/test_effect_schema.py`
- Create: `tests/test_effect_fields.py`
- Create: `tests/test_effect_fingerprint.py`
- Create: `tests/test_effect_trust.py`

**Interfaces:**
- Produces: `EffectDefinitionV2`, `EffectInstanceV2`, `KernelDefinitionV2`, `ParameterDefinitionV2`, `FieldBindingV2`.
- Produces: `effect_snapshot_dict(instance) -> dict[str, object]` and `effect_fingerprint(instance) -> str`.
- Produces: `stack_fingerprint(instances) -> tuple[str, tuple[dict[str, object], ...]]`.
- Produces: `FIELD_SPECS`, `validate_field_binding(binding, kernel) -> tuple[EffectIssue, ...]`.
- Produces: `TrustStore` and `content_trust_digest(definition) -> str`.

- [ ] **Step 1: Write failing schema, field, fingerprint, and trust tests**

Tests must cover strict unknown-key rejection, enum validation, parameter range/type validation, duplicate IDs, legal particle/voxel fields, rejected cross-domain bindings, fingerprint changes for every behavior-affecting property, stable canonical ordering, built-in trust, local trust by exact digest, and trust invalidation after source edits.

- [ ] **Step 2: Run the focused tests and verify they fail**

```bash
python -m pytest tests/test_effect_schema.py tests/test_effect_fields.py \
  tests/test_effect_fingerprint.py tests/test_effect_trust.py -q
```

Expected: import failures because `sim.effects` does not yet exist.

- [ ] **Step 3: Implement enums and immutable strict dataclasses**

Define:

```python
class EffectDomain(str, Enum):
    PARTICLES = "PARTICLES"
    ACTIVE_VOXELS = "ACTIVE_VOXELS"

class EffectStage(str, Enum):
    BEFORE_EMISSION = "BEFORE_EMISSION"
    AFTER_PARTICLE_TRANSFER = "AFTER_PARTICLE_TRANSFER"
    BEFORE_PRESSURE = "BEFORE_PRESSURE"
    AFTER_PARTICLE_UPDATE = "AFTER_PARTICLE_UPDATE"

class EffectCadence(str, Enum):
    EVERY_SUBSTEP = "EVERY_SUBSTEP"
    ONCE_PER_FRAME = "ONCE_PER_FRAME"
    INITIALIZATION = "INITIALIZATION"

class FieldAccess(str, Enum):
    READ = "READ"
    WRITE_CURRENT = "WRITE_CURRENT"
    READ_WRITE_CURRENT = "READ_WRITE_CURRENT"
    ATOMIC_SCATTER = "ATOMIC_SCATTER"
    UNSAFE_ARBITRARY = "UNSAFE_ARBITRARY"

class ParameterKind(str, Enum):
    FLOAT = "FLOAT"
    INT = "INT"
    BOOL = "BOOL"
    VECTOR = "VECTOR"
    COLOR = "COLOR"
    ENUM = "ENUM"
    OBJECT = "OBJECT"
```

Dataclasses use deterministic `to_dict()` and strict `from_dict()` methods. Runtime object pointers are represented in snapshots by stable binding records, not Python object identity.

- [ ] **Step 4: Implement the authoritative field registry**

Each `FieldSpec` contains key, dtype declaration, logical domain, units, legal stages, allowed access modes, runtime accessor name, current-element expression, prerequisite set, and structural/persistent flags. Include all existing `FIELD_ACCESSORS` entries, but reject particle fields from voxel kernels and grid fields from particle kernels in Safe mode.

- [ ] **Step 5: Implement canonical serialization and SHA-256**

Use UTF-8 JSON with sorted keys, no NaN/Infinity, and compact separators. Include definition/API/ABI identity, source, bindings, parameters, controller snapshots, cadence/window, order, mute/solo/enabled, and trust content digest. Aggregate stack order must affect the stack fingerprint.

- [ ] **Step 6: Implement a local digest trust store**

`TrustStore(path)` reads/writes schema-versioned JSON atomically with fsync. Package trust claims are ignored. `trust(digest, label)`, `block(digest, reason)`, `status(digest, built_in=False)`, and `revoke(digest)` are required.

- [ ] **Step 7: Run focused tests**

```bash
python -m pytest tests/test_effect_schema.py tests/test_effect_fields.py \
  tests/test_effect_fingerprint.py tests/test_effect_trust.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add sim/effects tests/test_effect_*.py
git commit -m "feat: define versioned custom effect contracts"
```

---

### Task 2: Static validation, generated source, real Warp compilation, and sandbox execution

**Files:**
- Create: `sim/effects/validation.py`
- Create: `sim/effects/source.py`
- Create: `sim/effects/compiler.py`
- Create: `sim/effects/sandbox.py`
- Create: `tests/test_effect_validation.py`
- Create: `tests/test_effect_source.py`
- Create: `tests/test_effect_compiler.py`
- Create: `release/validate_custom_effects_warp.py`
- Modify: `sim/custom.py`

**Interfaces:**
- Consumes: Task 1 schema/field/trust interfaces.
- Produces: `validate_effect(instance, *, safe_mode=True) -> ValidationResult`.
- Produces: `generate_kernel_source(instance, kernel_index) -> GeneratedKernelSource` with line maps.
- Produces: `compile_effect(instance, *, device) -> CompiledEffect`.
- Produces: `sandbox_test(compiled, *, device) -> SandboxReport`.
- Produces: `build_candidate_effects(instances, *, device, trust_store) -> CandidateBuildResult`.

- [ ] **Step 1: Write failing validation/source/compiler tests**

Negative fixtures must include scalar/vec3 assignment mismatch, missing Warp built-in, duplicate argument, invalid identifier, wrong-domain field, forbidden structural write, arbitrary scatter in Safe mode, unbounded loop, `print` in production code, out-of-range parameter, and source-line mapping. Positive tests cover particle and active-voxel kernels.

- [ ] **Step 2: Run tests with the supplied Warp package and verify failure**

```bash
PYTHONPATH=/mnt/data/warp_lang_1_16_external:$PWD \
python -m pytest tests/test_effect_validation.py tests/test_effect_source.py \
  tests/test_effect_compiler.py -q
```

- [ ] **Step 3: Implement strict AST validation**

Validate legal names, declared parameters/fields, bounded constant loops, field subscripts, current-element write expressions, access permissions, safe/expert capabilities, duplicate identifiers, and stage/domain compatibility. Diagnostics carry effect ID, kernel ID, source line, column, code, severity, and corrective hint.

- [ ] **Step 4: Implement deterministic generated source and line maps**

Generate real files in a content-addressed cache directory. Names use SHA-256, not SHA-1. Preserve the user body line offset and map generated traceback/compile locations to original Text lines. Runtime parameter arrays are passed as typed launch inputs rather than captured as source literals where Warp supports it.

- [ ] **Step 5: Implement forced Warp compilation**

Import the generated module, call `wp.load_module(module=<module>, device=<device>)`, synchronize, and record Warp version, module path, device alias, source digest, compile duration, and diagnostics. Import success alone must not set compiled status.

- [ ] **Step 6: Implement bounded sandbox launches**

Create small guarded particle and voxel fixtures. Launch each kernel with legal field arrays, synchronize, verify canaries, expected mutation bounds, dtypes/shapes, and finite values. Reject buffer corruption or non-finite output.

- [ ] **Step 7: Make `sim.custom` a v1 compatibility wrapper**

Legacy `CustomSnippet` conversion builds one-kernel v2 effects. `CompiledSnippet` delegates to v2 compilation. `SnippetManager.set_snippets()` becomes fail-closed and returns a structured candidate result instead of silently skipping invalid entries.

- [ ] **Step 8: Add supplied-Warp validation script**

The script loads all built-ins and negative fixtures, uses the exact supplied Warp root, executes CPU compile/sandbox/determinism checks, writes atomic JSON, and returns nonzero on any false positive.

- [ ] **Step 9: Run focused tests and the supplied-Warp gate**

```bash
PYTHONPATH=/mnt/data/warp_lang_1_16_external:$PWD \
python -m pytest tests/test_effect_validation.py tests/test_effect_source.py \
  tests/test_effect_compiler.py -q

PYTHONPATH=/mnt/data/warp_lang_1_16_external:$PWD \
python release/validate_custom_effects_warp.py \
  --expected-warp-root /mnt/data/warp_lang_1_16_external \
  --device cpu \
  --report-json /mnt/data/custom_effects_warp_validation_dev.json
```

- [ ] **Step 10: Commit**

```bash
git add sim/effects sim/custom.py tests release/validate_custom_effects_warp.py
git commit -m "feat: compile and sandbox custom effects truthfully"
```

---

### Task 3: Transactional runtime manager, scheduling, and profiling

**Files:**
- Create: `sim/effects/runtime.py`
- Create: `tests/test_effect_runtime.py`
- Modify: `sim/solver.py`
- Modify: `diagnostics/profiler.py`
- Modify: `blender/simulation_admission.py`

**Interfaces:**
- Consumes: `CompiledEffect`, `CandidateBuildResult`.
- Produces: `EffectRuntimeManager`.
- Produces: `EffectRuntimeManager.build_candidate(...)` and `swap_candidate(candidate)`.
- Produces: `requirements_for_stage(stage) -> EffectRequirements`.
- Produces: per-effect `EffectRuntimeStats`.

- [ ] **Step 1: Write failing runtime tests**

Cover atomic swap, failed candidate preserving the prior manager/fingerprint, fail-closed enabled errors, mute/solo/order, frame window, cadence, initialization-only behavior, once-per-frame behavior, and capability-derived topology/SDF/collision requirements.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/test_effect_runtime.py -q
```

- [ ] **Step 3: Implement candidate/applied runtime state**

A manager contains immutable compiled entries grouped by public stage. Candidate construction produces either a complete valid candidate or errors with no partial runtime. `swap_candidate()` is the only mutation point.

- [ ] **Step 4: Implement stage/cadence scheduling**

Map public stages to the existing solver hook sequence. Track frame/substep/initialization calls so once-per-frame and initialization-only kernels cannot execute multiple times.

- [ ] **Step 5: Replace global `has_any()` topology behavior**

Solver topology, SDF, collision, and particle prerequisites are derived from enabled applied effect capabilities. Particle-velocity-only effects must not set `rebuild_each_substep`.

- [ ] **Step 6: Add profiling and safe runtime failure capture**

Record calls, elements, submission duration, prerequisites, and last runtime error per effect/kernel. Runtime errors stop the effect stack at a safe boundary and surface structured state; they do not silently disappear.

- [ ] **Step 7: Run runtime and existing solver tests**

```bash
python -m pytest tests/test_effect_runtime.py tests/test_bake_state_machine.py \
  tests/test_checkpoint_contract.py -q
```

- [ ] **Step 8: Commit**

```bash
git add sim/effects/runtime.py sim/solver.py diagnostics/profiler.py \
  blender/simulation_admission.py tests/test_effect_runtime.py
git commit -m "feat: apply effect stacks transactionally"
```

---

### Task 4: Built-in effect library and advanced parameter/controller support

**Files:**
- Create: `sim/effects/builtins.py`
- Create: `sim/effects/library.py`
- Create: `tests/test_effect_builtins.py`
- Create: `resources/effects/*.json`
- Create: `resources/effects/sources/*.py`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Produces: `builtin_effect_definitions() -> tuple[EffectDefinitionV2, ...]`.
- Produces: `EffectLibrary` with search/category/compatibility APIs.

- [ ] **Step 1: Write failing built-in tests**

Require unique IDs, strict schema, built-in digest trust, legal bindings, meaningful descriptions, typed ranges/units, deterministic compile/sandbox behavior, and controller declarations where spatial placement is required.

- [ ] **Step 2: Implement eight production built-ins**

Create Vortex Force, Curl Turbulence, Radial Pulse, Planar Split, Velocity Damping, Temperature Lift, Particle Speed Limit, and Voxel Cooling. Use vector/color/bool/enum/object parameter kinds. Vortex, Pulse, and Split use controller-object transforms rather than raw coordinate constants.

- [ ] **Step 3: Implement multi-kernel effect support**

The schema/runtime already accepts multiple kernels. Add one built-in demonstration whose kernels run at two stages while presenting one artist effect instance, proving coherent effect-level management.

- [ ] **Step 4: Run built-in tests and supplied-Warp validation**

```bash
PYTHONPATH=/mnt/data/warp_lang_1_16_external:$PWD \
python -m pytest tests/test_effect_builtins.py -q

PYTHONPATH=/mnt/data/warp_lang_1_16_external:$PWD \
python release/validate_custom_effects_warp.py \
  --expected-warp-root /mnt/data/warp_lang_1_16_external \
  --device cpu \
  --report-json /mnt/data/custom_effects_builtins_validation_dev.json
```

- [ ] **Step 5: Commit**

```bash
git add sim/effects resources/effects tests/test_effect_builtins.py THIRD_PARTY_NOTICES.md
git commit -m "feat: add artist-ready Warp effect library"
```

---

### Task 5: Blender scene model, migration, trust, and draft/applied state

**Files:**
- Create: `blender/effects.py`
- Create: `blender/effect_library.py`
- Create: `tests/test_blender_effect_adapter_source.py`
- Modify: `properties.py`
- Modify: `blender/snippets.py`
- Modify: `blender/pending_settings.py`
- Modify: `blender/scene_bridge.py`
- Modify: `blender/simulation_admission.py`

**Interfaces:**
- Produces: `effect_instance_from_pg(scene, pg) -> EffectInstanceV2`.
- Produces: `build_effect_stack(scene, *, trust_store) -> EffectStackBuild`.
- Produces: `migrate_legacy_snippets(scene) -> MigrationReport`.
- Produces: scene-level draft/tested/applied fingerprints and status.

- [ ] **Step 1: Write failing source/adapter tests**

Cover property registration source, v1 migration preservation, `VOXEL_TILE` migration warning, unknown embedded code disabled/untrusted, stable controller binding snapshots, draft fingerprint invalidation on every behavior-affecting edit, and legacy operator compatibility.

- [ ] **Step 2: Add v2 PropertyGroups**

Define effect instances, typed parameter values, kernel field bindings, diagnostics, profiling, package metadata, trust state, and scene stack state. Keep legacy collections for migration only and hide them after successful conversion.

- [ ] **Step 3: Implement Blender adapters and migration**

Convert PropertyGroups to strict snapshots. Migrate legacy snippets without rewriting source. Create local trust-store paths through `bpy.utils.extension_path_user`. Unknown scene code remains disabled until Review & Trust.

- [ ] **Step 4: Integrate candidate builds into initialization and Apply**

Scene admission rejects invalid, dirty, untrusted, or untested enabled effects. A live solver receives only a complete candidate. Draft failure leaves the existing applied fingerprint and runtime unchanged.

- [ ] **Step 5: Run adapter and lifecycle tests**

```bash
python -m pytest tests/test_blender_effect_adapter_source.py \
  tests/test_registration_source.py tests/test_namespace_safety.py -q
```

- [ ] **Step 6: Commit**

```bash
git add blender/effects.py blender/effect_library.py blender/snippets.py \
  blender/pending_settings.py blender/scene_bridge.py blender/simulation_admission.py \
  properties.py tests/test_blender_effect_adapter_source.py
git commit -m "feat: persist trusted effect stacks in Blender scenes"
```

---

### Task 6: Artist-facing Custom Effects UI and workflow

**Files:**
- Create: `blender/effect_ui.py`
- Create: `tests/test_effect_ui_source.py`
- Modify: `operators.py`
- Modify: `panels.py`
- Modify: `__init__.py`

**Interfaces:**
- Produces: Custom Effects `UIList` and panel.
- Produces operators for add, reorder, mute, solo, duplicate, remove, test, compile/apply, revert, preview, controller create/select, trust, and untrust.
- Preserves legacy snippet operator IDs as forwarding adapters.

- [ ] **Step 1: Write failing UI source tests**

Require UIList registration, visible status language, no raw stage/domain/field controls in Artist mode, stack operations, controller actions, compile/apply/revert/preview actions, trust warning, and direct access outside the deeply nested Advanced panel.

- [ ] **Step 2: Implement Custom Effects stack UI**

Use `UIList` rows with enabled/mute/solo, effect name, status icon, cost indicator, and reorder controls. Provide a searchable intent/category library and typed parameter rendering.

- [ ] **Step 3: Implement controller workflow**

Create/reuse owned Empty controllers with effect-specific display shapes, names, transforms, and owner markers. Three-step Vortex path: Add Vortex, Compile & Apply, Preview One Frame.

- [ ] **Step 4: Implement transactional actions and state display**

`Test Draft` runs real compile/sandbox. `Compile & Apply` builds the complete candidate. `Revert Draft` restores the applied snapshot. Dirty state is automatic. Production controls show blocking reasons.

- [ ] **Step 5: Convert legacy snippet operators to adapters**

Old add/load/test/apply IDs forward to v2 operations or migration messages. Remove the inline line editor and raw shortcut panels from the public class list.

- [ ] **Step 6: Run UI source tests and Blender registration validation**

```bash
python -m pytest tests/test_effect_ui_source.py tests/test_registration_source.py -q
/mnt/data/blender-5.2.0-linux-x64/blender --command extension validate "$PWD"
```

- [ ] **Step 7: Commit**

```bash
git add blender/effect_ui.py operators.py panels.py __init__.py tests/test_effect_ui_source.py
git commit -m "feat: add artist-facing custom effects workflow"
```

---

### Task 7: Warp Code workspace, source-mapped diagnostics, API browser, and profiling UI

**Files:**
- Create: `blender/effect_workspace.py`
- Create: `tests/test_effect_workspace_source.py`
- Modify: `operators.py`
- Modify: `panels.py`
- Modify: `properties.py`

**Interfaces:**
- Produces: Open in Text Editor, Go to Error Line, Show Generated Source, API browser, Safe/Expert controls, cadence/stage/domain controls, field access editing, and profiling display.

- [ ] **Step 1: Write failing workspace source tests**

Require Text datablock editing, no inline per-line editor, source-mapped error navigation, generated-source action, registry-derived field/stage options, safe/expert acknowledgement, cadence controls, and profiling labels.

- [ ] **Step 2: Implement Text Editor integration**

Open or switch a `TEXT_EDITOR` area to the selected effect source. `Go to Error Line` positions the cursor at mapped line/column. Generated source is written to a read-only-named Text datablock with digest metadata.

- [ ] **Step 3: Implement API browser and permissions**

Render fields filtered by domain/stage with dtype, units, access, persistence, and cost/prerequisite descriptions. Expert capabilities require a scene acknowledgement and appear in reports.

- [ ] **Step 4: Add profiling evidence and target runtime display**

Show selected device, Warp version, compiled source digest, compile/sandbox time, calls, elements, submission time, and last runtime error.

- [ ] **Step 5: Run source tests and Blender validation**

```bash
python -m pytest tests/test_effect_workspace_source.py -q
/mnt/data/blender-5.2.0-linux-x64/blender --command extension validate "$PWD"
```

- [ ] **Step 6: Commit**

```bash
git add blender/effect_workspace.py operators.py panels.py properties.py \
  tests/test_effect_workspace_source.py
git commit -m "feat: add Warp Code authoring workspace"
```

---

### Task 8: Cache, checkpoint, resume, diagnostics, and headless policy

**Files:**
- Create: `tests/test_effect_cache_contract.py`
- Create: `tests/test_effect_headless_policy.py`
- Modify: `io/cache_schema.py`
- Modify: `io/cache.py`
- Modify: `io/checkpoint.py`
- Modify: `blender/production_bake.py`
- Modify: `blender/command_line.py`
- Modify: `blender/cache_manager.py`
- Modify: `diagnostics/report.py`
- Modify: `release/run_headless_bake.py`

**Interfaces:**
- Consumes: aggregate stack fingerprint/inventory.
- Produces: cache/checkpoint fields `effect_signature` and `effect_inventory`.
- Produces: resume decision code `EFFECT_SIGNATURE_MISMATCH`.
- Produces: CLI `--custom-code-policy deny|builtins-only|trusted`.

- [ ] **Step 1: Write failing cache/headless tests**

Test manifest/checkpoint round-trip, mismatch rejection before mutation, legacy playback, no-resume with active effects and absent identity, code/parameter/order/cadence/controller mutation, report inventory, and each headless trust policy.

- [ ] **Step 2: Extend cache-v2 and checkpoint schemas compatibly**

New writes include signature and ordered inventory. Older cache-v2 manifests remain playable. Checkpoint validation occurs before solver mutation. A scene with active effects cannot resume from a manifest/checkpoint lacking identity.

- [ ] **Step 3: Integrate production-bake and cache inspection**

Bake preflight requires a fully applied stack. Manifest transactions include the exact applied signature. Cache inspection reports effect compatibility and inventory.

- [ ] **Step 4: Add headless custom-code policy**

Default `builtins-only`. `deny` rejects all custom effects. `trusted` accepts built-ins and locally trusted exact digests. Reports include policy, decision, signature, and inventory; failures use a distinct exit code without executing code.

- [ ] **Step 5: Extend support reports**

Include draft/tested/applied signatures, trust statuses, API/ABI versions, effect inventory, compile targets, and runtime profiling. Redact home paths as existing reports do.

- [ ] **Step 6: Run focused and regression tests**

```bash
python -m pytest tests/test_effect_cache_contract.py tests/test_effect_headless_policy.py \
  tests/test_cache_schema.py tests/test_checkpoint_contract.py \
  tests/test_bake_state_machine.py tests/test_command_line.py -q
```

- [ ] **Step 7: Commit**

```bash
git add io blender diagnostics release/run_headless_bake.py \
  tests/test_effect_cache_contract.py tests/test_effect_headless_policy.py
git commit -m "feat: bind effects to cache resume and farm policy"
```

---

### Task 9: Effect Package v2, library management, and legacy JSON migration

**Files:**
- Create: `sim/effects/package.py`
- Create: `tests/test_effect_package.py`
- Modify: `blender/effect_library.py`
- Modify: `operators.py`
- Modify: `panels.py`

**Interfaces:**
- Produces: `write_effect_package(path, definition) -> PackageReport`.
- Produces: `read_effect_package(path) -> EffectDefinitionV2`.
- Produces: `migrate_legacy_preset(payload) -> EffectDefinitionV2`.
- Produces: library import/export/install/update/duplicate/rename/delete/reveal/conflict APIs.

- [ ] **Step 1: Write failing package tests**

Cover strict manifest, ZIP path traversal rejection, digest mismatch, unknown schema, unsupported API, atomic write interruption, import trust state, conflict policy, legacy migration, and no overwrite of legacy files.

- [ ] **Step 2: Implement strict package format**

Use `effect.json`, source files, optional preview/demo metadata, and per-file SHA-256. Validate every archive member, reject symlinks/path traversal/duplicates, and ignore package trust claims.

- [ ] **Step 3: Implement user library operations**

Store packages in extension-managed user data. Writes use sibling temporary files plus fsync and atomic replace. Conflict handling offers keep both, replace installed version, or cancel; scene snapshots are never silently replaced.

- [ ] **Step 4: Implement legacy JSON migration**

Parse the existing schema strictly, map one snippet to one effect, preserve source and constants, map `VOXEL_TILE`, write a new package, and leave the original untouched.

- [ ] **Step 5: Add Blender library operations and diagnostics UI**

Expose import/export, duplicate, rename, delete, reveal, update comparison, and malformed-package diagnostics.

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_effect_package.py tests/test_effect_builtins.py -q
```

- [ ] **Step 7: Commit**

```bash
git add sim/effects/package.py blender/effect_library.py operators.py panels.py \
  tests/test_effect_package.py
git commit -m "feat: package and share versioned custom effects"
```

---

### Task 10: Real demos, Blender integration validation, and correctness analysis

**Files:**
- Create: `release/generate_custom_effect_demos.py`
- Create: `release/validate_custom_effects_blender.py`
- Create: `resources/examples/05_custom_effects_vortex.blend`
- Create: `resources/examples/06_custom_effects_stack.blend`
- Create: `resources/examples/07_warp_code_authoring.blend`
- Create: `resources/examples/custom_effects/README.md`
- Create: `tests/test_custom_effect_demo_assets.py`
- Modify: `release/generate_examples.py`
- Modify: `release/validate_examples_blender.py`
- Modify: `release/smoke_test_blender.py`

**Interfaces:**
- Produces three reproducible `.blend` demos, CPU Warp result NPZ/JSON, F12 PNG evidence, and atomic validation report.

- [ ] **Step 1: Write failing demo asset tests**

Require generator/validator scripts, three scene names, effect inventory, controller objects, Text datablocks, applied signatures, demo metadata, and no bundled runtime payloads.

- [ ] **Step 2: Implement deterministic supplied-Warp demo simulations**

Use the real v2 compiler/runtime on CPU fixtures to simulate multiple steps for Vortex, Damping, and a multi-effect stack. Save initial/final arrays and analytic metrics: radius/tangential velocity change, kinetic-energy damping ratio, unchanged canaries, finite values, deterministic rerun equality, and stack-order sensitivity.

- [ ] **Step 3: Generate Blender demo scenes**

Create scenes containing effect stacks, controllers, source Text datablocks, initial/final point visualizations, annotations, materials, cameras, and report custom properties. The scenes must reopen with exact applied signatures and trust states.

- [ ] **Step 4: Render F12 evidence with supplied Blender**

Render clear before/after or layered visualizations in Eevee. The validator analyzes images and scene data, not merely file existence.

- [ ] **Step 5: Validate UI operations and save/reopen**

In supplied Blender background mode, add a Vortex from the library, create/select controller, Test, Compile & Apply, reorder/mute/solo/duplicate, open Text Editor, map a negative compile error to source line, export/import package, save/reopen, and confirm untrusted imported code remains disabled.

- [ ] **Step 6: Run demo tests and validation**

```bash
python -m pytest tests/test_custom_effect_demo_assets.py -q

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
FLUIDSIM_EXTERNAL_WARP_ROOT=/mnt/data/warp_lang_1_16_external \
/mnt/data/blender-5.2.0-linux-x64/blender --background --factory-startup \
  --offline-mode --python-use-system-env --python-exit-code 1 \
  --python release/generate_custom_effect_demos.py -- \
  --module fluidsim_snippets_repo \
  --output resources/examples \
  --work-dir /mnt/data/custom_effect_demo_work \
  --report-json /mnt/data/custom_effect_demo_generation.json

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
FLUIDSIM_EXTERNAL_WARP_ROOT=/mnt/data/warp_lang_1_16_external \
/mnt/data/blender-5.2.0-linux-x64/blender --background --factory-startup \
  --offline-mode --python-use-system-env --python-exit-code 1 \
  --python release/validate_custom_effects_blender.py -- \
  --module fluidsim_snippets_repo \
  --examples resources/examples \
  --work-dir /mnt/data/custom_effect_blender_validation \
  --expected-warp-root /mnt/data/warp_lang_1_16_external \
  --report-json /mnt/data/custom_effects_blender_validation.json
```

- [ ] **Step 7: Commit**

```bash
git add release resources/examples tests/test_custom_effect_demo_assets.py
git commit -m "test: add real custom effect demos and Blender proof"
```

---

### Task 11: Documentation, versioning, release workflow, deep review, and final artifacts

**Files:**
- Modify: `version.py`
- Modify: `blender_manifest.toml`
- Modify: `CHANGELOG.md`
- Modify: `docs/user_guide.md`
- Modify: `docs/technical_report.md`
- Create: `docs/custom_effects.md`
- Create: `docs/custom_effects_api.md`
- Create: `docs/custom_effects_handoff.md`
- Modify: `.github/workflows/release-checks.yml`
- Modify: `release/build_extension.py`
- Modify: `release/delivery_policy.py`
- Modify: `tests/test_build_extension.py`
- Modify: `tests/test_release_workflow_source.py`

**Interfaces:**
- Produces: release `1.0.251`, Linux x64 extension, source ZIP, demos ZIP, validation bundle, JSON reports, Git bundle, command ledger, and SHA-256 sums.

- [ ] **Step 1: Update version and documentation**

Document Artist workflow, Warp Code API, trust boundary, cache/resume identity, headless policy, package format, migration, demos, supplied-runtime evidence, known limits, and CUDA/Windows gates. Remove stale public `Custom Warp Snippets` claims except migration notes.

- [ ] **Step 2: Extend release workflow**

Add pure tests, supplied-Warp CPU gate, supplied-Blender integration gate, archive checks, and a required NVIDIA `cuda:0` job for release tags. Keep CUDA results unresolved locally.

- [ ] **Step 3: Run the full regression suite and compile checks**

```bash
python -m pytest -q
PYTHONPATH=/mnt/data/warp_lang_1_16_external:$PWD python -m pytest -q
python -m compileall -q -f .
git diff --check
```

- [ ] **Step 4: Run all supplied-Warp and supplied-Blender gates**

Run Phase 3, 4, and 5 regression validators, custom effects Warp validation, custom effects Blender validation, example validation, source extension validation, clean-profile package build, installed smoke twice, and installed headless no-CUDA process validation.

- [ ] **Step 5: Perform a deep review and fix findings**

Inspect all custom-effects code paths for signature completeness, transactional state, trust bypasses, package traversal, stale status, Blender lifecycle leaks, line-map correctness, topology scheduling, demo correctness, and release payload policy. Add regression tests before every bug fix.

- [ ] **Step 6: Build release artifacts**

Create:

```text
FluidSim_Custom_Effects_Complete_1.0.251_NoBundledRuntimes.zip
fluidsim_warp-1.0.251-linux_x64.zip
FluidSim_Custom_Effects_Demos_1.0.251.zip
FluidSim_Custom_Effects_Render_Validation_1.0.251.zip
FluidSim_Custom_Effects_Validation_Bundle_1.0.251.zip
FluidSim_Custom_Effects_Validation_1.0.251.json
FluidSim_Custom_Effects_Release_Summary_1.0.251.json
FluidSim_Custom_Effects_Agent_Handoff_1.0.251.md
FluidSim_Custom_Effects_1.0.251_Git.bundle
FluidSim_Custom_Effects_Commands_1.0.251.txt
FluidSim_Custom_Effects_1.0.251_SHA256SUMS.txt
```

Every ZIP must pass CRC, delivery-policy, and bundled-runtime scans. The Git bundle must verify and contain complete history.

- [ ] **Step 7: Run final checksum and repository verification**

```bash
sha256sum -c /mnt/data/FluidSim_Custom_Effects_1.0.251_SHA256SUMS.txt
git status --short --branch
git bundle verify /mnt/data/FluidSim_Custom_Effects_1.0.251_Git.bundle
```

Expected: clean worktree, all checksums pass, bundle complete.

- [ ] **Step 8: Commit final release state**

```bash
git add .
git commit -m "release: ship custom effects workflow 1.0.251"
```
