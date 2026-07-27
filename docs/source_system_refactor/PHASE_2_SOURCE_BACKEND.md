# Phase 2 — Source Backend Normalization and Capability Completion

## Goal statement

Implement a single truthful source-region pipeline that can normalize analytic
regions, mesh solid volumes, mesh surface shells, point/particle inputs, imported
volume grids, and height fields into GPU-resident source representations on the
flat sparse 8×8×8 leaf table, then apply liquid, sink, force, collision/mask, and
advanced field operations without duplicating geometry logic.

Phase 2 completes behavior; it does not perform the final panel redesign.
Capabilities remain hidden or experimental until their behavioral and runtime
gates pass.

## Entry criteria

- Phase 1 is Accepted or Accepted with documented deferral.
- Current UI makes no claims about missing Phase 2 behavior.
- `docs/SOURCE_EMISSION_PRODUCT_PLAN.md` and `docs/source_shape_spec.md` are the
  approved behavior contracts.
- Baseline GPU profile, memory profile, and source-focused test results are saved.
- Decisions about optional external VDB dependencies and performance thresholds
  are explicitly recorded before their work packages begin.

## Exit criteria

The backend supports and validates:

1. Analytic solid source regions.
2. Watertight Mesh Solid Volume source regions.
3. Mesh Surface Shell source regions.
4. Point/particle source inputs with attribute transfer.
5. Imported fog/SDF/scalar/vector volume grids flattened into sparse source tiles.
6. Height-region sourcing with only implemented back/profile modes.
7. Initial Fill and Continuous Inflow over applicable liquid regions.
8. Rename-stable runtime source identity for one solver session.
9. Correct sink, force, mask/collision, and advanced field operations.
10. Cached/dynamic source invalidation without full-domain or full-particle
    host readback in the hot path.

Items 1–4 and 6–10 form the core backend acceptance. Item 5 is a separately
reviewed **Phase 2V Volume milestone** executed after the core compiler exists,
as required by `SOURCE_EMISSION_PRODUCT_PLAN.md`. Full Phase 2 **Accepted**
status requires Phase 2V. If Blender 5.1 cannot provide a truthful grid-access
path, the phase is Blocked until the user selects an implementation option or
explicitly narrows the program goal; it is not silently Accepted with deferral.

## Truth contract

- Solid Volume means interior sampling of a validated watertight region.
- Surface Shell means bounded narrow-band sampling and is never used as a silent
  fallback for invalid solid input.
- Target Particles / Voxel controls sampling occupancy, not world-volume flow rate.
- Initial Fill fires once per source identity and resets only on Initialize/Reset.
- Continuous Inflow tops the region up while enabled and is substep invariant.
- Imported VDB is staging data; live source storage uses the flat leaf table.
- Point sources preserve point attributes and do not pretend to fill a volume
  unless explicitly rasterized.
- FIELD source behavior is field/pressure influence, not liquid emission.
- Capacity-limited, invalid, unsupported, and partially imported sources report
  visible structured errors/warnings.

## Proposed module seam

The writer must design this interface twice before implementation and record why
the selected version has greater depth and locality. The target shape is:

```python
SourcePlan(
    stable_id,
    operation,
    lifecycle,
    region,
    payload,
    activation,
    blend,
)

CompiledSourceBatch = SourceCompiler.compile(plans, frame_context)
SourcePipeline.apply(compiled_batch, dt)
```

The exact types may differ, but these invariants may not:

- Blender adapters return immutable, Blender-independent values.
- No Blender object, PropertyGroup, mathutils proxy, or evaluated mesh lifetime
  crosses into persistent solver state.
- Region compilation owns geometry-specific complexity.
- SourcePipeline owns lifecycle and operation ordering.
- The interface does not expose per-adapter implementation details.
- Tests and Blender use the same public seam.

Likely new ownership:

- `sim/source_regions.py` — region plans, compiled tile/point representations,
  source compiler, cache keys, and invalidation contract.
- `sim/sourcing.py` — source operation execution and lifecycle, consuming
  compiled regions.
- `blender/scene_inputs.py` — Blender adapters only.

File names are recommendations, not permission to layer a pass-through module.
Apply the deletion test: removing the compiler module should force geometry,
cache, validation, and normalization complexity back into multiple callers.

## Work packages and dependencies

### P2.0 — Contract freeze and failing matrix

**Dependencies:** Phase 1 handoff.

Create a claim matrix across:

- region input × operation × lifecycle × payload × evaluation cadence;
- valid, invalid, capacity-limited, animated, reset, and rename cases;
- CPU adapter, GPU compilation, solver operation, Blender integration.

Add failing behavioral tests before changing the interface. Mark unsupported
combinations explicitly rather than allowing accidental behavior.

### P2.1 — Stable source identity and immutable plans

**Dependencies:** P2.0.

- Add a rename-stable runtime source key derived from Blender object identity
  plus the owning solver/session generation; do not key Initial Fill by object name.
- Keep the key host-only and non-persistent unless a later saved-data requirement
  is separately approved. Initialize/Reset creates a new lifecycle session, so a
  persistent UUID is not required by the current contract.
- Convert all Blender values to owned primitives/arrays at the adapter seam.
- Separate operation, region, lifecycle, and payload enums/types.
- Preserve legacy property translation in one compatibility adapter.

**Required tests:** rename does not re-arm Initial Fill within a session;
duplicated objects receive distinct runtime identities; Initialize/Reset re-arms
as specified; save/reopen starts a fresh session; no RNA proxy survives adaptation.

### P2.2 — Compiled sparse region foundation

**Dependencies:** P2.1.

- Define sparse source tiles using fixed 8×8×8 leaves and flat leaf-table
  addressing.
- Support coverage/mask and optional SDF payloads.
- Allocate only leaves intersecting source bounds.
- Define cache keys and invalidation for static and every-frame evaluation.
- Define deterministic sampling seeds from stable source ID, voxel, frame, and
  sample index.
- Establish device ownership, synchronization, memory release, and failure paths.

**Non-negotiable:** no hierarchical live tree and no whole-domain dense mask.

### P2.3 — Analytic and Mesh Solid/Surface regions

**Dependencies:** P2.2.

- Route analytic SDF regions through the compiler interface, preserving current
  output as the reference implementation.
- Add Mesh Source Region: `SOLID_VOLUME` / `SURFACE_SHELL`.
- Solid Volume:
  - validate watertightness only for solid consumers;
  - compute containment/SDF for candidate voxels/samples;
  - sample the interior, not triangle projections.
- Surface Shell:
  - compile unsigned-distance/narrow-band coverage, or rasterize deterministic
    area-weighted surface points into the same region representation;
  - preserve explicit shell half-width.
- Reuse shared mesh analysis without weakening sink validation.
- Never fall back from Solid Volume to Surface Shell.

**Required comparisons:** a watertight cube source approximates an analytic box
at matched resolution and particles/voxel; an open cube fails Solid Volume and
passes Surface Shell.

### P2.4 — Point/particle adapter

**Dependencies:** P2.2. It may run in parallel with P2.3 only after the compiled-
region interface is frozen and each writer uses an isolated worktree.

Support owned GPU point inputs from:

- Blender PointCloud;
- first particle system;
- evaluated mesh vertices;
- generated area-weighted surface scatter;
- future external point caches through the same adapter.

Preserve velocity, temperature, viscosity, density attribute, and lifespan when
present. Distinguish:

- Direct one-shot append.
- Region rasterization/top-up for lifecycle-aware emission.

Do not call vertex injection a volume fill.

### P2.5 / Phase 2V — Volume-grid adapter milestone

**Dependencies:** accepted P2.2 core compiler and an approved discovery checkpoint.

This is a separate acceptance milestone inside Phase 2, not an optional adapter
hidden inside the core implementation. Discovery must prove how Blender
Volume/OpenVDB grids can be read on the target Blender 5.1 runtime without an
unapproved dependency. If direct access is unavailable, stop, mark Phase 2V and
full Phase 2 Blocked, and present evidence-backed options; do not fake volume
support or silently omit it from an Accepted full-program handoff.

When approved:

- Import fog, SDF, scalar, vector, and vector-component grids.
- Transform/resample into solver coordinates.
- Flatten active values into source leaves/tiles.
- Release all hierarchical staging structures before live simulation.
- Preserve mask, scalar/vector payload, grid transform, and missing-grid errors.
- Cache static grids and invalidate animated/evaluated inputs explicitly.

### P2.6 — Height region completion

**Dependencies:** P2.2.

- Represent height fill through the common region interface.
- Implement distinct Infinite, Flat, and Profile semantics only if their input
  contract is defined and testable.
- Add an explicit profile input or keep Profile hidden.
- Add lifecycle behavior; do not leave height fill as an unexplained always-on
  exception.

### P2.7A — Liquid lifecycle and sink operations

**Dependencies:** P2.3; P2.4 or P2.6 only for their respective region adapters.

- Route Initial Fill and Continuous Inflow through compiled regions.
- Preserve substep invariance, overlap occupancy, capacity reporting, and reset.
- Apply sink masks for analytic, solid, shell, and accepted imported masks.
- Gate with lifecycle, overlap, capacity, and sink-removal tests before merge.

### P2.7B — Force, collision, and activation operations

**Dependencies:** P2.2 and the region adapter used by each operation.

- Preserve analytic force-region ordering after P2G.
- Add collision/activation mask operations only where required by the original
  contract and validated independently.
- Do not combine force velocity, emitted-particle initial velocity, and collider
  velocity transfer behind one ambiguous operation.
- Gate with force-stage, collision precedence, and activation tests before merge.

### P2.7C — Advanced field-source operations

**Dependencies:** accepted P2.3/P2.4 region/payload adapters and accepted Phase
2V for imported volume payloads.

Implement and gate separately:

1. pressure sourcing;
2. scalar fields;
3. vector fields;
4. staggered velocity;
5. constant, particle-attribute, and imported-volume payloads;
6. only blend modes proven for each target/payload combination.

Each numbered operation receives its own focused tests and capability-matrix row.
FIELD mode stays hidden until its complete operation-specific behavioral and
Blender acceptance lane passes.

### P2.8 — Caching, profiling, and recovery

**Dependencies:** all enabled adapters.

- Dirty-track geometry, transform, payload, lifecycle, and activation separately.
- Avoid rebuilding static sources every substep.
- Avoid CPU readback of live solver particles for source compilation.
- Bound source tile/point memory and report capacity errors.
- Release device memory on reset, unregister, load, and failed initialization.
- Compare dam-break and animated source performance to the approved baseline.

No agent may invent an acceptable regression percentage after seeing results.

## Agent workflow policy

In addition to the common program policy:

- The phase lead must freeze and document the SourcePlan/compiled-region
  interface before adapter implementation is delegated.
- One writer owns the compiler/pipeline integration path at a time.
- Adapter workers receive the frozen interface, an isolated worktree when they
  write concurrently, and an explicit capability/test slice.
- Read-only discovery work may run in parallel, but it may not make product,
  dependency, storage, or fallback decisions.
- Every accepted work package updates the capability matrix before the next
  package is exposed or delegated.
- New adapter paths remain unreachable from the normal Blender translator until
  their focused and Blender gates pass. This internal capability gate is not a
  saved user preference; tests must prove legacy descriptors still select the
  old accepted path while a package is pending.

### Parallel work policy

Allowed parallelism after interfaces freeze:

- Mesh region implementation and point adapter in isolated worktrees.
- Volume access discovery as read-only research.
- Independent tests/fixtures that do not edit shared source files.

Not allowed:

- Multiple writers in `sim/sourcing.py`, `sim/source_regions.py`, or
  `blender/scene_inputs.py`.
- UI implementation in Phase 3 before a capability's Phase 2 gate passes.
- Volume adapter and live-grid storage decisions made by different writers.

The phase lead integrates one work package at a time and runs the focused gate
after each integration.

## Required validation

### Core source suite

```bash
python -m pytest -q \
  tests/test_sourcing_pipeline.py \
  tests/test_mesh_source_pipeline.py \
  tests/test_source_emission_lifecycle.py \
  tests/test_source_field_force_contract.py \
  tests/test_force_subsystem_contracts.py \
  tests/test_sparse_leaf_grid_contract.py \
  tests/test_memory_safety_recovery.py \
  tests/test_performance_hotpath_contracts.py
```

Add focused tests for source-region compilation, point attributes, volume import,
stable identity, height profiles, and field operations. Those tests become
mandatory once their work package lands.

### Blender/runtime gates

```bash
blender --background --factory-startup --python tests/blender_mesh_source_acceptance.py
blender --background --factory-startup --python tests/blender_force_region_acceptance.py
blender --background --factory-startup --python tests/blender_memory_safety_gate.py
```

The following scripts and markers must be added by the owning work package and
become mandatory when that package is integrated:

```bash
blender --background --factory-startup \
  --python tests/blender_source_region_acceptance.py -- \
  --json-out artifacts/source_region_acceptance.json
# SOURCE-REGION-ACCEPTANCE-PASS

blender --background --factory-startup \
  --python tests/blender_point_source_acceptance.py -- \
  --json-out artifacts/point_source_acceptance.json
# POINT-SOURCE-ACCEPTANCE-PASS

blender --background --factory-startup \
  --python tests/blender_source_lifecycle_acceptance.py -- \
  --json-out artifacts/source_lifecycle_acceptance.json
# SOURCE-LIFECYCLE-ACCEPTANCE-PASS

blender --background --factory-startup \
  --python tests/blender_animated_source_acceptance.py -- \
  --json-out artifacts/animated_source_acceptance.json
# ANIMATED-SOURCE-ACCEPTANCE-PASS
```

Phase 2V additionally requires:

```bash
blender --background --factory-startup \
  --python tests/blender_volume_source_acceptance.py -- \
  --json-out artifacts/volume_source_acceptance.json
# VOLUME-SOURCE-ACCEPTANCE-PASS
```

Every script must write and validate this canonical JSON shape before printing
its positive marker:

```json
{
  "schema": "fluidsim-source-acceptance-v1",
  "ok": true,
  "capability": "mesh_solid_volume",
  "identity": {
    "blender_version": "5.1.0",
    "blender_hash": "adfe2921d5f3",
    "addon_module": "fluidSim_warp_blender51",
    "addon_version": [1, 0, 0],
    "addon_path": "C:/absolute/path/to/__init__.py",
    "device": "cuda:0",
    "device_name": "NVIDIA ..."
  },
  "fixture": {
    "name": "watertight_cube",
    "blend_path": "C:/absolute/or/empty/path.blend"
  },
  "measurements": {
    "active_voxels": 0,
    "emitted_particles": 0,
    "coverage_ratio": 0.0,
    "attribute_checks": {
      "velocity": true,
      "temperature": true,
      "viscosity": true,
      "density": true,
      "lifespan": true
    }
  },
  "invalid_input": {
    "case": "open_mesh_as_solid",
    "rejected": true,
    "message": "non-empty actionable message"
  },
  "timings_ms": {
    "compile": 0.0,
    "step_median": 0.0,
    "step_p95": 0.0
  },
  "invariants": [
    {
      "id": "solid_particles_inside",
      "passed": true,
      "expected": "all sampled particles inside",
      "actual": "all sampled particles inside"
    }
  ],
  "errors": []
}
```

Types are fixed: booleans are JSON booleans; counts are integers; ratios/timings
are numbers; identity/messages are strings; `addon_version` is an integer array;
`invariants` is an object array; `errors` is a string array. A capability that
does not use a measurement or attribute keeps the canonical key with `null`.
Additional capability-specific data is allowed only under
`measurements.details` as a JSON object. Validators require schema equality,
all canonical keys/types, every invariant `passed: true`, empty `errors`, exit
code zero, and the exact positive marker. Report existence or non-crash is
insufficient.

### GPU profile

Add and run an exact source-pipeline profiler:

```bash
blender --background --factory-startup \
  --python tools/profile_source_pipeline.py -- \
  --frames 30 \
  --json-out artifacts/source_pipeline_profile.json
# SOURCE-PIPELINE-PROFILE-PASS
```

The JSON contains warmup/measured frame counts, device identity, peak allocated
source memory, compile time, median/p95 step time, and per-case results for:

- analytic source baseline;
- static solid mesh;
- animated shell mesh;
- point rasterization;
- Phase 2V volume-grid sampling;
- no-source control.

Phase 2V's case is mandatory for full Phase 2 acceptance. Thresholds are
recorded before implementation; the profiler does not invent pass limits.

### Production subset

```bash
FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py
```

Full GUI production acceptance is reserved for Phase 3 unless Phase 2 changes a
visible workflow or the phase lead requires it.

## Reviewer gates

### R1 — module/interface review

A fresh reviewer must verify:

- Source compiler is a deep module, not a descriptor pass-through.
- Blender adapters contain no solver policy.
- SourcePipeline contains no Blender-specific geometry acquisition.
- Operation, region, lifecycle, and payload remain independent.
- Tests cross the production interface.

### R2 — original-semantics truth review

A fresh reviewer must compare behavior to `source_shape_spec.md` and the original
kernel inventory. It must explicitly reject:

- FIELD liquid emission;
- shell-as-solid fallback;
- labels/tests that imply unsupported volume import;
- point injection represented as volume fill;
- source-name lifecycle identity.

### R3 — GPU/sparse-grid review

Must verify:

- fixed 8×8×8 flat leaf-table storage;
- bounded allocation and teardown;
- no unintended full-domain dense buffers;
- no hot-path full particle readback;
- static/dynamic cache invalidation correctness;
- synchronization and device ownership.

### R4 — behavioral acceptance review

Must inspect quantitative fixtures, not screenshots alone:

- occupancy/coverage against analytic references;
- shell thickness;
- watertight rejection;
- lifecycle/substep invariance;
- attribute transfer;
- capacity and error behavior;
- source motion and object-velocity inheritance.

No blocker/high finding may remain. Performance deviations require an approved
waiver with measurements.

## Phase handoff requirements

The Phase 2 handoff must additionally include:

- Final SourcePlan and compiled-region interface documentation.
- Adapter/operation capability matrix with Real/Partial/Missing verdicts.
- Runtime source-identity, duplication, rename, reset, and reopen semantics.
- GPU memory layout and cache-invalidation diagram.
- Quantitative mesh solid/shell and volume/point validation results.
- Performance and memory JSON artifacts.
- Error/warning catalog surfaced to Phase 3 UI.
- Exact list of capabilities approved for exposure.
- Exact list that must remain hidden or Experimental.
- A Phase 3 UI schema mapping each approved capability to controls and help text.

## Rollback

Each new adapter path remains unreachable from the normal Blender translator
until accepted; direct tests may exercise it through the frozen interface. The
acceptance commit enables translation for that capability and adds a regression
that legacy descriptors still resolve correctly. Legacy analytic and shell
behavior remain available during migration. Rollback removes one adapter's
translation/implementation without rewriting saved scenes; compatibility
translation must continue to emit valid SourcePlans.
