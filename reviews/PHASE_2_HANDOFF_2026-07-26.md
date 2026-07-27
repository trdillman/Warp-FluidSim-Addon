# Phase 2 Handoff — Source Backend Normalization and Capability Completion

## Verdict

**Accepted.** Phase 2 core and mandatory Phase 2V are implemented through the frozen backend seam. The focused suite, five quantitative Blender/CUDA lanes, approved Balanced profile, headless production subset, and consolidated R1–R4 review are green after all blocker/high findings were repaired.

This is backend acceptance, not Phase 3 UI exposure. Point-rasterized, Volume, Mesh Solid, and advanced field plans remain reachable through the frozen seam/direct acceptance until Phase 3 adds a reviewed translator/control schema.

## Delivered scope

- Immutable, Blender-independent `SourcePlan` axes: stable identity, operation, lifecycle, cadence, region, payload, activation, and blend.
- Deep `SourceCompiler` producing flat fixed 8×8×8 source leaves and owned point payloads.
- Analytic SDF, watertight Mesh Solid Volume, explicit Mesh Surface Shell, direct/rasterized points, Flat/Profile height, and imported OpenVDB fog/SDF/scalar/vector/component grids.
- Initial Fill/Once and Continuous Inflow/Top Up lifecycle, session-stable rename identity, reset/reopen re-arm, overlap occupancy, capacity results, and diagnostics.
- Liquid, sink, collision, activation, pressure, scalar, vector, staggered MAC velocity, and accepted legacy post-P2G FORCE behavior.
- Compiler cache ownership, one retained generation per stable source, deterministic seeds, bounded preflight, empty-leaf compaction, teardown, and recovery.
- Canonical report validator, five Blender acceptance lanes, and the measured source profiler.

## Frozen interface and ownership

The two candidate interfaces and the selected deeper design are recorded in
`docs/source_system_refactor/PHASE_2_SOURCEPLAN_CONTRACT.md`.

```text
Blender/compatibility adapters
  -> immutable SourcePlan
  -> SourceCompiler.compile(plans, CompileContext)
  -> CompiledSourceBatch[CompiledRegion]
  -> SourcePipeline operation/lifecycle ordering
  -> solver flat leaf table and GPU fields/particles
```

- `sim/source_regions.py`: geometry validation, bounds, candidate leaves, mesh/OpenVDB staging, transforms, deterministic seeds, cache keys, diagnostics, uploads, and release.
- `sim/source_compat.py`: sole legacy descriptor-to-plan adapter.
- `sim/sourcing.py`: operation ordering, lifecycle, and compiled GPU application; no Blender geometry acquisition.
- `blender/scene_inputs.py`: copies Blender-owned values/arrays and creates session/object identities; no solver policy.
- No `bpy`, RNA, mathutils proxy, evaluated mesh, persistent `wp.Mesh`, or OpenVDB tree crosses into plan/live solver state.

Deleting `SourceCompiler` would force mesh containment, OpenVDB conversion, bounds, sparse allocation, caching, diagnostics, and device ownership into multiple callers; it is not a pass-through module.

## GPU storage and cache layout

```text
Source bounds
  -> candidate leaf coordinates [N,3]
  -> compact covered leaves only
  -> flat payload [N,512] / [N,512,3]
  -> solver leaf_table lookup (8x8x8 LEAF_DIM)
```

- No source hierarchy and no whole-domain dense live mask.
- OpenVDB hierarchy is staging only; each grid's authored index-to-world transform is composed with the plan/object transform, then staging grids/accessors are released.
- Production-scale Static/First Sim Frame meshes use Warp CUDA BVH staging. Every Sim Frame compiled meshes are intentionally limited to 40,000 voxel/triangle comparisons, derived from the approved interactive limits; larger animated inputs receive `SOURCE_DYNAMIC_MESH_BUDGET` and must use First Sim Frame or lower source resolution.
- Superseded static or dynamic cache generations are released. Reset/teardown clears all retained entries.
- Source-memory profile values are conservative flat host/staging payload-byte proxies, not whole-device allocator telemetry.

## Capability matrix

| Capability | Verdict | Reachability after Phase 2 |
|---|---|---|
| Analytic solid liquid region | Real | Existing translator/UI |
| Mesh Solid Volume | Real; watertight required; no shell fallback | Frozen seam/direct acceptance; Phase 3 translator/UI |
| Mesh Surface Shell | Real; explicit half-width and voxel-floor policy | Existing shell translator; compiled backend accepted |
| Direct points | Real; all owned attributes preserved | Existing compatibility operator |
| Rasterized point top-up | Real | Frozen seam/direct acceptance; Phase 3 translator/UI |
| Volume fog/SDF/scalar/vector/components | Real via Blender-bundled `openvdb` staging | Frozen seam/direct acceptance; Phase 3 translator/UI |
| Height Flat | Real | Existing translator/UI |
| Height Profile | Real only with explicit profile input | Hidden until Phase 3 supplies an input control |
| Height Infinite | Missing primary-source semantics | Rejected and hidden |
| Sink/collision/activation | Real | Existing role paths where already translated |
| Pressure FIELD | Real; post-solve/pre-projection; emits no liquid | Hidden until Phase 3 schema |
| Scalar/vector/staggered velocity | Real; post-P2G, face-specific MAC sampling | Hidden until Phase 3 schema |
| FORCE | Real through accepted legacy analytic post-P2G path | Existing FORCE role; compiled FORCE plans reject actionably |

## Identity and lifecycle

- New runtime key: `session:<generation>:object:<pointer>`.
- Rename preserves identity; duplicate objects have distinct keys.
- Initialize/Reset/load creates a new generation and re-arms Once/Initial Fill.
- Disable/re-enable, rename, movement, and reevaluation do not re-arm a consumed Once source.
- Save/reopen starts a fresh runtime session; no persistent UUID or saved-data migration was added.
- Existing RNA identifiers/defaults and Phase 1 compatibility descriptors remain valid.

## Error/warning catalog for Phase 3

Errors are structured as severity/code/message/stable ID and prevent only the affected source from applying.

- `MESH_EMPTY`, `MESH_NOT_WATERTIGHT`, `SOURCE_COMPILE_FAILED`.
- `SOURCE_LEAF_CAPACITY`, `SOURCE_MEMORY_CAPACITY`.
- `SOURCE_DYNAMIC_MESH_BUDGET`, `SOURCE_MESH_CUDA_REQUIRED`.
- `COMPILED_FORCE_UNSUPPORTED` (use accepted legacy analytic FORCE path).
- Missing/unreadable named Volume grids and non-linear VDB transforms are actionable compile errors.
- `TARGET_OCCUPANCY_CLAMPED` warns above the GPU limit of 64 particles/voxel.
- `SHELL_WIDTH_VOXEL_FLOOR` reports when effective shell half-width becomes half a voxel.
- Runtime application reports emitted count and `capacity_limited`; partial fills are never presented as complete.

## Quantitative Blender acceptance

Runtime identity for all reports: Blender 5.1.0 `adfe2921d5f3`, add-on `fluidSim_warp_blender51` version `[1,0,241]`, CUDA device NVIDIA GeForce RTX 4080 SUPER. Reproducible `.blend` fixtures are under `artifacts/phase2_fixtures/`.

```text
blender --background --factory-startup --python tests/blender_source_region_acceptance.py -- --json-out artifacts/source_region_acceptance.json
SOURCE-REGION-ACCEPTANCE-PASS

blender --background --factory-startup --python tests/blender_point_source_acceptance.py -- --json-out artifacts/point_source_acceptance.json
POINT-SOURCE-ACCEPTANCE-PASS

blender --background --factory-startup --python tests/blender_source_lifecycle_acceptance.py -- --json-out artifacts/source_lifecycle_acceptance.json
SOURCE-LIFECYCLE-ACCEPTANCE-PASS

blender --background --factory-startup --python tests/blender_animated_source_acceptance.py -- --json-out artifacts/animated_source_acceptance.json
ANIMATED-SOURCE-ACCEPTANCE-PASS

blender --background --factory-startup --python tests/blender_volume_source_acceptance.py -- --json-out artifacts/volume_source_acceptance.json
VOLUME-SOURCE-ACCEPTANCE-PASS
```

Evidence includes:

- Mesh solid coverage equals analytic box coverage; all solid samples are inside.
- Open solid is rejected while explicit shell succeeds; measured shell band reaches the requested width within half-voxel sampling and never exceeds it.
- Direct point velocity, temperature, viscosity, density, and lifespan arrays transfer exactly.
- Once lifecycle emits 512/0 on first/second application; one- versus four-substep compiled fills both produce 512.
- Rename stable, duplicate distinct, Reset re-emits 512, reopen re-emits 512.
- Animated source recompiles spatially at the moved position and transfers object velocity.
- Pressure produces zero particles and a projected velocity delta of 6.0 over a zero-gravity no-pressure control.
- Staggered velocity produces independent MAC component maxima `[1,2,3]`.
- Phase 2V uses the installed Blender `openvdb` binding and validates fog, SDF, scalar, vector, component, staggered, authored-grid transform, plan transform, and missing-grid rejection.
- Each lane validates capacity preflight and a bounded 32-particle partial runtime fill.

## Focused and production validation

Final focused command covers the required core suite plus all new contract/compiler/adapter/operation/report/review-regression tests:

```text
115 passed in 4.66s
```

Artifact: `artifacts/phase2_final_focused_tests.log`.

Approved Balanced profile:

```text
blender --background --factory-startup --python tools/profile_source_pipeline.py -- --frames 120 --json-out artifacts/source_pipeline_profile.json
SOURCE-PIPELINE-PROFILE-PASS
```

- All profile cases compile without hidden error diagnostics.
- No-source median/p95 regression: -1.96% / -1.20%.
- Analytic median/p95 regression: +1.19% / +1.98%.
- Animated shell: 4.40 ms median / 6.62 ms p95 in the accepted fixture.
- Near-budget animated mesh (36,864 comparisons): 11.82 ms / 13.03 ms.
- Point rasterization: 1.14 ms / 1.56 ms.
- Volume grid sampling: 2.29 ms / 3.36 ms, nonzero retained compiled bytes, no errors.
- 50,000-triangle static mesh at 0.02 m: CUDA BVH compile 224 ms, nonzero coverage, no diagnostics.
- One cache generation retained; reset returns to zero source cache bytes.
- Every approved 10%/15%, 16.7/33.3 ms, 512 MiB, and 8 MiB check is true.

Production subset:

```text
FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py
USER-PATH-PROFILE-PASS
HEADLESS-GATE-PASS
```

Artifact: `artifacts/phase2_final_production_gate.log`. GUI Phase 3 production acceptance was intentionally not requested because Phase 2 adds no final UI exposure.

## Independent reviewer disposition

- Initial R1/R3 review rejected unbounded dynamic cache retention and unrepresentative mesh scaling. Both were repaired with one-generation eviction, cumulative live-cache profiling, CUDA BVH static staging, and a measured dynamic work cap.
- Initial R2/R4 review rejected incomplete quantitative shell/capacity/motion/substep/report evidence. All lanes were strengthened and rerun.
- Consolidated fresh R1–R4 review `360eb6e7` verified the substantive H1 and M1–M3 repairs. Its only final high was a stale test expecting the removed `SOURCE_MESH_WORK_LIMIT` code. The assertion now expects `SOURCE_DYNAMIC_MESH_BUDGET`; the complete suite passes 115 tests.
- The same final pass added CPU-versus-CUDA-BVH shell coverage/SDF parity, resolving the review's remaining correctness seam.
- No blocker/high finding remains after parent disposition.

## Changed files by Phase 2 ownership

### Backend
- `fluidSim_warp_blender51/sim/source_regions.py` (new)
- `fluidSim_warp_blender51/sim/source_compat.py` (new)
- `fluidSim_warp_blender51/sim/sourcing.py`
- `fluidSim_warp_blender51/sim/solver.py`
- `fluidSim_warp_blender51/sim/pressure.py`
- `fluidSim_warp_blender51/sim/enums.py`
- `fluidSim_warp_blender51/blender/scene_inputs.py`

### Contracts and tools
- `docs/source_system_refactor/PHASE_2_SOURCEPLAN_CONTRACT.md` (new)
- `tools/profile_source_pipeline.py` (new)
- `tools/validate_source_acceptance_report.py` (new)

### Tests/runtime lanes
- `tests/test_source_plan_contract.py`
- `tests/test_source_region_compilation.py`
- `tests/test_source_compatibility.py`
- `tests/test_source_identity.py`
- `tests/test_point_source_pipeline.py`
- `tests/test_height_source_pipeline.py`
- `tests/test_volume_source_pipeline.py`
- `tests/test_compiled_source_operations.py`
- `tests/test_source_acceptance_report.py`
- `tests/test_phase2_review_repairs.py`
- `tests/_phase2_source_acceptance_common.py`
- five `tests/blender_*_source_acceptance.py` capability scripts

The repository was heavily dirty before Phase 2. This list identifies Phase 2-owned paths/hunks only; aggregate diff statistics are not attributed wholesale. No reset, clean, stash, stage, or unrelated rewrite occurred.

## Residual limitations and rollback

- Every Sim Frame compiled meshes are intentionally low-complexity; larger inputs must use First Sim Frame or lower source resolution. Static/First Sim Frame production-scale meshes use CUDA BVH.
- Height Infinite remains hidden/rejected; Profile remains hidden until a real input control exists.
- Point-rasterized and Volume construction remain seam/direct-test reachable until Phase 3 adds the reviewed Blender translator/UI.
- Memory measurements are flat source staging/cache payload-byte proxies.
- Rollback removes one translator/capability gate and its compiler path while preserving legacy descriptors and saved RNA. No `.blend` migration is required.

## Phase 3 prerequisites and schema mapping

Phase 3 may expose only the Real rows above and must:

1. Map operation, region, lifecycle, cadence, payload, and blend independently.
2. Keep FIELD non-liquid and show pressure/scalar/vector/staggered controls only for their accepted operation.
3. Show Solid Volume only with watertight validation; never offer shell fallback.
4. Label direct points separately from rasterized top-up.
5. Add Volume filepath/grid selectors that surface named-grid and transform errors.
6. Add Profile height only with an explicit profile input; keep Infinite absent.
7. Present dynamic mesh budget, capacity, occupancy clamp, and shell-width diagnostics adjacent to the owning source.
8. Preserve existing RNA identifiers/defaults and gate new translator paths with Blender acceptance.

Implementation prompt:

> Implement Phase 3 from `docs/source_system_refactor/PHASE_3_UI_REFACTOR.md` using the Real capability rows and frozen SourcePlan contract in this handoff. Preserve all compatibility RNA/defaults, keep Height Infinite absent, keep FIELD non-liquid, and expose point/Volume/Mesh Solid/advanced-field translators only with their focused UI and Blender gates. Do not weaken the dynamic-mesh, capacity, transform, or error-reporting contracts.

## Worktree state

`git diff --cached --name-only` is empty: **no files staged**. The final handoff intentionally leaves the user's pre-existing dirty worktree intact.
