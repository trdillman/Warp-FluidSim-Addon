# Frozen Phase 2 SourcePlan contract

## Interface decision (designed twice)

### Candidate A — descriptor union (rejected)

`SourcePlan(kind, SourceShapeDesc | MeshSourceDesc | PointDesc | VolumeDesc)`
would leave geometry dispatch, Warp mesh/volume lifetimes, bounds, validation and
cache invalidation in `SourcePipeline`. It fails the deletion test: removing a
nominal compiler would merely reveal the same adapter-specific branches in the
pipeline and solver.

### Candidate B — immutable plan and compiled region (selected)

```text
Blender adapter / compatibility adapter
  -> SourcePlan(stable_id, operation, lifecycle, cadence,
                region, payload, activation, blend)
  -> SourceCompiler.compile(plans, CompileContext)
  -> CompiledSourceBatch[CompiledRegion]
  -> SourcePipeline.apply_compiled(batch, dt)
```

Candidate B is deeper and has greater locality. `sim/source_regions.py` alone
owns geometry validation, candidate-leaf enumeration, deterministic seeds,
OpenVDB staging, flat-leaf payload construction, cache keys, capacity checks,
device upload and release. The pipeline owns operation order and lifecycle but
never reacquires geometry. Blender adapters copy arrays before evaluated-mesh
release. Deleting the compiler would force substantial mesh containment,
OpenVDB, sparse allocation, cache and diagnostics logic into multiple callers.

## Frozen ownership and storage

- A `SourcePlan` is frozen and contains only enums, strings, numbers, tuples and
  owned read-only contiguous NumPy arrays.
- Blender objects, RNA/PropertyGroup values, mathutils proxies, evaluated meshes,
  `wp.Mesh`, OpenVDB grids and accessors cannot enter persistent plan state.
- `CompiledRegion` uses `leaf_coords[N,3]` plus slot-major payload arrays
  `[N,512]` (or `[N,512,3]`). Leaves are fixed 8x8x8 and addressed through the
  solver's existing flat leaf table. There is no live hierarchical source grid
  and no whole-domain dense source mask.
- OpenVDB is Blender-bundled staging. Named grids are read with their authored
  index-to-world transforms, composed with the object/plan transform, sampled
  into source leaves, and all trees/accessors are dropped before `compile()` returns.
- Compiler cache entries exclusively own device uploads. At most one generation
  per stable source is retained for every cadence; replacement releases the old
  upload. `release()` runs on reset, teardown and failure recovery.

## Independent axes

- Operation: Liquid, Sink, Force, Collision, Activation, Pressure, Scalar Field,
  Vector Field, Staggered Velocity.
- Region: Analytic SDF, Mesh Solid, Mesh Shell, Points, Volume Grid, Height.
- Lifecycle: Once, Top Up, Direct Append, Every Frame.
- Cadence: Static, First Sim Frame, Every Sim Frame.
- Payload: None, Liquid Attributes, Pressure, Scalar, Vector, Staggered Vector,
  Mask.

Invalid combinations fail plan construction. Field payloads cannot be labeled
liquid. Direct Append is liquid points only. Pressure/scalar/vector operations
require their corresponding payloads.

## Capability matrix

| Capability | Compiler | Operation/runtime | Translation gate |
|---|---|---|---|
| Analytic solid | Real | Legacy + compiled | Legacy accepted |
| Mesh Solid Volume | Real; watertight required; production-scale static meshes use Warp CUDA BVH | Compiled GPU coverage/top-up | Programmatic/acceptance until Phase 3 UI |
| Mesh Surface Shell | Real; explicit width; Every Sim Frame compiled meshes are intentionally low-complexity and rejected above the measured 40k-comparison interactive budget | Legacy + compiled; CUDA BVH is Static/First Sim Frame only | Legacy accepted |
| Direct points | Real, owned velocity/temperature/viscosity/density/lifespan arrays | Compiled direct append preserves every array | Compatibility operator accepted |
| Rasterized points | Real | Compiled top-up | Frozen seam/direct acceptance only until Phase 3 adds construction/translation |
| Volume fog/SDF/scalar/vector | Real via bundled OpenVDB staging | Compiled masks/fields | Frozen seam/direct acceptance only until Phase 3 adds construction/translation |
| Height Flat/Profile | Real; Profile requires input | Compiled/legacy | Profile hidden without input |
| Height Infinite | Missing primary-source semantics; rejected | None | Hidden; plan construction fails truthfully |
| Sink/collision/activation | Real | collision > sink > liquid precedence | Role translation accepted where existing |
| Pressure | Real compiled operation; collision/sink skip | Blended after pressure solve and before projection; FIELD emits no liquid | FIELD stays hidden until Phase 3 |
| Scalar/vector | Real compiled operation | Applied after P2G so transfer cannot overwrite it | Hidden until Phase 3 schema |
| Staggered velocity | Real compiled operation | Imported vector grids sampled independently at x/y/z face positions and applied to MAC components after P2G | Hidden until Phase 3 schema |
| Force | Analytic region | Existing post-P2G force stage | FORCE role accepted |

Point direct append is never reported as volume coverage. Invalid solid input is
never converted to a shell.

## Identity and lifecycle

New plans use `session:<generation>:object:<pointer>`. Rename preserves the key;
duplicates have distinct pointers; Initialize/Reset/load increments the session
generation. Legacy `source_id=blender-object:<pointer>` remains readable for
Phase 1 evidence and saved-scene compatibility. Once latches re-arm only on
pipeline reset.

## Diagnostics

Structured diagnostics contain severity, code, message and stable source ID.
Current codes include `MESH_EMPTY`, `MESH_NOT_WATERTIGHT`,
`SOURCE_COMPILE_FAILED`, `SOURCE_LEAF_CAPACITY`, and
`SOURCE_MEMORY_CAPACITY`. Errors prevent the affected source from applying;
there is no fallback. Capacity-limited particle application reports emitted
count and `capacity_limited`.

## Cache and performance contract

Cache keys include stable ID, independent plan axes, owned geometry/payload,
domain transform, voxel size and session. Every-frame cadence additionally
includes frame/substep; static and first-frame plans reuse device allocations.
Superseded static or dynamic generations are evicted and released. Dynamic mesh
CPU work is capped at 40,000 voxel/triangle comparisons, derived from a measured
near-limit case against the already-approved interactive limits; larger animated
meshes must use First Sim Frame or lower source resolution. Production-scale
static meshes use Warp's CUDA mesh BVH and remain supported. Memory gates report
flat source staging/cache payload bytes (a conservative host-array proxy), not a
vendor allocator's whole-device telemetry. Approved Balanced gates are fixed
before implementation:

- existing no-source/analytic median regression <= 10%, p95 <= 15%;
- static sources do not rebuild or allocate after warmup;
- dynamic compile+apply median <= 16.7 ms, p95 <= 33.3 ms;
- peak staging <= 512 MiB;
- reset returns within 8 MiB of baseline.
