# Custom Effects and Warp Code API

## Versioned contracts

Release 1.0.251 defines:

```text
Effect schema version: 2
Effect API version:    2
Effect ABI version:    1
Package schema:        2
```

An `EffectDefinitionV2` owns metadata, typed parameters, and one or more `KernelDefinitionV2` records. An `EffectInstanceV2` owns scene execution state: stable instance ID, order, enable/mute/solo state, typed values, controller snapshots, frame window, safe/expert acknowledgement, and display name.

Unknown fields, unsupported versions, invalid identifiers, duplicate parameter/kernel/instance IDs, non-finite numbers, and out-of-range values are rejected.

## Domains

| Domain | Current index | Base context |
|---|---|---|
| `PARTICLES` | `particle_index` | `position`, `velocity`, particle arrays; dead entries are skipped by the generated wrapper. |
| `ACTIVE_VOXELS` | `flat_voxel_index` | leaf/local/global coordinates, `world_position`, voxel type, and active sparse topology. |

The former public `VOXEL_TILE` mode is migrated to `ACTIVE_VOXELS` with a warning. It did not provide distinct tiled semantics.

## Public stages

| Stage | Solver boundary |
|---|---|
| `BEFORE_EMISSION` | Before source emission for the substep. |
| `AFTER_PARTICLE_TRANSFER` | After particle-to-grid transfer, before the main force sequence. |
| `BEFORE_PRESSURE` | After forces/viscosity/collision preparation, before pressure projection. |
| `AFTER_PARTICLE_UPDATE` | After grid-to-particle transfer and particle update. |

## Cadence

- `EVERY_SUBSTEP`
- `ONCE_PER_FRAME`
- `INITIALIZATION`

Instance and kernel frame windows support start, end, and step. Cadence is independent from stage.

## Field registry

The API browser is generated from `sim/effects/fields.py`. Each field declares dtype, logical domain, units, legal stages, access modes, runtime accessor, current index, prerequisite set, persistence, structural status, and description.

Particle fields:

```text
particle_position       vec3  m
particle_velocity       vec3  m/s
particle_age            float s
particle_life           float s
particle_temperature    float scene temperature
particle_viscosity      float m²/s
particle_density        float kg/m³
particle_time_offset    float s
```

Active-voxel fields:

```text
velocity_u/v/w          float m/s
weight_u/v/w            float weight
temperature             float scene temperature
viscosity               float m²/s
density                 float kg/m³
pressure                float solver pressure
divergence              float 1/s
surface_sdf              float m, read-only, requires current surface SDF
collision_sdf            float m, read-only, requires collision field
churn                    float 1/s
voxel_type               int bitmask, structural read-only
voxel_counts             int particles, structural read-only
```

Safe mode permits `READ`, `WRITE_CURRENT`, and `READ_WRITE_CURRENT` only where the registry allows them. `ATOMIC_SCATTER` and `UNSAFE_ARBITRARY` require Expert mode and explicit acknowledgement. Structural fields remain read-only.

## Source environment

Generated wrappers always import Warp as `wp` and provide:

```text
frame
time
real_time_step
solver_time_step
time_scale
substeps
voxel_size
```

| Name | Meaning |
|---|---|
| `frame` | Timeline frame being simulated (the frame windows use the same clock). |
| `time` | Simulated seconds since the simulation started. |
| `real_time_step` | Simulated seconds per display frame: `time_scale / fps`. This is how far `time` advances each frame, so at Time Scale 2 and 24 fps it is 2/24, not 1/24. Use it for once-per-frame effects. |
| `solver_time_step` | Simulated seconds per solver substep: `real_time_step / substeps`. Use it for every-substep effects. |
| `time_scale` | The domain's Time Scale. Divide by it to get wall-clock (playback) seconds. |
| `substeps` | Substeps taken this frame. |
| `voxel_size` | Simulation voxel size in metres. |

Both time steps are in simulated time, so a rate in 1/s (such as Voxel Cooling's) stays in step with the liquid at any Time Scale.

Particle code additionally receives `particle_index`, `position`, and `velocity`. Voxel code receives sparse-coordinate context and `world_position`. Bound fields and typed parameters are explicit launch inputs rather than arbitrary global lookup.

The static validator rejects imports, unsupported constructs, unknown names, undeclared field/parameter use, domain mismatches, illegal writes, unbounded loops, and unsafe capabilities in Safe mode.

## Compilation and diagnostics

The pipeline is:

```text
strict snapshot
→ field/capability validation
→ AST validation
→ deterministic generated source and line map
→ Python module import
→ wp.load_module(device=selected_device)
→ bounded launch
→ device synchronization
→ canary and finite checks
→ complete candidate
→ atomic runtime swap
```

Diagnostics include effect ID, instance ID, kernel ID, code, severity, user source line/column, and a corrective hint. Instance IDs prevent two instances of one definition from sharing the wrong UI error.

Runtime launch errors raise `EffectExecutionError`, latch the manager failed, stop later execution, and remain blocked until a successful candidate swap. The live solver does not silently continue with a partial effect stack.

## Capability-derived scheduling

Prerequisites are derived from fields and kernel declarations:

```text
PARTICLES
TOPOLOGY
SURFACE_SDF
COLLISION
```

A particle-velocity-only effect does not force sparse topology or SDF rebuilding. Active-voxel effects require topology. SDF and collision fields request their corresponding current solver data before the stage.

## Fingerprints

`execution_stack_fingerprint()` filters the scene through enabled/mute/solo semantics, rejects duplicate instance IDs, sorts by authored order plus stable input position, and returns a SHA-256 plus ordered inventory. Disabled or muted definitions do not affect execution identity; any effective source, value, controller, order, stage, cadence, frame, binding, capability, API, ABI, or package change does.

## Package limits

The reader rejects archives exceeding:

```text
32 members
1 MiB per uncompressed member
8 MiB total uncompressed data
200:1 compression ratio per member
```

These limits are enforced from ZIP metadata before `archive.read()`.
