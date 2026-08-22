# FluidSim Custom Effects v2 Design

**Status:** Approved for implementation by the user on 2026-08-17  
**Baseline:** FluidSim Warp 1.0.250, commit `33562fae3629b03f567131e6d08312134b374e30`  
**Target release:** 1.0.251  
**Validation runtimes:** supplied Blender 5.2.0 LTS and supplied NVIDIA Warp 1.16.0+cu12

## Product objective

Replace the prototype-facing **Custom Warp Snippets** feature with a shippable **Custom Effects** workflow while retaining a developer-grade **Warp Code** workspace. Artists should add and control intent-driven effects without handling internal field arrays or kernel hooks. Technical artists should receive a versioned API, real Warp compilation, sandbox execution, source-mapped diagnostics, strict permissions, packaging, profiling, and deterministic cache identity.

## User stories

### Artist

As a liquid artist, I can add an effect such as a vortex, wave, turbulence, pulse, drain, damping field, or temperature lift; position it with a controller; tune meaningful parameters; test and apply it; preview it; and bake or resume it reproducibly without understanding solver internals.

### Technical artist

As a technical artist, I can author effect code in Blender's Text Editor, select legal fields and access modes from an authoritative registry, compile on the selected Warp backend, run a bounded sandbox test, inspect generated source and source-mapped diagnostics, profile execution, and package the exact tested revision.

### Pipeline

As a pipeline lead, I can enforce trusted-code policy, inspect effect fingerprints and inventories, reject incompatible resume, run deterministic headless policy, and reproduce validation with the supplied Blender and Warp packages.

## Public workflow

The default feature is **Custom Effects**:

1. Add a built-in or installed effect from an intent-based library.
2. Adjust typed parameters and, when applicable, position a controller object.
3. Use **Compile & Apply**. The currently applied stack remains active unless the entire draft stack passes validation, real Warp compilation, sandbox launch, and trust checks.
4. Preview one frame or use the normal live/bake workflow.

The advanced **Warp Code** panel exposes source, domain, stage, cadence, field access, generated source, diagnostics, trust, and profiling. Raw implementation controls are hidden from Artist mode.

## Effect model

`EffectDefinitionV2` is immutable content metadata:

- stable effect ID and semantic version;
- name, summary, description, category, author, license;
- minimum effect API and FluidSim version;
- trust origin and bundled digest;
- typed parameter definitions;
- optional controller and scope definitions;
- one or more kernel definitions;
- cost and capability metadata.

`EffectInstanceV2` is scene-authored state:

- instance UUID and definition snapshot;
- enabled, mute, solo, and stack order;
- typed parameter values;
- controller and scope bindings;
- frame range and cadence;
- draft, tested, and applied fingerprints;
- trust status and diagnostic state.

Legacy snippets migrate to one-kernel legacy effect instances. Their original source and Text datablocks are preserved. Unknown embedded code is disabled and untrusted until the user trusts the exact content digest.

## Typed parameters

Supported parameter kinds:

- `FLOAT`, `INT`, `BOOL`;
- `VECTOR`, `COLOR`;
- `ENUM`;
- `OBJECT` controller references.

Definitions include units, hard/soft ranges, precision, descriptions, animation policy, and whether a value is runtime-bound or compile-structural. Built-ins use runtime-bound scalar/vector values where possible so tuning does not trigger unnecessary recompilation.

## Kernel model

Each kernel declares:

- domain: `PARTICLES` or `ACTIVE_VOXELS`;
- stage: `BEFORE_EMISSION`, `AFTER_PARTICLE_TRANSFER`, `BEFORE_PRESSURE`, or `AFTER_PARTICLE_UPDATE`;
- cadence: `EVERY_SUBSTEP`, `ONCE_PER_FRAME`, or `INITIALIZATION`;
- source Text datablock or embedded source;
- typed field bindings and access capabilities;
- parameter bindings;
- prerequisite capabilities such as topology, SDF, collision data, or particles.

`VOXEL_TILE` is not a public v2 mode. Legacy instances migrate to `ACTIVE_VOXELS` and receive a migration notice.

## Field and access safety

One authoritative field registry defines name, Warp dtype, logical domain, units, legal stages, persistence, structural impact, runtime accessor, and allowed access modes.

Access modes are:

- `READ`;
- `WRITE_CURRENT`;
- `READ_WRITE_CURRENT`;
- `ATOMIC_SCATTER`;
- `UNSAFE_ARBITRARY`.

Safe mode permits only legal domain/stage combinations and current-element writes. Expert capabilities require explicit acknowledgement and remain visible in reports. Invalid combinations fail before source generation.

## Validation and compilation

Testing is a staged pipeline:

1. Strict schema and identifier validation.
2. AST safety and field-access validation.
3. Deterministic generated source plus line map.
4. Python import.
5. Forced Warp JIT compilation with `wp.load_module(device=...)`.
6. Bounded sandbox launch with canary buffers and finite-value checks.
7. Device synchronization.
8. Structured diagnostics mapped to effect/kernel/source lines.

Statuses are `UNTRUSTED`, `DRAFT`, `INVALID`, `TESTED`, `APPLIED`, `DIRTY`, and `RUNTIME_ERROR`. Import success alone is never called compilation success.

## Runtime and scheduling

A candidate runtime manager is built from the complete enabled draft stack. Every enabled effect must be trusted, valid, compiled, and sandbox-tested. Only then is the candidate atomically swapped into the solver. Failure preserves the prior applied manager and fingerprint.

Scheduling is capability-derived. A particle-only effect does not force sparse-topology rebuilds. SDF, collision, or topology prerequisites are prepared only for effects that declare them. Cadence is enforced independently of frame windows.

Runtime diagnostics track per effect/kernel:

- compile and sandbox duration;
- calls and elements per frame;
- submission duration;
- required prerequisites;
- last runtime error.

## Fingerprints and persistence

Canonical JSON serialization and SHA-256 cover every behavior-affecting value:

- effect API/ABI and package identity;
- enabled/mute/solo/order;
- source and kernel structure;
- domains, stages, cadence, and frame windows;
- field names and access modes;
- parameter definitions and values;
- controller/scope bindings and transforms;
- trust-relevant content digest.

The aggregate effect-stack signature and inventory are included in scene/settings signatures, cache-v2 manifest metadata, checkpoints, production-bake reports, headless reports, and support reports. A mismatch rejects resume before solver mutation with `EFFECT_SIGNATURE_MISMATCH`. Older caches without effect identity remain playable, but cannot resume when custom effects are active.

## Trust and headless policy

Trust attaches to exact SHA-256 content, stored locally outside scene-authored claims. States are `BUILT_IN_TRUSTED`, `LOCAL_TRUSTED`, `UNTRUSTED_IMPORTED`, and `BLOCKED`. Editing content invalidates local trust.

Headless policy is explicit:

- `deny`;
- `builtins-only`;
- `trusted`.

The default is `builtins-only`. Policy failures are machine-readable and do not execute code.

## Effect Package v2

Packages use a strict ZIP/JSON format containing:

- `effect.json` manifest;
- source files;
- optional preview image and demo metadata;
- content digests;
- author/license/origin metadata;
- compatibility declarations.

Import, export, install, duplicate, rename, delete, reveal, conflict resolution, and update inspection are supported. Package-provided trust claims are ignored. Writes are atomic and fsync-backed. Malformed packages fail closed with actionable diagnostics.

## Blender UI

### Custom Effects panel

- `UIList` with select, reorder, mute, solo, duplicate, remove, status, and cost indicator;
- searchable Add Effect browser grouped by intent;
- typed parameter controls and controller creation/select actions;
- Draft/Tested/Applied/Dirty/Error/Untrusted state;
- Compile & Apply, Revert Draft, and Preview One Frame;
- concise effect help and performance prerequisites.

### Warp Code panel

- Text datablock selector and Open in Text Editor;
- domain, stage, cadence, field/access declarations;
- Safe/Expert mode;
- static, compile, sandbox, and runtime diagnostics;
- Go to Error Line;
- API browser generated from the field registry;
- generated-source view;
- target runtime/device evidence and profiling.

Legacy operator IDs forward to v2 operations for one compatibility release.

## Built-in effects and demos

The bundled library includes at minimum:

- Vortex Force with controller, strength, radius, falloff, suction, and lift;
- Curl Turbulence;
- Radial Pulse;
- Planar Split;
- Velocity Damping;
- Temperature Lift;
- Particle Speed Limit;
- Voxel Cooling.

Real supplied-Warp CPU demonstrations validate deterministic particle and voxel outcomes for representative stacks. Supplied-Blender demos include configured effect stacks, controllers, Text datablocks, visualized before/after point states, validation reports, and F12 evidence renders. CUDA-specific full-solver behavior remains an explicit hardware gate when no compatible NVIDIA device is exposed.

## Error handling

- Enabled invalid or untrusted effects block initialization and production bake; they are never silently omitted.
- Failed Test or Apply preserves the last applied stack.
- Runtime failures are recorded at a safe boundary and surfaced with effect/kernel/source context.
- Diagnostics are structured and not truncated to a single UI string.
- Imported package, migration, cache, checkpoint, and trust errors fail closed.

## Testing and release gates

### Pure Python

Schema, canonical serialization, fingerprint mutation coverage, identifier validation, field/domain/stage/access matrix, AST capability checks, package migration, atomic persistence, trust transitions, candidate swap, cache/checkpoint signature handling, and headless policy.

### Supplied Warp

Every built-in and negative fixture undergoes real static validation, `wp.load_module`, bounded CPU sandbox launch, canary/finite checks, profiling capture, and deterministic output checks. CUDA compile/launch remains required on compatible supplied-package hardware.

### Supplied Blender

Artist workflow, controllers, stack operations, draft/applied state, Text Editor integration, source-mapped diagnostics, trust, save/reopen, cache signature mutation, resume rejection, package import/export, headless policy, lifecycle, demo generation, and F12 render evidence.

## Completion criteria

The redesign is complete when invalid code cannot report compile success, invalid or untrusted enabled effects cannot run or be silently skipped, Apply is atomic, cache identity includes the exact effect stack, Artist mode hides internal solver plumbing, Warp Code uses Blender's Text Editor and source-mapped diagnostics, package/trust/headless behavior is strict, representative demos are generated and analyzed, and all available supplied-package gates pass. CUDA-only claims remain unresolved unless executed on compatible NVIDIA hardware.
