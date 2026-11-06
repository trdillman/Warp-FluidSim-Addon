# Frozen scene inputs and cache continuation

Production Bake now saves the host inputs captured by **First Sim Frame** sources
(mesh, points, volumes and fields) and **Static** colliders alongside each solver
checkpoint. Resume and Re-bake restore these snapshots before admission evaluates
the scene. Every Sim Frame inputs and moving-collider velocity history are still
primed at the last committed frame. Finishing, cancelling or failing a bake restores
the preview's original playhead and input snapshots before restoring its solver.
If preview restoration fails, the baking solver is released and the user must
Initialize; it is never left running as the previous preview.

## Storage and compatibility

Checkpoint format remains **2**. Its existing extensible maps carry:

- `scalars.blender_frozen_inputs`: schema version 1 and records keyed by persistent
  source/dependency identities, not process-local Blender pointers;
- `arrays.blender_frozen_*`: named, numeric NumPy arrays containing point positions
  and world-space mesh vertices/triangles.

Records also contain frozen analytic placement and VDB path/transform plus the
captured VDB content digest. A missing or modified VDB is refused; file names alone
are not accepted as evidence of the original data. Snapshot schema, dimensions,
mesh indices, array inventory, finite values and scene roster are validated before
replacement admission. The surrounding checkpoint's source/settings identity and
SHA-256 bind the extension to the committed transaction. Old format-2 readers
already preserve arbitrary scalar/array entries, so no extra top-level checkpoint
metadata key or pickle payload was introduced.

The cache-v3 manifest separately records optional `input_identity_schema_version`.
Missing means 0. New Production Bakes write revision 2, which covers the resolved
input dependencies. Resume/Re-bake refuses an older revision with
`INPUT_IDENTITY_SCHEMA_REQUIRED`; this protects against histories whose old
fingerprints omitted topology or referenced input data. Old caches remain readable
for playback/export in the updated add-on. Older add-on releases with strict
manifest readers may refuse newly stamped manifests; use the current add-on to
read them.

A revision-2 checkpoint without the frozen extension may continue only when its
scene requires no frozen snapshots. That is an explicit proof of an empty state.
Otherwise continuation reports `FROZEN_INPUT_STATE_REQUIRED` and requires a restart.
It does not invent a capture epoch: source disable/re-enable events can discard and
recapture snapshots, and old records do not prove which original epoch produced
the checkpoint. Invalid or mismatched snapshots report `FROZEN_INPUT_STATE_INVALID`.
These refusals do not remove committed frames or their export/playback access.

## Verification boundary

`tests/cache_io/test_frozen_input_state.py`,
`tests/cache_io/test_bake_integrity.py`, and
`tests/runtime/test_frozen_scene_inputs.py` cover serialization, validation before
mutation, restoration ordering, native-input siblings, shared mesh uploads and the
Every Sim Frame control. Existing solver, rigid and narrow-band checkpoint tests
remain required for numerical continuation.

`tests/acceptance/blender_frozen_input_resume_acceptance.py` runs in two separate
Blender processes (`--phase write`, then `--phase read`, with the same
`--state-root`). It uses actual evaluated Blender mesh/point/volume/field inputs,
Static colliders and moving-collider history through CPU staging and checkpoint I/O.
Its `FROZEN-INPUT-RESUME-HOST-PASS` marker explicitly does **not** certify a liquid
solve. Use private absolute Blender config/scripts/extensions/datafiles paths for
both invocations. Full CUDA Production Bake continuation and the deforming-collider
acceptance still belong to the certified release gate; the host acceptance does not
replace them or expand same-GPU/same-Warp determinism guarantees.

## Actual CUDA Production Bake continuation runner

Run the separate production path with an unused evidence directory:

```sh
python tools/gate/run_frozen_input_acceptance.py --blender /absolute/path/to/blender --device cuda:0 --state-root /absolute/path/to/new-evidence
```

The runner creates private absolute Blender profile directories and launches each
phase in a fresh background factory-startup process. It checks CUDA availability
before add-on registration or fixture creation and never substitutes a CPU solver.
Use the certified Blender/Warp runtime for qualification; Blender 5.2.1 is a
diagnostic runtime until separately certified. The runner inherits `PYTHONPATH`
for managed diagnostic environments; a certified installation should resolve its
own qualified dependencies.

`tests/acceptance/blender_frozen_production_acceptance.py` executes:

1. CUDA preflight, retaining actual runtime and source identities.
2. A complete six-frame Production Bake with animated FIRST_SIM_FRAME mesh and
   point sources, a frozen volume sink and analytic scalar field, Static and
   Every Sim Frame colliders. It saves the fixture after assigning persistent IDs.
3. A separate-process bake cancelled after frame 3, retaining its real cache and
   checkpoints with the original six-frame manifest range.
4. Fresh-process Resume: exact solver checkpoint, lifecycle and frozen-state
   import before stepping, plus a contrasting checkpoint-frame recapture. Later
   frozen inputs and moving-collider velocities must equal the uninterrupted run.
5. Another fresh-process Re-bake from frame 4, asserting the same invariants and
   byte-identical preservation of frames 1–3 and their checkpoints.
6. Actual preview-at-frame-5 restoration after finish, cancellation and an injected
   step failure, each with Live both off and on. Solver checkpoint and frozen
   snapshots restore exactly, and next-frame collider history matches the control.

All cache generation goes through `BakeRun` and `ProductionBakeBackend`; the test
subclass observes actual admission/commits and injects one named failure. Scene
input observation delegates to the real evaluator unchanged. Nonempty liquid is
required on every committed frame. The retained NPZ/cache files and per-array
numerical differences use their original order, exposing counts, shapes, dtypes,
exact equality, maximum absolute differences and RMS; no sorting masks drift.

**This runner currently cannot qualify a release.** There is no approved numerical
continuation tolerance for this mixed compiled-emitter fixture. It enforces exact
state-transfer contracts but records later fluid-array differences as diagnostics.
A complete structural run returns exit **2** with
`pending_compiled_emitter_numerical_policy`; unavailable CUDA or a failed invariant
returns exit **1**. It emits no release PASS marker. Establish and approve the
numerical policy on the certified GPU/runtime before making this a release gate;
do not loosen comparisons merely to obtain a pass. Existing rigid/narrow-band and
deforming-collider gates remain separate requirements.
