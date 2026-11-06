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
