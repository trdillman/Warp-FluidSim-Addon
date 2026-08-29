# FluidSim Phase 4 Handoff — Production Whitewater Outputs

## Release state

- Branch: `phase4/production-whitewater`
- Version: `1.0.249`
- Base: Phase 3 commit `5d26a9a7c12682edf63e2db11159fc757785ccfc`
- Supplied Blender: 5.2.0 LTS, Linux x64
- Supplied Warp: 1.16.0+cu12, module version 1.16.0
- Executed Warp device: CPU
- CUDA certification: open; no usable NVIDIA driver/device was exposed

## Delivered

Phase 4 converts the solver's prior aggregate whitewater behavior into one
persistent typed particle pool. Stable values are foam `0`, bubble `1`, and
spray `2`; value `3` is reserved and rejected. Initial classification uses
liquid SDF depth, speed, and churn. Per-step transitions use separate enter and
exit thresholds so particles do not flicker between types at one boundary.

The single pool now carries aligned int32 type state through emission,
advection, compaction, host capture, cache-v2 frames, checkpoints, resume, and
Blender publication. Foam, bubbles, and spray have independent lifespan and
motion controls while sharing the existing whitewater capacity and admission
policy. Type counts are reduced on the device and reported with frame/runtime
statistics.

Blender exposes three publication policies:

- `Combined` — one backward-compatible `FluidSim Whitewater` object.
- `Separate` — `FluidSim Foam`, `FluidSim Bubbles`, and `FluidSim Spray`.
- `Combined + Separate` — both views for independent art direction.

Separate objects are filtered host/publication views; they are not additional
simulation pools. The bundled `fluidsim-render-p4-v1` asset contains six owned
Geometry Nodes groups for surface, liquid particles, combined whitewater, foam,
bubbles, and spray. Installation remains idempotent and does not overwrite
user-owned modifiers or node groups.

## Compatibility contract

- Cache-v1 and Phase-3 cache-v2 frames remain playable.
- Whitewater without recorded type state is published as legacy-unclassified
  `-1` on the combined output only. It is never silently relabelled as foam.
- Newly written Phase-4 cache-v2 frames require one valid int32 type per
  whitewater point and attribute schema version 2.
- Phase-3 checkpoints are intentionally not resumable in Phase 4. Deterministic
  type/transition state did not exist and is not reconstructed heuristically.
- Phase-4 checkpoints require `whitewater_type` and
  `whitewater_type_schema_version = 1` before any solver mutation.

## Validation commands

```bash
python -m pytest -q
python -m compileall -q -f .
python ../tools/release/build_extension.py --verify-only --root .

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
python ../tools/release/validate_phase4_warp.py \
  --expected-warp-root /mnt/data/warp_lang_1_16_external \
  --module fluidsim_phase4_repo \
  --device cpu \
  --report-json /mnt/data/phase4_warp_validation.json

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
FLUIDSIM_EXTERNAL_WARP_ROOT=/mnt/data/warp_lang_1_16_external \
/mnt/data/blender-5.2.0-linux-x64/blender \
  --background --factory-startup --offline-mode --python-use-system-env \
  --python-exit-code 1 --python ../tools/release/validate_phase4_blender.py -- \
  --module fluidsim_phase4_repo \
  --work-dir /mnt/data/phase4_blender_validation_work \
  --report-json /mnt/data/phase4_blender_validation.json
```

The NVIDIA release workflow now invokes the same Warp validator with
`--device cuda:0` before the existing installed-package GPU regression suite.
That gate was defined but could not be executed on the CPU-only validation host.

## Verified in the available environment

- Pure-Python suite: 176 passed.
- Python compilation and source verification: passed.
- Supplied Warp CPU kernels: classification parity, typed admission,
  type-specific lifetimes, type counts, aligned compaction, checkpoint restore,
  reserved-value rejection, and one-pool invariant passed.
- Supplied Blender: all six output shells, P4 Geometry Nodes, typed cache
  publication, legacy-unclassified preservation, user-modifier protection,
  save/reopen, and Eevee F12 three-material rendering passed.
- Phase-3 Blender and Warp regression validators passed against the Phase-4
  source.
- Four generated example scenes reopened and validated with the P4 output
  structure.
- Clean-profile package install, enable/disable/re-enable, two-process smoke,
  external Warp execution, and uninstall passed.

## Remaining gates

Run the mandatory NVIDIA workflow with the supplied Warp package on compatible
hardware. Open checks include CUDA full-solver numerics, uninterrupted versus
resumed deterministic type transitions, GPU growth/compaction under production
loads, VRAM admission/out-of-memory behavior, performance, and renderer-specific
GPU regressions. Windows x64 and Cycles-specific validation also remain open.
Phase 5 surface-tension physics has not been started.
