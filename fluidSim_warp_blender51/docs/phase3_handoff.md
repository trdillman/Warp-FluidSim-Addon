# FluidSim Phase 3 Handoff — Artist-Ready Rendering

## Release state

- Branch: `phase3/artist-ready-rendering`
- Version: `1.0.248`
- Base: Phase 2 commit `4e8272a80319394e7cd329bb7e1a3b6e6d7c16b6`
- Supplied Blender: 5.2.0 LTS, Linux x64
- Supplied Warp: 1.16.0+cu12, module version 1.16.0
- Executed Warp device: CPU
- CUDA certification: open; no usable NVIDIA driver/device was exposed

## Delivered

Phase 3 adds a single authoritative output-attribute schema, opt-in persistent
liquid color/source-ID/UID channels, derived speed/vorticity/motion data,
strict cache-v2 inventory validation, typed Blender attribute publication,
attribute-aware default materials, and a revisioned bundled Geometry Nodes
asset for surface, liquid-particle, and whitewater rendering.

Point render groups realize instances before material assignment. This is
required for point-domain named attributes to survive into shader-visible F12
geometry. Installation and explicit updates touch only FluidSim-owned modifiers;
user modifiers and user node groups are not overwritten.

## Validation commands

```bash
python -m pytest -q
python -m compileall -q -f .
python ../tools/release/build_extension.py --verify-only --root .

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
python ../tools/release/validate_phase3_warp.py \
  --expected-warp-root /mnt/data/warp_lang_1_16_external \
  --report-json /mnt/data/phase3_warp_validation.json

PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
/mnt/data/blender-5.2.0-linux-x64/blender \
  --background --factory-startup --offline-mode --python-use-system-env \
  --python-exit-code 1 --python ../tools/release/validate_phase3_blender.py -- \
  --module fluidsim_phase3_work \
  --work-dir /mnt/data/phase3_blender_validation_work \
  --report-json /mnt/data/phase3_blender_validation.json
```

## Remaining gates

Run the existing NVIDIA regression workflow using the supplied Warp package on
compatible NVIDIA hardware. Required open checks include CUDA solver numerics,
CUDA uninterrupted-versus-resumed determinism, optional-channel behavior under
GPU particle growth/compaction, VRAM admission and out-of-memory handling,
performance, motion-blur rendering on the target GPU renderer path, and Windows
validation when matching supplied runtimes become available.
