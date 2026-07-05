# Contributing

This repository should stay clean enough to distribute as a Blender add-on and
review as source code. Generated simulation caches, reports, exports, and local
scratch work do not belong in Git.

## Before touching `master`

Run the production gate from the repository root:

```bash
python3 tools/run_production_gate.py
```

The gate checks repository hygiene, runs the CUDA/Warp solver tests through
Blender 5.1's bundled Python, and proves the Blender viewport/live handler runs
the simulation for at least two frames.

For solver, scene-bridge, viewport, or cache changes that could affect
interactive playback cost, also run the user-path regression profile. It
protects the exact path a user exercises after Quick Setup and Initialize
Solver: scene sync/collider gather, solver step, viewport datablock update, and
optional cache write. This is opt-in so the default production gate stays
small:

```bash
FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py
```

Or run it directly with a focused report:

```bash
blender --background --factory-startup --python tools/profile_user_path.py -- \
  --frames 6 --voxel-size 0.08 --max-particles 200000 \
  --collider-subdiv 2 --json-out artifacts/user_path_profile.json
```

The supported controls are intentionally narrow: `--frames`, `--voxel-size`,
`--max-particles`, `--particles`, `--cache`, `--collider-subdiv`, the mean-time
ceilings, and the minimum active particle/voxel counts. The JSON report is the
debug surface: it records `contract.parameter_surface`,
`contract.inspection_surface`, per-check pass/fail rows, per-frame sync/solve/
view/cache timings, viewport output counts, and final scene statistics.

If Blender is installed in a non-default location, override these paths:

```bash
BLENDER_EXE="/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
BLENDER_PY="/mnt/c/Program Files/Blender Foundation/Blender 5.1/5.1/python/bin/python.exe" \
python3 tools/run_production_gate.py
```

## Artifact policy

Keep source assets in Git:

- add-on Python package files under `fluidSim_warp_blender51/`
- tests under `tests/`
- reusable tools under `tools/`
- documentation and contribution policy files
- curated `.blend` demo scenes and preview images

Keep generated output out of Git:

- `.npz` simulation frame caches
- generated reports, OBJ/PLY exports, and temporary diagnostics
- local snippet-library experiments
- scratch Blender helper scripts
- review/swarm output folders

Use `artifacts/` for regenerated local outputs and `_quarantine/` for scratch
or drift files that should be preserved locally but not distributed.
