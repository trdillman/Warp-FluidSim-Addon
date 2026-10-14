# Contributing

This repository should stay clean enough to distribute as a Blender add-on and
review as source code. Generated simulation caches, reports, exports, and local
scratch work do not belong in Git.

## Continuous integration

Every pull request runs `.github/workflows/ci.yml` on GitHub-hosted runners.
The workflow lints, runs the tests, validates the kernels on Warp CPU, then
builds, installs and smoke-tests the extension in Blender. The runners have no
GPU, so CUDA-only tests skip there. A green CI run is required before merging,
but it is not GPU proof. See `docs/RELEASING.md` for the release procedure and
the manual GPU gate.

## Before touching `master`

Run the production gate from the repository root:

```bash
python3 tools/run_production_gate.py
```

The gate checks repository hygiene, runs the CUDA/Warp solver tests through
Blender 5.2's bundled Python, proves the Blender viewport/live handler runs the
simulation in background mode, and, with `--include-gui`, runs the foreground
GUI acceptance lane and writes `artifacts/production_gate_manifest.json`.

The gate takes Warp from `PYTHONPATH` and refuses to start unless Blender's
Python imports the certified version (`CERTIFIED_WARP_VERSION` in
`fluidSim_warp_blender51/version.py`). It sets `PYTHONNOUSERSITE=1` and passes
`--python-use-system-env` to Blender, so a Warp in your Python user site is
never used. Point `PYTHONPATH` at a directory holding the certified Warp wheel
installed with `pip install --target`, plus pytest.

For solver, scene-bridge, viewport, or cache changes that could affect
interactive playback cost, also run the user-path regression profile. It
protects the exact path a user exercises after Quick Setup and Initialize
Solver: scene sync/collider gather, solver step, viewport datablock update, and
optional cache write. This is opt-in so the default production gate stays
small:

```bash
FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py
```

Or run it directly with a focused report. Keep the private extensions directory:
`--factory-startup` alone deletes your installed extensions' wheels (Warp).

```bash
BLENDER_USER_EXTENSIONS="$PWD/artifacts/gate_blender_user_extensions" \
blender --background --factory-startup --python tools/profiling/profile_user_path.py -- \
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
BLENDER_EXE="/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" \
BLENDER_PY="/mnt/c/Program Files/Blender Foundation/Blender 5.2/5.2/python/bin/python.exe" \
PYTHONPATH="/path/to/warp-1.18.0-site" \
python3 tools/run_production_gate.py --include-gui
```

## Artifact policy

Keep source assets in Git:

- add-on Python package files under `fluidSim_warp_blender51/`
- tests under `tests/`
- reusable tools under `tools/`
- documentation and contribution policy files
- canonical playable `.blend` demos under `demos/`, their builders, inventory
  entries, runtime acceptance, and preview images
- clearly labelled distributed examples and validation/review assets under
  `fluidSim_warp_blender51/resources/examples/`

Keep generated output out of Git:

- `.npz` simulation frame caches
- generated reports, OBJ/PLY exports, and temporary diagnostics
- local snippet-library experiments
- scratch Blender helper scripts
- unpromoted review/swarm working output

## Demo scene policy

Follow `docs/DEMO_SCENE_CONTRACT.md`. A new or changed `demos/*.blend` must
reopen at `frame_start`, resolve an authoritative Domain object, initialize a
non-empty solver from owned sources or seeds, advance through the production
frame handlers, and publish changing owned output. Add it to
`tools/demos/check_demo_blends.py` and add focused runtime acceptance for the feature
it advertises.

Do not call a static `.blend`, rendered comparison, CPU fixture, benchmark,
package-authoring example, or review scene a demo. Store and label those assets
according to their narrower purpose. A preview image or validation report is
supporting evidence, not a substitute for live playback proof.

Keep generated review and scan output under `artifacts/` while it is being
produced. Promote only the small, curated record needed by an active task or
historical provenance; do not commit reproducible scratch output wholesale.

Use `artifacts/` for regenerated local outputs and `_quarantine/` for scratch
or drift files that should be preserved locally but not distributed.
