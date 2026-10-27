# Isolated CPU-only Blender UI testing

## Purpose and boundaries

Run genuine Blender 5.2.2 foreground UI on a Linux cloud desktop, using Mesa llvmpipe software OpenGL. This lane validates addon registration, visible control layout, genuine missing-CUDA presentation, and presentation-only scene property persistence. It is **not** CUDA, live fluid simulation, performance, cached playback, export correctness or production release qualification. No fake Warp/GPU modules are used.

## Usage

From a Linux cloud **desktop terminal**, in the FluidSim repository:

```bash
bash tools/cloud_ui/run.sh --download
```

The official Blender archive is downloaded only if the executable is absent, checked against the pinned publisher SHA256, and extracted under `artifacts/cloud_ui/runtime`. Every run checks executable SHA256, exact version and build hash. The hash receipt is in `tools/cloud_ui/runtime-lock.json`; update deliberately when qualifying a new runtime.

Optional environment variables:
- `FLUIDSIM_SOURCE_ROOT`: source checkout root, defaults to this repository
- `CLOUD_UI_WORK_ROOT`: evidence/profile/runtime root, defaults to `<source>/artifacts/cloud_ui`
- `CLOUD_UI_RUNTIME_ROOT`: existing pinned Linux runtime directory
- `BLENDER_EXECUTABLE`: explicit pinned Linux executable, still hash checked
- `CLOUD_UI_RUN_ID`: unique run name using letters, digits, hyphen and underscore

Paths are normalized to absolute paths. Existing run IDs are rejected. Use the **same WORK root for every run on one desktop** so the nonblocking flock serializes GUI sessions. Do not start other GUI lanes concurrently.

This cloud system's shell tool and desktop share files but not the display namespace. Launch through the cloud desktop's Terminal app (Sky/CUA), not a user desktop, Windows gateway, or by guessing DISPLAY in a restricted shell. The runtime failure message explicitly asks for a desktop terminal if no display is present. Sky screenshots provide the visual evidence; CLI background registration is not a substitute.

The launcher creates fresh private config/scripts/extensions/datafiles/resources/cache/temp directories and seeds only that new test profile before the canonical snapshot. It disables splash and autosave in that profile. It does not modify a normal or production profile. CUDA/Warp cache paths are reserved privately but Warp installation is not part of this lane. Do not run a dependency installer or live simulation to make this missing-runtime test pass.

## What happens

1. Runtime pin, run ownership, and display preflight.
2. Background fixture creation in the private profile.
3. Repository canonical `tools/gate/launch_profile_safe_addon_review.py` copies and verifies addon identity, snapshots private profile, and opens foreground Blender.
4. `foreground_probe.py` verifies real window/runtime state, changes only `scene.fluidsim.ui_mode` to ADVANCED, saves the fixture, and writes readiness evidence. It does **not** assert visual panel draw success.
5. Inspect foreground UI manually through supported cloud CUA, capture screenshots into the run folder. Click the FluidSim sidebar tab. A narrow sidebar can truncate the runtime explanation. Do not save preferences during the review.
6. Quit this owned Blender window normally (Ctrl+Q). Canonical final report must say `ok: true`, `profile.unchanged: true`, and owned-process/temp-copy cleanup true. `review_ready` alone is not completion.
7. Launcher starts a separate isolated background Blender process to reopen saved RNA and verifies ADVANCED mode with live simulation false. This result is distinct from foreground visual evidence.

## Acceptance checklist

- [ ] Exact executable archive/build/hash and source revision recorded
- [ ] Genuine foreground window and software OpenGL renderer recorded
- [ ] Real FluidSim sidebar visible in a screenshot; inspect Basic/Advanced layout
- [ ] Missing-runtime message is genuine and no solver mocked
- [ ] Presentation-only setting saved and reopened in fresh process
- [ ] No tracebacks in final accepted run logs
- [ ] Owned process exited, temporary addon copy removed, private baseline unchanged
- [ ] Preferences installation UI, setup assets, live simulation, demos, playback/export explicitly marked untested unless separately tested

The canonical temporary-registration launcher does not establish normal installed-extension Preferences UI. During the initial review, FluidSim was not listed in Preferences → Add-ons; installation/runtime button visibility was therefore not validated.

## Regression guards

```bash
python3 -m unittest discover -s tests/cloud_ui -v
bash -n tools/cloud_ui/run.sh
python3 -m py_compile tools/cloud_ui/*.py
```

Four guard tests verify invalid run ID, reuse of evidence directory, wrong executable, and concurrent lock rejection without launching Blender.

## Known probe development findings

An early probe reopened its current `.blend` inside a timer and then accessed UI context. First it produced `context.screen is None`; a second attempt crashed while mutating UI after reopen. The current probe uses a fresh process for reopen, avoiding stale UI context. Exact native root cause was not established; these failures are not attributed to product code. A fresh profile's first-run Continue button also saved preferences and correctly failed the canonical drift check. Current fixture creation seeds the test profile **before** the before/after snapshot. Failed-run reports are retained separately from final evidence.
