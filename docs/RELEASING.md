# Releasing FluidSim Warp

A release needs two kinds of proof. Hosted CI runs on every pull request.
The manual GPU gate runs once per release on a machine with an NVIDIA GPU.
Either one alone is not enough.

## What hosted CI proves

`.github/workflows/ci.yml` runs on every pull request, on every push to
`master` and on every `v*` tag. Its runners have no GPU. `CUDA_VISIBLE_DEVICES`
is set to `-1`, so CUDA-only tests skip rather than fail.

| Job | Proves |
| --- | --- |
| `checks` | ruff is clean, and release metadata matches between `version.py`, `blender_manifest.toml`, the Warp lock and the delivery policy. Every module compiles and the full pytest suite passes. The whitewater, surface-tension and Custom Effects kernels compile and pass their validators on Warp **CPU**. |
| `package` | The pinned Blender (checked by sha256) builds the `.zip` packages, validates them, installs the Linux one and smoke-tests it against an external Warp, then uninstalls it. |

The packages and JSON reports are attached to each run as artifacts.

## What only the GPU gate proves

These are proven only by the gate:

- the CUDA solver
- the three kernel validators on `cuda:0`
- live viewport playback
- the foreground GUI acceptance lane

Run the gate on the release commit from the repository root. Run it detached,
with a durable log, and don't touch Blender windows while it runs:

```bash
python -u tools/run_production_gate.py --include-gui > artifacts/production_gate.log 2>&1
```

It prints `FULL-PRODUCTION-GATE-PASS` when it succeeds. It then writes
`artifacts/production_gate_manifest.json`, which records:

- the commit
- the FluidSim version
- the Blender executable
- the sha256 of the demo analysis report

Every run deletes the old manifest first, and only a passing `--include-gui`
run writes a new one. If the manifest's commit is not the tagged commit, the
gate has not been run for this release. The demo report hash only changes when
simulation results change. After a solver or Warp change a new hash is
expected; otherwise, a changed hash needs an explanation before release.

## Steps

1. **Version.** Set `VERSION` (and `BUILD_CHANNEL` if it changes) in
   `fluidSim_warp_blender51/version.py`. Set the same `version` in
   `fluidSim_warp_blender51/blender_manifest.toml`. CI's verify step fails if
   the two disagree.
2. **Changelog.** Add a section for the version to
   `fluidSim_warp_blender51/CHANGELOG.md`. Say what a user will notice, and
   call out any change that alters simulation results or breaks older files.
3. **Pull request.** Open a pull request into `master` and wait for both CI
   jobs to pass. Merge it.
4. **GPU gate.** On the merged commit, run the gate as described above. Also
   run `python tools/demos/analyze_demos.py`, which must report every demo check
   passing.
5. **Tag.** Tag the merged commit `v<version>` and push the tag. CI runs again
   on the tag and uploads the packages for that exact commit.
6. **GitHub release.** Create the release from the tag and attach:
   - the Windows and Linux `.zip` packages and `SHA256SUMS.txt`. Build them
     with `python tools/release/build_extension.py --blender <blender>
     --output dist` on the tagged commit, or take the Linux one from the
     tag's CI run.
   - `artifacts/production_gate_manifest.json` from step 4.

   Paste the changelog section into the release notes.
7. **Update your own Blender.** Blender never updates an installed add-on
   from this repository, so a stale copy can keep loading unnoticed. Run
   `python tools/release/install_local.py --blender <blender> --zip <package>`.
   It installs the package, warns about legacy `fluidSim_warp_blender51`
   add-on folders, and prints the FluidSim version, the Warp build that
   actually loaded, and whether CUDA is usable. It fails when any of them is
   wrong. `--probe-only` checks without installing. The Status & Runtime
   panel also names the loaded build.

## Sharing Warp with other extensions

FluidSim does not ship a Warp wheel. **Install Warp** in add-on preferences
downloads the locked `warp-lang` 1.17.0 wheel from PyPI and extracts it into
the directory Blender assigns to this extension
(`bpy.utils.extension_path_user`). That is not the shared
`extensions/.local` wheel store, and it is not Blender's user site-packages.
If Warp is already loaded from another path, the button refuses and does not
write. Python imports an existing `sys.modules["warp"]` before consulting
`sys.path`, so changing path order after Warp Foundry or another extension
imported Warp does not switch versions. FluidSim's development override only
changes `sys.path` after explicit opt-in and is not a fix for an already
imported Warp module.

FluidSim certifies Warp 1.17.0. Its runtime rejects Warp older than 1.17.0
or a different major version before initializing it; newer 1.x versions
remain untested and show a warning. A second press when this extension's own
copy is already 1.17.0 reports success and writes nothing. An older copy in
that same directory is replaced. A copy anywhere else, including another
add-on or the user site-packages, is left in place until the user removes it.
The button also stops when `bpy.app.online_access` is off. To use FluidSim
and Warp Foundry in the same Blender process, align them on a version tested
by both projects, then restart Blender. Until that is validated, use separate
Blender processes/profiles for the extensions. Reinstalling FluidSim alone
cannot replace a `warp` module another extension already loaded. Do not
delete `sys.modules` entries, shadow the package with a private path, or swap
Warp's native libraries in process.

## Changing a pinned dependency

- **Blender.** In `ci.yml`, update `BLENDER_VERSION` and `BLENDER_SHA256`
  together, taking the hash from Blender's published `.sha256` file. CI
  downloads from an official mirror, because `download.blender.org` answers
  scripted downloads with a bot challenge. A test keeps `BLENDER_VERSION`
  equal to `CERTIFIED_BLENDER_VERSION` in `version.py`.
- **Warp.** Update `CERTIFIED_WARP_VERSION` in `version.py` and the
  `warp-lang` pin in `fluidSim_warp_blender51/requirements-dev.txt`
  together; a test keeps them equal. A Warp change can alter results, so
  expect the demo report hash to change, and record the new reference in
  the release notes.
