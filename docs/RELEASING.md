# Releasing FluidSim Warp

A release needs a passing local CPU/package gate on the exact release commit
and a production GPU gate on a machine with an NVIDIA GPU. GitHub Actions
workflows are disabled. Keep the local gate receipts with the review and release.

## Storage workflow

Normal `python tools/run_production_gate.py` automatically runs the vendored
storage regression suite, reports `pre-production`, and reports `end-task` in
`finally`, including production and hygiene-test failures. Hygiene-test failure
blocks production; storage warnings remain diagnostics and do not change its
exit code. Reports never delete, move or recycle files. No external installation
is needed; see `tools/storage_hygiene/PROVENANCE.md`.

At task startup run `python tools/storage_hygiene/storage_hygiene.py check`.
Before allocating output, record absolute path, project, owner, run ID, source
SHA, state, budget/expiry, active lease, retention reason and protected evidence.
For an owned output policy set `FLUIDSIM_STORAGE_POLICY` to its JSON path, using
the vendored policy schema; roots cannot overlap. Stale warnings require a
completed registered disposable root and a complete scan. Default container
scans are bounded lower bounds and deliberately exclude protected material.
On non-Windows hosts, the bundled Windows paths remain valid policy entries
but are reported as unmeasured with unknown budgets; no Windows drive or root
is probed or translated to a mounted path. Native owned-output policies still
use the same bounded scanning and protection rules.
Overrides retain all bundled denied/protected paths and the current checkout
as never-probe holds. Measure owned disposable outputs outside those holds;
an override cannot authorize probing retained source/proof or denied material.
The four-container telemetry pass remains mandatory with an override, which adds
a separate owned-output pass. Their measurements are not additive. Observed
file-age warnings include retained/unknown material and never authorize cleanup;
age is neither task age nor evidence of obsolescence. Incomplete budget assessment
is unknown unless its measured lower bound already exceeds the budget. Traversal
remains bounded at 20,000 entries/0.5 seconds per root, plus OS metadata latency;
the default four-root allowance is about two seconds per phase. Existing denied
paths and newly observed inaccessible paths remain excluded on later phases.

At task end reconcile records and report retained/cleanup-eligible outputs.
Clean only owned disposable artifacts through a verified recoverable route;
keep source, dirty work, autosaves, qualified packages/baselines and failure
evidence. A warning grants no deletion authority. Permanent deletion requires
exact approval. Missing/invalid mandatory policy or unavailable mandatory
pre-production telemetry blocks admission with exit 2, before production starts.
Invalid optional overrides remain diagnostic while the default pass is mandatory.
End-task telemetry failure is diagnostic and preserves the completed production
exit code or exception. Valid incomplete, aging, budget and low-space results
remain warnings.
The default policy warns below 30 GiB free and retains known denied paths as
never-probe. Storage-hook tests alone do not requalify GPU/GUI behavior.

## What the local gate proves

Run `tools/release/run_local_gate.py` from a clean checkout at the commit under
review. Install the exact pins in `fluidSim_warp_blender51/requirements-dev.txt`
in an isolated Python environment first. The gate checks those versions and
sets `CUDA_VISIBLE_DEVICES=-1` for its children. It runs ruff, source/package
metadata verification, compileall, the full pytest suite, and the phase 4,
phase 5 and Custom Effects validators on Warp CPU. A failed command produces
no pass receipt.

Supply a reviewed SHA-256 digest for the certified Blender executable.
The gate checks its bytes, builds and inspects both platform packages, then
installs, smoke-tests and uninstalls the package for the **host** platform in
isolated Blender user directories. It writes `local-gate-report.json`, the
package report, CPU validator reports and package checksums under `--output`.
An approved digest must come from a retained, trusted Blender distribution;
the digest argument is an input to review, not proof of provenance by itself.

Run the gate on both Windows and Linux before a cross-platform release. Each
host smoke-tests its own package. Keep both receipts, Blender distribution
checksums and the resulting packages. A Windows receipt alone does not prove
Linux installation. Use an owned output path outside the checkout, for example:

```bash
python tools/release/run_local_gate.py --output <owned-output> \
  --blender <certified-blender> --blender-sha256 <approved-executable-sha256>
```

`--checks-only` runs the CPU lane without claiming a package smoke pass.

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
run writes a new one.

Validation is scoped to what changed. The release needs one passing
`--include-gui` run on the release commit, and its `demo_artifact_sha256` must
match the reference recorded for that line (a different hash means simulation
results changed and needs an explanation, not a rerun). Demo bakes are bitwise
deterministic since d2a6b6d, so repeating the run proves nothing new; the old
three-consecutive-passes rule is retired.

The paired performance budget (about twenty minutes of Blender trials) is only
measured when it has to be. By default the gate reuses
`artifacts/performance_budget/summary.json` when it passed and no budget
source changed since the commit it measured: the add-on (minus its docs and
Markdown), `tools/repo_addon_loader.py`, the benchmark tools, the budget evaluator,
`tools/demos`, and the saved paddle/vortex scenes that the gate does not rebuild.
Documentation, rebuilt dam/crown blends and test-only commits therefore keep the
measured budget. The manifest records `performance_budget_reused` and
`performance_budget_measured_at`. `--measure-budget` forces a fresh
measurement; `--reuse-budget` fails instead of measuring.

For changes that do not touch solver or simulation output, run the focused
tests for the area (for example `tests/gate` for gate tooling) or `--quick`;
a full gate is only needed for the release commit or a change to simulation
results.

`--quick` is the tier for localized changes: hygiene, the Python and CUDA
checks, the release validators and the background acceptance scripts, ending
in `QUICK-GATE-PASS` (about six minutes). It skips the demo rebuild, GUI lanes
and budget, writes no manifest and is never release evidence. In every tier
the independent Python checks run in parallel, and the background acceptance
scripts share one Blender session (`tools/gate/run_blender_batch.py`); any
script that does not pass there is rerun alone, and that run decides. Every
step prints its wall time, summarised at the end and in
`artifacts/gate_timing.json`.

If the manifest's commit is not the tagged commit, the
gate has not been run for this release. The demo report hash only changes when
simulation results change. After a solver or Warp change a new hash is
expected; otherwise, a changed hash needs an explanation before release.

## Steps

1. **Version.** Set `VERSION` (and `BUILD_CHANNEL` if it changes) in
   `fluidSim_warp_blender51/version.py`. Set the same `version` in
   `fluidSim_warp_blender51/blender_manifest.toml`. The local verify step fails if
   the two disagree.
2. **Changelog.** Add a section for the version to
   `fluidSim_warp_blender51/CHANGELOG.md`. Say what a user will notice, and
   call out any change that alters simulation results or breaks older files.
3. **Pull request.** Open a pull request into `master`. Run the local gate on
   its exact commit on Windows and Linux, attach both receipts and reports for
   owner review, then merge after review.
4. **GPU gate.** On the merged commit, run the gate as described above. Also
   run `python tools/demos/analyze_demos.py`, which must report every demo check
   passing.
5. **Tag.** Rerun both local gates on the merged commit, retain their packages
   and receipts, then tag that exact commit `v<version>` and push the tag.
6. **GitHub release.** Create the release from the tag and attach:
   - the Windows and Linux `.zip` packages and `SHA256SUMS.txt` produced by
     the local gate on the tagged commit, with both host receipts.
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

Blender installs extension wheels into its Python environment. A bundled
`warp_lang` wheel supplies the top-level `warp` package; it does not give
each extension a private `warp` module. Python imports an existing
`sys.modules["warp"]` before consulting `sys.path`, so changing path order
after Warp Foundry or another extension imported Warp does not switch versions.
FluidSim's development override only changes `sys.path` after explicit opt-in
and is not a fix for an already imported Warp module.

FluidSim 2.0 bundles and certifies Warp 1.17.0. Its runtime rejects Warp
older than 1.17.0 or a different major version before initializing it;
newer 1.x versions remain untested and show a warning. To use FluidSim and
Warp Foundry in the same Blender process, align their bundled Warp versions
on a version tested by both projects, then restart Blender. Until that is
validated, use separate Blender processes/profiles for the extensions.
Reinstalling FluidSim alone cannot replace a `warp` module another
extension already loaded. Do not delete `sys.modules` entries, shadow the
package with a private path, or swap Warp's native libraries in process.

## Changing a pinned dependency

- **Blender.** Update `CERTIFIED_BLENDER_VERSION` in `version.py`, obtain
  reviewed Windows and Linux distributions and published archive checksums,
  and record the executable SHA-256 values used by the local gates. Retain the
  distribution provenance with the gate receipts. Validate both platforms.
- **Warp.** Update `CERTIFIED_WARP_VERSION` in `version.py` and the
  `warp-lang` pin in `fluidSim_warp_blender51/requirements-dev.txt`
  together; a test keeps them equal. A Warp change can alter results, so
  expect the demo report hash to change, and record the new reference in
  the release notes.
