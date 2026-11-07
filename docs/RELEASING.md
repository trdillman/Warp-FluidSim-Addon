# Releasing FluidSim Warp

A release needs local source/package checks and a manual GPU gate on a
machine with an NVIDIA GPU. GitHub Actions workflows have been removed;
historical CI results and artifact uploads are not current merge gates.

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

## Local source and package checks

Run these checks locally on the candidate commit and retain their logs.
For CPU-only regression checks, set `CUDA_VISIBLE_DEVICES=-1`; those results
do not qualify CUDA execution or the foreground GUI.

| Check | Proves |
| --- | --- |
| Source | ruff is clean, and release metadata matches between `version.py`, `blender_manifest.toml`, the Warp lock and the delivery policy. Every module compiles and the full pytest suite passes. The whitewater, surface-tension and Custom Effects kernels compile and pass their validators on Warp **CPU**. |
| Package | The certified Blender builds and validates the platform `.zip` packages. Run the install/smoke/uninstall lane on the host platform with the locked Warp wheel provisioned into this extension's own user directory; the zip itself has no wheels. |

Keep the packages, checksums and JSON reports as local qualification evidence.

The optional `tools/release/run_local_gate.py` helper runs the source checks,
CPU validators and host-platform package smoke together. Use a clean checkout,
the pinned development environment, an owned output directory outside the
checkout, and a reviewed SHA-256 for the certified Blender executable:

```bash
python tools/release/run_local_gate.py --output <owned-output> \
  --blender <certified-blender> --blender-sha256 <approved-executable-sha256>
```

The helper writes `local-gate-report.json` only after successful checks and a
final unchanged-commit/clean-checkout check. `--checks-only` excludes package
smoke and records that exclusion. Its receipt does not replace the GPU/GUI gate
or qualify an untested operating system. Retain Blender distribution provenance
alongside the executable digest; supplying a digest alone does not establish it.


## What only the GPU gate proves

### Cache-directory ownership

Use a distinct cache directory for each independently running Blender process.
Before Bake, Resume or Re-bake, stop any other process that can write that
directory, including background jobs. Process-local bake guards do not provide
a cross-process lock. Concurrent writers to one directory are unsupported;
atomic frame/checkpoint writes do not make that workflow safe. Retain a copy
of valuable caches before recovery or continuation experiments. Cross-process
refusal/recovery must be implemented and exercised before concurrent writing
can be advertised as supported.

### Runtime evidence

These are proven only by the gate:

- the CUDA solver
- the three kernel validators on `cuda:0`
- live viewport playback
- the foreground GUI acceptance lane

Run the gate from a clean, committed release candidate. Commit verifier changes
before requesting approval or measuring a budget. Product and verifier inputs
are frozen at admission; a changed HEAD or changed source during the run
prevents a receipt. The only generated-file exception is `demos/dam_break.blend`
and `demos/crown_splash.blend`: changed bytes require equal observed authored
scene semantics before and after rebuilding. The receipt records both byte
hashes and that semantic fingerprint. Unsupported semantic data fails closed.
An add-on-code change never qualifies for this exception.

Qualification requires an approved Blender executable digest and a reviewed
JSON distribution record outside the checkout. The record must contain:

```json
{
  "blender_download_url": "https://download.blender.org/release/Blender5.2/<official-archive>.tar.xz",
  "blender_archive_sha256": "<64 lowercase hexadecimal characters>",
  "blender_executable_sha256": "<64 lowercase hexadecimal characters>",
  "bundled_python_sha256": "<64 lowercase hexadecimal characters>",
  "verifier_fingerprint": "<64 lowercase hexadecimal characters>",
  "reviewed_by": "<reviewer identity>"
}
```

Use the official `.zip` URL for Windows. Retain the official archive and its
published checksum alongside this record; writing a JSON assertion is not an
independent verification of the distribution. The verifier fingerprint includes
tracked tools and tests, including the evaluator and pinned baseline. Obtain it
from the committed checkout for review:

```bash
python -c 'from tools.gate.qualification import freeze_source; from pathlib import Path; print(freeze_source(Path.cwd())["verifier_fingerprint"])'
```

Set the following variables to absolute paths and reviewed values. The retained
Warp archive must match `tools/release/wheels.lock.json`; the loaded Warp source
and native files, including JIT headers, must match that archive. Python bytecode and caches are not
compared. `PYTHONPATH` must contain the certified installed Warp plus pytest.

```bash
export BLENDER_EXE="/path/to/certified-blender/blender"
export BLENDER_PY="/path/to/certified-blender/5.2/python/bin/python3.13"
export BLENDER_SHA256="<approved executable SHA-256>"
export BLENDER_PROVENANCE="/owned/evidence/blender-provenance.json"
export FLUIDSIM_WARP_WHEEL="/owned/downloads/<locked-host-Warp-wheel>.whl"
export PYTHONPATH="/owned/runtime/warp-1.19.0-site"
python -u tools/run_production_gate.py --include-gui --measure-budget > artifacts/production_gate.log 2>&1
```

Use the equivalent environment assignments on Windows and its bundled
`python.exe`. The gate rejects Blender 5.1 and any build other than the certified
version in `version.py`. An available Blender 5.2.1 used for diagnostic checks
does not qualify a 5.2.0 release contract. The bundled Python must belong to the
approved distribution and match Blender's observed Python version/prefix.
`nvidia-smi` must be available to record the actual driver for the observed
Warp GPU UUID. Do not touch Blender windows while foreground acceptance runs.

Each production invocation owns private absolute Blender configuration, scripts,
extensions, datafiles, temporary files and caches. Caller profile overrides are
replaced; configured `PYTHONPATH` is retained for this source gate, while
`PYTHONHOME` and `PYTHONSTARTUP` are removed. The observed bundled Python prefix
must remain inside the approved distribution.

Approval policy and measurement are separate records. The gate observes Blender
version/build/executable hash, Python identity, Warp version/distribution/loaded
path/locked-wheel hash, OS/architecture, GPU UUID/model and driver. It repeats
source and runtime checks before issuing a receipt. Every invocation, including
`--quick`, moves the previous current manifest into its own new
`artifacts/qualification_runs/<run_id>/prior_manifest.json` before storage
admission. A storage or admission failure therefore leaves no current passing
manifest. Existing evidence is retained, not rewritten as a new pass.

A successful full run prints `FULL-PRODUCTION-GATE-PASS` and atomically writes
both its retained run manifest and `artifacts/production_gate_manifest.json`.
The receipt contains the commit, FluidSim version, `run_id`,
`source_fingerprint`, `verifier_fingerprint`, `environment_fingerprint`, the
observed environment and separate compatibility policy, demo-analysis hash,
generated-demo exceptions, and the path/hash of the retained performance report.
The report's raw trial paths and hashes must still resolve. Preserve each
`artifacts/performance_budget/runs/<attempt_id>/` directory with its summary,
raw reports, settings and logs, including failed attempts. A later measurement
uses a new directory and cannot overwrite a previous run's raw evidence.

Validation is scoped to what changed. The release needs one passing
`--include-gui` run on the release commit, and its `demo_artifact_sha256` must
match the reference recorded for that line (a different hash means simulation
results changed and needs an explanation, not a rerun). Demo bakes are bitwise
deterministic since d2a6b6d, so repeating the run proves nothing new; the old
three-consecutive-passes rule is retired.

The paired performance budget reuses only a completed passing measurement with
required provenance and retained raw reports. Each reused demo must match the
current source inputs, verifier and observed environment; legacy evidence,
missing reports, a different GPU/driver/Blender/Warp/platform, or changed verifier
requires new measurement. The current pinned comparison is release 4.0.1, as
recorded in `REFERENCE_COMMITS` and `REFERENCE_VERSIONS`. Candidate modifications
must not silently replace baseline or policy pins: those files belong to the
reviewed verifier fingerprint. Numerical thresholds are unchanged.

Documentation-only commits may reuse matching evidence. A generated demo can
reuse only matching semantic inputs; saved workload or verifier changes require
measurement. An interrupted campaign cannot be promoted merely because its raw
rows happen to pass. Revalidation never edits an earlier receipt's SHA or verdict.
The manifest records `performance_budget_reused` and the measured commit per
demo. `--measure-budget` forces fresh measurement; `--reuse-budget` fails instead
of measuring. Warp 1.19 still needs actual GPU/GUI qualification; source and
CPU-only tests are not that evidence.

A measurement stops each demo early when its first ABBA and BAAB blocks (four
trials per variant) settle it: the order-balanced host ratio clears the budget
by `EARLY_STOP_MARGIN` (0.06, above the largest 2-block vs 4-block gap seen on
the 4.0 campaign, 0.051) and the accuracy guard holds, or exceeds it by that
margin. Anything closer runs all four blocks and is judged exactly as before.
A clear four-demo pass takes about eight minutes instead of twenty;
`tools/benchmark_versions.py --demo-budget --full-schedule` always runs every
block. Trial children skip Blender's teardown after writing their report.

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
script that does not pass there is rerun alone, and that run decides. In a
full run the release validators and acceptance batch, the longest acceptance
script (its own batch), and the demo rebuild, analysis and blend checks run as
three concurrent lanes; `QUIET_ACCEPTANCE` (scripts that read free VRAM, bound
wall time, or open the rebuilt demos) then runs alone before the GUI lanes. Every
step prints its wall time, summarised at the end and in
`artifacts/gate_timing.json`.

If the manifest's commit is not the tagged commit, the
gate has not been run for this release. The demo report hash only changes when
simulation results change. After a solver or Warp change a new hash is
expected; otherwise, a changed hash needs an explanation before release.

## Steps

1. **Version.** Set `VERSION` (and `BUILD_CHANNEL` if it changes) in
   `fluidSim_warp_blender51/version.py`. Set the same `version` in
   `fluidSim_warp_blender51/blender_manifest.toml`. The local source verifier
   fails if the two disagree.
2. **Changelog.** Add a section for the version to
   `fluidSim_warp_blender51/CHANGELOG.md`. Say what a user will notice, and
   call out any change that alters simulation results or breaks older files.
3. **Pull request.** Open a pull request into `master`, resolve conflicts,
   complete the applicable local source/package checks and owner review,
   then merge it. Do not restore the removed GitHub workflows.
4. **GPU gate.** On the merged commit, run the gate as described above. Also
   run `python tools/demos/analyze_demos.py`, which must report every demo check
   passing.
5. **Tag.** After release qualification, tag the merged commit `v<version>`
   and push the tag. Build packages locally from that exact commit.
6. **GitHub release.** Create the release from the tag and attach:
   - the Windows and Linux `.zip` packages and `SHA256SUMS.txt`. Build them
     with `python tools/release/build_extension.py --blender <blender>
     --output dist` on the tagged commit.
   - `artifacts/production_gate_manifest.json` from step 4.

   Paste the changelog section into the release notes.
7. **Update your own Blender.** Blender never updates an installed add-on
   from this repository, so a stale copy can keep loading unnoticed. Run
   `python tools/release/install_local.py --blender <blender> --zip <package>`.
   It installs the package, warns about legacy `fluidSim_warp_blender51`
   add-on folders, and prints the FluidSim version, the Warp build that
   actually loaded, and whether CUDA is usable. It fails when any of them is
   wrong. This default does not download Warp. Add `--install-warp` to
   explicitly download the certified runtime through the installed
   extension's checksum-verified installer. If the extension is already
   installed, `--probe-only --install-warp` provisions Warp without replacing
   the extension package. `--probe-only` by itself is read-only and performs
   no network access. The Status & Runtime panel also names the loaded build.

## Sharing Warp with other extensions

FluidSim does not ship a Warp wheel. **Install Warp** in add-on preferences
downloads the locked `warp-lang` 1.19.0 wheel from PyPI and extracts it into
the directory Blender assigns to this extension
(`bpy.utils.extension_path_user`). That is not the shared
`extensions/.local` wheel store, and it is not Blender's user site-packages.
If Warp is already loaded from another path, the button refuses and does not
write. Python imports an existing `sys.modules["warp"]` before consulting
`sys.path`, so changing path order after Warp Foundry or another extension
imported Warp does not switch versions. FluidSim's development override only
changes `sys.path` after explicit opt-in and is not a fix for an already
imported Warp module.

FluidSim certifies Warp 1.19.0. Its runtime rejects Warp older than 1.17.0
or a different major version before initializing it; newer 1.x versions
remain untested and show a warning. A second press when this extension's own
copy is already 1.19.0 reports success and writes nothing. An older copy in
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

- **Blender.** Update `CERTIFIED_BLENDER_VERSION` in `version.py` and the
  corresponding local harness defaults together. Obtain Blender from an
  official distribution and verify its published SHA-256 before qualification.
- **Warp.** Update `CERTIFIED_WARP_VERSION` in `version.py` and the
  `warp-lang` pin in `fluidSim_warp_blender51/requirements-dev.txt`
  together; a test keeps them equal. A Warp change can alter results, so
  expect the demo report hash to change, and record the new reference in
  the release notes.

## Cloud UI smoke testing

See [isolated cloud Blender UI testing](CLOUD_BLENDER_UI.md) and the
[2026-10-15 integration provenance](CLOUD_BLENDER_UI_INTEGRATION_20261015.md).
The retained Linux Blender 5.2.2 evidence is for PR30 commit `295c4cdf`,
not a qualification of this checkout or a production acceptance pass.
