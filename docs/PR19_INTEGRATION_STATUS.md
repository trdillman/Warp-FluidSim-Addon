# PR #19 integration status

## Current integration (2026-10-13)

The owner requested reconciliation and merge of the Warp install change,
while keeping GitHub workflows removed. The integration combines PR head
`7f4f51329878f6de98971d63bf2b1c5d3f0d4650` with master
`8b037aae48b12af60cd6d10212d4eae1956d1dff`. FluidSim remains **3.2.0**;
current master features, cache-v3 behavior, registration and bundled demos
are retained. Extension archives contain no Warp wheels. The explicit
Preferences installer and `install_local.py --install-warp` provision the
locked Warp 1.17.0 CUDA 12 runtime only into the extension-owned user directory.

The deleted GitHub workflows remain deleted. Workflow-specific tests have
been replaced by applicable local checks and a regression against workflow
restoration. Historical artifact quota failures are not current merge gates.
Local checks and an independent review qualify this integration; release
tagging still requires the GPU/GUI evidence described in `RELEASING.md`.

Fresh source checks: 146 focused tests passed; full CPU regression passed
2,676 tests with 299 skipped; add-on and touched-tool lint, compilation and
source/lock/manifest verification passed. All three Warp CPU validators passed.
Independent source review found a loaded-runtime repair defect when owned
version metadata is missing. Both regression cases failed before the fix;
113 affected installer/runtime/discovery/build/helper checks passed afterward.
The installer now stages any loaded owned runtime for restart independently
of its version metadata. Independent correctness and intent review passed.

Isolated Blender installer qualification is in progress. The first attempt
used a Blender 5.2.0 installation with foreign Warp preinstalled in its Python
site; its ownership refusal produced 5 failures and 2 passes. Evidence is
retained in `real-blender-installer.log`. The bounded fallback uses a private
local copy of the same executable (matching SHA-256), with its copied foreign
Warp preserved outside that runtime. The copied Python finds no Warp before
installation; all profiles are private and CUDA is hidden. This is installer
qualification, not CUDA solver, GUI or fresh official-archive provenance.
Logs and packages are retained locally under
`artifacts/pr19-integration-20261013/`. No tag or release publication is requested.

## Historical candidate evidence (2026-10-09)

The remaining sections record the earlier 2.2.0 proposal and its checks.
Those results and blockers describe that historical candidate, not the
current integration or current process availability.

## Candidate and preserved release

Proposed FluidSim 2.2.0 reconciles installer head
`514e61bdb5b75868ebc05666f15e81d8cc1db016` with master
`cc364034854135340e27d362e10f5d494ff71608`. The existing 2.1.0 release,
tags, binaries and historical qualification records are unchanged.

The extension zip has no bundled Warp wheels. Preferences explicitly downloads
the locked Warp 1.17.0 CUDA 12 wheel into Blender's extension-owned user directory.
Foreign-copy writes remain refused; runtime version admission is unchanged.
An external runtime is displayed only while process opt-in is active and its
loaded module exactly matches the configured, applied developer path.
A successful CUDA runtime refresh reschedules deferred kernel prewarm; staged
repair still waits for restart. Master's registration table remains authoritative.

## Review findings

| Finding | Candidate resolution |
| --- | --- |
| Certified-copy repair unavailable | Retained forced staged repair, loaded-native no-mutation and fresh-process coverage. |
| Missing factory startup | Retained factory startup and private extension/profile directories. |
| Upgrade guidance | Retained explicit 2.1-to-2.2 Install Warp step. |
| Local helper cannot provision Warp | Retained explicit `--install-warp`; corrected agent-facing recovery command. |
| Authorized override hides controls | Added loaded-path authorization check and external runtime UI lane; incompatible-version rejection retained. |
| One-shot prewarm retires before install | Available runtime refresh reschedules the existing owned timer. |
| Historical host appears to qualify 2.2 | Technical report labels historical evidence and links this candidate record. |

Independent read-only review checked the reconciliation and implementation.
Its runtime, documentation and explicit-Blender-selection findings were addressed.
An explicitly missing configured Blender now fails rather than skipping qualification;
the focused regression failed before the fix and passes afterward. Unit regressions first reproduced the
two runtime/UI defects, then passed after the repair: 34 installer tests on
Python 3.13 with pinned development dependencies.

## Qualification gates

Windows CPU regression passed: **1,929 passed, 191 skipped** (CUDA hidden).
The initial run exposed an uncertified Blender 5.1 PATH selection and two incomplete
runtime test doubles; these were corrected before the passing rerun.
The 34 installer unit tests, lint, metadata/lock verification, compilation and
three Warp CPU validators passed.

Hosted CI run [37932715129](https://github.com/trdillman/liquid_warp_blender51_port/actions/runs/37932715129)
at candidate `9f440a73c9326bfaa910dec1e4948413290b90d3` passed lint, tests,
all three CPU validators, package build/install/smoke/uninstall, and all six
isolated Linux Blender 5.2 installer checks. Both mandatory artifact uploads
failed because GitHub artifact storage quota was exhausted. CI remains failed;
no upload gate was waived and no old artifacts were deleted. Latest reviewed source candidate `557d764d81b3b76f03e5e860341a6c0cb85cb4d0`
repeated all validation and package checks successfully in
[run 37933282104](https://github.com/trdillman/liquid_warp_blender51_port/actions/runs/37933282104);
only the two mandatory artifact uploads failed. Independent source review
reported no remaining substantive findings at that candidate. All five
existing inline review threads are resolved. This bounded approval does not
waive Windows/CUDA/GUI or artifact gates.

The isolated Windows Blender 5.2 installer lane passed **6 tests in 2,008.21s**
with CUDA hidden: offline refusal, fresh installation/idempotency/loaded repair,
foreign-copy refusal, corrupted-owned-copy repair/restart, explicit local helper,
and authorized external override/version rejection/opt-in revocation.

This Windows fixture package was built before later test, workflow and documentation
commits. SHA-256 comparison confirmed its preferences, discovery, installer,
runtime probe, prewarm and registration modules match reviewed source candidate
`557d764d81b3b76f03e5e860341a6c0cb85cb4d0` byte for byte. The retained local
`isolated-package-runtime-identity.json` records that bounded identity;
this is not final-package or GPU/GUI qualification. Hosted CI built the reviewed
candidate itself. Task-start and end-task storage checks exited successfully;
passing receipts and failed-attempt evidence are retained. The isolated
installer lane hides CUDA to avoid competing with other local GPU work; it
cannot qualify solver execution, viewport behavior or GPU performance.

At that time, Windows/CUDA and foreground GUI production qualification were
blocked by other Blender/Python sessions using the target GPU. The earlier
candidate was held pending its checks and owner approval. Historical receipts
and metadata-only `release_ready` fields do not qualify a new release.
