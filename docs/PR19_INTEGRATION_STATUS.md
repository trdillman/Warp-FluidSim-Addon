# PR #19 integration status

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
no upload gate was waived and no old artifacts were deleted. The subsequent
test-selection/cancellation correction requires its own CI result.

The isolated Windows Blender 5.2 installer checks are still in progress. The isolated
installer lane hides CUDA to avoid competing with other local GPU work; it
cannot qualify solver execution, viewport behavior or GPU performance.

Windows/CUDA and foreground GUI production qualification are **BLOCKED**:
other Blender/Python processes are using the target GPU. Do not stop or alter
those sessions. Do not merge or tag 2.2.0 until candidate-specific mandatory
checks and owner/repository approval gates pass. No historical 2.1.0 receipt or
metadata-only `release_ready` field waives these gates.
