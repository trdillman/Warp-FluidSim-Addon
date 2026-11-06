# FluidSim 4.1 stabilization status — 6 November 2026

Source remediation is proposed in seven draft fix PRs; the combined candidate passes the pinned CPU regression suite. Production release remains **NO-GO** until reviewed fixes are integrated and the exact candidate and installed packages pass certified target-host qualification. No release tag, production publication, or remote merge was performed by this remediation task.

The original review covered `80bf4aa`; refreshed master `f8d5b32` adds PR53's reviewed 4.0.1 performance reference. The original checkout and its untracked review/plan documents were preserved. Work uses separate branches/worktrees, with `codex/release-integration-20261106` as the reviewable combined candidate. It includes existing PR47 as the installer dependency; it does not include PR49/51/52 or merge anything to master.

| Area | Finding | PR | Status |
| --- | --- | --- | --- |
| Build profile isolation | R5 | [#54](https://github.com/trdillman/Warp-FluidSim-Addon/pull/54) | Source review clean; certified package acceptance pending |
| Installer retries / Preferences | R1, R9 | [#55](https://github.com/trdillman/Warp-FluidSim-Addon/pull/55), stacked on [#47](https://github.com/trdillman/Warp-FluidSim-Addon/pull/47) | Source review clean; certified foreground matrix pending |
| Source checks / retained evidence | Task 1 | [#56](https://github.com/trdillman/Warp-FluidSim-Addon/pull/56) | Full CPU suite 3,338 passed, 345 skipped; Ruff clean |
| Input identity / verified bytes | R2, R12 | [#58](https://github.com/trdillman/Warp-FluidSim-Addon/pull/58) | Source review clean; certified continuation/performance pending |
| Frozen inputs / preview restoration | R3, R8 | [#59](https://github.com/trdillman/Warp-FluidSim-Addon/pull/59), stacked on #58 | Source review clean; host staging passes; actual CUDA bake/restore pending |
| Paused display / cutover / fallback attributes | R6, R7, R13 | [#57](https://github.com/trdillman/Warp-FluidSim-Addon/pull/57) | Source review clean; native 5.2.1 cache diagnostic passes; certified CUDA/GUI pending |
| Source/environment qualification binding | R4, R10, R11 | [#60](https://github.com/trdillman/Warp-FluidSim-Addon/pull/60), stacked on #57 and including #54 | Source review clean; actual GPU/runtime/package qualification pending |

## Combined verification

Product/verifier source at `fb455288269946325a60eb7caccd1d6c472e9f0f`:

- Full pinned Python 3.13.5 / Warp 1.19.0 suite: **3,544 passed, 345 skipped**, zero failures, 147.80 seconds. Skip reasons retained; missing CUDA/Blender cases are not passes.
- Configured add-on Ruff: clean. In-memory compilation: **653 tracked Python files**, zero errors. Release metadata verification: passed, metadata only.
- Shared Admission merge resolved preserving newer retirement cleanup and frozen-input restoration; focused integrated boundary tests: **124 passed**.
- First aggregate run retained **nine failures**. Corrections kept the Live/counter/stride assertions, updated scoped doubles and declared settings access, retained the current changelog heading, and classified manual diagnostics. Final full run above verifies those corrections together.
- Real production-gate invocation ran mandatory storage checks, then refused admission because reviewed certified executable provenance was absent (`BLENDER_SHA256`); exit 1, no authoritative production manifest. No fabricated provenance or hardware bypass was supplied.
- Independent review fixes cover installer range/visible-control checks, inactive/domain identity semantics, secondary session cleanup, pre-cutover preview restoration, bounded immutable frame reads, Warp JIT headers, source-admission races and process-profile isolation.

The following documentation-only commit records these results; it is not a new production qualification receipt. Exact commands, stdout, failed attempts and per-lane reports are retained in the evidence roots below.

## Remaining acceptance implementation and execution

| Boundary | Evidence available | Required before production release |
| --- | --- | --- |
| Installer | Host transport/ownership tests; actual RNA icons; installed no-Warp foreground script | Complete and run the native good/damaged/foreign/offline/cancel/repeated-click matrix on the final installed ZIP |
| Frozen input continuation | Checkpoint/state tests; two fresh Blender processes staging real inputs | Complete and run actual CUDA ProductionBake uninterrupted/Resume/Re-bake and preview finish/cancel/admission/step-failure acceptance; direct SceneInputs staging does not cover this |
| Qualification environment | Host negative controls; actual 5.2.1 scene stability, save/reopen and mutations | Approved 5.2.0 provenance, locked Warp/native payload observation, GPU/driver, paired budgets, GUI, exact installed archive and platform qualification |
| Existing PR blockers | Existing owners report native evidence in #49/#51 | Independently review/integrate those fixes and rerun on the final source; their prior results do not certify this candidate |

## Integration and release sequence

1. Review the independent fix PRs and existing prerequisites. #55 is stacked on #47; merge #47 first and retarget #55. Cache continuation requires both #58 and #59 plus the shared Admission interface in #57. #60 includes the #54 helper and targets #57; retarget after its prerequisites merge. The combined candidate resolves shared-file overlaps and refreshes the derived ownership map.
2. Include reviewed existing blockers: [#51](https://github.com/trdillman/Warp-FluidSim-Addon/pull/51) covers native export/depsgraph safety; [#49](https://github.com/trdillman/Warp-FluidSim-Addon/pull/49) covers whitewater acceptance revision tracking. Competing alternatives #50/#48 are not additional fixes to combine blindly. Review [#52](https://github.com/trdillman/Warp-FluidSim-Addon/pull/52) independently for Deck scope. None is merged by this task.
3. Freeze the merged candidate and verifier source. Use the qualification inputs documented in the updated release procedure; refuse missing identity/provenance instead of reusing an old receipt.
4. Run pinned source/CPU, exact certified Blender **5.2.0**, locked Warp **1.19.0**, CUDA, foreground GUI, nine-demo, four-demo performance, and installed exact-ZIP acceptance on each supported platform. Windows is the existing certified platform; Linux requires separate qualification.
5. Complete the actual installer matrix (missing/good/damaged/foreign Warp, offline wheel, cancellation/repeated clicks), fresh-process animated frozen-input Resume/Re-bake, preview finish/cancel/failure restoration, paused cache edits, and fault-recovery acceptance. Diagnostic CPU/Blender 5.2.1 observations do not satisfy these target-host checks.
6. Retain exact source/verifier/runtime/package digests, review the requirements-to-results matrix, and approve publication of the qualified bytes. Recheck unused version/tag before reserving 4.1.0; never retarget an existing tag or rebuild supposedly equivalent release bytes.
7. After publication, verify clean consumer download/install and retain a rollback package and support triage checklist, as specified in the full plan.

## Evidence limits

This host has no NVIDIA device/driver. Available Blender is 5.2.1, not certified 5.2.0. CPU and RNA/profile diagnostics are reported separately; skipped required hardware lanes remain unqualified. Cache identity hashing and immutable frame snapshots have measured overhead in the lane reports and need representative production performance review.

Retained local evidence and review reports live under `/workspace/scratch/fluidsim-remediation-20261106`; the original review evidence is under `/workspace/scratch/fluidsim-review-80bf4aa-20261106`. Ownership, leases, budgets and retention are recorded. No evidence cleanup was performed.

## Execution decisions

- Authorized independent draft PRs and isolated worktrees preserve the original checkout. No remote merge, tag or release publication occurred.
- Existing failing baseline checks were repaired as part of the requested remediation, with the original failures retained.
- CPU and Blender 5.2.1 diagnostics support review only; certified hardware and the remaining native acceptance implementation are explicit release prerequisites.
- Cloud skill helper scripts had no local executable path; equivalent brief, ledger, diff-package and review files were retained in the owned run root.
- The POSIX open-inode replacement test is explicitly excluded on Windows because ordinary Windows opens lack delete sharing; universal bounded-read, growth/shrink and post-close replacement controls remain. Windows native acceptance is still required.
- Evidence and worktrees were retained under the recorded storage budget instead of deleting them; this preserves failure proofs and user work at the cost of retained scratch storage.
