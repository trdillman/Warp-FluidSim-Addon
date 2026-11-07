# FluidSim 4.1 stabilization status — 6 November 2026

The original R1–R13 code repairs are present in the combined candidate. Production release remains **NO-GO** until the exact candidate and installed packages pass certified target-host qualification. This status distinguishes implemented repairs from runtime acceptance; historical test counts are not evidence for later product changes.

The original review covered `80bf4aa`. Master was refreshed to `f8d5b32` while preserving its untracked review/plan documents. The combined PR #61 candidate advanced to `d77d4bf`, including export, whitewater, Deck, visibility/lifecycle and optional GPU preview integrations. PR49/51/62 are already ancestors and must not be applied twice. PR #63 (`276fb87`) adds the installed installer matrix, CUDA continuation diagnostics and dedicated UI review. PR #63 was subsequently merged into the integration branch by another session, advancing it to `53dbfc4` with three retained-log ignore exceptions. The UI follow-up targets that current integration branch; no merge to master, release tag or publication is performed by this task.

## Original findings

| Finding | Repair and review PR | Remaining acceptance |
| --- | --- | --- |
| R1 Preferences controls; R9 productive download retries | Supported icon and retained-progress retry budget; [#55](https://github.com/trdillman/Warp-FluidSim-Addon/pull/55), with #47 | Seven installed foreground scenarios on the final package |
| R2 consumed input identity; R12 verified frame bytes | Resolved dependencies and schema coverage; bounded immutable read; [#58](https://github.com/trdillman/Warp-FluidSim-Addon/pull/58). Later integration includes consumed analytic FIELD volume values | Actual partial-bake/edit/refusal and representative performance on certified hardware |
| R3 frozen inputs; R8 preview restoration | Persisted frozen state; original-frame restoration and failure cleanup; [#59](https://github.com/trdillman/Warp-FluidSim-Addon/pull/59) | CUDA ProductionBake continuation/preview restoration and reviewed compiled-emitter numerical policy |
| R4 qualification identity; R10 performance environment; R11 exact runtime admission | Source/verifier/runtime fingerprints, admission invalidation and exact-runtime/provenance refusal; [#60](https://github.com/trdillman/Warp-FluidSim-Addon/pull/60) | Current-source, final-package GPU/GUI/performance campaign |
| R5 inherited build profile | Private Blender profile and process environment; [#54](https://github.com/trdillman/Warp-FluidSim-Addon/pull/54) | Certified package build/install/profile integrity |
| R6 paused cache display; R7 retirement failure; R13 fallback attributes | Retained publication, coherent cutover and cleanup, independent viewport attributes; [#57](https://github.com/trdillman/Warp-FluidSim-Addon/pull/57) | Installed foreground/CUDA behavior and fault recovery |
| Original failing source checks | Current fixtures, declared-field scan and generated ownership map; [#56](https://github.com/trdillman/Warp-FluidSim-Addon/pull/56) | Re-run on each final changed candidate |

The independent follow-up audit and its source/test references are recorded in `ORIGINAL_FINDINGS_AUDIT_20261106.md`. UI U1–U3 are fixed in the follow-up: see [implementation and final verification](UI_REMEDIATION_20261107.md). The original [UI review](UI_WORKFLOW_REVIEW_20261106.md) is retained as the pre-fix record.

## Acceptance runners and evidence boundaries

- PR #63 defines seven isolated installed-profile installer scenarios: missing, good owned, damaged owned, foreign preloaded, offline local wheel, cancellation, and repeated clicks. The local wheel must match the lock. Actual foreground matrix execution remains pending; the native startup/import regression passed in Blender 5.2.1.
- The six-process CUDA continuation runner uses actual BakeRun/ProductionBakeBackend, exact checkpoint/frozen-state/history assertions, cache-prefix controls and preview finish/cancel/failure cases. It retains numerical differences and deliberately reports unqualified until the compiled-emitter numerical policy is reviewed. Positive CUDA execution has not run on this host.
- At PR #63 source `276fb87` (product/verifier `b80b76f`), the full pinned suite passed **3,731 tests, 345 skipped, 4 warnings**. That is the pre-UI-fix baseline. Missing GPU/GUI cases do not count as passes.
- The dedicated UI review at product `d77d4bf` passed 370 focused host tests and real Blender 5.2.1 background setup/cache/save-reopen and Alembic/USD/VDB readback checks. Recorded layout draws establish bindings and emitted actions/text, not native visual rendering or interaction.
- Existing native evidence from prior PRs is attributable only to its recorded source/runtime. It does not certify this final candidate or optional GPU preview.

## Release sequence

1. Review and integrate the combined fixes and follow-ups while preserving their dependency order. #63 is already merged into the #61 integration branch; the UI follow-up targets that branch. Do not replay older topic commits already present in #61.
2. Freeze the product/verifier commit and exact distribution provenance. Follow [RELEASING.md](RELEASING.md); missing identity must refuse rather than reuse a stale receipt.
3. Run source checks and the certified Blender **5.2.0**, locked Warp **1.19.0**, CUDA, foreground GUI, nine-demo and four-demo performance campaigns. Windows is the existing certification platform; Linux requires its own evidence.
4. Complete installed exact-ZIP installer, bake/resume/re-bake, preview restoration, paused display, export, lifecycle and fault-recovery stories. Qualify the optional GPU preview separately before advertising it as accepted.
5. Keep one writer per cache directory; concurrent writing is unsupported. Cross-process lock/refusal/recovery is not established by process-local guards or atomic file writes.
6. Seal exact source/verifier/runtime/archive digests and the requirements-to-results matrix. Recheck version/tag availability and obtain publication signoff only after qualified artifacts exist. Never replace an existing tag or rebuild allegedly equivalent bytes.
7. Verify consumer download/install after publication, retain rollback packages, and execute the support/rollback procedure in the full production plan.

## Evidence and retention

This host has no NVIDIA device/driver or usable foreground Blender session. Available Blender is 5.2.1 rather than the certified 5.2.0. No synthetic approval record or runtime bypass is supplied to the production gate. Source fixes and diagnostic runs do not close those limitations.

Original evidence is retained under `/workspace/scratch/fluidsim-review-80bf4aa-20261106`; fixes, reviews, failed attempts and follow-up evidence are retained under `/workspace/scratch/fluidsim-remediation-20261106`. Original worktrees and user changes remain preserved. Storage records retain ownership, leases, budgets and retention; no cleanup was performed.
