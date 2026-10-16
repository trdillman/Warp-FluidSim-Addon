# Completed b29b968 performance evidence

**Aggregate FAIL / regression.** Crown regression, Dam order disagreement/inconclusive, Paddle and Vortex pass. This is the bounded b29b968 versus 3.2.1 comparison, not qualification of current master or the full 4.0 release.

- [Full completed report](REPORT.md)
- [Aggregate paired-block reducer output](aggregate-budget.json)
- [Counts, iterations and noise summary](final-summary.json)
- [Explicit user noise-policy decision](noise-policy.json)
- [Source/runtime/input provenance](provenance.json)
- [Copy/hash verification](copy-verification.json)
- [Durable raw artifact directory](../../../../artifacts/performance-20261015-completed/)
- [Full source-to-destination SHA-256 manifest](../../../../artifacts/performance-20261015-completed/copy-manifest-sha256.json)

Absolute raw root: `F:/FluidSim-performance-fix-20261015/artifacts/performance-20261015-completed`.
The directory is intentionally ignored by Git under the established `/artifacts/` convention. Retain it as protected gate evidence when archiving this worktree; it is not disposable cache.

`rerun/evidence-vortex-quiet/combined-runs.json` contains the final 64 matched reports. `rerun/evidence-attempt4/correction-runs.json` contains the eight correction reports. All original rejected/superseded trials, prelaunch failures, per-trial monitor receipts and logs remain under `rerun/`; the frozen before/after/reference source trees, saved blends and original diagnostics are copied under `original-investigation/`. `frozen-settings/` preserves the four input payloads.

Earlier completed demos retain the original process-presence monitor. The fresh 16-trial Vortex set uses Tyler's approved quiet-work-machine policy: background readings at or below 10% before/after trials, with full-interval activity logs and explicit WDDM attribution limitations. The background allowance is unrelated to the unchanged 1.10/1.05 candidate/baseline performance ceilings. The previously rejected Vortex trial remains rejected.

Copied runner scripts are historical execution provenance. Do not blindly rerun their embedded task paths; new measurement requires an explicit new owned evidence root and the pinned source/input identities. Documentation closeout performed no new GPU tests or product changes.
