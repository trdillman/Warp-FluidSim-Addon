# Post-3.0 review candidate - 2026-10-10

Base: 04d732a13038cab058c6ead088ccc6b5caa7d77d. Merged to master on 2026-10-10 as
cc53ab5, followed by the PCG stall fix (2945240) and an Append-acceptance gate fix
(aeba03a). Unreleased: no tag or publish yet.

## Merge verification (2026-10-10)

At aeba03a: full suite 2,530 passed, 2 skipped, 1 xfailed. One
`tools/run_production_gate.py --include-gui` run: FULL-PRODUCTION-GATE-PASS,
DEMO-ANALYSIS-PASS (21/21), PERFORMANCE-BUDGET-PASS, and `demo_artifact_sha256`
`8baa886c018245f3ea9187933b7d4108891a79998de861da24578e113efbbaf3`, the 3.0.0
reference unchanged. The Append acceptance first ran in the gate here: as
committed in cc53ab5 it raised TypeError before any check (fixed in aeba03a,
14/14). The sections below are the candidate's own pre-merge record.

## Implementations

Append lane: 7e6d0c671f9a060d213dfd1c332412ccc46287da (tag `archive/post30-append-20261010`) - `docs/POST_3_0_APPEND_MIGRATION.md`.
VDB lane: 88f177f9b20c7080e592eb8dc74978e8d3f1388d (tag `archive/post30-vdb-20261010`) - `docs/VDB_CACHE_EXPORT.md`.
Diagnosis lane: 57047ebaf56372ed8f6e6d5286669d33b80989a9 (tag `archive/post30-diagnosis-20261010`) - `docs/JACOBI_COLUMN_DIAGNOSIS_20261010.md`; the diagnostic summary/history JSON are on that tag, and the probe is in master as `tools/diagnostics/diagnose_resting_column.py`.
This review branch contains the combined Append/VDB implementation snapshot; its production solver/default/version/manifest sources are unchanged from the base.

## Verification

- Full Python regression with CUDA hidden: **2,296 passed, 226 skipped** (212.73 seconds). Skipped GPU tests are not counted as runtime passes.
- Focused gate/producer/export/import contract slice: 28 passed, 3 skipped.
- Final real Blender 5.2: **14/14 Append checks and 12/12 synthetic-cache VDB checks**. The OpenVDB reader produced moving-sphere meshes at correct world bounds (maximum 0.002105 error at voxel size 0.05).
- Real ProductionBakeBackend, Warp 1.17.0, RTX 4080 SUPER: two committed frames with **9,032 particles each**, cached origin captured from the actual solver grid, public VDB export, independent Blender OpenVDB read-back.
- A fresh Blender process with every Warp import rejected and no domain object exported and read that same real cache successfully.
- The unchanged 3.0 cache reader read both new frames, retaining optional origin metadata. No format-number change.
- New Python files pass Ruff. The selected modified pre-existing files have the same 18 Ruff findings as the baseline; none were introduced. They were not silently bulk-reformatted or repaired.
- Changed Python files compile; git diff --check passed. The actual DCC-tested implementation hashes are recorded in candidate-source-hashes.json.

## Diagnosis, not a solver patch

(Since fixed in master by 2945240; see the note at the top of the diagnostic report.)

The Jacobi-preconditioned PCG column collapses because the common stall rule stops valid non-monotonic progress at iteration 25. Increasing the limit from 200 to 800 does not help; postponing that checkpoint in a diagnostic control holds the column. Full-band particle count remains 49,152. The same predicate exists in v2.0.0 and v2.1.0 source; those binaries were not replayed. See the full diagnostic report for scope, true-residual controls and qualifications. No production termination policy or default was changed.

## Review limits and retained failures

This is self-reviewed work; the permitted agent routes could not dispatch an independent reviewer. It is not an installed-extension qualification, a foreground file-browser review, or a full GUI production/release gate. Do not tag/publish from this evidence alone.

Failed baseline controls, the unlinked-transform first attempt, missing acceptance-marker/gate integration and updated producer-double failures, and both launcher timeouts remain preserved. All introduced test failures were corrected; no assertions or release thresholds were weakened. A canonical launch readiness timeout and a profile-check timeout were followed by the existing isolated production-gate runner; only owned processes/profiles were used.

The candidate's raw evidence folder (`F:/FluidSim-post30-20261010/evidence/`:
logs, probe JSON, source hashes and the retained failed attempts) was retired
on 2026-10-10 after the merge and is no longer available. The figures above are
the record of it. The Jacobi summary and history JSON survive on tag
`archive/post30-diagnosis-20261010` under `docs/evidence/post30/`.
