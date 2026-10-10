# Post-3.0 local review candidate â€” 2026-10-10

Base: 04d732a13038cab058c6ead088ccc6b5caa7d77d. Unreleased; no master/tag/publish or production-profile installation.

## Implementations

Append lane: 7e6d0c671f9a060d213dfd1c332412ccc46287da â€” `docs/POST_3_0_APPEND_MIGRATION.md`.
VDB lane: 88f177f9b20c7080e592eb8dc74978e8d3f1388d â€” `docs/VDB_CACHE_EXPORT.md`.
Diagnosis lane: 57047ebaf56372ed8f6e6d5286669d33b80989a9 â€” `docs/JACOBI_COLUMN_DIAGNOSIS_20261010.md`; the executable probe and diagnostic summary/history JSON are on that separate branch.
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

The Jacobi-preconditioned PCG column collapses because the common stall rule stops valid non-monotonic progress at iteration 25. Increasing the limit from 200 to 800 does not help; postponing that checkpoint in a diagnostic control holds the column. Full-band particle count remains 49,152. The same predicate exists in v2.0.0 and v2.1.0 source; those binaries were not replayed. See the full diagnostic report for scope, true-residual controls and qualifications. No production termination policy or default was changed.

## Review limits and retained failures

This is self-reviewed work; the permitted agent routes could not dispatch an independent reviewer. It is not an installed-extension qualification, a foreground file-browser review, or a full GUI production/release gate. Do not tag/publish from this evidence alone.

Failed baseline controls, the unlinked-transform first attempt, missing acceptance-marker/gate integration and updated producer-double failures, and both launcher timeouts remain preserved. All introduced test failures were corrected; no assertions or release thresholds were weakened. A canonical launch readiness timeout and a profile-check timeout were followed by the existing isolated production-gate runner; only owned processes/profiles were used.

Evidence root: `F:/FluidSim-post30-20261010/evidence/`. Key files: `review-full-cpu-green.log`, `final-combined-acceptance/gate_probe.json`, `production-bake-vdb-retry/report.json`, `warp-free-real-cache/gate_probe.json`, `old-3.0-reader.log`, `jacobi-diagnostic.json`, `jacobi-summary.json`, `candidate-source-hashes.json`, `protected-source-checks.json`. Keep the original failed attempts with the final results.
