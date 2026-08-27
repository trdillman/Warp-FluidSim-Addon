# FluidSim Swarm Review Completion Plan

## Constraints

- Current `master` is authoritative.
- Do not merge the historical species prototype or archived experiments.
- Preserve unrelated dirty files in the original checkout.
- Use RED then GREEN tests for every newly confirmed defect.
- Commit only files owned by this completion slice.

## Tasks

- [x] Reconstruct the canonical 18-claim ledger from commit `9a890eb`.
- [x] Run six independent read-only verification lanes with Luna/high for
  regular lanes and Terra/high for stronger lanes; exclude security.
- [x] Confirm already-integrated ownership, playback, cache identity, export,
  metadata, and finite-value corrections.
- [x] Add RED malformed-rank tests; implement exact source-array validation.
- [x] Add RED permission/type tests; persist and enforce legacy snippet
  permissions and reject unsupported FUNCTION snippets before compilation.
- [x] Add RED pending/visibility/action tests; implement truthful Reset,
  Reinitialize, Hidden, and empty-action behavior.
- [x] Add RED full-schema benchmark and failure-cleanup lease tests; retain
  structured benchmark failures and hold cache ownership through cleanup.
- [x] Repair duplicate-registry test teardown and document export atomicity
  limits.
- [x] Run claim-focused integration: `309 passed`.
- [x] Run full Python regression: `1135 passed`.
- [x] Run Blender Phase 4 live interaction:
  `PHASE4-LIVE-INTERACTION-PASS`.
- [x] Run Blender handler persistence: `HANDLER-PERSISTENCE-GATE-PASS`.
- [x] Run Blender background acceptance: `BACKGROUND-ACCEPTANCE-PASS`.
- [x] Run CUDA cache/export acceptance: `CACHE-EXPORT-PIPELINE-PASS`.
- [x] Resolve independent Terra/high final review to PASS with no actionable
  production finding.
- [x] Commit the isolated correction slice as `fe9c9e4` and integrate it into `master`
  without staging the original checkout's unrelated dirty file.
- [x] Stabilize the documented root pytest import path and preserve loaded
  submodule identity during root-package test reloads in `72733fb`.
- [x] Re-run the authoritative full and Blender gates on integrated `master`:
  `1136 passed`, `PHASE4-LIVE-INTERACTION-PASS`,
  `HANDLER-PERSISTENCE-GATE-PASS`, `BACKGROUND-ACCEPTANCE-PASS`, and
  `CACHE-EXPORT-PIPELINE-PASS`.
