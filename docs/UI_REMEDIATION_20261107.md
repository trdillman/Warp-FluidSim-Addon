# UI fixes and original-findings follow-up — 7 November 2026

All three reviewed UI defects are repaired. The independent original-findings audit confirms source repairs and targeted regressions for R1–R13; it found no additional concrete code defect. Production qualification remains **NO-GO** pending certified native evidence.

## Artist-visible changes

- **U1:** Render Outputs, Render Surface Mesh and Render Particles now describe their rendering purpose and explain viewport independence. The **Liquid Viewport** summary uses the existing display policy, including Preview, Off and missing-surface fallback. Whitewater remains independent. A read-only output lookup ensures panel drawing cannot create/link collections or adopt legacy output IDs.
- **U2:** Sidebar/header/Properties transport and Liquid Deck provide **Exit Cache Playback**. They invoke the existing stop owner and retain baked files. Play/Pause semantics are unchanged; the Bake wheel exposes Exit Cache while cache mode is active.
- **U3:** Sidebar/header show **Cache Playback Error** with bounded actionable detail and a Status & Recovery link. The retained error takes precedence over a benign activity message. Returning to a valid cached frame clears the error through the existing retry/recovery path.

Independent review approved implementation `76c52fd690bff00a4f0a6923d68c060a5543b600`; integration commit is `b47cef1`. The reviewer found no actionable regression in ownership, deferred visibility/export hold, cache lifecycle or sibling UI presentation. Existing saved property identifiers/defaults/callbacks remain unchanged.

## Verification

| Check | Result / boundary |
| --- | --- |
| Final full suite at integrated product/verifier `b47cef1` | **3,746 passed, 345 skipped, 4 existing warnings**, 159.45 seconds; locked Warp 1.19 environment, CUDA hidden, real Blender startup regression enabled |
| Focused UI/runtime/ownership checks | **330 passed**, including all previously failing fixture/map cases |
| Extended real Blender paused-cache acceptance | **36/36 passed**, private profile, Blender 5.2.1; actual RNA, cache publication, output IDs and public stop operator; recorded panel draws, not framebuffer or mouse interaction |
| Configured add-on Ruff | Passed; broader optional scan retains 19 pre-existing findings outside the configured scope, no new findings |
| In-memory syntax | **670 tracked Python files**, no errors |
| Metadata / whitespace | Passed; metadata-only result is not release qualification |
| Production gate at `b47cef1` | Storage pre/end hooks ran; 38 hygiene tests passed. Admission refused missing reviewed `BLENDER_SHA256`; exit 1, no passing production manifest |
| Concurrent integration update | PR #63 was merged externally; fetched `53dbfc4`, whose only delta from its head is three exact retained-log `.gitignore` exceptions. Merged without changing tested product/verifier bytes; actual tracked-ignored-file check passed |

The initial complete UI run failed nine checks: eight memory-recovery UI doubles omitted the newly read `current_error` contract, and the generated ownership map was stale. The doubles and generated map were corrected without removing their original negative/recovery assertions. The passing final suite includes those cases. Test-first UI failures and the retained-error/benign-status failure are preserved as evidence.

Full-suite command from the integration worktree:

```sh
FLUIDSIM_INSTALLER_TEST_BLENDER=/workspace/.tools/blender-5.2.1-linux-x64/blender \
CUDA_VISIBLE_DEVICES=-1 \
/workspace/scratch/fluidsim-review-80bf4aa-20261106/venv/bin/python -m pytest -q -rs
```

## Original findings and release scope

The [R1–R13 audit](ORIGINAL_FINDINGS_AUDIT_20261106.md) maps each original defect to current owners/tests and remaining runtime acceptance. The release-status document now reflects integrated export/whitewater/Deck dependencies and implemented acceptance runners. Contribution instructions target Warp 1.19, the archive plan is marked historical, and release notes distinguish the current unqualified candidate from prior Windows certification.

The supported cache workflow is explicitly one writer per directory. Cross-process locking/refusal is not implemented or claimed. Source integration of optional GPU preview does not qualify its runtime dependencies.

Certified Blender 5.2.0 foreground readability/DPI/input, actual CUDA bake/resume/restoration, approved compiled-emitter numerical comparison, optional GPU preview, exact installed archives and platform campaigns remain open. This host has neither a usable foreground display nor NVIDIA hardware; Blender 5.2.1 background diagnostics do not substitute for those requirements. No release/tag/master merge was performed by this task.

Evidence is retained under `/workspace/scratch/fluidsim-remediation-20261106`: `ui-release-final-tests.log`, `ui-release-production-gate.log`, `ui-release-metadata.json`, `ui-fixes/report.md`, `ui-fixes/evidence-index.json`, `ui-fixes/native-v2/`, `ui-fixes-independent-review.md`, and `original-findings-audit.md`. Original work, failed attempts and all qualification evidence are preserved; no cleanup was performed.
