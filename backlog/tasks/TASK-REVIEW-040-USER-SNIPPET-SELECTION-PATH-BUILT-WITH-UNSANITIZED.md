# TASK-REVIEW-040: User snippet selection path built with unsanitized enum suffix

## State

ready-for-agent

## Severity

low

## What to build

On line 1109, the path is constructed as os.path.join(user_snippet_dir(), selection[4:]) where selection is a string like 'USR:filename.json'. While the enum items are built with os.path.basename (properties.py:272), a crafted .blend file could set the RNA enum value to 'USR:../../malicious.json'. The subsequent re-validation via load_user_presets (line 1110-1112) mitigates this because load_user_presets only lists files directly in user_snippet_dir(), so a traversed path would not match. However, the traversal path itself is constructed before validation.

**Location:** `fluidSim_warp_blender51/operators.py:1109`
**Dimension:** Trust & Safety

## Impact

Currently mitigated by the found-list re-validation check, but the unsanitized path construction is a fragile defense-in-depth gap. A refactor that removes the re-validation step would expose the traversal.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:1109` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Sanitize selection[4:] with os.path.basename before joining, or validate that the resulting path is within user_snippet_dir() using pathlib.Path.resolve() and relative_to().
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SEC-007` in `reviews/swarm-2026-08-08/findings/security.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Trust & Safety lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
