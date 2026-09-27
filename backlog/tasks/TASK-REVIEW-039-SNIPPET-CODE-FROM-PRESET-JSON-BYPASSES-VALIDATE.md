# TASK-REVIEW-039: Snippet code from preset JSON bypasses validate_snippet at compile boundary

## State

obsolete

## Severity

low

## What to build

CompiledSnippet._compile() (line 905) generates source from sn.code and writes/loads it without calling validate_snippet(). The validate_snippet function (line 191) enforces a _FORBIDDEN_NODES denylist (no imports, no exec, no function defs, etc.) and an allowed-names check. However, validation is only a UI pre-flight step; it is not enforced in the compile path. A preset JSON dropped into the user snippet directory (custom.py load_user_presets, line 613) can contain arbitrary code in its 'code' field that reaches _compile() without validation. The template structure (textwrap.indent on line 912) currently prevents module-level breakout, but the missing validation at the compile boundary means new template changes could silently introduce an execution path.

**Location:** `fluidSim_warp_blender51/sim/custom.py:905`
**Dimension:** Trust & Safety

## Impact

Defense-in-depth gap. Currently mitigated by template indentation keeping user code inside a Warp kernel body, but the validation boundary is fragile. If the template or indentation logic changes, unvalidated preset code could execute at module level during exec_module.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/custom.py:905` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Call validate_snippet(sn.code, sn.run_over, sn.argument_names()) at the top of _compile() and refuse to compile if any ERROR-level issues are found. This enforces the security boundary at the point of code execution, not just in the UI.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SEC-006` in `reviews/swarm-2026-08-08/findings/security.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Trust & Safety lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Obsolete in 1.1.0. The legacy snippet runtime was retired in commit `b34192e` ("refactor(effects): retire legacy snippet runtime"). `CompiledSnippet` and `SnippetManager` remain in `fluidSim_warp_blender51/sim/custom.py` but have no callers in add-on runtime code (only `tests/test_snippet_library.py`). Even there, `SnippetManager.set_snippets` runs `validate_snippet` and drops ERROR-level snippets before building a `CompiledSnippet`.

User code now goes through Custom Effects v2: `compile_effect` in `sim/effects/compiler.py` calls `validate_effect` and refuses to compile invalid effects, with trust and sandbox checks in `sim/effects/trust.py` and `sim/effects/sandbox.py`.

## Blocked by

None - can start immediately
