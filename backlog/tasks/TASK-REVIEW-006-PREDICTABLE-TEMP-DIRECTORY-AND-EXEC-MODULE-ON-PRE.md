# TASK-REVIEW-006: Predictable temp directory and exec_module on pre-existing files enables local arbitrary code execution

## State

ready-for-agent

## Severity

high

## What to build

CompiledSnippet._compile() writes generated Python source to a fixed-name directory os.path.join(tempfile.gettempdir(), "fluidsim_warp_snippets") (line 935) with a filename derived from a SHA1 content hash (line 933-934). The guard 'if not os.path.exists(path)' on line 938 means that if a file already exists at the computed path, the add-on skips writing its own content and proceeds to load and exec_module the existing file (line 948: spec.loader.exec_module(module)). On multi-user Linux systems where /tmp is world-writable and shared, a local attacker can pre-create the directory and plant a file or symlink at the predicted filename. The module name is deterministic for any built-in preset because both the template and the preset code are public in the source tree, allowing the SHA1 hash to be computed in advance. exec_module executes all module-level code in the file, giving the attacker arbitrary Python execution within the Blender process.

**Location:** `fluidSim_warp_blender51/sim/custom.py:935`
**Dimension:** Trust & Safety

## Impact

Local arbitrary code execution in the Blender process on multi-user systems. An attacker who shares a /tmp directory can achieve code execution by pre-planting a file whose name matches a content hash computed from known built-in preset code.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/custom.py:935` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Use tempfile.mkdtemp(prefix='fluidsim_snippet_') to create a private per-compile directory instead of a shared fixed name, or verify file ownership/permissions before loading. Alternatively, write the file with mkstemp (which creates the file atomically with O_CREAT|O_EXCL) and always overwrite rather than skipping when the file exists.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SEC-001` in `reviews/swarm-2026-08-08/findings/security.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the Trust & Safety lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
