# Display-cycle integration review — 3 November 2026

PR #44 head 9782207386757b82f70f3f48a53a6ad19af34fc5 was combined with
PR #46 head 4393bd09123fe8a725fc64ff4a3815f3b37dc6c4 on master
665bd48aaf02522db5e69f014ea0b66f79b3a886. Git auto-merged without conflicts.

Independent source review found two shared-path blockers: legacy particle-only
files migrated to a blank surface view, and paused-cache mode changes could
not recover geometry cleared under render visibility policy. Both were fixed
before merge. Viewport geometry needs are now separate payload facts; render
flags remain authored. Cache publication explicitly retains already-read mesh
and points across modes. Missing mesh falls back to raw points. Preview renderer
failure defers refresh outside drawing; unregister removes the timer.

Legacy particle-only files migrate to Preview. Surface mode with disabled
meshing shows particles and reports the fallback. The Look property inventory
and parity fixtures now describe the display enum rather than the compatibility
Boolean. No solver settings or render/export selections are rewritten.

Final bounded verification (system Python, CUDA_VISIBLE_DEVICES=-1):
- combined-final-focused.log: 747 passed (all UI tests, cache playback and
  pending-settings registry).
- combined-output-contracts.log: 39 passed (publication buffers, publication
  visibility and scene settings).
- final-lifecycle.log: 64 passed (preview, Deck parity and registration
  atomicity), including the additional deferred-fallback lifecycle regression.
- git diff --check passed. Ruff comparison against the combined baseline found
  no new diagnostics; pre-existing diagnostics remain outside this scope.

Independent reviewer re-reviewed the repair and reported no remaining concrete
blocker. These are code/test results, not final-candidate Blender/GPU/framebuffer
acceptance. Owner requested code debugging without additional UI probes. A full
Python-suite attempt was interrupted at 17% while the candidate was still being
repaired; it is not reported as a completed full-suite pass. Earlier failing
iteration logs are retained locally but are not final pass receipts.
