# Export display lifecycle verification

The fix in `c9f92bed` builds on PR #51 (`1aa88b75`). Export-time display requests now remain queued until the outermost visibility hold releases. Loading another file or unregistering the add-on cancels the timer and clears pending scene state.

Three regression cases failed before the fix and passed afterward; the focused preview suite passed 59 tests. Independent source review found no remaining actionable issue.

`report.json` binds the tested source files, Blender executable, dependency root, acceptance scripts and raw logs by SHA-256. All five real Blender 5.2 checks passed: Alembic export, USD export, handler persistence, cache/bake/export pipeline, and export from a live session. Runs used isolated Blender profile directories and Warp 1.19.0. GPU runs held the shared native validation mutex. This is background runtime evidence; interactive GUI acceptance and release qualification remain separate.

The broader source suite on the original PR #51 base plus this fix completed with 3,688 passes, five failures, five skips and one expected failure. The five failures concern missing host-test adapters, an RNA method mistaken for a settings field, and stale generated ownership documentation. Existing PR #56 owns their repairs. Its commits were incorporated without conflict as merge `957ac0a3`; no duplicate repair was written. All nine affected test files then completed with **243 passes and one skip** in 136.42 seconds, including all five previously failing cases. This targeted rerun is distinct from a second complete suite run.

Raw logs remain in this local evidence directory. They are not force-added through the repository log exclusion policy. The initial handler attempt in `../roi-export-20261106/` failed because the test wrapper disabled CUDA; the corrected GPU-enabled check passed and that setup failure is retained.
