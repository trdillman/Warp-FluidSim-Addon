# Native acceptance follow-up — 6 November 2026

Based on PR61 candidate d77d4bfbcb7c435a34e983c3e7b01a2a9f1d1974, preserving concurrent integration of exports, whitewater, Deck and optional GPU preview. PR49/51/62 are already ancestors; none was applied twice. No product behavior is changed by this follow-up.

The installer runner now defines seven independent installed-profile scenarios: missing, owned, damaged, foreign preloaded, offline local install, cancellation and repeated clicks. It requires an explicit checksum-locked local wheel and private profiles; it performs no automatic network download. Actual foreground execution remains pending. An actual Blender5.2.1 startup regression confirms native-mode imports reach the intended foreground requirement from an unrelated working directory.

The frozen-input runner now executes six fresh-process actual CUDA ProductionBake phases, with exact checkpoint/frozen-state and history checks, cache-prefix controls, preview finish/cancel/failure cases, and retained numerical differences. It uses the shared device selector and private Blender/Python/cache/temp environment. CPU selection refuses before fixture creation. No numerical acceptance threshold was invented: complete structural execution remains explicitly unqualified until the numerical policy is reviewed. Positive CUDA execution remains unrun on this host.

Independent review found and corrected a native-script sibling import failure and inherited Python/cache state in the CUDA runner. Full current-candidate baseline passed 3697 tests with345skipped. Follow-up verification logs and review reports are retained under /workspace/scratch/fluidsim-remediation-20261106; final results are recorded in the pull request description.

The dedicated [UI workflow review](UI_WORKFLOW_REVIEW_20261106.md) records three unresolved P2 usability findings, a story matrix, actual background Blender/export evidence and explicit foreground/CUDA limits. These findings concern misleading labels/status, cache-to-live discoverability and hidden cache-error details. They have been reviewed, not silently fixed or treated as passing production stories.

Run the documented launcher help for exact invocation and fixture requirements:

- python tests/acceptance/run_installer_preferences_matrix.py --help
- python tools/gate/run_frozen_input_acceptance.py --help

Before release: complete native installer execution, actual CUDA continuation and approved numerical comparison, foreground usability/interaction review, optional GPU-preview qualification, and exact final-archive platform acceptance. No merge, release tag or production publication occurred in this follow-up.
