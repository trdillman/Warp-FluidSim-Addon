# Worktree Consolidation Evidence — 2026-08-10

## Authority

- Target branch: `master`
- Frozen baseline: `a5f99edcf48de8eb42930a07faeb401cb31f9779`
- Integration source: `codex/architecture-resolution-ui-integration-20260807` at `0e257b6b4b4cb454de70eb71d076a412e08bd11e`
- Recovery archive: `liquid_warp_blender51_port_worktree_evidence_20260810`
- Recovery bundle SHA-256: `ae90466d893d78f348a36f288fda82c311df72648562c90e98cc97daeaee5feb`
- Curated ignored evidence: 87 records, 85 unique files after SHA-256 deduplication
- Baseline Python suite: 564 passed and four pre-existing system-Warp CUDA conditional-graph failures

The sibling archive contains the verified Git bundle, baseline worktree and
branch inventories, the complete baseline test receipt, a machine-readable
curated evidence manifest, and `SHA256SUMS.txt`. Reproducible demo caches, NPZ
frames, Python/pytest caches, agent inputs/transcripts, and build outputs were
excluded.

## Branch disposition

### Preserved

- `master`
- `codex/native-gpu-interop` — archived experiment, dirty worktree preserved
- `codex/native-gpu-interop-main-20260715` — archived experiment, dirty worktree preserved
- `claude/particle-surface-rendering-apis-f62356` — archived experiment preserved
- `worktree-stflip-implementation` — archived experiment preserved

### Merge source and contained branches

- `codex/architecture-resolution-ui-integration-20260807` — sole merge source
- `codex/architecture-deepening-20260805` — contained by the merge source
- `codex/merge-resolution-quality-20260805` — already merged into `master`
- `feature/resolution-quality-standardization` — already merged into `master`
- `phase4/live-interaction-merge-20260728` — already merged into `master`

### Patch-equivalent branches to retire

- `agent/res-quality-t01-source-truth`
- `agent/res-quality-t03-whitewater-truth`
- `codex/diagnose-simulation-slowdown`

### Superseded branches to retire without cherry-picking

- `agent/res-quality-t02-session-capacity`
- `codex/audit-corrections`
- `codex/orbit-st-flip-free-surface-completion`
- `codex/preserve-master-dirty-20260703-044839`
- `codex/review-and-refactor-my-code`

These branches encode older capacity bypass/live-update behavior, beta
ST-FLIP state, recovery snapshots, or demo-routing implementations superseded
by the current fail-closed capacity, production-gate, and runtime contracts.

## Removal gate

No listed branch or worktree may be removed until the consolidated tree passes
the focused regression lanes, reproduces no new full-suite failures relative
to the frozen baseline, passes the GUI-inclusive production gate and policy
evidence validator, and the sibling recovery bundle verifies successfully.
