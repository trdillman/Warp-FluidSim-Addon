# Retire redundant FluidSim checkouts

## State

ready-for-agent

## Goal

Revalidate and safely retire the clean Phase-4 landing worktree, then optionally archive and remove two stale unregistered physical checkout directories. Preserve all explicitly classified experiments.

## Safe registered candidate

- `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_phase4_commit`
- Branch: `phase4/live-interaction-merge-20260728`
- Review snapshot: clean; zero commits absent from `master`; approximately 243 MB.
- Re-run live ancestry/status checks before `git worktree remove`. Retain the branch ref unless separately authorized.

## Manual archive-then-delete candidates

These are unregistered directories whose `.git` files point at worktree metadata that no longer exists:

- `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_first_checkout`
- `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_viewport_viz`

The review found July 4–5 snapshots, mostly regenerable artifacts, totaling about 699 MB. Git cannot protect them because they are already orphaned. Archive unique source/docs with manifests and SHA-256 checksums before filesystem deletion.

## Explicitly excluded experiments — do not touch

- `codex/native-gpu-interop`
- `codex/native-gpu-interop-main-20260715`
- `claude/particle-surface-rendering-apis-f62356`
- `worktree-stflip-implementation` (locked)

Do not merge, extract, prune, unlock, or clean these as part of this task. Keep `worktree_archives\2026-07-29` intact.

## Acceptance criteria

- [ ] Phase-4 checkout is still clean and fully contained in current `master`.
- [ ] Orphan directories are inventoried and archived before deletion.
- [ ] Archive manifests/checksums are verified.
- [ ] All four experiments and existing archives remain untouched.
- [ ] Cleanup report records reclaimed paths, retained refs, and exact verification commands.
