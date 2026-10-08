# Worker brief — lane <LETTER> <name>

Fill every field. Dispatch the whole file as the subagent's prompt. Works for Sonnet 5.5
(Claude Code `Agent`, `isolation: "worktree"`) and GPT-6.1-Sol (Codex) lanes alike.

## Goal
<One sentence: the user-visible outcome of this lane when merged.>

## Spec
Read `docs/specs/3.0/SPEC-<name>.md` in full. In your first progress message, restate
the "Trap this spec prevents" section in your own words. Do not start editing before
you have done that.

## You own
<Exact file list from the spec's "Owner module" line.>
You may create new files only under those directories or where the spec names them.

## You do not touch
- `properties.py` unless listed above
- `tools/run_production_gate.py`, GUI gate runs, GPU benchmarks (orchestrator only)
- any `.blend` under `resources/` unless the spec says "demos re-save"
- other lanes' files; if you need a change there, stop and report the exact change

## Environment facts you need
- Repo root: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`;
  package `fluidSim_warp_blender51/`.
- Blender 5.2 portable; headless runs need `--python-use-system-env`,
  `PYTHONNOUSERSITE=1`, and a private `BLENDER_USER_EXTENSIONS` or they can wipe the
  user's Warp install. Check for "Warp 1.17.0 initialized" in the log.
- Tests: `python -m pytest tests/<area> -q` from repo root; your lane's areas: <list>.
- Commit inside your worktree only, Conventional-Commit subject, body states what the
  spec's acceptance items you satisfied and which you did not.

## Order of work
1. Restate the trap.
2. Write the acceptance tests named in the spec first; commit them red.
3. Implement; commit green.
4. Run your test areas plus `tests/repo` and `tests/runtime`; paste exact failures.
5. Report using the format below.

## Stop conditions
Stop and report (do not work around) if: a test outside your area goes red; you need a
file you do not own; a spec statement turns out to be false in the code; the headless
Blender probe fails for a reason unrelated to your lane; two bounded attempts at a fix
fail.

## Report format
```
LANE <LETTER> <name> — <DONE | BLOCKED | PARTIAL>
worktree: <path>  head: <sha>
trap restated: <yes/no, one line>
acceptance: <item>: <PASS/FAIL/NOT RUN> (one line each, from the spec)
tests: <command> → <N passed, M failed>, failures: <exact names>
spec corrections: <anything the spec got wrong about the code>
needs from orchestrator: <files outside lane, decisions, GPU runs>
```
