#!/usr/bin/env bash
# Executor-session runner for the visc-surface-tension-opt autonomous task.
# Usage: bash .autonomous/visc-surface-tension-opt/run-session.sh <session-number>
set -uo pipefail

PROJECT_ROOT="C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port"
SKILL_DIR="C:/Users/Tyler/.claude-desktop/plugins/cache/claude-code-settings/autonomous-skill/1.0.1/skills/autonomous-skill"
TASK_DIR=".autonomous/visc-surface-tension-opt"
SESSION_NUM="${1:-1}"
LOG="$TASK_DIR/session-${SESSION_NUM}.log"

cd "$PROJECT_ROOT" || exit 1

EXECUTOR_PROMPT=$(cat "$SKILL_DIR/templates/executor-prompt.md")
TASK_LIST=$(cat "$TASK_DIR/task_list.md")
PROGRESS=$(cat "$TASK_DIR/progress.md" 2>/dev/null || echo "No previous progress notes")

read -r -d '' GUARDRAILS <<'EOF'
=== NON-NEGOTIABLE PROJECT GUARDRAILS ===

The worktree is DIRTY with ~24 unrelated in-progress UI/effects/demo files that
belong to the user. PRESERVE ALL OF THEM.
  - NEVER run: git stash, git reset, git clean, git revert, or `git checkout --`
    on any file you did not yourself edit in this session.
  - NEVER run `git add -A` or `git add .`. Stage explicit paths only.
  - NEVER merge or cherry-pick from the branch `fix/settings-apply-domain-object`.

There is ONE GPU (cuda:0). Never run two benchmark or profiling processes
concurrently. All benchmarks are strictly sequential.

Do NOT change DEFAULT_VISCOSITY_TOLERANCE or DEFAULT_VISCOSITY_ITERATIONS.
That is a physics decision reserved for the user.

EVIDENCE STANDARD (this project's core requirement):
  - Recompute every headline number from raw JSON on disk. Never quote a number
    you did not compute yourself.
  - No speedup may be claimed without a PAIRED before/after benchmark, both
    sides run from one binary via a mode flag, sequentially, on the same GPU.
  - The benchmark's demonstrated trial spread is +/-5 %. An effect smaller than
    that CANNOT be claimed. Refuting an item is a valid, valuable result.
  - Label conclusions Verified / Likely / Assumed.
  - If a command cannot run, report the exact command and its exact error.
    NEVER estimate, extrapolate, or fabricate a measurement.
  - Correctness outranks speed. A faster SDF that silently truncates the
    curvature support band is a regression, not an optimization.

The full test suite baseline is 1402 passed. Run `python -m pytest -q` before
marking any task done. Do not commit a red suite.

Work through the task list IN ORDER. Complete tasks thoroughly rather than
starting many. Update task_list.md checkboxes and append to progress.md before
you finish, so the next session can resume without re-planning.
EOF

claude -p "Continue working on the task.
Task Directory: $TASK_DIR
Project root: $PROJECT_ROOT

$GUARDRAILS

Current task_list.md:
$TASK_LIST

Previous progress notes:
$PROGRESS

$EXECUTOR_PROMPT" \
  --max-turns 120 \
  --dangerously-skip-permissions \
  --append-system-prompt "You are the Executor Agent for the visc-surface-tension-opt task. Complete tasks in order and update task_list.md and progress.md in $TASK_DIR. Obey the project guardrails in the prompt absolutely: preserve the dirty worktree, never stage files outside the explicit list, never run two GPU benchmarks at once, and never claim a speedup without a paired measurement." \
  > "$LOG" 2>&1

STATUS=$?
echo "session $SESSION_NUM exit=$STATUS"
echo "--- progress ---"
TOTAL=$(grep -c '^\- \[' "$TASK_DIR/task_list.md" 2>/dev/null || echo 0)
DONE=$(grep -c '^\- \[x\]' "$TASK_DIR/task_list.md" 2>/dev/null || echo 0)
echo "$DONE/$TOTAL tasks completed"
exit $STATUS
