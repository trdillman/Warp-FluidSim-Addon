# Codex Windows and context-mode: high-value prevention changes

Date: 2026-07-10

## Live result

`ctx_doctor` is healthy on this machine: context-mode v1.0.169, Codex CLI
0.144.1, SQLite/FTS5, server smoke, and registered hooks all pass. Do not
change context-mode's MCP registration, hooks, or environment globally based
on this investigation.

The local `C:\Users\Tyler\.codex\tools\context-mode-health-check.ps1` is
stale: it treats two removed launchers as required and fails before exercising
the active plugin-managed server. This is a diagnostic defect, not evidence
that context-mode fails to start.

## Recommended changes, ordered by value

### 1. Replace the stale health check with one plugin-aware manual action

Keep it manually invoked after plugin/config changes; do not run it as a
startup hook. It should perform, in order:

1. `codex mcp list` and `codex mcp get context-mode`.
2. `ctx_doctor`.
3. One real `ctx_execute` smoke call.

It must check the current plugin-managed configuration, not hard-code a
historical launcher. This removes false failures while catching the real three
boundaries: discovery, server initialization, and tool invocation.

### 2. Add a short first-use recovery rule to the global AGENTS.md

Add this exact rule near the machine-local routing guidance:

> First context-mode/MCP failure: run `ctx_doctor` once, then one minimal real
> tool smoke. If either fails, retain the exact error and active server/config
> identity; do not retry the same call. Use the native tool path for the task
> and repair the server separately.

This prevents early-turn retry loops without adding a hook or running a costly
preflight for every task.

### 3. Add a Windows shell-lane rule to the global AGENTS.md

> Use PowerShell syntax only in a native PowerShell terminal. context-mode's
> shell sandbox is Git-Bash/MSYS; for a PowerShell cmdlet invoke
> `pwsh -NoProfile -Command ...`, and never mix Windows drive paths or Bash
> heredocs across the two shells.

This is a small, high-frequency prevention rule. It targets malformed shell
commands rather than masking them with retries or longer timeouts.

### 4. Make Windows stdio MCP configuration explicit, only where needed

For each proven-problematic stdio server, use an absolute executable, explicit
`args`, explicit `cwd`, and server-scoped `env_vars`/`env`. Forward only what
the server needs. At minimum, trace whether `SystemRoot`/`windir` is present;
for npm, Docker, or `.cmd` launchers, also consider `ComSpec`, `APPDATA`,
`LOCALAPPDATA`, `PROGRAMDATA`, and `USERPROFILE`.

Do not set global full environment inheritance pre-emptively: it broadens
secret exposure and hides which server actually has an inherited-environment
defect. When a `.cmd` shim is required, use an explicit `cmd.exe /d /s /c`
wrapper rather than relying on bare `npx`, `node`, or PATH resolution.

### 5. Bound startup impact per server

- Use `startup_timeout_sec = 30` only for a server with measured cold-start
  work such as native dependency loading or first-run `npx` installation.
- Keep the default timeout for normal servers; raising every timeout turns a
  fast launch error into a slow conversation start.
- Keep optional services non-required. Set `required = true` only where a
  task cannot continue without that service.

## Do not add

- No blanket `[shell_environment_policy] inherit = "all"`.
- No all-session MCP health hook.
- No duplicated launcher or duplicate MCP ownership.
- No unconditional timeout increase.

## Evidence and sources

- Live 2026-07-10: `ctx_doctor` passed context-mode server, FTS5/SQLite,
  hooks, and Codex CLI resolution (v1.0.169 / 0.144.1).
- Live 2026-07-10: the standalone health script failed because
  `context-mode-codex-win-launcher.mjs` and `context-mode-launcher.mjs` no
  longer exist, despite the active plugin server being healthy.
- Codex MCP config supports `command`, `args`, `cwd`, `env`, `env_vars`,
  startup timeouts, and required-server behavior: [MCP docs](https://learn.chatgpt.com/docs/extend/mcp),
  [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
- Windows child-environment loss causing npm/Docker MCP startup failures is
  documented as issue evidence, not an official root cause:
  [openai/codex#4180](https://github.com/openai/codex/issues/4180),
  [openai/codex#3311](https://github.com/openai/codex/issues/3311).
- Windows MCP startup is eager and can amplify optional-server latency:
  [openai/codex#15947](https://github.com/openai/codex/issues/15947).
- For app-only Windows process creation failures, use a no-profile PowerShell
  probe first and the documented sandbox fallback; do not wipe `.codex` state:
  [Windows sandbox guide](https://learn.chatgpt.com/docs/windows/windows-sandbox),
  [issue evidence #13917](https://github.com/openai/codex/issues/13917).

## Completion criteria for an implementation pass

1. The health action passes without legacy-launcher assumptions.
2. A fresh Codex session has one successful `ctx_execute` call.
3. Each edited stdio MCP proves `initialize` and one real tool invocation.
4. The exact error is retained when a server still fails; optional failures do
   not block unrelated work.
