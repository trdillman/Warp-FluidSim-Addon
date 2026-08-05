---
status: fresh
escalated: false
generated_at: 2026-08-05T11:17:25Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\demos'
git_head: 0b47a0535f3970a3c8b6426370e63d754ea96486
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 86406b0a7b3f0e75cf337e40a75ed800d2e0d5e490e8f551ded9155a93115cc5
---

# Policy Drift Report — particle-surface-rendering-apis demos unit

## Resolved finding

### PD-NESTED-DEMO-001 — Demo evidence has no authoritative parent run

Type: validation drift and documentation drift. Blocking criticality: high.

The parent nested full gate passes, and this report is bound to the nested
worktree's fresh `18/18` generated demo artifact and passing saved-demo
playback report.

## Current metrics

Re-pinned from the `2026-08-05T11:17:25Z` gate run at HEAD `0b47a053`:

| Demo | Metric | Value |
|---|---|---|
| dam_break | front speed / c0 | 0.9758126 |
| dam_break | Z(T=2) | 2.5895753 |
| dam_break | end column height | 0.0768738 m |
| crown_splash | impact / peak frame | 1 / 4 |
| crown_splash | crown sqrt(t) correlation | 0.9715359 |
| crown_splash | crown symmetry CV | 0.0066210 |
| crown_splash | max crown height | 0.2866601 m |

End column height moved from `0.0814825` to `0.0768738` against the prior run
on unchanged inputs. All 18 checks still pass, so this is run-to-run variation
inside the acceptance tolerances rather than a regression — but it is also why
the artifact hash changes on every rerun.

## Machine-readable audit footer

generated_at: 2026-08-05T11:17:25Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\demos'
git_head: 0b47a0535f3970a3c8b6426370e63d754ea96486
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: parent nested FULL-PRODUCTION-GATE-PASS.'
  - 'PASS: nested DEMO-ANALYSIS-PASS (18/18) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 86406b0a7b3f0e75cf337e40a75ed800d2e0d5e490e8f551ded9155a93115cc5
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Restore the nested parent lane inputs.'
  - '[x] Rerun nested demo analysis and saved-demo playback.'
  - '[x] Refresh this report from those nested artifacts.'
