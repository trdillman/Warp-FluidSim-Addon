# UI and release follow-up implementation plan

> **For agentic workers:** Use subagent-driven development for the UI implementation and independent review; root owns integration and release records.

**Goal:** Address UI findings U1–U3 and reconcile every original finding R1–R13 against the integrated candidate.

**Architecture:** Preserve viewport/render separation, Playback retry semantics, and existing cache-stop lifecycle ownership. UI adapters display authoritative state and invoke existing operators. No solver, cache format, or certification policy change is intended.

**Tech Stack:** Existing Python/Blender add-on; host pytest and private-profile Blender diagnostics. Certified release target remains Blender 5.2.0/Warp 1.19.0.

**Spec:** [UI findings and story matrix](../../UI_WORKFLOW_REVIEW_20261106.md) and [original findings](../../PRODUCTION_READINESS_REVIEW_20261106.md). User requested implementation of both sets of findings.

## Global constraints

- Preserve original work and retained evidence; isolate edits from PR63 acceptance-only changes.
- Use the current shared owners; do not make viewport changes mutate render/export choices.
- Background RNA/recorded draws do not establish foreground GUI acceptance.
- No CUDA, numerical, final-package or platform qualification claims without execution.

## Review focus

- Surface/preview/off and absent-surface fallback must match the actual viewport policy.
- Cache exit must remain available while paused and without Warp, retain cache files, and preserve generic Play/Pause semantics.
- Missing/corrupt cache errors must be visible in sidebar/header; valid-frame recovery must clear them.
- Domain/Scene/Physics/sidebar/Deck sibling controls must remain coherent.
- Original fixes must be traced to current owners/tests rather than inferred from PR titles.

## Tasks

- [x] UI specialist: add failing regressions for the three reviewed stories; fix labels/summary in properties.py and blender/ui_scene.py/ui_stages.py using liquid_display policy, expose existing cache_playback_stop in panels.py/ui_scene.py/deck/controls.py, and display existing activity error details in panels.py/ui_header.py. Verify affected sibling drawers and actual private-profile Blender cache/recovery paths; commit.
- [x] Independent original-findings audit: map R1–R13 to current code, tests and remaining native acceptance. Escalate any concrete unfixed code defect for implementation.
- [x] Root: correct stale Warp contribution example, mark the archive plan historical, document single-writer cache scope, reconcile changelog and remediation status with current source/evidence.
- [x] Independent reviewer: inspect the complete UI diff and tests; route confirmed issues to implementer and re-review fixes.
- [x] Root: integrate commits, run configured lint, full regression suite and mandatory production gate attempt; retain failed admission and native limitations. Prepare a separate draft PR targeting the integration branch now that #63 is merged, with findings disposition and verification evidence.
