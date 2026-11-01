# Permanent ST-FLIP retirement — 1 November 2026

Source base: `502ca66e6372ea7874da77ed38f72659abc12e61` (authenticated GitHub master).
Branch: `codex/retire-stflip-20261101`. Scope: documentation and project policy only.

The owner permanently removes ST-FLIP from global project scope. The ordinary
FLIP/APIC solver remains in scope. The implementation, settings and UI had
already been removed in 3.0 (recorded in RELEASE_3_0_HANDOFF.md, lane D,
`6ce6b33`). No solver repair or new implementation removal is required here.

## Inventory and disposition

| Location | Disposition |
| --- | --- |
| AGENTS.md and README.md | Permanent retirement; native interop/rendering reactivation with fresh qualification, no merge/release authority |
| docs/RELEASE_4_0_PLAN.md | Remove future ST-FLIP revival proposal; retain research citation as historical |
| docs/RELEASE_4_1_PLAN.md, row 23 | Retire item, preserve row numbering |
| docs/RELEASE_4_1_PLAN.md, row 25 | Reflect owner's interop reactivation; dependency/runtime qualification remains open |
| Package technical_report.md | Label 1.0.255 CUDA solver receipt historical, including removed ST-FLIP capability |
| CHANGELOG, migration guides, 2.1/3.0 handoffs/plans, archive tags | Retain historical release and removal evidence |
| tools/performance_budget.py and tests/gate/test_performance_budget.py | Retain removed-setting compatibility for historical release comparisons; not production settings |
| sim/particles.py time_offset comment, ownership-map historical comment | Retain explanation of public cache/effects channel and past setting removal |
| Ordinary FLIP/APIC source/tests | Unchanged |

A bounded tracked-file inventory before host disconnection found no remaining
production ST-FLIP implementation or UI. Matches were historical records,
retirement commentary, benchmark compatibility and future revival documentation.
No foreign experiment directories, source history, archives or packages deleted.

## Verification and limits

Exact-once text-target checks passed for the proposed changes. Surrounding
AGENTS.md content is preserved; no unrelated local tracker header is copied.
Every changed file is Markdown. No source, tests, scenes, version or packages
are changed. Runtime tests and production/GPU/GUI gates were not run and are not
needed to validate this documentation-only change; no new runtime acceptance
is claimed. GitHub commit/readback validation is recorded in the PR.

The Windows host became unavailable during local preparation. A clean isolated
clone/branch at the base SHA was verified before disconnection. A later local
mutation command failed to start with an exec-server transport error; its effects
are unverified and are not used as this candidate's source. The candidate is
constructed from exact base file contents through authenticated GitHub tools.
The dirty canonical checkout and its original work are not this PR's source.

Storage startup check ran before host disconnection and returned exit 0 with
bounded/incomplete telemetry and warnings; no cleanup authority was inferred.
Raw host telemetry could not be recovered durably after disconnection. No gate
was admitted. This small committed report is retained as PR audit evidence.
No merge, release, tag or product publication performed. Tracker reconciliation remains
owned by the parent agent, separate from this PR.
