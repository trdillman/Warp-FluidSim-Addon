# Superseded: synchronize technical demo evidence

## State

superseded

## Superseded by

This combined slice is superseded by
`backlog/tasks/TASK-POLICY-DRIFT-011-DEMO-EVIDENCE-RECURRENCE.md` for the
technical report and
`demos/backlog/tasks/TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`
for the checked-in demo JSON and Markdown. Do not execute this card in
parallel with those narrower owners.

## What to build

Regenerate the checked-in demo JSON/Markdown evidence and the dependent
technical-report wording from the same reproducible run as the parent full
production gate. Keep the parent report explicit about the difference between
current generated artifacts and checked-in review evidence.

## Acceptance criteria

- [ ] `demos/analysis_report.json` matches the current generated analysis artifact.
- [ ] `demos/ANALYSIS.md` reports the same metrics, source run, and 18/18 result.
- [ ] `fluidSim_warp_blender51/docs/technical_report.md` uses the same metrics and run provenance.
- [ ] The parent and nested policy reports cite the synchronized source artifact and remain bounded to its executed scale.

## Evidence

- Current full lane: `rtk python tools/run_production_gate.py --include-gui`
  passed at HEAD `695fad96072e5f00d34cf2c33c132c2a1a896092` with
  `FULL-PRODUCTION-GATE-PASS` and `DEMO-ANALYSIS-PASS (18/18)`.
- Current generated artifact has dam-break front ratio `1.0121877903`, column
  height `0.0852106214 m`, and crown height `0.2748437524 m`.
- Checked-in JSON/Markdown/technical report still have front ratio
  `0.9758125585`, column height `0.0952136219 m`, and crown height
  `0.4082177579 m`; synchronization is unresolved.
- Current `rtk python -m pytest -q` passed `343` tests.

## Blocked by

Superseded; use the two narrower cards above.
