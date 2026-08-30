# Synchronize demos unit evidence

## State

ready-for-agent

## What to build

Synchronize this demos unit's checked-in analysis JSON and Markdown with the
current parent production-gate artifact. The saved scenes and live playback
checks pass, but promoted values and the recorded artifact hash are stale.

## Acceptance criteria

- [ ] Use the current parent lane artifact as the source of truth for every
      promoted demo metric.
- [ ] Update `demos/analysis_report.json` and `demos/ANALYSIS.md` atomically
      with the same values and SHA-256.
- [ ] Confirm `blender --background --python tools/check_demo_blends.py` and
      `python tools/analyze_demos.py` still pass after synchronization.
- [ ] Confirm the parent policy-drift validator no longer reports checked-in
      demo evidence drift.

## Blocked by

None - can start immediately
