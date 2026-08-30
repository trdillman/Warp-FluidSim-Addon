# Calibrate paddle spin-up acceptance

## State

ready-for-agent

## What to build

Replace the moving-collider analyzer's explicitly uncalibrated fixed angular
momentum margin with a measured, documented acceptance envelope. The current
real 96-frame, 158,220-particle bake passes, but the source admits that the
threshold is a placeholder.

## Acceptance criteria

- [ ] Measure repeated real `paddle_stir` bakes across the supported frame,
      resolution, and particle-count envelope.
- [ ] Derive and document a threshold from those measurements rather than
      retaining an unexplained magic number.
- [ ] Add a negative or degraded-motion case that proves the check rejects
      missing spin-up and still accepts valid scale variants.
- [ ] Remove the `UNCALIBRATED` qualification and rerun
      `python tools/analyze_demos.py paddle_stir` with a recorded `10/10`
      result.

## Blocked by

None - can start immediately
