# Solver completion clock

This adds the completed-state clock required by the frozen FSPT P0 contract.
It supplies a measurement input; it does not qualify realtime performance or
change which images the viewport may retain.

## Boundary and identity

`Session.step` samples host `time.monotonic()` immediately after its successful
`_step_growing_leaves` call returns. The existing solver has completed its
mandatory device synchronization, rigid-state/readback work and frame statistics
at that boundary. The new code adds no device fence, readback, optional surface
extraction or renderer work.

The immutable `SolverCompletion` binds the actual solver object, scene and
domain/output owner, surface epoch, solver frame, liquid-state revision,
simulation time and actually solved timeline frame. `current_solver_completion`
checks these facts without advancing or materializing anything. Moving only the
Blender timeline cursor does not change the captured source frame or timestamp.
A receipt with unknown solved timeline retains that limitation; it does not
establish exact-frame render eligibility.

Each attempted Session solve clears the previous receipt before animated-effect
synchronization or solver work. Failed solves therefore cannot reuse it even
when their public counters remain unchanged. Replacement cutover and runtime
release also clear it. A subsequent optional publication failure preserves the
fact that the solver completed; the existing publication quarantine continues
to decide whether any surface is eligible.

## Publication and measurement

`SurfacePublication.state_ready_at` receives the clock only from a matching live
Session receipt. Admission before a solve, direct solver stepping outside that
Session boundary, mismatched state and cache playback yield `None`. Cache data
does not claim a live wall-clock completion time.

Cadence skips keep the former surface's original clock. A pending synchronization
keeps the exact candidate clock through its later commit. Forcing another
surface build for the same completed state does not refresh its age. The
existing captured-image identity and generic publication serialization carry
the field without relabeling earlier images.

Later performance tooling must use the matched captured publication's
`state_ready_at` and actual presentation time. `published_at` remains a distinct
publication-completion clock. The existing historical-image retention policy
still uses publication time, and its diagnostic age must not be reported as the
P0 completed-state freshness metric. Changing that policy or qualifying the
joint quality/overhead/freshness targets requires separate work.

Viewport presentation diagnostics now carry `solver_state_age_seconds` separately
from `publication_age_seconds`. They use the completed image's captured receipt,
including when a newer live surface exists. A missing or cached completion clock
stays unknown; mixed inputs cannot synthesize a complete clock. Invalid known
clock ordering is rejected. The asynchronous acceptance verifier retains the
raw receipt and measures `solver_state_to_draw_seconds` at independent draw entry.
That endpoint precedes texture upload and completed presentation, so it remains
a lower-bound diagnostic, not the P0 end-to-end freshness result.

## Verification

Host controls cover optional-work delay, same-frame repeated solves, no-output
stepping, solver/scene/domain/epoch/frame/revision/time mismatch, failed solve,
read-only clock access, publication cadence, forced rebuild, delayed sync,
cache/direct-step exclusion and replacement/release. Deliberate mutants that
retain a failed clock, ignore a changed revision, or substitute publication time
each fail their respective controls.

`tests/acceptance/blender_fspt_state_clock_acceptance.py` exercises the real CUDA
solver through `Session.step`, then the real evaluated Blender surface output.
It independently records the optional extraction boundary, injects a test-only
30 ms delay, checks clock ordering and exact state, retains raw surface arrays,
and rejects a deliberately substituted publication clock. It also exercises
same-frame solves, actual direct solver advancement, failure invalidation and
clock queries with GPU/readback/stepping calls forbidden. Native and installed
qualification must be reported separately from the host results.

The native clock lane passed at source
`2893409902e2c1e41e134a0ab8a2ed42dbb0b70d`. Its retained report is
`artifacts/fspt-p0/bridge-qualification/state-clock-2893409902-20261128T042503214331Z/state-clock/acceptance.json`,
SHA-256 `607def3c3b550a2550b37e22aaabd7efc97a9862694b79c0f6fc4535eb45d2c8`.
The earlier `fe9a9627` attempt stopped in its verifier because it treated the
solver's string device alias as a Warp device object. Resolving that alias with
`wp.get_device` corrected the verifier without changing solver or clock behavior.
