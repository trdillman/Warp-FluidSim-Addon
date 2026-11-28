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
