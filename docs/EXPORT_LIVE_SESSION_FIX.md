# Export live-session preservation correction

Base: PR #30 at 295c4cdf291cb96d02b3ded75f9e7eae315aed0a.
Review finding: discussion_r4278308803.

## Cause and behavior

Shared-path defect: both Alembic and USD enter cache playback through
_SceneState. Playback releases the authoritative singleton Session; restoring
mode flags cannot reconstruct particle or rigid-body state. Turning Live off
only pauses that Session. Cache-only VDB export does not use this path.

The correction refuses Alembic/USD export whenever Session.active is true,
including paused and foreign-scene sessions. It checks at planning and again
before constructing restoration state, so direct callers and plans retained
across the file picker cannot bypass it. Refusal performs no mode transition
or restoration. The error explains the data-loss risk and offers a separate
Blender instance, or an explicit switch to Cache Playback if the artist intends
to discard live state. Inactive/cache-only export behavior is preserved.

## Verification

- Red control: original code failed all eight active-session cases; 12
  inactive success/cancellation/exception cases passed.
- Focused host suite: 105 passed, including 28 lifecycle tests. Uses real
  Session ownership and Playback transitions with fake Blender IO. Includes
  paused/foreign sessions, retained plans, repeated export, operator refusal,
  successful completion, cancellation, exceptions, partial setup failure,
  scene restoration, and return-to-live admission.
- Full CPU suite on Blender Python 3.13 and Warp 1.18.0: 2,886 passed,
  339 skipped. Collected before the final four operator-report cases;
  those are included in the later 105-test focused run. CUDA was hidden.
  An earlier system-Python/Warp-1.17 attempt was stopped and is not qualification.
- Real Blender 5.2.0 LTS cached exports: Alembic and USD each exported twice
  in one session, checked restoration after each export, and passed file
  read-back geometry/animation assertions. CUDA disabled for this lane.
- Real background Blender/CUDA lifecycle acceptance: EXPORT_LIVE_SESSION_OK,
  RTX 4080 SUPER, Warp 1.18.0, 16,352 particles, one Dynamic collider.
  Twelve refused calls across both exporters and Live true/false preserved
  particle positions, complete rigid state, Session identity/key/frame and
  cache-file hashes. The same solver then advanced again.
- Foreground Blender/CUDA lifecycle repeat also passed the mandatory marker,
  exited cleanly, and reported background=false with the same particle/body
  counts. Initial foreground attempts were stopped or failed teardown;
  the final fixture disables Live before editing settings and schedules a
  timer-based quit after startup (the callback returns None).
- Independent standards and correctness reviewers found no blocking defect.
  Their return-to-live/owner/cursor coverage gaps were corrected and re-reviewed.
- Ruff and git diff --check passed. Production-gate inventory coverage: six passed.

## Reproduction and boundaries

The real CUDA check is tests/acceptance/blender_export_live_session_acceptance.py
and is registered in BACKGROUND_ACCEPTANCE with its mandatory success marker.
Use the repository's isolated production-gate environment with the certified
Warp runtime on PYTHONPATH. The normal --quick gate includes this check.
The focused development command is:

    python -m pytest -q tests/runtime/test_interchange_lifecycle.py tests/runtime/test_scene_bridge_cache_playback.py tests/cache_io/test_usd_export_capacity.py fluidSim_warp_blender51/tests/test_interchange_source.py

All Blender processes used private config/scripts/extensions/data/temp paths.
Evidence logs remain in the task worktree's ignored artifacts directory.
This is not a full release/production gate. Direct solver stepping is not
timeline-driven playback qualification. Actual file-picker ESC cancellation was not exercised. Exporter cancellation
and injected exceptions are covered at host seams, not by manual GUI operation.

## Retained evidence hashes

- artifacts/export-final-focused.log: SHA-256 214b3fa6fffecddc9dc6796ee338daae0322c4585fc7aa1e2565e808fceb2274
- artifacts/pytest-certified.log: SHA-256 96e47f222fe03d204c236eebcdda0968612984732390d8c2be5559adb20d443a
- artifacts/blender-export-final.log: SHA-256 04e5b748b597aa66bf92666583802268f7a7ca0d493548690ede4fe78964c0f9
- artifacts/blender-live-background2.log: SHA-256 1c0323d7a9d168066c6d02a051e75cee9b93c16456c9588bafe3503c1b8d1746
- artifacts/blender-live-foreground-final.log: SHA-256 e82c6b1582bff19b98b4795e91a81e6c6e38fba03a20a559462fda1fdc889c4a
