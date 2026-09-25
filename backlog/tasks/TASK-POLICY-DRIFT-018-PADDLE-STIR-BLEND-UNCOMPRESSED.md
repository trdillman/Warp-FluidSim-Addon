# Re-save the paddle_stir demo blend compressed

## State

done (2026-09-24)

## What to build

`demos/paddle_stir.blend` is the only demo blend in the repository stored
uncompressed, which makes it 20x larger than the file the builder emits and
3.4x larger than the next-largest demo.

Diagnosed 2026-09-22 at HEAD 4303e00, not inferred:

- checked-in file: 5,731,753 bytes, header `42 4c 45 4e 44 45 52` (`BLENDER`)
- default rebuild via `tools/build_demo_scenes.py -- paddle_stir`:
  280,220 bytes, header `28 b5 2f fd` (Zstandard)
- `demos/dam_break.blend`, `demos/crown_splash.blend` and
  `demos/chromatic_vortex.blend` all carry the Zstandard magic

The divergence is *not* a content difference. Both files were inspected
headlessly and hold the same datablocks: 9 objects, 5 meshes, and the same
`Paddle Stir Pool Seed` mesh of 158,220 vertices and 0 polygons (the seed
`make_particle_seed` builds at `tools/build_demo_scenes.py:1740`, matching the
158,220 particles the shipped settings simulate). The only inventory
difference is three extra text datablocks. Compression accounts for the whole
20x.

The uncompressed save appears to date from `bb6d3fc` (2026-08-22, "prepare
Blender 5.1 release 1.0.255"), where the file jumped 3,040,951 -> 5,731,753
bytes. It has never been near the builder's 280 KB.

This is a repository-weight and reproducibility issue, not a correctness one:
the demo loads, plays back and passes the gate as-is.

## Acceptance criteria

- [x] Re-save `demos/paddle_stir.blend` compressed, from the builder rather
      than by hand, so the checked-in file is the file the builder emits.
- [x] Confirm the result carries the Zstandard magic `28 b5 2f fd` and that no
      other demo blend is uncompressed.
- [x] Confirm the rebuilt scene still holds the 158,220-vertex
      `Paddle Stir Pool Seed` and still reports
      `saved live-playback-ready frame 1 with 158,220 particles`.
- [x] Run `python tools/analyze_demos.py paddle_stir` and record
      `DEMO-ANALYSIS-PASS (10/10)`.
- [x] Run `python tools/run_production_gate.py --include-gui` and record
      `FULL-PRODUCTION-GATE-PASS`, then re-run
      `python tools/validate_policy_drift_reports.py`.
- [x] Decide whether the builder should assert compression on save so the
      divergence cannot silently recur, and note the decision here.

## Resolution (2026-09-24, RTX 4080 SUPER, Blender 5.1.0)

Rebuilt with
`blender --background --factory-startup --python tools/build_demo_scenes.py -- paddle_stir`
and no `FLUIDSIM_PADDLE_*` variables set:

- `demos/paddle_stir.blend`: 5,731,753 -> 206,335 bytes, header `28 b5 2f fd`.
  All four demo blends now carry the Zstandard magic.
- Builder log: `paddle_stir: saved live-playback-ready frame 1 with 158,220 particles`.
- `demos/paddle_stir/meta.json` is byte-identical to the committed file; the
  preview PNG re-rendered (156,113 -> 156,004 bytes).
- `python tools/analyze_demos.py paddle_stir` -> `DEMO-ANALYSIS-PASS (10/10)`
  with spin-up ratio 0.9437 and steady circulation 0.17241, the same values
  the `b5e0050` recheck measured (see TASK-POLICY-DRIFT-016).
- `tools/check_demo_blends.py` -> `DEMO-BLEND-CHECK-PASS`.
- `python tools/run_production_gate.py --include-gui` ->
  `FULL-PRODUCTION-GATE-PASS` (GUI 53/53, `DEMO-ANALYSIS-PASS (18/18)`,
  demo artifact sha256 `a32872b8...250926`, unchanged from the previous gate).

Decision: the guard lives in the checker, not the builder. The builder
already saves compressed; the drift came from a hand save outside it, which
a builder assertion would never see. `tools/check_demo_blends.py` now fails
any demo blend without the Zstandard magic (`saved_compressed`), and the
production gate runs that checker on every demo.

## Blocked by

None - can start immediately. Note that a default rebake overwrites
`demos/paddle_stir.blend` and `demos/paddle_stir_preview.png`; both are
committed, so `git checkout --` restores them.
