# Native transport and PRK integration acceptance

Verified 1 November 2026 on Windows, RTX 4080 SUPER, driver 610.88,
Blender 5.2.0 build `fbe6228777e7`, Vulkan and Warp 1.18.0.
The opt-in GPU Particle Preview uses the private SDK's GPU texture transport
and the adapted external PRK `ExternalTexturePoints` API, which invokes its
existing `PointsStrategy`. Private dependency source and wheels are not in this
public repository. The SDK adaptation is private PR
https://github.com/trdillman/blender-vulkan-interop/pull/2.

## Results

- `acceptance-report.json`: fresh managed Blender run passed. A real Quick Liquid
  solver produced 5,984 particles; real timeline handlers advanced scene/solver
  frame 1 to 13, positions changed, and installed PRK drew the SDK frame.
  Paused disable restored the owned output, re-enable resumed drawing, and
  release restored output with no pending retirement. Explicit diagnostic
  position readbacks belong to the verifier, not the new transport.
- `SOURCE_FINGERPRINT.json`: all 239 candidate add-on Python files byte-matched
  the managed installed copy used by the final run; the verifier is hash-bound.
- `sdk-consumer.json`: zero, three, 4,095, 4,096, 4,097 and 4,194,304 positions
  passed texture correctness and completed close; known RGB samples matched.
- `prk-framebuffer.json`: the actual installed PRK strategy drew the three known
  red positions; a background negative-control pixel remained transparent black.
- `missing-prk.json`: real missing-package control left the solver working and
  authoritative particle output visible, with no GPU owner falsely reported.
- `active-load-fixed.json`: save with a live transport, reopen with persistent
  output visible, no retained/pending transport, then reinitialize and draw again.
  `active-load-before-save-fix.json` retains the failure that exposed serialized
  temporary visibility. `roundtrip.json` is the earlier inactive-owner check.
- `timeline-proof.json` and `live-timeline.png`: additional real timeline run and
  visually inspected output with the source displayed as wire. This earlier
  probe mixed manual steps and timeline steps; use the final acceptance report
  for aligned scene/solver frames.
- Worker focused FluidSim regression before the final load/save fix: 202 passed,
  two skipped. The final owner/registry/registration suite passed 62 tests.
  Independent final parent check: 35 passed (`parent-cpu-review.log`). SDK host suite: 156 passed.
  PRK host subset: 56 passed, 18 skipped; Blender-only PRK tests cannot collect in
  plain host Python. Those counts do not replace native acceptance above.

Only the integrated point-preview path is qualified here. Other PRK modes and
arbitrary platforms/builds are not certified by this run. Persistent output
publication remains unchanged and still gathers host data for Blender geometry,
rendering and exports. No full-path readback elimination, performance improvement,
production-gate pass, release qualification, merge or publication is claimed.

## Reproduce the real workflow

Install the matching private SDK `0.1.0a3.dev0`, PRK distribution
`0.1.0+fluidsim.1`, and Warp 1.18.0 into a dedicated directory named
`site-packages`, using the same local user as the managed launcher. Do not modify
the regular Blender profile or replace modules in an already-running process.
The installed PRK API reports base version `0.1.0`; its wheel/source hashes identify
the adaptation. Example paths below identify the verified host, not universal
installation locations.

Call the canonical Blender MCP `launch` with these overrides:

```json
{
  "executable": "C:/Users/Tyler/Downloads/blender-5.2.0-windows-x64/blender-5.2.0-windows-x64/blender.exe",
  "gpu_backend": "vulkan",
  "port": "auto",
  "profile_root": "C:/workspace/fluidsim-native-interop-20261101/blender-profile-clean",
  "addon_sources": [{"module": "fluidSim_warp_blender51", "path": "C:/workspace/fluidsim-native-interop-clean-20261101/fluidSim_warp_blender51"}],
  "addons": ["fluidSim_warp_blender51"],
  "python_dependencies": [
    {"module": "warp", "path": "C:/workspace/interop-acceptance-20261101/site-packages", "version": "1.18.0"},
    {"module": "blender_vulkan_interop", "path": "C:/workspace/interop-acceptance-20261101/site-packages", "version": "0.1.0a3.dev0"},
    {"module": "prk", "path": "C:/workspace/interop-acceptance-20261101/site-packages", "version": "0.1.0"}
  ],
  "no_window_focus": true,
  "working_directory": "C:/workspace/fluidsim-native-interop-clean-20261101",
  "wait_for_ready_sec": 60
}
```

Require a fresh empty managed scene and one window. If the copied private profile
opens its temporary Preferences window, close only that owned window (screen
name `temp`), preserving the main window. Through the managed listener run:

```python
import runpy
probe = runpy.run_path('C:/workspace/fluidsim-native-interop-clean-20261101/tests/acceptance/blender_gpu_particle_preview_acceptance.py')
probe['start'](
    'C:/workspace/fluidsim-native-interop-20261101/blender-profile-clean/acceptance-report.json',
    'C:/workspace/fluidsim-native-interop-20261101/blender-profile-clean',
)
```

Read the report until `status` is `passed` or `failed`, with a 90-second deadline.
Close only the owned Blender process afterward. The probe validates its profile
root before creating objects and never saves preferences.

## Failure history and retained limits

Earlier attempts correctly failed closed on a checkout-normalized ABI profile,
unqualified GPU context, or unavailable dependency. Runtime testing exposed a
paused-disable visibility bug and invalid close-from-draw scheduling; both were
repaired before the final run. Those initial observations were tool outputs and
were not saved as standalone failure receipts. They are not relabelled as passes.
The private SDK PR documents the exact-byte profile repair and preserves its
integrity anchor. The global production gate and full installed release/package
  qualification were not run for this experimental integration PR.

The final review also found that file load could cancel a nonpersistent
retirement timer. A persistent, distinct load-pre handler now retires owners
before window destruction. Active-save testing then exposed persisted temporary
visibility; save-pre/post handlers repair that shared lifecycle path. The final
active-load and reusable acceptance reports cover both fixes; earlier acceptance
and source fingerprints are retained explicitly as before-fix history.

The broader host regression attempt was blocked by automatic approval review,
which reported an untrusted telemetry connection. Its log stops at 12% and has
no completion verdict. `BROADER_CPU_BLOCKED.json` preserves the stated reason;
the endpoint and source-level cause were not identified. No full-suite pass is
claimed and that connection was not retried.
