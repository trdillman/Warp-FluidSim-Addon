# FSPT qualification status

**2026-11-28 08:37 UTC checkpoint.** P0/M0 is complete. Current Windows source
`ce6afc9a` contains the presentation scheduling fix. Its targeted CPU render
suite and provisional package checks passed. Both successor GUI lanes stopped
before exclusive GPU admission. All reports retain their executed identities. The full production gate remains blocked and P2/P3 remain open.

| Scope | Bound outcome |
| --- | --- |
| Current Windows source | Branch `experimental/fspt-p1-20261125`, commit `ce6afc9ae8b607a33b5c429e6743a60a86df8b1e`, tree `4f349bc837b8610b35a39f204f038d2db43c3ea4`. Native freeze `artifacts/fspt-p0/presentation-first-bind-20261128T081859669187Z/source-freeze.json`. The remote branch returned this exact commit at 08:22:26 UTC. [Native freeze observations](../../evidence/fspt-session-20261128/native-ce6afc9a-observations.json) |
| Current frozen inputs | 884 inputs, inventory SHA-256 `0710366113654f45c4756b8a3164df52b22b8f6aa58c67e4d1f90e370c3001a4`. Source fingerprint `894057a2db428085028ab3e6388e42caf4bd209d526dfe989472616c98756943`. Verifier fingerprint `b236a2c95e631a040d9958eb506ea859e5b899223bf71d4fd130fcff2646927d`. [Independent verifier provenance](../../evidence/fspt-session-20261128/review-ce6afc9a.json) SHA-256 `9c84bc64a0410ca1581057dd934ab8c53288661f79540b0f8fb1bb2c7826ade4`. |
| P0/M0 | **COMPLETE** within the frozen runtime, graphics, OptiX and fluid feasibility contract. Original P0 receipts retain their identities. [P0 reconciliation](P0_RECONCILIATION.md) |
| ce6afc9a targeted Windows host | **PASSED**, 441 CPU render checks. Product-file Ruff passed. The touched test has three pre-existing Ruff diagnostics, and its baseline and candidate diagnostic JSON matched byte for byte. This is targeted verification. |
| ce6afc9a standalone native GUI | **ADMISSION BLOCKED.** Both requested lanes queued at position 1, ended with broker exit 75 and produced no lease or acceptance report. The 50-cycle viewport and 240-state/600-second Live case have no successor product result. [Native admission and receipt hashes](#successor-gui-admission) |
| 0607 Windows full CPU host | **VERIFIED**, exit 0, 4,659 passed and 444 skipped, four warnings in 553.84 seconds. Native `artifacts/fspt-p0/host-regression/0607daad9354-external-20261128T071439274723Z/`. `status.json` SHA-256 `0ac737a8564558f0521a81f7c9ae7678abf8890e59ae7d8fb252c19ad35ac6e7`. |
| 5c91 broker and controlled device viewport/worker Final | Both **VERIFIED**, exit 0, under exact `5c91c818…`. Independent 0607 evidence-hygiene rebind confirmed identical product and verifier inputs between 5c91 and 0607. The ce6afc9a scheduler changes two product inputs and one verifier input. [Device qualification and originals](DEVICE_VIEWPORT_QUALIFICATION.md) |
| 0607 standalone GUI | Static viewport **VERIFIED**, 50 cycles under clean exclusive lease `d99ecdc67770`. Live soak **FAILED**, exit 1 under clean exclusive lease `f265e23c1ffe`. The initial 240 distinct images completed. Seven new states and six changed images were recorded in the soak before the unchanged 10-second progress guard fired. [Standalone evidence and original screenshot captions](ASYNC_VIEWPORT_QUALIFICATION.md#exact-0607-standalone-native-viewport-and-live-trials) |
| 0607 provisional package | **PROVISIONAL_VERIFIED** for the standard builder, Linux archive construction/inspection, Windows generic fresh installed double smoke and uninstall, and CPU-only Warp. [Package receipt](../../evidence/fspt-session-20261128/provisional-0607-receipt.json) SHA-256 `7e2be1f8fac9c9c79dc57de49e73ad8e3986454ef2c6ffc47ffa9b3fa471d502`. Windows ZIP `9dc97824b62b99556c87df92d6336d47522c4a33e5c7f66814413b03fd440207`; Linux ZIP `85c79518a75546a8c4b967c44cbec8b6c19e6d65f4801cb6c26c8232bf17ea6e`. Exact 0607 installed FSPT controller did not launch. |
| ce6afc9a provisional package | **PROVISIONAL_VERIFIED** at 08:33:13 UTC. The canonical CPU add-on build ran Ruff and 767 tests passed with 7 skipped. Both platform archives contain 202 Python files and nine demos. Two fresh Windows generic smoke processes passed in the private test profile, followed by verified uninstall. Linux scope is archive inspection. [Package evidence and ZIP hashes](#successor-provisional-package) |
| ce6afc9a GPU runtime observation and named installed FSPT | Runtime observation **ADMISSION BLOCKED**, broker exit 75 after a 24-second lease wait. The installed FSPT lanes **DID NOT RUN** because the fresh runtime context prerequisite was unavailable. [Native runtime observation status](../../evidence/fspt-session-20261128/runtime-ce6afc9a-observation-status.json) |
| Full normal Windows production gate and performance budget | **BLOCKED**. Both full-gate attempts ended on exclusive-broker contamination of unproven PID ownership. No qualifying full manifest or fresh standalone normal performance-budget result exists at this checkpoint. |

## Successor GUI admission

The attempt ran from 08:21:22 to 08:25:20 UTC in
`artifacts/fspt-p0/bridge-qualification/gui-ce6afc9ae8-20261128T082122009491Z/`.
The broker reported “queued for exclusive GPU lease, position 1” and “no
exclusive lease granted (state=queued)” for both lanes. Both broker invocations
returned 75, and neither lane wrote a lease, acceptance report, draw rows or
original screenshots. The wrapper’s aggregate status is `FAILED`. Its source
receipts before and after have matching SHA-256
`d75a1e5a0d1f3d51dc2e9820a9293d1cd3633633ab07675afb1f2bc948148fb7`.

| Native file | Evidence |
| --- | --- |
| `status.json` | SHA-256 `400f1334d55b1d0b260e80339f883e3ce710d328715943c425f0d42a81c745d7` |
| `async-viewport/terminal.json` | Exit 75 at 08:23:21 UTC, SHA-256 `a33d83c3b89369c4b9a156d1a74cd43be36959c647e112e5c0e932b893005ceb` |
| `async-live/terminal.json` | Exit 75 at 08:25:18 UTC, SHA-256 `be9012089e54ddea49919231cc0b534b412ec935e9db044c09b3846762798941` |
| Both `stderr.log` files | Identical, SHA-256 `daa71082ae97d1f1e3a81296bc2deaa6d070dcdc75d000e4fc12899f6cf3371a` |

[Native admission and remote push projection](../../evidence/fspt-session-20261128/successor-gui-admission-and-push.json)

The next successor run still needs the named 50-cycle viewport and 240-state,
600-second Live acceptance. This attempted run requested the same 250 ms
retained-age and 10-second progress guards. Its operational Live deadline was
1,200 seconds with a 1,230-second lease request. The canonical production gate
command and timing remain as specified. The 0607 draw and screenshot data
are the prior measurement; successor timing comparison awaits an admitted
native run.

## 0607 failure and the ce6afc9a change

All 240 initial 0607 Live observer rows were retained images at 525 × 298.
During uninterrupted playback, 22.910 seconds elapsed between accepted solver
frames 15 and 143. The separate soak later failed. The original screenshots
labeled live-240, reopen-live and soak-000 show the purple waiting framebuffer
after earlier observer acceptance. The paused-refinement original alone shows
the visible fluid in that four-PNG set. [Caption and hash details](ASYNC_VIEWPORT_QUALIFICATION.md#original-0607-desktop-captures)

A bounded 0607 per-generation diagnostic traced 47 late first draw entries.
Each image was eligible under the 250 ms age limit at collection, then one
following-generation capture occupied Blender's main thread before that draw.
The median capture took 109.13 ms, including 55.86 ms of independent audit.
The median collection-to-draw interval was 113.88 ms, with no
intervening solver callback in those late cases. The diagnostic retained
originals and did not function as acceptance. Its trace SHA-256 is
`9812f253d722ab52d305daa01753db66c435b4a86e454e44224f01b97ed94cbb`.
[Raw diagnostic trace](../../evidence/fspt-session-20261128/native-080500-trace.json)

The successor gives a newly accepted eligible evaluated Live image one timer
turn to draw before the next capture. It still checks scene, input, source,
camera and publication compatibility, and the deferral expires on the next
admitted timer turn if no draw occurs. The fix leaves the independent audit,
retention bound and image-progress guard intact. A successor native GUI verdict still requires an admitted run on the exact ce6afc9a source.

## Successor provisional package

The Windows controller finished at 08:33:13 UTC with
`PROVISIONAL_VERIFIED` and disposition `PROVISIONAL_FULL_GATE_BLOCKED`.
The canonical builder exited 0, checked compilation and Ruff, and
reported 767 add-on tests passed and 7 skipped in 40.92 seconds. The Windows
extension was installed in a private profile and two separate CPU-only smoke
processes (PIDs 27704 and 43464) passed. Uninstall was verified. Warp 1.19.0
reported CUDA unavailable for this CPU lane. Each ZIP contains 202 packaged
Python files and nine demos. The Linux archive was inspected on Windows.

| ZIP | Bytes | Independently recomputed SHA-256 |
| --- | ---: | --- |
| `fluidsim_warp-4.1.0-windows_x64.zip` | 8,483,340 | `6c07df0d034789e41b989c5473765e8393f314079e71acee5785e766713eac7b` |
| `fluidsim_warp-4.1.0-linux_x64.zip` | 8,483,342 | `22b2d89e69afea972c5ecfd415dabcfae4a21b8ad2ffc29f651b4a0fa5ef75c0` |

The native output is
`Documents/FluidSim-FSPT-ce6afc9a-20261128T082930Z/package/`.
The [exact provisional receipt](../../evidence/fspt-session-20261128/provisional-ce6afc9a-receipt.json)
hashes to `7849553ff2e8474ae7b072c16169518e5402e149416cbd0d602af534ae3d94e0`.
The canonical builder's release-ready field records its generic packaging
scope. Strict production qualification, GPU renderer acceptance from the
installed Windows ZIP and native Linux execution retain separate requirements.
[Builder report](../../evidence/fspt-session-20261128/provisional-ce6afc9a-builder-report.json) ·
[Windows smoke report](../../evidence/fspt-session-20261128/provisional-ce6afc9a-windows-smoke-report.json) ·
[Independent read and hashes](../../evidence/fspt-session-20261128/provisional-ce6afc9a-independent-observations.json)

## Runtime observation and installed FSPT

The exact-source runtime observer requested its own exclusive lease and stopped
with broker exit 75 after a 24-second queue allowance. It produced no Blender
environment report or independently observed runtime context. Its native
[status receipt](../../evidence/fspt-session-20261128/runtime-ce6afc9a-observation-status.json)
hashes to `339407832530d37debad01ecd346038d309e84043057fe7fbbb048c6b33038c7`.
The named installed FSPT lanes did not start. The package’s two generic CPU
smoke processes are the fresh Windows installed checks in this checkpoint.

## Full gate attempts

Paths are below native `artifacts/fspt-p0/`. The 5c91 pre-rebind hygiene attempt is a separate historical result.

| Exact 0607 lease | Outer and inner status directories | Observed stop |
| --- | --- | --- |
| `d62204b8ff4b` | `exclusive-production/0607daad93-20261128T072026585325Z/` and `production-qualification/0607daad93-20261128T072027912267Z/` | Failed about 07:23:02 UTC. Unattributed PIDs 32960 and 52820. Cleanup confirmed by process absence. |
| `2414e127b80c` | `exclusive-production/0607daad93-20261128T072802222163Z/` and `production-qualification/0607daad93-20261128T072804448041Z/` | Failed about 07:30:39 UTC. Unattributed PID 28720. Owned cleanup exit 0. |

The guard decisions and failed attempts remain part of the record. The source and verifier rebind does not convert either attempt to a passing release gate.

## Outstanding acceptance

Keep the frozen joint 24 fresh presented states/s, simulation/wall ≥1,
100/250 ms p95/worst freshness, 1.05 solver interference, 80% combined VRAM,
1080p quality/throughput and separate optical, export and installed gates.
A report from the standalone evaluated fixture supplies bounded native
evidence, with the measured stages and routes recorded explicitly.
[Acceptance contract](P0_ACCEPTANCE_CONTRACT.md)
