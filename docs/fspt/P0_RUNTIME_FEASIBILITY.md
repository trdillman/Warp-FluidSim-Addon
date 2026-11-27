# FSPT P0 runtime and native feasibility

## Decision and evidence boundary

Keep the certified Blender 5.2.0 / Python 3.13 / Warp 1.19 runtime. The existing
Warp CUDA software-BVH backend remains the reference implementation. Qualify
native RTX tracing and GPU image presentation independently before integrating
either into the renderer or making performance claims.

This P0 work supplies an executable target-runtime probe and an isolated native
OptiX probe. Compilation, API presence, triangle tracing, renderer integration,
installed-package behavior and realtime performance are separate results. A
successful triangle trace does not complete the later rendering milestones.
The joint quality/compatibility/performance requirements remain in
`P0_ACCEPTANCE_CONTRACT.md` and
`tests/render/fixtures/fspt_acceptance_contract_v1.json`.

## Runtime distribution review

On 27 November 2026, the retained official Windows archive and checksum file
were re-read from the reviewed distribution record at
`D:\Codex\fluid77-5775f3b2-nov21\production-reviewed\blender-provenance.json`.
The installed executable and bundled Python bytes also match the corresponding
members inside that archive. The retained distribution record itself hashes to
`25c3748f9268d278dedf19254469edd7622d5731f45bf53b3d0ee0ed270215d6`.

| Item | SHA-256 |
| --- | --- |
| Official Blender 5.2.0 Windows x64 ZIP | `2d184b626c001692c362291911293b6a297179d618d95e9e9192c3a80318adc4` |
| Retained official checksum file | `f35709c2eb91fbb58ebbd354285039df62217e6dbda9c3a4713ec113d728057f` |
| Installed and archived `blender.exe` | `0060916d6921eb4d46c57254609d805a2ea711917399051391a52ba14beb6327` |
| Installed and archived bundled `python.exe` | `7d984ee02e70c22f04d5e26f1328865d75a5da14b677890896a294f35815d37a` |

The archive source recorded by the reviewed distribution is the
[official Blender archive](https://download.blender.org/release/Blender5.2/blender-5.2.0-windows-x64.zip),
with the [official checksum list](https://download.blender.org/release/Blender5.2/blender-5.2.0.sha256).
This revalidation preserves the existing runtime policy. The old record's
source/verifier fingerprints belong to its old candidate. They must not be
reused as approval of this session's verifier. The final frozen candidate needs
its own reviewed binding through the existing production qualification code.

## Executable API and graphics probe

`tests/acceptance/blender_fspt_p0_runtime_acceptance.py` runs in the actual
Blender process and records its executable, build, Python, add-on source tree,
Warp payload, CUDA device, driver module and private profile paths. Its complete
mode requires a real Blender window; `--core-only` explicitly marks graphics
`NOT_RUN`.

| Boundary | Execution and retained evidence |
| --- | --- |
| RenderEngine | Actual `begin_result`, `update_result`, progress and `end_result` on a generated 16×16 linear RGBA ramp; RNA signatures retained. The ten required pass names execute `register_pass`, `add_pass`, and synthetic pass-buffer write/read checks through the actual `update_render_passes` callback. This proves API transport, not FSPT path semantics or multilayer export. |
| Camera / scene | Actual camera projection-matrix call; evaluated/depsgraph and region APIs inspected; four authored `.blend` files hashed and appended without activating or solving them. |
| DOF / motion blur | Actual camera DOF and render motion-blur RNA plus authored values for dam break, crown splash, paddle stir and chromatic vortex. API availability does not imply renderer support. |
| Image and color | Independently reopened 32-bit EXR must preserve the linear ramp within `1e-6`; encoded 16-bit RGBA PNG must match analytic Standard/sRGB values within `2/65535`. The fixture is opaque; premultiplied/straight alpha conversion and zero-coverage compositing remain separate required tests. |
| UI ownership | Real main-thread timer, `view_update` and `view_draw` callbacks; private splash setting, event-simulation availability and launch flag recorded. |
| Blender graphics | Actual `GPUTexture` RGBA32F host upload, exact texture readback, shader/batch draw into `GPUOffScreen`, and checked readback within `1e-6`; platform backend/vendor/renderer/version recorded. |
| Display | Public display-space shader bind/unbind is executed; viewport transform parity is not inferred from this alone. The screenshot and raw RGBA fixture are retained. |
| CUDA | Actual Warp kernel result, pinned-host copy, checked event completion, native context/stream association and nonblocking flag. Stream destruction must succeed. |
| Native ABI | Optional, explicitly bound DLL/PTX pair executes the OptiX test described below through `ctypes.CDLL` in Blender's existing Python. |

The public GPU API inventory records whether the actual texture type exposes
any named native-handle candidates. It does not reinterpret an opaque Python
address as an OpenGL/Vulkan/CUDA resource. Public Blender API references:
[GPU module](https://docs.blender.org/api/5.2/gpu.html),
[GPU types](https://docs.blender.org/api/5.2/gpu.types.html), and
[image textures](https://docs.blender.org/api/5.2/gpu.texture.html).
Runtime execution, including failure receipts, remains the acceptance evidence.

The current `blender/fspt_scene.py` explicitly rejects authored depth of field.
Its camera snapshot contains a single projection/view matrix and no shutter
interval or temporal geometry samples. DOF and motion blur remain required
Final endpoints; the Live preset may disable them. This inventory preserves
those requirements instead of treating their absence from a small fixture as
proof of support.

## Bounded RTX route on the current Python ABI

NVIDIA's [warp-optix repository](https://github.com/NVIDIA/warp-optix) advertises
Python 3.12 wheels and describes the package as alpha. That advertised binary
route does not establish compatibility with this Blender's Python 3.13. The
probe therefore uses a small C ABI without CPython headers or an extension
wheel. This is a feasibility route, not an adopted production backend.

The locally retained build inputs are OptiX SDK 9.0.0, CUDA Toolkit 12.9.2
(`nvcc` 12.9.86), and Visual Studio 2022 Build Tools with selected MSVC
14.39.33519 and Windows SDK 10.0.26100.0. No toolkit, SDK, wheel or driver is
downloaded or installed. `tools/fspt/build_optix_probe.py` creates a fresh
artifact directory, pins source/tool/header/import-library hashes, saves both
compiler logs and return codes, and binds the resulting DLL/PTX/notices hashes.
Its `VERIFIED` result means CPU compilation only.

The native test uses the existing Warp CUDA context and a checked nonblocking
stream. It builds one indexed triangle GAS, compiles ray-generation/miss/hit
programs, creates the SBT/pipeline with computed stack sizes, and traces two
rays. The required output is exactly `(t=1, u=.25, v=.5, hit=1)` for the central
ray and `(-1, -1, -1, 0)` for the miss. The result is copied to pinned host
storage after checked completion and saved as a NumPy artifact. A wrong-context
negative control must fail before creating an OptiX context or enqueuing work.

Geometry, SBT, launch parameters, modules and acceleration allocations remain
owned until the supplied stream retires. Every native cleanup result is
checked. A failed retirement quarantines the native resources for the lifetime
of the disposable probe process. The probe includes completed host uploads of
small control records and makes no asynchronous-preparation or performance
claim.

The compiled helper notices accompany every probe build in
`OPTIX_PROBE_NOTICES.txt`. The SDK and driver are not bundled. This preserves
the observed helper attribution and does not claim SDK redistribution rights
or installed-product qualification. A product route still needs an explicitly
reviewed dependency/package contract and support strategy.

## Execution and remaining integration gates

The CPU build can run independently:

```text
python tools/fspt/build_optix_probe.py --out-dir <fresh-owned-artifact-directory>
```

Run the committed Blender script only through the existing automatic admission
route, with a bounded owned lease, `--factory-startup`, and private absolute
`BLENDER_USER_CONFIG`, `BLENDER_USER_EXTENSIONS` and `BLENDER_USER_SCRIPTS`.
Pass `--json-out <fresh-report>` and, for RTX, `--optix-build <build.json>` after
Blender's `--`. Do not force a graphics backend. One GPU job runs at a time.

The full script emits `FSPT-P0-RUNTIME-PASS` only after the actual API, file,
graphics and CUDA assertions complete. Its report preserves an explicit
`NOT_RUN` if no native build is supplied. Compile success or core-only success
must not be promoted to full runtime success.

Before adopting RTX, prove dynamic GAS/IAS lifetime and rebuild/refit policy,
the same optical/material contract, cancellation and device admission, bounded
memory, solver interference and installed packaging. Before adopting GPU image
sharing, prove actual graphics/CUDA device matching, supported external image
and semaphore handles, resize/backend teardown and display color parity. Until
then, completed pinned-host RGBA presentation is the explicit fallback.

Native results for the final candidate are recorded separately in the session
evidence manifest. This document specifies the reproducible probe and its
limits; it does not replace the existing production-gate receipt.
