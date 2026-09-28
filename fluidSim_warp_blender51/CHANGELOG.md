# Changelog

## 2.0.0 — unreleased

### Upgrade notes

- **Open 1.0.x files in 1.1 first.** 2.0 opens files saved by 1.1.0 or later.
  A file saved by 1.0.x is flagged when it opens (a load message and an alert
  on the Setup and Physics tabs): its domain settings were not carried over,
  and a scene whose Custom Warp Snippets 1.1 never converted to Custom Effects
  will not start a simulation. The stored data is kept, so opening and
  re-saving the file in 1.1.0 recovers it.
- **1.0.x caches must be baked again.** A cache written by 1.0.x
  (`fluidsim_*.npz` frames with `fluidsim_sequence.json`) is recognized and
  refused rather than played; free it and bake again.
- **ST-FLIP is Experimental.** Its controls are labelled Experimental, and
  **Variable Projection** is now off by default. A saved ST-FLIP scene that
  never changed Variable Projection runs with it off; turn it back on in
  **Spatiotemporal Processing** to keep the 1.x result.
- **Scripts that write a setting to the wrong group now fail.** Writing a
  domain setting to `scene.fluidsim` (or a scene setting to
  `obj.fluidsim_domain`) used to create a plain Python attribute that the
  solver never read. It now raises `AttributeError` naming the right group.

See `docs/migrating_to_2_0.md` for the full upgrade path.

### Platform

- Blender 5.2 LTS only; Blender 5.1 is no longer supported.
- NVIDIA Warp 1.17.0 is bundled as checksum-locked wheels and installed with
  the extension. An external Warp install is neither used nor needed.
- Every gate lane (pytest, CUDA regressions, Blender background acceptance,
  release validators, the demo bake and the GUI production gate) runs against
  Blender 5.2 and Warp 1.17.

### Fixed

- A production bake with an animated source or collider (such as the
  paddle_stir demo) played back frame 1 and then refused every later frame as
  baked from different scene settings, and resuming it depended on the
  playhead. Playback now compares only the frame's settings signature, and
  bake and resume signatures are pinned to the bake start frame.
- NaN or infinite face weights no longer reach the pressure solve.
- Whitewater collision normals near the edge of a sparse collider now use the
  same gradient helper as the liquid instead of an out-of-range fallback.
- The ST-FLIP Jacobi solve reports its real residual instead of a constant.
- The particles-per-voxel ceiling (64) is one constant instead of six copies.
- **Apply capacity suggestion** can be undone with Ctrl+Z.
- The capacity dialog and its status lines no longer show garbled characters.
- Live Tuning knobs bind to the group that declares each setting, so changes
  made there reach the solver.
- One default per setting: RNA defaults, solver defaults and fallbacks agree,
  and length and speed settings carry Blender units.
- The height-field *Infinite* back mode, which the solver could not run, is
  removed. A file that saved it reports an actionable error.
- GPU self-validations release their solvers.
- An evaluated mesh used for point sourcing is released even if reading its
  vertices fails.

### Changed

- A failed CUDA-graph capture in the pressure or viscosity solve is shown as
  an error (`stat_native_solver_message`) instead of silently running slower.
- Examples 01–04 were regenerated with 1.1 so 2.0 opens them without flagging
  them as pre-1.1 files.
- Demo bakes are bitwise reproducible on the same GPU, driver and Warp build.
  The whitewater collision fix changes the demo reference hash.

### Known limitations

- Only Windows 11 x64 on a single GPU model (RTX 4080 SUPER) has a recorded
  GPU gate run. Linux x64 is packaged from the same source but not certified.
- A cache frame records the simulation settings, not source or collider
  animation. Changing an animation after a bake does not invalidate the
  cache; bake again yourself.
- If an older Warp in Blender's user site-packages is loaded ahead of the
  bundled one, FluidSim refuses to run and names the Warp version it found.
  Remove the old Warp from the user site-packages.
- Speed matches 1.1.0. Against 1.0.x, scenes dominated by particle-to-grid
  transfer run about 4% slower (paddle_stir demo), which is the cost of the
  deterministic transfer 1.1.0 introduced. The other demos run at the same
  speed or faster.
- The reported slowdown after repeated resets with viscosity and surface
  tension on has not been reproduced outside the viewport.

### Removed

- The retired 1.x snippet runtime and its library: example and saved snippet
  presets, the snippet validator and auto-fix, shortcuts and the hidden
  library/editor settings.
- The Custom Warp Snippet settings, the `fluidsim.snippet_apply` operator and
  the snippet-to-Custom-Effects conversion. 1.1 already converted saved
  snippets when it opened a file.
- The 1.0.x settings migration and cache format 1 playback (see Upgrade notes).

## 1.1.0 — per-object domains, a Basic/Advanced interface and deterministic bakes

This is the bridge release to 2.0. Open and re-save your 1.0.x files in 1.1.0:
2.0 will only open files saved by 1.1.0 or later.

### Upgrade notes

- Domain settings now live on the Domain object (`obj.fluidsim_domain`) rather
  than on the scene. Files from 1.0.x migrate automatically when they are
  opened; save them afterwards to keep the migrated form.
- Simulation results differ from 1.0.255. Several GPU steps now run in a fixed
  order (see Changed), the viscosity solver now reaches its tolerance, and
  several demo scenes were recalibrated. Re-bake caches you want to compare.
- Blender 5.1, external Warp 1.16.0, Windows x64 and Linux x64 remain the
  supported contract.

### Added

- **Liquid color mixing** on the GPU, shown in the new `chromatic_vortex` demo.
- **Per-object domains.** Each Domain object carries its own settings, the
  **Add → Fluid Domain** menu creates one, and a scene can hold several. The
  **Physics Properties** tab is the home for domain settings; the Scene
  panels edit the active domain.
- **Basic and Advanced modes.** Basic, the default, shows the settings most
  scenes need; Advanced adds the solver panels.
- **Quality tiers** (Draft, Preview, Final) now scale voxel size, substeps,
  pressure iterations and particles per voxel. In 1.0.x the setting had no
  effect.
- **Transport and Lifecycle panels.** Playback controls and destructive actions
  have their own panels; destructive actions are styled as alerts.
- **Viewport HUD toggle** in the Display panel, and a compact **Live Tuning**
  shelf in the 3D View sidebar that replaces the full settings mirror.
- **Diagnostics** gathers the solver telemetry in one place.
- **Whitewater display fraction** thins the whitewater points drawn in the
  viewport without changing the simulation.
- **Custom Effects** parameters marked runtime-bindable update live, without a
  rebuild. Writing Warp code for an effect is now a developer preference
  (**Enable Effect Authoring**), and trust granted while editing lasts for the
  session only.
- **Artist preview.** One preview path for the viewport, and scene actions that
  roll back cleanly if setup fails.
- **Demo scene contract.** The four shipped demos (`dam_break`, `paddle_stir`,
  `crown_splash`, `chromatic_vortex`) must reopen, initialise and play from
  their own sources, and a check proves each one.

### Changed

- **Deterministic bakes.** Four order-dependent GPU steps (the pressure solver's
  dot product, particle-to-grid binning, shape emission and whitewater
  recycling) now run in a fixed order. Baking `dam_break` or `crown_splash`
  twice on the same GPU gives bit-identical frames. Mesh emitters and color
  mixing still accumulate in arrival order, so other scenes can differ
  slightly between bakes.
- The viscosity solver's iteration ceiling rose from 40 to 64 so it can reach
  its tolerance, and its convergence is now reported in the frame statistics.
- Surface-tension band propagation is the default SDF band mode.
- The legacy snippet runtime was removed. Custom Effects use the v2 pipeline.
- Developer tooling moved out of the add-on package into `tools/`.

### Performance

- Viscosity runs its solver loop inside a CUDA conditional graph. With
  viscosity on, a frame costs 1.55x the baseline instead of 7.80x.
- Surface attributes are gathered on the GPU: a surface frame fell from 175.5 ms
  to 20.7 ms in the benchmark scene.
- Live output copies only the used part of each device buffer (p95 −10%).
- Smaller wins: fused particle compaction counts, no duplicate whitewater
  reduction, and less per-substep launch overhead.

### Fixed

- Presets, live settings and quality tiers now apply to the domain after
  migration. Previously they wrote to the old scene settings.
- Edits to domain settings now show as pending and apply on reset. The export
  attribute toggles are visible again.
- Baking from the command line no longer crashes on exit.
- `crown_splash` whitewater now emits foam; the old threshold was unreachable.
- Emitters keep emitting after the particle capacity grows.
- The surface SDF band is wide enough for smoothing and filtering.
- Malformed user snippet presets are skipped with a message instead of failing.
- Errors now always outrank warnings in status displays.

### Security

- Custom Effects modules are written to a private per-user directory and are
  executed only when their digest matches.
- **Restart** refuses to delete a folder that is not a FluidSim cache.
- Closed a legacy path that could bypass Custom Effects trust.

### Release process

- Every pull request now runs hosted CI: lint, tests, CPU kernel validation, and
  a Blender build, install and smoke test. See `docs/RELEASING.md`.
- The GPU production gate validates the release kernels on CUDA and records the
  FluidSim version in its manifest.

## 1.0.255 — contextual Physics Properties and full viewport workflow

### Added

- Added contextual **Physics Properties → FluidSim** panels for the selected
  domain, generated outputs, and input-role objects without adding a second
  settings model.
- Added the complete Scene/Custom Effects workflow to **3D View → FluidSim**
  beneath one collapsed **Full Settings** parent, while retaining the compact
  activity/status and lifecycle panel as the primary viewport surface.
- Added guarded domain assignment/removal and confirmed single-output removal.
  Destructive output actions revalidate stable tags, ownership, and exact object
  identity immediately before deletion.

### Preserved during integration

- Scene/Object RNA, Pending Settings, Simulation Admission, Playback,
  Production Bake, and Output Publication remain the authoritative owners.
- Blender 5.1+, Windows x64, Linux x64, external Warp 1.16.0, and binary-free
  split-platform packaging remain the supported and validated contract.
- Removed legacy cache controls remain removed; the viewport mirrors only the
  current authoritative Scene Properties workflow.

### Fixed

- Multi-name kernel registration now validates the complete identity set before
  mutation, so duplicate or colliding names cannot leave partial registry or
  source-alias state.

## 1.0.253 — restore Windows x64 extension packaging

### Fixed

- Restored the long-standing Windows x64 product platform in the Blender
  extension manifest and split-platform release output.
- Separated supported product platforms from the subset of external runtime
  artifacts available on the current validation host. A Linux-only supplied
  Warp wheel can no longer silently reduce the extension to Linux-only.
- Added release tests that require both `windows-x64` and `linux-x64`, require
  both split archives, and require CI to retain both artifacts.

### Packaging and validation

- The release builder now emits `windows_x64` and `linux_x64` archives from the
  same binary-free Python/assets source. Neither archive bundles Blender or Warp.
- Windows Blender 5.1/Warp 1.16 execution is proven by the GUI-inclusive
  production gate on an RTX 4080 SUPER. Linux remains supported by the
  previously established runtime proof and the same binary-free source archive.

## 1.0.252 — native Properties workflow and Custom Effects UI integration

### Added

- Added a workflow-ordered Scene Properties hierarchy for Session, Setup,
  Domain, Height Fill, Physics, Custom Effects, Surface, Whitewater, Outputs,
  Cache/Production Bake, and Diagnostics.
- Added canonical role-aware Object Properties panels for Sources, Sinks,
  Colliders, Forces, Field Sources, and Volume Grid payloads.
- Added a compact 3D View status/launcher and a Scene Properties navigation
  action without duplicating authored settings.
- Added a mutually exclusive Start/Resume/Restart production-bake presentation
  layer while retaining strict execute-time verification.
- Added supplied-Blender validation for panel metadata, all authored draw paths,
  ten object-role/geometry combinations, Field Source payload coherence,
  save/reopen persistence, and registration cleanup.

### Changed

- Integrated Custom Effects v2 as a Scene Properties root between Physics and
  Surface. Artist flow is Effect Stack → Selected Effect → Workflow → Status;
  package administration and Warp Code are closed child panels.
- Warp Code now prefers an existing Blender Text Editor before repurposing the
  current area.
- Preferences now expose runtime/device selection and only the explicitly
  enabled developer override. The previous verbose toggle remains internal.
- Cache mutation dialogs display the exact resolved path or orphan inventory
  and revalidate it immediately before mutation.
- Field Source payload selection and region coverage stay coherent: Volume Grid
  payload selects Volume Grid coverage, while analytic coverage restores a
  Constant payload.

### Fixed

- Fixed Stop Live Preview, Stop Cache Playback, Cancel Bake, and Release Solver
  disappearing or becoming unusable after runtime readiness degraded.
- Fixed Step One Frame opening the multi-frame dialog and disabled stepping
  during cache playback or production bake.
- Fixed Surface + Particles controls being hidden because the UI tested the
  nonexistent `BOTH` identifier instead of `SURFACE_PARTICLES`.
- Fixed Height Fill errors appearing in a conditional source panel instead of
  the authoritative Session & Status surface.
- Fixed installed-package maintenance actions appearing for embedded-only
  Custom Effects.
- Fixed arbitrary existing directories qualifying for destructive Cache
  management. Filesystem services now fail closed unless the target is a
  recognized FluidSim cache or interrupted FluidSim transaction.
- Fixed cache-management visibility for an empty authored path and added
  byte-for-byte move verification for pre-manifest interrupted transactions.

### Compatibility and validation

- Cache-v1 playback, cache-v2 transactions/resume, Phase-3 attributes, Phase-4
  typed whitewater, Phase-5 surface tension, and Custom Effects v2 identities
  remain compatible. No solver or persistent effect schema was removed.
- External Warp 1.16.0+cu12 remains the runtime dependency. Blender and Warp
  runtimes are not bundled.
- Linux x64 is validated with the supplied Blender 5.2.0 LTS and supplied Warp
  1.16.0 package. The release workflow now requires and uploads a native
  Properties UI report alongside solver, Custom Effects, package, and installed
  lifecycle evidence.
- Updated the Phase-5 Blender regression to verify the native Scene Properties
  surface-tension panel and its Simulation Physics parent after removal of the
  legacy sidebar panel.
- CUDA solver certification and Windows x64 remain external gates.

## 1.0.251 — shippable Custom Effects and Warp Code workflow

### Added

- Replaced the prototype-facing Custom Warp Snippets UI with an artist-facing
  Custom Effects stack and a separate developer-grade Warp Code workspace.
- Added eight typed built-in effects, controller-object workflows, ordered
  mute/solo/duplicate/reorder operations, draft/tested/applied states, live
  dirty detection, profiling, and real one-frame preview actions.
- Added strict Effect schema/API 2 and ABI 1 contracts, an authoritative typed
  field/capability registry, deterministic generated source, source maps, real
  `wp.load_module` compilation, and bounded synchronized sandbox launches.
- Added exact-content SHA-256 trust, imported-code blocking, headless
  `deny|builtins-only|trusted` policy, and process exit code `17` for policy
  rejection before compilation or solver allocation.
- Added strict `.fseffect` packages with per-file hashes, path/symlink/duplicate
  rejection, decompression resource limits, atomic installation, identity-safe
  Keep Both behavior, semantic update comparison, safe package rename,
  malformed-package diagnostics, and library reveal/delete operations.
- Added three generated Blender demos with supplied-Warp numerical fixtures and
  Eevee F12 evidence: Vortex Force, stack-order behavior, and Warp Code
  authoring/trust.

### Changed

- Cache-v2 manifests, frame payloads, checkpoints, production-bake reports,
  headless reports, cache inspection, and support reports now carry the exact
  effective Custom Effects signature and ordered inventory.
- Production admission recomputes current Text datablock, parameter, controller,
  order, cadence, frame, binding, and capability state instead of trusting
  stored UI callback hashes.
- Runtime prerequisites are capability-derived. Particle-only effects no longer
  force sparse topology/SDF rebuilds.
- Legacy snippet operator IDs remain one-release migration adapters. Legacy
  `VOXEL_TILE` maps to Active Voxels with a warning; imported legacy code is
  disabled and untrusted.

### Fixed

- Fixed import-only false-positive validation by forcing Warp JIT compilation
  and a synchronized guarded launch.
- Fixed resume continuing after source/parameter/controller changes by binding
  the effective stack identity to every transactional cache/checkpoint seam.
- Fixed failed Apply removing the prior working stack; candidate replacement is
  atomic and invalid enabled effects fail closed.
- Fixed runtime errors allowing later effects or frames to continue; execution
  now raises a structured error and remains latched failed until a successful
  candidate swap.
- Fixed built-in trust spoofing, package identity shadowing, filename-only Keep
  Both behavior, duplicate instance ambiguity, stale UI status, and diagnostics
  attaching to the wrong instance.
- Fixed effect-package zip-bomb exposure by checking member count, per-member
  size, total uncompressed size, and compression ratio before decompression.

### Compatibility and validation

- Cache v1 remains playback-only. Older cache-v2 frames remain playable, but a
  resume with active Custom Effects requires matching effect identity; rebake
  when identity is absent or differs.
- External Warp 1.16.0+cu12 remains the runtime dependency; no Blender or Warp
  runtime is bundled.
- Supplied Blender 5.2.0 LTS and supplied Warp 1.16.0 CPU proof cover the product
  and integration paths on Linux x64. CUDA numerics/determinism/performance and
  Windows x64 remain external release gates.

## 1.0.250 — stable surface-tension physics and final foundation release

- Added continuum-surface-force capillarity from the simulation liquid SDF.
- Added a Brackbill-style capillary timestep limit, explicit clamp diagnostics,
  curvature/acceleration statistics, and optional 24-byte-per-voxel scratch state.
- Added physical Water Surface Tension and stylized High Tension presets.
- Added supplied-Warp CPU kernel validation and mandatory CUDA release gates.
- Preserved cache-v1 and older cache-v2 frame playback. Surface tension adds
  no persistent stochastic state, but cross-release production resume remains
  fail-closed: rebake under 1.0.250 to obtain its exact solver/settings signature.
- Fixed duplicate Spray UI labeling and stale pre-advection SDF reuse when
  capillarity and whitewater are both active.
- Fixed same-key SDF reuse after continuous source emission, first-frame
  capillary substep bypass, reversible optional-scratch accounting, and
  per-frame maximum capillary diagnostics across multiple substeps.
- Fixed the authored Time Scale = 0 pause path so capillary preview and
  particle sanitation remain finite instead of dividing by zero.
- Fixed direct particle injection invalidation so same-frame physical SDF and
  render-mesh caches cannot omit newly appended particles.
- Kept static/source tests independent of Warp so the release identity and
  packaging gate remains runnable before an external runtime is provisioned.
- Expanded SDF-normal support for smoothing, made optional scratch allocation
  transactional, and closed validation-resource leaks on failure paths.
- Strengthened the supplied-Warp proof to exercise the real solver limiter
  before first-frame emission, and removed checkout-name assumptions from all
  Phase-3/4/5 Warp regression harnesses.
- Expanded the physical SDF and sparse topology halo for every capillary
  stencil, including authored splat radii, and included the resulting topology
  and optional scratch costs in capacity and VRAM-safe recommendations.
- Prevented legacy presets from retaining hidden capillary state after a Phase-5
  preset, and made release JSON/checksum metadata strict atomic replacements.

All notable user-facing changes are recorded here. FluidSim follows a three-part release version and declares cache compatibility explicitly.

## 1.0.249 — deterministic typed whitewater outputs

### Added

- Stable persistent whitewater type values: foam `0`, bubble `1`, and spray `2`; value `3` remains reserved and is rejected by current cache/checkpoint validators.
- Deterministic host/device classification with SDF depth, speed, churn, and hysteretic per-step transitions that prevent threshold flicker.
- Type-specific foam advection/preservation, bubble fluid drag/buoyancy, spray air drag/gravity, and independent lifespan multipliers.
- A mandatory int32 `fluidsim_whitewater_type` point attribute in Phase-4 cache-v2 frames and solver checkpoints, with an explicit whitewater-type checkpoint schema version.
- Separate `FluidSim Foam`, `FluidSim Bubbles`, and `FluidSim Spray` Blender outputs and materials, alongside the backward-compatible `FluidSim Whitewater` combined output.
- A P4 Geometry Nodes asset containing six revisioned groups: surface, liquid particles, combined whitewater, foam, bubbles, and spray.
- Supplied-Warp validation for host/device classification parity, typed admission, type-specific lifetimes, aligned compaction, type counts, checkpoint restoration, and the one-pool invariant.
- Supplied-Blender validation for all six output shells, typed cache publication, legacy-unclassified preservation, user-modifier protection, save/reopen, and an Eevee F12 three-material render.

### Changed

- Whitewater remains one GPU simulation pool. Separate artist outputs are filtered publication views and do not triple solver storage.
- The Splash + Whitewater preset publishes both combined and separate typed views; other workflows retain the combined-only preview default.
- Legacy cache-v1 and Phase-3 cache-v2 whitewater without type state is preserved as `-1` unclassified on the combined output. It is never silently labelled foam, bubble, or spray.
- Whitewater type is mandatory and cannot be disabled in the render-attribute inventory.

### Compatibility

- Cache-v1 and Phase-3 cache-v2 frame playback remains supported. Earlier whitewater is combined/unclassified because no physical type state was recorded.
- Phase-3 solver checkpoints are not resumable as Phase 4 because deterministic type and transition state cannot be reconstructed; rebake to create Phase-4 checkpoints.
- Blender 5.2.0 LTS and Linux x64 remain the validated host/platform. NVIDIA Warp 1.16.0+cu12 remains external and is not bundled.
- CUDA solver determinism, NVIDIA memory/performance behavior, and Windows x64 remain open hardware/platform gates on this host.

## 1.0.248 — artist-ready attributes and render nodes

### Added

- One authoritative Phase-3 attribute schema for surface, liquid-particle, and whitewater outputs, including stable Blender names, domains, data types, cache names, and checkpoint policy.
- Opt-in persistent liquid-particle color, source-ID, and UID channels. UID allocation is monotonic and transactional; source identity is deterministic across Blender sessions.
- Derived speed and projected-grid vorticity output, plus Blender motion data stored as per-scene-frame displacement in the conventional `velocity` attribute while `fluidsim_velocity` remains world units per second.
- Strict cache-v2 attribute inventory metadata and fail-closed shape, dtype, finiteness, alignment, and UID validation. Legacy cache frames without a Phase-3 inventory remain readable.
- A generated, revisioned Geometry Nodes asset for surface, liquid particles, and whitewater, together with idempotent installation and explicit update controls that never claim user-owned modifiers or node groups.
- Attribute-aware default materials for source color and viscosity, host-readback/cache-cost estimates, and a dedicated Render Attributes panel.
- Supplied-Warp validation for optional-channel allocation, source metadata, scan compaction, checkpoint restoration, prepared-packet velocity preservation, and the vorticity kernel.
- Supplied-Blender validation for attribute publication, cache round-trip, save/reopen, typed modifier inputs, evaluated Geometry Nodes, motion data, user-asset protection, and non-empty F12 renders.

### Changed

- Point render groups realize particle instances before material assignment so point-domain simulation attributes survive into shader-visible F12 geometry.
- Guided setup installs the current bundled render-node revision on all three owned output objects.
- Persistent render-attribute inventory changes are Reset-tier and participate in solver/cache compatibility signatures; viewport debug coloring remains a separate presentation path.

### Validation status

- The pure-Python suite, source verification, supplied-Warp CPU execution, supplied-Blender attribute publication, save/reopen, Geometry Nodes evaluation, and Eevee F12 rendering pass for this increment.
- The validation host exposed no NVIDIA CUDA driver/device. CUDA solver numerics, CUDA determinism, GPU memory behavior, GPU performance, and hardware renderer regressions remain explicit release gates.

### Compatibility

- Blender 5.2.0 LTS and Linux x64 remain the validated host/platform.
- NVIDIA Warp 1.16.0+cu12 remains an external dependency and is not bundled.
- Cache v1 playback remains supported. Existing cache-v2 frames without the Phase-3 attribute schema remain readable; newly enabled persistent channels require Reset/rebake and are checkpointed.

## 1.0.247 — resumable production workflow and guided first run

### Added

- Strict cache-v2 manifests, transactional frame/checkpoint commits, SHA-256 and file-size verification, contiguous committed-frame enforcement, and fail-closed corruption/orphan handling.
- Solver checkpoint export/import for liquid, whitewater, sparse topology, source lifecycle/RNG, ST-FLIP temporal/RNG/reference-mass state, solver frame, and simulation time.
- Responsive one-frame production-bake controller with safe cancellation, restart, resume, Blender-frame restoration, and live-session restoration.
- Farm-friendly background bake command with JSONL progress, atomic final reports, signal cancellation, and process exit codes `0`, `2`, and `10`–`16`.
- Six strict product presets, atomic user-preset storage, selected-object guided setup modes, owned FluidSim collection structure, stable domain/output creation, and capacity-preflight reporting.
- Four generated example scenes covering live inflow, moving colliders, viscous liquid, and whitewater cache-v2 bake/resume.

### Changed

- `fluidsim.setup_scene` now implements the artist-facing selected-object workflow while preserving its public operator ID.
- New production bakes write cache v2; cache v1 remains readable for playback.
- The installable extension includes only the installed headless entry point from `release/`; build, smoke, wheel-fetch, and example-generation tooling remains excluded.

### Validation status

- Pure-Python, source verification, supplied-Blender registration, installed-package lifecycle, preset/example save-reopen, and supplied-Warp CPU execution are release evidence for this increment.
- CUDA solver numerics, CUDA resume determinism, NVIDIA memory behavior, and GPU performance remain an explicit hardware release gate; they are not certified by a CPU-only environment.

### Compatibility

- Blender 5.2.0 LTS and Linux x64 remain the validated host/platform in this deliverable.
- NVIDIA Warp 1.16.0+cu12 remains an external dependency and is not bundled.
- Cache v1 playback remains available; resume requires cache v2.

## 1.0.245 — external runtime packaging policy

### Changed

- Removed NVIDIA Warp and all Blender/Warp runtime binaries from FluidSim delivery archives.
- Warp 1.16.0+cu12 remains checksum-locked as an external validation/runtime dependency only.
- Release inspection now rejects bundled `.whl` files, Blender runtime archives, Warp package directories, and Warp distribution metadata.
- Blender smoke tests load Warp from a separately provisioned validation environment and prove it is outside the installed FluidSim extension.

### Compatibility

- Solver numerics and cache-v1 compatibility are unchanged.
- Blender 5.2.0 LTS / Linux x64 remains the certified P0 target.

## 1.0.244 — P0 release foundation

### Added

- Canonical `fluidsim_warp` extension identity and centralized runtime version metadata.
- GPL-3.0-or-later license file and a complete embedded Warp-license inventory.
- Certified P0 target: Blender 5.2.0 LTS on Linux x64 with NVIDIA CUDA.
- Namespace-safe generated custom Warp modules and extension-scoped user snippet storage.
- Checksum-locked, package-contained NVIDIA Warp 1.16.0+cu12 binary.
- Cached runtime/device preflight, enumerated CUDA selection, and privacy-safe support diagnostics.
- Deterministic source verification, platform archive inspection, and SHA-256 artifact manifests.
- Clean-profile install, enable, disable, re-enable, fresh-process reload, bundled-Warp execution, and uninstall smoke coverage.
- Self-hosted NVIDIA gate definition for all 16 GPU validations plus ten preview frames.

### Changed

- Blender extension metadata now comes exclusively from `blender_manifest.toml`.
- Registration no longer searches external Python installations for Warp.
- Initialize, live playback, stepping, and cache baking are blocked with actionable messages until the packaged runtime and selected CUDA device are available.
- Render-handler teardown removes only exact callbacks owned by FluidSim.

### Compatibility

- Existing cache-v1 frame output remains readable. P0 does not change the cache payload format or solver numerics.
- Windows x64 is not shipped in this P0 archive because a matching Warp 1.16.0+cu12 Windows wheel and Windows Blender 5.2 runtime were not supplied for validation.
