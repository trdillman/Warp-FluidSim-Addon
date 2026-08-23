# Changelog

All notable user-facing changes are recorded here. FluidSim follows a three-part release version and declares cache compatibility explicitly.

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

### Fixed

- Cache playback retries a frame after failed publication and reinitializes singleton solver sessions when Blender scene ownership changes.
- Lazy particle and whitewater outputs share one refreshed live-count snapshot, so attributes requested before positions remain complete without redundant host-count reads.
- Cache metadata, point export, benchmark comparison, profiler windows, particle-capacity growth, source sampling, custom snippets, and kernel registration now reject malformed or ambiguous inputs without leaving partial state.

### Validation status

- The pure-Python suite, source verification, supplied-Warp CPU execution, supplied-Blender attribute publication, save/reopen, Geometry Nodes evaluation, and Eevee F12 rendering pass for this increment.
- The validation host exposed no NVIDIA CUDA driver/device. CUDA solver numerics, CUDA determinism, GPU memory behavior, GPU performance, and hardware renderer regressions remain explicit release gates.

### Compatibility

- The source extension is compatible with Blender 5.1 or newer and does not
  restrict installation by operating system. Blender 5.1/Windows x64 is covered
  by the repository production gate; Blender 5.2.0 LTS/Linux x64 remains the
  certified 1.0.248 release target.
- Blender 5.2.0 LTS and Linux x64 remain the validated host/platform.
- NVIDIA Warp 1.12.1 through 1.16.x is accepted by the runtime compatibility
  gate; 1.16.0+cu12 remains the certified external dependency and is not bundled.
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
