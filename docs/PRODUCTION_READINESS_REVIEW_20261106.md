# FluidSim 4.1 production readiness review

This is the review snapshot of `80bf4aa`. Remediation is now underway; see [current PR and qualification status](REMEDIATION_STATUS_20261106.md). Subsequent source fixes do not rewrite this original evidence.

Re-reviewed 6 November 2026 against freshly fetched GitHub `master`, `80bf4aaae0a108602ef1ef73cdf6c74e30dc8354`, in `/workspace/Warp-FluidSim-Addon`. `git fetch origin` followed by the ahead/behind check returned `0 0`. Product source was clean; only the prior review/plan were untracked. This replaces the earlier same-day assessment of `502ca66`, whose documents and evidence remain preserved. It reviews the current tree and its 11-commit update, not only a pull-request diff.

**Decision: NO-GO for production release from this candidate.** Installation has a reproducible UI compatibility defect; cache identity and continuation omit relevant input state; qualification can accept evidence for different source bytes. The documented source checks also fail. Stabilize the existing 4.1 features, then qualify the final source and installed packages. No product fixes, release, tag, merge, or publication were performed in this review.

The execution sequence is in [the production release plan](superpowers/plans/2026-11-06-production-release.md). Findings below distinguish observed Python behavior, actual Blender introspection, static analysis, and unexecuted GPU/GUI claims.

## Scope and architecture

The repository contains 808 tracked files and 639 Python files across the add-on, tests, and tools. Review concentrated on production boundaries: installation, admission/replacement, scene input capture, playback/publication, cache/checkpoint/resume, release packaging, and qualification evidence. It included source tracing, existing tests, focused fault probes, and a full repository pytest invocation. It is not a claim that every numerical kernel or every line was independently audited.

The architecture has useful boundaries worth preserving:

| Boundary | Current owners | Release significance |
| --- | --- | --- |
| Artist actions and settings | `operator_domains/`, `properties.py`, `blender/pending_settings.py`, `blender/deck/`, `blender/liquid_display.py` | Keep setting policy centralized; test real controls and frame transitions |
| Admission and active runtime | `blender/simulation_admission.py`, `blender/scene_bridge.py` | Candidate validation must precede a coherent authority change |
| Evaluated scene inputs | `blender/scene_inputs.py`, `blender/evaluated_geometry.py` | Identity, frozen input lifetime, and resume must describe the same data |
| GPU simulation | `sim/solver.py`, pressure/source/rigid/narrow-band modules | CUDA, accuracy, resource growth and timing need target-runtime evidence |
| Published output | `blender/output_publication.py`, `blender/playback.py` | Preview is non-authoritative; display edits still need coherent current-frame output |
| Production bake and persistence | `blender/cache_bake.py`, `blender/production_bake.py`, `io/` | Committed frame prefix, checkpoint continuation and restoration must survive interruption |
| Distribution and qualification | `warp_installer.py`, `tools/release/`, `tools/run_production_gate.py` | A source pass, a ZIP build, and installed CUDA/GUI usability are separate claims |

Positive controls include checksum-locked Warp wheels, refusal to overwrite foreign Warp installations, pickle-free cache/checkpoint reads, atomic checkpoint writes, content-digest effect trust, bounded effect archives, explicit architecture ownership, and substantial regression coverage. These controls do not close the specific gaps below. Large modules such as `pressure.py` (2,794 lines), `solver.py` (2,623), and `output_publication.py` increase review cost, but a broad refactor is not required to ship the stabilization release.

## Changes since the earlier review

| Area | Current finding |
| --- | --- |
| Source/release target | Current GitHub master is `80bf4aa`; metadata still says 4.1.0; remote refs checked during sync listed v4.0.0/v4.0.1 and no v4.1 tag |
| Warp | Pin moved to **1.19.0**; both locks/build/version/requirements agree, and both platform lock entries match current PyPI metadata. The upgrade explicitly has **no completed 1.19 GPU qualification** |
| Paused liquid display | **Partially fixed upstream:** cached liquid geometry and preview points survive OFF/PREVIEW/SURFACE changes. Preserve that behavior; do not repeat the old broad failure claim |
| Remaining cached edits | Whitewater Combined→Separate and display fractions still fail to rebuild paused output (R6) |
| New live display regression | Viewport-only fallback particles can lose selected attribute coloring (R13) |
| Cache identity | R2 remains, but is narrowed to Resume/Re-bake: ordinary playback intentionally checks settings only, not source history |
| Full regression suite | Current full pinned-runtime run has **five failures**. The preceding 435-test focused pass did not establish a clean full suite or repair the old isolated bpy failure |
| Experiments | ST-FLIP is permanently out. Native interop/rendering APIs are reactivated for recovered-source review and new isolated plans; not automatically approved for merge or release |

## Executed verification

Fresh evidence root: `/workspace/scratch/fluidsim-review-80bf4aa-20261106`. `ownership.json` records source SHA, ownership, budget and retention. Earlier documents are preserved in its `prior-documents/` tree. Product source was not modified during this review.

| Check | Result | Boundary |
| --- | --- | --- |
| Full repository pytest, **pinned Warp 1.19.0** | **3,332 passed, 345 skipped, 5 failed**, 134.24 seconds, exit 1 | Fresh isolated Python 3.13.5 environment; pytest 9.1.1, Ruff 0.16.2, NumPy 2.3.4, Warp 1.19.0; CUDA hidden |
| Same tree on existing Warp 1.17.0 environment | **3,332 passed, 345 skipped, 5 failed**, 105.64 seconds | Contrasting compatibility run; does not qualify the 1.19 target |
| Five failing tests together in fresh pinned process | **All five failed again** | Confirms failures do not depend on the older Warp environment |
| Configured add-on Ruff | **2 errors** | Executed from add-on root; configured include set is narrower than all product code |
| Source metadata verification | **Passed** | `build_extension.py --verify-only`; its `release_ready: true` means metadata only |
| Syntax | **639 Python files, zero errors** | In-memory compilation without executing each module |
| Warp 1.19 CPU whitewater validator | **Passed, 9 checks** | Actual locked-version Python runtime and CPU kernels; `cuda_certified: false` |
| Warp 1.19 CPU surface-tension validator | **Passed, 14 checks** | Actual CPU kernels; `cuda_certified: false` |
| Warp 1.19 CPU Custom Effects validator | **Passed** | Real built-in compilation/sandbox and expected invalid-kernel rejection; CPU only |
| Current Blender icon probe | **Invalid `GPU` icon confirmed again** | Linux Blender 5.2.1 LTS, build `9e2066aef7ef`, private profile; exact certified 5.2.0 drawing not executed |
| Installer interruption probe | **64 KiB productive transfer still fails after 8 attempts** | Actual downloader with controlled transport faults; 1 MiB-progress completion and zero-progress stop controls |
| Current cache, display, cutover and gate probes | **Findings reproduced; positive controls retained** | Actual Python policy/owners with scoped doubles/fault injection; not complete GPU/GUI user journeys |
| Full production/GUI gate | **BLOCKED here** | No NVIDIA driver/device; available Blender is 5.2.1, while certification requires 5.2.0 |
| Final ZIP build/install/GPU/GUI acceptance | **NOT_RUN** | No new release package built or certified |

The pinned runtime was provisioned in the evidence root; the existing sibling virtual environment was left unchanged. Its actual import path and device list are in `pinned-runtime.json`. Warp 1.19 initializes on CPU and reports CUDA unavailable. Wheel-lock metadata consistency is separately recorded in `gate-review/pypi-lock-comparison.json`; this does not establish installed final-archive provenance.

### Source-check failures and repair boundaries

| Test | Cause observed | Required repair |
| --- | --- | --- |
| `tests/cache_io/test_frame_output.py::test_output_cost_counters_separate_all_four_phases` | Host-independent test stubs `_require_bpy`, but `prepare_outputs` imports `materials.py` and then missing bpy | Scope material/output doubles correctly; retain all four counter assertions and actual Blender acceptance |
| `tests/perf/test_performance_hotpath_contracts.py::test_skipped_surface_publish_preserves_existing_mesh` | Old SimpleNamespace payload lacks new `retain_cache_geometry` field | Use current PublicationPayload/default fields; preserve skipped-surface/no-swap behavior assertion |
| `tests/runtime/test_settings_getattr_names.py::test_defaulted_settings_reads_name_declared_fields` | Scanner treats RNA `is_property_set` method as a declared settings field | Recognize this non-field attribute and preserve misspelled-field negative controls |
| `tests/test_ownership_map.py::test_map_is_current` | Generated ownership map was not regenerated after display changes | Regenerate using `tools/ownership_map.py`; review output |
| `tests/test_ownership_map.py::test_declared_owner_is_the_declaring_class` | Map lacks new Scene-owned `liquid_display_mode` | Same regeneration, then verify owner/policy consistency |

These are release-check failures, not proof that the corresponding production workflow crashes. They still block the required source gate. The two Ruff findings remain B007 at `fluidSim_warp_blender51/tests/test_properties_ui_contract.py:142` and I001 at `fluidSim_warp_blender51/tests/test_properties_ui_release_source.py:1`. One initial Ruff invocation from the wrong working-directory/config scope selected no files; it is retained as `ruff-scoping-attempt.log` and is not counted as a pass. The corrected add-on-root invocation is `ruff.log` and fails.

## Findings

Priorities: P1 should block the candidate; P2 is a concrete correctness/reliability issue scheduled for this stabilization release. Runtime frequency is not inferred from fault-injection tests.

### R1 — P1 — Preferences cannot draw its installation controls

**Location:** `fluidSim_warp_blender51/preferences.py:263`.

The first Preferences label uses `icon="GPU"`. Actual Blender 5.2.1 RNA does not contain that icon, so drawing the Preferences body raises before the Install Warp and Install from File controls. The scan found this as the only invalid literal icon in product Python. `docs/CERTIFICATION_4_0_1_20261017.md` independently records the same visible failure on 5.2.2; the bad code remains at this HEAD.

**Scope:** shared Preferences entry point, regardless of demo or CUDA availability. Exact 5.2.0 and Windows drawing were not executed here. Replace with a supported enum value, validate all shipped literal icons against certified Blender, and test the installed Preferences UI with missing, owned and foreign Warp states. Mock layouts accepting arbitrary strings do not protect this boundary.

### R2 — P1 — Cache identity omits consumed source data

**Locations:** `blender/cache_identity.py:25–35`, `:75–95`, `:99–118`, under the add-on root.

Meshes contribute coordinates and polygon count but no connectivity. Non-mesh objects contribute only type and point count. Pointer properties are excluded, and role-NONE objects are skipped, even when an emitter refers to them as its actual mesh/point/volume input. Production `SceneInputs` consumes those references, triangulation, point positions, and VDB path/data.

Changing topology with constant coordinates/counts, replacing a referenced point input, or changing a VDB payload can therefore leave the source signature and frame stamp unchanged. Both topology and referenced-point collisions were reproduced. Resume/Re-bake can accept an old checkpoint for different simulation inputs. Ordinary cache playback deliberately uses settings-only compatibility (`playback.py:419–443`), so this finding applies to source-history-dependent Resume/Re-bake rather than bypassing a source-aware playback guard. Direct tagged vertex-coordinate and matrix changes are contrasting covered paths.

Fix the shared identity seam to fingerprint resolved effective dependencies, including topology and consumed external data. Version the identity semantics and refuse continuation only when the required input proof is missing; preserve supported legacy playback/export and safe, explicitly verified reconstruction paths. Tests belong in `tests/cache_io/test_cache_io.py`, `test_source_frame_stamp.py`, `test_cache_v3.py`, plus real partial-bake/edit/refusal acceptance.

### R3 — P1 — Resume loses frozen input state

**Locations:** `blender/production_bake.py:900–914`, `blender/scene_inputs.py:937–955` and `:1310–1327`, `sim/sourcing.py:316–328`.

Resume performs fresh admission at the last committed frame. This correctly primes moving-collider history, but also creates fresh FIRST_SIM_FRAME source snapshots and Static-collider snapshots. Those host-owned snapshots are absent from solver checkpoint state. An unchanged animated input can consequently be frozen at the checkpoint frame instead of at the original first frame.

The actual frozen-input method kept point `[1,0,0]` in the uninterrupted case and captured `[10,0,0]` after admission at the checkpoint. A stronger current probe also drove the actual Production Bake initialization method with scoped scene/solver doubles: bake start 1, committed frame 10, initialization at 10, and frozen point 10 before next frame 11. Source tracing establishes affected mesh/point/volume/field and static-collider lanes; a complete Blender/GPU continuation comparison remains required. Changing admission globally to the start frame would break the contrasting EVERY_SIM_FRAME moving-collider lane.

Preserve or deterministically reconstruct frozen input state independently of dynamic motion history. Resume and Re-bake need shared coverage, including an uninterrupted control and a native-input sibling. Do not claim bitwise continuation for every emitter merely because rigid checkpoint tests pass.

### R4 — P1 — Qualification evidence can describe different source bytes

**Locations:** `tools/run_production_gate.py:546–558`, `:711–725`, `:765–780`; `tools/benchmark_versions.py:175–190`.

The gate logs dirty state but does not freeze a clean candidate or compare start/end source identity. A fresh budget accepts a matching inventory from an uncommitted tree. During reuse, a mismatched inventory can fall through to a commit-diff check. Measuring a dirty solver change and reverting it at the same HEAD was reproduced as accepted reusable evidence. The manifest then names HEAD, without binding the working tree that earlier acceptance lanes actually ran.

**Scope:** all fresh/reused qualification paths, not just one demo. The intended allowance for regenerated dam/crown compressed files must not permit changed solver bytes. Freeze source and verifier fingerprints, reject arbitrary dirty inputs and mid-run changes, and bind every accepted measurement to actual bytes. Preserve only explicit, narrowly tested generated-demo normalization. Add rejection controls to `tests/gate/test_production_gate_semantics.py`, `test_demo_budget_runner.py`, and `test_performance_evidence_integrity.py`.

### R5 — P1 — Package builds inherit the regular Blender profile

**Location:** `tools/release/build_extension.py:856–879`.

The validate/build subprocesses use `--factory-startup` without a private environment. `_run` inherits the caller's profile; `run_local_gate.py` does not supply isolation. The repository explicitly documents that factory startup can synchronize away other installed extensions' shared wheels. Smoke-test profile isolation occurs later and cannot protect these earlier subprocesses.

**Scope:** direct build and local package gate unless the caller independently isolated all Blender paths. Unsafe invocation is confirmed statically; deletion of user-profile wheels was deliberately not reproduced. Give every Blender build/validate/generation launch private config/scripts/extensions/datafiles and prove parent-profile inventories unchanged, including failure paths. Add captured-environment assertions in `fluidSim_warp_blender51/tests/test_build_extension.py`.

### R6 — P2 — Paused-cache whitewater mode and sampling edits stay stale

**Locations:** `blender/pending_settings.py:1243`, `blender/output_publication.py:1717`, `:1748`, `blender/playback.py:568`.

**Partially fixed upstream:** actual cache payloads now retain liquid mesh/particles even when hidden, and preserve preview points. Current OFF→SURFACE/PREVIEW controls work without re-reading the frame. That portion of the prior finding is closed.

Geometry-dependent whitewater and sampling edits still use the no-session metadata-only callback. Paused Combined→Separate hides the combined output but does not populate typed whitewater that publication previously cleared. Changing particle display fraction from 1 to .5 leaves six points until an explicit republication produces three. The current-policy probe used actual cache preparation/publication and Pending Settings; republication produced the missing foam and corrected sample count.

**Scope:** panels and Deck controls share the callback. Repair only geometry-dependent cached edits through authoritative current-frame republication without creating a solver; preserve the working mode-switch retention and cheap material/radius changes. Test whitewater modes/fractions immediately while paused, plus attribute modes and Warp-unavailable playback as acceptance coverage. This host-policy result is not claimed as a foreground visual test.

### R7 — P2 — Retirement failure leaves an active session holding a released solver

**Locations:** `blender/scene_bridge.py:193–206`, `blender/simulation_admission.py:706–710`, `:745`, `:768–772`, `:801`.

`Session.install_solver()` assigns the candidate before retiring old inputs/runtime. If retirement raises, Admission has not marked the replacement committed and discards the now-authoritative candidate. Controlled old-runtime release failure reproduced both Initialize and Reset retaining `active=True` with a released candidate. Reset additionally reports that the live solver was preserved.

**Scope:** shared replacement seam and its reinitialize/reset/apply/rewind/bake callers, conditional on retirement error. No spontaneous CUDA cleanup failure was observed. Make authority transfer explicit before retirement; failure must leave a usable installed runtime or a truthful inactive state. Cover old-input cleanup, old-solver release and CUDA-memory-release failures using the real Session/Admission seam.

### R8 — P2 — Returning from Bake restores preview inputs at the wrong frame

**Location:** `blender/production_bake.py:1113–1134`.

Preview restoration initializes/synchronizes while the scene is still at the last baked frame, restores solver state, and only then restores the original playhead. The real method with a session double recorded admission/sync at frame 20 followed by restoration of playhead 5. Frozen inputs retain bake-end state; moving-collider history is also primed at the wrong time. This affects finish, cancellation and failure restoration. Fix frame ordering together with R3's host input snapshot restoration; verify against an untouched live-preview control.

### R9 — P2 — Productive small downloads exhaust a no-progress budget

**Location:** `fluidSim_warp_blender51/warp_installer.py:189`.

Retries reset their failure counter only after at least 1 MiB of progress. A controlled 2,097,155-byte transfer delivering 65,536 bytes per interruption stopped after eight attempts with only 524,288 bytes received. One-MiB chunks completed in three attempts; zero-progress stopped after eight. This contradicts the documented consecutive-no-progress policy and affects all online installs/repairs; local-file installation bypasses it.

Track productive progress independently of the read-buffer size, retain the bounded no-progress stop, and test 64 KiB, boundary-adjacent segments, ignored Range, short clean response, incorrect content and checksum rejection. Preserve a termination bound when the server repeatedly restarts instead of making retained progress.

### R10 — P2 — Performance reuse lacks a current environment identity

**Locations:** `tools/run_production_gate.py:635`, `tools/performance_budget.py:413`, `tools/benchmark_demo_trial.py:166`.

The reducer compares device/Warp fields between trials, but neither reducer nor reuse binds them to the current qualified host. Clean, byte-identical synthetic evidence with Warp 1.17.2 passed while this checkout targets certified Warp 1.19.0. Reports identify `cuda:0`, which cannot establish GPU model, driver, platform or Blender identity. A committed dependency-pin change often triggers invalidation indirectly, but moved evidence and changed external runtimes remain unbound.

Add an observed environment fingerprint to trials and reuse eligibility. First Linux certification and any runtime/driver/GPU/Blender change require a new measurement. Reject missing provenance instead of treating it as equivalent.

### R11 — P2 — Production admission retains Blender 5.1 exemptions

**Locations:** `tools/run_production_gate.py:65–73`, `tools/release/validate_phase5_blender.py:54`, `tools/release/validate_properties_ui_blender.py:1177`.

The production runner falls back to 5.1 and does not perform an early exact-version/digest check. Two validators explicitly exempt 5.1 from the certified-version check. The manifest stores a path, not the observed executable identity. This contradicts the 5.2.0 qualification contract. The separate local gate already checks exact version and digest. Remove the qualification escape hatches and align preflight. This finding does not assert that every later acceptance test would pass on 5.1.

### R12 — P2 — Frame checksum and parsing can inspect different bytes

**Location:** `fluidSim_warp_blender51/io/cache.py:728–736`.

The reader verifies a path and then reopens it through `np.load`. Controlled replacement between those operations returned changed coordinates even though the new file's checksum did not match the manifest. Activation requires concurrent frame-file mutation, not serialized normal playback. Playback and VDB export share the reader. The checkpoint sibling already hashes and parses one byte snapshot (`io/checkpoint.py:268–281`). Use that pattern or an equivalent bounded immutable read, and test replacement plus corruption without weakening alignment/size checks.

### R13 — P2 — Live viewport-only particles omit selected attribute coloring

**Location:** `fluidSim_warp_blender51/blender/output_publication.py:1244–1260`; empty-mesh fallback at `:1300`.

The new display policy gathers particle positions when the viewport needs them independently of `show_particles`, but attribute gathering and selected-value computation still require that flag. It now governs render/export visibility. Surface mode with meshing disabled, or with an empty extracted surface, can therefore show fallback particles without the selected HEIGHT/SPEED visualization when particle rendering is disabled.

A current probe produced six viewport points and no attributes with `show_particles=False`; changing only the render flag yielded `__visualization__`. The cached path preserved HEIGHT coloring with the same render flag disabled, and empty-surface fallback reproduced the live omission. Scope is the shared `prepare_live_payload` owner used by playback and property-triggered publication. Existing new geometry tests stub attributes to empty, so cannot detect this.

Gather selected attributes whenever the actual viewport/live payload needs them, including fallback discovered after extraction. Extend `tests/ui/test_liquid_preview.py` with attribute-bearing fixtures and assert render visibility stays unchanged. Confirm actual Blender viewport coloring on certified runtime; the current proof uses host payloads, not framebuffer pixels.

## Readiness gaps and bounded risks

- **Installed delivery:** the main GPU gate runs repository source. Local package smoke hides CUDA. The final ZIPs need installed CUDA/GUI/bake/export evidence bound to their own hashes; rebuilding after qualification creates untested bytes.
- **Platforms:** `version.py` certifies Windows x64 and explicitly leaves Linux unqualified. Manifest/build support for Linux is not Linux production proof. No current-candidate full gate receipt was available in this checkout.
- **Concurrent cache ownership:** process-local guards do not establish exclusive ownership across two Blender processes. Define and test safe refusal/recovery for a second writer, or constrain/document the supported workflow. No concurrent full bake was executed here.
- **Release documents:** `docs/RELEASING.md` now correctly names Warp 1.19. `CONTRIBUTING.md:67` still points at a 1.18 environment and the older 4.1 plan still describes 1.18 certification. The old archive-integration plan is not a current execution checklist. Reconcile the changelog with subsequently merged display/Deck/settings/allocation work and actual qualification results.
- **Scope protection:** ST-FLIP is permanently removed. Native interop/rendering APIs are authorized for recovered-source review and new isolated plans, but do not yet have merge/release qualification. Keep them separate from this stabilization release. Surface Nets is a proposal supported by a synthetic CPU trial, not a shipped/qualified backend. Learned pressure guessing remains hidden/off with no weights. Do not broaden determinism across GPUs or Warp versions.
- **Qualification admission failure:** storage admission can fail before old production-manifest invalidation. Consumers must bind a receipt to the intended run rather than infer success from its presence. Include failed-admission invalidation in R4's tests.
- **Lifecycle acceptance:** disable/re-enable, load during bake, undo/redo, handler persistence, cache cursor reset, GPU OOM/recovery and resource retirement need final installed-runtime coverage. Source concerns without a concrete reproducer are not promoted here to additional confirmed defects.

## Recommended release scope

Keep the next candidate as **4.1 stabilization**: existing Body Type/rigid contact, nine demos, surface-tension fixes, integrated settings, Deck/allocation work, Warp 1.19, and the new display/navigation changes, plus the repairs above. The remote tag check during sync listed v4.0.0 and v4.0.1 and no v4.1 tag. Recheck GitHub tags/releases at candidate freeze; use 4.1.0 only if still unused, otherwise the next unused patch version. Never replace an existing release/tag.

Three viable schedules exist:

1. **Recommended: Windows production first, Linux qualification as a separate completion gate.** Matches the existing certification policy. Linux cannot be described as production-certified until its actual installed NVIDIA/GUI/export matrix passes. A Linux artifact published earlier must be explicitly experimental and require the owner's scope decision.
2. **Simultaneous Windows/Linux production release.** Complete the same hardening, then hold publication until both platform matrices pass. Choose this if equal platform support is required at launch.
3. **Expand into the old 4.2/performance backlog.** Adds numerical change and requalification risk before fixing evidence/data-integrity defects. Defer warm-start changes, new exporters, new demos and experiments.

A new **Warp 1.19 GPU/GUI/performance campaign** is mandatory: the new pin is a target, not accomplished certification. Rebind actual imported paths, runtime builds, demo hashes and final archive evidence. Compare numerical outputs and budget on the approved same-runtime baseline; never infer 1.18→1.19 bitwise equivalence from unchanged solver source.

Existing gate thresholds stay authoritative. Historical four-demo budgets are not evidence for the five newer demos; all nine need functional/physics acceptance. A release date should follow successful gates and available GPU hosts, not substitute for them.

## Reproduction and evidence index

Commands run from `/workspace/Warp-FluidSim-Addon` unless stated otherwise. Repeated pytest attempts need new temporary directories to preserve evidence.

```bash
CUDA_VISIBLE_DEVICES=-1 /workspace/scratch/fluidsim-review-80bf4aa-20261106/venv/bin/python -m pytest -q \
  --basetemp=/workspace/scratch/fluidsim-review-80bf4aa-20261106/pytest-pinned-tmp \
  -o cache_dir=/workspace/scratch/fluidsim-review-80bf4aa-20261106/pytest-pinned-cache
# Ruff: same interpreter, `-m ruff check .`, cwd fluidSim_warp_blender51/.
python tools/release/build_extension.py --verify-only
python /workspace/scratch/fluidsim-review-80bf4aa-20261106/probe_download_progress.py
python -B /workspace/scratch/fluidsim-review-80bf4aa-20261106/gate-review/reproduce_budget_binding.py
PYTHONDONTWRITEBYTECODE=1 python /workspace/scratch/fluidsim-review-80bf4aa-20261106/cache-review/probe.py
PYTHONDONTWRITEBYTECODE=1 /workspace/scratch/fluidsim-review-80bf4aa-20261106/venv/bin/python \
  /workspace/scratch/fluidsim-review-80bf4aa-20261106/runtime-review/probe_display.py
PYTHONDONTWRITEBYTECODE=1 /workspace/scratch/fluidsim-review-80bf4aa-20261106/venv/bin/python \
  /workspace/scratch/fluidsim-review-80bf4aa-20261106/runtime-review/probe_cutover.py
```

CPU validators: run each `tools/release/validate_{phase4,phase5,custom_effects}_warp.py` under the pinned interpreter with `--device cpu --expected-warp-root /workspace/scratch/fluidsim-review-80bf4aa-20261106/venv/lib/python3.13/site-packages/warp --report-json <owned-report>`, `CUDA_VISIBLE_DEVICES=-1`, and a private `WARP_CACHE_PATH`. Their real compilation/output/negative-control reports are retained; no CUDA execution is inferred.

Key evidence: `pytest-pinned-full.log`, `pytest-full.log` (1.17 control), `pytest-failures.log`, `ruff.log`, `metadata.json`, `syntax.json`, `pinned-runtime.json`, `pinned-environment-install.log`, the three CPU validator JSON/log pairs, `blender-icons.log`, `download-progress.json`, and the cache/runtime/gate probe directories. The actual Blender enum probe used private config/scripts/extensions/datafiles and `--background --factory-startup --python-exit-code 1`. No current-result claims rely solely on prior-head logs.
