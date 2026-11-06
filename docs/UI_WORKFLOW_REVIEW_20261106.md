# FluidSim UI and production-workflow review — 6 November 2026

Three actionable **P2 usability findings** remain: misleading viewport labels for independent render/export controls, no direct way back from cache playback to live work, and hidden cache-error detail in the primary viewport surfaces. No renderer-policy change is recommended. Native setup, paused cache publication, save/reopen, and Alembic/USD/VDB cache-export diagnostics passed within the limits below. This is **not foreground visual approval or release qualification**.

## Reviewed source and evidence boundaries

Product source: `d77d4bfbcb7c435a34e983c3e7b01a2a9f1d1974`, detached worktree `/workspace/scratch/fluidsim-remediation-20261106/ui-specialist/source`. Worktree remains clean. Root's later acceptance-only commits were not substituted for this product revision. No product edits, install/system mutations, pushes, PR comments, subagents, or normal Blender-profile access were performed.

Evidence root: `/workspace/scratch/fluidsim-remediation-20261106/ui-specialist`. Ownership was recorded before allocating worktree/probe artifacts. Every Blender subprocess used `--background --factory-startup --python-exit-code 1` and private absolute configuration/scripts/extensions/datafiles/temp/cache directories. Actual binary: Blender **5.2.1 LTS**, build `9e2066aef7ef`; this is not the certified **5.2.0** runtime. CUDA was hidden; the cache lanes require no solver. The paused-cache acceptance explicitly prevented importing Warp. No screenshot was captured.

Foreground capability was investigated: DISPLAY, WAYLAND_DISPLAY and XDG_RUNTIME_DIR were unset; no X socket, Xvfb or Weston was available. Xorg and xdpyinfo exist, which does not establish a usable foreground display. The connected Blender MCP status request returned `UNAVAILABLE`, timing out at `127.0.0.1:19879`. No existing user's Blender session was modified. `environment-discovery.json` records these observations; an isolated foreground lane remains unexecuted.

Evidence labels used here:

- **Native background**: real Blender RNA, registered operators, output IDs, file handlers, actual exporter/importer/readback. No mocked solver claim.
- **Native RNA + recorded draw**: production panel functions receive actual scene/property state and a recording layout. This establishes bindings, emitted text/actions and exceptions; it does not establish native UILayout rendering, geometry, truncation, contrast, mouse interaction or accessibility.
- **Host tests**: CPU pytest, including explicit host doubles. These supplement production callers; they do not certify runtime interaction.
- **Blocked**: required foreground/CUDA/certified runtime evidence was not obtained.

## Findings

### U1 — P2: Label the independent output controls according to what they actually control

**Locations:** `fluidSim_warp_blender51/properties.py:909` (Liquid View and its viewport tooltip), `properties.py:916` (Surface Mesh/Particles tooltips), `blender/ui_stages.py:360` and `:381` (both controls in Look), `blender/ui_scene.py:913` and `:922` (Visible summary), `:938` and `:955` (both controls in Display). Deep owner: `blender/liquid_display.py:44`.

**User story:** while reviewing a paused cache, choose Liquid View → Hidden to hide liquid, or Particles to inspect the particles.

**Reproduction:** load/publish the retained synthetic cache at frame 7 with a nonempty surface and six particles; leave Liquid Display = Surface Mesh; change the actual RNA Liquid View value through its callback. Observe actual output `hide_get()` flags and record the native-state panel adapters. A control run uses Surface/Surface; another uses Surface/Off.

| Liquid View | Liquid Display | Actual native visibility | UI summary |
| --- | --- | --- | --- |
| Surface | Surface Mesh | Surface visible, particles hidden | Visible: Surface |
| Hidden | Surface Mesh | Surface visible, particles hidden | Visible: None |
| Particles | Surface Mesh | Surface visible, particles hidden | Visible: Particles |
| Surface | Off | Both hidden | Visible: Surface |

**Scope and root cause:** shared presentation mismatch, not broken rendering. `liquid_display.viewport_families()` intentionally derives viewport geometry from Liquid Display and surface availability, independently of the authored render/export flags. Those legacy flags still have meaningful publication/render/export duties. The UI and tooltip nevertheless promise that Liquid View chooses viewport Hidden/Particles, and the summary calls that legacy selection "Visible." Scene Properties, its Physics mirrors, and the sidebar expose this mismatch. The native test exercises actual cache publication and visibility, not merely a property mock.

**Expected/minimal recommendation:** preserve viewport/render/export independence. Relabel/rehome the legacy selection and visibility fields as render/output choices, explain their scope, and derive the viewport summary from the authoritative display policy (including preview/fallback/off), rather than `liquid_view_mode`. Do not make an export-affecting setting implicitly follow a viewport click. Verify both Scene/Physics mirrors and sidebar text after the change.

**Evidence:** `workflow_findings.json` → `findings.competing_view_controls`, `workflow_findings_probe.py`, `workflow_findings/process.log`. Four native contrasting states; 35 retained surface vertices. No framebuffer evidence claimed.

### U2 — P2: Offer a direct cache-to-live transition beside cache transport

**Locations:** `blender/ui_scene.py:1036` (`draw_cache_playback`), `panels.py:122` (`draw_transport`), `blender/deck/controls.py:99` (Deck transport) and `:249` (Bake wheel Play Cache). Existing transition owner: `operator_domains/session.py:750` (`cache_playback_stop`); existing refusal: `blender/pending_settings.py:878`.

**User story:** play a baked cache, pause, change simulation settings, then return to live preview to iterate.

**Reproduction:** enter the cache through `playback.prepare_cache_preview`, the exact owner called by Play Cache, then inspect the paused cache state. Sidebar/header/Properties cache controls offer Play; Deck transport offers only `screen.animation_play`; the Bake wheel has Play Cache again. None offers Stop Cache Playback / Return to Live. The authoritative pending owner refuses apply with `Cache Preview is selected; switch to Live Preview before applying simulation changes`. A native call to the registered `fluidsim.cache_playback_stop` succeeds, clears both cache/live flags and returns to idle. The command palette indexes this operator, proving a workaround exists.

**Scope:** shared missing affordance across sidebar, header, Properties and Deck, not a missing lifecycle implementation or a claim that no recovery is possible. Cache playback releases the solver; the normal Play action follows the still-selected cache state. Both runtime-ready and playback-only drawers use the same cache transport helper; GPU execution was not required to observe the omitted action. Starting a new bake is not a substitute for the artist's requested live-preview iteration.

**Expected/minimal recommendation:** add an explicit `Return to Live`/`Stop Cache Playback` action in the shared transport/cache drawer and Deck mapping, using the existing owner. Explain that it leaves the baked files intact. Test successful cache → pause → stop-cache → live-start on the certified GUI/CUDA lane. Keep generic Play/Pause semantics intact.

**Evidence:** `workflow_findings.json` → `findings.cache_exit`, including actual deepest-owner refusal and the native public stop-operator success. `native_ui_probe.json` additionally records the stage, Properties, header, Deck slots and palette counterpart. No simulated GUI clicks were used.

### U3 — P2: Show persistent cache failures beside viewport transport

**Locations:** `panels.py:161`, `blender/ui_header.py:141`. Shared activity already contains `detail` in `blender/playback.py:122`; cache failure is recorded at `blender/playback.py:624`. Contrasting correct presentation: `blender/ui_scene.py:334` and `:362`. Diagnostic overlay exclusion: `blender/viewport_overlay.py:37`.

**User story:** review a partial cache, advance beyond the last committed frame, understand why the output stopped changing, then recover.

**Reproduction:** publish valid frame 7; arm cache playback with registered `fluidsim.cache_playback_start`; advance to absent frame 8 through native frame handling. The real cache reader raises FileNotFoundError. `fs.status`, `current_error(scene)` and `activity_snapshot(...).detail` all contain the missing-frame/path explanation. Sidebar/header draw records still say **Cache Playback Ready** and omit that explanation; they show only the separate missing-CUDA notice. Properties Status & Runtime includes the actual error. Returning to valid frame 7 republishes six particles and clears the error.

**Scope:** shared presentation loss for cache failures; sidebar and header are affected, Properties is a contrasting functioning sibling. The diagnostic HUD cannot compensate in the reproduced no-solver/no-pending cache state because it returns early. This also demonstrates that generic runtime readiness is not an adequate explanation for cache-only failure. No claim is made that missing frames must stop the timeline: existing playback policy deliberately keeps cache mode armed and retries subsequent/repaired frames.

**Expected/minimal recommendation:** render the existing activity detail/current error in the status strip and header with a concise actionable error treatment and a path to full details/cache controls. Avoid asserting "Ready" as the only activity message when cached-frame publication failed. Preserve cache retry/catch-up behavior; no solver policy rewrite is needed. Verify missing/corrupt/range failures and recovery alongside a successful cached frame.

**Evidence:** `workflow_findings.json` → `findings.missing_frame` and final recovery checks; `workflow_findings/process.log`. Actual native handler/cache error, two affected draw adapters and the Properties control checked.

## User-story matrix

A pass is only for the explicitly described lane. "Partial" does not imply a passing end-to-end production story.

| Story / expected outcome | Result | Evidence and limits |
| --- | --- | --- |
| Fresh registration, genuine missing-Warp message, online and file install controls | Partial | Actual registration/missing runtime, Preferences recording adapter includes both registered operators. Installed ZIP Preferences rendering and actual clicks not exercised. |
| Online/offline install, good/damaged/foreign runtime, cancellation/repeated click recovery | Blocked | Root's new installer matrix is separate; this agent did not install dependencies or substitute mocks for native execution. Certified foreground matrix still needed. |
| Selected mesh → Set Up Only → owned domain/source/outputs | Pass, native background | `quick_setup/process.log`: real operator, source/material preservation, repeat setup, transactional rollback at three injected internal failure points. `native_ui_probe.json`: designated object-owned domain and source. |
| Add/tag collider; edit only designated-domain settings | Pass, native background/RNA | Actual collider tagging; authoritative domain RNA and native navigation. No physical collision simulation claim. |
| Scene / Physics / object / sidebar panels stay bound to current owner | Pass, recorded draw only | 204 successful recorded draws across domain/source/collider selection, including fresh Preferences/Setup; no missing property/operator exceptions. No actual layout readability claim. |
| Open full settings from domain vs unrelated object | Pass, native background | Registered navigation selects Physics for domain and Scene otherwise. Workspace duplication/native popup behavior remains untested. |
| Liquid Deck palette authoring changes the same property | Pass, partial | Actual palette `commit_inline` changes domain viscosity RNA and returns "takes effect on the next run" without allocating a solver. Real modal events, Ctrl+K/Alt+RMB, undo and gizmo drags remain foreground-blocked. |
| Live play → pause → resume / step | Blocked | No usable CUDA/foreground session. A public Play Cache attempt correctly refuses the background context before starting UI transport. |
| Apply/reset/reinitialize/revert with pending solver changes | Partial | 370 host regressions include pending/apply/deck/bake guards. Idle native palette feedback is correct. Running solver confirmation and cutover need CUDA/GUI. Cache→live affordance fails U2. |
| Preview / Surface Mesh / Off publication handoff while paused | Pass, native IDs/callbacks | `paused_cache.json`: retained surface/point handoff without rereads. Focused native U1 probe verifies Off hides both. Smooth Preview pixels/renderer never tested. |
| Legacy Liquid View labels predict viewport result | **Fail U1** | Four native contrasting output states and corresponding misleading summaries. Runtime independence is intentional. |
| Optional GPU Particle Preview, fallback, graphics lifecycle | Partial / blocked native | Flag exposed in recorded Properties adapter; host ownership/visibility tests pass. Native CUDA/OpenGL/Vulkan drawing, cleanup and fallbacks remain unqualified. |
| Begin Production Bake with safe path / active ownership | Partial / blocked native | Source poll/confirmation review and host bake-safety tests. No actual ProductionBake CUDA run in this lane. |
| Cancel → verified Resume / animation change → Re-bake preserving prefix | Blocked | Root's actual-bake fresh-process harness and no-CUDA preflight report remain separate. Synthetic UI caches explicitly lack current full input identity and correctly offer Restart, not fabricated Resume. |
| Paused cache: whitewater Separate/Both, display fraction, HEIGHT attribute | Pass, native background | 12 paused-cache checks: three whitewater types, immediate 6→3 particle fraction, varying display colors, no reread for radius-only edit, frame unchanged, no Warp/solver. |
| Cache → live iteration without command-search workaround | **Fail U2** | Missing direct switch; existing public Stop Cache Playback succeeds. |
| Missing cached frame produces visible actionable viewport error | **Fail U3** | Real missing frame8 error absent from sidebar/header; retained in Properties. |
| Recover from missing cached frame | Pass, native background | Return to valid frame7 republishes six particles and clears error. Retry policy preserved. |
| Save/reopen with object-owned solver settings and display preferences | Pass, native background | Native save and reopen preserve max_substeps, ADVANCED and OFF; handlers remain registered; no live solver allocated. No GPU-state restoration claim. |
| Export cache surface as Alembic and read back animation | Pass, native background | 25 assertions; public exporter, real Alembic import, four frames' vertex counts/bounds, empty/range refusal and scene-state restoration; synthetic cache. |
| Export surface/particles as USD with varying counts and attributes | Pass, native background | 35 assertions; real exporter and pxr readback, 300/2600/40/1800 particle sequence, speeds/positions/nonzero widths, surface counts and restored state; synthetic cache. |
| Export level set as VDB with baked world origin | Pass, native background | 12 assertions; public exporter → Blender OpenVDB reader/mesher; two frames, geometry/bounds, unchanged cache and scene; moved-domain legacy-origin refusal. |
| Export while live solver is active, including paused Live | Source/host evidence only here | Public idle guard intentionally refuses active sessions. Prior internal ordering concern remains withdrawn. No active-CUDA export acceptance claimed. |
| Foreground widths/DPI/contrast/keymaps/hover/confirmation/file browser | Blocked | No usable foreground session; no screenshots. Must inspect exact installed package on certified host. |

## Executed checks and retained artifacts

1. `run_native.py`: existing quick-setup and paused-cache acceptances, exit0/markers confirmed. Paused cache **12 checks passed**. `native_runs.json` retains exact subprocess commands/log paths.
2. `run_ui_probe.py`: **15 native checks passed**, **204 recorded panel/Preferences draws**, save/reopen; `native_ui_probe.json`.
3. `run_findings.py`: **9 checks passed**, meaning all three findings and the successful contrasting recovery/control lanes were reproduced; `workflow_findings.json`. These are finding confirmations, not product acceptance passes.
4. `run_exports.py`: wrappers invoke the repository acceptance `main` functions directly so their output archives/caches are retained instead of deleted by their script entrypoint cleanup. No exporter is replaced. Alembic **25**, USD **35**, VDB **12** assertions passed. See `export_runs.json`, `{alembic,usd,vdb}/process.log`, and `{alembic,usd,vdb}-artifacts/`. Native archives and independent readback results are retained.
5. Pinned Python `/workspace/scratch/fluidsim-review-80bf4aa-20261106/venv/bin/python`, `CUDA_VISIBLE_DEVICES=-1`: `pytest -q tests/ui/test_operator_panel_behavior.py tests/ui/test_phase4_pending_interaction.py tests/ui/test_deck_parity.py tests/ui/test_deck_state_follow_through.py tests/ui/test_artist_apply.py tests/ui/test_artist_bake_safety.py tests/ui/test_deck_palette_follow_through.py tests/ui/test_liquid_preview.py tests/runtime/test_pending_settings_registry.py tests/test_gpu_particle_owner.py` → **370 passed**, four existing package reassignment deprecation warnings, 5.99s. Full log `host-regressions.log`. This is host coverage, not live GUI/CUDA.

Failed exploratory artifacts are retained: `native_ui/probe-draft1.log` and `probe-draft2.log` are reviewer harness mistakes (Preferences proxy missing its own fields; wrong panel collection constant), corrected before the final native probe. `followthrough/native_ui_probe.json` has **16/17 passing assertions**: its speculation that a missing cached frame would disarm playback was false. Source explicitly retains cache retry behavior; that is **not reported as a product defect**. Its exit0 is not treated as a passing report. The final focused reproducer asserts the actual contract and records successful recovery.

Read the export continuation review as corroborating context only: `/workspace/scratch/fluidsim-remediation-20261106/continuation-export-review.md`. PR49/51/62 are already included in this source. This review does not revive the withdrawn active-session/internal-ordering concern, nor rely on stale integration statements in the earlier remediation document.

## Remaining acceptance and retention

Fix/review U1–U3 while preserving the existing independent owners, then run the focused native story probes and the real foreground loops. Certified Blender5.2.0/Warp1.19/CUDA execution must still cover actual live animation, pending-setting restart confirmation, modal cancel/resume/rebake, optional GPU preview lifecycle, installed Preferences matrix and final-source/package GUI/export interactions. Existing CPU and synthetic-cache results cannot close those lanes.

All artifacts and the isolated source worktree are retained under the owned evidence root; no cleanup was performed. No production release or end-to-end GUI success is claimed.
