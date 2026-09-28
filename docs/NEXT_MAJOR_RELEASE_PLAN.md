# FluidSim Warp 2.0: status review and major-release plan

Status as of 2026-09-27, `master` at `0a89232`. Every finding below was checked against
the live code, git history, or a fresh run in this review; claims taken from docs are
marked as such.

## 1. Summary

The solver and the artist workflow are complete for what the product set out to do,
and they are well verified **on one machine**. What stands between 1.0.255-beta and
a stable 2.0 is mostly release engineering:

- **Verification that runs by itself.** The CI workflow never runs.
- **A current platform.** Blender 5.2 LTS is out and Warp is pinned to an exact older version.
- **An installation an artist can complete.** Warp must be hand-installed into Blender's Python.
- **One security fix and one regressed defect.**
- **The compatibility debt** accumulated across the 1.x series.

**Recommendation:** make 2.0 a *stable, installable, LTS-certified* release rather than a
feature release. Use the major-version break to delete the legacy compatibility surface.
Ship a final 1.x "bridge" release first, so users' files migrate before that code goes
away. New features (species v2, Alembic/VDB export, frontier performance) go to 2.1+.

## 2. Current status

### Strengths (verified)

| Area | Evidence |
|---|---|
| Test suite | `python -m pytest -q -p no:cacheprovider`: **1,565 passed, 0 failed**, 5 m 12 s (this review, Windows, HEAD `0a89232`) |
| Physics validity | `demos/ANALYSIS.md`: **18/18 PASS**. Dam-break front tracks shallow-water theory at 1.01·√(gH); crown radius follows √t (r = 0.971) |
| Determinism | Demo bakes bitwise-identical since `d2a6b6d`; full GUI gate reproduced 3/3 with a stable artifact hash |
| Feature coverage | Everything in `PROMPT.md` exists: sparse tiled grid, PIC/FLIP, 5 particle trace modes up to RK3, sort/bin/relax, all source shapes, moving colliders, sinks, Jacobi/PCG, viscosity, SDF surface, whitewater, forces, custom kernels, profiling, production bake, command-line bake, Geometry Nodes render assets |
| Performance | 4 profiled optimizations retained (surface attribute transfer 175.5 → 20.7 ms/frame). Viscosity overhead cut from 7.80× to 1.55× with CUDA conditional graphs |
| Architecture | The architecture-deepening, resolution-quality, fail-closed capacity and settings-split (phases 0–2) plans are all implemented. `scene_bridge.py` went from a 2,016-line mixed owner to 567 lines |

### Gaps (verified)

| # | Gap | Evidence |
|---|---|---|
| G1 | **No working CI.** The workflow sits at `fluidSim_warp_blender51/.github/workflows/` (GitHub only reads the repo root), triggers on `main` (the branch is `master`), names a stale `1.0.253` artifact and needs a self-hosted runner. It has never run: the repo's Actions history is 18 runs of one-off automation workflows on side branches (2026-08-23/25), mostly failed or cancelled, and none on `master` | `release-checks.yml` |
| G2 | **Unreleased work.** 181 of 387 non-merge commits (47%) landed after the 1.0.255 build (2026-08-22): UI redesign, settings split, artist workflows, determinism, paddle calibration and performance. The version was never bumped, the CHANGELOG stops at 1.0.255, and no release tags exist | `git rev-list`, `version.py`, `CHANGELOG.md` |
| G3 | **Security (P0).** Custom Effects writes generated code to the shared system temp dir (`<tmp>/fluidsim_warp_effects_v2`) and then imports and executes it. On a multi-user Linux host, another local user can plant or swap that file | `sim/effects/source.py:45-60` → `sim/effects/compiler.py:56-68`; the only caller passes no private dir (`compiler.py:180`) |
| G4 | **Defect #5 regressed by the settings split.** Pending-change detection reads 35 fields from `scene.fluidsim`, but they now exist only on `FLUIDSIM_PG_domain`: `height_*` (8), `*_attribute_*` (25) and `profiling_enabled/sync`. The lookup always returns `None`, so editing them never raises "Reset/Reinitialize pending". Found by static check of the class declarations; not yet reproduced in Blender. The existing test uses a fake scene that hides it | `blender/pending_settings.py:415-440` |
| G5 | **Single-host evidence.** CUDA proof exists only for Windows + Blender 5.1 + RTX 4080 SUPER. Linux was last validated with Blender 5.2 LTS and **Warp on CPU**, and the changelog itself lists Linux CUDA as an open "external gate" | `CHANGELOG.md` 1.0.251/1.0.253 |
| G6 | **Platform drift.** Blender 5.2 LTS (released 2026-07-14, supported to July 2028) is uncertified. Its Geometry Nodes input path is coded but untested, and it *silently* skips an input if both API paths fail. Warp is pinned to exactly `1.16.0`, and any other version disables the add-on. Warp 1.17 is current; 1.18 moves PyPI wheels to CUDA 13 (R580+ driver, Turing/sm_75+) | `blender/geometry_nodes.py:137-158`, `runtime/warp_runtime.py:11,106-119` |
| G7 | **Installation.** Artists must pip-install a ~150 MB CUDA wheel of one exact version into Blender's Python. The user guide says to "provision" Warp but gives no steps | `docs/user_guide.md:21-27`, `tools/release/delivery_policy.py` |
| G8 | **Hidden numerical failures.** ST Jacobi reports residual −1 and "valid" (REVIEW-004). The native PCG fallback is never shown to the user (REVIEW-016). NaN face weights pass the MAC sanitize (REVIEW-014) | `sim/pressure.py:1840-1843, 2115-2118`, `sim/solver.py:295-316` |
| G9 | **ST-FLIP is advertised but has not passed its own gates.** The tagline says "beta ST-FLIP", and enabling it switches on Variable Projection (Beta) by default. The 1.0.234 assessment advised against that, and its reference-error and conservation gates are still open. *Closed in 2.0: the tagline dropped ST-FLIP (`13929b3`); D6 labels ST-FLIP Experimental with Variable Projection off by default (`3ba90d5`)* | `blender_manifest.toml` tagline, `properties.py` `spacetime_variable_projection`, `FEATURE_VALUE_ASSESSMENT_1.0.234.md` |
| G10 | **Compatibility debt.** About 3–4k LOC of migrations and shims plus dead code: legacy snippets (`sim/custom.py`, 1,355 LOC, runtime already retired), schema-0 settings migration, source schema < v4, `LEGACY_ABSOLUTE` sampling, cache-v1 playback, 13 unused enums including MultiGrid | agent audit, spot-checked |
| G11 | **Gate coverage.** 22 of 41 `tests/blender_*` acceptance scripts are not run by `tools/run_production_gate.py`, and 5 test files contain no test functions. *Closed in 2.0 (REVIEW-009): every Blender acceptance script and release validator is gated or excused with a reason, pinned by `tests/test_production_gate_coverage.py`* | REVIEW-009 triage |
| G12 | **Docs drift.** Among 195 tracked `.md` files:<br>• finished plans still say "not started" or "uncommitted"<br>• the user guide cites Linux-only evidence and "1,033 tests"<br>• `CONTEXT.md` says bakes are not bitwise-repeatable<br>• the lock file says "validated in Blender 5.2.0 LTS" while the certified minimum is 5.1 | per-document audit |

### Backlog (all 50 REVIEW cards re-triaged against live code)

| Status | Count | Cards |
|---|---|---|
| Open P0 | 1 | 006 (temp-dir `exec_module`; moved from dead legacy code into live Custom Effects) |
| Open P1 | 5 | 004, 011, 016, 017, 022 |
| Open P2 | 23 | 008, 009, 013, 014, 015, 019, 021, 024, 025, 026, 027, 028, 029, 031, 036, 037, 038, 041, 042, 044, 045, 046, 047 |
| Fixed, card not closed | 14 | 001, 002, 003, 005, 007, 012, 018, 020, 023, 030, 033, 034, 035, 048 |
| Obsolete | 7 | 010, 032, 039, 040, 043, 049, 050 |

`TASK-WORKTREE-PRUNE-001`: the physical cleanup is done, but its archive location and
the deleted experiment refs are unconfirmed. All `TASK-POLICY-DRIFT-*` cards are closed.

### Earlier user-reported defects

| Defect | Status |
|---|---|
| #1: presets/quality/live settings applied to stale scene settings | **Fixed** (`f347528`, `b57b7cf`, `1832f08`, `c4d18b4`, `b6fe26c`) |
| #5: pending detection reads stale settings | **Partially fixed, then regressed for 35 fields** (G4) |
| #2: slowdown compounding per reset | **Open, parked, not reproduced.** Clue: noticed with viscosity + surface tension on; no probe has run with those on |
| #4: large error from custom snippets | **Largely addressed** by retiring the legacy runtime; not reproduced. Compiler errors are still not summarized (`sim/effects/compiler.py:76`) |

## 3. What 2.0 means

- **Stable channel:** version `2.0.0`, `BUILD_CHANNEL = "stable"`, tagged in git.
- **Certified matrix:** one Blender LTS line plus one Warp line, each proven by a recorded
  GPU gate run on every platform the manifest lists.
- **Installable:** from a clean Blender profile to a running simulation using only documented steps.
- **One owner per setting:** each authored field is declared on exactly one PropertyGroup.
  The UI writes it where the solver reads it, and change detection reads the same place.
  This removes the class of bug behind defects #1 and #5.
- **Documented break:** 2.0 opens files saved by the 1.x bridge release or later. Older
  files are migrated by opening and saving them once in the bridge release.
- **Honest labels:** anything that hasn't passed its gates is labelled Experimental and not advertised.

**Non-goals for 2.0:** new solver features, species v2, Alembic/VDB export, macOS/CPU
support, multigrid, broad refactors of large modules that nothing else touches.

## 4. Workstreams

Sizes are relative effort: S is a day or two, M about a week, L several weeks. Each
workstream lists its exit evidence.

### WS1: Release engineering and CI (M)

1. Move the workflow to the repo-root `.github/workflows/`, trigger it on `master` and on
   PRs, and derive artifact names from `version.py`.
2. **Hosted CPU job** (free GitHub runners, every PR):
   - ruff
   - the non-CUDA pytest subset
   - `build_extension.py` packaging
   - delivery policy
   - manifest and version-consistency checks
3. **GPU job:** either a self-hosted runner on the RTX 4080 SUPER box, or a required manual
   gate whose manifest is attached to the release (decision D9).
4. Test infrastructure:
   - a shared `cuda_device` fixture
   - `cuda`/`blender`/`slow` markers
   - a `FLUIDSIM_TEST_DEVICE` override (REVIEW-026, -044)
   - fix or delete the 5 empty test files
5. Wire the 22 unwired Blender acceptance scripts into the gate, or delete them with a reason (REVIEW-009).
6. Tag every release; bump the version whenever `master` ships.

*Exit:* a PR cannot merge with red CPU CI; a release tag carries a GPU-gate manifest for each certified platform.

### WS2: Correctness and security blockers (M)

| Item | Fix | Test |
|---|---|---|
| REVIEW-006 (P0) | Generate Custom Effects modules into a per-user private dir (the Blender extension user dir, mode 0700) and check ownership before import. In the same pass, do REVIEW-019 (cubin integrity) and REVIEW-021 (cache path containment) | Planted-file test on the generated-module path |
| Defect #5 regression | Route each field by the class that declares it, not by a hand-kept list | Assert the signature field sets against the *real* `FLUIDSIM_PG_scene`/`FLUIDSIM_PG_domain` annotations, and a Blender test that edits `height_enabled` and expects "pending" |
| REVIEW-004, -016, -014 | Report real residuals, show native-PCG fallback in stats/UI, reject NaN weights | Unit tests per path |
| REVIEW-017, -022, -011 | Whitewater collision normal fallback; memoize playback metadata and scene hash per cache identity; UNDO on capacity apply | Unit test; playback per-frame timing; operator test |
| Defect #2 | One bounded repro attempt with viscosity + surface tension on across repeated resets. If it's not reproduced, document it as a known issue | Timing probe |

*Exit:* no open P0/P1, and each fix has a regression test that fails before the change.

### WS3: Platform certification (M)

1. Install Blender 5.2 LTS and run the full GUI gate, demo analysis and `check_demo_blends` on it.
2. Make the Geometry Nodes input setter fail loudly (log and test) instead of silently skipping.
3. Move from Warp 1.16.0 to 1.17.x (cu12). The 1.17 changelog has no breaking changes on
   paths this add-on uses. Its breaking items are `jax_kernel` and `load_module` on empty modules.
4. Replace the exact-version pin with a tested range: block below the floor, warn on untested newer versions.
5. Re-baseline the determinism hash and demo analysis after the Warp bump (numerics may shift);
   18/18 must still pass.
6. Linux: run the GPU gate on a Linux + NVIDIA host, or ship Linux as "packaged, not certified" (D4).
7. Set ruff's target to the Python bundled by the certified Blender (3.13 in 5.1; confirm for 5.2).

*Exit:* a gate manifest for each certified Blender × Warp × OS combination, recorded in the release notes.

### WS4: Installation and first run (M–L, depends on D3)

The single biggest barrier to adoption. The options:

- **(a) Bundle the hash-locked Warp wheels** through the extension manifest's `wheels`
  (split-platform builds already exist). Adds ~150 MB per platform. This reverses the
  1.0.245 "external runtime" policy, which is encoded in `delivery_policy.py` and
  `test_manifest_wheels.py`; the reason for that policy isn't recorded.
- **(b) Guided installer:** an operator that installs the locked wheel into a private
  site-packages after verifying its hash. Needs a declared network permission.
- **(c) Documentation only:** step-by-step commands per OS, plus a preflight message that
  names the exact fix.

Recommended: **(a)**, if the size is acceptable for the distribution channel. Otherwise (b).
Do (c) regardless.

*Exit:* a clean-profile install on Windows and Linux reaches a running dam-break preview
using only the written steps, performed and recorded.

### WS5: Breaking cleanup (L)

Only after the bridge release (Phase 0) has shipped.

1. **One-owner settings model** (split proposal phases 3–5):
   - remove `FLUIDSIM_PG_legacy_settings`, `legacy_scene_settings()`, `migration.py` and the
     schema-0 load_post hook
   - remove the `setting_owner`/`_authoring_owner` divergence and the 131 defaulted `getattr`s
2. **Delete dead and legacy code:**
   - `CompiledSnippet`/`SnippetManager` and the `FLUIDSIM_PG_snippet*` RNA
   - `blender/snippets.py` and the retired operator forwards
   - `write_sequence_metadata`
   - the 13 unused enums, including MultiGrid/V_CYCLE, and the unused `k_ww_compact_scatter` argument
3. **Remove source and cache compatibility paths:**
   - source schema < v4, `sim/source_compat.py`, `LEGACY_ABSOLUTE` sampling
   - cache-v1 playback (bridge-release users re-bake), legacy output-object adoption
4. **Remove UI choices that can only fail, or finish them:** height-field `INFINITE`,
   FIELD source mode (emits no liquid), compiled FORCE plans.
5. **ST-FLIP (D6):** label it Experimental, drop it from the tagline, and default Variable Projection to off.
6. **Settings drift:** one pass over defaults and subtypes (REVIEW-028, -029, -037, -045, -046, -047).
7. **Optional (D8):** rename the package folder `fluidSim_warp_blender51` → `fluidsim_warp`,
   since the old name encodes a Blender version 2.0 won't target.

*Exit:*
- the full suite and gate are green
- `ownership_map` shows no field declared on two groups
- a 1.x-bridge-saved copy of each demo opens in 2.0 and bakes to the **same hash** as the
  2.0 build of that demo. Determinism turns this migration proof into an exact check.

### WS6: Documentation and repo hygiene (S–M)

- CHANGELOG entries for everything since 1.0.255, a "Known limitations" section, and a
  1.x → 2.0 migration guide.
- User guide: install steps, the real certified matrix, and no hard-coded test counts.
  Fix the `CONTEXT.md` determinism statement and the lock-file provenance text.
- Move finished plans to `docs/archive/` with a one-line status header. Keep `docs/` to
  the documents a new contributor needs.
- Close the 21 fixed/obsolete REVIEW cards, resolve PRUNE-001, and mark 032/050 won't-fix
  (they conflict with the enforced undo policy).

### WS7: Performance (2.1, unless D7 says otherwise)

Topology frontier (6 tasks, the next-highest ROI in the results doc), then the pressure
follow-on (5 tasks), the rest of the device-resident surface work, cache-cursor
integration, and viscosity SDF reuse. **For 2.0 itself:** add a performance budget to the
gate. The 4 demos may be at most 5% slower per frame than the 1.0.255 baseline.

## 5. Sequence

| Phase | Contents | Ships | Gate to leave the phase |
|---|---|---|---|
| **0. Bridge** | WS1 items 1–2 and 6; WS2 REVIEW-006 and defect #5; WS6 CHANGELOG catch-up and card closing | **1.1.0**: last 1.x release, carries every migration | CPU CI green on PRs; full GUI gate + demo analysis pass; tag pushed |
| **1. Platform** | WS3, WS4, WS1 items 3–5 | 2.0.0-alpha builds | Gate manifests for the chosen matrix; clean-install walkthrough recorded |
| **2. Break** | WS5 | 2.0.0-beta | Migration-equivalence hashes match; no dual-owned fields |
| **3. Harden** | WS2 remaining P1s and defect #2 attempt; chosen P2s; performance budget | 2.0.0-rc | No open P0/P1; performance within budget |
| **4. Release** | WS6 docs and migration guide; RC used on real scenes | **2.0.0** | Every item in §6 checked |

Phases 1 and 2 can overlap once the bridge has shipped. WS2 fixes for hidden numerics
can move earlier at any point, since they don't depend on the break.

## 6. Release checklist for 2.0.0

- [ ] CI: CPU job green on `master`; GPU gate manifest for every certified platform.
- [ ] Platform: Blender 5.2 LTS (and 5.1 only if D1 keeps it) × Warp 1.17.x certified;
      Linux either GPU-proven or labelled uncertified.
- [ ] 0 open P0/P1 cards, each fix with a regression test.
- [ ] Demo analysis 18/18 PASS on the certified stack; GUI gate reproducible 3/3 with the new hash.
- [ ] Performance within 5% of the 1.0.255 baseline on all 4 demos.
- [ ] Clean-profile install walkthrough performed on Windows and Linux.
- [ ] Bridge-saved demos bake to hash-identical results in 2.0.
- [ ] CHANGELOG, user guide, README, manifest and `version.py` agree; migration guide published.
- [ ] Experimental features are labelled and absent from the tagline.

## 7. Decisions needed

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | Which Blender versions 2.0 supports | **5.2 LTS only** (`blender_version_min = 5.2.0`) | Supported to July 2028; one line to certify. 5.1 is not an LTS release |
| D2 | Warp line | **1.17.x cu12 with a range pin** | Current and non-breaking for this code. 1.18's CUDA 13 wheels need R580+ drivers and drop GPUs older than RTX 20-series |
| D3 | Warp delivery | **Bundle the locked wheels** (reverses 1.0.245), else a guided installer | Removes the #1 install failure; the wheels are already hash-locked |
| D4 | Linux | **Certify it if a Linux + NVIDIA box is available; otherwise ship it labelled uncertified** | No Linux CUDA run has ever been recorded |
| D5 | File compatibility | **2.0 opens 1.1+ files; older files go through 1.1** | Lets WS5 delete ~3–4k LOC safely |
| D6 | ST-FLIP | **Experimental in 2.0**; promote in 2.x after its gates | Reference/conservation gates never ran |
| D7 | Frontier performance work | **2.1** | Keeps 2.0 about stability; the solver is already interactive on the target GPU |
| D8 | Rename package folder | Optional; do it in Phase 2 if at all | Name encodes Blender 5.1; churns every path |
| D9 | GPU CI | **Required manual gate with attached manifest** to start, self-hosted runner later | The gate already writes manifests; a runner on a desktop is fragile |

## 8. Risks

- **One GPU, one machine.** 8 GB cards and older architectures are untested. Mitigation:
  record VRAM headroom per demo and state a minimum GPU in the release notes.
- **Blender 5.2 API drift beyond Geometry Nodes** stays unknown until the gate runs there.
  Do WS3 item 1 first in Phase 1.
- **The Warp bump may change numerics**, which invalidates the determinism hash and could
  shift demo metrics. The re-baseline is planned (WS3 item 5), not a surprise.
- **Deleting migrations strands old files** if users skip 1.1. Mitigation: the migration
  guide, plus a clear error on schema-0 files that names 1.1.
- **Defect #2 may be real and user-visible.** It's bounded by one repro attempt, then documented.
- **Bundled wheels add ~150 MB per platform**; check the distribution channel's size limit before committing to D3(a).

## 9. After 2.0 (2.1+ candidates)

- Topology frontier and pressure follow-on performance.
- Species v2 (foam/spray/bubble). Only in archived experiments today, so it needs a fresh port and your approval.
- Alembic/VDB export (currently OBJ/PLY only).
- ST-FLIP promotion.
- Narrow-band particles; resolution divisions instead of voxel size.
- Pressure-field sourcing.
- Multigrid.

## Sources

- Blender 5.2 LTS release: https://www.blender.org/releases/5-2/
- Blender 5.2 Python API notes: https://developer.blender.org/docs/release_notes/5.2/python_api/
- Warp changelog (1.17.0, 2026-08-31): https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md
- Warp 1.18 CUDA 13 wheel change: https://github.com/NVIDIA/warp/issues/1955
