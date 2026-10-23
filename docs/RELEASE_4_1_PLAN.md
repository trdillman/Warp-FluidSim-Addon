# 4.1.0 Plan: Archive Review and Integration

Date: 2026-10-16, re-ranked 2026-10-17 · Baseline: origin/master `454839e` (v4.0.0)

## 0. Status on master (2026-10-23, `8cd30c0`)

Master already says 4.1.0 (`blender_manifest.toml`, CHANGELOG, user guide and
technical report). There is no `v4.1.0` tag or GitHub release yet; the last
tag is `v4.0.1` at `46cdf3c`. What shipped is narrower than §3: Wave 0 plus
new features (Body Type, rigid contact, five demos, surface-tension fixes,
the integrated settings UI). Waves A–C are mostly still open.

**Done**
- #1 4.0.1 installer branch: merged and tagged `v4.0.1` (`46cdf3c`).
- #2 Rewind crash and emptied-action fixes: landed with the rigid demos
  (in the 4.1.0 CHANGELOG Fixes).
- #3 Transfer clarity UI: `f007c58`.
- #7 Rigid demos: decided as Demo Gallery scenes (Density Ladder, Water
  Wheel), `9c796e5`, with the real-UI rigid authoring test.
- #9 Real-Blender installer test: closed by removing it (`10dfca9`); the
  faked-download tests cover the installer.
- Version bump to 4.1.0 (`70585dd`) and the 4.1.0 CHANGELOG entry.
- Integrated UI evidence: `docs/verification/ui41-20261020.json`.

**Not in the plan, but landed for 4.1**
- Collider Body Type (Static/Animated/Dynamic) and rigid-body contact (`11df9d2`).
- Viscosity Race, Deep Plunge, Zero-G Droplet demos; fine-grid surface
  tension fix (`7bb7b51`, `5396f87`).
- Every domain setting shown in the Scene tab and All Simulation Settings
  (PR #33).

**Still open**
- #4 Span cap: the setting is still in `properties.py:1607` and its hover
  text still says "removed in 4.1". Remove it, or change the text to 4.2.
- #5 `CONTRIBUTING.md:9` still names `.github/workflows/ci.yml`, which does
  not exist.
- #6 AGENTS.md Cloud Agent notes: not cherry-picked.
- #8 Publication buffer fix, #11 cloud UI smoke lane, #15 Alembic point
  clouds, #18 benchmark: not started.
- #10 Warp decision: the lock pins Warp 1.18.0 (not 1.17 as §2 says); the
  decision is not recorded here.
- #12–#14, #16 Solver wave (MG warm start, perf headroom, crown metric,
  residual margin): not started. Proposed: move to 4.2.
- #17 Linux certification: not done; listed as a known limitation in the
  4.1.0 CHANGELOG.
- Release steps (Wave D): the full GUI gate with hash match on the 4.1
  demos, build, install probe and checksums per `docs/RELEASING.md`, then
  the `v4.1.0` tag and GitHub release.

Scope: all 41 `archive/*` tags, the `release/4.0` worktree (clean), the
uncommitted cloud-UI work in the main checkout, deferred items recorded
in the 4.0 docs, and the three local branches started after 4.0.0
(`feat/warp-install-from-file`, `feat/rigid-body-demos`,
`feat/transfer-ui-clarity`). Method: `git cherry` / `git log` /
`git diff --stat` against origin/master and `git grep` for each feature.
Nothing was built or run for this review. Effort is estimated hours,
including tests and a GUI gate where the item changes sim output.

## 1. Summary

- **Three branches since 4.0.0 are finished but not merged**:
  - The 4.0.1 installer branch: Install from File, resumable downloads, and
    the rigid rotation-resume fix.
  - The Transfer clarity UI.
  - The rigid demos branch. It also fixes a crash on rewind.
  Merging them is now the cheapest, highest-value work.
- **Both the installer and rigid-demos branches bump the version to 4.0.1.**
  Merge the installer branch first, then rebase rigid-demos.
- **The rigid-demos branch adds two Demo Gallery scenes**, Density Ladder
  and Water Wheel. This conflicts with the rule that there are only four
  demos, so it needs a decision before it lands: make them examples, or
  widen the rule.
- **25 tags are already on master or superseded** (section 5).
- **Debt carried from 4.0**: Linux certification, little perf headroom,
  MG warm start, Alembic point clouds, a stale `ci.yml` reference, the
  span-cap removal promised for 4.1, and the Warp 1.17 decision.

## 2. Ranked by ROI (value ÷ effort)

| Rank | Item | Source | Value | Effort (h) | Changes sim output? | Status |
|---|---|---|---|---|---|---|
| 1 | Merge 4.0.1: Install Warp from File, resumable download, rigid rotations restored bit-exactly on resume | `feat/warp-install-from-file` 46cdf3c (pushed) | H | 0.5–1 | No (resume fix only) | Done; needs a PR and merge |
| 2 | Rewind crash fix (Reset freed the pose action inside the frame handler) + emptied-action fix | `feat/rigid-body-demos` 2b2877c | H | 1–2 to split from the demos | No | Done, bundled with the demos; split or land together |
| 3 | Transfer clarity UI: dropdown beside FLIP Ratio, tooltips, guide. APIC default declined (§6) | `feat/transfer-ui-clarity` | M | 0.25 | No | Done; uncommitted |
| 4 | Remove the surface span cap (CHANGELOG says "removed in 4.1") | CHANGELOG 4.0.0 | L-M | 1–2 | No | Open |
| 5 | Fix the stale `ci.yml` claim in `CONTRIBUTING.md:9` | origin docs | M | 1 | No | Open |
| 6 | Cherry-pick the Cloud Agent toolchain notes into `AGENTS.md` (4.0.1 also edits it: small conflict) | `archive/cloud-agent-env-notes-20261014` | L | 0.25 | No | Open |
| 7 | Rigid demos: decide examples or demos, then land the scenes and the real-UI rigid authoring test | `feat/rigid-body-demos` | M | 2–4 | No (new scenes) | Built; **needs a decision** |
| 8 | Publication buffer fix + buffer test + 2-frame viewport gate (leave out the demo .blend changes) | `archive/wt-publication-gate-20261001-20261010` | M | 3–5 | Low | Open |
| 9 | Real-Blender Warp installer test: 4.0.1 already adds a 63-line version; port only what remains of the 572-line archive | `archive/pr19-integration-snapshot-20261014` | M | 1–3 (was 3–5) | No | Partly done |
| 10 | Warp 1.17 decision: keep it or re-decide (accepted "for 4.0 only"; 4.0.1 still pins 1.17) | `RELEASE_4_0_PLAN.md` Open decisions 1 | M | 2–4 | No | Open |
| 11 | Cloud UI smoke lane: Windows skip for `fcntl`, rebase onto 4.0, fresh evidence; keep only the cloud_ui block of the local `.gitignore`/`RELEASING.md` diffs | local untracked files | L-M | 2–3 | No | Open |
| 12 | MG warm start from the previous frame's pressure ("next lever"); also restores perf headroom | `RELEASE_4_0_ACCEPTANCE.md` L83 | H | 8–12 | Yes (rebind) | Open |
| 13 | Perf headroom: crown_splash 1.132× vs 1.155 budget; dam_break swings 0.87–1.08× on the same code | `RELEASE_4_0_ACCEPTANCE.md` | H | 8–16 | Maybe | Open (with #12) |
| 14 | Make the crown metric and hash rebind less fragile (`crown_sqrt_t_corr` 0.971) | `technical_report.md:659`, `analyze_demos.py:369` | M | 4–6 | No | Open (pairs with #12) |
| 15 | Alembic: point clouds export as empty transforms on 5.2; surface only | CHANGELOG 4.0.0 | M | 4–8 | No | Open |
| 16 | MG residual ratios near the 6× limit (chromatic 5.42×) | `RELEASE_4_0_ACCEPTANCE.md` | M | 4–8 | Yes | Open (with #12) |
| 17 | Linux CI + Linux USD export certification (moved to 4.1) | `RELEASE_4_0_ACCEPTANCE.md`, CHANGELOG | H | 16–24 | No | Needs a Linux CUDA host |
| 18 | Cache-playback validation benchmark | `archive/codex/cache-cursor-recovery-20260827` | L-M | 2–4 | No | Optional |
| 19 | Coastal Cascade **or** Ocean Swell scene, shipped as an *example* | `archive/coastal-cascade-delivery-20261014`, `archive/ocean-swell-delivery-20261014` | M | 15–25 each | No (new scene) | Optional, one only |
| 20 | Species-v2 extended transport | `archive/codex/species-v2` | M | 16–30 | Yes, opt-in | Defer to 4.2 |
| 21 | Guide to Target effect | `RELEASE_4_0_PLAN.md` §8 | M | 12–20 | Opt-in | Defer |
| 22 | Rigid body-to-body / body-to-static contact | CHANGELOG 4.0.0 | M | 16+ | Opt-in | Defer |
| 23 | ST-FLIP: needs a conservation reference gate first | `archive/worktree-stflip-implementation` | M | 20–40 | Yes | Defer |
| 24 | Effect Graph node editor | `RELEASE_4_0_PLAN.md` §8 | M | 40+ | No | Defer |
| 25 | Native GPU interop requalification | `archive/wt-interop-requalification-*`, `codex/native-gpu-interop*` | L | 40+ | High risk | No: declined in 4.0 plan |
| — | Adaptive grids, `wp.Tape` fitting, zero-copy viewport | `RELEASE_4_0_PLAN.md` | L-M | 80+ / research | — | No |

## 3. 4.1.0 scope (balanced cut)

- **In scope:** items 1–17, about **65–110 h**. Items 1–3 cost about 2 h of
  that, because the work is already done.
- **Stretch:** items 18 and 19.
- **Shape:** the release keeps the 4.0 shape: one sim-changing batch, one hash
  rebind, one GUI gate.

## 4. Integration plan

Work from a `release/4.1` branch off `454839e`, using one worktree per lane
and merging in this order. Sim-changing work is grouped so that the demo
hash is rebound once.

**Wave 0: land finished branches (~half a day)**
- #1 merge `feat/warp-install-from-file` (as 4.0.1, or fold into 4.1).
- #3 commit and merge `feat/transfer-ui-clarity`.
- #2/#7 rebase `feat/rigid-body-demos` onto that; after the demos-vs-examples
  decision, land it whole or split the crash fix out.

**Wave A: no sim change (parallel, ~1–2 days)**
- A1 Docs/hygiene: #5 CONTRIBUTING, #6 AGENTS.md cherry-pick, #4 span-cap removal.
- A2 Publication fix #8: re-apply the `output_publication.py` change, test and
  2-frame gate onto 4.0; drop the archive's demo .blend/.png edits.
- A3 Tooling: #9 installer test remainder, #11 cloud UI lane (`.gitignore`
  must keep `!/tests/**/fixtures/*.npz`), #18 benchmark if time allows.
- A4 Export: #15 Alembic point clouds.
- Check: focused pytest per lane plus the `--quick` gate. The demo hash must
  still match `454839e`'s.

**Wave B: solver (serial, ~1 week)**
- B1 #12 MG warm start, then #16 residual margin. Measure paired against 4.0
  on all four demos.
- B2 #13 perf headroom: target crown ≤ 1.10× and a dam_break variance study
  (fix the measurement before the code).
- B3 #14 make crown metrics robust before rebinding.
- Check: paired benchmark, volume conservation on chromatic_vortex (the 4.0
  −2.9% trap), then rebind the gate reference.

**Wave C: decisions**
- #10 Warp decision, recorded in the plan doc. APIC stays opt-in (§6).

**Wave D: release**
- #17 Linux certification if a Linux CUDA host is available. Otherwise carry
  it again and say so in Known limitations.
- Version bump 4.1.0, CHANGELOG, one full GUI gate with hash match, then build
  and install-probe per `docs/RELEASING.md`.

**Do not land:**
- `wt-r40-mg`: it reverts the 4.0 MG tuning that fixed volume loss.
- The `rigid_coupling.py` part of `performance-budget-codex`: it brings back
  a per-substep host round trip.
- Any archive's re-saved demo .blend files.

## 5. Archives with nothing to land

Already on master: deck-risks-40, export-live-session, nearest-values-gpu,
usd-carried-40, post30-append (another form), post30-diagnosis, post30-vdb,
st-flip-final, fluidsim-1.0.252-superseded, release-prep-1.0.255.

Superseded or obsolete:
- r30-append-vdb, wt-r30-hash, wt-r30-i (except possibly
  `docs/FOLLOWUP_3_0_20261010.md`).
- local-release-gate, paired-budget-proposal, release-2.1.1-wip,
  performance-budget-codex, wt-performance-fix (rerun evidence only).
- wt-perfreview: a CPU refactor of the nearest lookup, superseded by the GPU
  version that shipped. Check before deleting.
- coastal-reference: overlaps cascade, "unqualified".
- wt-codex-linux-cert: notes for #17.
- automation/* (3), forensics/*, claude/particle-surface-rendering-apis,
  fluidsim-1.0.248-reconciled, codex/native-gpu-interop-main.

Recommend keeping all tags (cheap); no deletions are proposed here.

## 6. Decisions

- **2026-10-16: Transfer default stays FLIP (ratio 0.97); APIC stays opt-in.**
  - The only cost measurement (3.0, one dam_break run) put APIC at about 1.5×
    per frame plus 72 bytes per particle. It also moved narrow-band behaviour,
    and no visual comparison has shown a gain worth that.
  - `archive/lane-m-defaults-20261014` is kept for reference, not for landing.
  - Reopen only with a paired cost measurement *and* a side-by-side look
    comparison on the four demos.
  - Instead, 4.1 makes the switch visible: Transfer sits with FLIP Ratio in
    the Simulate stage, and the hover text and user guide explain both.

## 7. Notes

- Three items from session memory have no record in origin docs:
  - the R lane being 11% over budget;
  - 11 red workflow-reading tests;
  - a pending gate rebind.

  #5 and #14 are the closest matches.
- The local master checkout is at `296fa95`, behind origin `454839e`.
  Fast-forward it before starting Wave A.
