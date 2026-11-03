# FluidSim Warp for Blender 5.2

This repository implements an interactive, GPU-resident liquid simulation add-on for Blender 5.2 LTS using NVIDIA Warp/CUDA. Warp 1.19.0 is not bundled: after enabling the add-on, press **Install Warp** in add-on preferences to download the certified build. The production add-on lives in `fluidSim_warp_blender51/`.

## Repository map

| Path | Purpose |
| --- | --- |
| `fluidSim_warp_blender51/` | Production Blender add-on and simulation runtime. |
| `tests/` | Python, Blender, CUDA/Warp, and acceptance checks. |
| `tools/` | Production gates, validators, profilers, and demo tooling. |
| `demos/` | Canonical playable simulations governed by `docs/DEMO_SCENE_CONTRACT.md`; regenerated caches stay local. |
| `fluidSim_warp_blender51/resources/examples/` | Distributed workflow examples and validation/review assets; not canonical demos unless separately promoted and runtime-gated. |
| `docs/` | Design contracts, release procedure, and handoff documents. |
| `artifacts/`, `bench/` | Ignored local reports, caches, profiles, and benchmark output. |

Live code, runtime output, and current tests outrank historical contracts, review records, screenshots, and generated reports.

**Demo** is a runtime claim in this repository. A demo must reopen with an
authoritative Domain, initialize a non-empty solver from owned sources or
seeds, advance through the real frame handlers, and publish changing owned
output. Static geometry, renders, CPU fixtures, and package/UI review scenes
must be described as examples or evidence. See `docs/DEMO_SCENE_CONTRACT.md`.

## Authoritative guidance

- `AGENTS.md` defines architecture ownership, safety constraints, and the authoritative verification lanes.
- `CONTRIBUTING.md` defines contribution and artifact policy.

Common repository-root checks are:

```powershell
python -m pytest -q
python tools/run_production_gate.py
python tools/run_production_gate.py --include-gui
```

Use the focused commands in `AGENTS.md` for the subsystem being changed.
## Artist workflow

For everyday work, choose **Set Up & Preview** for a selected mesh or use the
Scene Properties controls to configure an existing domain, then press
**Preview**. Preview prepares the effects, solver, and outputs and starts the
Blender timeline. Use **Pause**, **Resume Preview**, **Apply Changes**, or the
explicit **Restart Simulation** action as needed; production bake and cached
preview remain separate workflows.

See [`docs/ARTIST_WORKFLOWS.md`](docs/ARTIST_WORKFLOWS.md) for the consequences
of each action, effect trust, safe versus structural changes, cached playback,
and the Advanced diagnostic controls.

## Experiment scope

ST-FLIP (Spatiotemporal FLIP / Variable Projection) is permanently outside
FluidSim's project scope by the owner's 1 November 2026 decision. Its solver,
settings and UI were removed in 3.0. Historical records and legacy benchmark
schema comparisons remain evidence, not implementation or revival candidates.
The ordinary FLIP/APIC solver remains in scope.

Native GPU interop (`codex/native-gpu-interop*`) and particle-surface rendering
APIs (`claude/particle-surface-rendering-apis-f62356`) are reactivated for
recovered-source review and a new implementation/validation plan. Use clean
isolated branches based on fresh `origin/master`; historical experiment results
do not qualify a current candidate. Dependency/runtime qualification is open.
Reviewable PRs require current tests and evidence; reactivation does not
authorize merge or release.
