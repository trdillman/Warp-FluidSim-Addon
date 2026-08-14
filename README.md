# FluidSim Warp for Blender 5.1

This repository implements an interactive, GPU-resident liquid simulation add-on for Blender 5.1 using NVIDIA Warp/CUDA. The production add-on lives in `fluidSim_warp_blender51/`; the similarly named `fluidSim_synth/` package is reference material, not runtime code.

## Repository map

| Path | Purpose |
| --- | --- |
| `fluidSim_warp_blender51/` | Production Blender add-on and simulation runtime. |
| `fluidSim_synth/` | Distilled FluidSim 1.0.226 contracts used as validation input. |
| `tests/` | Python, Blender, CUDA/Warp, and acceptance checks. |
| `tools/` | Production gates, validators, profilers, and demo tooling. |
| `demos/` | Curated source scenes and previews; regenerated caches stay local. |
| `docs/` | Design documents and maintained external-reference evidence. |
| `backlog/tasks/` | Canonical task queue; `backlog/reports/` holds historical scans. |
| `reviews/` | Current review surfaces, archived review records, and still-linked evidence. |
| `artifacts/`, `bench/` | Ignored local reports, caches, profiles, and benchmark output. |

Live code, runtime output, and current tests outrank historical contracts, review records, screenshots, and generated reports.

## Authoritative guidance

- `AGENTS.md` defines architecture ownership, safety constraints, and the authoritative verification lanes.
- `PROMPT.md` defines the synthesized implementation objective.
- `CONTEXT.md` defines the shared domain language and runtime ownership model.
- `CONTRIBUTING.md` defines contribution and artifact policy.

Common repository-root checks are:

```powershell
python -m pytest -q
python tools/run_production_gate.py
python tools/run_production_gate.py --include-gui
```

Use the focused commands in `AGENTS.md` for the subsystem being changed. After a GUI-inclusive pass and report refresh, run `python tools/validate_policy_drift_reports.py`.

## Archived experiments

The following checkouts are historical branched experiments and are **not merge candidates**:

- `codex/native-gpu-interop-main-20260715` in the `..._port_native_interop_main_20260715` checkout;
- `codex/native-gpu-interop` in the `..._port_native_interop` checkout;
- `claude/particle-surface-rendering-apis-f62356` in `.claude/worktrees/particle-surface-rendering-apis-f62356`;
- `worktree-stflip-implementation` in `.claude/worktrees/stflip-implementation`.

Do not merge, rebase, cherry-pick, or extract from them unless the experiment is explicitly reactivated with a new implementation and validation plan.
