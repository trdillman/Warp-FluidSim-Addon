# Inactive Simulation Panel Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent every advanced Simulation Physics panel from being polled or drawn without an authoritative domain settings owner, while preserving object-owned settings and the existing Basic/Advanced visibility contract.

**Architecture:** Keep `active_domain_settings(scene)` as the only settings authority and encode the domain-plus-Advanced precondition once in an internal Simulation Physics child-panel base. Apply that base to all twelve `sim_*` children while retaining the root panel's equivalent explicit poll and required public base; do not restore the removed `scene.fluidsim` fallback or add draw-time `None` guards that would hide a bad visibility contract. Strengthen both the fast panel-behavior test and the real Blender GUI inventory so the complete shared fan is covered.

**Tech Stack:** Python 3, pytest, Blender 5.1 Python/RNA UI, PowerShell, Git worktrees.

**Spec:** `AGENTS.md` architecture ownership and verified GUI workflow; `tests/blender_gui_acceptance.py` is the executable GUI contract.

## Global Constraints

- Preserve `obj.fluidsim_domain` as the authoritative owner; never fall back to `scene.fluidsim` for solver settings.
- Treat Pressure and Viscosity as the reproducers for a shared twelve-panel visibility defect.
- Do not modify, stage, rebuild, or restore the six existing dirty `crown_splash` and `dam_break` demo artifacts.
- Use test-first red-green development and fix the shared poll seam, not individual draw methods.
- Run the expensive GUI-inclusive production gate only from a fresh detached worktree at the exact integrated commit.

---

### Task 1: Encode the complete Simulation Physics visibility contract

**Files:**
- Modify: `tests/test_operator_panel_behavior.py:2715`
- Modify: `fluidSim_warp_blender51/blender/ui_scene.py:153-163`
- Modify: `fluidSim_warp_blender51/blender/ui_scene.py:443-663`

**Interfaces:**
- Consumes: `_domain_visible(context) -> bool`, `_advanced_visible(context) -> bool`, `FluidSimScenePanel.poll(context) -> bool`.
- Produces: `_FluidSimSimulationPanel.poll(context) -> bool`, inherited by every direct `FLUIDSIM_PT_sim_*` child; the root keeps the same explicit predicate on its required public base.

- [x] **Step 1: Add the failing behavior test**

  Extend the existing Advanced-mode test with an explicit direct-poll contract for the complete child fan. The test must derive the fan from `SCENE_PANEL_CLASSES`, so adding a future child without the shared base is caught:

```python
def test_advanced_sim_panels_require_an_authoritative_domain(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")

    inactive = SimpleNamespace(
        scene=SimpleNamespace(
            fluidsim=SimpleNamespace(ui_mode="ADVANCED", domain_object=None)
        )
    )
    domain_settings = _PermissiveSettings({})
    active = SimpleNamespace(
        scene=SimpleNamespace(
            fluidsim=SimpleNamespace(
                ui_mode="ADVANCED",
                domain_object=SimpleNamespace(fluidsim_domain=domain_settings),
            )
        )
    )
    basic = SimpleNamespace(
        scene=SimpleNamespace(
            fluidsim=SimpleNamespace(
                ui_mode="BASIC",
                domain_object=active.scene.fluidsim.domain_object,
            )
        )
    )
    children = tuple(
        cls for cls in ui_scene.SCENE_PANEL_CLASSES
        if getattr(cls, "bl_parent_id", "") == "FLUIDSIM_PT_scene_simulation"
    )

    assert len(children) == 12
    assert ui_scene.FLUIDSIM_PT_scene_simulation.poll(inactive) is False
    assert all(cls.poll(inactive) is False for cls in children)
    assert ui_scene.FLUIDSIM_PT_scene_simulation.poll(basic) is False
    assert all(cls.poll(basic) is False for cls in children)
    assert ui_scene.FLUIDSIM_PT_scene_simulation.poll(active) is True
    assert all(cls.poll(active) is True for cls in children)
    for cls in children:
        panel = cls()
        panel.layout = _Layout()
        panel.draw(active)
        assert panel.layout.prop_calls
        assert all(call["data"] is domain_settings for call in panel.layout.prop_calls)
```

- [x] **Step 2: Run the focused test and verify RED**

  Run:

```powershell
python -m pytest -q tests/test_operator_panel_behavior.py::test_advanced_sim_panels_require_an_authoritative_domain
```

  Expected: FAIL because the twelve children inherit `FluidSimScenePanel.poll`, which returns true for an inactive Advanced scene even though `_domain_owner(context)` is `None`.

- [x] **Step 3: Add one shared internal panel base**

  In `ui_scene.py`, after `_advanced_visible`, add:

```python
class _FluidSimSimulationPanel(FluidSimScenePanel):
    """Shared visibility contract for the Advanced domain-physics fan."""

    @classmethod
    def poll(cls, context):
        return _domain_visible(context) and _advanced_visible(context)
```

  Change all twelve `FLUIDSIM_PT_sim_*` classes to inherit `_FluidSimSimulationPanel` instead of `FluidSimScenePanel`. Keep the root `FLUIDSIM_PT_scene_simulation` class directly on `FluidSimScenePanel` with its equivalent explicit poll because the public Scene-panel topology contract requires that direct root base. Do not change any draw method and do not alter `_domain_owner`.

- [x] **Step 4: Run the focused test and verify GREEN**

  Run the same single-test command. Expected: PASS with all thirteen panels false without a domain and true with an object-owned domain in Advanced mode.

- [x] **Step 5: Run the related fast panel/ownership suite**

  Run:

```powershell
python -m pytest -q tests/test_pending_settings_registry.py tests/test_phase4_pending_interaction.py tests/test_operator_panel_behavior.py tests/test_scene_bridge_settings.py
```

  Expected: all tests PASS; the existing assertion that `active_domain_settings` returns `None` without a domain remains unchanged.

### Task 2: Cover the shared fan in real Blender

**Files:**
- Modify: `tests/blender_gui_acceptance.py:44-60`

**Interfaces:**
- Consumes: registered Blender panel identifiers and each class's real `poll`/`draw` behavior.
- Produces: GUI smoke coverage for all twelve Simulation Physics children in both inactive and initialized states.

- [x] **Step 1: Expand the GUI panel inventory**

  Add the ten missing identifiers so `PANEL_IDS` contains this complete fan:

```python
"FLUIDSIM_PT_sim_time",
"FLUIDSIM_PT_sim_gravity",
"FLUIDSIM_PT_sim_pressure",
"FLUIDSIM_PT_sim_viscosity",
"FLUIDSIM_PT_sim_surface_tension",
"FLUIDSIM_PT_sim_color_mixing",
"FLUIDSIM_PT_sim_relax",
"FLUIDSIM_PT_sim_limits",
"FLUIDSIM_PT_sim_forces",
"FLUIDSIM_PT_sim_dissipation",
"FLUIDSIM_PT_sim_attraction",
"FLUIDSIM_PT_sim_spacetime",
```

  The inactive smoke must skip them through `poll`; after setup, set `scene.fluidsim.ui_mode = "ADVANCED"` so they must be callable and draw against the actual domain owner.

- [x] **Step 2: Run the isolated interactive GUI acceptance**

  Run with UTF-8 output from the repository root:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
blender --factory-startup --python tests/blender_gui_acceptance.py
```

  Expected: `GUI-ACCEPTANCE-PASS`; the inactive panel smoke has no `UILayout.prop(... None ...)` errors and the later initialized smoke draws the complete registered fan.

- [x] **Step 3: Inspect the exact source diff**

  Run:

```powershell
git diff --check
git diff -- fluidSim_warp_blender51/blender/ui_scene.py tests/test_operator_panel_behavior.py tests/blender_gui_acceptance.py docs/superpowers/plans/2026-09-12-inactive-simulation-panel-guard.md
```

  Confirm the diff contains no scene fallback, per-panel draw guard, demo edit, unrelated refactor, or broad formatting change.

### Task 3: Review, commit, and certify from a fresh worktree

**Files:**
- Review only: the four files listed in Task 2 Step 3.
- Create verification artifacts only inside a new detached gate worktree's ignored `artifacts/` directory.

**Interfaces:**
- Consumes: the integrated source/test diff and exact commit ID.
- Produces: an independent review verdict, one conventional commit, full GUI-inclusive gate evidence, and a policy-drift validation result.

- [x] **Step 1: Obtain independent read-only review**

  Ask a `deep_worker` to falsify both artifact correctness and intent coverage. It must specifically check that all twelve children share the owner-aware poll, no scene fallback exists, and the tests exercise inactive plus active sibling paths. The reviewer must not edit or rerun the heavy gate.

- [ ] **Step 2: Commit only owned files**

  Stage the four exact paths and commit:

```powershell
git add -- fluidSim_warp_blender51/blender/ui_scene.py tests/test_operator_panel_behavior.py tests/blender_gui_acceptance.py docs/superpowers/plans/2026-09-12-inactive-simulation-panel-guard.md
git commit -m "fix(ui): guard inactive simulation panels"
```

  Verify `git status --short` still lists only the six pre-existing dirty crown/dam demo artifacts.

- [ ] **Step 3: Create a fresh detached verification worktree**

  Resolve the new commit hash, choose a new explicit path under `C:\Users\Tyler\.codex\worktrees\912e\`, and run:

```powershell
git worktree add --detach <new-gate-worktree-path> <new-commit-hash>
```

  Do not reuse or clean `liquid_warp_gate_8cbca14`; its failed-gate artifacts are historical evidence.

- [ ] **Step 4: Run the full GUI-inclusive production gate**

  In the fresh gate worktree:

```powershell
$env:FLUIDSIM_USER_PATH_PROFILE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
python tools/run_production_gate.py --include-gui
```

  Expected: exit code 0 with `PRODUCTION-GATE-PASS`, including `GUI-ACCEPTANCE-PASS` and the saved-demo playback/report lane.

- [ ] **Step 5: Validate refreshed policy-drift reports only after the gate passes**

  In the same exact verification worktree, run:

```powershell
python tools/validate_policy_drift_reports.py
```

  Expected: exit code 0. If the production gate fails, preserve the exact logs and do not run this validator.

- [ ] **Step 6: Final reconciliation**

  Confirm the primary worktree remains on `codex/simplification-2026-09-12`, HEAD is the new commit, and the six protected demo modifications are still unstaged and unchanged. Report the commit, focused test counts, GUI marker, full-gate verdict, policy-drift verdict, and any residual risk without claiming beyond the evidence.
