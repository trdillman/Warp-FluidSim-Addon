"""Automatic sparse-leaf growth: planning, admission cutover and the step retry."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import simulation_admission as admission
from fluidSim_warp_blender51.sim import memory_safety
from fluidSim_warp_blender51.sim.fields import TopologyCapacityError
from fluidSim_warp_blender51.sim.leaf_capacity import (
    MAX_GROWTHS_PER_STEP,
    checkpoint_leaf_allocation,
    leaf_growth_target,
)
from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings


# --- planning --------------------------------------------------------------

def test_target_is_geometric_rounded_and_contains_the_demand():
    assert leaf_growth_target(8192, 9398, total_cells=13328) == 12288
    assert leaf_growth_target(1000, 5000, total_cells=100_000) == 5120
    assert leaf_growth_target(512, 513, total_cells=100_000) == 768


def test_target_never_exceeds_the_domain_or_the_hard_limit():
    assert leaf_growth_target(8192, 9398, total_cells=9400) == 9400
    assert leaf_growth_target(
        200_000, 250_000, total_cells=10**9, hard_limit=262_144) == 262_144
    # Demand beyond what the domain can hold is reported, not hidden.
    assert leaf_growth_target(1800, 2500, total_cells=1800) == 1800


def test_checkpoint_allocation_is_the_grown_capacity_or_the_live_count():
    assert checkpoint_leaf_allocation(SimpleNamespace(
        scalars={"leaf_capacity": 12288, "leaf_count": 10000})) == 12288
    assert checkpoint_leaf_allocation(SimpleNamespace(
        scalars={"leaf_count": 10000})) == 10000
    assert checkpoint_leaf_allocation(SimpleNamespace(scalars={})) is None
    assert checkpoint_leaf_allocation(SimpleNamespace(
        scalars={"leaf_capacity": True, "leaf_count": 0})) is None
    assert checkpoint_leaf_allocation(SimpleNamespace(frame=1)) is None


def test_preflight_raises_leaf_allocation_to_fit_a_grown_checkpoint(monkeypatch):
    authored = _settings(leaf_capacity=8192)
    seen = {}

    def plan(_self, _scene, settings, _resources, **_kwargs):
        seen["leaf_capacity"] = int(settings.leaf_capacity)
        raise RuntimeError("planned")

    monkeypatch.setattr(admission, "build_settings", lambda _scene: authored)
    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: {})
    monkeypatch.setattr(admission.SimulationAdmission, "_plan_scene", plan)
    scene = SimpleNamespace(objects=(), frame_start=1)

    with pytest.raises(RuntimeError, match="planned"):
        _owner(_Session(None))._preflight(scene, runtime_leaf_capacity=12288)
    assert seen["leaf_capacity"] == 12288

    seen.clear()
    with pytest.raises(RuntimeError, match="planned"):
        _owner(_Session(None))._preflight(scene, runtime_leaf_capacity=64)
    assert seen["leaf_capacity"] == 8192


def test_target_never_shrinks_and_rejects_bad_factors():
    assert leaf_growth_target(4096, 10, total_cells=100_000) == 6144
    assert leaf_growth_target(4096, 10, total_cells=2048) == 4096
    with pytest.raises(ValueError):
        leaf_growth_target(1, 2, total_cells=10, factor=0.5)
    with pytest.raises(ValueError):
        leaf_growth_target(1, 2, total_cells=10, factor=float("nan"))


# --- admission cutover -----------------------------------------------------

class _Runtime:
    def __init__(self, name, settings, *, total_cells=13328, capacity=8192):
        self.name = name
        self.settings = settings
        self.device = "cuda:0"
        self.released = False
        self.events = []
        self.grid = SimpleNamespace(num_leaf_cells=total_cells, capacity=capacity)
        self.particles = SimpleNamespace(capacity=800_000)
        self.features = SimpleNamespace(effects=SimpleNamespace(
            mark_initialized_through=lambda frame: self.events.append(
                ("effects", frame)),
            adopt_cadence_history=lambda other: self.events.append(
                ("history", other))))
        self.state = {"frame": 17}
        self.import_error = None

    def export_checkpoint_state(self):
        self.events.append("export")
        return self.state

    def import_checkpoint_state(self, state):
        self.events.append(("import", state))
        if self.import_error is not None:
            raise self.import_error

    def resume_refused_frame(self):
        self.events.append("resume")

    def release(self):
        self.released = True


class _Session:
    def __init__(self, solver):
        self.solver = solver
        self.profiler = None
        self.scene_inputs = SimpleNamespace(
            export_frozen_state=lambda: "FROZEN")
        self.errors = []
        self.begin_run_calls = 0
        self.release_calls = []

    def owns_scene(self, _scene):
        return self.solver is not None

    def install_solver(self, candidate, *, profiler, device, scene_inputs,
                       admitted_effects=()):
        old, self.solver = self.solver, candidate
        old.release()

    def begin_run(self, *_args, **_kwargs):
        self.begin_run_calls += 1

    def set_error(self, _scene, message):
        self.errors.append(str(message))

    def release(self, _scene=None, *, preserve_error=False):
        self.release_calls.append(preserve_error)


class _Snapshot:
    def __init__(self, settings):
        self.settings = settings
        self.scene_inputs = SimpleNamespace(clear=lambda: None)
        self.released = 0
        self.effects = ()

    def transfer_scene_inputs(self):
        return self.scene_inputs

    def release_transients(self):
        self.released += 1


def _settings(**overrides):
    base = SolverSettings(
        device="cuda:0", leaf_capacity=8192, max_particles=800_000)
    return replace(base, **overrides)


def _owner(session):
    return admission.SimulationAdmission(
        session, scene_identity=lambda _scene: 1,
        applied_state_capture=lambda *_a, **_k: pytest.fail(
            "growth must not rewrite the applied settings snapshot"),
        reset_settings_apply=lambda *_a: None)


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda *_a, **_k: None)
    solver = _Runtime("old", _settings())
    session = _Session(solver)
    owner = _owner(session)
    seen = {}

    def preflight(scene, **kwargs):
        seen["preflight"] = kwargs
        seen["order"] = list(solver.events)
        settings = kwargs["applied_allocation"]
        snapshot = _Snapshot(replace(_settings(), leaf_capacity=settings.leaf_capacity,
                                     max_particles=kwargs["runtime_capacity"]))
        seen["snapshot"] = snapshot
        return snapshot, None

    candidate = _Runtime("new", _settings(leaf_capacity=12288), capacity=12288)
    seen["candidate"] = candidate

    def prepare(scene, settings, snapshot, profiler):
        seen["prepared_settings"] = settings
        return candidate, ()

    monkeypatch.setattr(owner, "_preflight", preflight)
    monkeypatch.setattr(owner, "_prepare_solver_replacement", prepare)
    monkeypatch.setattr(owner, "_validate_effect_snapshot", lambda *_a: None)
    return SimpleNamespace(solver=solver, session=session, owner=owner,
                           seen=seen, candidate=candidate)


def _refusal(required=9398, capacity=8192, substep=0):
    error = TopologyCapacityError(required, capacity)
    error.substep_index = substep
    return error


def test_growth_replaces_the_solver_with_the_same_state(rig):
    result = rig.owner.grow_leaf_capacity(
        SimpleNamespace(), _refusal(), effect_frame=18)

    assert result.status == "GROWN" and result.grew
    assert (result.previous, result.current, result.required) == (8192, 12288, 9398)
    # The state is captured before the replacement is built, restored into the
    # candidate before the swap, and the old solver is retired only afterwards.
    assert rig.solver.events == ["export"]
    assert rig.candidate.events == [
        ("import", rig.solver.state), ("effects", 17),
        ("history", rig.solver.features.effects), "resume"]
    assert rig.session.solver is rig.candidate
    assert rig.solver.released is True
    assert rig.candidate.released is False
    assert rig.seen["snapshot"].released == 1
    # The replacement is allocated at the grown leaf capacity and keeps the
    # runtime particle pool; the run bookkeeping is not restarted.
    assert rig.seen["prepared_settings"].leaf_capacity == 12288
    assert rig.seen["prepared_settings"].max_particles == 800_000
    assert rig.seen["preflight"]["operation"] == "RESET"
    assert rig.session.begin_run_calls == 0


def test_growth_changes_allocation_only(rig):
    # Authored edits that are still pending (a Reset-only setting changed on
    # the scene) must not ride along: the running solver's settings carry over.
    rig.solver.settings = _settings(open_boundaries=True)

    result = rig.owner.grow_leaf_capacity(SimpleNamespace(), _refusal())

    assert result.grew
    prepared = rig.seen["prepared_settings"]
    assert prepared.open_boundaries is True
    assert prepared.leaf_capacity == 12288
    assert rig.seen["snapshot"].settings is prepared


def test_growth_keeps_frozen_scene_inputs(rig):
    rig.owner.grow_leaf_capacity(SimpleNamespace(), _refusal())

    assert rig.seen["preflight"]["frozen_input_state"] == "FROZEN"


@pytest.mark.parametrize("disabled_by", ["setting", "mid_frame", "domain_full"])
def test_growth_is_refused_without_touching_the_live_solver(rig, disabled_by):
    error = _refusal()
    if disabled_by == "setting":
        rig.solver.settings = _settings(auto_grow_particles=False)
        status = "DISABLED"
    elif disabled_by == "mid_frame":
        error.substep_index = 3
        status = "BLOCKED"
    else:
        rig.solver.grid.num_leaf_cells = 9000
        status = "BLOCKED"
    result = rig.owner.grow_leaf_capacity(SimpleNamespace(), error)

    assert result.status == status and not result.grew and result.reason
    assert rig.solver.events == []
    assert rig.session.solver is rig.solver and rig.solver.released is False
    assert "preflight" not in rig.seen


def test_unknown_substep_is_treated_as_unsafe(rig):
    error = TopologyCapacityError(9398, 8192)
    assert error.substep_index is None
    assert rig.owner.grow_leaf_capacity(SimpleNamespace(), error).status == "BLOCKED"


def test_preflight_refusal_keeps_the_running_solver(rig, monkeypatch):
    def refuse(*_a, **_k):
        raise RuntimeError("FluidSim memory safeguard stopped the reset")

    monkeypatch.setattr(rig.owner, "_preflight", refuse)
    result = rig.owner.grow_leaf_capacity(SimpleNamespace(), _refusal())

    assert result.status == "BLOCKED"
    assert "memory safeguard" in result.reason
    assert rig.session.solver is rig.solver and rig.solver.released is False


def test_restore_failure_discards_the_candidate_and_keeps_the_solver(rig):
    rig.candidate.import_error = ValueError("checkpoint optional particle inventory")
    result = rig.owner.grow_leaf_capacity(SimpleNamespace(), _refusal())

    assert result.status == "FAILED"
    assert "inventory" in result.reason
    assert rig.candidate.released is True
    assert rig.session.solver is rig.solver and rig.solver.released is False
    assert rig.seen["snapshot"].released == 1
    assert rig.session.errors == [] and rig.session.release_calls == []


def test_allocation_failure_keeps_the_solver(rig, monkeypatch):
    def fail(*_a):
        raise MemoryError("CUDA out of memory")

    monkeypatch.setattr(rig.owner, "_prepare_solver_replacement", fail)
    result = rig.owner.grow_leaf_capacity(SimpleNamespace(), _refusal())

    assert result.status == "FAILED" and "out of memory" in result.reason
    assert rig.session.solver is rig.solver and rig.solver.released is False


def test_retirement_failure_after_the_swap_clears_the_session(rig, monkeypatch):
    def retire_badly(candidate, snapshot, *, profiler, settings):
        rig.session.solver = candidate
        raise admission.ReplacementRetirementError("old runtime would not release")

    cleared = []
    monkeypatch.setattr(rig.owner, "_commit_solver_replacement", retire_badly)
    monkeypatch.setattr(
        rig.owner, "_release_failed_cutover",
        lambda scene, error: cleared.append(str(error)))

    with pytest.raises(admission.ReplacementRetirementError):
        rig.owner.grow_leaf_capacity(SimpleNamespace(), _refusal())
    assert cleared and rig.candidate.released is False


# --- the step retry --------------------------------------------------------

def _scene_bridge(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(evaluated_depsgraph_get=lambda: object())
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(frame_change_post=[]))
    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    sys.modules.pop("fluidSim_warp_blender51.blender.playback", None)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    return importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")


class _SteppingSolver:
    def __init__(self, name, outcomes):
        self.name = name
        self.outcomes = list(outcomes)
        self.effect_frame = None
        self.calls = 0

    def step_frame(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _GrowingAdmission:
    def __init__(self, session, replacements, results):
        self.session = session
        self.replacements = list(replacements)
        self.results = list(results)
        self.calls = []

    def grow_leaf_capacity(self, scene, error, *, effect_frame=None):
        self.calls.append((error, effect_frame))
        result = self.results.pop(0)
        if result.grew:
            self.session.solver = self.replacements.pop(0)
        return result


def _session(monkeypatch, solver, replacements, results):
    scene_bridge = _scene_bridge(monkeypatch)
    session = scene_bridge.Session()
    session.solver = solver
    admission_owner = _GrowingAdmission(session, replacements, results)
    session._admission = lambda: admission_owner
    return session, admission_owner


def _grown(previous, current):
    from fluidSim_warp_blender51.sim.leaf_capacity import LeafGrowthResult

    return LeafGrowthResult("GROWN", previous, current, current - 1)


def test_refused_frame_is_repeated_on_the_grown_solver(monkeypatch):
    old = _SteppingSolver("old", [_refusal()])
    new = _SteppingSolver("new", ["stats"])
    old.effect_frame = 18
    session, owner = _session(monkeypatch, old, [new], [_grown(8192, 12288)])

    assert session._step_growing_leaves(SimpleNamespace()) == "stats"
    assert (old.calls, new.calls) == (1, 1)
    assert new.effect_frame == 18
    assert owner.calls[0][1] == 18
    assert session.last_leaf_growth.current == 12288
    assert "8,192" in session._leaf_growth_notice and "12,288" in session._leaf_growth_notice


def test_a_refused_growth_re_raises_the_original_error_with_the_reason(monkeypatch):
    from fluidSim_warp_blender51.sim.leaf_capacity import LeafGrowthResult

    error = _refusal()
    old = _SteppingSolver("old", [error])
    blocked = LeafGrowthResult("BLOCKED", 8192, 8192, 9398, "not enough memory")
    session, _ = _session(monkeypatch, old, [], [blocked])

    with pytest.raises(TopologyCapacityError) as caught:
        session._step_growing_leaves(SimpleNamespace())
    assert caught.value is error
    assert caught.value.growth_note == "not enough memory"
    assert session.solver is old


def test_growth_is_bounded_per_step(monkeypatch):
    solvers = [_SteppingSolver(f"s{i}", [_refusal(capacity=1000 * (i + 1))])
               for i in range(MAX_GROWTHS_PER_STEP + 1)]
    session, owner = _session(
        monkeypatch, solvers[0], solvers[1:],
        [_grown(1000 * i, 1000 * (i + 1)) for i in range(1, MAX_GROWTHS_PER_STEP + 1)])

    with pytest.raises(TopologyCapacityError) as caught:
        session._step_growing_leaves(SimpleNamespace())
    assert len(owner.calls) == MAX_GROWTHS_PER_STEP
    assert "did not settle" in caught.value.growth_note


def test_other_errors_are_not_retried(monkeypatch):
    old = _SteppingSolver("old", [RuntimeError("boom")])
    session, owner = _session(monkeypatch, old, [], [])
    with pytest.raises(RuntimeError, match="boom"):
        session._step_growing_leaves(SimpleNamespace())
    assert owner.calls == [] and old.calls == 1
