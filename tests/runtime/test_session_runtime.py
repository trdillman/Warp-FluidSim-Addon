"""The Solver Session owns its runtime fields; Simulation Admission only asks.

Admission admits a candidate solver and then has to cut the Session over to it.
That cutover, the per-run bookkeeping reset and the scene binding all live on
``Session`` so the fields they touch can only change together. These tests pin
that interface and keep Admission on the public side of it.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest

from test_scene_bridge_settings import _install_scene_bridge_stubs

_ADMISSION = (
    pathlib.Path(__file__).resolve().parents[2]
    / "fluidSim_warp_blender51" / "blender" / "simulation_admission.py")

_PER_RUN_FIELDS = (
    "start_frame", "cache_written_start", "cache_written_end",
    "cache_settings_records", "last_frame", "frame_stats",
)


@pytest.fixture
def scene_bridge(monkeypatch):
    return _install_scene_bridge_stubs(monkeypatch)


class _Solver:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.released = False

    def release(self):
        self.released = True


class _Inputs:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


def _dirty(session):
    session.start_frame = 17
    session.cache_written_start = 3
    session.cache_written_end = 9
    session.cache_settings_records = {"3": "record"}
    session.last_frame = 9
    session.frame_stats = [object()]


def test_begin_run_restores_every_per_run_field_to_a_fresh_sessions_value(scene_bridge):
    session = scene_bridge.Session()
    fresh = scene_bridge.Session()
    _dirty(session)

    session.begin_run()

    for name in _PER_RUN_FIELDS:
        assert getattr(session, name) == getattr(fresh, name), name


def test_begin_run_records_the_start_frame(scene_bridge):
    session = scene_bridge.Session()

    session.begin_run(12)

    assert session.start_frame == 12


def test_begin_run_restarts_the_growth_marker_only_when_asked(scene_bridge):
    session = scene_bridge.Session()
    session.last_reported_growth_capacity = 4096

    session.begin_run(1)
    assert session.last_reported_growth_capacity == 4096

    session.begin_run(1, reported_growth_capacity=8192)
    assert session.last_reported_growth_capacity == 8192

    session.begin_run(1, reported_growth_capacity=0)
    assert session.last_reported_growth_capacity == 0


def test_install_solver_swaps_authority_and_retires_the_old_runtime(
        scene_bridge, monkeypatch):
    from fluidSim_warp_blender51.sim import memory_safety

    freed = []
    monkeypatch.setattr(
        memory_safety, "release_cuda_memory", lambda device=None: freed.append(device))
    session = scene_bridge.Session()
    old_solver, old_inputs = _Solver("cuda:1"), _Inputs()
    session.solver, session.scene_inputs = old_solver, old_inputs
    candidate, new_inputs, profiler = _Solver(), _Inputs(), object()

    session.install_solver(
        candidate, profiler=profiler, device="cuda:0", scene_inputs=new_inputs)

    assert session.solver is candidate
    assert session.profiler is profiler
    assert session.device == "cuda:0"
    assert session.scene_inputs is new_inputs
    assert old_solver.released and old_inputs.cleared
    assert not candidate.released and not new_inputs.cleared
    assert freed == ["cuda:1"]


def test_install_solver_keeps_inputs_the_candidate_already_shares(
        scene_bridge, monkeypatch):
    from fluidSim_warp_blender51.sim import memory_safety

    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda device=None: None)
    session = scene_bridge.Session()
    shared = _Inputs()
    session.scene_inputs = shared

    session.install_solver(
        _Solver(), profiler=None, device="cuda:0", scene_inputs=shared)

    assert not shared.cleared


def test_install_solver_into_an_empty_session_releases_nothing(scene_bridge):
    session = scene_bridge.Session()
    candidate = _Solver()

    session.install_solver(
        candidate, profiler=None, device="cuda:0", scene_inputs=_Inputs())

    assert session.solver is candidate
    assert session.active


def test_bound_scene_is_owned_by_the_session(scene_bridge):
    scene = SimpleNamespace()
    session = scene_bridge.Session()
    session.solver = _Solver()
    assert not session.owns_scene(scene)

    session.bind_scene(scene_bridge.scene_identity(scene), "digest")

    assert session.owns_scene(scene)
    assert session.scene_key == "digest"
    assert not session.owns_scene(SimpleNamespace())


def test_releasing_the_runtime_begins_a_fresh_run_and_unbinds_the_scene(
        scene_bridge, monkeypatch):
    from fluidSim_warp_blender51.sim import memory_safety

    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda device=None: None)
    session = scene_bridge.Session()
    fresh = scene_bridge.Session()
    session.solver = _Solver()
    session.bind_scene(101, "digest")
    session.last_reported_growth_capacity = 4096
    _dirty(session)

    session.force_release()

    for name in _PER_RUN_FIELDS:
        assert getattr(session, name) == getattr(fresh, name), name
    assert session.last_reported_growth_capacity == 0
    assert session.scene_identity is None and session.scene_key == ""
    assert not session.active


def test_admission_never_assigns_or_reaches_into_session_privates():
    """Admission asks the Session to change; it does not write its fields."""
    tree = ast.parse(_ADMISSION.read_text(encoding="utf-8"))

    def on_session(node):
        return (isinstance(node, ast.Attribute) and node.attr == "session"
                and isinstance(node.value, ast.Name) and node.value.id == "self")

    writes, privates = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and on_session(node.value):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                writes.append(f"session.{node.attr} (line {node.lineno})")
            if node.attr.startswith("_"):
                privates.append(f"session.{node.attr} (line {node.lineno})")

    assert writes == [], f"Admission assigns Session fields: {writes}"
    assert privates == [], f"Admission reaches Session privates: {privates}"
