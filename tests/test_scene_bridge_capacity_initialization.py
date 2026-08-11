import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


def _modules(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.context = SimpleNamespace()
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(frame_change_post=[]))
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    output_publication = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    output_publication.refresh_existing_output_settings = lambda *_args, **_kwargs: None
    output_publication.update_outputs = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication", output_publication)
    blender_package = importlib.import_module("fluidSim_warp_blender51.blender")
    monkeypatch.setattr(
        blender_package, "output_publication", output_publication, raising=False)
    for name in (
        "fluidSim_warp_blender51.blender.scene_bridge",
        "fluidSim_warp_blender51.blender.sources",
    ):
        sys.modules.pop(name, None)
    sources = importlib.import_module("fluidSim_warp_blender51.blender.sources")
    bridge = importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")
    return bridge, sources, output_publication


class _Solver:
    def __init__(self, capacity=4, host_count=0):
        self.particles = SimpleNamespace(capacity=capacity, host_count=host_count)
        self.sourcing = SimpleNamespace(last_apply_result={})
        self.injected = []

    def inject_particle_packets_exact(self, packets):
        packets = tuple(packets)
        required = sum(len(packet.positions) for packet in packets)
        available = self.particles.capacity - self.particles.host_count
        admitted = required <= available
        emitted = required if admitted else 0
        if admitted:
            self.injected.extend(
                (np.array(packet.positions, copy=True), packet)
                for packet in packets)
            self.particles.host_count += required
        result = SimpleNamespace(
            admitted=admitted, emitted=emitted, required=required,
            available=available, capacity_limited=not admitted,
            total=self.particles.host_count)
        self.sourcing.last_injection_result = result
        self.sourcing.last_apply_result = {
            "admitted": admitted, "required": required,
            "available": available, "emitted": emitted,
            "capacity_limited": not admitted,
        }
        if not admitted:
            self.sourcing.last_apply_result["deferred"] = True
        return result


def _seed(sources, name, count):
    return sources.PreparedParticleSeed(
        stable_id=name,
        positions=np.zeros((count, 3), np.float32),
        velocity=(0.0, 0.0, 0.0),
        temperature=0.0,
        viscosity=0.0,
        density=1.0,
        life=0.0,
        use_lifespan=False,
    )


def test_prepared_auto_seed_batch_rejects_aggregate_before_first_injection(monkeypatch):
    _bridge, sources, _viewport = _modules(monkeypatch)
    solver = _Solver(capacity=4, host_count=1)
    sess = SimpleNamespace(solver=solver)

    emitted = sources.commit_prepared_seeds(
        sess, (_seed(sources, "a", 2), _seed(sources, "b", 2)))

    assert emitted == 0
    assert solver.injected == []
    result = solver.sourcing.last_apply_result
    assert result["admitted"] is False
    assert result["required"] == 4
    assert result["available"] == 3
    assert result["emitted"] == 0
    assert result["capacity_limited"] is True
    assert result["deferred"] is True


def test_prepared_auto_seed_batch_commits_all_when_exactly_fitting(monkeypatch):
    _bridge, sources, _viewport = _modules(monkeypatch)
    solver = _Solver(capacity=4, host_count=1)
    sess = SimpleNamespace(solver=solver)

    emitted = sources.commit_prepared_seeds(
        sess, (_seed(sources, "a", 1), _seed(sources, "b", 2)))

    assert emitted == 3
    assert [len(item[0]) for item in solver.injected] == [1, 2]
    assert solver.particles.host_count == 4


def test_manual_object_injection_raises_instead_of_reporting_zero_as_success(
        monkeypatch):
    _bridge, sources, _viewport = _modules(monkeypatch)
    solver = _Solver(capacity=4, host_count=3)
    sess = SimpleNamespace(solver=solver)
    monkeypatch.setattr(
        sources, "prepare_object_particles",
        lambda *_args, **_kwargs: _seed(sources, "manual", 2))

    with pytest.raises(RuntimeError, match="particles required 2, available 1"):
        sources.inject_object_particles(object(), object(), sess)

    assert solver.injected == []


def test_empty_object_injection_clears_stale_capacity_refusal(monkeypatch):
    _bridge, sources, _viewport = _modules(monkeypatch)
    solver = _Solver(capacity=4, host_count=1)
    solver.sourcing.last_injection_result = SimpleNamespace(admitted=False)
    sess = SimpleNamespace(solver=solver)
    monkeypatch.setattr(sources, "prepare_object_particles", lambda *_args: None)

    emitted = sources.inject_object_particles(object(), object(), sess)

    assert emitted == 0
    result = solver.sourcing.last_injection_result
    assert result.admitted is True
    assert result.required == 0
    assert result.emitted == 0
    assert result.total == 1


def test_session_step_rejects_capacity_limited_frame_before_publication(monkeypatch):
    scene_bridge, _sources, viewport = _modules(monkeypatch)

    class StepSolver:
        def __init__(self):
            self.sourcing = SimpleNamespace(last_apply_result={
                "admitted": False,
                "required": 9,
                "available": 8,
                "emitted": 0,
                "capacity_limited": True,
            })

        def step_frame(self):
            return SimpleNamespace(as_dict=lambda: {"frame": 1})

    sess = scene_bridge.Session()
    sess.solver = StepSolver()
    sess.scene_identity = "scene"
    monkeypatch.setattr(scene_bridge, "_scene_identity", lambda _scene: "scene")
    updates = []
    monkeypatch.setattr(
        viewport, "update_outputs",
        lambda *_args, **_kwargs: updates.append("viewport"),
    )
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(cache_enabled=False),
        frame_current=1,
    )

    with pytest.raises(RuntimeError, match="required 9.*available 8"):
        sess.step(scene)

    assert updates == []
    assert sess.frame_stats == []


@pytest.mark.parametrize(("pause_requested", "live_after"), (
    (False, True),
    (True, False),
))
def test_deferred_growth_never_releases_solver_or_partially_publishes_source(
        monkeypatch, pause_requested, live_after):
    scene_bridge, _sources, viewport = _modules(monkeypatch)

    stats = SimpleNamespace(
        active_particles=8,
        active_voxels=16,
        whitewater_particles=0,
        step_time=0.01,
        substeps=1,
        cfl_substeps_clamped=False,
        as_dict=lambda: {"frame": 1},
    )

    class StepSolver:
        frame = 1
        last_particle_growth = None

        def __init__(self):
            self.particles = SimpleNamespace(host_count=8, capacity=8)
            self.sourcing = SimpleNamespace(last_apply_result={
                "admitted": False,
                "required": 2,
                "available": 0,
                "emitted": 0,
                "capacity_limited": True,
                "deferred": True,
                "pause_requested": pause_requested,
                "growth_status": "BLOCKED",
                "growth_reason": "guarded",
            })

        def step_frame(self):
            return stats

    solver = StepSolver()
    sess = scene_bridge.Session()
    sess.solver = solver
    sess.scene_identity = "scene"
    monkeypatch.setattr(scene_bridge, "_scene_identity", lambda _scene: "scene")
    updates = []
    monkeypatch.setattr(
        viewport, "update_outputs",
        lambda *_args, **_kwargs: updates.append("viewport"),
    )
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(cache_enabled=False, live=True),
        frame_current=1,
        objects=[],
    )

    assert sess.step(scene) is stats

    assert sess.solver is solver
    assert scene.fluidsim.live is live_after
    assert updates == ["viewport"]
    assert "solver preserved" in scene.fluidsim.status or "solver continued" in scene.fluidsim.status
    assert scene_bridge.current_error(scene) == ""


def test_capacity_error_message_lists_particle_and_leaf_requirements(monkeypatch):
    scene_bridge, _sources, _viewport = _modules(monkeypatch)
    blockers = (
        SimpleNamespace(code="PARTICLE_CAPACITY", required=101, available=100),
        SimpleNamespace(code="LEAF_CAPACITY", required=17, available=16),
    )

    message = scene_bridge.capacity_blocking_message(blockers)

    assert "particles required 101, available 100" in message
    assert "leaves required 17, available 16" in message
    assert "No partial source emission was committed" in message
