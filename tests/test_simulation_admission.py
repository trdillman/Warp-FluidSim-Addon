"""Characterization and ownership contracts for Simulation Admission."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import simulation_admission as admission


class _Runtime:
    def __init__(self, name):
        self.name = name
        self.released = False
        self.device = "cuda:0"
        self.settings = SimpleNamespace(device="cuda:0")

    def release(self):
        self.released = True


class _Session:
    def __init__(self):
        self.solver = _Runtime("old")
        self.error = ""
        self.release_calls = []
        self.capacity_plan = None
        self.capacity_plan_memory_bypassed = False
        self.scene_inputs = SimpleNamespace(clear=lambda: None)
        self.last_reported_growth_capacity = 0
        self.profiler = None
        self.device = "cuda:0"
        self.memory_assessment = {}
        self.scene_identity = 101
        self.scene_key = "old-key"
        self.start_frame = 1
        self.cache_written_start = None
        self.cache_written_end = None
        self.cache_settings_records = {}
        self.last_frame = -1
        self.frame_stats = []

    def owns_scene(self, _scene):
        return self.solver is not None

    def release(self, _scene=None, *, preserve_error=False):
        self.release_calls.append(preserve_error)
        if self.solver is not None:
            self.solver.release()
        self.solver = None
        if not preserve_error:
            self.error = ""

    def set_error(self, _scene, message):
        self.error = str(message)

    def clear_error(self, _scene=None):
        self.error = ""


class _Snapshot:
    configured_device = "cuda:0"
    profiling_enabled = False
    profiling_sync = False
    capacity_plan = SimpleNamespace(blockers=())
    settings = SimpleNamespace(device="cuda:0")
    scene_identity = 101
    scene_key = "key"
    frame_start = 1

    def __init__(self):
        self.released = 0
        self.scene_inputs = SimpleNamespace(clear=lambda: None)

    def transfer_scene_inputs(self):
        value, self.scene_inputs = self.scene_inputs, None
        return value

    def release_transients(self):
        self.released += 1


def _scene():
    return SimpleNamespace(
        fluidsim=SimpleNamespace(status="", snippets=[]),
    )


def _owner(session, applied=None):
    return admission.SimulationAdmission(
        session,
        scene_identity=lambda _scene: 101,
        snippet_builder=lambda _scene: (),
        applied_state_capture=applied or (lambda *_args, **_kwargs: None),
        reset_settings_apply=lambda *_args: None,
    )


def test_preflight_failure_preserves_existing_solver(monkeypatch):
    session = _Session()
    old_solver = session.solver
    owner = _owner(session)
    monkeypatch.setattr(
        owner, "_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("capacity rejected")))

    with pytest.raises(RuntimeError, match="capacity rejected"):
        owner.initialize(_scene())

    assert session.solver is old_solver
    assert old_solver.released is False
    assert session.release_calls == []
    assert "capacity rejected" in session.error


@pytest.mark.parametrize("failure_phase", ("allocate", "configure", "seed", "output"))
def test_candidate_failures_preserve_live_session_and_post_cutover_failure_cleans(
        monkeypatch, failure_phase):
    session = _Session()
    old_solver = session.solver
    owner = _owner(session)
    snapshot = _Snapshot()
    monkeypatch.setattr(
        owner, "_preflight",
        lambda *_args, **_kwargs: (
            snapshot, {"safe": True, "estimated_bytes": 1}, None))

    def fail(name):
        raise RuntimeError(f"{name} failed")

    candidate = _Runtime("replacement")

    def prepare(*_args):
        if failure_phase == "allocate":
            fail("allocate")
        if failure_phase == "configure":
            fail("configure")
        if failure_phase == "seed":
            fail("seed")
        return candidate, ()

    monkeypatch.setattr(owner, "_prepare_solver_replacement", prepare)
    monkeypatch.setattr(
        owner, "_prepare_outputs",
        ((lambda *_args: fail("output")) if failure_phase == "output"
         else (lambda *_args: None)))

    with pytest.raises(RuntimeError, match=f"{failure_phase} failed"):
        owner.initialize(_scene())

    if failure_phase == "output":
        assert old_solver.released is True
        assert candidate.released is True
        assert session.solver is None
        assert session.release_calls == [True]
    else:
        assert old_solver.released is False
        assert session.solver is old_solver
        assert session.release_calls == []
    assert f"{failure_phase} failed" in session.error
    assert snapshot.released == 1


def test_successful_admission_uses_one_snapshot_and_orders_cutover(monkeypatch):
    session = _Session()
    old_solver = session.solver
    events = []
    owner = _owner(session, applied=lambda *_args, **_kwargs: events.append("applied"))
    snapshot = _Snapshot()
    calls = []

    def preflight(*_args, **_kwargs):
        calls.append("snapshot")
        events.append("preflight")
        return snapshot, {"safe": True, "estimated_bytes": 1}, None

    monkeypatch.setattr(owner, "_preflight", preflight)
    candidate = _Runtime("replacement")

    def prepare(*_args):
        events.extend(("allocate", "configure", "seeds"))
        return candidate, ()

    original_commit = owner._commit_solver_replacement

    def commit(*args, **kwargs):
        events.append("cutover")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(owner, "_prepare_solver_replacement", prepare)
    monkeypatch.setattr(owner, "_commit_solver_replacement", commit)
    monkeypatch.setattr(
        owner, "_prepare_outputs",
        lambda *_args: events.append("outputs"))

    result = owner.initialize(_scene())

    assert result is session
    assert calls == ["snapshot"]
    assert events == [
        "preflight", "allocate", "configure", "seeds", "cutover",
        "outputs", "applied",
    ]
    assert old_solver.released is True
    assert session.solver.name == "replacement"
    assert snapshot.released == 1


def test_source_compiler_releases_when_compile_raises(monkeypatch):
    from fluidSim_warp_blender51.sim import source_regions

    released = []

    class Compiler:
        def compile(self, *_args, **_kwargs):
            raise RuntimeError("compile fault")

        def release(self):
            released.append(True)

    monkeypatch.setattr(source_regions, "SourceCompiler", Compiler)
    scene = SimpleNamespace(
        frame_current=3,
        frame_start=1,
        objects=[],
        fluidsim=SimpleNamespace(
            domain_object=None, profiling_enabled=False, profiling_sync=False,
            height_enabled=False,
        ),
    )
    settings = SimpleNamespace(
        device="cpu", domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0), voxel_size=0.1,
        leaf_capacity=8, max_particles=8,
    )
    depsgraph = SimpleNamespace(updates=[])
    _install_bpy_depsgraph_stub(monkeypatch, depsgraph)

    with pytest.raises(RuntimeError, match="compile fault"):
        admission.capture_snapshot(
            scene, settings,
            {"gpu_total_bytes": 1024, "gpu_free_bytes": 1024},
            depsgraph=depsgraph, scene_identity=1,
            snippet_builder=lambda _scene: ())

    assert released == [True]


def test_snapshot_captures_one_owned_scene_evaluation(monkeypatch):
    from fluidSim_warp_blender51.sim import source_regions

    compile_calls = []
    releases = []

    class Compiler:
        def compile(self, plans, context):
            compile_calls.append((tuple(plans), context.frame, context.device))
            return SimpleNamespace(footprints=(), diagnostics=(), sources=())

        def release(self):
            releases.append(True)

    monkeypatch.setattr(source_regions, "SourceCompiler", Compiler)
    plan = SimpleNamespace(blockers=())
    monkeypatch.setattr(
        admission, "_plan_capacity", lambda *_args, **_kwargs: plan)
    depsgraph = SimpleNamespace(updates=[])
    _install_bpy_depsgraph_stub(monkeypatch, depsgraph)
    scene = SimpleNamespace(
        frame_current=7, frame_start=2, objects=[],
        fluidsim=SimpleNamespace(
            domain_object=None, profiling_enabled=True, profiling_sync=False,
            height_enabled=False),
    )
    settings = SimpleNamespace(
        device="cpu", domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0), voxel_size=0.1,
        leaf_capacity=8, max_particles=8)

    snapshot = admission.capture_snapshot(
        scene, settings, {}, depsgraph=depsgraph,
        scene_identity=123, scene_key="owned-key",
        snippet_builder=lambda _scene: ("owned-snippet",))
    try:
        assert snapshot.scene_identity == 123
        assert snapshot.scene_key == "owned-key"
        assert snapshot.frame == 7 and snapshot.frame_start == 2
        assert snapshot.depsgraph_generation[0] == id(depsgraph)
        assert snapshot.configured_device == "cpu"
        assert snapshot.domain_transform is None
        assert snapshot.domain_min == (0.0, 0.0, 0.0)
        assert snapshot.snippets == ("owned-snippet",)
        assert compile_calls == [((), 7, "cpu")]
        assert releases == [True]
    finally:
        snapshot.release_transients()


def test_capacity_replan_reuses_original_owned_demand(monkeypatch):
    inputs = SimpleNamespace(clear=lambda: None)
    original_footprints = (object(),)
    snapshot = admission.AdmissionSnapshot(
        scene_identity=1, scene_key="key", frame=1, frame_start=1,
        depsgraph_generation=(2, ()), configured_device="cpu",
        domain_transform=None, domain_min=(0, 0, 0), domain_max=(1, 1, 1),
        settings=SimpleNamespace(device="cpu"),
        profiling_enabled=False, profiling_sync=False,
        scene_inputs=inputs, descriptors=(), extra_plans=(), colliders=(),
        force_fields=(), height_source=None, seeds=(), snippets=(),
        footprints=original_footprints, collider_footprints=(), resources={},
        capacity_plan=object(),
    )
    candidate = SimpleNamespace(device="cpu")
    seen = []
    monkeypatch.setattr(
        admission, "_plan_capacity",
        lambda settings, footprints, seeds, colliders, resources, **_kwargs: (
            seen.append((settings, footprints, seeds, colliders, resources))
            or "candidate-plan"))

    assert admission.replan_snapshot(snapshot, candidate) is snapshot
    assert snapshot.capacity_plan == "candidate-plan"
    assert snapshot.scene_inputs is inputs
    assert seen == [(candidate, original_footprints, (), (), {})]


def _install_bpy_depsgraph_stub(monkeypatch, depsgraph):
    fake_bpy = types.ModuleType("bpy")
    calls = []

    def evaluated_depsgraph_get():
        calls.append(depsgraph)
        return depsgraph

    fake_bpy.context = SimpleNamespace(
        evaluated_depsgraph_get=evaluated_depsgraph_get)
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    return calls


def test_capture_acquires_one_depsgraph_and_shares_it_with_every_gatherer(
        monkeypatch):
    depsgraph = SimpleNamespace(updates=[])
    acquisitions = _install_bpy_depsgraph_stub(monkeypatch, depsgraph)
    seen = []
    cleared = []

    class Inputs:
        _session_generation = 4

        def __init__(self):
            self.device = "cuda:0"

        def begin_sync(self):
            return 11

        def gather_sources(self, _scene, actual, **_kwargs):
            seen.append(("sources", actual))
            return ()

        def gather_source_plans(self, _scene, actual):
            seen.append(("plans", actual))
            return ()

        def gather_colliders(self, _scene, actual, **_kwargs):
            seen.append(("colliders", actual))
            return ()

        def gather_force_fields(self, _scene, actual):
            seen.append(("forces", actual))
            return ()

        def gather_height_source(self, _scene, actual):
            seen.append(("height", actual))
            return None

        def materialize_device_meshes(
                self, _descriptors, _colliders, device):
            seen.append(("staging", device))

        def clear(self):
            cleared.append(True)

    compile_contexts = []

    class Compiler:
        def compile(self, _plans, context):
            compile_contexts.append(context)
            return SimpleNamespace(footprints=(), diagnostics=(), sources=())

        def release(self):
            seen.append(("compiler-release", None))

    monkeypatch.setattr(admission, "SceneInputs", Inputs)
    from fluidSim_warp_blender51.blender import sources
    from fluidSim_warp_blender51.sim import source_regions
    monkeypatch.setattr(sources, "prepare_auto_seeds", lambda _scene, actual: (
        seen.append(("seeds", actual)) or ()))
    monkeypatch.setattr(source_regions, "SourceCompiler", Compiler)
    monkeypatch.setattr(
        admission, "_plan_capacity",
        lambda *_args, **_kwargs: SimpleNamespace(blockers=()))
    scene = SimpleNamespace(
        frame_current=2, frame_start=1, objects=[],
        fluidsim=SimpleNamespace(
            domain_object=None, profiling_enabled=False,
            profiling_sync=False))
    settings = SimpleNamespace(
        device="cuda:7", domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0), voxel_size=0.1,
        leaf_capacity=8, max_particles=8)

    snapshot = admission.capture_snapshot(scene, settings, {})
    try:
        assert acquisitions == [depsgraph]
        assert [actual for label, actual in seen
                if label in {
                    "sources", "plans", "colliders", "forces", "height",
                    "seeds"}] == [depsgraph] * 6
        assert ("staging", "cpu") in seen
        assert len(compile_contexts) == 1
        assert compile_contexts[0].device == "cpu"
        assert compile_contexts[0].upload is False
        assert snapshot.configured_device == "cuda:7"
        assert snapshot.depsgraph is depsgraph
    finally:
        snapshot.release_transients()
    assert cleared == [True]


def test_capture_rejects_depsgraph_generation_invalidation(monkeypatch):
    depsgraph = SimpleNamespace(updates=[])
    _install_bpy_depsgraph_stub(monkeypatch, depsgraph)
    cleared = []
    released = []

    class Inputs:
        _session_generation = 1
        device = "cpu"

        def begin_sync(self):
            return 1

        def gather_sources(self, _scene, _depsgraph, **_kwargs):
            return ()

        def gather_source_plans(self, _scene, _depsgraph):
            depsgraph.updates.append(
                SimpleNamespace(id=SimpleNamespace(original=object())))
            return ()

        def gather_colliders(self, _scene, _depsgraph, **_kwargs):
            return ()

        def gather_force_fields(self, _scene, _depsgraph):
            return ()

        def gather_height_source(self, _scene, _depsgraph):
            return None

        def materialize_device_meshes(self, *_args):
            return 0

        def clear(self):
            cleared.append(True)

    class Compiler:
        def compile(self, _plans, _context):
            return SimpleNamespace(footprints=(), diagnostics=(), sources=())

        def release(self):
            released.append(True)

    monkeypatch.setattr(admission, "SceneInputs", Inputs)
    from fluidSim_warp_blender51.blender import sources
    from fluidSim_warp_blender51.sim import source_regions
    monkeypatch.setattr(sources, "prepare_auto_seeds", lambda *_args: ())
    monkeypatch.setattr(source_regions, "SourceCompiler", Compiler)
    monkeypatch.setattr(
        admission, "_plan_capacity",
        lambda *_args, **_kwargs: SimpleNamespace(blockers=()))
    scene = SimpleNamespace(
        frame_current=1, frame_start=1, objects=[],
        fluidsim=SimpleNamespace(
            domain_object=None, profiling_enabled=False,
            profiling_sync=False))
    settings = SimpleNamespace(
        device="cuda:0", domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0), voxel_size=0.1,
        leaf_capacity=8, max_particles=8)

    with pytest.raises(RuntimeError, match="depsgraph generation invalidated"):
        admission.capture_snapshot(scene, settings, {})

    assert cleared == [True]
    assert released == [True]


def test_scene_input_clear_breaks_cuda_mesh_descriptor_references():
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs

    inputs = SceneInputs()
    mesh = object()
    source = SimpleNamespace(mesh=mesh)
    collider = SimpleNamespace(mesh=mesh)
    inputs._evaluated_mesh_cache = {
        "mesh": {"mesh": mesh, "device": "cuda:0"}}
    inputs._mesh_consumers = [source, collider]

    inputs.clear()

    assert source.mesh is None
    assert collider.mesh is None
    assert inputs.last_clear_device_resource_count == 1
    assert inputs.total_device_resource_releases == 1
    assert inputs._evaluated_mesh_cache == {}
