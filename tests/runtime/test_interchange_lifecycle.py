"""Export lifecycle regressions: real Session and playback, fake Blender IO."""
from contextlib import nullcontext
import importlib
import sys
from types import SimpleNamespace, ModuleType

import pytest

from test_scene_bridge_cache_playback import _install_scene_bridge_stubs


class Obj:
    type = "MESH"
    name = "surface"
    hide_viewport = True
    hidden = True
    selected = False
    modifiers = ()

    def __init__(self):
        self.data = SimpleNamespace(as_pointer=lambda: 42)

    def select_get(self):
        return self.selected

    def select_set(self, value):
        self.selected = value

    def hide_get(self):
        return self.hidden

    def hide_set(self, value):
        self.hidden = value


class Objects(list):
    active = None


@pytest.fixture
def export_env(monkeypatch):
    bridge = _install_scene_bridge_stubs(monkeypatch)
    playback = bridge.playback
    monkeypatch.setattr(playback, "_bridge", lambda: bridge)
    bpy = sys.modules["bpy"]
    props = ModuleType("bpy.props")
    props.IntProperty = props.StringProperty = lambda **kw: None
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    bpy.types = SimpleNamespace(Operator=type("Operator", (), {}))
    obj, other = Obj(), Obj()
    other.selected = True
    objects = Objects([obj, other])
    objects.active = other
    bpy.context.view_layer = SimpleNamespace(objects=objects)
    bpy.ops = SimpleNamespace(wm=SimpleNamespace(
        alembic_export=lambda **kw: {"FINISHED"},
        usd_export=lambda **kw: {"FINISHED"}))
    scene = SimpleNamespace(
        frame_current=7, frame_start=1, frame_end=20, objects=objects,
        fluidsim=SimpleNamespace(live=False, cache_playback_enabled=False,
                                 cache_path="cache", status="before"))
    scene.frame_set = lambda frame: setattr(scene, "frame_current", frame)
    name = "fluidSim_warp_blender51.blender.interchange"
    monkeypatch.delitem(sys.modules, name, raising=False)
    interchange = importlib.import_module(name)
    monkeypatch.setattr(interchange, "playback", playback)
    from fluidSim_warp_blender51.blender import production_bake
    monkeypatch.setattr(production_bake, "bake_is_active", lambda s: False)
    monkeypatch.setattr(interchange, "available_frames", lambda s: (1, 2))
    monkeypatch.setattr(interchange.output_publication, "output_object", lambda *a: obj)
    monkeypatch.setattr(interchange, "_range_point_counts", lambda *a: (0, 0))
    monkeypatch.setattr(interchange.rigid_publication, "carried_objects", lambda *a: ())
    monkeypatch.setattr(interchange._UsdSamplers, "attach", lambda *a: None)
    monkeypatch.setattr(interchange.output_publication, "hold_point_capacity", nullcontext)
    obj.hide_render = False
    return SimpleNamespace(module=interchange, bridge=bridge, playback=playback,
                           bpy=bpy, scene=scene, obj=obj, other=other)


def snapshot(env):
    s = env.scene
    return (s.frame_current, s.frame_start, s.frame_end, s.fluidsim.live,
            s.fluidsim.cache_playback_enabled, s.fluidsim.cache_path,
            tuple((o.select_get(), o.hide_get(), o.hide_viewport) for o in s.objects),
            env.bpy.context.view_layer.objects.active)


def install_solver(env, foreign=False):
    sess = env.bridge.session()
    solver = SimpleNamespace(particles=[(1, 2, 3)], rigid_poses=[(4, 5, 6)])
    sess.solver = solver
    sess.bind_scene(id(object()) if foreign else id(env.scene), "cache-key")
    sess.last_frame = 7
    sess.cache_settings_records = {7: "preserve"}
    return sess, solver


@pytest.mark.parametrize("fmt", ["alembic", "usd"])
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("foreign", [False, True])
def test_active_session_refused_without_mutation(export_env, fmt, live, foreign):
    e = export_env
    e.scene.fluidsim.live = live
    sess, solver = install_solver(e, foreign)
    owner = (sess.scene_identity, sess.scene_key)
    e.playback._CACHE_CURSORS[id(e.scene)] = 7
    cursors = dict(e.playback._CACHE_CURSORS)
    before = snapshot(e)
    plan = (e.module.ExportPlan("cache", (1, 2), e.obj) if fmt == "alembic"
            else e.module.UsdExportPlan("cache", (1, 2), e.obj, ()))
    # Repeated direct calls also cover a plan retained across a file picker.
    for _ in range(2):
        with pytest.raises(e.module.ExportRefused, match="live solver"):
            getattr(e.module, "export_" + fmt)(e.scene, "out", plan)
        assert sess.solver is solver
        assert (sess.scene_identity, sess.scene_key) == owner
        assert e.playback._CACHE_CURSORS == cursors
        assert solver.particles == [(1, 2, 3)]
        assert solver.rigid_poses == [(4, 5, 6)]
        assert sess.last_frame == 7
        assert sess.cache_settings_records == {7: "preserve"}
        assert snapshot(e) == before
    if not foreign:
        assert e.playback.set_live_enabled(e.scene, True)
        assert sess.solver is solver
        assert sess.owns_scene(e.scene)
    planner = e.module.plan_export if fmt == "alembic" else e.module.plan_usd_export
    with pytest.raises(e.module.ExportRefused, match="live solver"):
        planner(e.scene)


@pytest.mark.parametrize("fmt", ["alembic", "usd"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("outcome", ["success", "cancel", "exception"])
def test_inactive_export_restores_scene(export_env, fmt, cached, outcome):
    e = export_env
    e.scene.fluidsim.cache_playback_enabled = cached
    e.scene.fluidsim.live = cached
    before = snapshot(e)
    plan = (e.module.ExportPlan("cache", (1, 2), e.obj) if fmt == "alembic"
            else e.module.UsdExportPlan("cache", (1, 2), e.obj, ()))

    def exporter(**kwargs):
        e.scene.frame_set(2)
        assert e.scene.fluidsim.cache_playback_enabled
        if outcome == "exception":
            raise RuntimeError("injected exporter failure")
        return {"FINISHED"} if outcome == "success" else {"CANCELLED"}

    setattr(e.bpy.ops.wm, fmt + "_export", exporter)
    for _ in range(2):
        if outcome == "success":
            assert getattr(e.module, "export_" + fmt)(e.scene, "out", plan) == []
        else:
            with pytest.raises(RuntimeError):
                getattr(e.module, "export_" + fmt)(e.scene, "out", plan)
        assert snapshot(e) == before
        assert not e.bridge.session().active


@pytest.mark.parametrize("fmt", ["alembic", "usd"])
def test_inactive_live_mode_admits_once_then_refuses_repeat(export_env, monkeypatch, fmt):
    e = export_env
    from fluidSim_warp_blender51.runtime import warp_runtime
    monkeypatch.setattr(warp_runtime, "prewarm_simulation_kernels", lambda: None)
    admitted = []
    sess = e.bridge.session()

    def initialize(scene):
        admitted.append(scene)
        install_solver(e)

    monkeypatch.setattr(sess, "initialize", initialize)
    e.scene.fluidsim.live = True
    plan = (e.module.ExportPlan("cache", (1, 2), e.obj) if fmt == "alembic"
            else e.module.UsdExportPlan("cache", (1, 2), e.obj, ()))
    before = snapshot(e)
    assert getattr(e.module, "export_" + fmt)(e.scene, "out", plan) == []
    assert admitted == [e.scene]
    assert sess.owns_scene(e.scene)
    assert snapshot(e) == before
    solver = sess.solver
    with pytest.raises(e.module.ExportRefused):
        getattr(e.module, "export_" + fmt)(e.scene, "out", plan)
    assert admitted == [e.scene]
    assert sess.solver is solver


@pytest.mark.parametrize("fmt", ["alembic", "usd"])
def test_partial_entry_failure_restores_scene(export_env, monkeypatch, fmt):
    e = export_env
    before = snapshot(e)

    def fail(*args):
        e.scene.frame_set(2)
        raise RuntimeError("injected setup failure")

    if fmt == "alembic":
        plan = e.module.ExportPlan("cache", (1, 2), e.obj, (e.other,))
        monkeypatch.setattr(e.module.rigid_publication, "publish_cache_frames", fail)
    else:
        plan = e.module.UsdExportPlan("cache", (1, 2), e.obj, ())
        monkeypatch.setattr(e.module._UsdSamplers, "attach", fail)
    with pytest.raises(RuntimeError, match="injected setup"):
        getattr(e.module, "export_" + fmt)(e.scene, "out", plan)
    assert snapshot(e) == before
    assert not e.bridge.session().active

@pytest.mark.parametrize("fmt", ["alembic", "usd"])
@pytest.mark.parametrize("entry", ["invoke", "execute"])
def test_operator_reports_refusal_and_cancels(export_env, fmt, entry):
    e = export_env
    install_solver(e)
    cls = (e.module.FLUIDSIM_OT_export_alembic if fmt == "alembic"
           else e.module.FLUIDSIM_OT_export_usd)
    op = cls()
    op.filepath = "output"
    op.frame_start = op.frame_end = 0
    reports = []
    op.report = lambda level, message: reports.append((level, message))
    context = SimpleNamespace(scene=e.scene)
    before = snapshot(e)
    result = op.invoke(context, None) if entry == "invoke" else op.execute(context)
    assert result == {"CANCELLED"}
    assert reports[0][0] == {"ERROR"}
    assert "particle and rigid-body state" in reports[0][1]
    assert "paused" in reports[0][1]
    assert snapshot(e) == before
