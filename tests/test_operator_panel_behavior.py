"""Focused UI/operator regression tests for Wave 2 remediation."""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_bpy_stub(monkeypatch, *, device="cuda:7"):
    fake_bpy = types.ModuleType("bpy")

    class _Operator:
        def report(self, *_args, **_kwargs):
            return None

    class _Panel:
        pass

    def _prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(Operator=_Operator, Panel=_Panel)
    fake_bpy.props = SimpleNamespace(
        BoolProperty=lambda **kwargs: _prop(**kwargs),
        EnumProperty=lambda **kwargs: _prop(**kwargs),
        IntProperty=lambda **kwargs: _prop(**kwargs),
        StringProperty=lambda **kwargs: _prop(**kwargs),
    )
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(
            addons={
                "fluidSim_warp_blender51": SimpleNamespace(
                    preferences=SimpleNamespace(device=device)
                )
            }
        )
    )
    fake_bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None,
        unregister_class=lambda _cls: None,
    )
    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_bpy.props)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    return fake_bpy


def _reload_module(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_validation_operator_uses_configured_device(monkeypatch):
    _install_bpy_stub(monkeypatch, device="cuda:3")
    operators = _reload_module("fluidSim_warp_blender51.operators")

    seen = {}

    class _Result:
        name = "device route"
        passed = True

        def as_dict(self):
            return {"name": self.name, "passed": self.passed}

    fake_validation = types.ModuleType("fluidSim_warp_blender51.sim.validation")
    def _run_all(device):
        seen["device"] = device
        return [_Result()]

    fake_validation.run_all = _run_all
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.sim.validation", fake_validation)
    monkeypatch.setattr(
        operators,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: SimpleNamespace(validation_results=[])),
    )
    context = SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(status="")))

    result = operators.FLUIDSIM_OT_run_validation().execute(context)

    assert result == {"FINISHED"}
    assert seen["device"] == "cuda:3"


def test_export_frame_invoke_opens_file_selector(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    calls = []
    context = SimpleNamespace(
        window_manager=SimpleNamespace(fileselect_add=lambda op: calls.append(op))
    )
    op = operators.FLUIDSIM_OT_export_frame()

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert calls == [op]


def test_undo_metadata_is_limited_to_mutating_operators(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")

    undoable = {
        operators.FLUIDSIM_OT_setup_scene,
        operators.FLUIDSIM_OT_tag_object,
        operators.FLUIDSIM_OT_snippet_add,
        operators.FLUIDSIM_OT_snippet_remove,
        operators.FLUIDSIM_OT_snippet_argument_add,
        operators.FLUIDSIM_OT_snippet_argument_remove,
        operators.FLUIDSIM_OT_clear_error,
    }
    not_undoable = {
        operators.FLUIDSIM_OT_initialize,
        operators.FLUIDSIM_OT_release,
        operators.FLUIDSIM_OT_run_validation,
        operators.FLUIDSIM_OT_run_benchmark,
        operators.FLUIDSIM_OT_write_report,
        operators.FLUIDSIM_OT_export_frame,
        operators.FLUIDSIM_OT_export_cache_sequence,
    }

    for cls in undoable:
        assert "UNDO" in getattr(cls, "bl_options", set()), cls.__name__
    for cls in not_undoable:
        assert "UNDO" not in getattr(cls, "bl_options", set()), cls.__name__


class _Layout:
    def __init__(self):
        self.labels = []
        self.operators = []
        self.props = []
        self.enabled = True

    def column(self, align=False):
        return self

    def row(self, align=False):
        return self

    def box(self):
        return self

    def separator(self):
        return None

    def prop(self, *args, **_kwargs):
        if len(args) >= 2:
            self.props.append(args[1])
        return None

    def label(self, text="", icon="NONE"):
        self.labels.append({"text": text, "icon": icon})

    def operator(self, operator_id, **kwargs):
        self.operators.append({"id": operator_id, **kwargs})
        return SimpleNamespace()


def test_simulation_panel_shows_persistent_error_with_clear_action(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            status="Solver error: boom",
            live=False,
            stat_particles=0,
            stat_voxels=0,
            stat_step_ms=0.0,
            stat_substeps=0,
            time_scale=1.0,
            cfl=3.0,
            min_substeps=1,
            max_substeps=4,
            use_scene_gravity=True,
            gravity=(0.0, 0.0, -9.81),
        )
    )
    scene_bridge.set_error(scene, "frame failed")

    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout
    panel.draw(SimpleNamespace(scene=scene))

    assert {"text": "frame failed", "icon": "ERROR"} in layout.labels
    assert any(op["id"] == "fluidsim.clear_error" for op in layout.operators)


def test_clear_error_operator_resets_session_and_scene_error(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")

    class _Scene(dict):
        pass

    scene = _Scene()
    scene.fluidsim = SimpleNamespace(status="Solver error: boom")
    scene_bridge.session().solver = object()
    scene_bridge.set_error(scene, "boom")
    assert scene_bridge.current_error(scene) == "boom"

    result = operators.FLUIDSIM_OT_clear_error().execute(SimpleNamespace(scene=scene))

    assert result == {"FINISHED"}
    assert scene_bridge.current_error(scene) == ""
    assert scene.fluidsim.status == "Solver ready"


def test_output_panel_exposes_cache_sequence_and_playback_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        liquid_view_mode="SURFACE",
        show_surface=True,
        surface_attribute_mode="NONE",
        show_particles=False,
        particle_display_fraction=1.0,
        particle_attribute_mode="NONE",
        show_whitewater=False,
        whitewater_attribute_mode="NONE",
        surface_material=None,
        whitewater_material=None,
        output_collection_name="FluidSim Output",
        show_overlay=True,
        cache_enabled=False,
        cache_playback_enabled=True,
        cache_path="//fluidsim_cache/",
        cache_use_custom_range=True,
        cache_frame_start=3,
        cache_frame_end=5,
    )
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_output()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert "cache_path" in layout.props
    assert "cache_playback_enabled" in layout.props
    assert "cache_use_custom_range" in layout.props
    assert "cache_frame_start" in layout.props
    assert "cache_frame_end" in layout.props
    assert any(op["id"] == "fluidsim.export_cache_sequence" for op in layout.operators)
