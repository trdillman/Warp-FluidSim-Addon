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
        CollectionProperty=lambda **kwargs: _prop(**kwargs),
        EnumProperty=lambda **kwargs: _prop(**kwargs),
        FloatProperty=lambda **kwargs: _prop(**kwargs),
        FloatVectorProperty=lambda **kwargs: _prop(**kwargs),
        IntProperty=lambda **kwargs: _prop(**kwargs),
        PointerProperty=lambda **kwargs: _prop(**kwargs),
        StringProperty=lambda **kwargs: _prop(**kwargs),
    )
    fake_bpy.types.PropertyGroup = object
    fake_bpy.types.Text = object
    fake_bpy.types.Object = object
    fake_bpy.types.Material = object
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(
            addons={
                "fluidSim_warp_blender51": SimpleNamespace(
                    preferences=SimpleNamespace(device=device)
                )
            }
        ),
        screen=SimpleNamespace(areas=[]),
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

    def prop(self, _data, property_name, **_kwargs):
        self.props.append(property_name)
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


def test_realtime_display_panel_exposes_surface_and_particle_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        liquid_view_mode="SURFACE_PARTICLES",
        show_surface=True,
        surface_enabled=True,
        particle_radius_scale=1.2,
        surface_dilate_iterations=0,
        surface_dilate_strength=1.0,
        surface_erode_iterations=0,
        surface_erode_strength=1.0,
        surface_smooth_type="BOX",
        surface_smooth_iterations=1,
        surface_smooth_strength=0.7,
        surface_attribute_mode="NONE",
        surface_material=None,
        show_particles=True,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        particle_attribute_mode="NONE",
        particle_material=None,
        show_whitewater=True,
        whitewater_enabled=True,
        whitewater_display_scale=1.0,
        whitewater_attribute_mode="NONE",
        whitewater_material=None,
    )
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_realtime_display()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    expected = {
        "liquid_view_mode",
        "show_surface",
        "surface_enabled",
        "particle_radius_scale",
        "surface_dilate_iterations",
        "surface_dilate_strength",
        "surface_erode_iterations",
        "surface_erode_strength",
        "surface_smooth_type",
        "surface_smooth_iterations",
        "surface_smooth_strength",
        "surface_attribute_mode",
        "surface_material",
        "show_particles",
        "particle_display_fraction",
        "particle_display_scale",
        "particle_attribute_mode",
        "particle_material",
        "show_whitewater",
        "whitewater_enabled",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
    }
    assert expected.issubset(set(layout.props))
    assert {
        "surface_filter_type",
        "surface_filter_iterations",
        "surface_filter_strength",
    }.isdisjoint(layout.props)


def test_realtime_display_panel_is_sole_owner_of_live_display_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        surface_enabled=True,
        whitewater_enabled=True,
        max_whitewater=1000,
        ww_churn_gain=1.0,
        ww_churn_decay=1.0,
        ww_churn_threshold=0.5,
        ww_emission_rate=2.0,
        ww_min_speed=0.5,
        ww_life_min=1.0,
        ww_life_max=3.0,
        ww_bubble_drag=4.0,
        ww_spray_drag=0.2,
        output_collection_name="FluidSim Output",
        show_overlay=True,
        cache_enabled=False,
        cache_path="",
    )
    scene = SimpleNamespace(fluidsim=fs)
    duplicate_owned = {
        "liquid_view_mode",
        "show_surface",
        "surface_enabled",
        "particle_radius_scale",
        "surface_dilate_iterations",
        "surface_dilate_strength",
        "surface_erode_iterations",
        "surface_erode_strength",
        "surface_smooth_type",
        "surface_smooth_iterations",
        "surface_smooth_strength",
        "surface_attribute_mode",
        "surface_material",
        "show_particles",
        "particle_display_fraction",
        "particle_display_scale",
        "particle_attribute_mode",
        "particle_material",
        "show_whitewater",
        "whitewater_enabled",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
    }

    for panel_cls in (panels.FLUIDSIM_PT_surface, panels.FLUIDSIM_PT_output):
        layout = _Layout()
        panel = panel_cls()
        panel.layout = layout
        panel.draw(SimpleNamespace(scene=scene))
        assert duplicate_owned.isdisjoint(layout.props), panel_cls.__name__


def test_realtime_property_callbacks_request_surface_or_output_refresh(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")

    calls = []
    scene = SimpleNamespace()
    fake_bridge = SimpleNamespace(
        request_scene_update=lambda scene, action, reason: calls.append(
            (scene, action, reason)
        )
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "scene_bridge", fake_bridge, raising=False)

    properties._request_surface_refresh(None, SimpleNamespace(scene=scene))
    properties._request_output_refresh(None, SimpleNamespace(scene=scene))

    assert calls == [
        (scene, "SURFACE", "Surface display settings changed"),
        (scene, "OUTPUT", "Viewport display settings changed"),
    ]


def test_surface_refresh_updates_active_solver_and_output(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    output_calls = []
    surface_calls = []
    fake_viewport = SimpleNamespace(
        update_outputs=lambda scene, sess: output_calls.append((scene, sess))
    )
    import fluidSim_warp_blender51.blender.viewport as viewport

    monkeypatch.setattr(viewport, "update_outputs", fake_viewport.update_outputs)
    solver = SimpleNamespace(
        settings=SimpleNamespace(),
        _surface_mesh_cache=object(),
        _surface_mesh_key=(10, 1.0),
        _surface=lambda: surface_calls.append("rebuilt"),
    )
    sess = scene_bridge.session()
    sess.solver = solver
    fs = SimpleNamespace(
        surface_enabled=False,
        particle_radius_scale=2.0,
        surface_filter_type="BOX",
        surface_filter_iterations=1,
        surface_filter_strength=0.7,
        surface_dilate_iterations=1,
        surface_dilate_strength=0.2,
        surface_erode_iterations=4,
        surface_erode_strength=0.45,
        surface_smooth_type="MEAN_CURVATURE_FLOW",
        surface_smooth_iterations=3,
        surface_smooth_strength=0.6,
        status="",
    )
    scene = SimpleNamespace(fluidsim=fs)

    scene_bridge.request_scene_update(scene, "SURFACE", "Surface changed")

    assert solver.settings.surface_enabled is False
    assert solver.settings.particle_radius_scale == 2.0
    assert solver.settings.surface_dilate_iterations == 1
    assert solver.settings.surface_dilate_strength == 0.2
    assert solver.settings.surface_erode_iterations == 4
    assert solver.settings.surface_erode_strength == 0.45
    assert solver.settings.surface_smooth_type.name == "MEAN_CURVATURE_FLOW"
    assert solver.settings.surface_smooth_iterations == 3
    assert solver.settings.surface_smooth_strength == 0.6
    assert solver._surface_mesh_cache is None
    assert solver._surface_mesh_key is None
    assert surface_calls == ["rebuilt"]
    assert output_calls == [(scene, sess)]


def test_whitewater_refresh_updates_active_solver_and_output(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    output_calls = []
    fake_viewport = SimpleNamespace(
        update_outputs=lambda scene, sess: output_calls.append((scene, sess))
    )
    import fluidSim_warp_blender51.blender.viewport as viewport

    monkeypatch.setattr(viewport, "update_outputs", fake_viewport.update_outputs)
    solver = SimpleNamespace(settings=SimpleNamespace(whitewater_enabled=True))
    sess = scene_bridge.session()
    sess.solver = solver
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(whitewater_enabled=False, status="")
    )

    scene_bridge.request_scene_update(scene, "WHITEWATER", "Whitewater changed")

    assert solver.settings.whitewater_enabled is False
    assert output_calls == [(scene, sess)]


def test_output_refresh_updates_existing_outputs_without_active_solver(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    refresh_calls = []
    fake_viewport = SimpleNamespace(
        refresh_existing_output_settings=lambda scene: refresh_calls.append(scene)
    )
    import fluidSim_warp_blender51.blender.viewport as viewport

    monkeypatch.setattr(
        viewport,
        "refresh_existing_output_settings",
        fake_viewport.refresh_existing_output_settings,
    )
    scene_bridge.session().solver = None
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))

    scene_bridge.request_scene_update(scene, "OUTPUT", "Viewport display settings changed")

    assert refresh_calls == [scene]
    assert scene.fluidsim.status == "Viewport display settings changed"


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
