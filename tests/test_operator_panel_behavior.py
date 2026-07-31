"""Focused UI/operator regression tests for Wave 2 remediation."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import textwrap
import time
import types
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_bpy_stub(monkeypatch, *, device="cuda:7"):
    fake_bpy = types.ModuleType("bpy")

    class _Operator:
        _poll_message = ""

        @classmethod
        def poll_message_set(cls, message):
            cls._poll_message = message

        def report(self, *_args, **_kwargs):
            return None

    class _Panel:
        pass

    class _AddonPreferences:
        pass

    def _prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(
        Operator=_Operator,
        Panel=_Panel,
        AddonPreferences=_AddonPreferences,
    )
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
    fake_bpy.types.Image = object
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


def test_pending_operators_route_context_scene_and_contain_failures(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    scene = SimpleNamespace(fluidsim=SimpleNamespace())
    calls = []
    fake_bridge = SimpleNamespace(
        apply_pending_reinitialize=lambda actual: calls.append(("reinit", actual)) or True,
        apply_pending_reset=lambda actual: calls.append(("reset", actual)) or True,
        revert_pending_values=lambda actual: calls.append(("revert", actual)) or True,
    )
    monkeypatch.setattr(operators, "_bridge", lambda: fake_bridge)
    context = SimpleNamespace(scene=scene)

    assert operators.FLUIDSIM_OT_apply_pending_reinitialize().execute(context) == {"FINISHED"}
    assert operators.FLUIDSIM_OT_apply_pending_reset().execute(context) == {"FINISHED"}
    assert operators.FLUIDSIM_OT_revert_pending().execute(context) == {"FINISHED"}
    assert calls == [("reinit", scene), ("reset", scene), ("revert", scene)]

    fake_bridge.apply_pending_reset = lambda _scene: (_ for _ in ()).throw(
        RuntimeError("blocked"))
    assert operators.FLUIDSIM_OT_apply_pending_reset().execute(context) == {"CANCELLED"}


def test_multi_frame_step_resyncs_scene_before_every_simulated_frame(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    scene = SimpleNamespace()
    sess = SimpleNamespace(active=True, error="")
    calls = []
    fake_bridge = SimpleNamespace(
        session=lambda: sess,
        sync_scene_to_solver=lambda actual_scene, actual_sess: calls.append(
            ("sync", actual_scene, actual_sess)),
        step_frame=lambda actual_scene, actual_sess: calls.append(
            ("step", actual_scene, actual_sess)),
    )
    monkeypatch.setattr(operators, "_bridge", lambda: fake_bridge)
    operator = operators.FLUIDSIM_OT_step()
    operator.frames = 3

    assert operator.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert [kind for kind, _scene, _sess in calls] == [
        "sync", "step", "sync", "step", "sync", "step"]
    assert all(actual_scene is scene and actual_sess is sess
               for _kind, actual_scene, actual_sess in calls)


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
        self.prop_calls = []
        self.prop_enum_calls = []
        self.popovers = []
        self.enabled = True

    def column(self, align=False):
        return self

    def row(self, align=False):
        return self

    def box(self):
        return self

    def separator(self):
        return None

    def prop(self, *args, **kwargs):
        if len(args) >= 2:
            self.props.append(args[1])
            self.prop_calls.append({"name": args[1], **kwargs})
        return None

    def prop_enum(self, _data, name, value, **kwargs):
        self.props.append(name)
        self.prop_enum_calls.append({"name": name, "value": value, **kwargs})
        return None

    def label(self, text="", icon="NONE"):
        self.labels.append({"text": text, "icon": icon})

    def operator(self, operator_id, **kwargs):
        self.operators.append({"id": operator_id, **kwargs})
        return SimpleNamespace()

    def popover(self, **kwargs):
        self.popovers.append(kwargs)
        return None


def test_pending_banner_and_fixed_header_expose_real_actions(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(live=True),
        fluidsim_pending_action="REINIT_PENDING",
        fluidsim_pending_count=2,
        fluidsim_pending_reasons="Reinitialize: voxel_size\nReinitialize: max_particles",
        fluidsim_pending_blocked="",
    )
    layout = _Layout()

    panels._draw_pending_banner(layout, scene, show_actions=True)
    header = SimpleNamespace(layout=layout)
    panels._draw_header_interaction(header, SimpleNamespace(scene=scene))

    assert any(label["text"] == "2 change(s) staged for restart"
               for label in layout.labels)
    assert {op["id"] for op in layout.operators} >= {
        "fluidsim.apply_pending_reinitialize", "fluidsim.revert_pending"}
    assert any(op.get("text") == "Restart Now" for op in layout.operators)
    assert any(op.get("text") == "Discard Changes" for op in layout.operators)
    assert getattr(layout, "alert", False) is False
    assert layout.popovers == [{
        "panel": "FLUIDSIM_PT_interaction_popover", "text": "2", "icon": "INFO"}]


def test_header_registration_is_symmetric(monkeypatch):
    bpy = _install_bpy_stub(monkeypatch)

    class Header:
        callbacks = []

        @classmethod
        def append(cls, callback):
            cls.callbacks.append(callback)

        @classmethod
        def remove(cls, callback):
            if callback in cls.callbacks:
                cls.callbacks.remove(callback)

    bpy.types.VIEW3D_HT_header = Header
    panels = _reload_module("fluidSim_warp_blender51.panels")

    panels.register()
    assert Header.callbacks == [panels._draw_header_interaction]
    panels.unregister()
    assert Header.callbacks == []


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


def test_simulation_panel_offers_safe_settings_for_gpu_capacity_error(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        status="Init failed", live=False, stat_particles=0, stat_voxels=0,
        stat_step_ms=0.0, stat_substeps=0, time_scale=1.0, cfl=3.0,
        min_substeps=1, max_substeps=4, use_scene_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    ))
    scene_bridge.set_error(scene, "CUDA error 2: out of memory")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene))

    assert any(
        op["id"] == "fluidsim.apply_safe_settings" for op in layout.operators
    )


def test_simulation_panel_offers_explicit_run_anyway_for_gpu_capacity_error(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        status="Init failed", live=False, stat_particles=0, stat_voxels=0,
        stat_step_ms=0.0, stat_substeps=0, time_scale=1.0, cfl=3.0,
        min_substeps=1, max_substeps=4, use_scene_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    ))
    scene_bridge.set_error(scene, "FluidSim memory safeguard stopped initialization")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene))

    assert any(
        op["id"] == "fluidsim.initialize"
        and op.get("text") == "Ignore Estimate & Initialize"
        for op in layout.operators
    )


def test_initialize_operator_passes_explicit_memory_override(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    seen = []
    solver = SimpleNamespace(
        device="cuda:0",
        grid=SimpleNamespace(capacity=1),
        particles=SimpleNamespace(capacity=1),
    )
    bridge = SimpleNamespace(
        initialize_session=lambda scene, **kwargs: (
            seen.append((scene, kwargs)) or SimpleNamespace(solver=solver)
        )
    )
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    op = operators.FLUIDSIM_OT_initialize()
    op.ignore_memory_safeguard = True

    assert op.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert seen == [(scene, {"ignore_memory_safeguard": True})]


def test_capacity_warning_offers_approval_gated_recommended_option(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    suggestion = SimpleNamespace(
        key="preserve_simulation",
        label="Preserve authored domain and resolution",
        changes={"max_particles": 120},
        tradeoffs=("Raises only fixed GPU capacity.",),
    )
    plan = SimpleNamespace(
        particle_demand=SimpleNamespace(value=120, quality=SimpleNamespace(value="exact")),
        leaf_demand=SimpleNamespace(value=1, quality=SimpleNamespace(value="exact")),
        estimated_bytes=1024,
        budget_bytes=2048,
        blockers=(SimpleNamespace(
            code="PARTICLE_CAPACITY", required=120, available=100,
            message="particle capacity"),),
        suggestions=(suggestion,),
    )
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        status="Capacity warning", live=False, stat_particles=0, stat_voxels=0,
        stat_step_ms=0.0, stat_substeps=0, time_scale=1.0, cfl=3.0,
        min_substeps=1, max_substeps=4, use_scene_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    ))
    scene_bridge.set_error(scene, "source capacity blocked")
    monkeypatch.setattr(
        scene_bridge, "session",
        lambda: SimpleNamespace(
            active=False, capacity_plan=plan,
            current_error=lambda actual: "source capacity blocked"),
    )
    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene))

    assert any(
        op["id"] == "fluidsim.apply_capacity_suggestion"
        and op.get("text") == "Review Recommended Changes & Run"
        for op in layout.operators
    )
    assert not any(
        op["id"] == "fluidsim.apply_safe_settings" for op in layout.operators)
    assert any("Nothing has been changed" in label["text"]
               for label in layout.labels)


def test_capacity_option_changes_nothing_until_approval_then_retries(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="preserve_simulation",
        label="Preserve authored domain and resolution",
        changes={"max_particles": 120, "leaf_capacity": 4},
        estimated_bytes=1024,
        budget_bytes=2048,
    )
    plan = SimpleNamespace(
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=())
    int_property = SimpleNamespace(type="INT")
    fs = SimpleNamespace(
        max_particles=100,
        leaf_capacity=2,
        domain_object=None,
        status="Capacity warning",
        bl_rna=SimpleNamespace(properties={
            "max_particles": int_property,
            "leaf_capacity": int_property,
        }),
    )
    scene = SimpleNamespace(fluidsim=fs)
    initialized = []
    bridge = SimpleNamespace(
        session=lambda: SimpleNamespace(capacity_plan=plan),
        refresh_capacity_plan=lambda actual: plan,
        clear_error=lambda actual: None,
        initialize_session=lambda actual: (
            initialized.append(actual) or SimpleNamespace(
                solver=SimpleNamespace(device="cuda:0"))),
    )
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    dialogs = []
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda actual, width: (
                dialogs.append((actual, width)) or {"RUNNING_MODAL"})),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "preserve_simulation"

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert dialogs == [(op, 560)]
    assert fs.max_particles == 100
    assert fs.leaf_capacity == 2
    assert op.expected_scene_signature == "scene-v1"
    assert op.execute(context) == {"FINISHED"}
    assert fs.max_particles == 120
    assert fs.leaf_capacity == 4
    assert initialized == [scene]


def test_realtime_display_panel_exposes_surface_and_particle_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        liquid_view_mode="SURFACE_PARTICLES",
        show_surface=True,
        surface_enabled=True,
        particle_radius_scale=1.2,
        surface_splat_radius=2,
        surface_resolution_mode="MULTIPLIER",
        surface_resolution_multiplier=2.0,
        surface_voxel_size=0.0,
        surface_leaf_capacity_scale=8.0,
        surface_max_dense_dim=512,
        surface_update_cadence="FRAME",
        live_surface_publish_cadence=4,
        surface_build_lowres_sdf=True,
        surface_filter_type="BOX",
        surface_filter_iterations=1,
        surface_filter_strength=0.7,
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
            "surface_splat_radius",
            "surface_resolution_mode",
            "surface_resolution_multiplier",
            "surface_leaf_capacity_scale",
            "surface_max_dense_dim",
            "surface_update_cadence",
            "live_surface_publish_cadence",
            "surface_build_lowres_sdf",
            "surface_filter_type",
            "surface_filter_iterations",
            "surface_filter_strength",
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
    }
    assert expected.issubset(set(layout.props))
    assert {
        "show_whitewater",
        "whitewater_enabled",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
    }.isdisjoint(layout.props)


def test_whitewater_panel_owns_all_whitewater_controls(monkeypatch):
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
        show_whitewater=True,
        whitewater_display_scale=1.0,
        whitewater_attribute_mode="NONE",
        whitewater_material=None,
    )
    scene = SimpleNamespace(fluidsim=fs)
    expected = {
        "show_whitewater",
        "whitewater_enabled",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
        "max_whitewater",
        "ww_churn_gain",
        "ww_churn_decay",
        "ww_churn_threshold",
        "ww_emission_rate",
        "ww_min_speed",
        "ww_life_min",
        "ww_life_max",
        "ww_bubble_drag",
        "ww_spray_drag",
    }

    layout = _Layout()
    panel = panels.FLUIDSIM_PT_surface()
    panel.layout = layout
    panel.draw_header(SimpleNamespace(scene=scene))
    panel.draw(SimpleNamespace(scene=scene))
    assert expected.issubset(layout.props)

    realtime_source = inspect.getsource(panels.FLUIDSIM_PT_realtime_display.draw)
    assert all(name not in realtime_source for name in expected)


def test_collider_panel_has_only_per_object_material_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    source = inspect.getsource(panels.FLUIDSIM_PT_colliders.draw)

    assert 'col.prop(pg, "collider_friction")' in source
    assert 'col.prop(pg, "collider_restitution")' in source
    assert 'col.prop(fs, "friction")' not in source
    assert 'col.prop(fs, "restitution")' not in source


def test_realtime_property_callbacks_request_surface_or_output_refresh(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    from fluidSim_warp_blender51.blender.scene_bridge import SceneUpdate

    calls = []
    scene = SimpleNamespace()
    fake_bridge = SimpleNamespace(
        request_scene_update=lambda scene, action: calls.append((scene, action))
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "scene_bridge", fake_bridge, raising=False)

    properties._request_surface_refresh(None, SimpleNamespace(scene=scene))
    properties._request_output_refresh(None, SimpleNamespace(scene=scene))

    assert calls == [
        (scene, SceneUpdate.SURFACE),
        (scene, SceneUpdate.OUTPUT),
    ]


def test_hot_solver_property_callback_requests_live_update(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    from fluidSim_warp_blender51.blender.scene_bridge import SceneUpdate

    calls = []
    scene = SimpleNamespace()
    fake_bridge = SimpleNamespace(
        request_scene_update=lambda scene, action: calls.append((scene, action))
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "scene_bridge", fake_bridge, raising=False)

    properties._request_live(None, SimpleNamespace(scene=scene))

    assert calls == [(scene, SceneUpdate.LIVE)]


def test_live_solver_settings_apply_without_resetting_particle_state(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    particle_state = object()
    solver = SimpleNamespace(
        settings=SimpleNamespace(flip_ratio=0.95, cfl=1.0),
        particles=particle_state,
    )
    sess = scene_bridge.session()
    sess.solver = solver
    reset_calls = []
    monkeypatch.setattr(sess, "reset", lambda _scene: reset_calls.append("reset"))
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(flip_ratio=0.25, cfl=2.0, status=""),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.LIVE)

    assert solver.settings.flip_ratio == 0.25
    assert solver.settings.cfl == 2.0
    assert solver.particles is particle_state
    assert reset_calls == []
    assert scene.fluidsim.status == ""


def test_live_setting_errors_are_atomic_contained_and_visible(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(sample_type="previous", flip_ratio=0.9)
    solver = SimpleNamespace(settings=settings, particles=object())
    sess = scene_bridge.session()
    sess.solver = solver
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            sample_type="NOT_A_REAL_ENUM", flip_ratio=0.2, status="Frame 8"),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.LIVE)

    assert sess.solver is solver
    assert settings.sample_type == "previous"
    assert settings.flip_ratio == 0.9
    assert "NOT_A_REAL_ENUM" in sess.current_error(scene)
    assert scene.fluidsim.status.startswith("Live settings error:")


def test_particles_per_voxel_live_refresh_rescales_automatic_st_mass(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(
        particles_per_voxel=8.0,
        spacetime_enabled=True,
        spacetime_reference_mass=0.0,
    )
    solver = SimpleNamespace(
        settings=settings,
        particles=object(),
        _st_reference_mass_cached=4.0,
    )
    solver._calibrate_st_reference_mass = lambda: (_ for _ in ()).throw(
        AssertionError("existing calibration should be density-rescaled"))
    sess = scene_bridge.session()
    sess.solver = solver
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(particles_per_voxel=12.0, status="Frame 4"),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.LIVE)

    assert settings.particles_per_voxel == 12.0
    assert solver._st_reference_mass_cached == 6.0
    assert scene.fluidsim.status == "Frame 4"


def test_live_gravity_uses_scene_or_custom_branch_without_reset(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(gravity=(0.0, 0.0, 0.0))
    sess = scene_bridge.session()
    sess.solver = SimpleNamespace(settings=settings, particles=object())
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            gravity=(1.0, 2.0, 3.0), use_scene_gravity=True, status="stats"),
        use_gravity=True,
        gravity=(0.0, 0.0, -4.0),
    )

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.LIVE)
    assert settings.gravity == (0.0, 0.0, -4.0)

    scene.fluidsim.use_scene_gravity = False
    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.LIVE)
    assert settings.gravity == (1.0, 2.0, 3.0)
    assert scene.fluidsim.status == "stats"


def test_inactive_live_update_status_is_truthful(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene_bridge.session().solver = None
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.LIVE)

    assert scene.fluidsim.status == "Live setting stored; applies on initialization"


def test_timeline_reset_reapplies_current_solver_settings(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    import fluidSim_warp_blender51.blender.viewport as viewport

    from fluidSim_warp_blender51.sim.solver import SolverSettings

    reset_snapshots = []
    released = []
    planned_capacities = []
    prepared = []
    solver = SimpleNamespace(
        settings=SolverSettings(
            max_particles=1024, flip_ratio=0.95, spacetime_seed=1),
        particles=SimpleNamespace(capacity=4096),
        reset=lambda: reset_snapshots.append((
            solver.settings.flip_ratio, solver.settings.spacetime_seed)),
        release=lambda: released.append("old"),
        device="cuda:0",
    )
    sess = scene_bridge.session()
    sess.solver = solver
    capacity_plan = SimpleNamespace(blockers=())
    snapshot = SimpleNamespace(
        capacity_plan=capacity_plan, seeds=(), scene_inputs=sess.scene_inputs)
    monkeypatch.setattr(
        scene_bridge, "build_settings",
        lambda _scene: SolverSettings(
            device="cuda:0", max_particles=1024,
            flip_ratio=0.2, spacetime_seed=77))

    def plan_scene(
            _scene, settings, _resources, *,
            additional_transaction_bytes=0):
        assert additional_transaction_bytes > 0
        planned_capacities.append(settings.max_particles)
        return snapshot

    monkeypatch.setattr(sess, "_plan_scene", plan_scene)

    def prepare_replacement(_scene, settings, _snapshot, _profiler):
        candidate = SimpleNamespace(
            settings=settings,
            particles=SimpleNamespace(capacity=settings.max_particles),
        )
        prepared.append(candidate)
        return candidate, ()

    monkeypatch.setattr(sess, "_prepare_solver_replacement", prepare_replacement)
    monkeypatch.setattr(viewport, "update_outputs", lambda *_args: None)
    scene = SimpleNamespace(
        frame_start=1,
        fluidsim=SimpleNamespace(
            flip_ratio=0.2, spacetime_seed=77, status="", snippets=[],
            gravity=(0.0, 0.0, -4.0), use_scene_gravity=False),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )

    sess.reset(scene)

    assert planned_capacities == [4096]
    assert reset_snapshots == []
    assert released == ["old"]
    assert sess.solver is prepared[0]
    assert sess.solver.particles.capacity == 4096
    assert sess.solver.settings.flip_ratio == 0.2
    assert sess.solver.settings.spacetime_seed == 77
    assert scene.fluidsim.status == "Simulation reset"


def test_solver_property_interaction_tiers_are_mechanically_exhaustive(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(properties.FLUIDSIM_PG_settings)))
    callbacks = {}
    for node in tree.body[0].body:
        if not (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.annotation, ast.Call)):
            continue
        update = next(
            (kw.value for kw in node.annotation.keywords if kw.arg == "update"),
            None,
        )
        if isinstance(update, ast.Name):
            callbacks[node.target.id] = update.id

    live_props = {name for name, callback in callbacks.items()
                  if callback == "_request_live"}
    assert live_props == set(bridge._LIVE_SETTING_FIELDS) | {
        "gravity", "use_scene_gravity"}
    declared_callbacks = {
        "_on_live_changed", "_request_rebuild", "_request_live",
        "_request_reset", "_request_surface_refresh",
        "_request_whitewater_refresh", "_apply_liquid_view_mode",
        "_sync_liquid_view_mode", "_request_output_refresh",
        "_request_particle_capacity", "_on_cache_playback_changed",
    }
    assert set(callbacks.values()) == declared_callbacks

    allocation = {"voxel_size", "leaf_capacity", "max_particles", "max_whitewater"}
    assert allocation.isdisjoint(
        bridge._LIVE_SETTING_FIELDS | bridge._RESET_SETTING_FIELDS)
    assert {
        name for name, callback in callbacks.items()
        if callback == "_request_particle_capacity"
    } == {
        "max_particles", "auto_grow_particles",
        "particle_growth_failure_policy",
    }
    assert bridge._HOT_SETTING_FIELDS.isdisjoint(bridge._RESET_SETTING_FIELDS)
    assert bridge._HOT_SETTING_FIELDS.isdisjoint(bridge._SURFACE_SETTING_FIELDS)
    assert bridge._LIVE_REFRESH_SETTING_FIELDS == {"particles_per_voxel"}
    enum_sources = {source for source, _enum in bridge._enum_setting_fields().values()}
    appliable = set(bridge._DIRECT_SETTING_FIELDS) | enum_sources
    assert (bridge._LIVE_SETTING_FIELDS | bridge._RESET_SETTING_FIELDS).issubset(
        appliable)
    assert bridge._RESET_SETTING_FIELDS.issubset(
        {name for name, callback in callbacks.items()
         if callback == "_request_reset"})
    assert bridge._SURFACE_SETTING_FIELDS.issubset(
        {name for name, callback in callbacks.items()
         if callback == "_request_surface_refresh"})
    bridge._validate_setting_registry()


def test_live_reconcile_is_bounded_after_registry_warmup(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(flip_ratio=0.9, cfl=1.0)
    sess = bridge.session()
    sess.solver = SimpleNamespace(settings=settings)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(flip_ratio=0.2, cfl=2.0),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )
    assert bridge._apply_realtime_solver_settings(scene, sess)

    started = time.perf_counter()
    for _ in range(500):
        assert bridge._apply_realtime_solver_settings(scene, sess)
    elapsed = time.perf_counter() - started

    assert elapsed / 500 < 0.001


def test_live_property_lifecycle_initializes_once_and_updates_status(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    init_calls = []
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    sess = scene_bridge.session()
    sess.solver = None

    def fake_initialize(scene_arg):
        init_calls.append(scene_arg)
        sess.solver = object()
        return sess

    monkeypatch.setattr(sess, "initialize", fake_initialize)

    scene_bridge.set_live_enabled(scene, True)
    scene_bridge.set_live_enabled(scene, True)
    scene_bridge.set_live_enabled(scene, False)

    assert init_calls == [scene]
    assert sess.active
    assert scene.fluidsim.status == "Live simulation disabled"


def test_live_property_callback_toggles_scene_bridge_directly(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")

    calls = []
    scene = SimpleNamespace()
    fake_bridge = SimpleNamespace(
        set_live_enabled=lambda scene, enabled: calls.append((scene, enabled))
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "scene_bridge", fake_bridge, raising=False)

    properties._on_live_changed(SimpleNamespace(live=True, status=""), SimpleNamespace(scene=scene))

    assert calls == [(scene, True)]


def test_scene_bridge_set_live_enabled_updates_status(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    sess = scene_bridge.session()

    def fake_initialize(_scene):
        sess.solver = object()
        return sess

    monkeypatch.setattr(sess, "initialize", fake_initialize)

    scene_bridge.set_live_enabled(scene, True)
    assert scene.fluidsim.status == "Live simulation enabled - press play"

    scene_bridge.set_live_enabled(scene, False)
    assert scene.fluidsim.status == "Live simulation disabled"


def test_frame_change_solver_error_disables_live_and_releases_session(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(
        frame_current=2,
        fluidsim=SimpleNamespace(
            live=True,
            cache_playback_enabled=False,
            status="",
        ),
    )
    sess = scene_bridge.session()
    sess.solver = object()
    sess.start_frame = 1
    sess.last_frame = 1

    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sess,
        "step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    scene_bridge._on_frame_change(scene, depsgraph=object())

    assert scene.fluidsim.live is False
    assert scene_bridge.session().active is False
    assert scene_bridge.current_error(scene) == "RuntimeError('boom')"
    assert scene.fluidsim.status == "Solver error: boom"


def test_preferences_warp_probe_is_cached_until_explicit_refresh(monkeypatch):
    _install_bpy_stub(monkeypatch)
    preferences = _reload_module("fluidSim_warp_blender51.preferences")
    calls = []
    fake_warp = SimpleNamespace(
        init=lambda: calls.append("init"),
        is_cuda_available=lambda: True,
        get_device=lambda _name: SimpleNamespace(name="Test GPU"),
        config=SimpleNamespace(version="test"),
    )
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    preferences._WARP_PROBE_RESULT = None

    first = preferences._probe_warp()
    second = preferences._probe_warp()
    assert first == second == (True, "Warp test - Test GPU")
    assert calls == ["init"]

    preferences._probe_warp(force=True)
    assert calls == ["init", "init"]


def test_addon_unregister_releases_runtime_handlers(monkeypatch):
    fake_bpy = _install_bpy_stub(monkeypatch)
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(
            frame_change_post=[],
            render_pre=[],
            render_post=[],
        )
    )
    addon = _reload_module("fluidSim_warp_blender51")
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    render = _reload_module("fluidSim_warp_blender51.blender.render")
    viewport = _reload_module("fluidSim_warp_blender51.blender.viewport")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    preferences = _reload_module("fluidSim_warp_blender51.preferences")
    properties = _reload_module("fluidSim_warp_blender51.properties")

    calls = []
    monkeypatch.setattr(viewport, "unregister_overlay", lambda: calls.append("viewport"))
    monkeypatch.setattr(render, "unregister_handlers", lambda: calls.append("render"))
    monkeypatch.setattr(scene_bridge, "unregister_handlers", lambda: calls.append("scene_handlers"))
    monkeypatch.setattr(scene_bridge, "release_session", lambda: calls.append("release"))
    monkeypatch.setattr(panels, "unregister", lambda: calls.append("panels"))
    monkeypatch.setattr(operators, "unregister", lambda: calls.append("operators"))
    monkeypatch.setattr(preferences, "unregister", lambda: calls.append("preferences"))
    monkeypatch.setattr(properties, "unregister", lambda: calls.append("properties"))

    addon.unregister()

    assert calls == [
        "viewport",
        "render",
        "scene_handlers",
        "release",
        "panels",
        "operators",
        "preferences",
        "properties",
    ]


def test_surface_refresh_updates_active_solver_and_output(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    output_calls = []
    surface_calls = []
    fake_viewport = SimpleNamespace(
        update_outputs=lambda scene, sess, **kwargs: output_calls.append(
            (scene, sess, kwargs))
    )
    import fluidSim_warp_blender51.blender.viewport as viewport

    monkeypatch.setattr(viewport, "update_outputs", fake_viewport.update_outputs)
    solver = SimpleNamespace(
        settings=SimpleNamespace(),
        rebuild_surface=lambda: surface_calls.append("rebuilt"),
    )
    sess = scene_bridge.session()
    sess.solver = solver
    fs = SimpleNamespace(
        surface_enabled=False,
        particle_radius_scale=2.0,
        surface_splat_radius=2,
        surface_resolution_mode="MULTIPLIER",
        surface_resolution_multiplier=2.0,
        surface_voxel_size=0.0,
        surface_leaf_capacity_scale=8.0,
        surface_max_dense_dim=512,
        surface_update_cadence="FRAME",
        live_surface_publish_cadence=4,
        surface_build_lowres_sdf=True,
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

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.SURFACE)

    assert solver.settings.surface_enabled is False
    assert solver.settings.particle_radius_scale == 2.0
    assert solver.settings.surface_dilate_iterations == 1
    assert solver.settings.surface_dilate_strength == 0.2
    assert solver.settings.surface_erode_iterations == 4
    assert solver.settings.surface_erode_strength == 0.45
    assert solver.settings.surface_smooth_type.name == "MEAN_CURVATURE_FLOW"
    assert solver.settings.surface_smooth_iterations == 3
    assert solver.settings.surface_smooth_strength == 0.6
    assert surface_calls == ["rebuilt"]
    assert output_calls == [(scene, sess, {"force_surface": True})]


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

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.WHITEWATER)

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

    scene_bridge.request_scene_update(scene, scene_bridge.SceneUpdate.OUTPUT)

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


def _phase1_object(role, **overrides):
    values = dict(
        role=role,
        enabled=True,
        geometry_mode="ANALYTIC",
        shape="BOX",
        mesh_object=None,
        mesh_evaluation_mode="FIRST_SIM_FRAME",
        mesh_surface_thickness=0.0,
        sink_region_mode="SOLID",
        mesh_error="",
        emission_mode="CONTINUOUS_INFLOW",
        emit_rate=8.0,
        initial_velocity=(0.0, 0.0, 0.0),
        use_object_velocity=True,
        temperature=0.0,
        viscosity=0.0,
        density=1.0,
        use_lifespan=False,
        life_min=2.0,
        life_max=4.0,
        force_type="OFF",
        force_strength=1.0,
        force_direction=(0.0, 0.0, 1.0),
        collider_friction=0.0,
        collider_restitution=0.0,
        transfer_velocity=True,
    )
    values.update(overrides)
    return SimpleNamespace(
        name=f"{role.title()} Object", type="MESH",
        fluidsim=SimpleNamespace(**values), display_type="TEXTURED",
        hide_render=False,
    )


def test_phase1_source_draw_exposes_real_lifecycle_and_hides_compatibility(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    source = _phase1_object("SOURCE")
    fs = SimpleNamespace(
        height_enabled=True, height_base=0.3,
        height_velocity=(0.0, 0.0, 0.0), height_emit_particles=True,
    )
    scene = SimpleNamespace(fluidsim=fs, objects=[source])
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_liquid_emitters()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene, active_object=source))

    assert {"emission_mode", "emit_rate", "geometry_mode"}.issubset(layout.props)
    assert {
        ("ANALYTIC", "Analytic Volume"),
        ("MESH", "Mesh Surface Shell"),
    }.issubset({(call["value"], call["text"])
                for call in layout.prop_enum_calls})
    assert "source_mode" not in layout.props
    assert "height_back_mode" not in layout.props
    assert any(call == {"name": "emit_rate", "text": "Target Particles / Voxel"}
               for call in layout.prop_calls)
    assert any(op["id"] == "fluidsim.tag_liquid_emitter"
               for op in layout.operators)
    assert not any(op["id"] == "fluidsim.inject_particles"
                   for op in layout.operators)


def test_phase1_particle_attributes_are_source_only_and_collapsed(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    source = _phase1_object("SOURCE", use_lifespan=True)
    context = SimpleNamespace(active_object=source)
    assert panels.FLUIDSIM_PT_liquid_emitter_attributes.poll(context)
    assert "DEFAULT_CLOSED" in panels.FLUIDSIM_PT_liquid_emitter_attributes.bl_options
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_liquid_emitter_attributes()
    panel.layout = layout
    panel.draw(context)
    assert {"temperature", "viscosity", "density", "use_lifespan",
            "life_min", "life_max"}.issubset(layout.props)
    assert any(call == {"name": "density", "text": "Density Attribute"}
               for call in layout.prop_calls)
    assert not panels.FLUIDSIM_PT_liquid_emitter_attributes.poll(
        SimpleNamespace(active_object=_phase1_object("SINK")))


def test_phase1_sink_uses_shared_region_editor_in_sink_workflow(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    sink = _phase1_object(
        "SINK", geometry_mode="MESH", mesh_error="Mesh must be watertight")
    scene = SimpleNamespace(objects=[sink], fluidsim=SimpleNamespace())
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_liquid_sinks()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene, active_object=sink))

    assert {"geometry_mode", "mesh_object", "mesh_evaluation_mode",
            "sink_region_mode"}.issubset(layout.props)
    assert "mesh_surface_thickness" not in layout.props
    assert {
        ("ANALYTIC", "Analytic Volume"), ("MESH", "Mesh")
    }.issubset({(call["value"], call["text"])
                for call in layout.prop_enum_calls})
    assert {"text": "Mesh must be watertight", "icon": "ERROR"} in layout.labels
    assert any(op["id"] == "fluidsim.tag_liquid_sink"
               for op in layout.operators)


def test_phase1_force_editor_requires_force_role_and_tag_action(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        directional_force=(0.0, 0.0, 0.0),
        omnidirectional_force_strength=0.0,
        omnidirectional_force_center=(0.0, 0.0, 0.0),
        omnidirectional_force_radius=1.0,
        force_diffusion=0.0,
        dissipation_enabled=False,
        dissipation_attribute="TEMPERATURE",
        dissipation_rate=0.0,
        dissipation_target=0.0,
        attraction_enabled=False,
        attraction_strength=0.0,
        attraction_max_distance=1.0,
    )
    scene = SimpleNamespace(fluidsim=fs)

    force_layout = _Layout()
    panel = panels.FLUIDSIM_PT_forces()
    panel.layout = force_layout
    panel.draw(SimpleNamespace(scene=scene, active_object=_phase1_object("FORCE")))
    assert {"enabled", "shape", "force_type"}.issubset(force_layout.props)
    assert any(op["id"] == "fluidsim.tag_force_region"
               for op in force_layout.operators)

    source_layout = _Layout()
    panel.layout = source_layout
    panel.draw(SimpleNamespace(scene=scene, active_object=_phase1_object("SOURCE")))
    assert "force_type" not in source_layout.props
    assert any("Select a tagged Force Region" in label["text"]
               for label in source_layout.labels)


def test_phase1_tag_operator_supports_force_and_has_disabled_state(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    obj = _phase1_object("NONE")
    op = operators.FLUIDSIM_OT_tag_object()
    op.role = "FORCE"
    context = SimpleNamespace(selected_objects=[obj])

    assert operators.FLUIDSIM_OT_tag_object.poll(context)
    assert op.execute(context) == {"FINISHED"}
    assert obj.fluidsim.role == "FORCE"
    assert obj.display_type == "WIRE"
    assert obj.hide_render is True
    assert not operators.FLUIDSIM_OT_tag_object.poll(
        SimpleNamespace(selected_objects=[]))
    assert "Select at least one object" in operators.FLUIDSIM_OT_tag_object._poll_message


def test_phase3_field_volume_input_is_drawn_once_when_region_and_payload_use_volume(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    field = _phase1_object(
        "FIELD", source_mode="FIELD", field_operation="SCALAR",
        field_region_kind="VOLUME_GRID", field_value_source="VOLUME_GRID",
        volume_input_object=None, volume_scalar_grid="temperature",
        volume_vector_grid="", volume_vector_x_grid="",
        volume_vector_y_grid="", volume_vector_z_grid="",
        blend_mode="REPLACE", field_amount=1.0,
        source_status_message="", source_status_severity="",
        source_status_code="", source_status_capacity_limited=False)
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_advanced_field_sources(); panel.layout = layout
    panel.draw(SimpleNamespace(
        scene=SimpleNamespace(objects=[field], fluidsim=SimpleNamespace()),
        active_object=field))
    assert [call["name"] for call in layout.prop_calls].count(
        "volume_input_object") == 1


def test_phase3_aggregate_emission_is_not_broadcast_as_per_source_truth():
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs

    def obj(pointer):
        pg = SimpleNamespace(
            role="SOURCE", source_mode="PARTICLES", enabled=True,
            source_status_code="", source_status_message="",
            source_status_severity="", source_status_emitted=0,
            source_status_capacity_limited=False)
        return SimpleNamespace(fluidsim=pg, as_pointer=lambda: pointer)

    first, second = obj(1), obj(2)
    inputs = SceneInputs()
    inputs.update_source_status(
        SimpleNamespace(objects=[first, second]), (),
        {"emitted": 12, "capacity_limited": True})
    assert first.fluidsim.source_status_emitted == 0
    assert second.fluidsim.source_status_emitted == 0
    second.fluidsim.enabled = False
    inputs.update_source_status(
        SimpleNamespace(objects=[first, second]), (),
        {"emitted": 12, "capacity_limited": True})
    assert first.fluidsim.source_status_emitted == 12
    assert first.fluidsim.source_status_capacity_limited is True


def test_phase1_point_injection_poll_is_truthful(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    sess = SimpleNamespace(active=True)
    monkeypatch.setattr(operators, "_bridge", lambda: SimpleNamespace(session=lambda: sess))

    mesh = SimpleNamespace(type="MESH", particle_systems=[])
    assert operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=mesh))
    unsupported = SimpleNamespace(type="LIGHT", particle_systems=[])
    assert not operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=unsupported))
    assert "mesh vertices" in operators.FLUIDSIM_OT_inject_particles._poll_message
    sess.active = False
    assert not operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=mesh))
    assert "Initialize the solver" in operators.FLUIDSIM_OT_inject_particles._poll_message


def test_phase1_point_injection_capacity_refusal_is_recoverable_cancel(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import sources, viewport

    result = SimpleNamespace(
        admitted=False, emitted=0, required=5, available=4,
        capacity_limited=True, total=1)
    solver = SimpleNamespace(sourcing=SimpleNamespace(
        last_injection_result=result))
    sess = SimpleNamespace(active=True, solver=solver)
    fatal_errors = []
    bridge = SimpleNamespace(
        session=lambda: sess,
        set_error=lambda *_args: fatal_errors.append(_args))
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    monkeypatch.setattr(sources, "inject_object_particles", lambda *_args: 0)
    viewport_updates = []
    monkeypatch.setattr(
        viewport, "update_outputs", lambda *_args: viewport_updates.append(_args))

    reports = []
    op = operators.FLUIDSIM_OT_inject_particles()
    op.report = lambda level, message: reports.append((level, message))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status="Solver ready"))
    obj = SimpleNamespace(name="Seed", type="MESH", particle_systems=[])

    assert op.execute(SimpleNamespace(
        scene=scene, active_object=obj)) == {"CANCELLED"}
    assert not viewport_updates
    assert not fatal_errors
    assert scene.fluidsim.status == "Solver ready"
    assert "5 slots" in reports[-1][1]
    assert "4" in reports[-1][1]
    assert "Max Particles" in reports[-1][1]


def test_phase1_empty_point_injection_clears_refusal_and_finishes(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import sources, viewport

    sourcing_state = SimpleNamespace(last_injection_result=SimpleNamespace(
        admitted=False, required=5, available=4))

    class Solver:
        sourcing = sourcing_state

        def inject_particle_packets_exact(self, packets):
            assert tuple(packets) == ()
            result = SimpleNamespace(
                admitted=True, emitted=0, required=0, available=4,
                capacity_limited=False, total=0)
            self.sourcing.last_injection_result = result
            return result

    sess = SimpleNamespace(active=True, solver=Solver())
    monkeypatch.setattr(
        operators, "_bridge", lambda: SimpleNamespace(session=lambda: sess))
    monkeypatch.setattr(sources, "prepare_object_particles", lambda *_args: None)
    viewport_updates = []
    monkeypatch.setattr(
        viewport, "update_outputs", lambda *_args: viewport_updates.append(_args))

    reports = []
    op = operators.FLUIDSIM_OT_inject_particles()
    op.report = lambda level, message: reports.append((level, message))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status="Solver ready"))
    obj = SimpleNamespace(name="Empty", type="MESH", particle_systems=[])

    assert op.execute(SimpleNamespace(
        scene=scene, active_object=obj)) == {"FINISHED"}
    assert len(viewport_updates) == 1
    assert reports[-1][0] == {"INFO"}
    assert "Injected 0 particles" in reports[-1][1]


def test_capacity_suggestion_poll_and_dialog_are_scene_owned(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="safe", label="Safe", changes={"max_particles": 12},
        estimated_bytes=1, budget_bytes=2)
    plan = SimpleNamespace(
        scene_signature="identical-content", suggestions=(suggestion,), blockers=())
    first = SimpleNamespace(fluidsim=SimpleNamespace(max_particles=8))
    second = SimpleNamespace(fluidsim=SimpleNamespace(max_particles=8))
    bridge.session()._set_capacity_plan(first, plan)
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)

    assert operators.FLUIDSIM_OT_apply_capacity_suggestion.poll(
        SimpleNamespace(scene=first)) is True
    assert operators.FLUIDSIM_OT_apply_capacity_suggestion.poll(
        SimpleNamespace(scene=second)) is False

    dialogs = []
    context_first = SimpleNamespace(
        scene=first,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda op, width: (
                dialogs.append((op, width)) or {"RUNNING_MODAL"})),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "safe"
    assert op.invoke(context_first, None) == {"RUNNING_MODAL"}

    refreshed = []
    monkeypatch.setattr(
        bridge, "refresh_capacity_plan",
        lambda scene: refreshed.append(scene) or plan)
    assert op.execute(SimpleNamespace(scene=second)) == {"CANCELLED"}
    assert refreshed == []
    assert second.fluidsim.max_particles == 8


def test_capacity_dialog_rechecks_same_scene_signature_before_mutation(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="safe", label="Safe", changes={"max_particles": 12},
        estimated_bytes=1, budget_bytes=2)
    original = SimpleNamespace(
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=())
    changed = SimpleNamespace(
        scene_signature="scene-v2", suggestions=(suggestion,), blockers=())
    fs = SimpleNamespace(max_particles=8, status="")
    scene = SimpleNamespace(fluidsim=fs)
    bridge.session()._set_capacity_plan(scene, original)
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda _op, width: {"RUNNING_MODAL"}),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "safe"
    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    monkeypatch.setattr(
        bridge, "refresh_capacity_plan",
        lambda actual: bridge.session()._set_capacity_plan(actual, changed))

    assert op.execute(context) == {"CANCELLED"}
    assert fs.max_particles == 8
    assert "changed" in fs.status.lower()
