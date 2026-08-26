"""Focused UI/operator regression tests for Wave 2 remediation."""

from __future__ import annotations

import importlib
import inspect
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
        op["id"] == "fluidsim.initialize" and op.get("text") == "Run Anyway"
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


def test_realtime_display_panel_exposes_surface_and_particle_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    sess = scene_bridge.session()
    sess.presentation_mode = "CPU Fallback"
    sess.presentation_reason = "forced unavailable"
    sess.presentation_bytes = 120000
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
        native_gpu_interop=True,
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
    assert {"text": "Mode: CPU Fallback", "icon": "ERROR"} in layout.labels
    assert {"text": "forced unavailable", "icon": "INFO"} in layout.labels
    assert {"text": "CPU transfer: 120,000 bytes", "icon": "NONE"} in layout.labels
    assert {
        "show_whitewater",
        "whitewater_enabled",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
    }.isdisjoint(layout.props)


def test_realtime_display_panel_reports_live_native_interop_activity(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    sess = scene_bridge.session()
    sess.presentation_mode = "Native GPU"
    sess.presentation_reason = ""
    sess.presentation_bytes = 0
    sess.native_presenter = SimpleNamespace(
        bridge=SimpleNamespace(
            stats={
                "imports": 6,
                "maps": 2,
                "published_frames": 9,
                "published_bytes": 86_400_000,
                "bulk_host_position_bytes": 0,
            }
        )
    )
    panels = _reload_module("fluidSim_warp_blender51.panels")

    class Settings(SimpleNamespace):
        def __getattr__(self, _name):
            return False

    fs = Settings(
        liquid_view_mode="PARTICLES",
        show_surface=False,
        surface_enabled=False,
        show_particles=True,
        native_gpu_interop=True,
        show_whitewater=False,
        whitewater_enabled=False,
    )
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_realtime_display()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert {"text": "Mode: Native GPU", "icon": "CHECKMARK"} in layout.labels
    assert {"text": "GPU publications: 9 frames", "icon": "NONE"} in layout.labels
    assert {
        "text": "Shared-buffer writes: 86,400,000 bytes",
        "icon": "NONE",
    } in layout.labels
    assert {"text": "Persistent imports/maps: 6 / 2", "icon": "NONE"} in layout.labels
    assert {"text": "Host position readback: 0 bytes", "icon": "NONE"} in layout.labels
    assert not any(label["text"].startswith("CPU transfer:") for label in layout.labels)


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


def test_lifecycle_handlers_release_before_load_undo_and_redo(monkeypatch):
    fake_bpy = _install_bpy_stub(monkeypatch)
    handlers = SimpleNamespace(
        frame_change_post=[],
        load_pre=[],
        undo_pre=[],
        redo_pre=[],
    )
    fake_bpy.app = SimpleNamespace(handlers=handlers)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    calls = []
    monkeypatch.setattr(scene_bridge, "release_session", lambda: calls.append("release"))

    scene_bridge.register_handlers()
    assert handlers.frame_change_post == [scene_bridge._on_frame_change]
    assert handlers.load_pre == [scene_bridge._on_lifecycle_pre]
    assert handlers.undo_pre == [scene_bridge._on_lifecycle_pre]
    assert handlers.redo_pre == [scene_bridge._on_lifecycle_pre]
    assert scene_bridge._on_lifecycle_pre._bpy_persistent is True

    scene_bridge._on_lifecycle_pre(None)
    assert calls == ["release"]
    scene_bridge.unregister_handlers()
    assert not any(
        (handlers.frame_change_post, handlers.load_pre, handlers.undo_pre, handlers.redo_pre)
    )


def test_lifecycle_release_clears_session_and_retains_failed_cleanup(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    sess = scene_bridge.session()
    allow_close = [False]

    class Presenter:
        def close(self):
            if not allow_close[0]:
                raise RuntimeError("teardown failed")

    presenter = Presenter()
    sess.solver = object()
    sess.native_presenter = presenter

    sess.release()

    assert not sess.active
    assert sess.native_presenter is None
    assert sess._native_cleanup_pending == [presenter]
    assert "teardown failed" in sess.native_interop_error

    allow_close[0] = True
    sess.release()
    assert sess._native_cleanup_pending == []


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
