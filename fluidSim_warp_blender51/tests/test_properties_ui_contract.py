from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from fluidSim_warp_blender51 import registration

ROOT = Path(__file__).resolve().parents[1]


def _load_playback_for_contract(monkeypatch):
    fake_bpy = ModuleType("bpy")
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(
            persistent=lambda function: function,
            frame_change_post=[],
            animation_playback_pre=[],
        )
    )
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    name = "fluidsim_testpkg.blender.playback_contract"
    spec = importlib.util.spec_from_file_location(name, ROOT / "blender" / "playback.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_activity_snapshot_has_one_authoritative_public_mode(monkeypatch):
    playback = _load_playback_for_contract(monkeypatch)

    def scene(**values):
        defaults = {
            "live": False,
            "cache_playback_enabled": False,
            "bake_status": "IDLE",
        }
        defaults.update(values)
        return SimpleNamespace(fluidsim=SimpleNamespace(**defaults))

    assert playback.activity_snapshot(scene(), owns_scene=False)["id"] == "IDLE"
    assert playback.activity_snapshot(scene(live=True), owns_scene=True)["id"] == "LIVE_PREVIEW"
    assert (
        playback.activity_snapshot(scene(live=True, cache_playback_enabled=True), owns_scene=False)[
            "id"
        ]
        == "CACHE_PLAYBACK"
    )
    assert (
        playback.activity_snapshot(
            scene(live=True, cache_playback_enabled=True, bake_status="RUNNING"),
            owns_scene=True,
        )["id"]
        == "PRODUCTION_BAKE"
    )


def test_common_panel_bases_target_native_blender_surfaces():
    source = (ROOT / "blender" / "ui_common.py").read_text(encoding="utf-8")
    assert "class FluidSimScenePanel" in source
    assert 'bl_space_type = "PROPERTIES"' in source
    assert 'bl_region_type = "WINDOW"' in source
    assert 'bl_context = "scene"' in source
    assert "class FluidSimObjectPanel" in source
    assert 'bl_context = "object"' in source
    assert "class FluidSimView3DPanel" in source
    assert 'bl_space_type = "VIEW_3D"' in source
    assert 'bl_category = "FluidSim"' in source


def test_common_ui_uses_lifecycle_owners_instead_of_raw_transport_state():
    source = (ROOT / "blender" / "ui_common.py").read_text(encoding="utf-8")
    assert "activity_snapshot" in source
    assert "pending_presentation" in source
    assert "owns_scene" in source
    assert "def active_object_role" in source
    assert "def canonical_region_kind" in source


def test_public_ui_never_draws_transport_or_visibility_aliases():
    public_sources = [
        ROOT / "panels.py",
        ROOT / "blender" / "ui_scene.py",
        ROOT / "blender" / "ui_object.py",
        ROOT / "blender" / "effect_ui.py",
        ROOT / "blender" / "effect_workspace.py",
        ROOT / "preferences.py",
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in public_sources if path.exists()
    )
    for forbidden in (
        'prop(fs, "live"',
        'prop(fs, "cache_playback_enabled"',
        'prop(fs, "show_surface"',
        'prop(fs, "show_particles"',
        'prop(self, "verbose"',
    ):
        assert forbidden not in combined


def _class_block(source: str, class_name: str) -> str:
    marker = f"class {class_name}"
    start = source.index(marker)
    next_class = source.find("\nclass ", start + len(marker))
    return source[start:] if next_class < 0 else source[start:next_class]


def _def_block(source: str, function_name: str) -> str:
    """Source of one top-level function, up to the next top-level def/class."""
    marker = f"def {function_name}("
    start = source.index(marker)
    ends = [
        index
        for index in (
            source.find("\ndef ", start + len(marker)),
            source.find("\nclass ", start + len(marker)),
        )
        if index >= 0
    ]
    return source[start : min(ends)] if ends else source[start:]


def test_scene_properties_root_order_and_context_are_stable():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    roots = (
        ("FLUIDSIM_PT_scene_lifecycle", 1, True),
        ("FLUIDSIM_PT_scene_control", 2, False),
        ("FLUIDSIM_PT_scene_setup", 10, False),
        ("FLUIDSIM_PT_scene_domain", 20, False),
        ("FLUIDSIM_PT_scene_height", 30, True),
        ("FLUIDSIM_PT_scene_simulation", 40, True),
        ("FLUIDSIM_PT_scene_surface", 60, True),
        ("FLUIDSIM_PT_scene_whitewater", 70, True),
        ("FLUIDSIM_PT_scene_outputs", 80, False),
        ("FLUIDSIM_PT_scene_diagnostics", 100, True),
    )
    for class_name, order, closed in roots:
        block = _class_block(source, class_name)
        assert f"bl_order = {order}" in block
        # Every Properties panel starts expanded so all settings are visible.
        assert "DEFAULT_CLOSED" not in block
        assert "FluidSimScenePanel" in block.splitlines()[0]


def test_scene_properties_children_follow_the_field_ownership_plan():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    parents = {
        "FLUIDSIM_PT_scene_pending": "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_scene_live_timeline": "FLUIDSIM_PT_scene_lifecycle",
        "FLUIDSIM_PT_scene_presets": "FLUIDSIM_PT_scene_setup",
        "FLUIDSIM_PT_scene_capacity": "FLUIDSIM_PT_scene_domain",
        "FLUIDSIM_PT_sim_time": "FLUIDSIM_PT_scene_simulation",
        "FLUIDSIM_PT_sim_gravity": "FLUIDSIM_PT_scene_simulation",
        "FLUIDSIM_PT_sim_pressure": "FLUIDSIM_PT_scene_simulation",
        "FLUIDSIM_PT_sim_viscosity": "FLUIDSIM_PT_scene_simulation",
        "FLUIDSIM_PT_sim_surface_tension": "FLUIDSIM_PT_scene_simulation",
        "FLUIDSIM_PT_surface_resolution": "FLUIDSIM_PT_scene_surface",
        "FLUIDSIM_PT_whitewater_classification": "FLUIDSIM_PT_scene_whitewater",
        "FLUIDSIM_PT_output_display": "FLUIDSIM_PT_scene_outputs",
        "FLUIDSIM_PT_output_materials": "FLUIDSIM_PT_scene_outputs",
        "FLUIDSIM_PT_output_attributes": "FLUIDSIM_PT_scene_outputs",
        "FLUIDSIM_PT_diagnostics_profiling": "FLUIDSIM_PT_scene_diagnostics",
        "FLUIDSIM_PT_diagnostics_benchmark": "FLUIDSIM_PT_scene_diagnostics",
        "FLUIDSIM_PT_diagnostics_report": "FLUIDSIM_PT_scene_diagnostics",
        # D4: runtime telemetry consolidated under Diagnostics (was scene_control).
        "FLUIDSIM_PT_scene_runtime_stats": "FLUIDSIM_PT_scene_diagnostics",
    }
    for child, parent in parents.items():
        block = _class_block(source, child)
        assert f'bl_parent_id = "{parent}"' in block


def test_height_fill_does_not_duplicate_domain_sampling():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    block = _class_block(source, "FLUIDSIM_PT_scene_height")
    assert "height_enabled" in block
    assert "height_emit_particles" in block
    assert "particles_per_voxel" not in block
    assert "inherited sampling" not in block.lower()


def test_phase4_and_phase5_controls_have_explicit_scene_owners():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    # D4: the surface-tension panel is edit-only; capillary-stability readbacks
    # (stat_capillary_dt / stat_substep_limiter) moved to the Diagnostics
    # telemetry panel, so they are no longer owned controls here.
    surface_tension = _class_block(source, "FLUIDSIM_PT_sim_surface_tension")
    for name in (
        "surface_tension_enabled",
        "surface_tension_coefficient",
        "surface_tension_density",
        "surface_tension_band_voxels",
        "surface_tension_safety",
        "surface_tension_smoothing_iterations",
    ):
        assert name in surface_tension
    classification = _class_block(source, "FLUIDSIM_PT_whitewater_classification")
    for name in ("ww_foam_band", "ww_spray_min_speed", "ww_type_hysteresis"):
        assert name in classification


def test_cache_playback_uses_explicit_start_and_stop_actions():
    source = (ROOT / "operator_domains" / "session.py").read_text(encoding="utf-8")
    assert 'bl_idname = "fluidsim.cache_playback_start"' in source
    assert 'bl_idname = "fluidsim.cache_playback_stop"' in source
    start = _class_block(source, "FLUIDSIM_OT_cache_playback_start")
    stop = _class_block(source, "FLUIDSIM_OT_cache_playback_stop")
    assert "cache_playback_enabled = True" in start
    assert "live = True" in start
    assert "live = False" in stop
    assert "cache_playback_enabled = False" in stop


def test_stop_and_release_actions_do_not_depend_on_runtime_readiness():
    source = (ROOT / "operator_domains" / "session.py").read_text(encoding="utf-8")
    release = _class_block(source, "FLUIDSIM_OT_release")
    toggle = _class_block(source, "FLUIDSIM_OT_toggle_live")
    assert "_FLUIDSIM_OT_session_base" not in release.splitlines()[0]
    assert "if not sess.active:" in release
    assert "_runtime_poll" not in release
    assert 'if getattr(fs, "live", False):' in toggle
    assert "return True" in toggle.split("def poll", 1)[1].split("def execute", 1)[0]


def test_root_registration_includes_scene_and_object_properties_modules():
    registered = {
        step.module for step in registration.STEPS if step.register and step.unregister
    }
    assert {"blender.ui_scene", "blender.ui_object"} <= registered


def test_transport_exposes_complete_state_dependent_artist_actions():
    # Transport has one drawing (``draw_transport``), shared by the pinned launcher
    # strip and the header popover. The Scene tab no longer repeats it.
    source = (ROOT / "panels.py").read_text(encoding="utf-8")
    assert "draw_transport(layout, context, state, runtime_ready)" in _class_block(
        source, "FLUIDSIM_PT_viewport_status"
    )
    block = source.split("def draw_transport", 1)[1].split("def draw_pending_apply", 1)[0]
    assert 'activity == "PRODUCTION_BAKE"' in block
    assert 'operator("fluidsim.preview_pause", text="Pause"' in block
    assert 'operator("screen.animation_play", text="Play"' in block
    # Retired: the Preview/Resume gate that had to re-arm playback.
    assert '"fluidsim.preview_start"' not in block
    assert '"fluidsim.cache_preview_start"' not in block
    assert '"fluidsim.restart_simulation"' in block
    assert '"fluidsim.bake_cancel"' in block
    assert 'text="Cancel Production Bake"' in block
    scene_source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    assert "FLUIDSIM_PT_scene_transport" not in scene_source


def test_step_one_and_step_many_have_distinct_execute_and_dialog_paths():
    # Manual stepping stays in the Lifecycle panel in every mode.
    scene_source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    block = _class_block(scene_source, "FLUIDSIM_PT_scene_lifecycle")
    assert "DEFAULT_CLOSED" not in block
    assert "_advanced_visible" not in block
    assert 'step_once.operator_context = "EXEC_DEFAULT"' in block
    assert 'step_many.operator_context = "INVOKE_DEFAULT"' in block
    operator_source = (ROOT / "operator_domains" / "session.py").read_text(encoding="utf-8")
    step_operator = _class_block(operator_source, "FLUIDSIM_OT_step")
    assert "invoke_props_dialog" in step_operator


def test_cache_management_requires_a_resolved_existing_directory():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    helper = source.split("def _cache_directory", 1)[1].split("def _draw_capacity_plan", 1)[0]
    assert "-> Path | None" in helper
    assert "if not raw:" in helper
    assert "return None" in helper
    block = _def_block(source, "cache_manage_available")
    assert "path is not None and is_managed_cache_directory(path)" in block


def test_production_bake_panel_presents_one_cache_aware_writer_action():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    block = _def_block(source, "draw_cache_bake")
    assert "production_bake_action" in block
    assert 'presentation.action == "RESUME"' in block
    assert 'presentation.action == "RESTART"' in block
    assert 'text="Resume Verified Bake"' in block
    assert '"Restart Production Bake…"' in block
    assert '"Start Production Bake"' in block
    assert block.count('"fluidsim.bake_resume"') == 1
    assert block.count('"fluidsim.bake_start"') == 1


def test_liquid_view_conditions_use_registered_combined_enum_identifier():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    assert '"SURFACE_PARTICLES"' in source
    for expression in (
        '{"SURFACE", "BOTH"}',
        '{"PARTICLES", "BOTH"}',
    ):
        assert expression not in source


def test_height_fill_error_is_owned_once_by_session_status():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    control = _class_block(source, "FLUIDSIM_PT_scene_control")
    height = _class_block(source, "FLUIDSIM_PT_scene_height")
    assert "height_error" in control
    assert "height_error" not in height


def test_manage_cache_panel_requires_a_recognized_fluidsim_cache_directory():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    block = _def_block(source, "cache_manage_available")
    assert "is_managed_cache_directory" in block
    stages = (ROOT / "blender" / "ui_stages.py").read_text(encoding="utf-8")
    shelf = _class_block(stages, "FLUIDSIM_STAGE_PT_bake_manage")
    assert "cache_manage_available(context)" in shelf


def test_live_timeline_recovery_is_nested_under_lifecycle_in_every_mode():
    source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    block = _class_block(source, "FLUIDSIM_PT_scene_live_timeline")
    assert 'bl_parent_id = "FLUIDSIM_PT_scene_lifecycle"' in block
    assert "_advanced_visible" not in block
    assert "max_frame_catchup" in block

