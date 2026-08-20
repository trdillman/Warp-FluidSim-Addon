from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "panels.py"
PREFERENCES = ROOT / "preferences.py"


def _block(source: str, class_name: str) -> str:
    start = source.index(f"class {class_name}")
    end = source.find("\nclass ", start + 1)
    return source[start:] if end < 0 else source[start:end]


def test_viewport_contains_only_a_compact_status_launcher():
    source = PANELS.read_text(encoding="utf-8")
    classes = source.split("CLASSES = (", 1)[1].split(")\n\n\ndef register", 1)[0]
    assert "FLUIDSIM_PT_viewport_status" in classes
    for legacy in (
        "FLUIDSIM_PT_simulation",
        "FLUIDSIM_PT_quality",
        "FLUIDSIM_PT_inputs",
        "FLUIDSIM_PT_physics",
        "FLUIDSIM_PT_surface",
        "FLUIDSIM_PT_output",
        "FLUIDSIM_PT_advanced",
        "FLUIDSIM_PT_snippets",
    ):
        assert legacy not in classes


def test_viewport_status_uses_derived_activity_and_lifecycle_actions():
    source = PANELS.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_viewport_status")
    assert "FluidSimView3DPanel" in block.splitlines()[0]
    assert "activity_ui_state" in block
    assert "runtime_status_for_device" in block
    assert 'pending["action"] != "NONE"' in block
    assert 'pending["visible"]' not in block
    for operator_id in (
        "fluidsim.toggle_live",
        "fluidsim.cache_playback_stop",
        "fluidsim.bake_cancel",
        "fluidsim.initialize",
        "fluidsim.release",
        "fluidsim.open_scene_properties",
    ):
        assert operator_id in block
    for forbidden in (
        "voxel_size",
        "particles_per_voxel",
        "pressure_solver",
        "cache_path",
        "whitewater_enabled",
        "surface_tension_enabled",
        "fluidsim_effects",
    ):
        assert forbidden not in block


def test_scene_properties_shortcut_uses_an_existing_properties_area():
    source = PANELS.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_OT_open_scene_properties")
    assert 'bl_idname = "fluidsim.open_scene_properties"' in block
    assert 'area.type == "PROPERTIES"' in block
    assert 'area.spaces.active.context = "SCENE"' in block
    assert "No Properties editor is open" in block


def test_preferences_show_runtime_and_conditional_override_but_hide_verbose():
    source = PREFERENCES.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_AP_preferences")
    assert 'box.label(text="Runtime"' in block
    assert 'box.prop(self, "device")' in block
    assert "runtime_status_for_device" in block
    assert 'os.environ.get("FLUIDSIM_ALLOW_EXTERNAL_WARP") == "1"' in block
    assert 'developer.prop(self, "developer_warp_path")' in block
    assert 'prop(self, "verbose")' not in block


def test_viewport_never_exposes_step_or_release_during_cache_playback_or_bake():
    source = PANELS.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_viewport_status")
    assert 'state["owns_scene"] and activity not in {"CACHE_PLAYBACK", "PRODUCTION_BAKE"}' in block


def test_viewport_step_one_frame_uses_execute_context_not_the_multi_step_dialog():
    source = PANELS.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_viewport_status")
    assert 'row.operator_context = "EXEC_DEFAULT"' in block
