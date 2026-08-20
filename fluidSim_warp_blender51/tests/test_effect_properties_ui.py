from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIST_UI = ROOT / "blender" / "effect_ui.py"
WARP_UI = ROOT / "blender" / "effect_workspace.py"


def _block(source: str, class_name: str) -> str:
    start = source.index(f"class {class_name}")
    end = source.find("\nclass ", start + 1)
    return source[start:] if end < 0 else source[start:end]


def test_custom_effects_is_a_closed_scene_properties_root_at_order_50():
    source = ARTIST_UI.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_custom_effects")
    assert 'bl_space_type = "PROPERTIES"' in block
    assert 'bl_region_type = "WINDOW"' in block
    assert 'bl_context = "scene"' in block
    assert "bl_order = 50" in block
    assert 'bl_options = {"DEFAULT_CLOSED"}' in block


def test_artist_workflow_is_always_primary_and_raw_mode_toggle_is_not_drawn():
    source = ARTIST_UI.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_custom_effects")
    selected = source.split("def _draw_selected_effect", 1)[1].split(
        "def _draw_effect_workflow", 1
    )[0]
    assert "_draw_artist_parameters(box, item, boxed=False)" in selected
    assert 'prop(stack, "artist_mode"' not in block
    assert "if stack.artist_mode" not in block
    for section in (
        "_draw_effect_stack",
        "_draw_selected_effect",
        "_draw_effect_workflow",
        "_draw_status",
    ):
        assert section in block


def test_effect_package_management_is_a_separate_closed_child_panel():
    source = ARTIST_UI.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_effect_packages")
    assert 'bl_parent_id = "FLUIDSIM_PT_custom_effects"' in block
    assert 'bl_space_type = "PROPERTIES"' in block
    assert 'bl_context = "scene"' in block
    assert 'bl_options = {"DEFAULT_CLOSED"}' in block
    for operator_id in (
        "fluidsim.effect_import_package",
        "fluidsim.effect_compare_update",
        "fluidsim.effect_export_package",
        "fluidsim.effect_rename_installed",
        "fluidsim.effect_delete_installed",
        "fluidsim.effect_reveal_library",
    ):
        assert operator_id in block


def test_warp_code_is_an_advanced_scene_properties_child_without_artist_mode_gate():
    source = WARP_UI.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_effect_warp_code")
    assert 'bl_parent_id = "FLUIDSIM_PT_custom_effects"' in block
    assert 'bl_space_type = "PROPERTIES"' in block
    assert 'bl_region_type = "WINDOW"' in block
    assert 'bl_context = "scene"' in block
    assert 'bl_options = {"DEFAULT_CLOSED"}' in block
    poll = block.split("def poll", 1)[1].split("def draw", 1)[0]
    assert "bool(stack.items)" in poll
    assert "bool(item.kernels)" in poll
    assert "artist_mode" not in poll


def test_effect_package_child_is_registered_after_the_root_panel():
    source = ARTIST_UI.read_text(encoding="utf-8")
    classes = source.split("CLASSES = (", 1)[1].split(")\n\n\ndef register", 1)[0]
    assert classes.index("FLUIDSIM_PT_custom_effects") < classes.index(
        "FLUIDSIM_PT_effect_packages"
    )


def test_selected_effect_avoids_redundant_nested_artist_boxes():
    source = ARTIST_UI.read_text(encoding="utf-8")
    artist = source.split("def _draw_artist_parameters", 1)[1].split("def _draw_status", 1)[0]
    assert "boxed=True" in artist
    assert "box = layout.box() if boxed else layout" in artist


def test_package_maintenance_actions_only_appear_for_an_installed_package():
    source = ARTIST_UI.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_effect_packages")
    assert "installed_package_for_effect" in block
    assert "Scene effect is embedded; no installed library package" in block
