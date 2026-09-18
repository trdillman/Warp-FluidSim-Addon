from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "blender" / "effect_ui.py"


def test_custom_effects_ui_is_a_direct_artist_workflow():
    source = UI.read_text(encoding="utf-8")
    assert "class FLUIDSIM_UL_effect_stack" in source
    assert 'bl_idname = "FLUIDSIM_PT_custom_effects"' in source
    assert "Custom Effects" in source
    assert "Test Draft" in source
    assert "Compile & Apply" in source
    assert "Revert Draft" in source
    assert "Preview One Frame" in source
    assert "Review & Trust" in source
    assert "Changes will be prepared when you preview" in source
    assert "Production Bake blocked" in source


def test_effect_stack_exposes_expected_artist_actions():
    source = UI.read_text(encoding="utf-8")
    for operator_id in (
        "fluidsim.effect_add",
        "fluidsim.effect_remove",
        "fluidsim.effect_duplicate",
        "fluidsim.effect_move",
        "fluidsim.effect_toggle_mute",
        "fluidsim.effect_toggle_solo",
        "fluidsim.effect_test_draft",
        "fluidsim.effect_apply_stack",
        "fluidsim.effect_revert_draft",
        "fluidsim.effect_preview_one_frame",
        "fluidsim.effect_create_controller",
        "fluidsim.effect_select_controller",
        "fluidsim.effect_trust",
        "fluidsim.effect_untrust",
        "fluidsim.effect_import_package",
        "fluidsim.effect_compare_update",
        "fluidsim.effect_export_package",
        "fluidsim.effect_rename_installed",
        "fluidsim.effect_reveal_library",
        "fluidsim.effect_delete_installed",
        "fluidsim.effect_refresh_status",
    ):
        assert operator_id in source
    assert "Select an effect first" in source


def test_artist_mode_hides_raw_kernel_plumbing():
    source = UI.read_text(encoding="utf-8")
    artist_draw = source.split("def _draw_artist_parameters", 1)[1].split(
        "def _draw_status", 1
    )[0]
    for raw_property in ("domain", "stage", "cadence", "fields", "safe_mode"):
        assert f'prop(.*"{raw_property}"' not in artist_draw
    assert "_draw_parameter" in artist_draw
    assert "object_value" in source


def test_legacy_public_panels_are_removed_and_v2_ui_registered():
    panels = (ROOT / "panels.py").read_text(encoding="utf-8")
    root = (ROOT / "__init__.py").read_text(encoding="utf-8")
    classes_block = panels.split("CLASSES = (", 1)[1].split(")\n\n\ndef register", 1)[0]
    assert "FLUIDSIM_PT_snippets" not in classes_block
    assert "FLUIDSIM_PT_snippet_editor" not in classes_block
    assert "FLUIDSIM_PT_snippet_shortcuts" not in classes_block
    assert "effect_ui.register" in root
    assert "effect_ui.unregister" in root


def test_panel_draw_path_never_mutates_the_active_effect_index():
    source = UI.read_text(encoding="utf-8")
    active_helper = source.split("def _active_effect", 1)[1].split(
        "def _normalize_orders", 1
    )[0]
    assert "stack.active_index =" not in active_helper


def test_package_import_reads_the_validated_installed_archive_not_report_metadata():
    source = UI.read_text(encoding="utf-8")
    assert "read_effect_package(report.path)" in source
    assert "report.definition" not in source


def test_only_non_mutating_legacy_apply_forward_is_registered():
    source = (ROOT / "operators.py").read_text(encoding="utf-8")
    classes_block = source.split("CLASSES = (", 1)[1].split(")\n\n\ndef register", 1)[0]
    assert classes_block.count("FLUIDSIM_OT_snippet_") == 1
    assert "FLUIDSIM_OT_snippet_apply" in classes_block


def test_library_failures_and_live_draft_drift_are_visible_in_the_panel():
    source = UI.read_text(encoding="utf-8")
    assert "library_diagnostics" in source
    assert "Installed package problem" in source
    assert "execution_stack_fingerprint" in source
    assert 'live_signature != stack.applied_signature' in source
    assert 'Changes will be prepared when you preview' in source
    assert 'Effect draft could not be inspected' in source
    assert "effect_refresh_status" in source


def test_keep_both_import_changes_effect_identity_instead_of_only_the_filename():
    source = (ROOT / "blender" / "effect_library.py").read_text(encoding="utf-8")
    assert "unique_copy_definition" in source
    assert "merge_effect_definitions" in source
    assert "cannot replace a built-in effect identity" in source


def test_effect_library_ui_exposes_update_comparison_and_installed_package_rename():
    source = UI.read_text(encoding="utf-8")
    assert 'bl_idname = "fluidsim.effect_compare_update"' in source
    assert 'bl_idname = "fluidsim.effect_rename_installed"' in source
    assert "compare_package_update" in source
    assert "rename_installed_package" in source
    assert "Compare Update" in source
    assert "Rename Package" in source
