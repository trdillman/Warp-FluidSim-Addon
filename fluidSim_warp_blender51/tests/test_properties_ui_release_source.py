from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"
VALIDATOR = TOOLS_RELEASE / "validate_properties_ui_blender.py"
SMOKE = TOOLS_RELEASE / "smoke_test_blender.py"


def test_properties_ui_blender_validator_covers_native_surfaces_and_lifecycle():
    source = VALIDATOR.read_text(encoding="utf-8")
    for marker in (
        "EXPECTED_SCENE_ROOTS",
        "EXPECTED_OBJECT_ROOTS",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_effect_packages",
        "FLUIDSIM_PT_effect_warp_code",
        "FLUIDSIM_PT_viewport_status",
        "FLUIDSIM_VIEW3D_PT_full_settings",
        "FLUIDSIM_PT_physics_root",
        "FLUIDSIM_OT_use_active_as_domain",
        "FLUIDSIM_OT_remove_selected_output",
        "FLUIDSIM_OT_open_scene_properties",
        "RecordingLayout",
        "save_as_mainfile",
        "open_mainfile",
        "unregister()",
        "register()",
        "_atomic_json",
    ):
        assert marker in source


def test_installed_smoke_requires_the_new_ui_modules_and_panel_inventory():
    source = SMOKE.read_text(encoding="utf-8")
    for path in (
        'extension_root / "blender" / "ui_common.py"',
        'extension_root / "blender" / "ui_scene.py"',
        'extension_root / "blender" / "ui_object.py"',
        'extension_root / "blender" / "ui_physics.py"',
        'extension_root / "blender" / "ui_view3d.py"',
    ):
        assert path in source
    for panel_id in (
        "FLUIDSIM_PT_scene_control",
        "FLUIDSIM_PT_scene_domain",
        "FLUIDSIM_PT_object_role",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_effect_packages",
        "FLUIDSIM_PT_effect_warp_code",
        "FLUIDSIM_PT_viewport_status",
        "FLUIDSIM_VIEW3D_PT_full_settings",
        "FLUIDSIM_VIEW3D_PT_scene_control",
        "FLUIDSIM_PT_physics_root",
        "FLUIDSIM_PT_physics_surface_output",
    ):
        assert panel_id in source


def test_properties_validator_is_tools_owned_and_not_packaged():
    manifest = (ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
    assert VALIDATOR.is_file()
    assert "validate_properties_ui_blender.py" not in manifest


def test_properties_validator_exercises_all_authored_panels_and_object_role_matrix():
    source = VALIDATOR.read_text(encoding="utf-8")
    for marker in (
        "ALL_AUTHORED_SCENE_PANELS",
        "OBJECT_ROLE_CASES",
        "SOURCE",
        "SINK",
        "COLLIDER",
        "FORCE",
        "FIELD",
        "VOLUME_GRID",
        "all_authored_draws",
        "viewport_mirrors",
        "physics_context",
        "role_matrix",
    ):
        assert marker in source
