from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "release" / "validate_properties_ui_blender.py"
SMOKE = ROOT / "release" / "smoke_test_blender.py"


def test_properties_ui_blender_validator_covers_native_surfaces_and_lifecycle():
    source = VALIDATOR.read_text(encoding="utf-8")
    for marker in (
        "EXPECTED_SCENE_ROOTS",
        "EXPECTED_OBJECT_ROOTS",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_effect_packages",
        "FLUIDSIM_PT_effect_warp_code",
        "FLUIDSIM_PT_viewport_status",
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
    ):
        assert panel_id in source


def test_properties_validator_remains_release_only_and_is_not_packaged():
    manifest = (ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
    assert '"release/validate_properties_ui_blender.py"' in manifest


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
        "role_matrix",
    ):
        assert marker in source
