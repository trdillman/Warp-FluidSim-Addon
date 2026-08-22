from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fluidsim_testpkg.blender import setup_wizard
from fluidsim_testpkg.presets import library

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRESETS = {
    "preview_water",
    "production_water",
    "splash_whitewater",
    "honey",
    "thick_liquid",
    "thin_liquid",
    "water_surface_tension",
    "stylized_high_tension",
}
EXPECTED_EXAMPLES = {
    "01_live_pour.blend",
    "02_moving_collider_splash.blend",
    "03_viscous_honey.blend",
    "04_whitewater_bake_resume.blend",
}


def test_builtin_preset_inventory_is_complete_and_artist_described():
    records = library.built_in_presets()
    assert {record.preset.id for record in records} == EXPECTED_PRESETS
    for record in records:
        preset = record.preset
        assert preset.description.strip()
        assert preset.expected_scale.strip()
        assert preset.performance_cost.strip()
        assert preset.requires_reset is True
        assert Path(record.path).name == f"{preset.id}.json"


def test_builtin_preset_json_is_canonical_and_versioned():
    for path in sorted((ROOT / "resources" / "presets").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["id"] == path.stem
        assert path.read_text(encoding="utf-8").endswith("\n")


def test_phase2_examples_are_generated_nonempty_blender_files():
    directory = ROOT / "resources" / "examples"
    files = {path.name for path in directory.glob("*.blend")}
    assert EXPECTED_EXAMPLES <= files
    for path in directory.glob("*.blend"):
        assert path.stat().st_size > 50_000
        magic = path.read_bytes()[:7]
        assert magic == b"BLENDER" or magic[:4] == b"(\xb5/\xfd"
    readme = (directory / "README.md").read_text(encoding="utf-8")
    for filename in EXPECTED_EXAMPLES:
        assert filename in readme
    assert "release/generate_examples.py" in readme
    assert "does not require CUDA" in readme


def test_example_generator_uses_guided_setup_and_has_validator():
    generator = (ROOT / "release" / "generate_examples.py").read_text(encoding="utf-8")
    validator = (ROOT / "release" / "validate_examples_blender.py").read_text(
        encoding="utf-8"
    )
    assert "bpy.ops.fluidsim.setup_scene" in generator
    assert "QUICK_LIQUID" in generator
    assert "QUICK_VISCOUS" in generator
    assert "ADD_COLLIDER" in generator
    assert "bpy.ops.wm.save_as_mainfile" in generator
    assert "bpy.ops.wm.open_mainfile" in validator
    assert "FLUIDSIM_EXAMPLES_VALIDATED" in validator


def test_selected_mesh_filter_rejects_non_mesh_objects():
    def obj(name: str, object_type: str, role: str):
        return SimpleNamespace(
            name=name,
            type=object_type,
            fluidsim=SimpleNamespace(role=role),
        )

    mesh = obj("Mesh", "MESH", "NONE")
    source = obj("Source", "MESH", "SOURCE")
    light = obj("Light", "LIGHT", "NONE")
    unsupported = obj("Force", "MESH", "FORCE")
    context = SimpleNamespace(selected_objects=(mesh, source, light, unsupported))
    assert setup_wizard._selected_meshes(context) == (mesh, source)


def test_installed_headless_entry_point_is_distinct_from_release_tooling():
    runner = (ROOT / "release" / "run_headless_bake.py").read_text(encoding="utf-8")
    manifest = (ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
    assert "os._exit" in runner
    assert '"release/run_headless_bake.py"' not in manifest
    for excluded in (
        "release/build_extension.py",
        "release/generate_examples.py",
        "release/generate_geometry_nodes.py",
        "release/smoke_test_blender.py",
        "release/validate_examples_blender.py",
        "release/validate_phase3_blender.py",
        "release/validate_phase3_warp.py",
        "release/validate_phase4_blender.py",
        "release/validate_phase4_warp.py",
    ):
        assert f'"{excluded}"' in manifest

def test_guided_workflow_exposes_three_action_live_preview_path():
    scene_ui = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    launcher = (ROOT / "panels.py").read_text(encoding="utf-8")
    operator = (ROOT / "operators.py").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert "Guided Setup…" in scene_ui
    assert "Start Live Preview" in scene_ui
    assert "Start Live Preview" in launcher
    assert "bridge.initialize_session(scene)" in operator
    assert "3. Press Play." in guide

