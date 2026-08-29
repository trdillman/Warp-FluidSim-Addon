from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"
GENERATOR = TOOLS_RELEASE / "generate_custom_effect_demos.py"
VALIDATOR = TOOLS_RELEASE / "validate_custom_effects_blender.py"
RUNTIME = TOOLS_RELEASE / "custom_effect_demo_runtime.py"
EXAMPLES = ROOT / "resources" / "examples"

DEMO_NAMES = (
    "05_custom_effects_vortex.blend",
    "06_custom_effects_stack.blend",
    "07_warp_code_authoring.blend",
)


def test_demo_generator_validator_and_runtime_are_present():
    assert GENERATOR.is_file()
    assert VALIDATOR.is_file()
    assert RUNTIME.is_file()
    generator = GENERATOR.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")
    runtime = RUNTIME.read_text(encoding="utf-8")
    for name in DEMO_NAMES:
        assert name in generator
        assert name in validator
    for token in (
        "EffectRuntimeManager",
        "build_candidate_effects",
        "tangential_speed_mean",
        "kinetic_energy_ratio",
        "stack_order_sensitivity",
        "deterministic_rerun_equal",
        "canaries_intact",
    ):
        assert token in runtime
    assert "expected-warp-root" in validator
    assert "F12" in validator or "render" in validator.lower()


def test_generated_demo_assets_and_documentation_are_packaged():
    for name in DEMO_NAMES:
        path = EXAMPLES / name
        assert path.is_file(), name
        assert path.stat().st_size > 1024
    readme = EXAMPLES / "custom_effects" / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    for phrase in (
        "Vortex Force",
        "Stack Order",
        "Warp Code",
        "supplied Warp 1.16.0",
        "CPU validation",
        "CUDA",
    ):
        assert phrase in text


def test_demo_assets_do_not_bundle_runtime_payloads():
    forbidden = {".whl", ".so", ".dll", ".exe", ".tar", ".xz"}
    root = EXAMPLES / "custom_effects"
    if root.exists():
        assert not [path for path in root.rglob("*") if path.suffix.lower() in forbidden]


def test_demo_scene_creation_allocates_a_world_for_factory_empty_runs():
    source = GENERATOR.read_text(encoding="utf-8")
    assert "bpy.data.worlds.new" in source
    assert "scene.world is None" in source


def test_demo_wire_geometry_does_not_attach_unsupported_modifier_id_properties():
    source = GENERATOR.read_text(encoding="utf-8")
    assert 'bevel["fluidsim_owned"]' not in source
    assert 'obj["fluidsim_owned"] = True' in source


def test_generator_analyzes_the_durable_saved_png_not_headless_render_result():
    source = GENERATOR.read_text(encoding="utf-8")
    render_block = source.split("def _render", 1)[1].split("def _set_parameter", 1)[0]
    assert "bpy.data.images.load" in render_block
    assert 'bpy.data.images.get("Render Result")' not in render_block


def test_blender_validator_treats_reported_operator_rejection_as_expected_and_preserves_line():
    source = VALIDATOR.read_text(encoding="utf-8")
    workflow = source.split("def _workflow_validation", 1)[1].split("def main", 1)[0]
    assert "except RuntimeError" in workflow
    assert "expected_rejection" in workflow
    assert "mapped_error_line =" in workflow
    return_block = workflow.split("return {", 1)[1]
    assert '"source_mapped_error_line": mapped_error_line' in return_block
    assert 'bad_pg.kernels[0].error_line' not in return_block


def test_standard_example_release_tools_include_custom_effect_demo_inventory():
    generator = (TOOLS_RELEASE / "generate_examples.py").read_text(encoding="utf-8")
    validator = (TOOLS_RELEASE / "validate_examples_blender.py").read_text(encoding="utf-8")
    for name in DEMO_NAMES:
        assert name in generator
        assert name in validator
    assert "CUSTOM_EFFECT_EXAMPLE_NAMES" in generator
    assert "CUSTOM_EFFECT_EXPECTED" in validator
    assert "effect_identity" in validator


def test_installed_smoke_proves_custom_effects_workflow_and_package_assets():
    smoke = (TOOLS_RELEASE / "smoke_test_blender.py").read_text(encoding="utf-8")
    for token in (
        "def _assert_custom_effects",
        "FLUIDSIM_PT_custom_effects",
        "FLUIDSIM_PT_effect_warp_code",
        'extension_root / "sim" / "effects" / "schema.py"',
        'extension_root / "resources" / "effects" / "vortex_force.json"',
        "bpy.ops.fluidsim.effect_test_draft(device=\"cpu\")",
        "bpy.ops.fluidsim.effect_apply_stack(device=\"cpu\")",
    ):
        assert token in smoke


def test_blender_validator_exercises_package_update_comparison_and_safe_rename():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "def _package_library_validation" in source
    assert "effect_compare_update" in source
    assert "compare_package_update" in source
    assert "rename_installed_package" in source
    assert "bpy.ops.fluidsim.effect_rename_installed" in source
    assert '"update_comparison": "NEWER"' in source
    assert '"package_rename": True' in source
