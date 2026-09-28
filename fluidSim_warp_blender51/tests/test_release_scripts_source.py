from __future__ import annotations

import ast
from pathlib import Path

from fluidsim_testpkg import version

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"


def test_runtime_release_package_contains_only_installed_command_shims():
    runtime_files = {
        path.name
        for path in (ROOT / "release").iterdir()
        if path.is_file() and path.suffix == ".py"
    }
    assert runtime_files == {"__init__.py", "run_headless_bake.py"}
    assert (TOOLS_RELEASE / "build_extension.py").is_file()
    assert (TOOLS_RELEASE / "smoke_test_blender.py").is_file()
    assert (TOOLS_RELEASE / "validate_phase5_blender.py").is_file()


def test_blender_smoke_proves_external_warp_cpu_execution_and_lifecycle():
    source = (TOOLS_RELEASE / "smoke_test_blender.py").read_text(encoding="utf-8")
    assert 'EXPECTED_WARP_VERSION = _IDENTITY["CERTIFIED_WARP_VERSION"]' in source
    assert 'EXPECTED_BLENDER_VERSION = tuple(_IDENTITY["CERTIFIED_BLENDER_VERSION"])' in source
    assert 'EXPECTED_FLUIDSIM_VERSION = tuple(_IDENTITY["VERSION"])' in source
    assert "bpy.app.version[:3]" in source
    assert "warp.__file__" in source or "wp.__file__" in source
    assert "BLENDER_USER_EXTENSIONS" in source
    assert "--python-use-system-env" not in source
    assert "@wp.kernel" in source
    assert "wp.launch" in source
    assert "device=\"cpu\"" in source
    assert "cpu_kernel_output" in source
    assert "fluidsim.refresh_runtime_probe" in source
    assert "fluidsim.setup_scene" in source
    assert "active_domain_settings(scene)" in source
    assert "_preset_field_owner(scene)" in source
    assert "source=domain_fs" in source
    assert "field_source=field_owner" in source
    assert source.count("target=domain_fs") >= 2
    assert source.count("field_target=field_owner") >= 2
    for domain_field in (
        "surface_attribute_velocity",
        "surface_attribute_motion",
        "particles_attribute_age",
        "particles_attribute_lifetime",
        "whitewater_attribute_type",
        "surface_tension_enabled",
        "surface_tension_coefficient",
    ):
        assert f"domain_fs.{domain_field}" in source
    assert "BUILTIN:preview_water" in source
    assert "EXPECTED_PRESETS" in source
    assert "bpy.ops.wm.save_as_mainfile" in source
    assert "bpy.ops.wm.open_mainfile" in source
    assert "run_headless_bake.py" in source
    assert source.count("_disable_and_verify(module)") >= 2
    assert "bpy.types.Object" in source
    assert "dir(bpy.app.handlers)" in source
    assert "_assert_no_owned_handlers" in source
    assert "FLUIDSIM_SMOKE_REPORT" in source
    assert "FLUIDSIM_SMOKE_OK" in source
    for development_only in (
        "custom_effect_demo_runtime.py",
        "generate_custom_effect_demos.py",
        "validate_custom_effects_blender.py",
        "validate_custom_effects_warp.py",
    ):
        assert development_only in source


def test_gpu_regression_targets_certified_runtime_and_full_suite():
    source = (TOOLS_RELEASE / "gpu_regression_blender.py").read_text(encoding="utf-8")
    assert "validation.run_all" in source
    assert 'f"{module}.diagnostics.validation"' in source
    assert 'EXPECTED_BLENDER_VERSION = tuple(_IDENTITY["CERTIFIED_BLENDER_VERSION"])' in source
    assert 'EXPECTED_WARP_VERSION = _IDENTITY["CERTIFIED_WARP_VERSION"]' in source
    assert "bpy.app.version[:3]" in source
    assert "expected_validation_count = len(validation.VALIDATION_TESTS)" in source
    assert "EXPECTED_VALIDATION_COUNT = 16" not in source
    assert "FLUIDSIM_EXTERNAL_WARP_ROOT" not in source
    assert "Warp loaded outside Blender's isolated extension wheel directory" in source
    assert '/ ".local"' in source
    assert "BLENDER_USER_EXTENSIONS" in source
    assert "--python-use-system-env" not in source
    assert "probe.module_file" in source
    assert "if not probe.module_file" in source
    assert "frames=10" in source
    assert "FLUIDSIM_GPU_REGRESSION_OK" in source


def _identity_loader(script: Path):
    """Run a release script's version.py loader without importing bpy or Warp."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    loader = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_source_release_identity"
    )
    namespace = {"Path": Path, "__file__": str(script)}
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["_source_release_identity"]


def test_in_blender_release_oracles_read_the_source_identity():
    # A version bump edits version.py only; the installed-package oracles follow.
    for name in (
        "smoke_test_blender.py",
        "validate_phase5_blender.py",
        "gpu_regression_blender.py",
    ):
        identity = _identity_loader(TOOLS_RELEASE / name)()
        assert identity["VERSION"] == version.VERSION, name
        assert identity["VERSION_STRING"] == version.VERSION_STRING, name
        assert identity["CERTIFIED_WARP_VERSION"] == version.CERTIFIED_WARP_VERSION, name
        assert (
            identity["CERTIFIED_BLENDER_VERSION"] == version.CERTIFIED_BLENDER_VERSION
        ), name
        source = (TOOLS_RELEASE / name).read_text(encoding="utf-8")
        assert "_IDENTITY = _source_release_identity()" in source, name
        assert '"1.16.0"' not in source and "(5, 1, 0)" not in source, name
