from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_blender_smoke_proves_external_warp_cpu_execution_and_lifecycle():
    source = (ROOT / "release" / "smoke_test_blender.py").read_text(encoding="utf-8")
    assert 'EXPECTED_WARP_VERSION = "1.16.0"' in source
    assert "EXPECTED_BLENDER_VERSION = (5, 2, 0)" in source
    assert "EXPECTED_FLUIDSIM_VERSION = (1, 0, 252)" in source
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
    source = (ROOT / "release" / "gpu_regression_blender.py").read_text(encoding="utf-8")
    assert "validation.run_all" in source
    assert "EXPECTED_BLENDER_VERSION = (5, 2, 0)" in source
    assert 'EXPECTED_WARP_VERSION = "1.16.0"' in source
    assert "bpy.app.version[:3]" in source
    assert "expected_validation_count = len(validation.VALIDATION_TESTS)" in source
    assert "EXPECTED_VALIDATION_COUNT = 16" not in source
    assert "BLENDER_USER_EXTENSIONS" in source
    assert "--python-use-system-env" not in source
    assert "probe.module_file" in source
    assert "if not probe.module_file" in source
    assert "frames=10" in source
    assert "FLUIDSIM_GPU_REGRESSION_OK" in source
