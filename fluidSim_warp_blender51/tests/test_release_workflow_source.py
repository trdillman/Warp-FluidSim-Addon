from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _workflow() -> str:
    return (ROOT / ".github" / "workflows" / "release-checks.yml").read_text(
        encoding="utf-8"
    )


def test_release_workflow_has_static_dual_package_and_gpu_gates():
    source = _workflow()
    assert "static:" in source
    assert "blender-package:" in source
    assert "gpu-regression:" in source
    assert "ubuntu-latest" in source
    assert "self-hosted" in source and "nvidia" in source
    assert "BLENDER_5_1_EXECUTABLE" in source
    assert "WARP_1_16_WHEEL_SOURCE" in source
    assert "release/fetch_warp_wheels.py" in source
    assert "--source" in source
    assert "release/build_extension.py" in source
    assert "release/gpu_regression_blender.py" in source
    assert "windows-latest" not in source
    assert "windows_x64" in source


def test_workflow_uploads_both_platform_packages_checksums_and_reports():
    source = _workflow()
    assert "SHA256SUMS.txt" in source
    assert "fluidsim_warp-1.0.255-linux_x64.zip" in source
    assert "fluidsim_warp-1.0.255-windows_x64.zip" in source
    assert "smoke-test-report.json" in source
    assert "gpu-regression-report.json" in source
    assert "custom-effects-warp-cpu-validation.json" in source
    assert "custom-effects-blender-validation.json" in source


def test_installed_release_oracles_match_the_1_0_255_identity():
    for relative in (
        "release/smoke_test_blender.py",
        "release/validate_phase5_blender.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "1.0.252" not in source
        assert "1.0.255" in source or "(1, 0, 255)" in source


def test_workflow_uses_current_actions_and_pinned_dev_tools():
    source = _workflow()
    assert "actions/checkout@v6" in source
    assert "actions/setup-python@v6" in source
    assert "actions/upload-artifact@v7" in source
    requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "pytest==9.1.1" in requirements
    assert "ruff==0.16.2" in requirements


def test_workflow_installs_pinned_requirements_file():
    requirements = ROOT / "requirements-dev.txt"
    assert requirements.is_file()
    assert requirements.read_text(encoding="utf-8").splitlines() == [
        "pytest==9.1.1",
        "ruff==0.16.2",
    ]
    source = _workflow()
    assert "cache-dependency-path: requirements-dev.txt" in source
    assert "python -m pip install --disable-pip-version-check -r requirements-dev.txt" in source


def test_workflow_uses_warp_only_as_external_validation_input():
    source = _workflow()
    assert "--allow-missing-wheels" not in source
    assert "FLUIDSIM_REQUIRE_WHEELS" not in source
    assert 'test -x "$BLENDER"' in source
    assert 'test -e "$WARP_WHEEL_SOURCE"' in source
    assert "--wheel-dir" in source
    assert "external-warp-wheel" in source
    assert "external-warp-site" in source
    assert "PYTHONPATH" in source


def test_workflow_has_custom_effect_cpu_blender_and_required_cuda_gates():
    source = _workflow()
    assert "release/validate_custom_effects_warp.py" in source
    assert "--device cpu" in source
    assert "release/validate_custom_effects_blender.py" in source
    assert "--device cuda:0" in source
    assert "custom-effects-warp-cuda-validation.json" in source


def test_workflow_runs_and_uploads_native_properties_ui_blender_gate():
    source = _workflow()
    assert source.count("release/validate_properties_ui_blender.py") == 1
    assert "properties-ui-blender-validation.json" in source
    assert "--module fluidsim_ci_source" in source
    assert "--expected-warp-root \"$RUNNER_TEMP/external-warp-site\"" in source
