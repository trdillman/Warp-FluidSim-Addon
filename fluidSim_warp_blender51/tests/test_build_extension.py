from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from fluidsim_testpkg.release import build_extension

EXTERNAL_WHEEL = "warp_lang-1.16.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
PAYLOAD = b"external-wheel-payload"
VERSION = "1.0.252"


def _manifest() -> str:
    return f'''schema_version = "1.0.0"
id = "fluidsim_warp"
name = "FluidSim Warp"
version = "{VERSION}"
tagline = "Interactive CUDA liquid simulation with beta ST-FLIP"
maintainer = "FluidSim"
type = "add-on"
blender_version_min = "5.1.0"
license = ["SPDX:GPL-3.0-or-later"]
platforms = ["linux-x64"]
tags = ["Physics"]

[permissions]
files = "Read and write simulation caches, exports, and presets"

[build]
paths_exclude_pattern = [
  "__pycache__/",
  ".*",
  "*.zip",
  "*.whl",
  "wheels/",
  "tests/",
  "release/build_extension.py",
  "release/delivery_policy.py",
  "release/fetch_warp_wheels.py",
  "release/custom_effect_demo_runtime.py",
  "release/generate_custom_effect_demos.py",
  "release/generate_examples.py",
  "release/generate_geometry_nodes.py",
  "release/gpu_regression_blender.py",
  "release/smoke_test_blender.py",
  "release/validate_examples_blender.py",
  "release/validate_custom_effects_blender.py",
  "release/validate_custom_effects_warp.py",
  "release/validate_phase3_blender.py",
  "release/validate_phase3_warp.py",
  "release/validate_phase4_blender.py",
  "release/validate_phase4_warp.py",
  "release/validate_phase5_blender.py",
  "release/validate_phase5_warp.py",
  "release/wheels.lock.json",
  "demo_outputs/",
  "artifacts/",
  "pyproject.toml",
  "requirements-dev.txt",
  "build/",
  "dist/",
]
'''


def _write_project(root: Path) -> None:
    (root / "release").mkdir(parents=True)
    (root / "wheels").mkdir()
    (root / "blender_manifest.toml").write_text(_manifest(), encoding="utf-8")
    (root / "version.py").write_text(
        'VERSION = (1, 0, 252)\nEXTENSION_ID = "fluidsim_warp"\nBLENDER_MIN_VERSION = (5, 1, 0)\n',
        encoding="utf-8",
    )
    (root / "__init__.py").write_text(
        "def register():\n    pass\n\ndef unregister():\n    pass\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("GPL-3.0-or-later\n", encoding="utf-8")
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "NVIDIA Warp 1.16.0+cu12 — Apache License, Version 2.0 — "
        "libmathdx-LICENSE.txt — nvrtc-LICENSE.txt\n",
        encoding="utf-8",
    )
    (root / "sim").mkdir()
    (root / "sim" / "surface_tension.py").write_text("# surface tension\n", encoding="utf-8")
    (root / "sim" / "surface_tension_math.py").write_text(
        "# surface tension math\n", encoding="utf-8")
    effect_sim = root / "sim" / "effects"
    effect_sim.mkdir()
    for name in ("schema.py", "runtime.py", "package.py"):
        (effect_sim / name).write_text(f"# custom effect {name}\n", encoding="utf-8")
    blender_dir = root / "blender"
    blender_dir.mkdir(exist_ok=True)
    for name in ("effects.py", "effect_ui.py", "effect_workspace.py"):
        (blender_dir / name).write_text(f"# custom effect {name}\n", encoding="utf-8")
    preset_dir = root / "resources" / "presets"
    preset_dir.mkdir(parents=True)
    (preset_dir / "water_surface_tension.json").write_text("{}\n", encoding="utf-8")
    (preset_dir / "stylized_high_tension.json").write_text("{}\n", encoding="utf-8")
    render_dir = root / "resources" / "geometry_nodes"
    render_dir.mkdir(parents=True)
    (render_dir / "fluidsim_render_nodes.blend").write_bytes(
        b"BLENDER" + b"\0" * 50_001
    )
    (root / "release" / "wheels.lock.json").write_text(
        json.dumps(
            {
                "warp_version": "1.16.0",
                "distribution_version": "1.16.0+cu12",
                "files": {
                    EXTERNAL_WHEEL: {
                        "platform": "linux-x64",
                        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
                        "size": len(PAYLOAD),
                        "source": "local-certified-binary",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    effects_dir = root / "resources" / "effects"
    sources_dir = effects_dir / "sources"
    sources_dir.mkdir(parents=True)
    for index, effect_id in enumerate((
        "vortex_force", "curl_turbulence", "radial_pulse", "planar_split",
        "velocity_damping", "temperature_lift", "particle_speed_limit", "voxel_cooling",
    )):
        (effects_dir / f"{effect_id}.json").write_text("{}\n", encoding="utf-8")
        (sources_dir / f"{effect_id}.py").write_text(f"# effect {index}\n", encoding="utf-8")
    examples = root / "resources" / "examples"
    renders = examples / "custom_effects" / "renders"
    renders.mkdir(parents=True)
    for stem in (
        "05_custom_effects_vortex", "06_custom_effects_stack", "07_warp_code_authoring",
    ):
        (examples / f"{stem}.blend").write_bytes(b"BLENDER" + b"\0" * 50_001)
        (renders / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 1_024)


def _write_package(path: Path, *, extra: dict[str, bytes] | None = None) -> None:
    generated = '\n[build.generated]\nplatforms = ["linux-x64"]\nwheels = []\n'
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("blender_manifest.toml", _manifest() + generated)
        archive.writestr("__init__.py", "def register(): pass\ndef unregister(): pass\n")
        archive.writestr("LICENSE", "GPL-3.0-or-later\n")
        archive.writestr(
            "THIRD_PARTY_NOTICES.md",
            "NVIDIA Warp 1.16.0+cu12 — Apache License, Version 2.0 — "
            "libmathdx-LICENSE.txt — nvrtc-LICENSE.txt\n",
        )
        archive.writestr("release/__init__.py", "\n")
        archive.writestr("release/run_headless_bake.py", "# installed CLI entry point\n")
        archive.writestr("blender/setup_wizard.py", "# guided setup\n")
        archive.writestr("blender/geometry_nodes.py", "# render nodes\n")
        archive.writestr("sim/attributes.py", "# attribute schema\n")
        archive.writestr("sim/surface_tension.py", "# surface tension\n")
        archive.writestr("sim/surface_tension_math.py", "# surface tension math\n")
        archive.writestr("sim/effects/schema.py", "# custom effects schema\n")
        archive.writestr("sim/effects/runtime.py", "# custom effects runtime\n")
        archive.writestr("sim/effects/package.py", "# custom effects package\n")
        archive.writestr("blender/effects.py", "# custom effects scene adapter\n")
        archive.writestr("blender/effect_ui.py", "# custom effects UI\n")
        archive.writestr("blender/effect_workspace.py", "# Warp Code workspace\n")
        archive.writestr("presets/schema.py", "# strict preset schema\n")
        archive.writestr("resources/presets/preview_water.json", "{}\n")
        archive.writestr("resources/presets/water_surface_tension.json", "{}\n")
        archive.writestr("resources/presets/stylized_high_tension.json", "{}\n")
        archive.writestr(
            "resources/geometry_nodes/fluidsim_render_nodes.blend",
            b"BLENDER" + b"\0" * 50_001,
        )
        for index, effect_id in enumerate((
            "vortex_force", "curl_turbulence", "radial_pulse", "planar_split",
            "velocity_damping", "temperature_lift", "particle_speed_limit", "voxel_cooling",
        )):
            archive.writestr(f"resources/effects/{effect_id}.json", "{}\n")
            archive.writestr(f"resources/effects/sources/{effect_id}.py", f"# effect {index}\n")
        for stem in (
            "05_custom_effects_vortex", "06_custom_effects_stack", "07_warp_code_authoring",
        ):
            archive.writestr(
                f"resources/examples/{stem}.blend", b"BLENDER" + b"\0" * 50_001
            )
            archive.writestr(
                f"resources/examples/custom_effects/renders/{stem}.png",
                b"\x89PNG\r\n\x1a\n" + b"\0" * 1_024,
            )
        for name, value in (extra or {}).items():
            archive.writestr(name, value)


def test_verify_source_tree_accepts_external_runtime_dependency_project(tmp_path: Path):
    _write_project(tmp_path)
    result = build_extension.verify_source_tree(tmp_path)
    assert result.platforms == ("linux-x64",)
    assert result.release_ready is True
    assert result.external_runtime_wheels == (EXTERNAL_WHEEL,)


def test_verify_source_tree_rejects_invalid_render_asset(tmp_path: Path):
    _write_project(tmp_path)
    asset = tmp_path / "resources" / "geometry_nodes" / "fluidsim_render_nodes.blend"
    asset.write_bytes(b"not-a-blend")
    with pytest.raises(RuntimeError, match="Geometry Nodes asset"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_requires_custom_effect_product_shell(tmp_path: Path):
    _write_project(tmp_path)
    (tmp_path / "sim" / "effects" / "runtime.py").unlink()
    with pytest.raises(RuntimeError, match="custom effect"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_rejects_wheel_payload_in_source_tree(tmp_path: Path):
    _write_project(tmp_path)
    (tmp_path / "wheels" / EXTERNAL_WHEEL).write_bytes(PAYLOAD)
    with pytest.raises(RuntimeError, match="runtime binary payloads must not be stored"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_rejects_manifest_wheel_declaration(tmp_path: Path):
    _write_project(tmp_path)
    manifest = (tmp_path / "blender_manifest.toml").read_text(encoding="utf-8")
    manifest = manifest.replace('tags = ["Physics"]', f'wheels = ["./wheels/{EXTERNAL_WHEEL}"]\ntags = ["Physics"]')
    (tmp_path / "blender_manifest.toml").write_text(manifest, encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not declare bundled wheels"):
        build_extension.verify_source_tree(tmp_path)


def test_inspect_packages_accepts_binary_free_platform_archive(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(package)
    records = build_extension.inspect_packages(tmp_path, [package])
    assert [record.platform for record in records] == ["linux-x64"]
    assert records[0].runtime_dependency == "external"
    assert records[0].sha256


@pytest.mark.parametrize(
    "name",
    [
        f"wheels/{EXTERNAL_WHEEL}",
        EXTERNAL_WHEEL,
        "vendor/blender-5.2.0-linux-x64.tar.xz",
        "vendor/blender-5.2.0-linux-x64.zip",
        "vendor/warp/bin/warp.so",
        "vendor/warp_lang-1.16.0.dist-info/METADATA",
    ],
)
def test_inspect_packages_rejects_packaged_blender_or_warp_runtime(tmp_path: Path, name: str):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(package, extra={name: b"forbidden"})
    with pytest.raises(RuntimeError, match="forbidden bundled runtime payload"):
        build_extension.inspect_packages(tmp_path, [package])


def test_inspect_packages_preserves_addon_blender_integration_source(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(package, extra={"blender/scene_bridge.py": b"# addon integration source\n"})
    records = build_extension.inspect_packages(tmp_path, [package])
    assert records[0].platform == "linux-x64"


def test_inspect_packages_rejects_release_only_content(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(package, extra={"tests/test_leak.py": b"pass\n"})
    with pytest.raises(RuntimeError, match="release-only path"):
        build_extension.inspect_packages(tmp_path, [package])


def test_build_runs_all_gates_and_emits_checksums(tmp_path: Path):
    _write_project(tmp_path)
    output = tmp_path / "dist"
    commands: list[list[str]] = []

    def fake_run(command: list[str], cwd: Path, *, env=None):
        commands.append(command)
        if command[1:4] == ["--command", "extension", "build"]:
            _write_package(output / f"fluidsim_warp-{VERSION}-linux_x64.zip")

    packages = build_extension.build(tmp_path, Path("/opt/blender"), output, runner=fake_run)
    flattened = [" ".join(command) for command in commands]
    assert any("-m ruff check" in command for command in flattened)
    assert any("-m pytest" in command for command in flattened)
    assert any("-m compileall" in command for command in flattened)
    assert any("extension validate" in command for command in flattened)
    assert any("extension build --split-platforms" in command for command in flattened)
    assert [path.name for path in packages] == [f"fluidsim_warp-{VERSION}-linux_x64.zip"]
    assert f"fluidsim_warp-{VERSION}-linux_x64.zip" in (output / "SHA256SUMS.txt").read_text()


def test_installed_smoke_uses_external_python_environment_and_uninstalls(tmp_path: Path):
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    package.write_bytes(b"zip")
    script = tmp_path / "smoke_test_blender.py"
    script.write_text("print('smoke')\n", encoding="utf-8")
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(command: list[str], cwd: Path, *, env=None):
        commands.append(command)
        environments.append(dict(env or {}))

    build_extension.run_installed_smoke(
        Path("/opt/blender"),
        package,
        tmp_path / "repo",
        script,
        runner=fake_run,
        extra_env={"PYTHONPATH": "/validation/warp-site"},
    )
    background = [command for command in commands if "--background" in command]
    assert len(background) == 2
    assert all("--python-use-system-env" in command for command in background)
    assert all(env.get("PYTHONPATH") == "/validation/warp-site" for env in environments)
    flattened = [" ".join(command) for command in commands]
    assert any("extension remove fluidsim_ci.fluidsim_warp" in command for command in flattened)


def test_verify_only_is_successful_without_runtime_binary_payloads(tmp_path: Path):
    _write_project(tmp_path)
    report = tmp_path / "metadata-report.json"
    exit_code = build_extension.main(
        ["--root", str(tmp_path), "--verify-only", "--json-report", str(report)]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["release_ready"] is True
    assert payload["source"]["external_runtime_wheels"] == [EXTERNAL_WHEEL]


def test_release_metadata_writes_are_atomic_and_strict(tmp_path: Path):
    output = tmp_path / "nested" / "dist"
    package = output / "fluidsim_warp-1.0.252-linux_x64.zip"
    record = build_extension.ArtifactRecord(
        path=package,
        platform="linux-x64",
        runtime_dependency="external",
        size=123,
        sha256="a" * 64,
    )

    checksum_path = build_extension._write_checksums(output, [record])
    report_path = output / "build-report.json"
    build_extension._write_json_report(report_path, {"status": "passed", "count": 1})

    assert checksum_path.read_text(encoding="utf-8") == (
        f"{'a' * 64}  {package.name}\n"
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "count": 1,
        "status": "passed",
    }
    assert list(output.glob(".*.tmp")) == []

    with pytest.raises(ValueError, match="Out of range float values"):
        build_extension._write_json_report(report_path, {"invalid": float("nan")})
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"
    assert list(output.glob(".*.tmp")) == []
