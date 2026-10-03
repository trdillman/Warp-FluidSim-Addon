from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from tools.release import build_extension

LINUX_WHEEL = "warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl"
WINDOWS_WHEEL = "warp_lang-1.17.0-py3-none-win_amd64.whl"
WHEELS = {
    "linux-x64": (LINUX_WHEEL, b"linux-wheel-payload"),
    "windows-x64": (WINDOWS_WHEEL, b"windows-wheel-payload"),
}
STALE_OWNER_PAYLOAD = b"stale candidate owner\n"
RETIRED_OWNER = "sim/custom.py"
VERSION = "1.0.255"


def _manifest() -> str:
    return f'''schema_version = "1.0.0"
id = "fluidsim_warp"
name = "FluidSim Warp"
version = "{VERSION}"
tagline = "Interactive CUDA liquid simulation for Blender"
maintainer = "FluidSim"
type = "add-on"
blender_version_min = "5.2.0"
license = ["SPDX:GPL-3.0-or-later"]
platforms = ["windows-x64", "linux-x64"]
tags = ["Physics"]

[permissions]
files = "Read and write simulation caches, exports, and presets"
network = "Download NVIDIA Warp runtime from PyPI when you press Install Warp"

[build]
paths_exclude_pattern = [
  "__pycache__/",
  ".*",
  "*.zip",
  "*.whl",
  "wheels/",
  "tests/",
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
    tool_root = root.parent / "tools" / "release"
    tool_root.mkdir(parents=True, exist_ok=True)
    (root / "wheels").mkdir()
    (root / "blender_manifest.toml").write_text(_manifest(), encoding="utf-8")
    (root / "version.py").write_text(
        'VERSION = (1, 0, 255)\nEXTENSION_ID = "fluidsim_warp"\n'
        'BLENDER_MIN_VERSION = (5, 2, 0)\nCERTIFIED_WARP_VERSION = "1.17.0"\n'
        'CERTIFIED_WARP_DISTRIBUTION = "1.17.0+cu12"\n',
        encoding="utf-8",
    )
    (root / "__init__.py").write_text(
        "def register():\n    pass\n\ndef unregister():\n    pass\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("GPL-3.0-or-later\n", encoding="utf-8")
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "NVIDIA Warp 1.17.0 — Apache License, Version 2.0 — "
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
    (tool_root / "wheels.lock.json").write_text(
        json.dumps(
            {
                "warp_version": "1.17.0",
                "distribution_version": "1.17.0+cu12",
                "files": {
                    name: {
                        "platform": platform,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                        "source": "pypi",
                    }
                    for platform, (name, payload) in WHEELS.items()
                },
            }
        ),
        encoding="utf-8",
    )
    (tool_root / "candidate_collision_guard_1_0_253.json").write_text(
        json.dumps(
            {
                "candidate_sha256": "0" * 64,
                "forbidden_sha256": {
                    "sim/surface_tension.py": hashlib.sha256(
                        STALE_OWNER_PAYLOAD
                    ).hexdigest(),
                    RETIRED_OWNER: hashlib.sha256(STALE_OWNER_PAYLOAD).hexdigest(),
                },
                "retired_owners": [RETIRED_OWNER],
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


def _write_package(
    path: Path,
    *,
    platform: str = "linux-x64",
    surface_tension_payload: bytes = b"# surface tension\n",
    extra: dict[str, bytes] | None = None,
    wheel_platform: str | None = None,
    wheel_payload: bytes | None = None,
    include_wheel: bool = False,
) -> None:
    wheel_name, payload = WHEELS[wheel_platform or platform]
    generated = f'\n[build.generated]\nplatforms = ["{platform}"]\n'
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("blender_manifest.toml", _manifest() + generated)
        if include_wheel:
            archive.writestr(f"wheels/{wheel_name}", wheel_payload or payload)
        archive.writestr("__init__.py", "def register(): pass\ndef unregister(): pass\n")
        archive.writestr("LICENSE", "GPL-3.0-or-later\n")
        archive.writestr(
            "THIRD_PARTY_NOTICES.md",
            "NVIDIA Warp 1.17.0 — Apache License, Version 2.0 — "
            "libmathdx-LICENSE.txt — nvrtc-LICENSE.txt\n",
        )
        archive.writestr("release/__init__.py", "\n")
        archive.writestr("release/run_headless_bake.py", "# installed CLI entry point\n")
        archive.writestr("blender/setup_wizard.py", "# guided setup\n")
        archive.writestr("blender/geometry_nodes.py", "# render nodes\n")
        archive.writestr("sim/attributes.py", "# attribute schema\n")
        archive.writestr("sim/surface_tension.py", surface_tension_payload)
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


def test_verify_source_tree_reports_one_locked_wheel_per_platform(tmp_path: Path):
    _write_project(tmp_path)
    result = build_extension.verify_source_tree(tmp_path)
    assert result.platforms == ("windows-x64", "linux-x64")
    assert result.release_ready is True
    assert result.bundled_warp_wheels == (LINUX_WHEEL, WINDOWS_WHEEL)
    assert result.wheel_for_platform("linux-x64") == LINUX_WHEEL
    assert result.wheel_for_platform("windows-x64") == WINDOWS_WHEEL


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


def test_verify_source_tree_accepts_fetched_locked_wheels(tmp_path: Path):
    _write_project(tmp_path)
    for name, payload in WHEELS.values():
        (tmp_path / "wheels" / name).write_bytes(payload)
    assert build_extension.verify_source_tree(tmp_path).bundled_warp_wheels


@pytest.mark.parametrize(
    "stray",
    ["wheels/warp_lang-1.16.0+cu12-py3-none-win_amd64.whl", "vendor/numpy.whl"],
)
def test_verify_source_tree_rejects_unlocked_wheels_in_source_tree(
    tmp_path: Path, stray: str
):
    _write_project(tmp_path)
    target = tmp_path / stray
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"stray")
    with pytest.raises(RuntimeError, match="only the locked Warp wheels"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_rejects_warp_identity_drift(tmp_path: Path):
    _write_project(tmp_path)
    version = tmp_path / "version.py"
    version.write_text(
        version.read_text(encoding="utf-8").replace('"1.17.0"', '"1.16.0"'),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="CERTIFIED_WARP_VERSION"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_rejects_stale_candidate_owner_payload(tmp_path: Path):
    _write_project(tmp_path)
    (tmp_path / "sim" / "surface_tension.py").write_bytes(STALE_OWNER_PAYLOAD)

    with pytest.raises(RuntimeError, match="stale candidate implementation"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_rejects_a_retired_owner_that_reappears(tmp_path: Path):
    _write_project(tmp_path)
    (tmp_path / RETIRED_OWNER).write_bytes(b"# any payload\n")

    with pytest.raises(RuntimeError, match="retired integration owner reappeared"):
        build_extension.verify_source_tree(tmp_path)


def test_collision_guard_retired_owners_must_be_protected(tmp_path: Path):
    _write_project(tmp_path)
    guard_path = tmp_path.parent / "tools" / "release" / "candidate_collision_guard_1_0_253.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    guard["retired_owners"] = ["sim/unknown.py"]
    guard_path.write_text(json.dumps(guard), encoding="utf-8")

    with pytest.raises(RuntimeError, match="retired_owners"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_rejects_manifest_wheels(tmp_path: Path):
    _write_project(tmp_path)
    manifest = (tmp_path / "blender_manifest.toml").read_text(encoding="utf-8")
    manifest = manifest.replace(
        "[permissions]",
        f'wheels = [\n  "./wheels/{LINUX_WHEEL}",\n]\n\n[permissions]',
    )
    (tmp_path / "blender_manifest.toml").write_text(manifest, encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be empty"):
        build_extension.verify_source_tree(tmp_path)


def test_verify_source_tree_requires_network_permission(tmp_path: Path):
    _write_project(tmp_path)
    manifest = (tmp_path / "blender_manifest.toml").read_text(encoding="utf-8")
    manifest = manifest.replace(
        'network = "Download NVIDIA Warp runtime from PyPI when you press Install Warp"\n',
        "",
    )
    (tmp_path / "blender_manifest.toml").write_text(manifest, encoding="utf-8")
    with pytest.raises(RuntimeError, match="network"):
        build_extension.verify_source_tree(tmp_path)


@pytest.mark.parametrize("platform", ["windows-x64", "linux-x64"])
def test_inspect_packages_accepts_archive_without_a_warp_wheel(
    tmp_path: Path,
    platform: str,
):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-{platform.replace('-', '_')}.zip"
    _write_package(package, platform=platform)
    records = build_extension.inspect_packages(tmp_path, [package])
    assert [record.platform for record in records] == [platform]
    assert records[0].runtime_dependency == "user-install"
    assert records[0].sha256


def test_inspect_packages_rejects_a_bundled_wheel(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(package, platform="linux-x64", include_wheel=True)
    with pytest.raises(RuntimeError, match="should not contain bundled wheels"):
        build_extension.inspect_packages(tmp_path, [package])


def test_inspect_packages_keeps_sorted_artifact_records_for_valid_archives(
    tmp_path: Path,
):
    _write_project(tmp_path)
    windows = tmp_path / f"fluidsim_warp-{VERSION}-windows_x64.zip"
    linux = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(windows, platform="windows-x64")
    _write_package(linux, platform="linux-x64")

    records = build_extension.inspect_packages(tmp_path, [windows, linux])

    assert [record.path for record in records] == [linux.resolve(), windows.resolve()]
    assert [record.platform for record in records] == ["linux-x64", "windows-x64"]
    assert [record.runtime_dependency for record in records] == [
        "user-install",
        "user-install",
    ]
    assert [record.size for record in records] == [
        linux.stat().st_size,
        windows.stat().st_size,
    ]
    assert [record.sha256 for record in records] == [
        hashlib.sha256(linux.read_bytes()).hexdigest(),
        hashlib.sha256(windows.read_bytes()).hexdigest(),
    ]


def test_inspect_packages_rejects_platform_payload_drift(tmp_path: Path):
    _write_project(tmp_path)
    windows = tmp_path / f"fluidsim_warp-{VERSION}-windows_x64.zip"
    linux = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(windows, platform="windows-x64")
    _write_package(
        linux,
        platform="linux-x64",
        extra={"sim/platform_drift.py": b"linux-only payload\n"},
    )

    with pytest.raises(RuntimeError, match="split-platform payloads differ"):
        build_extension.inspect_packages(tmp_path, [windows, linux])


def test_inspect_packages_rejects_stale_candidate_owner_payload(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-windows_x64.zip"
    _write_package(
        package,
        platform="windows-x64",
        surface_tension_payload=STALE_OWNER_PAYLOAD,
    )

    with pytest.raises(RuntimeError, match="stale candidate implementation"):
        build_extension.inspect_packages(tmp_path, [package])


def test_inspect_packages_rejects_a_retired_owner(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-windows_x64.zip"
    _write_package(
        package, platform="windows-x64", extra={RETIRED_OWNER: b"# any payload\n"})

    with pytest.raises(RuntimeError, match="contains retired integration owner"):
        build_extension.inspect_packages(tmp_path, [package])


@pytest.mark.parametrize(
    "name",
    [
        LINUX_WHEEL,
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


def test_inspect_packages_rejects_compiled_python_artifacts(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(package, extra={"sim/kernel.pyc": b"compiled bytecode"})

    with pytest.raises(
        RuntimeError,
        match=r"compiled Python artifact leaked into package: sim/kernel\.pyc",
    ):
        build_extension.inspect_packages(tmp_path, [package])


def test_build_runs_all_gates_and_emits_checksums(tmp_path: Path):
    _write_project(tmp_path)
    output = tmp_path / "dist"
    commands: list[list[str]] = []

    def fake_run(command: list[str], cwd: Path, *, env=None):
        commands.append(command)
        if "--command" in command and command[
            command.index("--command") + 1 : command.index("--command") + 3
        ] == ["extension", "build"]:
            _write_package(
                output / f"fluidsim_warp-{VERSION}-windows_x64.zip",
                platform="windows-x64",
            )
            _write_package(
                output / f"fluidsim_warp-{VERSION}-linux_x64.zip",
                platform="linux-x64",
            )

    fetched: list[Path] = []
    packages = build_extension.build(
        tmp_path, Path("/opt/blender"), output, runner=fake_run, fetcher=fetched.append)
    assert fetched == []
    flattened = [" ".join(command) for command in commands]
    assert any("-m ruff check" in command for command in flattened)
    assert any("-m pytest" in command for command in flattened)
    assert any("-m compileall" in command for command in flattened)
    assert any("extension validate" in command for command in flattened)
    assert any("extension build --split-platforms" in command for command in flattened)
    package_commands = [command for command in commands if "--command" in command]
    assert len(package_commands) == 2
    assert all("--factory-startup" in command for command in package_commands)
    assert all("--background" in command for command in package_commands)
    assert all("--offline-mode" in command for command in package_commands)
    assert [path.name for path in packages] == [
        f"fluidsim_warp-{VERSION}-linux_x64.zip",
        f"fluidsim_warp-{VERSION}-windows_x64.zip",
    ]
    assert f"fluidsim_warp-{VERSION}-linux_x64.zip" in (output / "SHA256SUMS.txt").read_text()
    assert f"fluidsim_warp-{VERSION}-windows_x64.zip" in (
        output / "SHA256SUMS.txt"
    ).read_text()


def test_installed_smoke_runs_like_an_artist_launch_and_uninstalls(tmp_path: Path):
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
    # No --python-use-system-env: Blender does not load Warp from PYTHONPATH.
    # Smoke setup places the locked wheel in the isolated extensions/.local
    # directory, which is where the preferences button installs it.
    assert all("--python-use-system-env" not in command for command in background)
    assert all(env.get("PYTHONPATH") == "/validation/warp-site" for env in environments)
    expected_cache = str(tmp_path / "repo-profile" / "warp-cache")
    assert all(env.get("WARP_CACHE_PATH") == expected_cache for env in environments)
    assert (tmp_path / "repo-profile" / "warp-cache").is_dir()
    expected_temp = str(tmp_path / "repo-profile" / "temp")
    assert all(env.get("TEMP") == expected_temp for env in environments)
    assert all(env.get("TMP") == expected_temp for env in environments)
    assert (tmp_path / "repo-profile" / "temp").is_dir()
    flattened = [" ".join(command) for command in commands]
    assert any("extension remove fluidsim_ci.fluidsim_warp" in command for command in flattened)


def test_blender_process_environment_recovers_missing_windows_architecture():
    source = {"PATH": "C:/tools"}

    recovered = build_extension._blender_process_environment(
        source, os_name="nt", compiled_platform="win-amd64")
    explicit = build_extension._blender_process_environment(
        {"PROCESSOR_ARCHITECTURE": "ARM64"},
        os_name="nt",
        compiled_platform="win-amd64",
    )

    assert recovered["PROCESSOR_ARCHITECTURE"] == "AMD64"
    assert source == {"PATH": "C:/tools"}
    assert explicit["PROCESSOR_ARCHITECTURE"] == "ARM64"


def test_verify_only_is_successful_without_runtime_binary_payloads(tmp_path: Path):
    _write_project(tmp_path)
    report = tmp_path / "metadata-report.json"
    exit_code = build_extension.main(
        ["--root", str(tmp_path), "--verify-only", "--json-report", str(report)]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["release_ready"] is True
    assert payload["source"]["bundled_warp_wheels"] == [LINUX_WHEEL, WINDOWS_WHEEL]


def test_release_metadata_writes_are_atomic_and_strict(tmp_path: Path):
    output = tmp_path / "nested" / "dist"
    package = output / "fluidsim_warp-1.0.255-linux_x64.zip"
    record = build_extension.ArtifactRecord(
        path=package,
        platform="linux-x64",
        runtime_dependency="bundled",
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
