from __future__ import annotations

import ast
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _assigned_value(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned in {path}")


def test_manifest_matches_runtime_version_and_identity():
    manifest = tomllib.loads((ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))
    version = _assigned_value(ROOT / "version.py", "VERSION")
    extension_id = _assigned_value(ROOT / "version.py", "EXTENSION_ID")
    blender_min = _assigned_value(ROOT / "version.py", "BLENDER_MIN_VERSION")
    assert manifest["version"] == ".".join(map(str, version))
    assert manifest["id"] == extension_id == "fluidsim_warp"
    assert manifest["license"] == ["SPDX:GPL-3.0-or-later"]
    assert manifest["platforms"] == ["windows-x64", "linux-x64"]
    assert manifest["blender_version_min"] == ".".join(map(str, blender_min)) == "5.2.0"
    assert version == (3, 2, 0)
    assert _assigned_value(ROOT / "version.py", "BUILD_CHANNEL") == "stable"


def test_supported_platforms_are_independent_of_current_runtime_evidence():
    supported = _assigned_value(ROOT / "version.py", "SUPPORTED_PLATFORMS")
    validated = _assigned_value(ROOT / "version.py", "VALIDATED_RUNTIME_PLATFORMS")

    assert supported == ("windows-x64", "linux-x64")
    # Only Windows has a recorded GPU gate run; Linux ships uncertified (D4).
    assert validated == ("windows-x64",)
    assert set(validated).issubset(supported)


def test_manifest_is_the_only_blender_metadata_surface():
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "bl_info" not in init_source
    assert "ensure_warp_discoverable" not in init_source


def test_distribution_contains_license_notices_and_changelog():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3" in license_text
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "NVIDIA Warp" in notices
    assert "warp-lang 1.17.0" in notices
    assert "CUDA 12" in notices
    assert "Apache License, Version 2.0" in notices
    assert "libmathdx-LICENSE.txt" in notices
    assert "nvrtc-LICENSE.txt" in notices
    assert (ROOT / "CHANGELOG.md").is_file()


def test_solver_constants_consume_canonical_identity():
    source = (ROOT / "sim" / "constants.py").read_text(encoding="utf-8")
    assert "from ..version import BUILD_LABEL, VERSION_STRING" in source
    assert "TOOL_VERSION = VERSION_STRING" in source
    assert "TOOL_BUILD = BUILD_LABEL" in source


def test_changelog_leads_with_the_current_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    version = ".".join(map(str, _assigned_value(ROOT / "version.py", "VERSION")))
    first_entry = next(
        line for line in changelog.splitlines() if line.startswith("## "))
    assert first_entry.startswith(f"## {version} ")


def test_user_guide_declares_certified_support_matrix():
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    for text in (
        "Blender 5.2.0",
        "Windows x64",
        "Linux x64",
        "NVIDIA CUDA",
        "Bundled NVIDIA Warp 1.17.0",
        "Cache v1 (1.0.x) is refused",
    ):
        assert text in guide
    assert "| Operating systems | Windows x64 and Linux x64 |" in guide
    version = ".".join(map(str, _assigned_value(ROOT / "version.py", "VERSION")))
    assert f"| FluidSim | {version} |" in guide
    assert f"FluidSim Warp {version} is" in guide
