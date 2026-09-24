from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"


def test_manifest_declares_no_bundled_python_wheels():
    manifest = tomllib.loads((ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))
    assert "wheels" not in manifest or manifest["wheels"] == []


def test_warp_lock_is_validation_metadata_only():
    manifest = tomllib.loads((ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))
    lock = json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(encoding="utf-8"))
    assert lock["warp_version"] == "1.16.0"
    assert lock["distribution_version"] == "1.16.0+cu12"
    runtime_platforms = {item["platform"] for item in lock["files"].values()}
    assert runtime_platforms == {"linux-x64", "windows-x64"}
    assert runtime_platforms <= set(manifest["platforms"])
    assert not list((ROOT / "wheels").glob("*.whl"))


def test_manifest_excludes_only_addon_owned_non_runtime_paths_from_build():
    manifest = tomllib.loads((ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))
    patterns = set(manifest["build"]["paths_exclude_pattern"])
    assert {
        "tests/",
        "demo_outputs/",
        "artifacts/",
        "pyproject.toml",
        "requirements-dev.txt",
        "__pycache__/",
        "*.zip",
        "*.whl",
        "wheels/",
    }.issubset(patterns)
    assert not any(pattern.startswith("release/") for pattern in patterns)
