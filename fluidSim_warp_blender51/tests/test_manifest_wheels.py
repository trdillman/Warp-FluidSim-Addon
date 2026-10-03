from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"


def _manifest() -> dict:
    return tomllib.loads((ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))


def _lock() -> dict:
    return json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(encoding="utf-8"))


def test_manifest_does_not_bundle_wheels():
    """Verify that the manifest does not declare bundled wheels (user installs via button)."""
    manifest = _manifest()
    assert "wheels" not in manifest or manifest.get("wheels") == []


def test_manifest_declares_network_permission():
    """Verify that the manifest declares network permission for PyPI downloads."""
    manifest = _manifest()
    assert "permissions" in manifest
    assert "network" in manifest["permissions"]
    assert "PyPI" in manifest["permissions"]["network"] or "download" in manifest["permissions"]["network"].lower()


def test_warp_lock_covers_every_platform_once_and_matches_identity():
    import ast

    tree = ast.parse((ROOT / "version.py").read_text(encoding="utf-8"))
    identity = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)
    }
    lock = _lock()
    assert lock["warp_version"] == identity["CERTIFIED_WARP_VERSION"]
    assert lock["distribution_version"] == identity["CERTIFIED_WARP_DISTRIBUTION"]
    platforms = [item["platform"] for item in lock["files"].values()]
    assert sorted(platforms) == sorted(_manifest()["platforms"])
    for name, item in lock["files"].items():
        assert name.startswith(f"warp_lang-{identity['CERTIFIED_WARP_VERSION']}-")
        assert item["url"].startswith("https://files.pythonhosted.org/")
        assert len(item["sha256"]) == 64 and item["size"] > 0


def test_manifest_excludes_only_addon_owned_non_runtime_paths_from_build():
    # Manifest-listed wheels are always packaged; the *.whl / wheels/ excludes
    # keep any stray, unlisted wheel out of the build.
    patterns = set(_manifest()["build"]["paths_exclude_pattern"])
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
