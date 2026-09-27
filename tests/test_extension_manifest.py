from __future__ import annotations

from pathlib import Path

import tomllib

from fluidSim_warp_blender51 import version as release_version

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "fluidSim_warp_blender51" / "blender_manifest.toml"


def _manifest():
    return tomllib.loads(MANIFEST_PATH.read_text(encoding="utf8"))


def test_source_manifest_matches_runtime_release_identity():
    manifest = _manifest()

    assert manifest["version"] == release_version.VERSION_STRING
    assert tuple(int(part) for part in manifest["blender_version_min"].split(".")) == release_version.BLENDER_MIN_VERSION


def test_source_manifest_remains_installable_on_proven_compatibility_targets():
    manifest = _manifest()

    assert release_version.BLENDER_MIN_VERSION == (5, 1, 0)
    assert release_version.VERSION_STRING == "1.1.0"
    supported = getattr(release_version, "SUPPORTED_PLATFORMS", None)
    assert supported == ("windows-x64", "linux-x64")
    assert tuple(manifest["platforms"]) == supported


def test_source_manifest_does_not_contain_builder_generated_metadata():
    manifest = _manifest()

    assert "generated" not in manifest.get("build", {})
