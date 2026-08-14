from __future__ import annotations

import tomllib
from pathlib import Path

from fluidSim_warp_blender51.version import BLENDER_MIN_VERSION, VERSION_STRING


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "fluidSim_warp_blender51" / "blender_manifest.toml"


def _manifest():
    return tomllib.loads(MANIFEST_PATH.read_text(encoding="utf8"))


def test_source_manifest_matches_runtime_release_identity():
    manifest = _manifest()

    assert manifest["version"] == VERSION_STRING
    assert tuple(int(part) for part in manifest["blender_version_min"].split(".")) == BLENDER_MIN_VERSION


def test_source_manifest_remains_installable_on_proven_compatibility_targets():
    manifest = _manifest()

    assert BLENDER_MIN_VERSION == (5, 1, 0)
    assert "platforms" not in manifest


def test_source_manifest_does_not_contain_builder_generated_metadata():
    manifest = _manifest()

    assert "generated" not in manifest.get("build", {})
