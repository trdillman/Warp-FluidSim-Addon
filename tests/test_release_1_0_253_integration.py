"""Integration contract for the adapted 1.0.253 release surface."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"
COLLISION_GUARD = ROOT / "release" / "candidate_collision_guard_1_0_253.json"


def _assigned(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {path}")


def test_release_identity_targets_blender_51_without_bundled_runtimes():
    manifest = tomllib.loads(
        (ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))

    assert manifest["id"] == "fluidsim_warp"
    assert manifest["version"] == "1.0.253"
    assert manifest["blender_version_min"] == "5.1.0"
    assert manifest["platforms"] == ["windows-x64", "linux-x64"]
    assert _assigned(ROOT / "version.py", "BLENDER_MIN_VERSION") == (5, 1, 0)
    assert _assigned(ROOT / "version.py", "SUPPORTED_PLATFORMS") == (
        "windows-x64",
        "linux-x64",
    )
    assert _assigned(ROOT / "version.py", "VALIDATED_RUNTIME_PLATFORMS") == (
        "linux-x64",
    )
    for forbidden in ("wheels", "runtimes", "vendor", "python"):
        assert not (ROOT / forbidden).exists()


def test_viewport_panel_remains_a_compact_properties_launcher():
    source = (ROOT / "panels.py").read_text(encoding="utf-8")

    assert "activity_ui_state(scene)" in source
    assert '"fluidsim.open_scene_properties"' in source
    assert 'prop(fs, "cache_' not in source
    assert 'prop(fs, "voxel_size"' not in source
    assert 'prop(fs, "advanced_' not in source


def test_adjudicated_owners_cannot_revert_to_stale_candidate_payloads():
    assert COLLISION_GUARD.is_file()
    guard = json.loads(COLLISION_GUARD.read_text(encoding="utf-8"))
    stale_hashes = guard["forbidden_sha256"]

    assert guard["candidate_sha256"] == (
        "354d32b97bf952c5bf07b62543dda66ed39b7368f0b9d7b91ad83521ac2061e7"
    )
    assert len(stale_hashes) == 27
    for relative, stale_hash in stale_hashes.items():
        payload = (ROOT / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() != stale_hash, relative
