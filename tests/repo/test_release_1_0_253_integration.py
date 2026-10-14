"""Integration contract for the current release surface."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
TOOLS_RELEASE = ROOT.parent / "tools" / "release"
COLLISION_GUARD = TOOLS_RELEASE / "candidate_collision_guard_1_0_253.json"
UPDATE_GUARD = TOOLS_RELEASE / "candidate_integration_1_0_255.json"


def _assigned(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {path}")


def test_release_identity_targets_blender_52_with_only_locked_warp_wheels():
    manifest = tomllib.loads(
        (ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))

    assert manifest["id"] == "fluidsim_warp"
    assert manifest["version"] == "3.2.1"
    assert manifest["blender_version_min"] == "5.2.0"
    assert manifest["platforms"] == ["windows-x64", "linux-x64"]
    assert _assigned(ROOT / "version.py", "BLENDER_MIN_VERSION") == (5, 2, 0)
    assert _assigned(ROOT / "version.py", "SUPPORTED_PLATFORMS") == (
        "windows-x64",
        "linux-x64",
    )
    assert _assigned(ROOT / "version.py", "VALIDATED_RUNTIME_PLATFORMS") == (
        "windows-x64",
    )
    for forbidden in ("runtimes", "vendor", "python"):
        assert not (ROOT / forbidden).exists()
    # wheels/ holds only the fetched, checksum-locked Warp wheels (git-ignored).
    locked = set(json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(
        encoding="utf-8"))["files"])
    fetched = {path.name for path in (ROOT / "wheels").glob("*")}
    assert fetched <= locked


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
    # 2.0 removed the legacy snippet runtime; a removed owner holds no payload.
    # The release build reads the same list (tools/release/build_extension.py).
    retired = set(guard["retired_owners"])
    assert retired == {"sim/custom.py"}
    assert retired <= stale_hashes.keys()
    for relative, stale_hash in stale_hashes.items():
        if relative in retired:
            assert not (ROOT / relative).exists(), relative
            continue
        payload = (ROOT / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() != stale_hash, relative


def test_1_0_255_update_records_incremental_scope_and_retained_owners():
    assert UPDATE_GUARD.is_file()
    record = json.loads(UPDATE_GUARD.read_text(encoding="utf-8"))

    assert record["candidate_sha256"] == (
        "49c598ee53606463eaf66c4a71fcf645986fb9b1ef3c99549b4b9c4245ff50eb"
    )
    assert record["incremental_surface"] == {
        "changed": 15,
        "added": 3,
        "removed": 0,
    }
    assert {
        "blender/ui_physics.py",
        "blender/ui_view3d.py",
        "blender/output_publication.py",
        "operators.py",
    }.issubset(record["adopt_or_adapt"])
    assert {
        "blender/pending_settings.py",
        "blender/playback.py",
        "blender/production_bake.py",
        "blender/simulation_admission.py",
        "sim/cuda_reduction.py",
        "sim/solver.py",
    }.issubset(record["retained_current_owners"])
