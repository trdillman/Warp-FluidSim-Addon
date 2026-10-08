from __future__ import annotations

import json

from fluidsim_testpkg.blender.ui_cache_state import production_bake_action
from fluidsim_testpkg.io.cache_schema import (
    LEGACY_2X_CACHE_MESSAGE,
    CacheManifestV3,
    write_manifest_atomic,
)


def _manifest(*, frame_end: int = 10) -> CacheManifestV3:
    return CacheManifestV3.new(
        solver_version="3.0.0",
        scene_key="sha256:" + "1" * 64,
        settings_signature="sha256:" + "2" * 64,
        source_signature="sha256:" + "3" * 64,
        frame_start=1,
        frame_end=frame_end,
        checkpoint_interval=1,
        effect_signature="4" * 64,
        effect_inventory=(),
        source_schema_version=1,
        transfer_mode="FLIP",
        grid_spec={
            "resolution_divisions": 64, "voxel_size": 0.0625,
            "domain_size": [4.0, 4.0, 4.0], "leaf_dim": 8},
        narrow_band={"enabled": False, "band_voxels": 0.0},
        platform={
            "os": "Windows-11", "gpu_name": "NVIDIA RTX", "driver": "560.94",
            "warp": "1.17.0"},
    )


def test_production_bake_starts_when_no_manifest_exists(tmp_path):
    state = production_bake_action(tmp_path)
    assert state.action == "START"
    assert state.message == "No cache manifest exists"
    assert state.destructive is False


def test_production_bake_resumes_an_existing_incomplete_manifest(tmp_path):
    write_manifest_atomic(tmp_path / "fluidsim_cache.json", _manifest())
    state = production_bake_action(tmp_path)
    assert state.action == "RESUME"
    assert "verify" in state.message.lower()
    assert state.destructive is False


def test_production_bake_restarts_a_complete_manifest(tmp_path):
    manifest = _manifest(frame_end=1)
    # The UI presentation only needs the manifest transaction boundary; exact
    # frame/checkpoint integrity remains an execute-time responsibility.
    from fluidsim_testpkg.io.cache_schema import FrameEntry

    manifest.frames[1] = FrameEntry(
        frame=1,
        frame_file="frames/frame_000001.npz",
        frame_sha256="a" * 64,
        frame_size=1,
        checkpoint_file="checkpoints/checkpoint_000001.npz",
        checkpoint_sha256="b" * 64,
        checkpoint_size=1,
        solver_frame=1,
        simulation_time=0.1,
        settings_signature="sha256:" + "2" * 64,
        source_signature="sha256:" + "3" * 64,
        completed_at="2026-08-18T00:00:00Z",
        source_frame_stamp="c" * 64,
        whitewater_count=0,
        particle_count=1,
    )
    write_manifest_atomic(tmp_path / "fluidsim_cache.json", manifest)
    state = production_bake_action(tmp_path)
    assert state.action == "RESTART"
    assert state.destructive is True


def test_production_bake_restarts_an_invalid_manifest(tmp_path):
    (tmp_path / "fluidsim_cache.json").write_text("{}", encoding="utf-8")
    state = production_bake_action(tmp_path)
    assert state.action == "RESTART"
    assert state.destructive is True
    assert "invalid" in state.message.lower()
    assert "v2" not in state.message.lower()
    assert state.message != LEGACY_2X_CACHE_MESSAGE


def test_production_bake_restarts_a_fluidsim_2x_cache_with_the_rebake_message(tmp_path):
    (tmp_path / "fluidsim_cache.json").write_text(
        json.dumps({"format_version": 2, "frames": {}}), encoding="utf-8")
    state = production_bake_action(tmp_path)
    assert state.action == "RESTART"
    assert state.destructive is True
    assert state.message == LEGACY_2X_CACHE_MESSAGE
