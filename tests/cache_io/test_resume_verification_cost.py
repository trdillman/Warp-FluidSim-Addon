"""Resume verifies every artifact each time but parses each checkpoint once (P07).

A resume asks for its decision two (GUI) or three (command line) times, and
each decision fully loaded every checkpoint of the history, hashing each one
twice. Now a checkpoint is hashed and parsed from one read, and a later
decision skips the parse of a checkpoint whose verified digest was already
parsed against the same expectations. Every decision still re-hashes every
frame and checkpoint, so corruption between decisions is still refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fixtures.tiny_cache import SOURCE_SIGNATURE, _payload

from fluidSim_warp_blender51 import blender as blender_package
from fluidSim_warp_blender51.blender import production_bake
from fluidSim_warp_blender51.blender.production_bake import ProductionBakeBackend
from fluidSim_warp_blender51.io import cache, cache_schema, checkpoint
from fluidSim_warp_blender51.version import VERSION_STRING

SIG = "settings-signature"


def _write_cache(base: Path, frames: int) -> None:
    manifest = cache_schema.CacheManifestV3.new(
        solver_version=VERSION_STRING, scene_key="key", settings_signature=SIG,
        source_signature=SOURCE_SIGNATURE, frame_start=1, frame_end=frames + 5,
        source_schema_version=1, transfer_mode="FLIP",
        grid_spec={"resolution_divisions": 16, "voxel_size": 0.1,
                   "domain_size": [1.6, 1.6, 1.6], "leaf_dim": 8},
        narrow_band={"enabled": False, "band_voxels": 0.0},
        platform={"os": "t", "gpu_name": "g", "driver": "0", "warp": "w"})
    rng = np.random.default_rng(1)
    for frame in range(1, frames + 1):
        stamp = hashlib.sha256(f"stamp-{frame}".encode()).hexdigest()
        payload = _payload(frame, SIG)
        frame_path, frame_sha, frame_size = cache.write_frame_v3_atomic(
            base, payload, frame, scene_key_value="key", settings_signature=SIG,
            source_signature=SOURCE_SIGNATURE, source_frame_stamp=stamp,
            simulation_time=frame / 24.0)
        ckpt = base / "checkpoints" / f"checkpoint_{frame:06d}.npz"
        digest = checkpoint.write_checkpoint_atomic(ckpt, checkpoint.CheckpointPayload(
            frame=frame, sim_time=frame / 24.0, settings_signature=SIG,
            source_signature=SOURCE_SIGNATURE,
            arrays={"particle_position": rng.random((64, 3), dtype=np.float32),
                    "field_pressure": rng.random(512, dtype=np.float32)}))
        manifest = manifest.with_committed_frame(cache_schema.FrameEntry(
            frame=frame, frame_file=frame_path.relative_to(base).as_posix(),
            frame_sha256=frame_sha, frame_size=frame_size,
            checkpoint_file=ckpt.relative_to(base).as_posix(),
            checkpoint_sha256=digest, checkpoint_size=ckpt.stat().st_size,
            completed_at="2026-10-12T00:00:00Z", solver_frame=frame,
            simulation_time=frame / 24.0, settings_signature=SIG,
            source_signature=SOURCE_SIGNATURE, source_frame_stamp=stamp,
            whitewater_count=0, particle_count=len(payload.positions)))
    cache_schema.write_manifest_atomic(cache_schema.manifest_path(base), manifest)


@pytest.fixture
def backend(monkeypatch, tmp_path):
    @contextlib.contextmanager
    def suppress():
        yield

    fake_playback = types.ModuleType("fluidSim_warp_blender51.blender.playback")
    fake_playback.suppress_frame_handlers = suppress
    fake_effects = types.ModuleType("fluidSim_warp_blender51.blender.effects")
    fake_effects.effect_identity = lambda _scene: ("", ())
    for name, module in (("playback", fake_playback), ("effects", fake_effects)):
        monkeypatch.setitem(sys.modules, f"fluidSim_warp_blender51.blender.{name}", module)
        monkeypatch.setattr(blender_package, name, module, raising=False)
    monkeypatch.setattr(production_bake, "_parsed_checkpoints", {})
    base = tmp_path / "cache"
    _write_cache(base, 4)
    instance = ProductionBakeBackend()
    instance.cache_path = lambda _scene: base
    instance._signatures_at = lambda _scene, _frame: {
        "scene_key": "key", "settings_signature": SIG,
        "source_signature": SOURCE_SIGNATURE}
    instance._first_changed_frame = lambda _scene, _manifest: None
    instance.base = base
    return instance


def _count_reads(monkeypatch):
    calls = []
    original = production_bake.read_checkpoint

    def counting(path, **kwargs):
        calls.append(Path(path).name)
        return original(path, **kwargs)

    monkeypatch.setattr(production_bake, "read_checkpoint", counting)
    return calls


def test_repeated_decisions_parse_each_checkpoint_once(backend, monkeypatch):
    reads = _count_reads(monkeypatch)
    scene = SimpleNamespace()
    decisions = [backend.resume_decision(scene) for _ in range(3)]
    assert {d.code for d in decisions} == {"RESUME_READY"}
    assert {d.resume_frame for d in decisions} == {5}
    assert len(decisions) == 3 and decisions[0] == decisions[1] == decisions[2]
    assert sorted(reads) == [f"checkpoint_{f:06d}.npz" for f in range(1, 5)]


def test_corruption_after_a_decision_is_still_refused(backend, monkeypatch):
    scene = SimpleNamespace()
    assert backend.resume_decision(scene).allowed
    path = backend.base / "checkpoints" / "checkpoint_000002.npz"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    decision = backend.resume_decision(scene)
    assert decision.code == "CACHE_INTEGRITY_FAILURE"
    assert "checkpoint 2 checksum mismatch" in decision.message
    frame = backend.base / "frames" / "frame_000003.npz"
    frame.write_bytes(frame.read_bytes()[:-1])
    assert backend.resume_decision(scene).code == "CACHE_INTEGRITY_FAILURE"


def test_a_checkpoint_replaced_under_a_new_digest_is_parsed_again(backend, monkeypatch):
    scene = SimpleNamespace()
    assert backend.resume_decision(scene).allowed
    reads = _count_reads(monkeypatch)
    manifest = cache_schema.load_manifest(cache_schema.manifest_path(backend.base))
    entry = manifest.frames[3]
    path = backend.base / entry.checkpoint_file
    # A well-formed checkpoint for the wrong frame, committed with its digest.
    digest = checkpoint.write_checkpoint_atomic(path, checkpoint.CheckpointPayload(
        frame=2, sim_time=3 / 24.0, settings_signature=SIG,
        source_signature=SOURCE_SIGNATURE))
    from dataclasses import replace
    manifest.frames[3] = replace(entry, checkpoint_sha256=digest,
                                 checkpoint_size=path.stat().st_size)
    cache_schema.write_manifest_atomic(cache_schema.manifest_path(backend.base), manifest)
    decision = backend.resume_decision(scene)
    assert reads == ["checkpoint_000003.npz"]
    assert decision.code == "CACHE_INTEGRITY_FAILURE"
    assert "checkpoint frame mismatch" in decision.message


def test_single_read_checkpoint_matches_a_file_read(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "checkpoint.npz"
    digest = checkpoint.write_checkpoint_atomic(path, checkpoint.CheckpointPayload(
        frame=7, sim_time=0.25, settings_signature="s", source_signature="q",
        arrays={"particle_position": rng.random((33, 3), dtype=np.float32),
                "field_pressure": rng.random(512, dtype=np.float32)},
        scalars={"leaf_count": 1}))
    size = path.stat().st_size
    verified = checkpoint.read_checkpoint(path, expected_sha256=digest, expected_size=size)
    unverified = checkpoint.read_checkpoint(path)
    assert verified.scalars == unverified.scalars
    assert set(verified.arrays) == set(unverified.arrays)
    for name, value in unverified.arrays.items():
        assert verified.arrays[name].dtype == value.dtype
        assert verified.arrays[name].tobytes() == value.tobytes()
    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        checkpoint.read_checkpoint(path, expected_sha256=digest, expected_size=size + 1)
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        checkpoint.read_checkpoint(path, expected_sha256="0" * 64, expected_size=size)
