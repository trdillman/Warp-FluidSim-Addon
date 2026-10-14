"""Cache v3.1: per-frame rigid-body records and rigid state on resume (4.0 lane R).

docs/specs/4.0/SPEC-rigid-coupling.md: each committed frame carries
``rigid_bodies: [{object_uid, pos, quat, lin_vel, ang_vel}]`` (plus the body's
rest centre of mass, which publication needs to turn the pose into a
``matrix_world``); a v3 cache loads with empty records; a v3.1 resume restores
the body state from the frame's checkpoint.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from fluidSim_warp_blender51.io import cache_schema, checkpoint
from fluidSim_warp_blender51.io.cache_schema import (
    CacheManifestV3,
    FrameEntry,
    RigidBodyRecord,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cache_v3 import _entry, _manifest  # noqa: E402

RECORD = {
    "object_uid": "cork",
    "pos": [0.1, 0.2, 0.3],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "lin_vel": [0.0, 0.0, -0.5],
    "ang_vel": [0.0, 0.25, 0.0],
    "rest_com": [0.1, 0.2, 0.4],
}


def _v3_dict():
    """A manifest as FluidSim 3.x wrote it: no minor version, no records."""
    manifest = _manifest().with_committed_frame(
        _entry(1), timestamp="2026-10-08T10:00:02Z")
    value = json.loads(json.dumps(manifest.to_dict()))
    value.pop("format_minor", None)
    for entry in value["frames"].values():
        entry.pop("rigid_bodies", None)
    return value


def test_v3_cache_loads_with_empty_rigid_records():
    manifest = CacheManifestV3.from_dict(_v3_dict())
    assert manifest.format_minor == 0
    assert manifest.frames[1].rigid_bodies == ()


def test_new_manifests_are_v3_1():
    assert cache_schema.FORMAT_MINOR_VERSION == 1
    assert _manifest().format_minor == 1
    assert _manifest().to_dict()["format_minor"] == 1


def test_frame_without_bodies_writes_no_rigid_key():
    record = _entry(1).as_record()
    assert "rigid_bodies" not in record


def test_rigid_records_round_trip_through_json():
    entry = _entry(1, rigid_bodies=(RigidBodyRecord.from_dict(RECORD),))
    manifest = _manifest().with_committed_frame(
        entry, timestamp="2026-10-08T10:00:02Z")
    loaded = CacheManifestV3.from_dict(json.loads(json.dumps(manifest.to_dict())))
    (body,) = loaded.frames[1].rigid_bodies
    assert body.as_dict() == RECORD
    assert loaded.frames[1] == entry


@pytest.mark.parametrize("change", [
    {"object_uid": ""},
    {"pos": [0.0, 0.0]},
    {"quat": [1.0, 0.0, 0.0]},
    {"lin_vel": [math.nan, 0.0, 0.0]},
    {"ang_vel": "spin"},
    {"extra": 1},
])
def test_malformed_rigid_record_is_refused(change):
    value = {**RECORD, **change}
    with pytest.raises(ValueError):
        _entry(1, rigid_bodies=(RigidBodyRecord.from_dict(value),)).validate()


def test_missing_rigid_record_key_is_refused():
    value = dict(RECORD)
    del value["quat"]
    with pytest.raises(ValueError):
        RigidBodyRecord.from_dict(value)


# ------------------------------------------------------------------ resume ----

def _solver(device, points):
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    return FluidSolver(SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(0.5, 0.5, 0.5),
        voxel_size=1.0 / 48.0, leaf_capacity=1024,
        max_particles=len(points) + 8192, surface_enabled=False,
        whitewater_enabled=False, device=device))


def test_v3_1_resume_restores_rigid_body_state(cuda_device, tmp_path):
    import warp as wp

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solver"))
    from _rigid_harness import H, box_collider, lattice

    points = lattice((0.0, 0.0, 0.0), (0.5, 0.5, 0.22), 0.5 * H)

    def collider():
        return box_collider(
            wp, cuda_device, (0.25, 0.25, 0.26), (0.06, 0.06, 0.04),
            dynamic=True, object_uid="cork", density=300.0, drag=1.0,
            initial_velocity=(0.1, 0.0, -0.2))

    original = _solver(cuda_device, points)
    try:
        original.inject_particles(points)
        body = collider()
        for _ in range(3):
            original.set_colliders([body])
            original.step_frame()
        saved = original.rigid_body_states()
        payload = checkpoint.capture_solver(
            original, settings_signature="settings", source_signature="sources")
        path = tmp_path / "checkpoint_000003.npz"
        digest = checkpoint.write_checkpoint_atomic(path, payload)
        for _ in range(3):
            original.set_colliders([body])
            original.step_frame()
        expected = original.rigid_body_states()
        expected_positions = original.particle_positions()
    finally:
        original.release()

    resumed = _solver(cuda_device, points)
    try:
        checkpoint.restore_solver(resumed, checkpoint.read_checkpoint(
            path, expected_sha256=digest, expected_frame=3))
        body = collider()
        resumed.set_colliders([body])
        restored = resumed.rigid_body_states()
        assert [r["object_uid"] for r in restored] == ["cork"]
        for key in ("pos", "quat", "lin_vel", "ang_vel"):
            assert np.array_equal(restored[0][key], saved[0][key]), key
        for _ in range(3):
            resumed.set_colliders([body])
            resumed.step_frame()
        actual = resumed.rigid_body_states()
        for key in ("pos", "quat", "lin_vel", "ang_vel"):
            assert np.array_equal(actual[0][key], expected[0][key]), key
        assert np.array_equal(resumed.particle_positions(), expected_positions)
    finally:
        resumed.release()


def test_read_frame_carries_the_frame_rigid_records(tmp_path):
    """Cache playback publishes poses from the frame it plays."""
    from dataclasses import replace

    from fluidSim_warp_blender51.io import cache

    sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
    from tiny_cache import build_tiny_cache

    base = build_tiny_cache(tmp_path / "c", frame_count=2)
    manifest_file = cache_schema.manifest_path(base)
    manifest = cache_schema.load_manifest(manifest_file)
    record = RigidBodyRecord.from_dict(RECORD)
    manifest.frames[2] = replace(manifest.frames[2], rigid_bodies=(record,))
    cache_schema.write_manifest_atomic(manifest_file, manifest)

    assert "rigid_bodies" not in cache.read_frame(str(base), 1)
    assert cache.read_frame(str(base), 2)["rigid_bodies"] == (record,)
