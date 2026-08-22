from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fluidsim_testpkg.io.checkpoint import (
    CheckpointPayload,
    capture_solver,
    read_checkpoint,
    restore_solver,
    write_checkpoint_atomic,
)


def _payload() -> CheckpointPayload:
    return CheckpointPayload(
        frame=12,
        sim_time=0.5,
        settings_signature="settings",
        source_signature="sources",
        arrays={
            "particle_position": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "particle_velocity": np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
            "particle_age": np.array([0.25], dtype=np.float32),
            "whitewater_position": np.zeros((0, 3), dtype=np.float32),
            "whitewater_velocity": np.zeros((0, 3), dtype=np.float32),
            "whitewater_type": np.zeros(0, dtype=np.int32),
        },
        scalars={"whitewater_emit_seed": 12345, "st_step_seed": 1729},
        source_state={"initial_fired": ["Emitter"], "emit_seed": 77},
    )


def test_checkpoint_round_trip(tmp_path: Path):
    payload = _payload()
    path = tmp_path / "checkpoint_000012.npz"
    digest = write_checkpoint_atomic(path, payload)
    restored = read_checkpoint(
        path,
        expected_sha256=digest,
        expected_size=path.stat().st_size,
        expected_frame=12,
        expected_settings_signature="settings",
        expected_source_signature="sources",
    )
    assert restored.frame == 12
    assert restored.scalars == payload.scalars
    assert restored.source_state == payload.source_state
    np.testing.assert_array_equal(
        restored.arrays["particle_position"], payload.arrays["particle_position"]
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_checksum_mismatch_is_rejected(tmp_path: Path):
    path = tmp_path / "checkpoint.npz"
    write_checkpoint_atomic(path, _payload())
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(path, expected_sha256="0" * 64)


def test_size_mismatch_is_rejected(tmp_path: Path):
    path = tmp_path / "checkpoint.npz"
    write_checkpoint_atomic(path, _payload())
    with pytest.raises(ValueError, match="size"):
        read_checkpoint(path, expected_size=path.stat().st_size + 1)


def test_nonfinite_array_is_rejected():
    payload = _payload()
    arrays = dict(payload.arrays)
    arrays["particle_age"] = np.array([np.nan], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        CheckpointPayload(
            payload.frame,
            payload.sim_time,
            payload.settings_signature,
            payload.source_signature,
            arrays,
            payload.scalars,
            payload.source_state,
        ).validated()


def test_object_array_is_rejected():
    payload = _payload()
    arrays = dict(payload.arrays)
    arrays["particle_bad"] = np.array([{"bad": True}], dtype=object)
    with pytest.raises(ValueError, match="object dtype"):
        CheckpointPayload(
            payload.frame,
            payload.sim_time,
            payload.settings_signature,
            payload.source_signature,
            arrays,
            payload.scalars,
            payload.source_state,
        ).validated()


def test_particle_channels_must_align():
    payload = _payload()
    arrays = dict(payload.arrays)
    arrays["particle_age"] = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="unaligned particle"):
        CheckpointPayload(
            payload.frame,
            payload.sim_time,
            payload.settings_signature,
            payload.source_signature,
            arrays,
            payload.scalars,
            payload.source_state,
        ).validated()


def test_capture_and_restore_use_solver_owned_seams():
    class FakeSolver:
        frame = 3
        sim_time = 0.125

        def __init__(self):
            self.restored = None

        def export_checkpoint_state(self):
            return {
                "arrays": {"grid_leaf_coords": np.zeros((0, 3), dtype=np.int32)},
                "scalars": {"frame": 3, "sim_time": 0.125, "seed": 4},
                "source_state": {"initial_fired": []},
            }

        def import_checkpoint_state(self, state):
            self.restored = state

    solver = FakeSolver()
    payload = capture_solver(
        solver, settings_signature="settings", source_signature="sources"
    )
    assert payload.frame == 3
    assert payload.scalars == {"seed": 4}
    restore_solver(solver, payload)
    assert solver.restored["scalars"]["frame"] == 3
    assert solver.restored["source_state"] == {"initial_fired": []}


def test_corrupt_npz_is_rejected(tmp_path: Path):
    path = tmp_path / "checkpoint.npz"
    path.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="could not be read"):
        read_checkpoint(path)


def test_whitewater_type_checkpoint_rejects_reserved_value():
    payload = _payload()
    arrays = dict(payload.arrays)
    arrays["whitewater_position"] = np.zeros((1, 3), dtype=np.float32)
    arrays["whitewater_velocity"] = np.zeros((1, 3), dtype=np.float32)
    arrays["whitewater_type"] = np.array([3], dtype=np.int32)
    with pytest.raises(ValueError, match="whitewater type"):
        CheckpointPayload(
            payload.frame, payload.sim_time, payload.settings_signature,
            payload.source_signature, arrays, payload.scalars, payload.source_state,
        ).validated()


def test_whitewater_type_checkpoint_round_trip():
    payload = _payload()
    arrays = dict(payload.arrays)
    arrays["whitewater_position"] = np.zeros((3, 3), dtype=np.float32)
    arrays["whitewater_velocity"] = np.zeros((3, 3), dtype=np.float32)
    arrays["whitewater_type"] = np.array([0, 1, 2], dtype=np.int32)
    typed = CheckpointPayload(
        payload.frame, payload.sim_time, payload.settings_signature,
        payload.source_signature, arrays, payload.scalars, payload.source_state,
    ).validated()
    np.testing.assert_array_equal(typed.arrays["whitewater_type"], [0, 1, 2])
