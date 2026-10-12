"""Atomic writers close their temp descriptor exactly once on failure.

``os.fdopen`` owns the descriptor once it returns; closing the number again
after the ``with`` block has closed it can close an unrelated file that
reused the number in threaded Blender.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
from fluidsim_testpkg.blender.command_line import write_json_atomic
from fluidsim_testpkg.io.cache import _write_npz_atomic
from fluidsim_testpkg.io.checkpoint import CheckpointPayload, write_checkpoint_atomic
from fluidsim_testpkg.presets.library import _write_atomic as write_preset_atomic


def _checkpoint_payload() -> CheckpointPayload:
    return CheckpointPayload(
        frame=1,
        sim_time=0.0,
        settings_signature="settings",
        source_signature="sources",
        arrays={
            "particle_position": np.zeros((1, 3), dtype=np.float32),
            "particle_velocity": np.zeros((1, 3), dtype=np.float32),
            "particle_age": np.zeros(1, dtype=np.float32),
            "whitewater_position": np.zeros((0, 3), dtype=np.float32),
            "whitewater_velocity": np.zeros((0, 3), dtype=np.float32),
            "whitewater_type": np.zeros(0, dtype=np.int32),
        },
        scalars={"whitewater_emit_seed": 1},
    )


WRITERS = {
    "cache_npz": lambda path: _write_npz_atomic(path, {"a": np.zeros(2)}),
    "checkpoint": lambda path: write_checkpoint_atomic(path, _checkpoint_payload()),
    "command_line_report": lambda path: write_json_atomic(path, {"a": 1}),
    "preset": lambda path: write_preset_atomic(path, {"a": 1}),
}


def _track_descriptor(monkeypatch):
    descriptors = []
    closes = []
    real_mkstemp = tempfile.mkstemp
    real_close = os.close

    def mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor, name

    def close(descriptor):
        if descriptor in descriptors:
            closes.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(os, "close", close)
    return descriptors, closes


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_failed_write_does_not_close_fdopen_owned_descriptor(
        writer, tmp_path: Path, monkeypatch):
    descriptors, closes = _track_descriptor(monkeypatch)

    def failing_fsync(_descriptor):
        raise OSError("forced write failure")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    target = tmp_path / "out" / "artifact.bin"
    with pytest.raises(OSError, match="forced write failure"):
        WRITERS[writer](target)

    assert len(descriptors) == 1
    # The ``with os.fdopen(...)`` block already closed it; a second close
    # would hit whatever file reused the number.
    assert closes == []
    assert list(target.parent.iterdir()) == []


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_descriptor_is_closed_when_fdopen_never_took_ownership(
        writer, tmp_path: Path, monkeypatch):
    descriptors, closes = _track_descriptor(monkeypatch)

    def failing_fdopen(*_args, **_kwargs):
        raise OSError("forced fdopen failure")

    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    target = tmp_path / "out" / "artifact.bin"
    with pytest.raises(OSError, match="forced fdopen failure"):
        WRITERS[writer](target)

    assert closes == descriptors and len(closes) == 1
    assert list(target.parent.iterdir()) == []
