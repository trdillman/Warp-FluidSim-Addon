import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.io.cache import read_frame, write_frame


class _PicklePayload:
    def __init__(self, marker_path):
        self.marker_path = marker_path

    def __reduce__(self):
        return (open, (self.marker_path, "w"))


def test_read_frame_blocks_pickle_object_arrays(tmp_path):
    cache_path = tmp_path / "fluidsim_00001.npz"
    marker_path = tmp_path / "pickle_was_loaded.txt"

    np.savez_compressed(
        cache_path,
        positions=np.array([_PicklePayload(str(marker_path))], dtype=object),
    )

    with pytest.raises(ValueError, match="format_version"):
        read_frame(str(tmp_path), 1)
    assert not marker_path.exists()


def test_read_frame_returns_none_for_corrupt_npz(tmp_path):
    cache_path = tmp_path / "fluidsim_00002.npz"
    cache_path.write_bytes(os.urandom(32))

    assert read_frame(str(tmp_path), 2) is None


def test_write_frame_stores_format_version_and_read_frame_loads_it(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "bpy",
        SimpleNamespace(path=SimpleNamespace(abspath=lambda value: value)),
    )
    solver = SimpleNamespace(
        frame=3,
        settings=SimpleNamespace(surface_enabled=False),
        particle_positions=lambda: np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        particle_velocities=lambda: np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        whitewater_positions=lambda: np.zeros((0, 3), dtype=np.float32),
    )
    scene = SimpleNamespace(fluidsim=SimpleNamespace(cache_path=str(tmp_path)))
    sess = SimpleNamespace(solver=solver)

    written = write_frame(scene, sess)
    loaded = read_frame(str(tmp_path), 3)

    assert written == str(tmp_path / "fluidsim_00003.npz")
    assert int(loaded["format_version"]) == 1
    np.testing.assert_array_equal(loaded["positions"], solver.particle_positions())


def test_read_frame_rejects_unsupported_format_version(tmp_path):
    cache_path = tmp_path / "fluidsim_00004.npz"
    np.savez_compressed(cache_path, format_version=np.array(2, dtype=np.int32))

    with pytest.raises(ValueError, match="Unsupported cache format_version 2"):
        read_frame(str(tmp_path), 4)


def test_read_frame_rejects_missing_format_version(tmp_path):
    cache_path = tmp_path / "fluidsim_00005.npz"
    np.savez_compressed(cache_path, positions=np.zeros((0, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="missing format_version"):
        read_frame(str(tmp_path), 5)


def test_write_frame_uses_same_directory_temp_and_atomic_replace(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "bpy",
        SimpleNamespace(path=SimpleNamespace(abspath=lambda value: value)),
    )
    scene = SimpleNamespace(fluidsim=SimpleNamespace(cache_path=str(tmp_path)))
    solver = SimpleNamespace(
        frame=6,
        settings=SimpleNamespace(surface_enabled=False),
        particle_positions=lambda: np.zeros((0, 3), dtype=np.float32),
        particle_velocities=lambda: np.zeros((0, 3), dtype=np.float32),
        whitewater_positions=lambda: np.zeros((0, 3), dtype=np.float32),
    )
    sess = SimpleNamespace(solver=solver)
    calls = {}
    real_savez_compressed = np.savez_compressed
    real_replace = os.replace

    def fake_save(path, **kwargs):
        calls["save_path"] = path
        assert os.path.dirname(path) == str(tmp_path)
        assert os.path.basename(path) != "fluidsim_00006.npz"
        real_savez_compressed(path, **kwargs)

    def fake_replace(src, dst):
        calls["replace"] = (src, dst)
        assert os.path.dirname(src) == os.path.dirname(dst) == str(tmp_path)
        real_replace(src, dst)

    monkeypatch.setattr("fluidSim_warp_blender51.io.cache.np.savez_compressed", fake_save)
    monkeypatch.setattr("fluidSim_warp_blender51.io.cache.os.replace", fake_replace)

    written = write_frame(scene, sess)

    assert written == str(tmp_path / "fluidsim_00006.npz")
    assert calls["replace"][1] == written
