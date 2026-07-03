import os

import numpy as np

from fluidSim_warp_blender51.io.cache import read_frame


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

    assert read_frame(str(tmp_path), 1) is None
    assert not marker_path.exists()


def test_read_frame_returns_none_for_corrupt_npz(tmp_path):
    cache_path = tmp_path / "fluidsim_00002.npz"
    cache_path.write_bytes(os.urandom(32))

    assert read_frame(str(tmp_path), 2) is None
