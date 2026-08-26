import os
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender.cache_identity import source_signature
from fluidSim_warp_blender51.io.cache import (
    available_frames,
    compact_settings_segments,
    read_frame,
    read_sequence_metadata,
    resolve_frame_range,
    settings_record,
    validate_contiguous_frames,
    write_frame,
    write_sequence_metadata,
)


class _SourceObject(dict):
    def __init__(self):
        super().__init__(fluidsim_persistent_id="stored-source-id")
        properties = {
            "role": SimpleNamespace(type="ENUM", is_array=False),
            "source_status_message": SimpleNamespace(type="STRING", is_array=False),
        }
        self.name = "Source"
        self.name_full = "Source"
        self.library = None
        self.type = "EMPTY"
        self.data = SimpleNamespace(points=())
        self.matrix_world = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.fluidsim = SimpleNamespace(
            role="SOURCE",
            source_status_message="",
            bl_rna=SimpleNamespace(properties=properties),
        )


def test_source_signature_ignores_runtime_presentation_status():
    source = _SourceObject()
    scene = SimpleNamespace(objects=(source,))
    before = source_signature(scene)

    source.fluidsim.source_status_message = "Emitted 42 particles"

    assert source_signature(scene) == before


def test_source_signature_routes_identity_through_persistent_source_owner(monkeypatch):
    from fluidSim_warp_blender51.blender import scene_inputs

    source = _SourceObject()
    source.pop("fluidsim_persistent_id")
    scene = SimpleNamespace(objects=(source,))
    monkeypatch.setattr(scene_inputs, "persistent_source_key", lambda _obj: "key-A")
    first = source_signature(scene)
    monkeypatch.setattr(scene_inputs, "persistent_source_key", lambda _obj: "key-B")

    assert source_signature(scene) != first


def test_compact_settings_segments_accepts_json_string_frame_keys():
    records = {
        "10": {"signature": "first"},
        "11": {"signature": "first"},
        "12": {"signature": "second"},
    }

    assert compact_settings_segments(records) == [
        {"frame_start": 10, "frame_end": 11, "signature": "first"},
        {"frame_start": 12, "frame_end": 12, "signature": "second"},
    ]


def test_compact_settings_segments_rejects_duplicate_normalized_frames():
    records = {
        1: {"signature": "first"},
        "1": {"signature": "second"},
    }

    with pytest.raises(ValueError, match="Duplicate settings record for frame 1"):
        compact_settings_segments(records)


def test_read_sequence_metadata_treats_non_object_json_as_corrupt(tmp_path):
    (tmp_path / "fluidsim_sequence.json").write_text(
        '["not", "metadata"]', encoding="utf-8")

    assert read_sequence_metadata(str(tmp_path)) is None


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
        settings=SimpleNamespace(surface_enabled=False, voxel_size=0.1),
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
    assert str(loaded["settings_signature"]) == settings_record(
        solver.settings)["signature"]
    np.testing.assert_array_equal(loaded["positions"], solver.particle_positions())
    with zipfile.ZipFile(written) as archive:
        assert archive.infolist()
        assert all(item.compress_type == zipfile.ZIP_DEFLATED
                   for item in archive.infolist())


def test_write_frame_explicit_timeline_index_overrides_solver_counter(
        tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "bpy",
        SimpleNamespace(path=SimpleNamespace(abspath=lambda value: value)),
    )
    solver = SimpleNamespace(
        frame=1,
        settings=SimpleNamespace(surface_enabled=False, voxel_size=0.1),
        particle_positions=lambda: np.zeros((0, 3), dtype=np.float32),
        particle_velocities=lambda: np.zeros((0, 3), dtype=np.float32),
        whitewater_positions=lambda: np.zeros((0, 3), dtype=np.float32),
    )
    scene = SimpleNamespace(fluidsim=SimpleNamespace(cache_path=str(tmp_path)))
    written = write_frame(scene, SimpleNamespace(solver=solver), frame=2)

    assert written == str(tmp_path / "fluidsim_00002.npz")
    assert read_frame(str(tmp_path), 2) is not None
    assert read_frame(str(tmp_path), 1) is None


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


def test_resolve_frame_range_defaults_to_scene_range():
    scene = SimpleNamespace(frame_start=12, frame_end=18)

    assert resolve_frame_range(
        scene,
        use_custom=False,
        frame_start=1,
        frame_end=2,
    ) == (12, 18)


def test_resolve_frame_range_uses_custom_override_and_rejects_inverted_range():
    scene = SimpleNamespace(frame_start=1, frame_end=10)

    assert resolve_frame_range(
        scene,
        use_custom=True,
        frame_start=3,
        frame_end=5,
    ) == (3, 5)
    with pytest.raises(ValueError, match="end frame"):
        resolve_frame_range(
            scene,
            use_custom=True,
            frame_start=6,
            frame_end=5,
        )


def test_sequence_metadata_round_trips_frame_range(tmp_path):
    path = write_sequence_metadata(str(tmp_path), 4, 6, source="simulation")

    metadata = read_sequence_metadata(str(tmp_path))

    assert path == str(tmp_path / "fluidsim_sequence.json")
    assert metadata["format_version"] == 1
    assert metadata["frame_start"] == 4
    assert metadata["frame_end"] == 6
    assert metadata["frame_count"] == 3
    assert metadata["source"] == "simulation"


def test_sequence_metadata_reports_mixed_settings_boundaries(tmp_path):
    first = settings_record(SimpleNamespace(cfl=2.0, flip_ratio=0.95))
    second = settings_record(SimpleNamespace(cfl=1.0, flip_ratio=0.5))

    write_sequence_metadata(
        str(tmp_path), 1, 4, source="simulation",
        settings_records={1: first, 2: first, 3: second, 4: second})
    metadata = read_sequence_metadata(str(tmp_path))

    assert metadata["mixed_settings"] is True
    assert metadata["settings_signature"] is None
    assert metadata["settings_segments"] == [
        {"frame_start": 1, "frame_end": 2, "signature": first["signature"]},
        {"frame_start": 3, "frame_end": 4, "signature": second["signature"]},
    ]
    assert metadata["settings_baseline"] == {
        "signature": first["signature"], "settings": first["settings"]}
    assert "settings_snapshots" not in metadata
    assert metadata["settings_changes"][second["signature"]] == {
        "changed": {"cfl": 1.0, "flip_ratio": 0.5}, "removed": []}


def test_write_sequence_metadata_rejects_duplicate_normalized_frame_keys(tmp_path):
    first = settings_record(SimpleNamespace(cfl=2.0))
    second = settings_record(SimpleNamespace(cfl=1.0))

    with pytest.raises(
            ValueError, match="Duplicate settings record for frame 1"):
        write_sequence_metadata(
            str(tmp_path), 1, 1,
            settings_records={1: first, "1": second})


def test_settings_record_encodes_nonfinite_values_as_standard_json(tmp_path):
    record = settings_record(SimpleNamespace(
        finite=1.0, nan=float("nan"), positive=float("inf"),
        negative=float("-inf")))

    write_sequence_metadata(
        str(tmp_path), 1, 1, settings_records={1: record})
    metadata = read_sequence_metadata(str(tmp_path))

    assert metadata["settings_baseline"]["settings"] == {
        "finite": 1.0,
        "nan": {"__fluidsim_nonfinite_float__": "NaN"},
        "negative": {"__fluidsim_nonfinite_float__": "-Infinity"},
        "positive": {"__fluidsim_nonfinite_float__": "Infinity"},
    }


def test_validate_contiguous_frames_reports_missing_frames(tmp_path):
    for frame in (1, 3):
        np.savez_compressed(
            tmp_path / f"fluidsim_{frame:05d}.npz",
            format_version=np.array(1, dtype=np.int32),
        )

    assert available_frames(str(tmp_path)) == [1, 3]
    assert validate_contiguous_frames(str(tmp_path), 1, 3) == [2]
