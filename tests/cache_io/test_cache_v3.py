"""Cache v3 acceptance tests (SPEC-cache-v3).

Every new manifest/frame field is required; a 2.x cache and a v1 checkpoint are
refused with a clear message; the per-frame source stamp is a pure function of
identity, world matrix and geometry; and a resume scan reports the first frame
whose animation changed.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fluidSim_warp_blender51.io import cache, cache_schema, checkpoint
from fluidSim_warp_blender51.io.cache_schema import (
    CacheManifestV3,
    FrameEntry,
    LegacyCacheFormatError,
    ResumeDecision,
    load_manifest,
    manifest_path,
    sha256_file,
    write_manifest_atomic,
)

REFUSAL_2X = (
    "This cache was baked by FluidSim 2.x and cannot be played by 3.0. "
    "Re-bake, or open it in 2.1.1."
)
PACKAGE = Path(cache_schema.__file__).resolve().parents[1]

GRID_SPEC = {
    "resolution_divisions": 64,
    "voxel_size": 0.0625,
    "domain_size": [4.0, 4.0, 4.0],
    "leaf_dim": 8,
}
NARROW_OFF = {"enabled": False, "band_voxels": 0.0}
NARROW_ON = {"enabled": True, "band_voxels": 3.0}
PLATFORM = {
    "os": "Windows-11",
    "gpu_name": "NVIDIA RTX",
    "driver": "560.94",
    "warp": "1.17.0",
}
NEW_MANIFEST_FIELDS = (
    "source_schema_version", "transfer_mode", "grid_spec", "narrow_band", "platform")
NEW_FRAME_FIELDS = ("source_frame_stamp", "whitewater_count", "particle_count")


def _stamp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(*, narrow_band=None, frame_end: int = 10) -> CacheManifestV3:
    return CacheManifestV3.new(
        solver_version="3.0.0",
        scene_key="scene",
        settings_signature="settings",
        source_signature="sources",
        frame_start=1,
        frame_end=frame_end,
        checkpoint_interval=1,
        checkpoint_format_version=checkpoint.CHECKPOINT_FORMAT_VERSION,
        source_schema_version=1,
        transfer_mode="FLIP",
        grid_spec=dict(GRID_SPEC),
        narrow_band=dict(narrow_band or NARROW_OFF),
        platform=dict(PLATFORM),
        timestamp="2026-10-08T10:00:00Z",
    )


def _entry(frame: int = 1, **changes) -> FrameEntry:
    values = dict(
        frame=frame,
        frame_file=f"frames/frame_{frame:06d}.npz",
        frame_sha256="a" * 64,
        frame_size=100,
        checkpoint_file=f"checkpoints/checkpoint_{frame:06d}.npz",
        checkpoint_sha256="b" * 64,
        checkpoint_size=200,
        completed_at="2026-10-08T10:00:01Z",
        solver_frame=frame,
        simulation_time=frame / 24.0,
        settings_signature="settings",
        source_signature="sources",
        source_frame_stamp=_stamp(f"frame-{frame}"),
        whitewater_count=0,
        particle_count=3,
    )
    values.update(changes)
    return FrameEntry(**values)


def _committed(count: int = 2) -> CacheManifestV3:
    manifest = _manifest()
    for frame in range(1, count + 1):
        manifest = manifest.with_committed_frame(
            _entry(frame), timestamp="2026-10-08T10:00:02Z")
    return manifest


# --------------------------------------------------------------- versions ----

def test_format_versions_are_bumped():
    assert cache_schema.FORMAT_VERSION == 3
    assert cache.CACHE_FORMAT_VERSION == 3
    assert checkpoint.CHECKPOINT_FORMAT_VERSION == 2
    assert cache.SCENE_KEY_VERSION == "v2"


def test_source_schema_version_constant_is_declared():
    tree = ast.parse((PACKAGE / "blender" / "sources.py").read_text(encoding="utf-8"))
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCE_SCHEMA_VERSION"
                for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert values == [1]


def test_a_2x_scene_key_can_never_match_a_3x_key():
    settings = {"voxel_size": 0.1}
    bundle = cache.signature_bundle_from_records(settings, ())
    legacy = hashlib.sha256(
        f"settings={bundle['settings_signature']}|"
        f"sources={bundle['source_signature']}".encode("utf-8")).hexdigest()
    assert bundle["scene_key"] != legacy


# ------------------------------------------------------- manifest round trip ----

def test_v3_manifest_round_trips(tmp_path: Path):
    path = tmp_path / "fluidsim_cache.json"
    manifest = _committed(2)
    write_manifest_atomic(path, manifest)

    loaded = load_manifest(path)
    assert loaded == manifest
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["format_version"] == 3
    assert raw["checkpoint_format_version"] == 2
    for name in NEW_MANIFEST_FIELDS:
        assert name in raw, name
    assert raw["grid_spec"] == GRID_SPEC
    assert raw["frames"]["1"]["source_frame_stamp"] == _stamp("frame-1")
    assert raw["frames"]["2"]["particle_count"] == 3
    assert loaded.frames[2].source_frame_stamp == _stamp("frame-2")


@pytest.mark.parametrize("field", NEW_MANIFEST_FIELDS)
def test_manifest_missing_any_new_field_fails_from_dict(field):
    payload = _committed(1).to_dict()
    del payload[field]
    with pytest.raises(ValueError, match=f"missing=.*{field}"):
        CacheManifestV3.from_dict(payload)


@pytest.mark.parametrize("field", NEW_FRAME_FIELDS)
def test_frame_entry_missing_any_new_field_fails_from_dict(field):
    payload = _committed(1).to_dict()
    del payload["frames"]["1"][field]
    with pytest.raises(ValueError, match=f"missing=.*{field}"):
        CacheManifestV3.from_dict(payload)
    entry = dict(payload["frames"]["1"])
    with pytest.raises(ValueError, match=f"missing=.*{field}"):
        FrameEntry.from_dict(entry)


def test_new_fields_have_no_defaults():
    for cls, names in ((CacheManifestV3, NEW_MANIFEST_FIELDS),
                       (FrameEntry, NEW_FRAME_FIELDS)):
        fields = cls.__dataclass_fields__
        for name in names:
            assert fields[name].default is dataclasses.MISSING, name
            assert fields[name].default_factory is dataclasses.MISSING, name


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_schema_version", 0, "source_schema_version"),
        ("source_schema_version", True, "source_schema_version"),
        ("transfer_mode", "PIC", "transfer_mode"),
        ("transfer_mode", "", "transfer_mode"),
        ("grid_spec", {**GRID_SPEC, "leaf_dim": 16}, "leaf_dim"),
        ("grid_spec", {**GRID_SPEC, "resolution_divisions": 0}, "resolution_divisions"),
        ("grid_spec", {**GRID_SPEC, "voxel_size": 0.0}, "voxel_size"),
        ("grid_spec", {**GRID_SPEC, "domain_size": [4.0, 4.0]}, "domain_size"),
        ("grid_spec", {"voxel_size": 0.1}, "grid_spec"),
        ("narrow_band", {"enabled": True, "band_voxels": 0.0}, "band_voxels"),
        ("narrow_band", {"enabled": "yes", "band_voxels": 0.0}, "narrow_band"),
        ("narrow_band", {"enabled": False}, "narrow_band"),
        ("platform", {"os": "Windows-11"}, "platform"),
        ("platform", {**PLATFORM, "gpu_name": ""}, "gpu_name"),
    ],
)
def test_invalid_new_manifest_values_are_rejected(field, value, message):
    payload = _committed(1).to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        CacheManifestV3.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_frame_stamp", "not-a-hash", "source_frame_stamp"),
        ("source_frame_stamp", "A" * 64, "source_frame_stamp"),
        ("whitewater_count", -1, "whitewater_count"),
        ("particle_count", -1, "particle_count"),
        ("particle_count", 1.5, "particle_count"),
    ],
)
def test_invalid_new_frame_values_are_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        _entry(**{field: value}).validate()


def test_checkpoint_format_in_manifest_must_match_the_reader():
    payload = _committed(1).to_dict()
    payload["checkpoint_format_version"] = 1
    with pytest.raises(ValueError, match="checkpoint_format_version"):
        CacheManifestV3.from_dict(payload)


# ------------------------------------------------------------- 2.x refusal ----

def _write_2x_cache(base: Path) -> None:
    """A minimal 2.x (format 2) manifest directory."""
    (base / "frames").mkdir(parents=True)
    (base / "checkpoints").mkdir()
    entry = {
        "frame": 1,
        "frame_file": "frames/frame_000001.npz",
        "frame_sha256": "a" * 64,
        "frame_size": 10,
        "checkpoint_file": "checkpoints/checkpoint_000001.npz",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_size": 10,
        "completed_at": "2026-09-01T10:00:01Z",
        "solver_frame": 1,
        "simulation_time": 0.04,
        "settings_signature": "settings",
        "source_signature": "sources",
    }
    manifest = {
        "format_version": 2,
        "checkpoint_format_version": 1,
        "solver_version": "2.1.0",
        "scene_key": "scene",
        "settings_signature": "settings",
        "source_signature": "sources",
        "frame_start": 1,
        "frame_end": 5,
        "checkpoint_interval": 1,
        "created_at": "2026-09-01T10:00:00Z",
        "updated_at": "2026-09-01T10:00:01Z",
        "effect_signature": "",
        "effect_inventory": [],
        "frames": {"1": entry},
    }
    (base / "fluidsim_cache.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_2x_cache_directory_is_refused_with_the_rebake_message(tmp_path: Path):
    _write_2x_cache(tmp_path)

    assert cache.detect_cache_format(tmp_path) == 2
    with pytest.raises(LegacyCacheFormatError) as caught:
        load_manifest(manifest_path(tmp_path))
    assert str(caught.value) == REFUSAL_2X
    assert isinstance(caught.value, ValueError)
    with pytest.raises(ValueError) as caught:
        cache.read_frame(str(tmp_path), 1)
    assert REFUSAL_2X in str(caught.value)
    with pytest.raises(ValueError) as caught:
        cache.read_sequence_metadata(str(tmp_path))
    assert REFUSAL_2X in str(caught.value)
    assert cache.available_frames(str(tmp_path)) == []


def test_2x_refusal_message_is_the_spec_wording():
    assert cache_schema.LEGACY_2X_CACHE_MESSAGE == REFUSAL_2X
    assert cache.LEGACY_2X_CACHE_MESSAGE == REFUSAL_2X


def test_unknown_format_numbers_are_not_called_2x(tmp_path: Path):
    path = tmp_path / "fluidsim_cache.json"
    payload = _committed(1).to_dict()
    payload["format_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported cache format 99") as caught:
        load_manifest(path)
    assert not isinstance(caught.value, LegacyCacheFormatError)


def test_v3_directory_detects_as_current_and_1_0_still_detects_as_1(tmp_path: Path):
    current = tmp_path / "current"
    current.mkdir()
    write_manifest_atomic(current / "fluidsim_cache.json", _committed(1))
    assert cache.detect_cache_format(current) == 3

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "fluidsim_sequence.json").write_text('{"format_version": 1}', encoding="utf-8")
    assert cache.detect_cache_format(legacy) == 1
    assert cache.detect_cache_format(tmp_path / "missing") == 0


# --------------------------------------------------- v1 checkpoint refusal ----

def _checkpoint_arrays(**extra):
    arrays = {"particle_positions": np.zeros((2, 3), dtype=np.float32)}
    arrays.update(extra)
    return arrays


def _write_raw_checkpoint(path: Path, *, version: int, arrays: dict) -> None:
    metadata = {
        "checkpoint_format_version": version,
        "frame": 1,
        "sim_time": 0.0,
        "settings_signature": "settings",
        "source_signature": "sources",
        "effect_signature": "",
        "effect_inventory": [],
        "scalars": {},
        "source_state": {},
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        },
    }
    np.savez(path, **{checkpoint._META_KEY: np.asarray(json.dumps(metadata)), **arrays})


def test_v1_checkpoint_is_refused_not_migrated(tmp_path: Path):
    path = tmp_path / "checkpoint_000001.npz"
    _write_raw_checkpoint(path, version=1, arrays=_checkpoint_arrays())
    with pytest.raises(ValueError, match="unsupported checkpoint format 1"):
        checkpoint.read_checkpoint(path)


def test_v2_checkpoint_round_trips_with_empty_reserved_slots(tmp_path: Path):
    path = tmp_path / "checkpoint_000001.npz"
    payload = checkpoint.CheckpointPayload(
        frame=1, sim_time=0.0, settings_signature="settings",
        source_signature="sources", arrays=_checkpoint_arrays())
    checkpoint.write_checkpoint_atomic(path, payload)

    loaded = checkpoint.read_checkpoint(path)
    assert loaded.format_version == 2
    assert loaded.arrays["particle_affine"].shape == (0, 9)
    assert loaded.arrays["particle_affine"].dtype == np.float32
    assert loaded.arrays["narrow_band_seed_state"].shape == (0,)
    np.testing.assert_array_equal(
        loaded.arrays["particle_positions"], np.zeros((2, 3), dtype=np.float32))


def test_v2_checkpoint_without_the_reserved_slots_is_rejected(tmp_path: Path):
    path = tmp_path / "checkpoint_000001.npz"
    _write_raw_checkpoint(path, version=2, arrays=_checkpoint_arrays())
    with pytest.raises(ValueError, match="particle_affine|narrow_band_seed_state"):
        checkpoint.read_checkpoint(path)


def test_misaligned_nonempty_particle_affine_is_rejected(tmp_path: Path):
    payload = checkpoint.CheckpointPayload(
        frame=1, sim_time=0.0, settings_signature="settings",
        source_signature="sources",
        arrays=_checkpoint_arrays(particle_affine=np.zeros((5, 9), dtype=np.float32)))
    with pytest.raises(ValueError, match="unaligned particle"):
        payload.validated()


# ------------------------------------------------------ liquid_sdf payload ----

def _sdf(leaves: int):
    return {
        "liquid_sdf_sparse": np.zeros((leaves, 512), dtype=np.float32),
        "liquid_sdf_leaf_coords": np.zeros((leaves, 3), dtype=np.int32),
        "liquid_sdf_voxel_size": np.array(0.0625 if leaves else 0.0, dtype=np.float32),
    }


def _frame_npz(*, frame: int = 1, sdf=None, stamp=None, particles: int = 3) -> dict:
    sdf = _sdf(0) if sdf is None else sdf
    zero3 = np.zeros((particles, 3), dtype=np.float32)
    column = np.zeros(particles, dtype=np.float32)
    return {
        "format_version": np.array(3, dtype=np.int32),
        "frame": np.array(frame, dtype=np.int32),
        "solver_frame": np.array(frame, dtype=np.int32),
        "simulation_time": np.array(frame / 24.0, dtype=np.float64),
        "settings_signature": np.array("settings"),
        "source_signature": np.array("sources"),
        "scene_key": np.array("scene"),
        "source_frame_stamp": np.array(stamp or _stamp(f"frame-{frame}")),
        "positions": zero3, "velocities": zero3.copy(),
        "particle_age": column, "particle_life": column.copy(),
        "particle_temperature": column.copy(), "particle_viscosity": column.copy(),
        "particle_density": column.copy(), "particle_time_offset": column.copy(),
        "whitewater": np.zeros((0, 3), dtype=np.float32),
        "whitewater_velocities": np.zeros((0, 3), dtype=np.float32),
        "whitewater_age": np.zeros(0, dtype=np.float32),
        "whitewater_life": np.zeros(0, dtype=np.float32),
        **sdf,
    }


def _write_v3_cache(base: Path, payload: dict, *, narrow_band=None,
                    particle_count: int = 3) -> None:
    (base / "frames").mkdir(parents=True)
    (base / "checkpoints").mkdir()
    frame_path = base / "frames" / "frame_000001.npz"
    np.savez(frame_path, **payload)
    checkpoint_path = base / "checkpoints" / "checkpoint_000001.npz"
    checkpoint_path.write_bytes(b"checkpoint")
    entry = _entry(
        1,
        frame_sha256=sha256_file(frame_path), frame_size=frame_path.stat().st_size,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_size=checkpoint_path.stat().st_size,
        particle_count=particle_count)
    manifest = _manifest(narrow_band=narrow_band).with_committed_frame(entry)
    write_manifest_atomic(manifest_path(base), manifest)


def test_v3_frame_with_disabled_narrow_band_reads_back_with_empty_sdf(tmp_path: Path):
    _write_v3_cache(tmp_path, _frame_npz())
    data = cache.read_frame(str(tmp_path), 1)
    assert data is not None
    assert data["liquid_sdf_sparse"].shape == (0, 512)
    assert data["liquid_sdf_leaf_coords"].shape == (0, 3)
    assert cache.available_frames(str(tmp_path)) == [1]
    assert cache.read_sequence_metadata(str(tmp_path))["format_version"] == 3


def test_enabled_narrow_band_with_empty_liquid_sdf_arrays_is_rejected(tmp_path: Path):
    _write_v3_cache(tmp_path, _frame_npz(sdf=_sdf(0)), narrow_band=NARROW_ON)
    with pytest.raises(ValueError, match="liquid_sdf"):
        cache.read_frame(str(tmp_path), 1)


def test_enabled_narrow_band_with_populated_liquid_sdf_arrays_is_accepted(tmp_path: Path):
    _write_v3_cache(tmp_path, _frame_npz(sdf=_sdf(2)), narrow_band=NARROW_ON)
    data = cache.read_frame(str(tmp_path), 1)
    assert data["liquid_sdf_sparse"].shape == (2, 512)


def test_disabled_narrow_band_with_nonempty_liquid_sdf_arrays_is_rejected(tmp_path: Path):
    _write_v3_cache(tmp_path, _frame_npz(sdf=_sdf(2)), narrow_band=NARROW_OFF)
    with pytest.raises(ValueError, match="liquid_sdf"):
        cache.read_frame(str(tmp_path), 1)


@pytest.mark.parametrize("missing", [
    "liquid_sdf_sparse", "liquid_sdf_leaf_coords", "liquid_sdf_voxel_size",
    "source_frame_stamp"])
def test_frame_payload_missing_a_required_v3_key_is_rejected_not_defaulted(
        tmp_path: Path, missing):
    payload = _frame_npz()
    del payload[missing]
    _write_v3_cache(tmp_path, payload)
    with pytest.raises(ValueError, match=missing):
        cache.read_frame(str(tmp_path), 1)


def test_frame_payload_stamp_must_match_the_manifest_entry(tmp_path: Path):
    _write_v3_cache(tmp_path, _frame_npz(stamp=_stamp("somebody-else")))
    with pytest.raises(ValueError, match="source_frame_stamp"):
        cache.read_frame(str(tmp_path), 1)


def test_manifest_particle_count_must_match_the_payload(tmp_path: Path):
    _write_v3_cache(tmp_path, _frame_npz(particles=3), particle_count=4)
    with pytest.raises(ValueError, match="particle_count"):
        cache.read_frame(str(tmp_path), 1)


def test_validate_liquid_sdf_names_the_rule():
    cache.validate_liquid_sdf(_frame_npz(sdf=_sdf(0)), enabled=False)
    cache.validate_liquid_sdf(_frame_npz(sdf=_sdf(3)), enabled=True)
    with pytest.raises(ValueError, match="narrow band is enabled"):
        cache.validate_liquid_sdf(_frame_npz(sdf=_sdf(0)), enabled=True)
    bad = _frame_npz(sdf=_sdf(3))
    bad["liquid_sdf_leaf_coords"] = np.zeros((2, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="leaf"):
        cache.validate_liquid_sdf(bad, enabled=True)


# ----------------------------------------------------------- frame stamping ----

IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _record(identity="paddle", *, matrix=IDENTITY, verts=None, properties=None,
            name_full=None):
    verts = np.arange(9, dtype=np.float32) if verts is None else verts
    return {
        "identity": identity,
        "name_full": name_full or identity.title(),
        "library": "",
        "role": "COLLIDER",
        "properties": properties or {"role": "COLLIDER", "friction": 0.5},
        "matrix_world": matrix,
        "geometry": {
            "kind": "MESH", "vertex_count": 3,
            "vertex_bytes": verts.tobytes(), "polygon_count": 1},
    }


def _moved(dx):
    return (IDENTITY[0][:3] + (dx,), IDENTITY[1], IDENTITY[2], IDENTITY[3])


def test_frame_stamp_is_a_sha256_and_is_deterministic():
    stamp = cache.source_frame_stamp_from_records((_record(),))
    assert len(stamp) == 64 and stamp == stamp.lower()
    assert cache.source_frame_stamp_from_records((_record(),)) == stamp


def test_frame_stamp_changes_with_matrix_geometry_and_identity():
    base = cache.source_frame_stamp_from_records((_record(),))
    assert cache.source_frame_stamp_from_records(
        (_record(matrix=_moved(0.25)),)) != base
    assert cache.source_frame_stamp_from_records(
        (_record(verts=np.arange(9, dtype=np.float32) + 1.0),)) != base
    assert cache.source_frame_stamp_from_records((_record(identity="other"),)) != base


def test_frame_stamp_excludes_properties_and_display_names():
    base = cache.source_frame_stamp_from_records((_record(),))
    assert cache.source_frame_stamp_from_records(
        (_record(properties={"role": "COLLIDER", "friction": 0.9}),)) == base
    assert cache.source_frame_stamp_from_records(
        (_record(name_full="Renamed"),)) == base
    # ...whereas the static signature does see properties.
    assert cache.source_signature_from_records(
        (_record(properties={"role": "COLLIDER", "friction": 0.9}),)
    ) != cache.source_signature_from_records((_record(),))


def test_frame_stamp_is_independent_of_record_order_and_covers_every_source():
    a, b = _record("a"), _record("b", matrix=_moved(1.0))
    assert cache.source_frame_stamp_from_records((a, b)) == \
        cache.source_frame_stamp_from_records((b, a))
    assert cache.source_frame_stamp_from_records((a, b)) != \
        cache.source_frame_stamp_from_records((a,))


def test_a_scene_without_sources_has_one_stable_stamp():
    assert cache.source_frame_stamp_from_records(()) == \
        cache.source_frame_stamp_from_records([])


# -------------------------------------------------------- resume animation ----

def test_resume_decision_carries_the_first_changed_frame():
    decision = ResumeDecision(False, None, "SOURCE_ANIMATION_CHANGED", "m",
                              first_changed_frame=7)
    assert decision.first_changed_frame == 7
    assert ResumeDecision(True, 3, "RESUME_READY", "m").first_changed_frame is None


def test_first_changed_source_frame_is_none_for_an_unchanged_animation():
    manifest = _committed(5)
    assert cache_schema.first_changed_source_frame(
        manifest, lambda frame: manifest.frames[frame].source_frame_stamp) is None


def test_first_changed_source_frame_finds_the_earliest_difference():
    manifest = _committed(6)
    changed = {4, 5, 6}

    def current(frame):
        return (_stamp("moved") if frame in changed
                else manifest.frames[frame].source_frame_stamp)

    assert cache_schema.first_changed_source_frame(manifest, current) == 4
    # A single changed middle frame is found even though the last still matches.
    changed = {3}
    assert cache_schema.first_changed_source_frame(manifest, current) == 3
    changed = {1}
    assert cache_schema.first_changed_source_frame(manifest, current) == 1


def test_first_changed_source_frame_with_no_committed_frames_is_none():
    assert cache_schema.first_changed_source_frame(_manifest(), lambda f: "x") is None


def test_truncated_before_drops_the_changed_frames_and_keeps_the_rest():
    manifest = _committed(6)
    cut = manifest.truncated_before(4)
    assert cut.committed_frames == (1, 2, 3)
    assert cut.frame_start == manifest.frame_start
    assert cut.frames[3] == manifest.frames[3]
    assert manifest.committed_frames == (1, 2, 3, 4, 5, 6)  # original untouched
    assert manifest.truncated_before(manifest.frame_start).committed_frames == ()
    assert manifest.truncated_before(99) == manifest
    with pytest.raises(ValueError, match="before the first frame"):
        manifest.truncated_before(0)
