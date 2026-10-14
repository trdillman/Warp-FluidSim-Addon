import json
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender.cache_identity import source_signature
from fluidSim_warp_blender51.io.cache import (
    available_frames,
    detect_cache_format,
    read_frame,
    read_sequence_metadata,
    resolve_frame_range,
    settings_record,
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


def _fluidsim_1_0_cache(tmp_path, *, metadata=True, frames=(1,)):
    if metadata:
        (tmp_path / "fluidsim_sequence.json").write_text(
            '{"format_version": 1}', encoding="utf-8")
    for frame in frames:
        np.savez_compressed(
            tmp_path / f"fluidsim_{frame:05d}.npz",
            format_version=np.array(1, dtype=np.int32),
            positions=np.zeros((1, 3), dtype=np.float32))


@pytest.mark.parametrize("metadata,frames", [(True, (1, 3)), (True, ()), (False, (2,))])
def test_fluidsim_1_0_cache_is_refused_with_a_rebake_message(tmp_path, metadata, frames):
    """2.0 reads cache format 2 only; a 1.0 cache must not read as empty."""
    _fluidsim_1_0_cache(tmp_path, metadata=metadata, frames=frames)

    assert detect_cache_format(tmp_path) == 1
    with pytest.raises(ValueError, match="FluidSim 1.0.*bake again"):
        read_frame(str(tmp_path), 1)
    assert read_sequence_metadata(str(tmp_path)) is None
    assert available_frames(str(tmp_path)) == []


def test_directory_without_a_cache_reads_nothing(tmp_path):
    assert detect_cache_format(tmp_path) == 0
    assert read_frame(str(tmp_path), 1) is None
    assert read_sequence_metadata(str(tmp_path)) is None
    assert available_frames(str(tmp_path)) == []
    assert detect_cache_format(tmp_path / "missing") == 0
    assert read_frame(str(tmp_path / "missing"), 1) is None


def test_the_cache_module_no_longer_writes_format_1():
    from fluidSim_warp_blender51.io import cache

    assert cache.CACHE_FORMAT_VERSION == 3
    for name in ("write_frame", "write_sequence_metadata", "compact_settings_segments"):
        assert not hasattr(cache, name), name


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


def test_settings_record_encodes_nonfinite_values_as_standard_json():
    record = settings_record(SimpleNamespace(
        finite=1.0, nan=float("nan"), positive=float("inf"),
        negative=float("-inf")))

    assert json.loads(json.dumps(record["settings"], allow_nan=False)) == {
        "finite": 1.0,
        "nan": {"__fluidsim_nonfinite_float__": "NaN"},
        "negative": {"__fluidsim_nonfinite_float__": "-Infinity"},
        "positive": {"__fluidsim_nonfinite_float__": "Infinity"},
    }


def _pre_d1_signature(settings):
    """The settings signature exactly as caches before owner decision D1 wrote
    it: every public field, the compute device included."""
    import hashlib
    from dataclasses import fields

    from fluidSim_warp_blender51.io.cache import _jsonable_setting

    values = {f.name: _jsonable_setting(getattr(settings, f.name))
              for f in fields(settings)
              if not f.name.startswith("_")
              and f.name != "live_surface_publish_cadence"
              # added in 4.0, after D1: no pre-D1 cache can have hashed it
              and f.name != "pressure_learned_guess"}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_cache_identity_does_not_depend_on_the_compute_device():
    # Review item B16, owner decision D1: changing the GPU preference (or
    # playing on another machine) rejected every cached frame.
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings

    on_0 = settings_record(SolverSettings(device="cuda:0"))
    on_1 = settings_record(SolverSettings(device="cuda:1"))
    assert "device" not in on_0["settings"]
    assert on_0["signature"] == on_1["signature"]


def test_caches_baked_before_d1_still_match_their_settings():
    from fluidSim_warp_blender51.io.cache import settings_signature_matches
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings

    baked = _pre_d1_signature(SolverSettings(device="cuda:0"))
    assert settings_signature_matches(SolverSettings(device="cuda:0"), baked)
    assert settings_signature_matches(SolverSettings(device="cuda:1"), baked)
    assert not settings_signature_matches(
        SolverSettings(device="cuda:0", cfl=SolverSettings().cfl + 1.0), baked)


def test_resume_can_rebuild_a_pre_d1_identity_bundle():
    import hashlib

    from fluidSim_warp_blender51.io.cache import (
        SCENE_KEY_VERSION,
        legacy_signature_bundles,
        signature_bundle_from_records,
    )
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings

    settings = SolverSettings(device="cuda:1")
    current = signature_bundle_from_records(settings, ())
    old_settings = _pre_d1_signature(settings)
    old_key = hashlib.sha256(
        f"scene-key={SCENE_KEY_VERSION}|settings={old_settings}|"
        f"sources={current['source_signature']}".encode("utf-8")).hexdigest()

    assert {"scene_key": old_key, "settings_signature": old_settings,
            "source_signature": current["source_signature"]} in (
        legacy_signature_bundles(current, settings))
