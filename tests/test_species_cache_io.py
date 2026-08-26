import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.io import cache
from fluidSim_warp_blender51.io.checkpoint import (
    CheckpointPayload,
    capture_solver,
    read_checkpoint,
    restore_solver,
    write_checkpoint_atomic,
)
from fluidSim_warp_blender51.sim.solver import (
    FluidSolver,
    _CHECKPOINT_FIELD_NAMES,
    validate_species_resume_state,
)


INCOMPATIBLE = "species resume state is incompatible; regenerate the cache"
PALETTE = {
    "ids": ["dye_a", "dye_b"],
    "absorption_rgb": [[0.05, 1.75, 2.75], [2.75, 1.25, 0.05]],
    "reference_path_length": 1.0,
}
PALETTE_JSON = json.dumps(PALETTE, sort_keys=True, separators=(",", ":"))
PALETTE_HASH = hashlib.sha256(PALETTE_JSON.encode()).hexdigest()


def _species_scalars(*, version=1, palette_hash=PALETTE_HASH):
    return {
        "species_schema_version": version,
        "species_palette_hash": palette_hash,
        "species_palette_json": PALETTE_JSON,
        "species_ids_json": '["dye_a","dye_b"]',
        "species_settings_json": (
            '{"diffusivity":0.01,"max_diffusion_substeps":16}'
        ),
        "species_cumulative_budget_json": (
            '{"sinks":[0.25,0.5],"sources":[1.25,2.5]}'
        ),
        "species_receipt_json": (
            '{"after":[2.0,1.0],"before":[1.0,0.5],'
            '"diffusion_substeps":2,"limiter_alpha":1.0,'
            '"max_loading":0.75,"min_concentration":0.0,'
            '"residual":[0.0,0.0],"sinks":[0.25,0.5],'
            '"sources":[1.25,1.0]}'
        ),
    }


def _species_arrays(count=2):
    return {
        "particle_represented_volume": np.array(
            [0.5, 0.75][:count], dtype=np.float32),
        "particle_species_concentration": np.array(
            [[0.25, 0.5, 0.0, 0.0], [0.75, 0.0, 0.0, 0.0]][:count],
            dtype=np.float32,
        ),
    }


def test_validate_species_resume_state_accepts_authoritative_enabled_payload():
    arrays = _species_arrays()
    restored = validate_species_resume_state(
        arrays,
        _species_scalars(),
        2,
        enabled=True,
        expected_palette_hash=PALETTE_HASH,
    )

    np.testing.assert_array_equal(restored["represented_volume"], arrays[
        "particle_represented_volume"])
    np.testing.assert_array_equal(restored["concentration"], arrays[
        "particle_species_concentration"])
    assert restored["ids"] == ("dye_a", "dye_b")
    assert restored["cumulative_budget"] == {
        "sources": (1.25, 2.5), "sinks": (0.25, 0.5)}
    assert restored["receipt"]["diffusion_substeps"] == 2


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("shape", np.zeros((2, 3), dtype=np.float32)),
        ("nonfinite", np.array(
            [[np.nan, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32)),
        ("simplex", np.array(
            [[0.75, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32)),
        ("hidden_lane", np.array(
            [[0.25, 0.5, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32)),
    ],
)
def test_validate_species_resume_state_rejects_corrupt_concentration(
        mutation, value):
    arrays = _species_arrays()
    arrays["particle_species_concentration"] = value

    with pytest.raises(ValueError, match=INCOMPATIBLE):
        validate_species_resume_state(
            arrays, _species_scalars(), 2, enabled=True,
            expected_palette_hash=PALETTE_HASH)


@pytest.mark.parametrize("scalars", [
    {},
    _species_scalars(version=99),
    _species_scalars(palette_hash="other-palette"),
])
def test_species_enabled_legacy_unknown_and_palette_mismatch_fail_closed(scalars):
    with pytest.raises(ValueError, match=INCOMPATIBLE):
        validate_species_resume_state(
            _species_arrays(), scalars, 2, enabled=True,
            expected_palette_hash=PALETTE_HASH)


def test_species_disabled_legacy_payload_remains_compatible():
    assert validate_species_resume_state(
        {}, {}, 2, enabled=False, expected_palette_hash=None) is None


def test_checkpoint_file_round_trips_every_species_resume_field(tmp_path):
    exported = {
        "arrays": _species_arrays(),
        "scalars": {"frame": 7, "sim_time": 0.35, **_species_scalars()},
        "source_state": {"schema_version": 1},
    }
    source = SimpleNamespace(
        frame=7,
        sim_time=0.35,
        export_checkpoint_state=lambda: exported,
    )
    payload = capture_solver(
        source, settings_signature="settings", source_signature="sources")
    path = tmp_path / "checkpoint.npz"

    digest = write_checkpoint_atomic(path, payload)
    loaded = read_checkpoint(path, expected_sha256=digest, expected_frame=7)
    candidate = SimpleNamespace(imported=None)
    candidate.import_checkpoint_state = lambda state: setattr(
        candidate, "imported", state)
    restore_solver(candidate, loaded)

    assert candidate.imported["scalars"]["species_schema_version"] == 1
    assert candidate.imported["scalars"]["species_palette_json"] == PALETTE_JSON
    np.testing.assert_array_equal(
        candidate.imported["arrays"]["particle_species_concentration"],
        exported["arrays"]["particle_species_concentration"],
    )


def _minimal_solver_state(*, species_version=1):
    count = 1
    arrays = {
        "grid_leaf_coords": np.empty((0, 3), dtype=np.int32),
        "grid_old_leaf_table": np.full(4, -1, dtype=np.int32),
        "particle_position": np.zeros((count, 3), dtype=np.float32),
        "particle_velocity": np.zeros((count, 3), dtype=np.float32),
        "particle_alive": np.ones(count, dtype=np.int32),
        "particle_time_offset": np.zeros(count, dtype=np.float32),
        "whitewater_position": np.empty((0, 3), dtype=np.float32),
        "whitewater_velocity": np.empty((0, 3), dtype=np.float32),
        "whitewater_age": np.empty(0, dtype=np.float32),
        "whitewater_life": np.empty(0, dtype=np.float32),
        "whitewater_alive": np.empty(0, dtype=np.int32),
        **_species_arrays(count),
    }
    for name in ("age", "life", "temperature", "viscosity", "density"):
        arrays[f"particle_attr_{name}"] = np.zeros(count, dtype=np.float32)
    arrays["particle_attr_represented_volume"] = arrays[
        "particle_represented_volume"].copy()
    arrays["particle_attr_species_concentration"] = arrays[
        "particle_species_concentration"].copy()
    for dotted_name in _CHECKPOINT_FIELD_NAMES:
        arrays[f"field_{dotted_name.replace('.', '_')}"] = np.empty(
            0, dtype=np.float32)
    scalars = {
        "frame": 3,
        "sim_time": 0.15,
        "leaf_count": 0,
        "particle_count": count,
        "whitewater_count": 0,
        "leaf_capacity": 2,
        "particle_capacity": 2,
        "whitewater_capacity": 0,
        "whitewater_emit_seed": 1,
        "st_step_seed": 2,
        "st_previous_dt": 0.05,
        "st_reference_mass_cached": 1.0,
        "particle_optional_channels": (
            "represented_volume,species_concentration"),
        **_species_scalars(version=species_version),
    }
    return {"arrays": arrays, "scalars": scalars, "source_state": {
        "schema_version": 1,
        "emit_seed": 1,
        "initial_fired": [],
        "session_generation": 0,
    }}


def test_invalid_species_checkpoint_is_rejected_before_candidate_mutation():
    candidate = object.__new__(FluidSolver)
    mutations = []
    candidate.settings = SimpleNamespace(species_enabled=True)
    candidate.species = SimpleNamespace(
        palette=SimpleNamespace(palette_hash=PALETTE_HASH))
    candidate.grid = SimpleNamespace(
        capacity=2,
        num_leaf_cells=4,
        restore_topology=lambda *_: mutations.append("topology"),
    )
    candidate.whitewater = SimpleNamespace(capacity=0)
    candidate.particles = SimpleNamespace(
        ATTR_NAMES=("age", "life", "temperature", "viscosity", "density"),
        optional_channels=("represented_volume", "species_concentration"),
        capacity=2,
    )
    candidate.ensure_particle_capacity = lambda *_: mutations.append("growth")
    state = _minimal_solver_state(species_version=99)
    before = copy.deepcopy(state)

    with pytest.raises(ValueError, match=INCOMPATIBLE):
        candidate.import_checkpoint_state(state)

    assert mutations == []
    for name, value in before["arrays"].items():
        np.testing.assert_array_equal(state["arrays"][name], value)


def _frame_payload(species_enabled=True):
    payload = {
        "positions": np.zeros((2, 3), dtype=np.float32),
        **_species_arrays(),
        **{name: np.asarray(value) for name, value in _species_scalars().items()},
    }
    if not species_enabled:
        for name in tuple(payload):
            if name.startswith("species_") or name.startswith("particle_species"):
                payload.pop(name)
        payload.pop("particle_represented_volume", None)
    return payload


def test_cache_frame_species_validation_supports_disabled_legacy_only():
    legacy = _frame_payload(species_enabled=False)

    assert cache.validate_species_frame_payload(
        legacy, species_enabled=False, expected_palette_hash=None) is None
    with pytest.raises(ValueError, match=INCOMPATIBLE):
        cache.validate_species_frame_payload(
            legacy, species_enabled=True,
            expected_palette_hash=PALETTE_HASH)


def test_public_cache_reader_refuses_legacy_frame_for_enabled_species(tmp_path):
    np.savez(
        tmp_path / "fluidsim_00001.npz",
        format_version=np.asarray(1, dtype=np.int32),
        positions=np.empty((0, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match=INCOMPATIBLE):
        cache.read_frame(
            str(tmp_path), 1, species_enabled=True,
            expected_species_palette_hash=PALETTE_HASH)


def test_cache_frame_species_validation_reports_exact_authoritative_counts():
    result = cache.validate_species_frame_payload(
        _frame_payload(), species_enabled=True,
        expected_palette_hash=PALETTE_HASH)

    assert result["particle_count"] == 2
    assert result["represented_volume_count"] == 2
    assert result["concentration_count"] == 2


def test_frame_payload_contains_authoritative_species_not_only_rgb():
    host = SimpleNamespace(
        positions=np.zeros((2, 3), dtype=np.float32),
        velocities=np.zeros((2, 3), dtype=np.float32),
        whitewater=np.empty((0, 3), dtype=np.float32),
        surface_mesh=(np.empty((0, 3), dtype=np.float32),
                      np.empty((0, 3), dtype=np.int32)),
        particle_attr=lambda _name: np.zeros(2, dtype=np.float32),
        whitewater_attr=lambda name: (
            np.empty((0, 3), dtype=np.float32) if name == "velocity"
            else np.empty(0, dtype=np.float32)),
        output_attributes=lambda *_args, **_kwargs: {},
    )
    species_state = {
        "arrays": _species_arrays(),
        "scalars": _species_scalars(),
    }
    solver = SimpleNamespace(
        frame=2,
        settings=SimpleNamespace(surface_enabled=False, voxel_size=0.1),
        stats=SimpleNamespace(),
        species=SimpleNamespace(
            palette=SimpleNamespace(palette_hash=PALETTE_HASH)),
        host_frame=lambda: host,
        export_species_resume_state=lambda count: species_state,
    )

    payload = cache._frame_payload(solver)

    np.testing.assert_array_equal(
        payload["particle_species_concentration"],
        species_state["arrays"]["particle_species_concentration"],
    )
    np.testing.assert_array_equal(
        payload["particle_represented_volume"],
        species_state["arrays"]["particle_represented_volume"],
    )
    assert str(payload["species_palette_hash"].item()) == PALETTE_HASH
    assert "particle_color" not in payload


def test_production_transaction_rejects_frame_checkpoint_species_mismatch(tmp_path):
    frame = _frame_payload()
    frame_path = tmp_path / "frame.npz"
    np.savez(frame_path, **frame)
    checkpoint = CheckpointPayload(
        frame=1,
        sim_time=0.1,
        settings_signature="settings",
        source_signature="sources",
        arrays=_species_arrays(),
        scalars=_species_scalars(),
        source_state={"schema_version": 1},
    )
    altered = dict(checkpoint.arrays)
    altered["particle_represented_volume"] = np.array(
        [9.0, 9.0], dtype=np.float32)
    mismatched = CheckpointPayload(
        frame=checkpoint.frame,
        sim_time=checkpoint.sim_time,
        settings_signature=checkpoint.settings_signature,
        source_signature=checkpoint.source_signature,
        arrays=altered,
        scalars=checkpoint.scalars,
        source_state=checkpoint.source_state,
    )

    with pytest.raises(ValueError, match="species frame/checkpoint mismatch"):
        cache.validate_species_bake_transaction(frame_path, mismatched)
