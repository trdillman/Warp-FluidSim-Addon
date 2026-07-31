"""Migration and effective-value contracts for source liquid sampling."""

from types import SimpleNamespace

from fluidSim_warp_blender51.blender.scene_inputs import _effective_source_target


def _scene(global_sampling=12):
    return SimpleNamespace(fluidsim=SimpleNamespace(
        voxel_size=0.1, particles_per_voxel=global_sampling))


def _source(**updates):
    values = dict(
        source_ui_schema_version=4,
        source_sampling_mode="INHERIT",
        source_sampling_multiplier=1.0,
        emit_rate=8.0,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_new_schema_inherits_global_sampling():
    assert _effective_source_target(_scene(12), _source()) == 12


def test_multiplier_and_bounds_share_backend_contract():
    assert _effective_source_target(
        _scene(12), _source(
            source_sampling_mode="MULTIPLIER",
            source_sampling_multiplier=2.0)) == 24
    assert _effective_source_target(
        _scene(64), _source(
            source_sampling_mode="MULTIPLIER",
            source_sampling_multiplier=8.0)) == 64
    assert _effective_source_target(
        _scene(12), _source(
            source_sampling_mode="MULTIPLIER",
            source_sampling_multiplier=0.0)) == 0


def test_old_schema_preserves_absolute_emit_rate_without_guessing_inherit():
    old = _source(
        source_ui_schema_version=3,
        source_sampling_mode="INHERIT",
        emit_rate=8.0)
    assert _effective_source_target(_scene(24), old) == 8
    old.emit_rate = 0.0
    assert _effective_source_target(_scene(24), old) == 0


def test_explicit_legacy_absolute_ignores_global_sampling():
    source = _source(
        source_sampling_mode="LEGACY_ABSOLUTE", emit_rate=5.0)
    assert _effective_source_target(_scene(32), source) == 5
