"""Resolution Divisions: the derived voxel size, tiers, wizard and bake record."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pending_settings_registry import _modules  # noqa: E402

from fluidSim_warp_blender51.sim.quality_contract import (  # noqa: E402
    AuthoredQualitySettings,
    MIN_VOXEL_SIZE,
    divisions_for_voxel_size,
    resolve_quality,
    voxel_size_for_divisions,
)


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _bounds(*size):
    return (0.0, 0.0, 0.0), tuple(float(v) for v in size)


def test_cube_two_metres_at_64_divisions_is_one_thirty_second():
    quality = resolve_quality(
        AuthoredQualitySettings(voxel_size=voxel_size_for_divisions(64, 2.0)),
        _bounds(2, 2, 2))
    assert quality.voxel_size == 0.03125
    assert quality.padded_voxel_dimensions == (64, 64, 64)


def test_non_cube_pads_per_axis_from_the_longest_axis_cell():
    quality = resolve_quality(
        AuthoredQualitySettings(voxel_size=voxel_size_for_divisions(64, 2.0)),
        _bounds(2, 1, 0.5))
    assert quality.voxel_size == 0.03125
    assert quality.padded_voxel_dimensions == (64, 32, 16)


def test_float32_voxel_never_pads_an_extra_leaf():
    """A float32 voxel a hair under longest/divisions must not ceil to divisions+1."""
    failures = []
    for divisions in range(8, 513):
        for longest in (1.0, 2.0, 2.03, 3.0, 4.0, 7.3):
            voxel = _f32(longest / divisions)
            quality = resolve_quality(
                AuthoredQualitySettings(voxel_size=voxel),
                _bounds(longest, longest * 0.5, longest * 0.25))
            if max(quality.unpadded_voxel_dimensions) != divisions:
                failures.append((divisions, longest))
    assert not failures, failures[:5]


def test_default_64_divisions_does_not_pad_to_72():
    voxel = _f32(2.03 / 64)
    quality = resolve_quality(
        AuthoredQualitySettings(voxel_size=voxel), _bounds(2.03, 2.03, 2.03))
    assert quality.padded_voxel_dimensions == (64, 64, 64)


def test_a_real_overshoot_still_adds_a_cell():
    quality = resolve_quality(
        AuthoredQualitySettings(voxel_size=1.0 / 64.5), _bounds(1, 1, 1))
    assert quality.unpadded_voxel_dimensions[0] == 65


def test_divisions_voxel_round_trip_and_floor():
    for divisions in (8, 64, 187, 512):
        voxel = voxel_size_for_divisions(divisions, 3.7)
        assert divisions_for_voxel_size(voxel, 3.7) == divisions
    assert divisions_for_voxel_size(100.0, 2.0) == 8
    assert voxel_size_for_divisions(1, 2.0) == 2.0 / 8
    with pytest.raises(ValueError):
        divisions_for_voxel_size(0.0, 2.0)


def test_min_cell_clamp_reports_the_capped_divisions():
    quality = resolve_quality(
        AuthoredQualitySettings(voxel_size=MIN_VOXEL_SIZE / 10), _bounds(1, 1, 1))
    assert quality.voxel_size == MIN_VOXEL_SIZE
    assert any(
        d.startswith("resolution capped at 10000 divisions by minimum cell size")
        for d in quality.diagnostics), quality.diagnostics


def test_grid_spec_divisions_equal_authored_divisions():
    from fluidSim_warp_blender51.blender import production_bake

    for divisions, size in ((64, (2.03, 1.0, 0.5)), (187, (5.6, 2.0, 1.6)),
                            (8, (3.0, 3.0, 3.0))):
        longest = max(size)
        quality = resolve_quality(
            AuthoredQualitySettings(
                voxel_size=_f32(voxel_size_for_divisions(divisions, longest))),
            _bounds(*size))
        assert production_bake._grid_spec_record(quality)[
            "resolution_divisions"] == divisions


def test_divisions_are_reinitialize_in_both_tier_lists(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)
    assert pending.SETTING_TRANSITION_REGISTRY["resolution_divisions"] is (
        pending.SettingTier.REINITIALIZE)
    assert pending.SETTING_TRANSITION_REGISTRY["voxel_size"] is (
        pending.SettingTier.REINITIALIZE)
    assert "resolution_divisions" in pending._REINIT_SETTING_FIELDS
    assert "resolution_divisions" not in pending._RESET_REGISTRY_FIELDS


def test_scaling_the_domain_raises_reinit_pending(monkeypatch):
    """Trap 3: the derived cell size follows the domain, and the REINIT
    signature already samples the domain bounds, so no RESET entry is needed."""
    _properties, pending, _bridge = _modules(monkeypatch)

    class Group:
        def __getattr__(self, _name):
            return 0

    bounds = {"value": ((0, 0, 0), (2, 2, 2))}
    monkeypatch.setattr(pending, "_signature_owners", lambda _s: (Group(), Group()))
    monkeypatch.setattr(pending, "active_domain_object", lambda _s: None)
    monkeypatch.setattr(pending, "_domain_bounds", lambda _s: bounds["value"])
    monkeypatch.setattr(pending, "_configured_device", lambda: "cpu")
    before = pending._reinit_signature(object())
    bounds["value"] = ((0, 0, 0), (4, 2, 2))
    after = pending._reinit_signature(object())
    assert pending._signature_changes(before, after) == ("domain_bounds",)


def test_voxel_size_is_derived_and_written_through_divisions(monkeypatch):
    properties, _pending, _bridge = _modules(monkeypatch)

    class Matrix:
        def __init__(self, scale):
            self.scale = scale

        def __matmul__(self, corner):
            return SimpleNamespace(**{
                axis: corner[i] * self.scale[i] for i, axis in enumerate("xyz")})

    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    obj = SimpleNamespace(bound_box=corners, matrix_world=Matrix((2.0, 1.0, 0.5)))
    group = SimpleNamespace(resolution_divisions=64, id_data=obj)
    assert properties._get_voxel_size(group) == 4.0 / 64
    properties._set_voxel_size(group, 4.0 / 100)
    assert group.resolution_divisions == 100
    properties._set_voxel_size(group, 1000.0)
    assert group.resolution_divisions == 8
    # No object: the default 4 x 4 x 4 domain.
    bare = SimpleNamespace(resolution_divisions=64, id_data=None)
    assert properties._get_voxel_size(bare) == 4.0 / 64


def test_wizard_restores_its_voxel_after_the_domain_is_resized():
    from fluidSim_warp_blender51.blender import setup_wizard

    updates = []
    domain = SimpleNamespace(scale=(2.0, 2.0, 2.0))

    class Fs:
        """voxel_size derived from the domain scale, as the RNA property is."""

        divisions = 64

        @property
        def voxel_size(self):
            return max(domain.scale) / self.divisions

        @voxel_size.setter
        def voxel_size(self, value):
            assert updates, "view layer must be updated before the write"
            self.divisions = divisions_for_voxel_size(value, max(domain.scale))

    fs = Fs()
    wizard_voxel = fs.voxel_size  # 2 / 64
    scene = SimpleNamespace(fluidsim=SimpleNamespace(domain_object=None))
    context = SimpleNamespace(
        view_layer=SimpleNamespace(update=lambda: updates.append(True)))
    bounds = ((0.0, 0.0, 0.0), (8.0, 4.0, 2.0))
    setup_wizard._fit_domain_keeping_voxel(
        context, scene, domain, fs, bounds, wizard_voxel)
    assert fs.divisions == 256
    assert fs.voxel_size == wizard_voxel


def _domain_object(stored_group, longest=4.0, divisions=64):
    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]

    class Matrix:
        def __matmul__(self, corner):
            return SimpleNamespace(x=corner[0] * longest / 2,
                                   y=corner[1] * longest / 4, z=corner[2] * 0.5)

    stored = {} if stored_group is None else {"fluidsim_domain": stored_group}
    obj = SimpleNamespace(bound_box=corners, matrix_world=Matrix(),
                          bl_system_properties_get=lambda: stored)
    obj.fluidsim_domain = SimpleNamespace(resolution_divisions=divisions, id_data=obj)
    return obj


def test_load_migration_maps_stored_voxel_size_to_divisions(monkeypatch):
    _modules(monkeypatch)
    import importlib

    file_compat = importlib.import_module("fluidSim_warp_blender51.blender.file_compat")
    legacy = _domain_object({"voxel_size": 0.05})
    already = _domain_object({"voxel_size": 0.05, "resolution_divisions": 33},
                             divisions=33)
    fresh = _domain_object({"time_scale": 1.0})
    plain = _domain_object(None)

    assert file_compat.migrate_domain_resolution(
        [legacy, already, fresh, plain], scenes=()) == 1
    assert legacy.fluidsim_domain.resolution_divisions == 80  # 4 m / 0.05 m
    assert already.fluidsim_domain.resolution_divisions == 33
    assert fresh.fluidsim_domain.resolution_divisions == 64
    assert plain.fluidsim_domain.resolution_divisions == 64
    # Stored data is read, never rewritten: a second pass sees the same legacy key
    # and must not run again once divisions are stored (the real RNA stores them).
    legacy.bl_system_properties_get()["fluidsim_domain"]["resolution_divisions"] = 80
    assert file_compat.migrate_domain_resolution([legacy], scenes=()) == 0
