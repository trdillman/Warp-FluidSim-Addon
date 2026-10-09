"""Narrow-band settings and the liquid-voxel stat agree across every registry."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pending_settings_registry import _modules

from fluidSim_warp_blender51.sim.frame_runtime import FrameStats

ADDON = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
FIELDS = ("narrow_band_enabled", "narrow_band_voxels")


def _domain_annotation(name):
    tree = ast.parse((ADDON / "properties.py").read_text(encoding="utf-8"))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if cls.name != "FLUIDSIM_PG_domain":
            continue
        for node in cls.body:
            if (isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == name):
                return node.annotation
    return None


def _keywords(call):
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
            if kw.arg != "update" and kw.arg != "description"}


def test_domain_properties_declare_the_documented_defaults_and_ranges():
    enabled = _domain_annotation("narrow_band_enabled")
    voxels = _domain_annotation("narrow_band_voxels")
    assert enabled.func.id == "BoolProperty"
    assert _keywords(enabled)["default"] is False
    assert voxels.func.id == "FloatProperty"
    options = _keywords(voxels)
    assert options["default"] == 3.0
    assert (options["min"], options["soft_min"], options["soft_max"]) == (
        1.0, 2.0, 6.0)
    for annotation in (enabled, voxels):
        update = [kw.value.id for kw in annotation.keywords if kw.arg == "update"]
        assert update == ["_request_reset"]


def test_both_fields_are_reset_tier_in_every_registry(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)
    from fluidSim_warp_blender51.blender import solver_settings
    from fluidSim_warp_blender51.presets import schema

    for name in FIELDS:
        assert name in pending._RESET_REGISTRY_FIELDS
        assert name in pending._RESET_SETTING_FIELDS
        assert name in solver_settings.DIRECT_SETTING_FIELDS
        assert name in schema.SETTING_SPECS
        assert name not in schema.SCENE_OWNED_SETTINGS
    assert pending.validate_transition_registry() is True
    solver_settings.validate_setting_registry()


def test_band_width_preset_range_matches_the_authored_hard_minimum():
    from fluidSim_warp_blender51.presets import schema

    spec = schema.SETTING_SPECS["narrow_band_voxels"]
    assert spec.validate("narrow_band_voxels", 1.0) == 1.0
    with pytest.raises(ValueError):
        spec.validate("narrow_band_voxels", 0.5)


def test_liquid_voxel_stat_is_a_scene_presentation_field(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)
    from fluidSim_warp_blender51.presets import schema

    assert "stat_liquid_voxels" in schema.SCENE_OWNED_SETTINGS
    assert "stat_liquid_voxels" in pending._PRESENTATION_FIELDS
    tree = ast.parse((ADDON / "properties.py").read_text(encoding="utf-8"))
    scene_fields = {
        node.target.id
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "FLUIDSIM_PG_scene"
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "stat_liquid_voxels" in scene_fields
    assert _domain_annotation("stat_liquid_voxels") is None


def test_frame_stats_carry_liquid_voxels_defaulting_to_zero():
    stats = FrameStats()
    assert stats.liquid_voxels == 0
    assert stats.as_dict()["liquidVoxels"] == 0


def test_stat_is_written_by_the_live_and_cache_paths():
    bridge = (ADDON / "blender" / "scene_bridge.py").read_text(encoding="utf-8")
    playback = (ADDON / "blender" / "playback.py").read_text(encoding="utf-8")
    stages = (ADDON / "blender" / "ui_stages.py").read_text(encoding="utf-8")
    assert "fs.stat_liquid_voxels = int(getattr(stats, \"liquid_voxels\", 0))" in bridge
    assert "fs.stat_liquid_voxels = 0" in playback
    assert '("stat_liquid_voxels", "Liquid voxels")' in stages


def test_manifest_block_reads_the_setting():
    from fluidSim_warp_blender51.blender.production_bake import (
        _narrow_band_record)

    off = SimpleNamespace(narrow_band_enabled=False, narrow_band_voxels=3.0)
    assert _narrow_band_record(off) == {"enabled": False, "band_voxels": 0.0}
    on = SimpleNamespace(narrow_band_enabled=True, narrow_band_voxels=4.5)
    assert _narrow_band_record(on) == {"enabled": True, "band_voxels": 4.5}
    assert _narrow_band_record(SimpleNamespace()) == {
        "enabled": False, "band_voxels": 0.0}
