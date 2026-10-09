"""The bake manifest stamps the transfer mode the solver actually ran."""

from __future__ import annotations

from types import SimpleNamespace

import sys

import pytest

from fluidSim_warp_blender51.blender import production_bake
from fluidSim_warp_blender51.blender.production_bake import ProductionBakeBackend
from fluidSim_warp_blender51.io.cache_schema import (
    SUPPORTED_TRANSFER_MODES,
    CacheManifestV3,
)
from fluidSim_warp_blender51.sim.enums import ParticleSampleType

PLATFORM = {
    "os": "Windows-11", "gpu_name": "NVIDIA RTX", "driver": "12.9", "warp": "1.17.0"}


def _backend(sample_type):
    solver = SimpleNamespace(
        settings=SimpleNamespace(sample_type=int(sample_type)),
        device="cuda:0",
        quality=SimpleNamespace(
            unpadded_voxel_dimensions=(32, 32, 32), voxel_size=0.125,
            domain_size=(4.0, 4.0, 4.0)),
    )
    return SimpleNamespace(
        signatures={
            "scene_key": "scene", "settings_signature": "settings",
            "source_signature": "sources"},
        checkpoint_interval=1,
        effect_signature="",
        effect_inventory=(),
        sess=SimpleNamespace(solver=solver),
    )


@pytest.mark.parametrize(
    ("sample_type", "stamp"),
    [("PIC", "PIC"), ("FLIP", "FLIP"), ("APIC", "APIC")],
)
def test_manifest_transfer_mode_follows_sample_type(
        monkeypatch, sample_type, stamp):
    monkeypatch.setattr(production_bake, "_platform_record", lambda _s: dict(PLATFORM))
    # blender.sources imports bpy; the manifest only needs its schema constant.
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.sources",
        SimpleNamespace(SOURCE_SCHEMA_VERSION=1))
    manifest = ProductionBakeBackend._new_manifest(
        _backend(ParticleSampleType[sample_type]), 1, 5)
    assert manifest.transfer_mode == stamp
    CacheManifestV3.from_dict(manifest.to_dict())  # the stamp is schema-valid


def test_supported_transfer_modes_cover_every_sample_type():
    assert set(SUPPORTED_TRANSFER_MODES) >= {t.name for t in ParticleSampleType}
