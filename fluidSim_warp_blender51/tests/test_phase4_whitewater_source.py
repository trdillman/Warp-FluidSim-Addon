from __future__ import annotations

import json
from pathlib import Path

from fluidsim_testpkg.blender import geometry_nodes, output_publication
from fluidsim_testpkg.sim import attributes
from fluidsim_testpkg.sim.whitewater_types import (
    WHITEWATER_TYPE_SCHEMA_VERSION,
    WhitewaterType,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase4_type_schema_and_public_name_are_stable():
    assert WHITEWATER_TYPE_SCHEMA_VERSION == 1
    assert [int(value) for value in WhitewaterType] == [0, 1, 2]
    spec = attributes.ATTRIBUTE_SPECS["type"]
    assert spec.blender_name == "fluidsim_whitewater_type"
    assert spec.data_type == "INT"
    assert spec.domain == "POINT"
    assert spec.cached is True and spec.checkpointed is True


def test_phase4_keeps_one_simulation_pool_and_persists_type_state():
    source = (ROOT / "sim" / "whitewater.py").read_text(encoding="utf-8")
    assert "self.type = wp.zeros" in source
    assert "self._type2 = wp.zeros" in source
    assert "candidate_types" in source
    assert "self.foam =" not in source
    assert "self.bubbles =" not in source
    assert "self.spray =" not in source
    assert "k_count_whitewater_types" in source
    assert "transition_whitewater_device" in source


def test_phase4_checkpoint_contract_is_explicit_and_fail_closed():
    source = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    assert '"whitewater_type"' in source
    assert '"whitewater_type_schema_version"' in source
    assert "WHITEWATER_TYPE_SCHEMA_VERSION" in source
    assert "unsupported whitewater type checkpoint schema" in source
    checkpoint = (ROOT / "io" / "checkpoint.py").read_text(encoding="utf-8")
    assert "validate_type_values" in checkpoint
    assert "allow_unclassified=False" in checkpoint


def test_phase4_has_combined_and_three_separate_owned_outputs():
    assert output_publication.WHITEWATER_OUTPUT_NAMES == (
        "FluidSim Whitewater",
        "FluidSim Foam",
        "FluidSim Bubbles",
        "FluidSim Spray",
    )
    assert set(geometry_nodes.GROUP_BY_OUTPUT) == {
        "FluidSim Surface",
        "FluidSim Particles",
        *output_publication.WHITEWATER_OUTPUT_NAMES,
    }
    assert geometry_nodes.ASSET_REVISION == "fluidsim-render-p4-v1"


def test_splash_preset_exposes_typed_output_policy():
    payload = json.loads(
        (ROOT / "resources" / "presets" / "splash_whitewater.json").read_text(
            encoding="utf-8"
        )
    )
    settings = payload["settings"]
    assert payload["source_version"] == "1.0.250"
    assert settings["whitewater_enabled"] is True
    assert settings["whitewater_output_mode"] == "BOTH"
    for key in (
        "ww_foam_band",
        "ww_spray_min_speed",
        "ww_type_hysteresis",
        "ww_foam_life_multiplier",
        "ww_bubble_life_multiplier",
        "ww_spray_life_multiplier",
    ):
        assert key in settings


def test_phase4_release_proof_scripts_are_present_but_not_packaged():
    manifest = (ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
    for relative in (
        "release/validate_phase4_warp.py",
        "release/validate_phase4_blender.py",
    ):
        assert (ROOT / relative).is_file()
        assert f'"{relative}"' in manifest
