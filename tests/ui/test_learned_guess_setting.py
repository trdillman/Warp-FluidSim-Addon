"""Learned pressure guess setting: RESET class, drawn under PCG, Live Readout row (4.0 lane N)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from test_operator_panel_behavior import _install_bpy_stub, _reload_module

ADDON = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"


def test_readout_row_only_while_the_setting_is_on(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_stages = _reload_module("fluidSim_warp_blender51.blender.ui_stages")

    def rows(on):
        domain = type("Domain", (), {"narrow_band_enabled": False,
                                     "pressure_learned_guess": on})()
        return dict(ui_stages.simulate_readouts(domain))

    assert "stat_learned_guess_message" not in rows(False)
    assert rows(True)["stat_learned_guess_message"] == "Learned guess"


def test_setting_is_reset_class_and_reaches_the_solver():
    from fluidSim_warp_blender51.blender import pending_settings, solver_settings
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings

    assert "pressure_learned_guess" in pending_settings._RESET_SETTING_FIELDS
    assert "pressure_learned_guess" in pending_settings._RESET_REGISTRY_FIELDS
    assert "pressure_learned_guess" not in pending_settings._LIVE_SETTING_FIELDS
    assert "pressure_learned_guess" in solver_settings.DIRECT_SETTING_FIELDS
    assert SolverSettings().pressure_learned_guess is False


def _property_call(name):
    tree = ast.parse((ADDON / "properties.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == name):
            return node.annotation
    raise AssertionError(f"{name} not declared")


def test_property_is_off_reset_and_has_a_tooltip():
    call = _property_call("pressure_learned_guess")
    kw = {k.arg: k.value for k in call.keywords}
    assert ast.literal_eval(kw["default"]) is False
    assert kw["update"].id == "_request_reset"
    assert "options" not in kw
    assert "resets the solver" in ast.literal_eval(kw["description"])


def test_setting_is_drawn_with_the_pcg_rows():
    # Every setting is reachable from All Simulation Settings; this one sits
    # with the PCG rows and greys out with them.
    drawn = re.compile(r"pcg\.prop\([^)]*[\"']pressure_learned_guess[\"']")
    hits = [p.name for p in (ADDON / "blender").glob("*.py")
            if drawn.search(p.read_text(encoding="utf-8"))]
    assert hits == ["ui_scene.py"]


def test_stat_is_published_and_cleared_on_playback():
    bridge = (ADDON / "blender" / "scene_bridge.py").read_text(encoding="utf-8")
    playback = (ADDON / "blender" / "playback.py").read_text(encoding="utf-8")
    assert "fs.stat_learned_guess_message = " in bridge
    assert 'fs.stat_learned_guess_message = ""' in playback
