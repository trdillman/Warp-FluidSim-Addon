"""Source-level pins for the Alembic export operator and the playback-only UI."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _function(source: str, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name)


def test_interchange_module_is_in_the_registration_table():
    registration = _source("registration.py")
    assert '"blender.interchange", "register", "unregister"' in registration


def test_export_follows_the_verified_playback_recipe():
    source = _source("blender/interchange.py")
    # Cache playback and live both on; stale cursors cleared; exporter evaluates RENDER.
    assert "set_cache_playback_enabled(scene, True)" in source
    assert "fs.live = True" in source
    assert "clear_cache_cursors()" in source
    assert 'evaluation_mode="RENDER"' in source
    # The frame handler publishes each frame, so it must never be suppressed.
    assert "suppress_frame_handlers(" not in source
    # Properties the 5.2 exporter does not have must not be passed.
    assert "visible_objects_only" not in source
    assert "export_attributes" not in source
    # Restore runs in a finally.
    export = _function(source, "export_alembic")
    assert any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(export))


def test_refusals_cover_every_documented_case():
    source = _source("blender/interchange.py")
    for needle in (
        "no baked cache", "not verified and contiguous", "production bake is running",
        "no FluidSim surface object",
    ):
        assert needle.lower() in source.lower(), needle


def test_bake_stage_offers_alembic_and_playback_only_hides_baking():
    source = _source("blender/ui_stages.py")
    bake = source[source.index("class FLUIDSIM_STAGE_PT_bake("):]
    bake = bake[:bake.index("class FLUIDSIM_STAGE_PT_bake_storage")]
    assert bake.count("fluidsim.export_alembic") == 2
    playback_branch = bake[
        bake.index("if _playback_only():"):bake.index('_prop(layout, scene, "export_attributes")')]
    assert "draw_cache_playback" in playback_branch
    assert "draw_cache_bake" not in playback_branch
    assert "fluidsim.export_frame" not in playback_branch


def test_simulate_stage_and_status_strip_show_the_playback_only_text():
    stages = _source("blender/ui_stages.py")
    simulate = stages[stages.index("class FLUIDSIM_STAGE_PT_simulate("):]
    simulate = simulate[:simulate.index("class FLUIDSIM_STAGE_PT_look")]
    assert "PLAYBACK_ONLY_TEXT" in simulate
    panels = _source("panels.py")
    assert "Playback only — CUDA unavailable" in panels
    assert "runtime_ui_state" in panels
