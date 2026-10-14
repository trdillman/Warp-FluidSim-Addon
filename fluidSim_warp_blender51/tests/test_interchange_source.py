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


def test_usd_export_follows_the_measured_sampling_recipe():
    """Blender 5.2 writes USD time samples for handler-published points only
    when the same datablock is rewritten in place and the object carries a
    time-dependent modifier (4.0 plan section 7 spike)."""
    source = _source("blender/interchange.py")
    export = ast.get_source_segment(source, _function(source, "export_usd"))
    assert "bpy.ops.wm.usd_export(" in export
    assert "export_animation=True" in export
    assert "selected_objects_only=True" in export
    assert 'evaluation_mode="RENDER"' in export
    # The 5.2 exporter has no frame-range arguments: it walks the scene range.
    assert "_set_frame_range(scene, plan.start, plan.end)" in export
    assert "self.frame_range" in source  # captured and restored
    # One datablock per output for the whole range.
    assert "hold_point_capacity()" in export
    assert "reserve_point_capacity(" in export
    assert "GeometryNodeInputSceneTime" in source
    # Everything temporary is undone in a finally, as for Alembic.
    tries = [node for node in ast.walk(_function(source, "export_usd"))
             if isinstance(node, ast.Try) and node.finalbody]
    assert tries
    final = "\n".join(ast.get_source_segment(source, stmt)
                      for stmt in tries[0].finalbody)
    assert "restore()" in final and "remove()" in final


def test_usd_export_writes_points_not_render_geometry():
    """The add-on's particle render group realizes ico-spheres; USD gets the
    cached points, so the samplers mute and later restore those modifiers."""
    source = _source("blender/interchange.py")
    samplers = source[source.index("class _UsdSamplers"):source.index("def export_usd(")]
    assert "show_render = False" in samplers
    remove = samplers[samplers.index("def remove("):]
    assert ".show_render = shown" in remove


def test_usd_operator_is_registered_and_offered_next_to_alembic():
    source = _source("blender/interchange.py")
    assert 'bl_idname = "fluidsim.export_usd"' in source
    classes = source[source.index("CLASSES = ("):]
    assert "FLUIDSIM_OT_export_usd" in classes[:classes.index(")")]
    stages = _source("blender/ui_stages.py")
    bake = stages[stages.index("class FLUIDSIM_STAGE_PT_bake("):]
    bake = bake[:bake.index("class FLUIDSIM_STAGE_PT_bake_storage")]
    assert bake.count("fluidsim.export_usd") == 2
    playback_branch = bake[
        bake.index("if _playback_only():"):bake.index('_prop(layout, scene, "export_attributes")')]
    assert "fluidsim.export_usd" in playback_branch


def test_usd_export_is_in_the_changelog():
    changelog = _source("CHANGELOG.md")
    section = changelog[changelog.index("## 4.0.0"):]
    section = section[section.index("### Changes"):]
    section = section[:section.index("\n### ", 1)]
    assert "USD" in section


def test_simulate_stage_and_status_strip_show_the_playback_only_text():
    stages = _source("blender/ui_stages.py")
    simulate = stages[stages.index("class FLUIDSIM_STAGE_PT_simulate("):]
    simulate = simulate[:simulate.index("class FLUIDSIM_STAGE_PT_look")]
    assert "PLAYBACK_ONLY_TEXT" in simulate
    panels = _source("panels.py")
    assert "Playback only — CUDA unavailable" in panels
    assert "runtime_ui_state" in panels
