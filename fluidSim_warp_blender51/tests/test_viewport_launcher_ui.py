from __future__ import annotations

from pathlib import Path

from fluidSim_warp_blender51 import registration

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "panels.py"
PREFERENCES = ROOT / "preferences.py"


def _block(source: str, class_name: str) -> str:
    start = source.index(f"class {class_name}")
    end = source.find("\nclass ", start + 1)
    return source[start:] if end < 0 else source[start:end]


def test_viewport_contains_only_a_compact_status_launcher():
    source = PANELS.read_text(encoding="utf-8")
    classes = source.split("CLASSES = (", 1)[1].split(")\n\n\ndef register", 1)[0]
    assert "FLUIDSIM_PT_viewport_status" in classes
    for legacy in (
        "FLUIDSIM_PT_simulation",
        "FLUIDSIM_PT_quality",
        "FLUIDSIM_PT_inputs",
        "FLUIDSIM_PT_physics",
        "FLUIDSIM_PT_surface",
        "FLUIDSIM_PT_output",
        "FLUIDSIM_PT_advanced",
        "FLUIDSIM_PT_snippets",
    ):
        assert legacy not in classes


def test_viewport_status_uses_derived_activity_and_artist_actions():
    source = PANELS.read_text(encoding="utf-8")
    strip = _block(source, "FLUIDSIM_PT_viewport_status")
    assert "FluidSimView3DPanel" in strip.splitlines()[0]
    assert "activity_ui_state" in strip
    # Transport and the pending-edit prompt are shared functions, drawn by the
    # strip and by the header popover alike.
    assert "draw_transport(layout, context, state, runtime_ready)" in strip
    assert "draw_pending_apply(layout, scene)" in strip
    assert "runtime_ui_state()" in strip
    runtime = source.split("def runtime_ui_state", 1)[1].split("def draw_transport", 1)[0]
    assert "runtime_status_for_device" in runtime
    transport = source.split("def draw_transport", 1)[1].split("def draw_pending_apply", 1)[0]
    pending_apply = source.split("def draw_pending_apply", 1)[1].split("class FLUIDSIM_PT_viewport_status", 1)[0]
    assert 'pending["action"] != "NONE"' in pending_apply
    assert 'pending["visible"]' not in pending_apply
    block = strip + transport + pending_apply
    for operator_id in (
        "fluidsim.preview_start",
        "fluidsim.preview_pause",
        "fluidsim.cache_preview_start",
        "fluidsim.apply_changes",
        "fluidsim.restart_simulation",
        "fluidsim.bake_cancel",
    ):
        assert operator_id in block
    # Status and transport only: settings live in the stages, and the old
    # "Open FluidSim Scene Properties" jump is gone from the strip.
    assert "fluidsim.open_scene_properties" not in block
    assert ".prop(" not in block
    assert "Solver not owned by this scene" not in block
    for legacy_manual in (
        "fluidsim.toggle_live",
        "fluidsim.initialize",
        "fluidsim.release",
        "fluidsim.step",
    ):
        assert legacy_manual not in block
    for forbidden in (
        "voxel_size",
        "particles_per_voxel",
        "pressure_solver",
        "cache_path",
        "whitewater_enabled",
        "surface_tension_enabled",
        "fluidsim_effects",
    ):
        assert forbidden not in block


def test_scene_properties_shortcut_uses_an_existing_properties_area():
    source = PANELS.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_OT_open_scene_properties")
    assert 'bl_idname = "fluidsim.open_scene_properties"' in block
    assert 'area.type == "PROPERTIES"' in block
    assert 'area.spaces.active.context = "SCENE"' in block
    assert "No Properties editor is open" in block


def test_preferences_show_runtime_and_conditional_override_but_hide_verbose():
    source = PREFERENCES.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_AP_preferences")
    assert 'box.label(text="Runtime"' in block
    assert 'box.prop(self, "device")' in block
    assert "runtime_status_for_device" in block
    assert 'os.environ.get("FLUIDSIM_ALLOW_EXTERNAL_WARP") == "1"' in block
    assert 'developer.prop(self, "developer_warp_path")' in block
    assert 'prop(self, "verbose")' not in block


def test_viewport_never_exposes_manual_step_or_release():
    source = PANELS.read_text(encoding="utf-8")
    block = _block(source, "FLUIDSIM_PT_viewport_status")
    assert 'operator("fluidsim.step"' not in block
    assert 'operator("fluidsim.release"' not in block


def test_viewport_moves_manual_step_to_advanced_lifecycle():
    viewport_source = PANELS.read_text(encoding="utf-8")
    viewport = _block(viewport_source, "FLUIDSIM_PT_viewport_status")
    assert 'operator("fluidsim.step"' not in viewport

    scene_source = (ROOT / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    lifecycle = _block(scene_source, "FLUIDSIM_PT_scene_lifecycle")
    assert 'bl_options = {"DEFAULT_CLOSED"}' in lifecycle
    assert 'step_once.operator_context = "EXEC_DEFAULT"' in lifecycle
    assert 'step_many.operator_context = "INVOKE_DEFAULT"' in lifecycle


def test_header_popover_shares_the_stage_drawings_and_the_add_on_registers_it():
    header = (ROOT / "blender" / "ui_header.py").read_text(encoding="utf-8")
    popover = _block(header, "FLUIDSIM_PT_header_popover")
    assert 'bl_space_type = "VIEW_3D"' in popover
    assert 'bl_region_type = "HEADER"' in popover
    # One drawing per control: the popover calls the shared stage/strip functions
    # and never authors a property itself, so it cannot drift from the sidebar.
    for shared in (
        "draw_transport(layout, context, state, runtime_ready)",
        "draw_pending_apply(layout, scene)",
        "draw_quality(layout, scene)",
        "draw_simulate_controls(layout, scene, domain)",
        "draw_guided_setup_buttons(box)",
    ):
        assert shared in popover, shared
    assert ".prop(" not in popover
    assert "VIEW3D_HT_header" in header
    assert "header.append(draw_header_button)" in header
    assert "header.remove(draw_header_button)" in header
    workspace = _block(header, "FLUIDSIM_OT_open_workspace")
    assert 'bl_idname = "fluidsim.open_workspace"' in workspace
    # The workspace is the user's own saved layout: never deleted, never created
    # by hand-built screens, and the area_dupli pop-out window stays out.
    assert "workspaces.remove" not in header
    assert "area_dupli" not in header
    assert "bpy.ops.workspace.duplicate()" in workspace

    # Registered after the stages it draws; ``unregister_all`` walks the same
    # table in reverse, so it is torn down before them.
    modules = [step.module for step in registration.STEPS if step.register and step.unregister]
    assert "blender.ui_header" in modules
    assert modules.index("blender.ui_stages") < modules.index("blender.ui_header")
