"""Workflow-oriented FluidSim panels for Blender Scene Properties."""

from __future__ import annotations

from pathlib import Path
import textwrap

import bpy

from .ui_common import (
    FluidSimScenePanel,
    activity_ui_state,
    draw_property,
    draw_property_grid,
    draw_readout_grid,
    draw_status_message,
    pending_ui_state,
)


def _layout(panel):
    layout = panel.layout
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout


def _domain_owner(context):
    """Return the settings datablock the active domain's controls must edit.

    Post-migration the solver reads ``active_domain_settings`` (i.e. the
    designated ``fluidsim.domain_object``'s ``fluidsim_domain``), so the Scene
    Properties domain panels must edit that same owner -- otherwise edits land
    on the stale ``scene.fluidsim`` copy the solver ignores. Re-designating a
    different domain object therefore re-targets these panels, which is how
    "multiple domains" surfaces in this single-scene UI.

    For unmigrated scenes ``active_domain_settings`` returns ``scene.fluidsim``
    unchanged, so every existing single-domain path stays byte-for-byte
    identical.
    """
    from .scene_bridge import active_domain_settings

    scene = context.scene
    return active_domain_settings(scene) or getattr(scene, "fluidsim", None)


def _runtime_ui_state():
    from ..preferences import configured_device
    from ..runtime.warp_runtime import probe_runtime, runtime_status_for_device

    probe = probe_runtime()
    selected = configured_device()
    ready, message = runtime_status_for_device(selected, probe=probe)
    return probe, selected, bool(ready), str(message)


def _draw_wrapped(layout, text, *, icon="NONE", width=72):
    for index, line in enumerate(textwrap.wrap(str(text), width=width) or [""]):
        layout.label(text=line, icon=icon if index == 0 else "NONE")


def _format_bytes(value):
    amount = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def _cache_directory(scene) -> Path | None:
    try:
        raw = str(bpy.path.abspath(scene.fluidsim.cache_path) or "").strip()
    except Exception:
        return None
    if not raw:
        return None
    return Path(raw)


def _draw_capacity_plan(layout, scene):
    from .scene_bridge import capacity_plan_for_scene

    fs = scene.fluidsim
    plan = capacity_plan_for_scene(scene)
    if plan is None:
        layout.label(text="Initialize or calculate safe options to inspect capacity.", icon="INFO")
        layout.operator("fluidsim.apply_safe_settings", text="Calculate Safe Options", icon="MEMORY")
        return
    estimated = int(getattr(plan, "estimated_bytes", 0) or 0)
    budget = int(getattr(plan, "budget_bytes", 0) or 0)
    layout.label(text=f"Estimated peak: {_format_bytes(estimated)}")
    layout.label(text=f"Live safety budget: {_format_bytes(budget)}")
    blockers = tuple(getattr(plan, "blockers", ()))
    for blocker in blockers[:4]:
        _draw_wrapped(layout, getattr(blocker, "message", str(blocker)), icon="ERROR")
    suggestions = tuple(getattr(plan, "suggestions", ()))
    if suggestions:
        layout.label(text="Verified options", icon="CHECKMARK")
        visible = suggestions if getattr(fs, "capacity_show_alternatives", False) else suggestions[:1]
        for suggestion in visible:
            box = layout.box()
            box.label(text=str(getattr(suggestion, "label", "Capacity option")))
            for tradeoff in tuple(getattr(suggestion, "tradeoffs", ()))[:3]:
                _draw_wrapped(box, tradeoff, icon="DOT")
            op = box.operator(
                "fluidsim.apply_capacity_suggestion",
                text="Review & Apply This Option…",
                icon="CHECKMARK",
            )
            op.suggestion_key = str(getattr(suggestion, "key", ""))
    row = layout.row(align=True)
    row.operator("fluidsim.apply_safe_settings", text="Recalculate", icon="FILE_REFRESH")
    if len(suggestions) > 1:
        row.prop(
            fs,
            "capacity_show_alternatives",
            text="Show Alternatives",
            toggle=True,
        )


def _draw_attribute_group(layout, fs, output_kind):
    from ..sim.attributes import OUTPUT_ATTRIBUTE_KEYS, OUTPUT_TOGGLE_NAMES

    grid = layout.grid_flow(
        row_major=True,
        columns=2,
        even_columns=True,
        even_rows=False,
        align=True,
    )
    for key in OUTPUT_ATTRIBUTE_KEYS[output_kind]:
        if output_kind == "whitewater" and key == "type":
            grid.label(text="Type (always published)", icon="CHECKMARK")
            continue
        draw_property(grid, fs, OUTPUT_TOGGLE_NAMES[output_kind][key])


def _surface_enabled(context):
    fs = getattr(getattr(context, "scene", None), "fluidsim", None)
    return bool(fs is not None and getattr(fs, "surface_enabled", False))


def _whitewater_enabled(context):
    fs = getattr(getattr(context, "scene", None), "fluidsim", None)
    return bool(fs is not None and getattr(fs, "whitewater_enabled", False))


def _advanced_visible(context):
    """True unless the user has chosen the Basic interface (D2 two-tier gate).

    Fail-open: a missing settings group or `ui_mode` never hides a panel, so a
    partially-initialised scene keeps the full surface rather than silently
    dropping deep-config panels.
    """
    fs = getattr(getattr(context, "scene", None), "fluidsim", None)
    return bool(fs is None or getattr(fs, "ui_mode", "ADVANCED") == "ADVANCED")


class FLUIDSIM_PT_scene_transport(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Transport"
    bl_idname = "FLUIDSIM_PT_scene_transport"
    bl_order = 0

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        state = activity_ui_state(scene)
        _probe, _selected, runtime_ready, _message = _runtime_ui_state()

        # Primary Start/Stop occupies its own full-width, emphasised row so it
        # never shares space with the step controls below it.
        primary = layout.row(align=True)
        primary.scale_y = 1.3
        if state["id"] == "LIVE_PREVIEW":
            primary.operator("fluidsim.toggle_live", text="Stop Live Preview", icon="PAUSE")
        elif state["id"] == "CACHE_PLAYBACK":
            primary.operator(
                "fluidsim.cache_playback_stop",
                text="Stop Cache Playback",
                icon="PAUSE",
            )
        elif state["id"] == "PRODUCTION_BAKE":
            primary.operator(
                "fluidsim.bake_cancel",
                text="Cancel Production Bake",
                icon="CANCEL",
            )
        else:
            primary.enabled = runtime_ready
            primary.operator("fluidsim.toggle_live", text="Start Live Preview", icon="PLAY")

        # Step One / Step Many are ALWAYS emitted (anti-reflow): greyed out when
        # the solver cannot single-step, never hidden, so buttons never move.
        can_step = bool(
            runtime_ready
            and state["owns_scene"]
            and state["id"] not in {"CACHE_PLAYBACK", "PRODUCTION_BAKE"}
        )
        step_once = layout.row(align=True)
        step_once.enabled = can_step
        step_once.operator_context = "EXEC_DEFAULT"
        op = step_once.operator("fluidsim.step", text="Step One Frame", icon="FRAME_NEXT")
        op.frames = 1
        step_many = layout.row(align=True)
        step_many.enabled = can_step
        step_many.operator_context = "INVOKE_DEFAULT"
        step_many.operator("fluidsim.step", text="Step Multiple Frames…", icon="FORWARD")


class FLUIDSIM_PT_scene_lifecycle(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Lifecycle"
    bl_idname = "FLUIDSIM_PT_scene_lifecycle"
    bl_order = 1

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        state = activity_ui_state(scene)
        _probe, _selected, runtime_ready, _message = _runtime_ui_state()
        owns = state["owns_scene"]

        # Constructive: bring a solver into existence for this scene.
        if not owns and state["id"] == "IDLE":
            init = layout.row(align=True)
            init.enabled = runtime_ready
            init.operator("fluidsim.initialize", text="Initialize Solver", icon="MOD_FLUIDSIM")

        # Destructive: physically separated and red-alerted so they cannot be
        # reached by muscle memory aiming at a safe control. Each carries a
        # one-line statement of what it throws away.
        if owns and state["id"] != "PRODUCTION_BAKE":
            layout.separator()
            danger = layout.column(align=True)
            danger.alert = True
            release = danger.column(align=True)
            release.label(text="Frees the GPU solver; requires Initialize to resume.", icon="X")
            release.operator("fluidsim.release", text="Release Solver", icon="X")
            if state["id"] not in {"CACHE_PLAYBACK", "PRODUCTION_BAKE"}:
                reset = danger.column(align=True)
                reset.label(text="Discards every simulated frame back to the start.", icon="FILE_REFRESH")
                reset.operator("fluidsim.reset", text="Reset Simulation", icon="FILE_REFRESH")


class FLUIDSIM_PT_scene_control(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Status & Runtime"
    bl_idname = "FLUIDSIM_PT_scene_control"
    bl_order = 2

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        fs = scene.fluidsim
        state = activity_ui_state(scene)
        from .scene_bridge import current_error

        row = layout.row(align=True)
        row.label(text=state["label"], icon=state["icon"])
        status = str(getattr(fs, "status", "") or "")
        if status:
            _draw_wrapped(layout, status)
        height_error = str(getattr(fs, "height_error", "") or "")
        if height_error:
            draw_status_message(layout, height_error, "ERROR")

        _probe, selected, runtime_ready, runtime_message = _runtime_ui_state()
        runtime = layout.row(align=True)
        runtime.label(
            text=(f"{selected} · {runtime_message}" if selected else runtime_message),
            icon="CHECKMARK" if runtime_ready else "ERROR",
        )
        runtime.operator("fluidsim.refresh_runtime_probe", text="Re-detect GPU" if not runtime_ready else "Re-detect", icon="FILE_REFRESH")

        # Compact one-line live telemetry: last step cost, substep count, and
        # whether the CFL condition forced a clamp. Read-only status only —
        # transport lives in FLUIDSIM_PT_scene_transport, lifecycle in
        # FLUIDSIM_PT_scene_lifecycle.
        step_ms = float(getattr(fs, "stat_step_ms", 0.0) or 0.0)
        substeps = int(getattr(fs, "stat_substeps", 0) or 0)
        cfl_clamped = bool(getattr(fs, "stat_cfl_clamped", False))
        layout.label(
            text=f"{step_ms:.1f} ms · {substeps} substeps · CFL {'clamped' if cfl_clamped else 'ok'}",
            icon="TIME",
        )

        error = current_error(scene)
        if error:
            box = layout.box()
            _draw_wrapped(box, error, icon="ERROR")
            box.operator("fluidsim.clear_error", text="Clear Error", icon="X")


class FLUIDSIM_PT_scene_pending(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Pending Settings"
    bl_idname = "FLUIDSIM_PT_scene_pending"
    bl_parent_id = "FLUIDSIM_PT_scene_control"

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context) and pending_ui_state(context.scene).get("action") != "NONE"

    def draw(self, context):
        layout = _layout(self)
        pending = pending_ui_state(context.scene)
        layout.label(
            text=pending.get("summary", f"{pending.get('count', 0)} staged change(s)"),
            icon="INFO",
        )
        for reason in tuple(pending.get("reasons", ()))[:5]:
            _draw_wrapped(layout, reason, icon="DOT")
        if pending.get("blocked_reason"):
            _draw_wrapped(layout, pending["blocked_reason"], icon="ERROR")
        row = layout.row(align=True)
        if pending.get("action") == "REINIT_PENDING":
            row.operator("fluidsim.apply_pending_reinitialize", text="Apply & Reinitialize", icon="FILE_REFRESH")
        else:
            row.operator("fluidsim.apply_pending_reset", text="Apply & Reset", icon="FILE_REFRESH")
        row.operator("fluidsim.revert_pending", text="Revert Pending Values", icon="LOOP_BACK")


class FLUIDSIM_PT_scene_live_timeline(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Live Timeline Recovery"
    bl_idname = "FLUIDSIM_PT_scene_live_timeline"
    bl_parent_id = "FLUIDSIM_PT_scene_control"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        layout.prop(context.scene.fluidsim, "max_frame_catchup")


class FLUIDSIM_PT_scene_setup(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Setup"
    bl_idname = "FLUIDSIM_PT_scene_setup"
    bl_order = 10

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        # Front door of the two-tier disclosure (design D2): the Interface switch
        # and the Quality target sit at the top of the first panel so every user
        # meets them before any deep-config surface.
        layout.prop(fs, "ui_mode", expand=True)
        layout.prop(fs, "quality")
        if getattr(fs, "setup_mode", ""):
            layout.label(text=f"Current setup: {fs.setup_mode.replace('_', ' ').title()}", icon="MOD_FLUIDSIM")
        layout.operator("fluidsim.setup_scene", text="Guided Setup…", icon="MOD_FLUIDSIM")


class FLUIDSIM_PT_scene_presets(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Product Presets"
    bl_idname = "FLUIDSIM_PT_scene_presets"
    bl_parent_id = "FLUIDSIM_PT_scene_setup"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "product_preset", text="Preset")
        row = layout.row(align=True)
        row.operator("fluidsim.apply_product_preset", text="Apply Selected Preset", icon="PRESET")
        row.operator("fluidsim.save_product_preset", text="Save Current Settings as Preset…", icon="FILE_TICK")
        if fs.active_preset_id:
            layout.label(text=f"Applied: {fs.active_preset_id}", icon="CHECKMARK")


class FLUIDSIM_PT_scene_domain(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Domain & Particles"
    bl_idname = "FLUIDSIM_PT_scene_domain"
    bl_order = 20

    def draw(self, context):
        layout = _layout(self)
        # The domain_object pointer is the scene-level identity selector (which
        # object *is* the domain); the settings it introduces live on that
        # object's datablock, so they follow the active domain owner.
        draw_property(layout, context.scene.fluidsim, "domain_object")
        owner = _domain_owner(context)
        draw_property_grid(
            layout,
            owner,
            (
                "voxel_size",
                "particles_per_voxel",
                "max_particles",
                "auto_grow_particles",
                "topology_expand",
                "open_boundaries",
            ),
        )


class FLUIDSIM_PT_scene_capacity(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Capacity & Safety"
    bl_idname = "FLUIDSIM_PT_scene_capacity"
    bl_parent_id = "FLUIDSIM_PT_scene_domain"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context) and _advanced_visible(context)

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "leaf_capacity")
        layout.prop(fs, "max_whitewater")
        if fs.auto_grow_particles:
            layout.prop(fs, "particle_growth_failure_policy")
        _draw_capacity_plan(layout, context.scene)


class FLUIDSIM_PT_scene_height(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Initial Liquid — Height Fill"
    bl_idname = "FLUIDSIM_PT_scene_height"
    bl_order = 30
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "height_enabled")
        if not fs.height_enabled:
            return
        layout.prop(fs, "height_base")
        layout.prop(fs, "height_back_mode")
        if fs.height_back_mode == "PROFILE":
            layout.prop(fs, "height_profile_image")
            layout.prop(fs, "height_profile_scale")
        layout.prop(fs, "height_emission_mode")
        layout.prop(fs, "height_velocity")
        layout.prop(fs, "height_emit_particles")


class FLUIDSIM_PT_scene_simulation(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Simulation Physics"
    bl_idname = "FLUIDSIM_PT_scene_simulation"
    bl_order = 40
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        # "Advanced Physics" gate (spec §5.3): the whole 12-panel sim_* fan
        # inherits this parent's visibility, so it collapses in Basic mode.
        return FluidSimScenePanel.poll(context) and _advanced_visible(context)

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.label(text=f"{fs.sample_type} transfer · {fs.pressure_solver} pressure", icon="PHYSICS")
        enabled = []
        if fs.viscosity_enabled:
            enabled.append("Viscosity")
        if fs.surface_tension_enabled:
            enabled.append("Surface tension")
        if fs.species_enabled:
            enabled.append("Color mixing")
        if fs.spacetime_enabled:
            enabled.append("ST-FLIP")
        layout.label(text="Active: " + (", ".join(enabled) if enabled else "Base liquid solver"))


class FLUIDSIM_PT_sim_time(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Time & Transfer"
    bl_idname = "FLUIDSIM_PT_sim_time"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        draw_property_grid(layout, fs, ("time_scale", "cfl", "min_substeps", "max_substeps", "sample_type"))
        if fs.sample_type == "FLIP":
            layout.prop(fs, "flip_ratio")
        layout.prop(fs, "trace_mode")
        layout.prop(fs, "count_stiffness")


class FLUIDSIM_PT_sim_gravity(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Gravity"
    bl_idname = "FLUIDSIM_PT_sim_gravity"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "use_scene_gravity")
        if not fs.use_scene_gravity:
            layout.prop(fs, "gravity")


class FLUIDSIM_PT_sim_pressure(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Pressure"
    bl_idname = "FLUIDSIM_PT_sim_pressure"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "pressure_solver")
        layout.prop(fs, "pressure_iterations")
        if fs.pressure_solver == "PCG":
            layout.prop(fs, "pressure_tolerance")
        layout.prop(fs, "pressure_threshold")


class FLUIDSIM_PT_sim_viscosity(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Viscosity"
    bl_idname = "FLUIDSIM_PT_sim_viscosity"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "viscosity_enabled")
        if not fs.viscosity_enabled:
            return
        layout.prop(fs, "viscosity_solver")
        layout.prop(fs, "viscosity_from_particles")
        if not fs.viscosity_from_particles:
            layout.prop(fs, "viscosity_constant")
        layout.prop(fs, "viscosity_iterations")
        if fs.viscosity_solver == "PCG":
            layout.prop(fs, "viscosity_tolerance")


class FLUIDSIM_PT_sim_surface_tension(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Surface Tension"
    bl_idname = "FLUIDSIM_PT_sim_surface_tension"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "surface_tension_enabled")
        if not fs.surface_tension_enabled:
            return
        draw_property_grid(
            layout,
            fs,
            (
                "surface_tension_coefficient",
                "surface_tension_density",
                "surface_tension_band_voxels",
                "surface_tension_safety",
                "surface_tension_smoothing_iterations",
            ),
        )
        # Capillary-stability readbacks are consolidated into the Diagnostics
        # telemetry panel (D4); this control panel stays edit-only.


class FLUIDSIM_PT_sim_color_mixing(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Color Mixing"
    bl_idname = "FLUIDSIM_PT_sim_color_mixing"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "species_enabled")
        if not fs.species_enabled:
            return
        layout.prop(fs, "species_mixing_rate")
        box = layout.box()
        box.label(text="Emitter colors are configured per fluid source.", icon="COLOR")
        box.label(text="Any number of source colors can mix.")


class FLUIDSIM_PT_sim_relax(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Particle Relaxation"
    bl_idname = "FLUIDSIM_PT_sim_relax"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "relax_enabled")
        if fs.relax_enabled:
            layout.prop(fs, "relax_strength")
            layout.prop(fs, "relax_mode")


class FLUIDSIM_PT_sim_limits(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Velocity Limits"
    bl_idname = "FLUIDSIM_PT_sim_limits"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "max_speed")
        if fs.max_speed > 0.0:
            layout.prop(fs, "limit_behavior")
            if fs.limit_behavior == "PERCENT":
                layout.prop(fs, "limit_percent")


class FLUIDSIM_PT_sim_forces(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Global Forces"
    bl_idname = "FLUIDSIM_PT_sim_forces"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "directional_force")
        layout.prop(fs, "omnidirectional_force_center")
        layout.prop(fs, "omnidirectional_force_strength")
        if fs.omnidirectional_force_strength != 0.0:
            layout.prop(fs, "omnidirectional_force_radius")
        layout.prop(fs, "force_diffusion")


class FLUIDSIM_PT_sim_dissipation(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Dissipation"
    bl_idname = "FLUIDSIM_PT_sim_dissipation"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "dissipation_enabled")
        if fs.dissipation_enabled:
            layout.prop(fs, "dissipation_attribute")
            layout.prop(fs, "dissipation_rate")
            layout.prop(fs, "dissipation_target")


class FLUIDSIM_PT_sim_attraction(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Attraction"
    bl_idname = "FLUIDSIM_PT_sim_attraction"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "attraction_enabled")
        if fs.attraction_enabled:
            layout.prop(fs, "attraction_strength")
            layout.prop(fs, "attraction_max_distance")


class FLUIDSIM_PT_sim_spacetime(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Spatiotemporal Processing"
    bl_idname = "FLUIDSIM_PT_sim_spacetime"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "spacetime_enabled")
        if not fs.spacetime_enabled:
            return
        draw_property_grid(
            layout,
            fs,
            (
                "spacetime_seed",
                "spacetime_reference_mass",
                "spacetime_jitter_strength",
                "spacetime_adaptive_jitter",
                "spacetime_phase_eta",
                "spacetime_phase_threshold",
                "spacetime_local_cfl",
                "spacetime_max_local_substeps",
            ),
        )
        layout.prop(fs, "spacetime_variable_projection")
        if fs.spacetime_variable_projection:
            draw_property_grid(
                layout,
                fs,
                (
                    "spacetime_liquid_density",
                    "spacetime_gas_density",
                    "spacetime_min_density",
                ),
            )
        if fs.surface_enabled:
            layout.prop(fs, "spacetime_resync_surface")


class FLUIDSIM_PT_scene_surface(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Surface Reconstruction"
    bl_idname = "FLUIDSIM_PT_scene_surface"
    bl_order = 60
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "surface_enabled")
        # Basic gets a single Smoothing knob (spec §8); the full filter chain
        # lives in the Advanced-only children below.
        if fs.surface_enabled:
            layout.prop(fs, "surface_smooth_iterations", text="Smoothing")


class _SurfaceChild:
    @classmethod
    def poll(cls, context):
        # Every surface child (resolution/cadence, splat, the full filter chain)
        # is deep-config, so the whole set is Advanced-only.  The parent keeps
        # the surface_enabled toggle (and Basic Smoothing surrogate) visible.
        return (
            FluidSimScenePanel.poll(context)
            and _surface_enabled(context)
            and _advanced_visible(context)
        )


class FLUIDSIM_PT_surface_resolution(_SurfaceChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Resolution & Cadence"
    bl_idname = "FLUIDSIM_PT_surface_resolution"
    bl_parent_id = "FLUIDSIM_PT_scene_surface"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "surface_resolution_mode")
        if fs.surface_resolution_mode == "MULTIPLIER":
            layout.prop(fs, "surface_resolution_multiplier")
        elif fs.surface_resolution_mode == "CUSTOM":
            layout.prop(fs, "surface_voxel_size")
        draw_property_grid(
            layout,
            fs,
            (
                "surface_leaf_capacity_scale",
                "surface_max_dense_dim",
                "surface_update_cadence",
                "live_surface_publish_cadence",
            ),
        )


class FLUIDSIM_PT_surface_reconstruct(_SurfaceChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Particle Splat"
    bl_idname = "FLUIDSIM_PT_surface_reconstruct"
    bl_parent_id = "FLUIDSIM_PT_scene_surface"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        draw_property_grid(layout, fs, ("particle_radius_scale", "surface_splat_radius", "surface_build_lowres_sdf"))


class FLUIDSIM_PT_surface_filters(_SurfaceChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Filters"
    bl_idname = "FLUIDSIM_PT_surface_filters"
    bl_parent_id = "FLUIDSIM_PT_scene_surface"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        # D9: the four post-process passes are chunked into their own labeled
        # boxes so the filter chain reads as four sequential sections rather than
        # a flat ten-prop wall.  Each strength knob stays gated on its pass having
        # a non-zero iteration count.
        layout = _layout(self)
        fs = context.scene.fluidsim

        flt = layout.box()
        flt.label(text="Filter", icon="FILTER")
        flt.prop(fs, "surface_filter_type")
        flt.prop(fs, "surface_filter_iterations")
        if fs.surface_filter_iterations > 0:
            flt.prop(fs, "surface_filter_strength")

        dilate = layout.box()
        dilate.label(text="Dilate", icon="FULLSCREEN_ENTER")
        dilate.prop(fs, "surface_dilate_iterations")
        if fs.surface_dilate_iterations > 0:
            dilate.prop(fs, "surface_dilate_strength")

        erode = layout.box()
        erode.label(text="Erode", icon="FULLSCREEN_EXIT")
        erode.prop(fs, "surface_erode_iterations")
        if fs.surface_erode_iterations > 0:
            erode.prop(fs, "surface_erode_strength")

        smooth = layout.box()
        smooth.label(text="Smooth", icon="MOD_SMOOTH")
        smooth.prop(fs, "surface_smooth_type")
        smooth.prop(fs, "surface_smooth_iterations")
        if fs.surface_smooth_iterations > 0:
            smooth.prop(fs, "surface_smooth_strength")


class FLUIDSIM_PT_scene_whitewater(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Whitewater"
    bl_idname = "FLUIDSIM_PT_scene_whitewater"
    bl_order = 70
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "whitewater_enabled")
        # Basic gets on/off + a single Amount knob (spec §8); per-stream emission,
        # classification, lifetime, and dynamics live in the Advanced children.
        if fs.whitewater_enabled:
            layout.prop(fs, "ww_emission_rate", text="Amount")


class _WhitewaterChild:
    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context) and _whitewater_enabled(context)


class _WhitewaterAdvancedChild(_WhitewaterChild):
    """Every whitewater child (per-stream emission, classification, lifetime,
    dynamics) is Advanced-only; Basic keeps just the on/off toggle plus the
    single Amount knob drawn on the parent panel."""

    @classmethod
    def poll(cls, context):
        return _WhitewaterChild.poll(context) and _advanced_visible(context)


class FLUIDSIM_PT_whitewater_emission(_WhitewaterAdvancedChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Emission"
    bl_idname = "FLUIDSIM_PT_whitewater_emission"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        draw_property_grid(
            layout,
            fs,
            (
                "ww_emission_rate",
                "ww_churn_threshold",
                "ww_churn_gain",
                "ww_churn_decay",
                "ww_min_speed",
                "ww_surface_band",
            ),
        )


class FLUIDSIM_PT_whitewater_classification(_WhitewaterAdvancedChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Classification"
    bl_idname = "FLUIDSIM_PT_whitewater_classification"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        draw_property_grid(layout, fs, ("ww_foam_band", "ww_spray_min_speed", "ww_type_hysteresis"))
        layout.label(text="Types are persistent: Foam 0 · Bubble 1 · Spray 2", icon="INFO")


class FLUIDSIM_PT_whitewater_life(_WhitewaterAdvancedChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Lifetime"
    bl_idname = "FLUIDSIM_PT_whitewater_life"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        draw_property_grid(layout, context.scene.fluidsim, ("ww_life_min", "ww_life_max"))


class FLUIDSIM_PT_whitewater_dynamics(_WhitewaterAdvancedChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Dynamics"
    bl_idname = "FLUIDSIM_PT_whitewater_dynamics"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        foam = layout.box()
        foam.label(text="Foam", icon="MOD_FLUIDSIM")
        draw_property_grid(
            foam,
            fs,
            ("ww_foam_advection", "ww_foam_preservation", "ww_foam_density_influence", "ww_foam_life_multiplier"),
        )
        bubbles = layout.box()
        bubbles.label(text="Bubbles", icon="PARTICLES")
        draw_property_grid(bubbles, fs, ("ww_bubble_drag", "ww_bubble_buoyancy", "ww_bubble_life_multiplier"))
        spray = layout.box()
        spray.label(text="Spray", icon="FORCE_WIND")
        draw_property_grid(spray, fs, ("ww_spray_drag", "ww_spray_gravity_scale", "ww_spray_life_multiplier"))


class FLUIDSIM_PT_scene_outputs(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Viewport & Outputs"
    bl_idname = "FLUIDSIM_PT_scene_outputs"
    bl_order = 80

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        surface = fs.liquid_view_mode in {"SURFACE", "SURFACE_PARTICLES"}
        particles = fs.liquid_view_mode in {"PARTICLES", "SURFACE_PARTICLES"}
        active = []
        if surface:
            active.append("Surface")
        if particles:
            active.append("Particles")
        if fs.whitewater_enabled and fs.show_whitewater:
            active.append("Whitewater")
        layout.label(text="Visible: " + (", ".join(active) if active else "None"), icon="HIDE_OFF")
        # Basic headline for the export inventory (spec §8); the 33 per-attribute
        # toggles live in the Advanced-only Persistent Attributes children.
        layout.prop(fs, "export_attributes")


class FLUIDSIM_PT_output_display(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Display"
    bl_idname = "FLUIDSIM_PT_output_display"
    bl_parent_id = "FLUIDSIM_PT_scene_outputs"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "liquid_view_mode")
        layout.prop(fs, "show_overlay", text="Viewport HUD")
        if fs.liquid_view_mode in {"SURFACE", "SURFACE_PARTICLES"} and fs.surface_enabled:
            layout.prop(fs, "surface_attribute_mode")
        if fs.liquid_view_mode in {"PARTICLES", "SURFACE_PARTICLES"}:
            layout.prop(fs, "particle_display_fraction")
            layout.prop(fs, "particle_display_scale")
            layout.prop(fs, "particle_attribute_mode")
        if fs.whitewater_enabled:
            layout.prop(fs, "show_whitewater")
            if fs.show_whitewater:
                layout.prop(fs, "whitewater_output_mode")
                layout.prop(fs, "whitewater_display_scale")
                layout.prop(fs, "whitewater_attribute_mode")


class FLUIDSIM_PT_output_materials(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Materials & Collection"
    bl_idname = "FLUIDSIM_PT_output_materials"
    bl_parent_id = "FLUIDSIM_PT_scene_outputs"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "output_collection_name")
        if fs.liquid_view_mode in {"SURFACE", "SURFACE_PARTICLES"}:
            layout.prop(fs, "surface_material")
        if fs.liquid_view_mode in {"PARTICLES", "SURFACE_PARTICLES"}:
            layout.prop(fs, "particle_material")
        if fs.whitewater_enabled:
            layout.prop(fs, "whitewater_material")
            if fs.whitewater_output_mode in {"SEPARATE", "BOTH"}:
                layout.prop(fs, "foam_material")
                layout.prop(fs, "bubble_material")
                layout.prop(fs, "spray_material")
        row = layout.row(align=True)
        install = row.operator("fluidsim.install_render_nodes", text="Install Render Nodes", icon="GEOMETRY_NODES")
        install.force = False
        update = row.operator("fluidsim.install_render_nodes", text="Update FluidSim Render Nodes…", icon="FILE_REFRESH")
        update.force = True


class FLUIDSIM_PT_output_attributes(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Persistent Attributes"
    bl_idname = "FLUIDSIM_PT_output_attributes"
    bl_parent_id = "FLUIDSIM_PT_scene_outputs"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        # The per-attribute export inventory (surface/particle/whitewater
        # toggles) is Advanced-only; its children inherit this gate.
        return FluidSimScenePanel.poll(context) and _advanced_visible(context)

    def draw(self, context):
        layout = _layout(self)
        layout.label(text="Inventory changes apply after Reset and affect cache identity.", icon="INFO")


class FLUIDSIM_PT_attrs_surface(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Surface Attributes"
    bl_idname = "FLUIDSIM_PT_attrs_surface"
    bl_parent_id = "FLUIDSIM_PT_output_attributes"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context) and _surface_enabled(context)

    def draw(self, context):
        _draw_attribute_group(_layout(self), context.scene.fluidsim, "surface")


class FLUIDSIM_PT_attrs_particles(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Particle Attributes"
    bl_idname = "FLUIDSIM_PT_attrs_particles"
    bl_parent_id = "FLUIDSIM_PT_output_attributes"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        _draw_attribute_group(_layout(self), context.scene.fluidsim, "particles")


class FLUIDSIM_PT_attrs_whitewater(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Whitewater Attributes"
    bl_idname = "FLUIDSIM_PT_attrs_whitewater"
    bl_parent_id = "FLUIDSIM_PT_output_attributes"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context) and _whitewater_enabled(context)

    def draw(self, context):
        _draw_attribute_group(_layout(self), context.scene.fluidsim, "whitewater")


class FLUIDSIM_PT_scene_cache(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Cache & Production Bake"
    bl_idname = "FLUIDSIM_PT_scene_cache"
    bl_order = 90
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        layout.prop(context.scene.fluidsim, "cache_path")


class FLUIDSIM_PT_cache_playback(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Playback"
    bl_idname = "FLUIDSIM_PT_cache_playback"
    bl_parent_id = "FLUIDSIM_PT_scene_cache"

    def draw(self, context):
        layout = _layout(self)
        state = activity_ui_state(context.scene)
        if state["id"] == "CACHE_PLAYBACK":
            layout.operator("fluidsim.cache_playback_stop", text="Stop Cache Playback", icon="PAUSE")
        else:
            layout.operator("fluidsim.cache_playback_start", text="Start Cache Playback", icon="PLAY")
        layout.label(text="Cache playback does not allocate the solver.", icon="INFO")


class FLUIDSIM_PT_cache_range(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Shared Playback / Export Range"
    bl_idname = "FLUIDSIM_PT_cache_range"
    bl_parent_id = "FLUIDSIM_PT_scene_cache"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "cache_use_custom_range")
        if fs.cache_use_custom_range:
            row = layout.row(align=True)
            row.prop(fs, "cache_frame_start")
            row.prop(fs, "cache_frame_end")


class FLUIDSIM_PT_cache_storage(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Storage"
    bl_idname = "FLUIDSIM_PT_cache_storage"
    bl_parent_id = "FLUIDSIM_PT_scene_cache"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "cache_compress")
        layout.prop(fs, "checkpoint_interval")


class FLUIDSIM_PT_cache_bake(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Production Bake"
    bl_idname = "FLUIDSIM_PT_cache_bake"
    bl_parent_id = "FLUIDSIM_PT_scene_cache"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        row = layout.row(align=True)
        row.prop(fs, "bake_start")
        row.prop(fs, "bake_end")
        active = fs.bake_status in {"PREFLIGHT", "INITIALIZING", "RUNNING", "CANCELLING"}
        layout.prop(fs, "bake_progress", text=fs.bake_status.title(), slider=True)
        if fs.bake_message:
            _draw_wrapped(layout, fs.bake_message, icon="INFO")
        actions = layout.row(align=True)
        if active:
            actions.operator("fluidsim.bake_cancel", text="Cancel Bake", icon="CANCEL")
        else:
            from .ui_cache_state import production_bake_action

            presentation = production_bake_action(_cache_directory(context.scene))
            if presentation.action != "START":
                _draw_wrapped(layout, presentation.message, icon="INFO")
            _probe, _selected, runtime_ready, _runtime_message = _runtime_ui_state()
            actions.enabled = runtime_ready
            if presentation.action == "RESUME":
                actions.operator(
                    "fluidsim.bake_resume",
                    text="Resume Verified Bake",
                    icon="RECOVER_LAST",
                )
            else:
                label = (
                    "Restart Production Bake…"
                    if presentation.action == "RESTART"
                    else "Start Production Bake"
                )
                icon = "FILE_REFRESH" if presentation.destructive else "PLAY"
                start = actions.operator("fluidsim.bake_start", text=label, icon=icon)
                start.restart_existing = presentation.destructive
        layout.operator("fluidsim.copy_bake_command", text="Copy Headless Bake Command", icon="COPYDOWN")


class FLUIDSIM_PT_cache_manage(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Manage Cache"
    bl_idname = "FLUIDSIM_PT_cache_manage"
    bl_parent_id = "FLUIDSIM_PT_scene_cache"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        if not FluidSimScenePanel.poll(context):
            return False
        from .cache_manager import is_managed_cache_directory

        path = _cache_directory(context.scene)
        return path is not None and is_managed_cache_directory(path)

    def draw(self, context):
        layout = _layout(self)
        path = _cache_directory(context.scene)
        if path is None:
            return
        layout.label(text=str(path), icon="FILE_FOLDER")
        row = layout.row(align=True)
        row.operator("fluidsim.cache_integrity_scan", text="Scan Integrity", icon="CHECKMARK")
        row.operator("fluidsim.cache_remove_orphans", text="Inspect & Remove Orphans…", icon="TRASH")
        row = layout.row(align=True)
        row.operator("fluidsim.cache_move", text="Move Cache…", icon="FILE_PARENT")
        row.operator("fluidsim.cache_rename", text="Rename Cache…", icon="GREASEPENCIL")
        row.operator("fluidsim.cache_open_folder", text="Open Cache Folder", icon="FILE_FOLDER")
        layout.operator("fluidsim.cache_free", text="Free Entire Cache…", icon="TRASH")


class FLUIDSIM_PT_scene_diagnostics(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Diagnostics"
    bl_idname = "FLUIDSIM_PT_scene_diagnostics"
    bl_order = 100
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        layout.operator("fluidsim.run_validation", text="Run Validation Suite", icon="CHECKMARK")


class FLUIDSIM_PT_scene_runtime_stats(FluidSimScenePanel, bpy.types.Panel):
    # Read-only solver telemetry consolidated here from the former control-panel
    # readbacks (D4): runtime counts, capillary stability, and whitewater
    # admission.  The ``bl_idname`` is preserved (the GUI-acceptance gate resolves
    # this panel by id) but it now parents under Diagnostics, and every value is a
    # label -- no control panel renders a ``stat_*`` readback (spec §11).
    bl_label = "Telemetry"
    bl_idname = "FLUIDSIM_PT_scene_runtime_stats"
    bl_parent_id = "FLUIDSIM_PT_scene_diagnostics"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        if not FluidSimScenePanel.poll(context):
            return False
        state = activity_ui_state(context.scene)
        return bool(state["owns_scene"] or state["id"] == "CACHE_PLAYBACK")

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        draw_readout_grid(
            layout,
            fs,
            (
                ("stat_particles", "Particles"),
                ("stat_voxels", "Voxels"),
                ("stat_whitewater", "Whitewater"),
                ("stat_step_ms", "Step (ms)"),
                ("stat_substeps", "Substeps"),
                ("stat_cfl_clamped", "CFL Clamped"),
            ),
            title="Runtime",
        )
        draw_readout_grid(
            layout,
            fs,
            (
                ("stat_capillary_dt", "Stable Timestep"),
                ("stat_capillary_substeps_required", "Required Substeps"),
                ("stat_substep_limiter", "Selected Limiter"),
                ("stat_capillary_clamped", "Clamped by Max Substeps"),
            ),
            title="Capillary Stability",
        )
        if getattr(fs, "stat_surface_tension_message", ""):
            _draw_wrapped(layout, fs.stat_surface_tension_message, icon="INFO")
        draw_readout_grid(
            layout,
            fs,
            (
                ("stat_whitewater_foam", "Foam"),
                ("stat_whitewater_bubbles", "Bubbles"),
                ("stat_whitewater_spray", "Spray"),
                ("stat_whitewater_invalid_types", "Invalid Types"),
                ("stat_whitewater_required", "Required"),
                ("stat_whitewater_admitted", "Admitted"),
                ("stat_whitewater_dropped", "Dropped"),
                ("stat_whitewater_policy", "Policy"),
                ("stat_whitewater_status", "Status"),
            ),
            title="Whitewater Admission",
        )


class FLUIDSIM_PT_diagnostics_profiling(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Profiling"
    bl_idname = "FLUIDSIM_PT_diagnostics_profiling"
    bl_parent_id = "FLUIDSIM_PT_scene_diagnostics"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "profiling_enabled")
        if fs.profiling_enabled:
            layout.prop(fs, "profiling_sync")
        from .scene_bridge import session
        sess = session()
        if sess.profiler is not None and sess.profiler.history:
            summary = sess.profiler.summary(last_n=30)
            box = layout.box()
            box.label(text="Average phase times", icon="TIME")
            from ..sim.constants import TIMER_PHASE_SLOTS

            phase_names = {
                f"compute_{slot}": phase.replace("_", " ").title()
                for phase, slot in TIMER_PHASE_SLOTS.items()
            }
            for key, label in phase_names.items():
                value = summary.get(key, 0.0)
                if float(value) > 0.01:
                    box.label(text=f"{label}: {float(value):.2f} ms")


class FLUIDSIM_PT_diagnostics_benchmark(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Benchmark"
    bl_idname = "FLUIDSIM_PT_diagnostics_benchmark"
    bl_parent_id = "FLUIDSIM_PT_scene_diagnostics"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "benchmark_frames")
        layout.prop(fs, "benchmark_baseline")
        layout.operator("fluidsim.run_benchmark", text="Run Benchmark…", icon="TIME")
        from .scene_bridge import session
        result = session().benchmark_result
        if result:
            box = layout.box()
            box.label(text=f"Average step: {result['timeStepAverage'] * 1000.0:.1f} ms")
            box.label(text=f"Voxels/s: {result['voxelsPerSecondAverage'] / 1e6:.1f} M")


class FLUIDSIM_PT_diagnostics_report(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Report & Export"
    bl_idname = "FLUIDSIM_PT_diagnostics_report"
    bl_parent_id = "FLUIDSIM_PT_scene_diagnostics"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "report_path")
        row = layout.row(align=True)
        row.operator("fluidsim.write_report", text="Write Diagnostic Report…", icon="TEXT")
        row.operator("fluidsim.export_frame", text="Export Current Frame…", icon="EXPORT")


SCENE_PANEL_CLASSES = (
    FLUIDSIM_PT_scene_transport,
    FLUIDSIM_PT_scene_lifecycle,
    FLUIDSIM_PT_scene_control,
    FLUIDSIM_PT_scene_pending,
    FLUIDSIM_PT_scene_live_timeline,
    FLUIDSIM_PT_scene_setup,
    FLUIDSIM_PT_scene_presets,
    FLUIDSIM_PT_scene_domain,
    FLUIDSIM_PT_scene_capacity,
    FLUIDSIM_PT_scene_height,
    FLUIDSIM_PT_scene_simulation,
    FLUIDSIM_PT_sim_time,
    FLUIDSIM_PT_sim_gravity,
    FLUIDSIM_PT_sim_pressure,
    FLUIDSIM_PT_sim_viscosity,
    FLUIDSIM_PT_sim_surface_tension,
    FLUIDSIM_PT_sim_color_mixing,
    FLUIDSIM_PT_sim_relax,
    FLUIDSIM_PT_sim_limits,
    FLUIDSIM_PT_sim_forces,
    FLUIDSIM_PT_sim_dissipation,
    FLUIDSIM_PT_sim_attraction,
    FLUIDSIM_PT_sim_spacetime,
    FLUIDSIM_PT_scene_surface,
    FLUIDSIM_PT_surface_resolution,
    FLUIDSIM_PT_surface_reconstruct,
    FLUIDSIM_PT_surface_filters,
    FLUIDSIM_PT_scene_whitewater,
    FLUIDSIM_PT_whitewater_emission,
    FLUIDSIM_PT_whitewater_classification,
    FLUIDSIM_PT_whitewater_life,
    FLUIDSIM_PT_whitewater_dynamics,
    FLUIDSIM_PT_scene_outputs,
    FLUIDSIM_PT_output_display,
    FLUIDSIM_PT_output_materials,
    FLUIDSIM_PT_output_attributes,
    FLUIDSIM_PT_attrs_surface,
    FLUIDSIM_PT_attrs_particles,
    FLUIDSIM_PT_attrs_whitewater,
    FLUIDSIM_PT_scene_cache,
    FLUIDSIM_PT_cache_playback,
    FLUIDSIM_PT_cache_range,
    FLUIDSIM_PT_cache_storage,
    FLUIDSIM_PT_cache_bake,
    FLUIDSIM_PT_cache_manage,
    FLUIDSIM_PT_scene_diagnostics,
    FLUIDSIM_PT_scene_runtime_stats,
    FLUIDSIM_PT_diagnostics_profiling,
    FLUIDSIM_PT_diagnostics_benchmark,
    FLUIDSIM_PT_diagnostics_report,
)


def register():
    for cls in SCENE_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(SCENE_PANEL_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
