"""Workflow-oriented FluidSim panels for Blender Scene Properties.

Transport (Play/Pause/Restart) is the 3D View launcher, and Cache &
Production Bake is the sidebar's Bake & Export stage; neither is repeated here.
The cache draw functions below are shared with that stage.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import bpy

from ..version import VERSION_STRING
from . import file_compat
from .scene_access import active_domain_settings
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


def resolution_readout(scene) -> tuple[str, ...]:
    """Read-only lines for the effective cell size and padded grid.

    The cell size is derived (longest domain axis / Resolution Divisions, then the
    Quality tier), so the artist is shown what the solver will use. Empty when it
    cannot be resolved (no domain, or a bounds read that fails).
    """
    try:
        from ..sim.quality_contract import (
            AuthoredQualitySettings,
            resolve_quality,
            scale_quality_value,
        )
        from .solver_settings import domain_bounds, scene_quality_fidelity

        fs = active_domain_settings(scene)
        if fs is None:
            return ()
        quality = resolve_quality(
            AuthoredQualitySettings(voxel_size=scale_quality_value(
                "voxel_size", float(fs.voxel_size), scene_quality_fidelity(scene))),
            domain_bounds(scene))
    except Exception:  # a label must never break panel drawing
        return ()
    nx, ny, nz = quality.padded_voxel_dimensions
    lines = [
        f"Cell size: {quality.voxel_size:.4g} m",
        f"Grid: {nx} x {ny} x {nz} cells",
    ]
    lines.extend(d for d in quality.diagnostics if d.startswith("resolution capped"))
    return tuple(lines)


def draw_resolution_readout(layout, scene) -> None:
    for line in resolution_readout(scene):
        if line.startswith("resolution"):
            layout.label(text=line, icon="ERROR")
        else:
            layout.label(text=line)


def _domain_owner(context):
    """Return the settings datablock the active domain's controls must edit.

    The solver reads ``active_domain_settings`` (the designated
    ``fluidsim.domain_object``'s ``fluidsim_domain``), so the Scene Properties
    domain panels must edit that same owner. Re-designating a different domain
    object therefore re-targets these panels, which is how "multiple domains"
    surfaces in this single-scene UI.

    ``None`` when no domain object is designated; panels that draw domain
    settings poll on :func:`_domain_visible`.
    """
    return active_domain_settings(getattr(context, "scene", None))


def _domain_visible(context):
    """Poll for panels that draw domain settings: hidden without a domain."""
    return FluidSimScenePanel.poll(context) and _domain_owner(context) is not None


def _runtime_ui_state():
    from ..preferences import configured_device
    from ..runtime.warp_runtime import probe_runtime, runtime_status_for_device

    probe = probe_runtime()
    selected = configured_device()
    ready, message = runtime_status_for_device(selected, probe=probe)
    return probe, selected, bool(ready), str(message)


def _build_identity():
    """Name the loaded build, so a stale install or a dev checkout is visible."""
    if (__package__ or "").startswith("bl_ext."):
        return f"FluidSim {VERSION_STRING} · installed extension"
    return f"FluidSim {VERSION_STRING} · loaded from {Path(__file__).resolve().parents[1]}"


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
        visible = suggestions if fs.capacity_show_alternatives else suggestions[:1]
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

    if fs is None:
        # No domain is designated (an output object can outlive its domain
        # assignment), so there are no domain attribute toggles to draw.
        return
    grid = layout.grid_flow(
        row_major=True,
        columns=2,
        even_columns=True,
        even_rows=False,
        align=True,
    )
    grid.active = bool(fs.export_attributes)
    for key in OUTPUT_ATTRIBUTE_KEYS[output_kind]:
        if output_kind == "whitewater" and key == "type":
            grid.label(text="Type (always published)", icon="CHECKMARK")
            continue
        draw_property(grid, fs, OUTPUT_TOGGLE_NAMES[output_kind][key])


def _gated(layout, enabled):
    """A column that stays visible but greys out while ``enabled`` is false.

    Properties lists every setting; a row whose feature or mode is off is
    shown inactive (still editable) rather than hidden.
    """
    column = layout.column()
    column.active = bool(enabled)
    return column


def _surface_enabled(context):
    fs = _domain_owner(context)
    return bool(fs is not None and fs.surface_enabled)


def _whitewater_enabled(context):
    fs = _domain_owner(context)
    return bool(fs is not None and fs.whitewater_enabled)


def _advanced_visible(context):
    """True unless the user has chosen the Basic interface (D2 two-tier gate).

    Fail-open on a missing settings group only: a scene with no FluidSim group
    keeps the full surface rather than silently dropping deep-config panels.
    ``ui_mode`` is a scene field, read directly.
    """
    fs = getattr(getattr(context, "scene", None), "fluidsim", None)
    return bool(fs is None or fs.ui_mode == "ADVANCED")


def _timeline_playing(context) -> bool:
    """Return Blender's actual playback state for the current UI context."""
    screen = getattr(context, "screen", None)
    return bool(screen is not None and getattr(screen, "is_animation_playing", False))


class _FluidSimSimulationPanel(FluidSimScenePanel):
    """Shared visibility contract for the domain-physics fan.

    Properties always shows every setting; the Basic/Advanced switch only trims
    the 3D View sidebar stages.
    """

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)


class FLUIDSIM_PT_scene_lifecycle(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Lifecycle"
    bl_idname = "FLUIDSIM_PT_scene_lifecycle"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context)

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        state = activity_ui_state(scene)
        _probe, _selected, runtime_ready, _message = _runtime_ui_state()
        owns = state["owns_scene"]

        can_step = bool(
            runtime_ready
            and owns
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
            release.label(text="Frees the GPU solver; Play can allocate it again.", icon="X")
            release.operator("fluidsim.release", text="Release Solver", icon="X")
            if state["id"] not in {"CACHE_PLAYBACK", "PRODUCTION_BAKE"}:
                reset = danger.column(align=True)
                reset.label(text="Discards every simulated frame back to the start.", icon="FILE_REFRESH")
                reset.operator("fluidsim.reset", text="Reset Simulation", icon="FILE_REFRESH")


class FLUIDSIM_PT_scene_live_timeline(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Live Timeline Recovery"
    bl_idname = "FLUIDSIM_PT_scene_live_timeline"
    bl_parent_id = "FLUIDSIM_PT_scene_lifecycle"

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context)

    def draw(self, context):
        layout = _layout(self)
        layout.prop(context.scene.fluidsim, "max_frame_catchup")


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
        status = str(fs.status or "")
        if status:
            _draw_wrapped(layout, status)
        domain = _domain_owner(context)
        height_error = str(domain.height_error or "") if domain is not None else ""
        if height_error:
            draw_status_message(layout, height_error, "ERROR")

        _probe, selected, runtime_ready, runtime_message = _runtime_ui_state()
        runtime = layout.row(align=True)
        runtime.label(
            text=(f"{selected} · {runtime_message}" if selected else runtime_message),
            icon="CHECKMARK" if runtime_ready else "ERROR",
        )
        runtime.operator("fluidsim.refresh_runtime_probe", text="Re-detect GPU" if not runtime_ready else "Re-detect", icon="FILE_REFRESH")
        _draw_wrapped(layout, _build_identity(), icon="INFO")

        # Compact one-line live telemetry: last step cost, substep count, and
        # whether the CFL condition forced a clamp. Read-only status only —
        # transport lives in the 3D View launcher (FLUIDSIM_PT_viewport_status),
        # lifecycle in FLUIDSIM_PT_scene_lifecycle.
        step_ms = float(fs.stat_step_ms or 0.0)
        substeps = int(fs.stat_substeps or 0)
        cfl_clamped = bool(fs.stat_cfl_clamped)
        layout.label(
            text=f"{step_ms:.1f} ms · {substeps} substeps · CFL {'clamped' if cfl_clamped else 'ok'}",
            icon="TIME",
        )

        error = current_error(scene)
        if error:
            box = layout.box()
            _draw_wrapped(box, error, icon="ERROR")
            if any(word in error.lower() for word in ("output", "render node", "material")):
                repair = box.operator(
                    "fluidsim.install_render_nodes",
                    text="Repair Outputs",
                    icon="GEOMETRY_NODES",
                )
                repair.force = False
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
        row.operator("fluidsim.apply_changes", text="Apply Changes", icon="CHECKMARK")
        row.operator("fluidsim.revert_pending", text="Revert Pending Values", icon="LOOP_BACK")


class FLUIDSIM_PT_scene_setup(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Setup"
    bl_idname = "FLUIDSIM_PT_scene_setup"
    bl_order = 10

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        file_compat.draw_pre_bridge_warning(layout, context.scene)
        # Front door of the two-tier disclosure (design D2): the Interface switch
        # and the Quality target sit at the top of the first panel so every user
        # meets them before any deep-config surface.
        layout.prop(fs, "ui_mode", expand=True)
        layout.prop(fs, "quality")
        if fs.setup_mode:
            layout.label(text=f"Current setup: {fs.setup_mode.replace('_', ' ').title()}", icon="MOD_FLUIDSIM")
        primary = layout.row(align=True)
        primary.scale_y = 1.25
        setup_preview = primary.operator(
            "fluidsim.setup_scene", text="Set Up & Play…", icon="PLAY"
        )
        setup_preview.start_preview = True
        setup_only = layout.operator(
            "fluidsim.setup_scene", text="Set Up Only…", icon="MOD_FLUIDSIM"
        )
        setup_only.start_preview = False


class FLUIDSIM_PT_scene_presets(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Product Presets"
    bl_idname = "FLUIDSIM_PT_scene_presets"
    bl_parent_id = "FLUIDSIM_PT_scene_setup"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "product_preset", text="Preset")
        row = layout.row(align=True)
        apply_row = row.row(align=True)
        apply_row.enabled = fs.product_preset not in {"", "NONE"}
        apply_row.operator("fluidsim.apply_product_preset", text="Apply Selected Preset", icon="PRESET")
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
                "resolution_divisions",
                "voxel_size",
                "particles_per_voxel",
                "max_particles",
                "auto_grow_particles",
                "topology_expand",
                "open_boundaries",
                "domain_sdf_walls",
            ),
        )
        if owner is not None:
            # ui_stages imports this module, so the shared drawer loads lazily.
            from .ui_stages import draw_narrow_band

            draw_narrow_band(layout, context.scene, owner)
        draw_resolution_readout(layout, context.scene)


class FLUIDSIM_PT_scene_capacity(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Capacity & Safety"
    bl_idname = "FLUIDSIM_PT_scene_capacity"
    bl_parent_id = "FLUIDSIM_PT_scene_domain"

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "leaf_capacity")
        layout.prop(fs, "max_whitewater")
        _gated(layout, fs.auto_grow_particles).prop(fs, "particle_growth_failure_policy")
        _draw_capacity_plan(layout, context.scene)


class FLUIDSIM_PT_scene_height(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Initial Liquid — Height Fill"
    bl_idname = "FLUIDSIM_PT_scene_height"
    bl_order = 30

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "height_enabled")
        body = _gated(layout, fs.height_enabled)
        body.prop(fs, "height_base")
        body.prop(fs, "height_back_mode")
        profile = _gated(body, fs.height_back_mode == "PROFILE")
        profile.prop(fs, "height_profile_image")
        profile.prop(fs, "height_profile_scale")
        body.prop(fs, "height_emission_mode")
        body.prop(fs, "height_velocity")
        body.prop(fs, "height_emit_particles")


class FLUIDSIM_PT_scene_simulation(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Simulation Physics"
    bl_idname = "FLUIDSIM_PT_scene_simulation"
    bl_order = 40

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)

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
        layout.label(text="Active: " + (", ".join(enabled) if enabled else "Base liquid solver"))


class FLUIDSIM_PT_sim_time(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Time & Transfer"
    bl_idname = "FLUIDSIM_PT_sim_time"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        draw_property_grid(layout, fs, ("time_scale", "cfl", "min_substeps", "max_substeps", "sample_type"))
        _gated(layout, fs.sample_type == "FLIP").prop(fs, "flip_ratio")
        layout.prop(fs, "trace_mode")
        layout.prop(fs, "count_stiffness")


class FLUIDSIM_PT_sim_gravity(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Gravity"
    bl_idname = "FLUIDSIM_PT_sim_gravity"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "use_scene_gravity")
        _gated(layout, not fs.use_scene_gravity).prop(fs, "gravity")


class FLUIDSIM_PT_sim_pressure(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Pressure"
    bl_idname = "FLUIDSIM_PT_sim_pressure"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "pressure_solver")
        layout.prop(fs, "pressure_iterations")
        pcg = _gated(layout, fs.pressure_solver == "PCG")
        pcg.prop(fs, "pressure_preconditioner")
        pcg.prop(fs, "pressure_tolerance")
        # Experimental: no trained weights ship yet, so the solve falls back to
        # a zero initial guess (same result) until a weights file is installed.
        pcg.prop(fs, "pressure_learned_guess")
        layout.prop(fs, "pressure_threshold")


class FLUIDSIM_PT_sim_viscosity(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Viscosity"
    bl_idname = "FLUIDSIM_PT_sim_viscosity"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "viscosity_enabled")
        body = _gated(layout, fs.viscosity_enabled)
        body.prop(fs, "viscosity_solver")
        body.prop(fs, "viscosity_from_particles")
        _gated(body, not fs.viscosity_from_particles).prop(fs, "viscosity_constant")
        body.prop(fs, "viscosity_iterations")
        _gated(body, fs.viscosity_solver == "PCG").prop(fs, "viscosity_tolerance")


class FLUIDSIM_PT_sim_surface_tension(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Surface Tension"
    bl_idname = "FLUIDSIM_PT_sim_surface_tension"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "surface_tension_enabled")
        draw_property_grid(
            _gated(layout, fs.surface_tension_enabled),
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


class FLUIDSIM_PT_sim_color_mixing(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Color Mixing"
    bl_idname = "FLUIDSIM_PT_sim_color_mixing"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "species_enabled")
        body = _gated(layout, fs.species_enabled)
        body.prop(fs, "species_mixing_rate")
        box = body.box()
        box.label(text="Emitter colors are configured per fluid source.", icon="COLOR")
        box.label(text="Any number of source colors can mix.")


class FLUIDSIM_PT_sim_relax(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Particle Relaxation"
    bl_idname = "FLUIDSIM_PT_sim_relax"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "relax_enabled")
        body = _gated(layout, fs.relax_enabled)
        body.prop(fs, "relax_strength")
        body.prop(fs, "relax_mode")


class FLUIDSIM_PT_sim_limits(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Velocity Limits"
    bl_idname = "FLUIDSIM_PT_sim_limits"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "max_speed")
        body = _gated(layout, fs.max_speed > 0.0)
        body.prop(fs, "limit_behavior")
        _gated(body, fs.limit_behavior == "PERCENT").prop(fs, "limit_percent")


class FLUIDSIM_PT_sim_forces(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Global Forces"
    bl_idname = "FLUIDSIM_PT_sim_forces"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "directional_force")
        layout.prop(fs, "omnidirectional_force_center")
        layout.prop(fs, "omnidirectional_force_strength")
        _gated(layout, fs.omnidirectional_force_strength != 0.0).prop(fs, "omnidirectional_force_radius")
        layout.prop(fs, "force_diffusion")


class FLUIDSIM_PT_sim_dissipation(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Dissipation"
    bl_idname = "FLUIDSIM_PT_sim_dissipation"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "dissipation_enabled")
        body = _gated(layout, fs.dissipation_enabled)
        body.prop(fs, "dissipation_attribute")
        body.prop(fs, "dissipation_rate")
        body.prop(fs, "dissipation_target")


class FLUIDSIM_PT_sim_attraction(_FluidSimSimulationPanel, bpy.types.Panel):
    bl_label = "Attraction"
    bl_idname = "FLUIDSIM_PT_sim_attraction"
    bl_parent_id = "FLUIDSIM_PT_scene_simulation"

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "attraction_enabled")
        body = _gated(layout, fs.attraction_enabled)
        body.prop(fs, "attraction_strength")
        body.prop(fs, "attraction_max_distance")


class FLUIDSIM_PT_scene_surface(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Surface Reconstruction"
    bl_idname = "FLUIDSIM_PT_scene_surface"
    bl_order = 60

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "surface_enabled")


class _SurfaceChild:
    """Always listed; Surface Reconstruction greys these while surfacing is off."""

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)


class FLUIDSIM_PT_surface_resolution(_SurfaceChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Resolution & Cadence"
    bl_idname = "FLUIDSIM_PT_surface_resolution"
    bl_parent_id = "FLUIDSIM_PT_scene_surface"

    def draw(self, context):
        layout = _gated(_layout(self), _surface_enabled(context))
        fs = _domain_owner(context)
        # Mode as buttons and both resolution rows always live (never greyed):
        # the user must be able to see and set every value up front.
        layout.row().prop(fs, "surface_resolution_mode", expand=True)
        layout.prop(fs, "surface_resolution_multiplier")
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

    def draw(self, context):
        layout = _gated(_layout(self), _surface_enabled(context))
        fs = _domain_owner(context)
        draw_property_grid(layout, fs, ("particle_radius_scale", "surface_splat_radius", "surface_build_lowres_sdf"))


class FLUIDSIM_PT_surface_filters(_SurfaceChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Filters"
    bl_idname = "FLUIDSIM_PT_surface_filters"
    bl_parent_id = "FLUIDSIM_PT_scene_surface"

    def draw(self, context):
        # D9: the four post-process passes are chunked into their own labeled
        # boxes so the filter chain reads as four sequential sections rather than
        # a flat ten-prop wall.  Each strength knob greys out while its pass has
        # zero iterations.
        layout = _gated(_layout(self), _surface_enabled(context))
        fs = _domain_owner(context)

        flt = layout.box()
        flt.label(text="Filter", icon="FILTER")
        flt.prop(fs, "surface_filter_type")
        flt.prop(fs, "surface_filter_iterations")
        _gated(flt, fs.surface_filter_iterations > 0).prop(fs, "surface_filter_strength")

        dilate = layout.box()
        dilate.label(text="Dilate", icon="FULLSCREEN_ENTER")
        dilate.prop(fs, "surface_dilate_iterations")
        _gated(dilate, fs.surface_dilate_iterations > 0).prop(fs, "surface_dilate_strength")

        erode = layout.box()
        erode.label(text="Erode", icon="FULLSCREEN_EXIT")
        erode.prop(fs, "surface_erode_iterations")
        _gated(erode, fs.surface_erode_iterations > 0).prop(fs, "surface_erode_strength")

        smooth = layout.box()
        smooth.label(text="Smooth", icon="MOD_SMOOTH")
        smooth.prop(fs, "surface_smooth_type")
        smooth.prop(fs, "surface_smooth_iterations")
        _gated(smooth, fs.surface_smooth_iterations > 0).prop(fs, "surface_smooth_strength")


class FLUIDSIM_PT_scene_whitewater(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Whitewater"
    bl_idname = "FLUIDSIM_PT_scene_whitewater"
    bl_order = 70

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "whitewater_enabled")


class _WhitewaterChild:
    """Always listed; these grey out while whitewater is off."""

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)


class FLUIDSIM_PT_whitewater_emission(_WhitewaterChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Emission"
    bl_idname = "FLUIDSIM_PT_whitewater_emission"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"

    def draw(self, context):
        layout = _gated(_layout(self), _whitewater_enabled(context))
        fs = _domain_owner(context)
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


class FLUIDSIM_PT_whitewater_classification(_WhitewaterChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Classification"
    bl_idname = "FLUIDSIM_PT_whitewater_classification"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"

    def draw(self, context):
        layout = _gated(_layout(self), _whitewater_enabled(context))
        fs = _domain_owner(context)
        draw_property_grid(layout, fs, ("ww_foam_band", "ww_spray_min_speed", "ww_type_hysteresis"))
        layout.label(text="Types are persistent: Foam 0 · Bubble 1 · Spray 2", icon="INFO")


class FLUIDSIM_PT_whitewater_life(_WhitewaterChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Lifetime"
    bl_idname = "FLUIDSIM_PT_whitewater_life"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"

    def draw(self, context):
        layout = _gated(_layout(self), _whitewater_enabled(context))
        draw_property_grid(layout, _domain_owner(context), ("ww_life_min", "ww_life_max"))


class FLUIDSIM_PT_whitewater_dynamics(_WhitewaterChild, FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Dynamics"
    bl_idname = "FLUIDSIM_PT_whitewater_dynamics"
    bl_parent_id = "FLUIDSIM_PT_scene_whitewater"

    def draw(self, context):
        layout = _gated(_layout(self), _whitewater_enabled(context))
        fs = _domain_owner(context)
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
        if _whitewater_enabled(context) and fs.show_whitewater:
            active.append("Whitewater")
        layout.label(text="Visible: " + (", ".join(active) if active else "None"), icon="HIDE_OFF")
        # Basic master switch for the export inventory (spec §8); the
        # per-attribute toggles live in the Advanced-only children.
        draw_property(layout, _domain_owner(context), "export_attributes")


class FLUIDSIM_PT_output_display(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Display"
    bl_idname = "FLUIDSIM_PT_output_display"
    bl_parent_id = "FLUIDSIM_PT_scene_outputs"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        # Display choices are scene-scoped; the two *_enabled gates are
        # domain-owned, so read them from the owner the solver uses.
        layout.prop(fs, "liquid_view_mode")
        layout.prop(fs, "show_overlay", text="Viewport HUD")
        surface = _gated(layout, fs.liquid_view_mode in {"SURFACE", "SURFACE_PARTICLES"} and _surface_enabled(context))
        surface.prop(fs, "surface_attribute_mode")
        particles = _gated(layout, fs.liquid_view_mode in {"PARTICLES", "SURFACE_PARTICLES"})
        particles.prop(fs, "particle_display_fraction")
        particles.prop(fs, "particle_display_scale")
        particles.prop(fs, "particle_attribute_mode")
        whitewater = _gated(layout, _whitewater_enabled(context))
        whitewater.prop(fs, "show_whitewater")
        shown = _gated(whitewater, fs.show_whitewater)
        shown.prop(fs, "whitewater_output_mode")
        shown.prop(fs, "whitewater_display_fraction")
        shown.prop(fs, "whitewater_display_scale")
        shown.prop(fs, "whitewater_attribute_mode")
        preview = layout.column(align=True)
        preview.prop(fs, "liquid_preview_enabled")
        sub = preview.column(align=True)
        sub.active = fs.liquid_preview_enabled
        sub.prop(fs, "liquid_preview_scale", text="Resolution")
        sub.prop(fs, "liquid_preview_radius_multiplier", text="Radius")
        sub.prop(fs, "liquid_preview_absorption")
        sub.prop(fs, "liquid_preview_whitewater", text="Whitewater")


class FLUIDSIM_PT_output_materials(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Materials & Collection"
    bl_idname = "FLUIDSIM_PT_output_materials"
    bl_parent_id = "FLUIDSIM_PT_scene_outputs"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        # Material pickers are scene-scoped; whitewater_enabled is not.
        layout.prop(fs, "output_collection_name")
        _gated(layout, fs.liquid_view_mode in {"SURFACE", "SURFACE_PARTICLES"}).prop(fs, "surface_material")
        _gated(layout, fs.liquid_view_mode in {"PARTICLES", "SURFACE_PARTICLES"}).prop(fs, "particle_material")
        whitewater = _gated(layout, _whitewater_enabled(context))
        whitewater.prop(fs, "whitewater_material")
        streams = _gated(whitewater, fs.whitewater_output_mode in {"SEPARATE", "BOTH"})
        streams.prop(fs, "foam_material")
        streams.prop(fs, "bubble_material")
        streams.prop(fs, "spray_material")
        row = layout.row(align=True)
        install = row.operator("fluidsim.install_render_nodes", text="Repair Outputs", icon="GEOMETRY_NODES")
        install.force = False
        update = row.operator("fluidsim.install_render_nodes", text="Force Render-Node Update…", icon="FILE_REFRESH")
        update.force = True


class FLUIDSIM_PT_output_attributes(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Persistent Attributes"
    bl_idname = "FLUIDSIM_PT_output_attributes"
    bl_parent_id = "FLUIDSIM_PT_scene_outputs"
    # Top-level in the Physics mirror (its Scene parent is not mirrored), so it
    # needs an explicit position below the simulation and surface groups.
    bl_order = 80

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context)

    def draw(self, context):
        layout = _layout(self)
        layout.label(text="Inventory changes apply after Reset and affect cache identity.", icon="INFO")


class FLUIDSIM_PT_attrs_surface(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Surface Attributes"
    bl_idname = "FLUIDSIM_PT_attrs_surface"
    bl_parent_id = "FLUIDSIM_PT_output_attributes"

    def draw(self, context):
        _draw_attribute_group(_layout(self), _domain_owner(context), "surface")


class FLUIDSIM_PT_attrs_particles(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Particle Attributes"
    bl_idname = "FLUIDSIM_PT_attrs_particles"
    bl_parent_id = "FLUIDSIM_PT_output_attributes"

    def draw(self, context):
        _draw_attribute_group(_layout(self), _domain_owner(context), "particles")


class FLUIDSIM_PT_attrs_whitewater(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Whitewater Attributes"
    bl_idname = "FLUIDSIM_PT_attrs_whitewater"
    bl_parent_id = "FLUIDSIM_PT_output_attributes"

    def draw(self, context):
        _draw_attribute_group(_layout(self), _domain_owner(context), "whitewater")


# Cache & Production Bake lives in the 3D View sidebar's Bake & Export stage
# (ui_stages.py), which calls these draw functions. They stay here because they
# share this module's runtime/cache helpers and the test seams that patch them.
def draw_cache_playback(layout, context):
    """Play-from-cache action; never allocates the solver."""
    state = activity_ui_state(context.scene)
    if state["id"] == "CACHE_PLAYBACK":
        if _timeline_playing(context):
            layout.operator("fluidsim.preview_pause", text="Pause", icon="PAUSE")
        else:
            layout.operator("screen.animation_play", text="Play", icon="PLAY")
    else:
        layout.operator("fluidsim.cache_preview_start", text="Play Cache", icon="PLAY")
    layout.label(text="Cache playback does not allocate the solver.", icon="INFO")


def draw_cache_bake(layout, context, show_all=False):
    """Production Bake range, progress and the one cache-aware start/resume/cancel action.

    ``show_all`` keeps the Advanced-only command row visible; Properties passes
    it because the Basic/Advanced switch only trims the 3D View sidebar.
    """
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
            _draw_wrapped(
                layout, presentation.message,
                icon="ERROR" if presentation.warning else "INFO",
            )
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
    if show_all or _advanced_visible(context):
        layout.operator("fluidsim.copy_bake_command", text="Copy Headless Bake Command", icon="COPYDOWN")


def cache_manage_available(context) -> bool:
    """True when ``cache_path`` already holds a recognized FluidSim cache."""
    from .cache_manager import is_managed_cache_directory

    path = _cache_directory(getattr(context, "scene", None))
    return path is not None and is_managed_cache_directory(path)


def draw_cache_manage(layout, context):
    """Integrity scan, move/rename/free and open-folder actions for the cache."""
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


class FLUIDSIM_PT_scene_cache_bake(FluidSimScenePanel, bpy.types.Panel):
    """Cache location, storage, frame range, playback, Production Bake and management.

    The 3D View sidebar's Bake & Export stage draws the same helpers; this is
    the Properties home so no cache setting is reachable only from the sidebar.
    """

    bl_label = "Cache & Bake"
    bl_idname = "FLUIDSIM_PT_scene_cache_bake"
    bl_order = 90

    def draw(self, context):
        # ui_stages imports this module, so the shared drawers load lazily.
        from .ui_stages import _prop, draw_cache_refusal

        layout = _layout(self)
        scene = context.scene
        fs = scene.fluidsim
        _prop(layout, scene, "cache_path")
        _prop(layout, scene, "cache_compress")
        _prop(layout, scene, "checkpoint_interval")
        _prop(layout, scene, "cache_use_custom_range")
        row = layout.row(align=True)
        row.active = bool(fs.cache_use_custom_range)
        _prop(row, scene, "cache_frame_start")
        _prop(row, scene, "cache_frame_end")
        if active_domain_settings(scene) is None:
            layout.label(text="Set up a liquid domain first to bake.", icon="INFO")
            return
        layout.separator()
        draw_cache_refusal(layout, context)
        draw_cache_playback(layout, context)
        layout.separator()
        draw_cache_bake(layout, context, show_all=True)
        if cache_manage_available(context):
            layout.separator()
            draw_cache_manage(layout, context)


class FLUIDSIM_PT_scene_diagnostics(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Diagnostics"
    bl_idname = "FLUIDSIM_PT_scene_diagnostics"
    bl_order = 100

    @classmethod
    def poll(cls, context):
        return FluidSimScenePanel.poll(context)

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
        if fs.stat_surface_tension_message:
            _draw_wrapped(layout, fs.stat_surface_tension_message, icon="INFO")
        if fs.stat_native_solver_message:
            _draw_wrapped(layout, fs.stat_native_solver_message, icon="ERROR")
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
    # Top-level in the Physics mirror (its Scene parent is not mirrored); last.
    bl_order = 100

    @classmethod
    def poll(cls, context):
        return _domain_visible(context)

    def draw(self, context):
        layout = _layout(self)
        fs = _domain_owner(context)
        layout.prop(fs, "profiling_enabled")
        if fs.profiling_enabled:
            layout.prop(fs, "profiling_sync")
        from .scene_bridge import session
        sess = session()
        # The Solver Session is a process singleton: its profiler times
        # whichever scene owns it, so another scene must not present them.
        if (sess.profiler is not None and sess.profiler.history
                and sess.owns_scene(context.scene)):
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

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "benchmark_frames")
        layout.prop(fs, "benchmark_baseline")
        layout.operator("fluidsim.run_benchmark", text="Run Benchmark…", icon="TIME")
        from .scene_bridge import session
        # Deliberately not scene-scoped: Run Benchmark times a fixed synthetic
        # scenario on the configured device, not this scene.
        result = session().benchmark_result
        if result:
            box = layout.box()
            box.label(text=f"Average step: {result['timeStepAverage'] * 1000.0:.1f} ms")
            box.label(text=f"Voxels/s: {result['voxelsPerSecondAverage'] / 1e6:.1f} M")


class FLUIDSIM_PT_diagnostics_report(FluidSimScenePanel, bpy.types.Panel):
    bl_label = "Report & Export"
    bl_idname = "FLUIDSIM_PT_diagnostics_report"
    bl_parent_id = "FLUIDSIM_PT_scene_diagnostics"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "report_path")
        row = layout.row(align=True)
        row.operator("fluidsim.write_report", text="Write Diagnostic Report…", icon="TEXT")
        row.operator("fluidsim.export_frame", text="Export Current Frame…", icon="EXPORT")


# The Scene tab registers only the panels that edit scene-owned settings or show
# scene-level runtime state. Every domain setting is authored in the Physics tab
# (``ui_physics`` mirrors the classes below) and in the staged sidebar; the
# Scene tab no longer carries a second copy of those controls.
SCENE_PANEL_CLASSES = (
    FLUIDSIM_PT_scene_lifecycle,
    FLUIDSIM_PT_scene_control,
    FLUIDSIM_PT_scene_pending,
    FLUIDSIM_PT_scene_live_timeline,
    FLUIDSIM_PT_scene_setup,
    FLUIDSIM_PT_scene_presets,
    FLUIDSIM_PT_scene_outputs,
    FLUIDSIM_PT_output_display,
    FLUIDSIM_PT_output_materials,
    FLUIDSIM_PT_scene_cache_bake,
    FLUIDSIM_PT_scene_diagnostics,
    FLUIDSIM_PT_scene_runtime_stats,
    FLUIDSIM_PT_diagnostics_benchmark,
    FLUIDSIM_PT_diagnostics_report,
)

# Draw sources for domain settings. These classes are never registered under
# the Scene tab: ``ui_physics._make_panel_mirrors`` builds the Physics-tab panels
# from them, and each draws through ``_domain_owner`` so it edits
# ``obj.fluidsim_domain``. Their ``bl_parent_id`` values name sibling sources
# (and, for the three that formerly sat under a Scene-tab parent, a parent that
# stays in the Scene tab); the mirror maps parents it does not mirror to the
# Physics root.
DOMAIN_DRAW_SOURCE_CLASSES = (
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
    FLUIDSIM_PT_scene_surface,
    FLUIDSIM_PT_surface_resolution,
    FLUIDSIM_PT_surface_reconstruct,
    FLUIDSIM_PT_surface_filters,
    FLUIDSIM_PT_scene_whitewater,
    FLUIDSIM_PT_whitewater_emission,
    FLUIDSIM_PT_whitewater_classification,
    FLUIDSIM_PT_whitewater_life,
    FLUIDSIM_PT_whitewater_dynamics,
    FLUIDSIM_PT_output_attributes,
    FLUIDSIM_PT_attrs_surface,
    FLUIDSIM_PT_attrs_particles,
    FLUIDSIM_PT_attrs_whitewater,
    FLUIDSIM_PT_diagnostics_profiling,
)


def register():
    for cls in SCENE_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    from ..unregister_guard import unregister_classes

    unregister_classes(SCENE_PANEL_CLASSES)
