"""UI panels for the FluidSim Warp add-on (3D Viewport sidebar > FluidSim).

Panel set mirrors the synthesized UI contract:
Simulation, Domain / Sparse Grid, Particles, Sources, Colliders / Sinks,
Forces / Attraction, Pressure Projection, Viscosity, Whitewater,
Realtime Surfacing / Particles, Viewport / Render Output, Custom Warp
Snippets, Diagnostics / Profiling, Validation / Benchmarks.
"""

from __future__ import annotations

import bpy

from .sim.constants import TIMER_PHASE_SLOTS
from .sim.enums import SourceGeometryMode


class FluidSimPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FluidSim"


def _draw_interaction_tier(layout, text, icon="INFO"):
    """Draw a structural update-timing label, not explanatory guidance."""
    row = layout.row() if hasattr(layout, "row") else layout
    row.alignment = "RIGHT"
    row.label(text=text, icon=icon)


def _pending_reason_label(scene, reason):
    """Translate cached internal setting names into user-facing RNA labels."""
    prefix, separator, field_name = str(reason).partition(":")
    if not separator:
        return str(reason)
    field_name = field_name.strip()
    rna = getattr(scene.fluidsim, "bl_rna", None)
    properties = getattr(rna, "properties", {})
    prop = properties.get(field_name)
    label = prop.name if prop is not None else field_name.replace("_", " ").title()
    return f"{label} will apply after restart"


def _draw_pending_banner(layout, scene, *, show_actions=False):
    """Draw one neutral pending summary; never inspect geometry or mutate RNA."""
    from .blender.pending_settings import pending_presentation

    pending = pending_presentation(scene)
    if pending["action"] == "NONE":
        return
    box = layout.box()
    box.label(
        text=f"{pending['count']} change(s) staged for restart",
        icon="INFO",
    )
    for reason in pending["reasons"][:3]:
        box.label(text=_pending_reason_label(scene, reason))
    if pending["count"] > 3:
        box.label(text=f"+ {pending['count'] - 3} more")
    if pending["blocked_reason"]:
        box.label(text=pending["blocked_reason"][:120], icon="INFO")
    if show_actions:
        row = box.row(align=True)
        if pending["action"] == "REINIT_PENDING":
            row.operator(
                "fluidsim.apply_pending_reinitialize",
                text="Restart Now", icon="FILE_REFRESH")
        else:
            row.operator(
                "fluidsim.apply_pending_reset",
                text="Reset Now", icon="FILE_REFRESH")
        row.operator(
            "fluidsim.revert_pending", text="Discard Changes", icon="LOOP_BACK")


def _draw_header_interaction(self, context):
    scene = getattr(context, "scene", None)
    if getattr(scene, "fluidsim", None) is None:
        return
    from .blender.pending_settings import pending_presentation

    pending = pending_presentation(scene)
    icon = "INFO" if pending["action"] != "NONE" else "MOD_FLUIDSIM"
    text = str(pending["count"]) if pending["count"] else ""
    self.layout.popover(panel="FLUIDSIM_PT_interaction_popover", text=text, icon=icon)


class FLUIDSIM_PT_interaction_popover(bpy.types.Panel):
    bl_label = "FluidSim Interaction"
    bl_idname = "FLUIDSIM_PT_interaction_popover"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        fs = scene.fluidsim
        from .blender.pending_settings import pending_presentation
        from .blender.scene_bridge import session

        pending = pending_presentation(scene)
        sess = session()
        layout.label(text="Solver active" if sess.owns_scene(scene)
                     else "Solver inactive", icon="PLAY" if sess.owns_scene(scene)
                     else "PAUSE")
        layout.prop(fs, "live", text="Live Playback")
        if pending["action"] == "NONE":
            layout.label(text="No pending settings", icon="CHECKMARK")
        else:
            _draw_pending_banner(layout, scene, show_actions=True)
        row = layout.row(align=True)
        row.operator("fluidsim.reset", text="Reset", icon="FILE_REFRESH")
        row.operator("fluidsim.release", text="Release", icon="X")


def _draw_capacity_plan(layout, plan, fs=None):
    if plan is None:
        return
    box = layout.box()
    if getattr(plan, "blockers", ()):
        box.label(text="Capacity adjustment recommended", icon="INFO")
        box.label(text="Nothing has been changed or partially emitted.")
    else:
        box.label(text="Source Capacity Planner", icon="MEMORY")
    particles = getattr(plan, "particle_demand", None)
    leaves = getattr(plan, "leaf_demand", None)
    if particles is not None:
        quality = getattr(getattr(particles, "quality", None), "value", "unknown")
        value = "unknown" if particles.value is None else f"{particles.value:,}"
        box.label(text=f"Initial particles: {value} ({quality.replace('_', ' ')})")
    if leaves is not None:
        quality = getattr(getattr(leaves, "quality", None), "value", "unknown")
        value = "unknown" if leaves.value is None else f"{leaves.value:,}"
        box.label(text=f"Initial sparse leaves: {value} ({quality.replace('_', ' ')})")
    estimated = int(getattr(plan, "estimated_bytes", 0) or 0)
    budget = int(getattr(plan, "budget_bytes", 0) or 0)
    box.label(text=(
        f"Estimated peak: {estimated / 1024**3:.2f} GiB / "
        f"live budget {budget / 1024**3:.2f} GiB"))
    box.label(text="Estimate only - not an OOM guarantee", icon="INFO")
    for blocker in getattr(plan, "blockers", ()):
        row = box.row()
        required = getattr(blocker, "required", None)
        available = getattr(blocker, "available", None)
        if required is None:
            text = getattr(blocker, "message", "Capacity demand is unknown")
        else:
            text = f"{blocker.code}: required {required:,}, available {available:,}"
        row.label(text=text[:120], icon="INFO")
    suggestions = tuple(getattr(plan, "suggestions", ()))
    if suggestions:
        column = box.column(align=True)
        column.label(text="Recommended change (review before applying):")
        show_alternatives = bool(
            fs is not None and getattr(fs, "capacity_show_alternatives", False))
        visible = suggestions if show_alternatives else suggestions[:1]
        for index, suggestion in enumerate(visible):
            option = column.box()
            prefix = "Recommended — " if index == 0 else "Alternative — "
            option.label(text=(prefix + suggestion.label)[:120])
            changes = ", ".join(
                f"{name.replace('_', ' ').title()}: "
                f"{value:,}" if isinstance(value, int)
                else f"{name.replace('_', ' ').title()}: {value}"
                for name, value in suggestion.changes.items())
            option.label(text=changes[:120])
            for tradeoff in suggestion.tradeoffs:
                option.label(text=f"{tradeoff}"[:120], icon="DOT")
            text = ("Review Recommended Changes & Run" if index == 0
                    else "Review This Option & Run")
            operator = option.operator(
                "fluidsim.apply_capacity_suggestion", text=text,
                icon="CHECKMARK" if index == 0 else "MODIFIER")
            operator.suggestion_key = getattr(suggestion, "key", "")
        if len(suggestions) > 1 and fs is not None:
            column.prop(
                fs, "capacity_show_alternatives",
                text="Show lower-priority alternatives")
    elif getattr(plan, "blockers", ()):
        box.label(
            text="No conservative adjustment fits the live safety budget.",
            icon="CANCEL",
        )


class FLUIDSIM_PT_simulation(FluidSimPanel, bpy.types.Panel):
    bl_label = "Simulation"
    bl_idname = "FLUIDSIM_PT_simulation"

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        from .blender.scene_bridge import current_error, session

        sess = session()
        _draw_pending_banner(layout, context.scene, show_actions=True)

        col = layout.column(align=True)
        col.operator("fluidsim.setup_scene", icon="MOD_FLUIDSIM")
        col.operator("fluidsim.initialize", icon="PLAY",
                     text="Initialize Solver" if not sess.active
                     else "Reinitialize Solver")

        row = layout.row(align=True)
        row.enabled = sess.active
        icon = "PAUSE" if fs.live else "PLAY"
        row.operator("fluidsim.toggle_live", icon=icon,
                     text="Live: On" if fs.live else "Live: Off")
        row.operator("fluidsim.reset", text="", icon="FILE_REFRESH")
        row.operator("fluidsim.release", text="", icon="X")

        row = layout.row(align=True)
        row.enabled = sess.active
        op = row.operator("fluidsim.step", text="Step 1")
        op.frames = 1
        op = row.operator("fluidsim.step", text="Step 10")
        op.frames = 10

        row = layout.row(align=True)
        expanded = getattr(fs, "playback_advanced_expanded", False)
        row.prop(
            fs, "playback_advanced_expanded", text="Advanced Playback",
            icon=("TRIA_DOWN" if expanded else "TRIA_RIGHT"), emboss=False)
        if expanded:
            advanced = layout.column(align=True)
            advanced.prop(
                fs, "max_frame_catchup",
                text="Maximum Playback Recovery Frames")

        box = layout.box()
        box.label(text=fs.status, icon="INFO")
        error = current_error(context.scene)
        if error:
            box.label(text=error[:120], icon="ERROR")
            from .sim.memory_safety import is_memory_capacity_error
            plan = getattr(sess, "capacity_plan", None)
            integrity_blocked = bool(plan and any(
                getattr(blocker, "code", "") != "MEMORY_ESTIMATE"
                for blocker in getattr(plan, "blockers", ())))
            if is_memory_capacity_error(error):
                recovery = box.column(align=True)
                if integrity_blocked:
                    recovery.label(
                        text="Review and approve a capacity option below",
                        icon="INFO",
                    )
                else:
                    recovery.label(text="GPU capacity risk detected", icon="INFO")
                    recovery.operator("fluidsim.apply_safe_settings", icon="MODIFIER")
                    run_anyway = recovery.operator(
                        "fluidsim.initialize", text="Ignore Estimate & Initialize",
                        icon="INFO")
                    run_anyway.ignore_memory_safeguard = True
            box.operator("fluidsim.clear_error", icon="X")
        _draw_capacity_plan(
            layout, getattr(sess, "capacity_plan", None), fs)
        if sess.active:
            col = box.column(align=True)
            col.label(text=f"Particles: {fs.stat_particles:,}")
            col.label(text=(
                f"GPU capacity: {sess.solver.particles.capacity:,} "
                f"(initial reserve {fs.max_particles:,})"))
            col.label(text=f"Active voxels: {fs.stat_voxels:,}")
            col.label(text=f"Step: {fs.stat_step_ms:.1f} ms "
                           f"({fs.stat_substeps} substeps)")
            if fs.stat_cfl_clamped:
                warn = col.row()
                warn.alert = True
                warn.label(text="CFL clamped by Max substeps", icon="ERROR")

        col = layout.column(align=True)
        _draw_interaction_tier(col, "Live")
        col.prop(fs, "time_scale")
        col.prop(fs, "cfl")
        row = col.row(align=True)
        row.prop(fs, "min_substeps", text="Min")
        row.prop(fs, "max_substeps", text="Max")
        col.prop(fs, "use_scene_gravity")
        sub = col.column(align=True)
        sub.enabled = not fs.use_scene_gravity
        sub.prop(fs, "gravity")


class FLUIDSIM_PT_domain(FluidSimPanel, bpy.types.Panel):
    bl_label = "Domain / Sparse Grid"
    bl_idname = "FLUIDSIM_PT_domain"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        _draw_pending_banner(layout, context.scene)
        _draw_interaction_tier(layout, "Reinitialize Required", "ERROR")
        layout.prop(fs, "domain_object")
        col = layout.column(align=True)
        col.prop(fs, "voxel_size")
        col.prop(fs, "leaf_capacity")
        _draw_interaction_tier(col, "Restart Required", "FILE_REFRESH")
        col.prop(fs, "topology_expand")
        col.prop(fs, "open_boundaries")
        col.prop(fs, "domain_sdf_walls")
        from .blender.scene_bridge import session
        from .blender.solver_settings import domain_bounds

        box = layout.box()
        # A draw callback runs on every redraw and an exception here takes the
        # whole sidebar down, so report the failure in the panel instead.
        try:
            lo, hi = domain_bounds(context.scene)
            dims = tuple(hi[i] - lo[i] for i in range(3))
            vox = tuple(int(d / max(fs.voxel_size, 1e-6)) for d in dims)
            box.label(text=f"Resolution: {vox[0]} x {vox[1]} x {vox[2]}")
            sess = session()
            if sess.active:
                g = sess.solver.grid
                box.label(text=f"Leaves: {g.active_leaf_count}/{g.capacity}")
        except Exception as exc:  # noqa: BLE001 - draw must never raise
            box.label(text=f"Domain unavailable: {exc}"[:120], icon="ERROR")


class FLUIDSIM_PT_particles(FluidSimPanel, bpy.types.Panel):
    bl_label = "Particles"
    bl_idname = "FLUIDSIM_PT_particles"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        from .blender.scene_bridge import session

        _draw_pending_banner(layout, context.scene)
        col = layout.column(align=True)
        _draw_interaction_tier(col, "Next Solver Start", "FILE_REFRESH")
        col.prop(fs, "max_particles")
        _draw_interaction_tier(col, "Live")
        col.prop(fs, "auto_grow_particles")
        growth = col.column(align=True)
        growth.enabled = fs.auto_grow_particles
        growth.prop(fs, "particle_growth_failure_policy")
        sess = session()
        if sess.active:
            growth.label(text=(
                f"Current GPU capacity: {sess.solver.particles.capacity:,}"))
        _draw_interaction_tier(col, "Live / Targeted Refresh")
        col.prop(fs, "particles_per_voxel")
        col.prop(fs, "sample_type")
        sub = col.column(align=True)
        sub.enabled = fs.sample_type == "FLIP"
        sub.prop(fs, "flip_ratio")
        col.prop(fs, "trace_mode")
        col.separator()
        col.prop(fs, "relax_enabled")
        sub = col.column(align=True)
        sub.enabled = fs.relax_enabled
        sub.prop(fs, "relax_strength")
        sub.prop(fs, "relax_mode")
        col.separator()
        col.prop(fs, "max_speed")
        sub = col.column(align=True)
        sub.enabled = fs.max_speed > 0
        sub.prop(fs, "limit_behavior")
        if fs.limit_behavior == "PERCENT":
            sub.prop(fs, "limit_percent")
        col.prop(fs, "count_stiffness")

        box = layout.box()
        box.label(text="ST-FLIP (Beta)", icon="EXPERIMENTAL")
        _draw_interaction_tier(
            box, "Enable/Reference/Seed: Restart Required", "FILE_REFRESH")
        box.prop(fs, "spacetime_enabled")
        sub = box.column(align=True)
        sub.enabled = fs.spacetime_enabled
        sub.prop(fs, "spacetime_jitter_strength")
        sub.prop(fs, "spacetime_adaptive_jitter")
        row = sub.row(align=True)
        row.prop(fs, "spacetime_phase_eta", text="Eta")
        row.prop(fs, "spacetime_phase_threshold", text="Threshold")
        sub.prop(fs, "spacetime_reference_mass")
        sub.prop(fs, "spacetime_variable_projection")
        projection = sub.column(align=True)
        projection.enabled = fs.spacetime_variable_projection
        row = projection.row(align=True)
        row.prop(fs, "spacetime_liquid_density", text="Liquid Density")
        row.prop(fs, "spacetime_gas_density", text="Gas Density")
        projection.prop(fs, "spacetime_min_density")
        row = sub.row(align=True)
        row.prop(fs, "spacetime_local_cfl", text="Local CFL")
        row.prop(fs, "spacetime_max_local_substeps", text="Max Steps")
        sub.prop(fs, "spacetime_resync_surface")
        sub.prop(fs, "spacetime_seed")


def _presentation_role(pg):
    if (getattr(pg, "role", "NONE") == "SOURCE"
            and getattr(pg, "source_mode", "PARTICLES") == "FIELD"):
        return "FIELD"
    return getattr(pg, "role", "NONE")


def _draw_role_actions(layout, tag_operator):
    row = layout.row(align=True)
    row.operator(tag_operator, icon="ADD")
    row.operator("fluidsim.clear_object_role", text="Clear", icon="X")


def _draw_source_status(layout, pg):
    message = getattr(pg, "source_status_message", "")
    if not message:
        return
    box = layout.box()
    severity = getattr(pg, "source_status_severity", "")
    box.alert = severity == "ERROR"
    icon = "ERROR" if severity == "ERROR" else "INFO"
    code = getattr(pg, "source_status_code", "")
    box.label(text=f"{code}: {message}" if code else message, icon=icon)
    if getattr(pg, "source_status_capacity_limited", False):
        box.label(text="Particle capacity limited the last source application", icon="ERROR")


def _draw_mesh_input(col, pg, *, solid, shell):
    col.prop(pg, "mesh_object")
    col.prop(pg, "mesh_evaluation_mode", text="Evaluation Cadence")
    if shell:
        col.prop(pg, "mesh_surface_thickness", text="Shell Half-width")
    if getattr(pg, "mesh_error", ""):
        error = col.box()
        error.alert = True
        error.label(text=pg.mesh_error, icon="ERROR")



def _draw_volume_input(col, pg, *, fields=False):
    col.prop(pg, "volume_input_object")
    if not fields:
        col.prop(pg, "volume_grid_type")
        if getattr(pg, "volume_grid_type", "FOG") == "SDF":
            col.prop(pg, "volume_sdf_grid")
        else:
            col.prop(pg, "volume_mask_grid")
            col.prop(pg, "volume_fog_threshold")


class FLUIDSIM_PT_liquid_emitters(FluidSimPanel, bpy.types.Panel):
    bl_label = "Liquid Emitters"
    bl_idname = "FLUIDSIM_PT_liquid_emitters"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        _draw_role_actions(layout, "fluidsim.tag_liquid_emitter")
        obj = context.active_object
        if obj is not None and _presentation_role(obj.fluidsim) == "SOURCE":
            pg = obj.fluidsim
            box = layout.box()
            box.label(text=f"{obj.name} (Liquid Emitter)", icon="OBJECT_DATA")
            col = box.column(align=True)
            col.prop(pg, "enabled")
            col.prop(pg, "emission_mode", text="Emission Behavior")
            legacy = int(getattr(pg, "source_ui_schema_version", 0) or 0) < 3
            if legacy:
                col.label(text="Legacy source region (saved behavior preserved)", icon="INFO")
                region = col.row(align=True)
                region.prop_enum(pg, "geometry_mode", "ANALYTIC", text="Analytic Volume")
                region.prop_enum(pg, "geometry_mode", "MESH", text="Mesh Surface Shell")
                kind = ("MESH_SHELL" if getattr(pg, "geometry_mode", "ANALYTIC") == "MESH"
                        else "ANALYTIC")
            else:
                region = col.row(align=True)
                region.prop_enum(pg, "source_region_kind", "ANALYTIC", text="Analytic Volume")
                region.prop_enum(pg, "source_region_kind", "MESH_SOLID", text="Mesh Solid Volume")
                region.prop_enum(pg, "source_region_kind", "MESH_SHELL", text="Mesh Surface Shell")
                region = col.row(align=True)
                region.prop_enum(pg, "source_region_kind", "POINTS", text="Points / Particles")
                region.prop_enum(pg, "source_region_kind", "VOLUME_GRID", text="Volume Grid")
                kind = getattr(pg, "source_region_kind", "ANALYTIC")
            if kind == "ANALYTIC":
                col.prop(pg, "shape")
            elif kind in {"MESH_SOLID", "MESH_SHELL"}:
                _draw_mesh_input(col, pg, solid=kind == "MESH_SOLID",
                                 shell=kind == "MESH_SHELL")
            elif kind == "POINTS":
                col.prop(pg, "point_input_object")
                col.prop(pg, "point_source_representation")
                if pg.point_source_representation == "RASTERIZED_REGION":
                    col.prop(pg, "point_rasterize_radius")
                    col.prop(pg, "mesh_evaluation_mode", text="Evaluation Cadence")
            elif kind == "VOLUME_GRID":
                _draw_volume_input(col, pg)
                col.prop(pg, "mesh_evaluation_mode", text="Evaluation Cadence")
            if kind != "POINTS" or pg.point_source_representation == "RASTERIZED_REGION":
                col.prop(pg, "emit_rate", text="Target Particles / Voxel")
            col.prop(pg, "initial_velocity")
            col.prop(pg, "use_object_velocity", text="Add Animated Object Velocity")
            _draw_source_status(box, pg)
        else:
            layout.label(text="Select a tagged Liquid Emitter", icon="INFO")
        tagged = [o for o in context.scene.objects
                  if _presentation_role(o.fluidsim) == "SOURCE"]
        if tagged:
            box = layout.box()
            box.label(text="Scene Emitters")
            for item in tagged:
                box.label(text=item.name, icon="OBJECT_DATA")


class FLUIDSIM_PT_liquid_emitter_attributes(FluidSimPanel, bpy.types.Panel):
    bl_label = "Particle Attributes"
    bl_idname = "FLUIDSIM_PT_liquid_emitter_attributes"
    bl_parent_id = "FLUIDSIM_PT_liquid_emitters"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and _presentation_role(obj.fluidsim) == "SOURCE"

    def draw(self, context):
        pg = context.active_object.fluidsim
        col = self.layout.column(align=True)
        col.prop(pg, "temperature")
        col.prop(pg, "viscosity")
        col.prop(pg, "density", text="Density Attribute")
        col.prop(pg, "use_lifespan")
        if pg.use_lifespan:
            row = col.row(align=True)
            row.prop(pg, "life_min")
            row.prop(pg, "life_max")


class FLUIDSIM_PT_liquid_sinks(FluidSimPanel, bpy.types.Panel):
    bl_label = "Liquid Sinks"
    bl_idname = "FLUIDSIM_PT_liquid_sinks"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        _draw_role_actions(layout, "fluidsim.tag_liquid_sink")
        obj = context.active_object
        if obj is not None and _presentation_role(obj.fluidsim) == "SINK":
            pg = obj.fluidsim
            box = layout.box()
            box.label(text=f"{obj.name} (Liquid Sink)", icon="MOD_FLUIDSIM")
            col = box.column(align=True)
            col.prop(pg, "enabled")
            legacy = int(getattr(pg, "source_ui_schema_version", 0) or 0) < 3
            if legacy:
                region = col.row(align=True)
                region.prop_enum(pg, "geometry_mode", "ANALYTIC", text="Analytic Volume")
                region.prop_enum(pg, "geometry_mode", "MESH", text="Mesh")
                kind = ("ANALYTIC" if pg.geometry_mode == "ANALYTIC" else
                        ("MESH_SHELL" if pg.sink_region_mode == "SURFACE" else "MESH_SOLID"))
                if pg.geometry_mode == "MESH":
                    col.prop(pg, "sink_region_mode")
            else:
                region = col.row(align=True)
                region.prop_enum(pg, "sink_region_kind", "ANALYTIC", text="Analytic Volume")
                region.prop_enum(pg, "sink_region_kind", "MESH_SOLID", text="Mesh Solid Interior")
                region.prop_enum(pg, "sink_region_kind", "MESH_SHELL", text="Mesh Surface Shell")
                region.prop_enum(pg, "sink_region_kind", "VOLUME_GRID", text="Volume Grid Mask")
                kind = pg.sink_region_kind
            if kind == "ANALYTIC":
                col.prop(pg, "shape")
            elif kind in {"MESH_SOLID", "MESH_SHELL"}:
                _draw_mesh_input(col, pg, solid=kind == "MESH_SOLID", shell=kind == "MESH_SHELL")
            elif kind == "VOLUME_GRID":
                _draw_volume_input(col, pg)
            _draw_source_status(box, pg)
        else:
            layout.label(text="Select a tagged Liquid Sink", icon="INFO")


class FLUIDSIM_PT_colliders(FluidSimPanel, bpy.types.Panel):
    bl_label = "Colliders"
    bl_idname = "FLUIDSIM_PT_colliders"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        _draw_role_actions(layout, "fluidsim.tag_collider")
        obj = context.active_object
        if obj is not None and _presentation_role(obj.fluidsim) == "COLLIDER":
            pg = obj.fluidsim
            box = layout.box()
            box.label(text=obj.name, icon="MOD_PHYSICS")
            col = box.column(align=True)
            col.prop(pg, "enabled")
            col.prop(pg, "collider_friction")
            col.prop(pg, "collider_restitution")
            col.prop(pg, "transfer_velocity", text="Transfer Animated Velocity")
            if pg.mesh_error:
                error = col.box()
                error.alert = True
                error.label(text=pg.mesh_error, icon="ERROR")
        else:
            layout.label(text="Select a tagged mesh Collider", icon="INFO")


class FLUIDSIM_PT_forces(FluidSimPanel, bpy.types.Panel):
    bl_label = "Forces"
    bl_idname = "FLUIDSIM_PT_forces"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        fs = context.scene.fluidsim
        global_box = layout.box()
        global_box.label(text="Global Grid Forces", icon="FORCE_FORCE")
        global_box.prop(fs, "directional_force")
        global_box.prop(fs, "omnidirectional_force_strength")
        radial = global_box.column(align=True)
        radial.enabled = fs.omnidirectional_force_strength != 0.0
        radial.prop(fs, "omnidirectional_force_center")
        radial.prop(fs, "omnidirectional_force_radius")
        global_box.prop(fs, "force_diffusion")
        global_box.prop(fs, "dissipation_enabled")
        dissipation = global_box.column(align=True)
        dissipation.enabled = fs.dissipation_enabled
        dissipation.prop(fs, "dissipation_attribute")
        dissipation.prop(fs, "dissipation_rate")
        dissipation.prop(fs, "dissipation_target")
        global_box.prop(fs, "attraction_enabled")
        attraction = global_box.column(align=True)
        attraction.enabled = fs.attraction_enabled
        attraction.prop(fs, "attraction_strength")
        attraction.prop(fs, "attraction_max_distance")
        physics = layout.box()
        physics.label(text="Blender Force Fields", icon="PHYSICS")
        _draw_role_actions(layout, "fluidsim.tag_force_region")
        if obj is not None and _presentation_role(obj.fluidsim) == "FORCE":
            pg = obj.fluidsim
            box = layout.box()
            box.label(text=obj.name, icon="FORCE_FORCE")
            col = box.column(align=True)
            col.prop(pg, "enabled")
            col.prop(pg, "shape")
            col.prop(pg, "force_type")
            sub = col.column(align=True)
            sub.enabled = pg.force_type != "OFF"
            sub.prop(pg, "force_strength")
            if pg.force_type == "DIRECTIONAL":
                sub.prop(pg, "force_direction")
            elif pg.force_type == "SPIN":
                sub.prop(pg, "force_direction", text="Spin Axis")
        else:
            layout.label(text="Select a tagged Force Region object", icon="INFO")


class FLUIDSIM_PT_domain_height_fill(FluidSimPanel, bpy.types.Panel):
    bl_label = "Domain Height Fill"
    bl_idname = "FLUIDSIM_PT_domain_height_fill"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        fs = context.scene.fluidsim
        layout = self.layout
        layout.prop(fs, "height_enabled", text="Enable Domain Height Fill")
        col = layout.column(align=True)
        col.enabled = fs.height_enabled
        col.prop(fs, "height_base", text="Fill Level")
        row = col.row(align=True)
        row.prop_enum(fs, "height_back_mode", "FLAT", text="Flat")
        row.prop_enum(fs, "height_back_mode", "PROFILE", text="Profile Image")
        if fs.height_back_mode == "PROFILE":
            col.prop(fs, "height_profile_image")
            col.prop(fs, "height_profile_scale")
        col.prop(fs, "height_emission_mode", text="Fill Lifecycle")
        col.prop(fs, "height_velocity", text="Initial Velocity")
        col.prop(fs, "height_emit_particles")
        col.prop(fs, "particles_per_voxel")
        if getattr(fs, "height_error", ""):
            error = col.box()
            error.alert = True
            error.label(text=fs.height_error, icon="ERROR")


class FLUIDSIM_PT_point_particle_injection(FluidSimPanel, bpy.types.Panel):
    bl_label = "Point / Particle Injection"
    bl_idname = "FLUIDSIM_PT_point_particle_injection"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.operator("fluidsim.inject_particles", text="Inject Active Object Points Once", icon="PARTICLES")
        obj = context.active_object
        if obj is not None:
            layout.prop(obj.fluidsim, "auto_inject_seed", text="Reinject on Initialize / Reset")


class FLUIDSIM_PT_advanced_field_sources(FluidSimPanel, bpy.types.Panel):
    bl_label = "Advanced Field Sources"
    bl_idname = "FLUIDSIM_PT_advanced_field_sources"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        _draw_role_actions(layout, "fluidsim.tag_field_source")
        obj = context.active_object
        if obj is None or _presentation_role(obj.fluidsim) != "FIELD":
            layout.label(text="Select a tagged Advanced Field Source", icon="INFO")
            return
        pg = obj.fluidsim
        box = layout.box()
        box.label(text=f"{obj.name} (Non-liquid)", icon="FORCE_FORCE")
        col = box.column(align=True)
        col.prop(pg, "enabled")
        col.prop(pg, "field_operation")
        col.prop(pg, "field_region_kind")
        if pg.field_region_kind == "ANALYTIC":
            col.prop(pg, "shape")
        else:
            _draw_volume_input(col, pg, fields=True)
        col.prop(pg, "field_value_source")
        if pg.field_value_source == "VOLUME_GRID":
            if pg.field_region_kind != "VOLUME_GRID":
                _draw_volume_input(col, pg, fields=True)
            if pg.field_operation in {"PRESSURE", "SCALAR"}:
                col.prop(pg, "volume_scalar_grid")
            else:
                col.prop(pg, "volume_vector_grid")
                if not pg.volume_vector_grid:
                    col.prop(pg, "volume_vector_x_grid")
                    col.prop(pg, "volume_vector_y_grid")
                    col.prop(pg, "volume_vector_z_grid")
        elif pg.field_operation in {"PRESSURE", "SCALAR"}:
            col.prop(pg, "field_scalar_value")
        else:
            col.prop(pg, "field_vector_value")
        col.prop(pg, "blend_mode", text="Blend")
        col.prop(pg, "field_amount")
        _draw_source_status(box, pg)


class FLUIDSIM_PT_pressure(FluidSimPanel, bpy.types.Panel):
    bl_label = "Pressure Projection"
    bl_idname = "FLUIDSIM_PT_pressure"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        _draw_interaction_tier(layout, "Live")
        col = layout.column(align=True)
        col.prop(fs, "pressure_solver")
        col.prop(fs, "pressure_iterations")
        col.prop(fs, "pressure_tolerance")
        col.prop(fs, "pressure_threshold")


class FLUIDSIM_PT_viscosity(FluidSimPanel, bpy.types.Panel):
    bl_label = "Viscosity"
    bl_idname = "FLUIDSIM_PT_viscosity"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.fluidsim, "viscosity_enabled", text="")

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        _draw_interaction_tier(layout, "Live")
        layout.enabled = fs.viscosity_enabled
        col = layout.column(align=True)
        col.prop(fs, "viscosity_solver")
        col.prop(fs, "viscosity_constant")
        col.prop(fs, "viscosity_from_particles")
        col.prop(fs, "viscosity_iterations")
        col.prop(fs, "viscosity_tolerance")


class FLUIDSIM_PT_surface(FluidSimPanel, bpy.types.Panel):
    bl_label = "Whitewater"
    bl_idname = "FLUIDSIM_PT_surface"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim

        layout.prop(fs, "whitewater_enabled")
        _draw_pending_banner(layout, context.scene)
        physics = layout.column(align=True)
        physics.enabled = fs.whitewater_enabled
        _draw_interaction_tier(physics, "Reinitialize Required", "ERROR")
        physics.prop(fs, "max_whitewater")
        _draw_interaction_tier(physics, "Live")
        physics.prop(fs, "ww_churn_gain")
        physics.prop(fs, "ww_churn_decay")
        physics.prop(fs, "ww_churn_threshold")
        physics.prop(fs, "ww_emission_rate")
        physics.prop(fs, "ww_min_speed")
        physics.separator()
        row = physics.row(align=True)
        row.prop(fs, "ww_life_min")
        row.prop(fs, "ww_life_max")
        row = physics.row(align=True)
        row.prop(fs, "ww_bubble_drag")
        row.prop(fs, "ww_spray_drag")

        layout.separator()
        layout.prop(fs, "show_whitewater", text="Show Whitewater in Viewport")
        display = layout.column(align=True)
        display.enabled = fs.show_whitewater
        display.prop(fs, "whitewater_display_scale")
        display.prop(fs, "whitewater_attribute_mode")
        display.prop(fs, "whitewater_material")


class FLUIDSIM_PT_realtime_display(FluidSimPanel, bpy.types.Panel):
    bl_label = "Realtime Surfacing / Particles"
    bl_idname = "FLUIDSIM_PT_realtime_display"

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim

        _draw_interaction_tier(layout, "Live / Targeted Refresh")
        layout.prop(fs, "liquid_view_mode")

        surface = layout.box()
        surface.label(text="Surface Mesh", icon="MESH_DATA")
        surface.prop(fs, "surface_enabled", text="Build Surface Mesh")
        settings = surface.column(align=True)
        settings.enabled = fs.surface_enabled
        settings.prop(fs, "surface_resolution_mode")
        if fs.surface_resolution_mode == "MULTIPLIER":
            settings.prop(fs, "surface_resolution_multiplier")
        elif fs.surface_resolution_mode == "CUSTOM":
            settings.prop(fs, "surface_voxel_size")
        settings.prop(fs, "particle_radius_scale")
        settings.prop(fs, "surface_splat_radius")
        settings.separator()
        settings.prop(fs, "surface_filter_type")
        row = settings.row(align=True)
        row.prop(fs, "surface_filter_iterations", text="Filter")
        row.prop(fs, "surface_filter_strength", text="Strength")
        row = settings.row(align=True)
        row.prop(fs, "surface_dilate_iterations", text="Dilate")
        row.prop(fs, "surface_dilate_strength", text="Strength")
        row = settings.row(align=True)
        row.prop(fs, "surface_erode_iterations", text="Erode")
        row.prop(fs, "surface_erode_strength", text="Strength")
        settings.prop(fs, "surface_smooth_type")
        row = settings.row(align=True)
        row.prop(fs, "surface_smooth_iterations", text="Smooth")
        row.prop(fs, "surface_smooth_strength", text="Strength")
        settings.separator()
        settings.prop(fs, "surface_update_cadence")
        settings.prop(fs, "live_surface_publish_cadence")

        surface.separator()
        surface.prop(fs, "show_surface", text="Show Surface in Viewport")
        display = surface.column(align=True)
        display.enabled = fs.show_surface
        display.prop(fs, "surface_attribute_mode")
        display.prop(fs, "surface_material")

        row = surface.row(align=True)
        row.prop(
            fs, "surface_advanced_expanded", text="Advanced Surface",
            icon=("TRIA_DOWN" if fs.surface_advanced_expanded
                  else "TRIA_RIGHT"), emboss=False)
        if fs.surface_advanced_expanded:
            advanced = surface.column(align=True)
            advanced.enabled = fs.surface_enabled
            advanced.prop(fs, "surface_leaf_capacity_scale", text="Leaf Scale")
            advanced.prop(fs, "surface_max_dense_dim", text="Dense Max")
            advanced.prop(fs, "surface_build_lowres_sdf")

        particles = layout.box()
        particles.label(text="Liquid Particles", icon="PARTICLE_DATA")
        particles.prop(fs, "show_particles", text="Show Liquid Particles")
        display = particles.column(align=True)
        display.enabled = fs.show_particles
        display.prop(fs, "particle_display_fraction")
        display.prop(fs, "particle_display_scale")
        display.prop(fs, "particle_attribute_mode")
        display.prop(fs, "particle_material")


class FLUIDSIM_PT_output(FluidSimPanel, bpy.types.Panel):
    bl_label = "Viewport / Render Output"
    bl_idname = "FLUIDSIM_PT_output"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        col = layout.column(align=True)
        col.prop(fs, "output_collection_name")
        col.prop(fs, "show_overlay")
        layout.separator()
        box = layout.box()
        box.label(text="Cache / Export", icon="DISK_DRIVE")
        box.prop(fs, "cache_path")
        box.prop(fs, "cache_enabled")
        box.prop(fs, "cache_playback_enabled")
        box.prop(fs, "cache_use_custom_range")
        sub = box.column(align=True)
        sub.enabled = fs.cache_use_custom_range
        row = sub.row(align=True)
        row.prop(fs, "cache_frame_start")
        row.prop(fs, "cache_frame_end")
        box.operator("fluidsim.export_cache_sequence", icon="RENDER_ANIMATION")
        box.operator("fluidsim.export_frame", icon="EXPORT")

def _snippet_state_icon(snip):
    """Icon reflecting validation state of a snippet's current source."""
    from .sim import custom

    code = snip.text_block.as_string() if snip.text_block else ""
    if snip.validated_hash != custom.code_hash(code):
        return "QUESTION"          # edited since last validation
    return "CHECKMARK" if snip.is_valid else "ERROR"


class FLUIDSIM_PT_snippets(FluidSimPanel, bpy.types.Panel):
    bl_label = "Custom Warp Snippets"
    bl_idname = "FLUIDSIM_PT_snippets"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim

        # -- Snippet library (Houdini-preset-style dropdown) ----------------
        box = layout.box()
        box.label(text="Snippet Library", icon="PRESET")
        row = box.row(align=True)
        row.prop(fs, "snippet_library_selection", text="")
        load_row = row.row(align=True)
        load_row.enabled = fs.snippet_library_selection not in {"", "NONE"}
        load_row.operator("fluidsim.snippet_load_preset", text="Load",
                          icon="IMPORT")

        # -- Snippet stack (layer multiple snippets, reorder by slot) -------
        row = layout.row(align=True)
        row.operator("fluidsim.snippet_add", icon="ADD")
        row.operator("fluidsim.snippet_remove", icon="REMOVE")
        row.operator("fluidsim.snippet_apply", icon="CHECKMARK")

        for i, snip in enumerate(fs.snippets):
            box = layout.box()
            row = box.row(align=True)
            row.prop(snip, "enabled", text="")
            row.prop(snip, "kernel_name", text="")
            row.label(text="", icon=_snippet_state_icon(snip))
            if i != fs.active_snippet:
                op = row.operator("fluidsim.snippet_select", text="",
                                  icon="RESTRICT_SELECT_OFF")
                op.index = i
            if i == fs.active_snippet:
                col = box.column(align=True)
                col.prop(snip, "order")
                col.prop(snip, "run_over")
                row = col.row(align=True)
                row.prop(snip, "frame_start", text="Start")
                row.prop(snip, "frame_end", text="End")
                row.prop(snip, "frame_step", text="Step")
                col.prop(snip, "text_block")
                col.separator()
                row = col.row(align=True)
                row.operator("fluidsim.snippet_argument_add",
                             text="Add Arg", icon="ADD")
                row.operator("fluidsim.snippet_argument_remove",
                             text="Remove Arg", icon="REMOVE")
                for arg in snip.arguments:
                    arow = col.row(align=True)
                    arow.prop(arg, "mode", text="")
                    if arg.mode == "FIELD":
                        arow.prop(arg, "field", text="")
                    else:
                        arow.prop(arg, "constant_name", text="")
                        arow.prop(arg, "constant_value", text="")
                if snip.last_error:
                    col.label(text=snip.last_error[:64], icon="ERROR")


class FLUIDSIM_PT_snippet_editor(FluidSimPanel, bpy.types.Panel):
    bl_label = "Snippet Editor"
    bl_idname = "FLUIDSIM_PT_snippet_editor"
    bl_parent_id = "FLUIDSIM_PT_snippets"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        if not (0 <= fs.active_snippet < len(fs.snippets)):
            layout.label(text="Add or load a snippet first", icon="INFO")
            return
        snip = fs.snippets[fs.active_snippet]

        row = layout.row(align=True)
        row.operator("fluidsim.snippet_validate", icon="CHECKMARK")
        row.operator("fluidsim.snippet_autofix", icon="TOOL_SETTINGS")
        row.operator("fluidsim.snippet_test", icon="EXPERIMENTAL")
        row.operator("fluidsim.snippet_save_preset", icon="FILE_TICK")

        code = snip.text_block.as_string() if snip.text_block else ""
        box = layout.box()
        header = box.row(align=True)
        header.label(text=f"Source ({snip.kernel_name})",
                     icon=_snippet_state_icon(snip))
        header.operator("fluidsim.snippet_line_add", text="", icon="ADD")
        lines = code.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        shown = lines[: fs.snippet_editor_lines]
        col = box.column(align=True)
        if not shown:
            col.label(text="(empty - use Add Line or the Shortcuts below)")
        for idx, line in enumerate(shown):
            row = col.row(align=True)
            row.label(text=f"{idx + 1:>3}  {line[:88]}")
            op = row.operator("fluidsim.snippet_edit_line", text="",
                              icon="GREASEPENCIL", emboss=False)
            op.line_index = idx
            op = row.operator("fluidsim.snippet_line_delete", text="",
                              icon="X", emboss=False)
            op.line_index = idx
        if len(lines) > len(shown):
            col.label(text=f"... {len(lines) - len(shown)} more line(s); "
                           f"raise Editor Lines or use the Text Editor")
        box.prop(fs, "snippet_editor_lines")

        if snip.validation_report:
            box = layout.box()
            box.label(text="Validation", icon="INFO")
            for line in snip.validation_report.split("\n")[:8]:
                icon = "ERROR" if line.startswith("ERROR") else "INFO"
                box.label(text=line[:96], icon=icon)


class FLUIDSIM_PT_snippet_shortcuts(FluidSimPanel, bpy.types.Panel):
    bl_label = "Shortcuts"
    bl_idname = "FLUIDSIM_PT_snippet_shortcuts"
    bl_parent_id = "FLUIDSIM_PT_snippets"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from .sim import custom

        layout = self.layout
        fs = context.scene.fluidsim
        if not (0 <= fs.active_snippet < len(fs.snippets)):
            layout.label(text="Add or load a snippet first", icon="INFO")
            return
        snip = fs.snippets[fs.active_snippet]

        # Bind a solver field / named constant (adds argument + code line).
        box = layout.box()
        box.label(text="Bind Data", icon="RNA")
        row = box.row(align=True)
        row.prop(fs, "shortcut_field", text="")
        row.operator("fluidsim.snippet_shortcut_field", text="Add Field")
        row = box.row(align=True)
        row.prop(fs, "shortcut_constant_name", text="")
        row.prop(fs, "shortcut_constant_value", text="")
        row.operator("fluidsim.snippet_shortcut_constant", text="Add")

        # One-click code shortcuts grouped by category.
        grouped = {}
        for shortcut in custom.shortcuts_for(snip.run_over):
            grouped.setdefault(shortcut.category, []).append(shortcut)
        for category in ("Context", "Time", "Math", "Constants"):
            shortcuts = grouped.pop(category, None)
            if not shortcuts:
                continue
            box = layout.box()
            box.label(text=category, icon="DOT")
            col = box.column(align=True)
            for i in range(0, len(shortcuts), 2):
                row = col.row(align=True)
                for shortcut in shortcuts[i:i + 2]:
                    op = row.operator("fluidsim.snippet_insert_shortcut",
                                      text=shortcut.label)
                    op.shortcut_id = shortcut.shortcut_id


class FLUIDSIM_PT_diagnostics(FluidSimPanel, bpy.types.Panel):
    bl_label = "Diagnostics / Profiling"
    bl_idname = "FLUIDSIM_PT_diagnostics"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        from .blender.scene_bridge import session

        sess = session()
        col = layout.column(align=True)
        col.prop(fs, "profiling_enabled")
        sub = col.column(align=True)
        sub.enabled = fs.profiling_enabled
        sub.prop(fs, "profiling_sync")
        col.prop(fs, "report_path")
        col.operator("fluidsim.write_report", icon="TEXT")

        if sess.profiler is not None and sess.profiler.history:
            box = layout.box()
            box.label(text="Average timers (ms):", icon="TIME")
            summary = sess.profiler.summary(last_n=30)
            names = {
                f"compute_{slot}": phase.replace("_", " ").title()
                for phase, slot in TIMER_PHASE_SLOTS.items()
            }
            for slot, label in names.items():
                v = summary.get(slot, 0.0)
                if v > 0.01:
                    box.label(text=f"{label}: {v:.2f}")
        elif fs.profiling_enabled and not sess.active:
            layout.label(text="Initialize the solver to profile", icon="INFO")


class FLUIDSIM_PT_validation(FluidSimPanel, bpy.types.Panel):
    bl_label = "Validation / Benchmarks"
    bl_idname = "FLUIDSIM_PT_validation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        from .blender.scene_bridge import session

        sess = session()
        layout.operator("fluidsim.run_validation", icon="CHECKMARK")
        if sess.validation_results:
            box = layout.box()
            passed = sum(1 for r in sess.validation_results if r["passed"])
            box.label(text=f"{passed}/{len(sess.validation_results)} passed")
            for r in sess.validation_results:
                icon = "CHECKMARK" if r["passed"] else "CANCEL"
                box.label(text=r["name"][:44], icon=icon)

        layout.separator()
        col = layout.column(align=True)
        col.prop(fs, "benchmark_frames")
        col.prop(fs, "benchmark_baseline")
        row = col.row(align=True)
        op = row.operator("fluidsim.run_benchmark", text="Run Benchmark")
        op.save_baseline = False
        op = row.operator("fluidsim.run_benchmark", text="Save Baseline")
        op.save_baseline = True
        if sess.benchmark_result:
            box = layout.box()
            d = sess.benchmark_result
            box.label(text=f"Avg step: {d['timeStepAverage'] * 1000:.1f} ms")
            box.label(text=f"Voxels/s: {d['voxelsPerSecondAverage'] / 1e6:.1f} M")
            box.label(text=f"Active voxels avg: {d['activeVoxelsAverage']:,}")
            if sess.benchmark_compare:
                box.label(text=f"vs baseline: {sess.benchmark_compare['result']}")


CLASSES = (
    FLUIDSIM_PT_interaction_popover,
    FLUIDSIM_PT_simulation,
    FLUIDSIM_PT_domain,
    FLUIDSIM_PT_particles,
    FLUIDSIM_PT_liquid_emitters,
    FLUIDSIM_PT_liquid_emitter_attributes,
    FLUIDSIM_PT_liquid_sinks,
    FLUIDSIM_PT_colliders,
    FLUIDSIM_PT_forces,
    FLUIDSIM_PT_domain_height_fill,
    FLUIDSIM_PT_point_particle_injection,
    FLUIDSIM_PT_advanced_field_sources,
    FLUIDSIM_PT_pressure,
    FLUIDSIM_PT_viscosity,
    FLUIDSIM_PT_surface,
    FLUIDSIM_PT_realtime_display,
    FLUIDSIM_PT_output,
    FLUIDSIM_PT_snippets,
    FLUIDSIM_PT_snippet_editor,
    FLUIDSIM_PT_snippet_shortcuts,
    FLUIDSIM_PT_diagnostics,
    FLUIDSIM_PT_validation,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    header = getattr(bpy.types, "VIEW3D_HT_header", None)
    if header is not None:
        try:
            header.remove(_draw_header_interaction)
        except Exception:
            pass
        header.append(_draw_header_interaction)


def unregister():
    header = getattr(bpy.types, "VIEW3D_HT_header", None)
    if header is not None:
        try:
            header.remove(_draw_header_interaction)
        except Exception:
            pass
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
