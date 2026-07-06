"""UI panels for the FluidSim Warp add-on (3D Viewport sidebar > FluidSim).

Panel set mirrors the synthesized UI contract:
Simulation, Domain / Sparse Grid, Particles, Sources, Colliders / Sinks,
Forces / Attraction, Pressure Projection, Viscosity, Surface / Whitewater,
Realtime Surfacing / Particles, Viewport / Render Output, Custom Warp
Snippets, Diagnostics / Profiling, Validation / Benchmarks.
"""

from __future__ import annotations

import bpy


class FluidSimPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FluidSim"


class FLUIDSIM_PT_simulation(FluidSimPanel, bpy.types.Panel):
    bl_label = "Simulation"
    bl_idname = "FLUIDSIM_PT_simulation"

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        from .blender.scene_bridge import current_error, session

        sess = session()

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

        box = layout.box()
        box.label(text=fs.status, icon="INFO")
        error = current_error(context.scene)
        if error:
            box.label(text=error[:120], icon="ERROR")
            box.operator("fluidsim.clear_error", icon="X")
        if sess.active:
            col = box.column(align=True)
            col.label(text=f"Particles: {fs.stat_particles:,}")
            col.label(text=f"Active voxels: {fs.stat_voxels:,}")
            col.label(text=f"Step: {fs.stat_step_ms:.1f} ms "
                           f"({fs.stat_substeps} substeps)")

        col = layout.column(align=True)
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
        layout.prop(fs, "domain_object")
        col = layout.column(align=True)
        col.prop(fs, "voxel_size")
        col.prop(fs, "leaf_capacity")
        col.prop(fs, "topology_expand")
        col.prop(fs, "open_boundaries")
        from .blender.scene_bridge import _domain_bounds, session

        lo, hi = _domain_bounds(context.scene)
        dims = tuple(hi[i] - lo[i] for i in range(3))
        vox = tuple(int(d / max(fs.voxel_size, 1e-6)) for d in dims)
        box = layout.box()
        box.label(text=f"Resolution: {vox[0]} x {vox[1]} x {vox[2]}")
        sess = session()
        if sess.active:
            g = sess.solver.grid
            box.label(text=f"Leaves: {g.active_leaf_count}/{g.capacity}")


class FLUIDSIM_PT_particles(FluidSimPanel, bpy.types.Panel):
    bl_label = "Particles"
    bl_idname = "FLUIDSIM_PT_particles"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        col = layout.column(align=True)
        col.prop(fs, "max_particles")
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


class FLUIDSIM_PT_sources(FluidSimPanel, bpy.types.Panel):
    bl_label = "Sources"
    bl_idname = "FLUIDSIM_PT_sources"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        scene = context.scene

        row = layout.row(align=True)
        op = row.operator("fluidsim.tag_object", text="Tag Source", icon="ADD")
        op.role = "SOURCE"
        op = row.operator("fluidsim.tag_object", text="Clear", icon="X")
        op.role = "NONE"
        layout.operator("fluidsim.inject_particles", icon="PARTICLES")

        # Active object source settings.
        obj = context.active_object
        if obj is not None and obj.fluidsim.role in {"SOURCE", "SINK", "FORCE"}:
            pg = obj.fluidsim
            box = layout.box()
            box.label(text=f"{obj.name} ({pg.role.title()})", icon="OBJECT_DATA")
            col = box.column(align=True)
            col.prop(pg, "enabled")
            col.prop(pg, "shape")
            col.prop(pg, "source_mode")
            if pg.role == "SOURCE":
                col.prop(pg, "emit_rate")
                col.prop(pg, "initial_velocity")
                col.prop(pg, "use_object_velocity")
                col.prop(pg, "temperature")
                col.prop(pg, "viscosity")
                col.prop(pg, "use_lifespan")
                if pg.use_lifespan:
                    row = col.row(align=True)
                    row.prop(pg, "life_min")
                    row.prop(pg, "life_max")

        # Height source.
        box = layout.box()
        box.prop(fs, "height_enabled")
        if fs.height_enabled:
            col = box.column(align=True)
            col.prop(fs, "height_base")
            col.prop(fs, "height_back_mode")
            col.prop(fs, "height_velocity")
            col.prop(fs, "height_emit_particles")

        # List of tagged sources.
        tagged = [o for o in scene.objects
                  if o.fluidsim.role in {"SOURCE", "SINK", "FORCE"}]
        if tagged:
            box = layout.box()
            for o in tagged:
                row = box.row()
                row.label(text=o.name,
                          icon="OUTLINER_OB_MESH" if o.type == "MESH"
                          else "OBJECT_DATA")
                row.label(text=o.fluidsim.role.title())


class FLUIDSIM_PT_colliders(FluidSimPanel, bpy.types.Panel):
    bl_label = "Colliders / Sinks"
    bl_idname = "FLUIDSIM_PT_colliders"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        row = layout.row(align=True)
        op = row.operator("fluidsim.tag_object", text="Tag Collider", icon="ADD")
        op.role = "COLLIDER"
        op = row.operator("fluidsim.tag_object", text="Tag Sink", icon="ADD")
        op.role = "SINK"

        obj = context.active_object
        if obj is not None and obj.fluidsim.role == "COLLIDER":
            pg = obj.fluidsim
            box = layout.box()
            box.label(text=obj.name, icon="MOD_PHYSICS")
            col = box.column(align=True)
            col.prop(pg, "enabled")
            col.prop(pg, "collider_friction")
            col.prop(pg, "collider_restitution")
            col.prop(pg, "transfer_velocity")

        col = layout.column(align=True)
        col.prop(fs, "restitution")
        col.prop(fs, "friction")

        tagged = [o for o in context.scene.objects
                  if o.fluidsim.role == "COLLIDER"]
        if tagged:
            box = layout.box()
            for o in tagged:
                box.label(text=o.name, icon="MESH_ICOSPHERE")


class FLUIDSIM_PT_forces(FluidSimPanel, bpy.types.Panel):
    bl_label = "Forces / Attraction"
    bl_idname = "FLUIDSIM_PT_forces"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.label(text="Per-object force regions (tag as Source):")
        if obj is not None and obj.fluidsim.role in {"SOURCE", "FORCE"}:
            pg = obj.fluidsim
            box = layout.box()
            box.label(text=obj.name, icon="FORCE_FORCE")
            col = box.column(align=True)
            col.prop(pg, "force_type")
            sub = col.column(align=True)
            sub.enabled = pg.force_type != "OFF"
            sub.prop(pg, "force_strength")
            if pg.force_type == "DIRECTIONAL":
                sub.prop(pg, "force_direction")
        else:
            layout.label(text="Select a tagged source object", icon="INFO")


class FLUIDSIM_PT_pressure(FluidSimPanel, bpy.types.Panel):
    bl_label = "Pressure Projection"
    bl_idname = "FLUIDSIM_PT_pressure"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        col = layout.column(align=True)
        col.prop(fs, "pressure_solver")
        col.prop(fs, "pressure_iterations")
        col.prop(fs, "pressure_tolerance")
        col.prop(fs, "multigrid")
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
        layout.enabled = fs.viscosity_enabled
        col = layout.column(align=True)
        col.prop(fs, "viscosity_solver")
        col.prop(fs, "viscosity_constant")
        col.prop(fs, "viscosity_from_particles")
        col.prop(fs, "viscosity_iterations")
        col.prop(fs, "viscosity_tolerance")


class FLUIDSIM_PT_surface(FluidSimPanel, bpy.types.Panel):
    bl_label = "Surface / Whitewater"
    bl_idname = "FLUIDSIM_PT_surface"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim
        col = layout.column(align=True)
        sub = col.column(align=True)
        sub.prop(fs, "max_whitewater")
        sub.prop(fs, "ww_churn_gain")
        sub.prop(fs, "ww_churn_decay")
        sub.prop(fs, "ww_churn_threshold")
        sub.prop(fs, "ww_emission_rate")
        sub.prop(fs, "ww_min_speed")
        row = sub.row(align=True)
        row.prop(fs, "ww_life_min")
        row.prop(fs, "ww_life_max")
        row = sub.row(align=True)
        row.prop(fs, "ww_bubble_drag")
        row.prop(fs, "ww_spray_drag")


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
        box.label(text="Optional Cache / Export", icon="DISK_DRIVE")
        box.prop(fs, "cache_enabled")
        sub = box.column(align=True)
        sub.enabled = fs.cache_enabled
        sub.prop(fs, "cache_path")
        box.operator("fluidsim.export_frame", icon="EXPORT")


class FLUIDSIM_PT_realtime_display(FluidSimPanel, bpy.types.Panel):
    bl_label = "Realtime Surfacing / Particles"
    bl_idname = "FLUIDSIM_PT_realtime_display"

    def draw(self, context):
        layout = self.layout
        fs = context.scene.fluidsim

        layout.prop(fs, "liquid_view_mode")

        box = layout.box()
        box.label(text="Surface Mesh", icon="MESH_DATA")
        box.prop(fs, "show_surface")
        sub = box.column(align=True)
        sub.enabled = fs.show_surface
        sub.prop(fs, "surface_enabled")
        sub.prop(fs, "particle_radius_scale")
        row = sub.row(align=True)
        row.prop(fs, "surface_dilate_iterations", text="Dilate")
        row.prop(fs, "surface_dilate_strength", text="Strength")
        row = sub.row(align=True)
        row.prop(fs, "surface_erode_iterations", text="Erode")
        row.prop(fs, "surface_erode_strength", text="Strength")
        sub.prop(fs, "surface_smooth_type")
        row = sub.row(align=True)
        row.prop(fs, "surface_smooth_iterations", text="Smooth")
        row.prop(fs, "surface_smooth_strength", text="Strength")
        sub.prop(fs, "surface_attribute_mode")
        sub.prop(fs, "surface_material")

        box = layout.box()
        box.label(text="Liquid Particles", icon="PARTICLE_DATA")
        box.prop(fs, "show_particles")
        sub = box.column(align=True)
        sub.enabled = fs.show_particles
        sub.prop(fs, "particle_display_fraction")
        sub.prop(fs, "particle_display_scale")
        sub.prop(fs, "particle_attribute_mode")
        sub.prop(fs, "particle_material")

        box = layout.box()
        box.label(text="Whitewater Particles", icon="OUTLINER_OB_POINTCLOUD")
        box.prop(fs, "show_whitewater")
        sub = box.column(align=True)
        sub.enabled = fs.show_whitewater
        sub.prop(fs, "whitewater_enabled")
        sub.prop(fs, "whitewater_display_scale")
        sub.prop(fs, "whitewater_attribute_mode")
        sub.prop(fs, "whitewater_material")


def _snippet_state_icon(snip):
    """Icon reflecting validation state of a snippet's current source."""
    from .sim import snippet_library

    code = snip.text_block.as_string() if snip.text_block else ""
    if snip.validated_hash != snippet_library.code_hash(code):
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
        row.operator("fluidsim.snippet_load_preset", text="Load",
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
        from .sim import snippet_library

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
        for shortcut in snippet_library.shortcuts_for(snip.run_over):
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
                "compute_0": "Sourcing", "compute_1": "Indexing",
                "compute_2": "P2G", "compute_3": "Forces",
                "compute_4": "Viscosity", "compute_5": "Pressure",
                "compute_6": "G2P", "compute_8": "Collision",
                "compute_10": "Relax", "compute_11": "Surface",
                "compute_12": "Whitewater", "compute_13": "Custom",
                "compute_14": "Lifecycle", "compute_15": "Topology",
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
    FLUIDSIM_PT_simulation,
    FLUIDSIM_PT_domain,
    FLUIDSIM_PT_particles,
    FLUIDSIM_PT_sources,
    FLUIDSIM_PT_colliders,
    FLUIDSIM_PT_forces,
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


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
