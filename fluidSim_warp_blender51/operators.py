"""Operators for the FluidSim Warp add-on."""

from __future__ import annotations

import json
import re

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty


def _bridge():
    from .blender import scene_bridge

    return scene_bridge


def _configured_device():
    try:
        prefs = bpy.context.preferences.addons[
            "fluidSim_warp_blender51"].preferences
        if prefs.device:
            return prefs.device
    except (KeyError, AttributeError):
        pass
    return "cuda:0"


class FLUIDSIM_OT_initialize(bpy.types.Operator):
    """Create (or recreate) the GPU solver from current scene settings"""

    bl_idname = "fluidsim.initialize"
    bl_label = "Initialize Solver"
    bl_options = {"REGISTER"}

    ignore_memory_safeguard: BoolProperty(
        name="Run Anyway",
        description="Ignore the predictive memory safeguard; real CUDA allocation failures still stop initialization",
        default=False,
        options={"SKIP_SAVE"},
    )

    def invoke(self, context, event):
        if self.ignore_memory_safeguard:
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        try:
            sess = _bridge().initialize_session(
                scene,
                ignore_memory_safeguard=self.ignore_memory_safeguard,
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Solver initialization failed: {exc}")
            scene.fluidsim.status = f"Init failed: {exc}"
            return {"CANCELLED"}
        dev = sess.solver.device
        cap = sess.solver.grid.capacity
        self.report({"INFO"},
                    f"Solver ready on {dev} ({cap} leaves, "
                    f"{sess.solver.particles.capacity:,} particle capacity)")
        return {"FINISHED"}


class FLUIDSIM_OT_release(bpy.types.Operator):
    """Release the GPU solver and free device memory"""

    bl_idname = "fluidsim.release"
    bl_label = "Release Solver"

    def execute(self, context):
        _bridge().release_session()
        context.scene.fluidsim.status = "Solver released"
        return {"FINISHED"}


class FLUIDSIM_OT_clear_error(bpy.types.Operator):
    """Clear the persistent simulation error shown in the UI"""

    bl_idname = "fluidsim.clear_error"
    bl_label = "Clear Simulation Error"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        bridge = _bridge()
        bridge.clear_error(context.scene)
        fs = context.scene.fluidsim
        fs.status = "Solver ready" if bridge.session().active else "Solver not initialized"
        return {"FINISHED"}


class FLUIDSIM_OT_apply_safe_settings(bpy.types.Operator):
    """Apply machine-aware memory settings and retry solver initialization."""

    bl_idname = "fluidsim.apply_safe_settings"
    bl_label = "Apply Safe Settings & Retry"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from .sim.memory_safety import (
            assess_configuration,
            estimate_solver_bytes,
            recommend_safe_settings,
            resource_snapshot,
        )

        bridge = _bridge()
        scene = context.scene
        fs = scene.fluidsim
        settings = bridge.build_settings(scene)
        resources = resource_snapshot(settings.device)
        assessment = assess_configuration(settings, resources)
        budget = assessment["budget_bytes"]
        recommended = dict(assessment["recommended"])

        error = bridge.current_error(scene)
        overflow = re.search(
            r"leaf capacity exceeded \((\d+) > (\d+)\)", error, re.IGNORECASE
        )
        if overflow:
            required = int(overflow.group(1))
            candidate = min(262_144, 1 << max(6, (required - 1).bit_length()))
            candidate_settings = vars(settings).copy()
            candidate_settings["leaf_capacity"] = candidate
            if estimate_solver_bytes(candidate_settings) <= budget:
                recommended["leaf_capacity"] = candidate
            else:
                ratio = max(1.0, required / max(int(settings.leaf_capacity), 1))
                recommended["voxel_size"] = float(settings.voxel_size) * (
                    ratio ** (1.0 / 3.0)
                ) * 1.1

        if not recommended:
            recovery_budget = min(
                budget, max(512 * 1024**2, int(assessment["estimated_bytes"] * 0.75))
            )
            recommended = recommend_safe_settings(settings, recovery_budget, {
                **resources,
                "gpu_utilization": max(85.0, resources["gpu_utilization"]),
            })
        if not recommended:
            self.report({"ERROR"}, "Safe Settings could not find a lower-risk configuration")
            return {"CANCELLED"}

        changed = []
        for name, value in recommended.items():
            if name == "surface_resolution_mode":
                value = "SIMULATION"
            if hasattr(fs, name):
                rna_property = fs.bl_rna.properties.get(name)
                if rna_property is not None and rna_property.type == "INT":
                    value = int(value)
                setattr(fs, name, value)
                changed.append(name)

        bridge.clear_error(scene)
        fs.status = "Safe settings applied: " + ", ".join(changed)
        try:
            bridge.initialize_session(scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Safe retry stopped: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, fs.status)
        return {"FINISHED"}


class FLUIDSIM_OT_reset(bpy.types.Operator):
    """Reset the simulation to an empty state"""

    bl_idname = "fluidsim.reset"
    bl_label = "Reset Simulation"

    def execute(self, context):
        sess = _bridge().session()
        if not sess.active:
            self.report({"WARNING"}, "Solver not initialized")
            return {"CANCELLED"}
        _bridge().reset_simulation(context.scene)
        return {"FINISHED"}


class FLUIDSIM_OT_step(bpy.types.Operator):
    """Advance the simulation by a number of frames"""

    bl_idname = "fluidsim.step"
    bl_label = "Step Frame"

    frames: IntProperty(name="Frames", default=1, min=1, max=250)

    def execute(self, context):
        scene = context.scene
        bridge = _bridge()
        sess = bridge.session()
        if not sess.active:
            try:
                sess = bridge.initialize_session(scene)
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, f"Solver initialization failed: {exc}")
                return {"CANCELLED"}
        try:
            for _ in range(self.frames):
                bridge.sync_scene_to_solver(scene, sess)
                bridge.step_frame(scene, sess)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Step failed: {exc}")
            sess.error = repr(exc)
            return {"CANCELLED"}
        return {"FINISHED"}


class FLUIDSIM_OT_toggle_live(bpy.types.Operator):
    """Toggle live simulation during animation playback"""

    bl_idname = "fluidsim.toggle_live"
    bl_label = "Toggle Live Simulation"

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _bridge()
        if not fs.live:
            if not bridge.session().active:
                try:
                    bridge.initialize_session(scene)
                except Exception as exc:  # noqa: BLE001
                    self.report({"ERROR"}, f"Solver initialization failed: {exc}")
                    return {"CANCELLED"}
            fs.live = True
            fs.status = "Live simulation enabled - press play"
        else:
            fs.live = False
            fs.status = "Live simulation disabled"
        return {"FINISHED"}


class FLUIDSIM_OT_setup_scene(bpy.types.Operator):
    """Create a ready-to-play demo setup: domain, source, and collider"""

    bl_idname = "fluidsim.setup_scene"
    bl_label = "Quick Setup"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim

        # Domain box - extended horizontally (X) and slightly taller (Z)
        # so the released water has room to spread and fall.
        # X range: -6..+6, Y range: -2..+2, Z range: 0..+6.
        mesh = bpy.data.meshes.new("FluidSim Domain")
        domain = bpy.data.objects.new("FluidSim Domain", mesh)
        scene.collection.objects.link(domain)
        import bmesh

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()
        domain.scale = (12.0, 4.0, 6.0)
        domain.location = (0.0, 0.0, 3.0)
        domain.display_type = "WIRE"
        domain.hide_render = True
        fs.domain_object = domain
        fs.relax_mode = "LOCAL_VOXEL"

        # Particle source box - kept inside the extended domain.
        # Higher up so the water has drop potential; offset in -X so it
        # spreads toward +X after splash.
        src_mesh = bpy.data.meshes.new("FluidSim Source")
        src = bpy.data.objects.new("FluidSim Source", src_mesh)
        scene.collection.objects.link(src)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(src_mesh)
        bm.free()
        src.location = (-4.0, 0.0, 4.0)
        src.scale = (1.2, 3.2, 2.2)
        from .blender import sources as _sources

        _sources.tag_source(src, shape="BOX", mode="PARTICLES")

        # Sphere collider - repositioned under the source so the stream
        # hits it and splashes outward into the longer +X run.
        col_mesh = bpy.data.meshes.new("FluidSim Collider")
        col = bpy.data.objects.new("FluidSim Collider", col_mesh)
        scene.collection.objects.link(col)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=0.75)
        bm.to_mesh(col_mesh)
        bm.free()
        col.location = (-2.0, 0.0, 0.75)
        from .blender import colliders as _colliders

        _colliders.tag_collider(col)

        scene.frame_start = 1
        scene.frame_end = 250
        scene.frame_set(1)
        self.report({"INFO"}, "Demo scene created - Initialize Solver, then play")
        return {"FINISHED"}


def _execute_role_tag(operator, context, role):
    from .blender import colliders as _colliders
    from .blender import sources as _sources

    count = 0
    skipped = 0
    for obj in context.selected_objects:
        if role == "SOURCE":
            _sources.tag_source(obj)
        elif role == "SINK":
            _sources.tag_sink(obj)
        elif role == "FORCE":
            _sources.tag_force(obj)
        elif role == "FIELD":
            _sources.tag_field(obj)
        elif role == "COLLIDER":
            if obj.type != "MESH":
                skipped += 1
                continue
            _colliders.tag_collider(obj)
        else:
            obj.fluidsim.role = "NONE"
            obj.display_type = "TEXTURED"
            obj.hide_render = False
        count += 1
    if count == 0:
        message = ("Colliders require a selected mesh object"
                   if role == "COLLIDER" else "No supported selected objects")
        operator.report({"WARNING"}, message)
        return {"CANCELLED"}
    detail = f"; skipped {skipped} non-mesh object(s)" if skipped else ""
    operator.report({"INFO"}, f"Tagged {count} object(s) as {role}{detail}")
    return {"FINISHED"}


class FLUIDSIM_OT_tag_object(bpy.types.Operator):
    """Tag the selected objects with a simulation role"""

    bl_idname = "fluidsim.tag_object"
    bl_label = "Tag Object"
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(
        name="Role",
        items=[
            ("SOURCE", "Liquid Emitter", "Emit liquid"),
            ("COLLIDER", "Collider", "Collide with liquid"),
            ("SINK", "Liquid Sink", "Remove liquid"),
            ("FORCE", "Force Region", "Apply an analytic force to liquid"),
            ("FIELD", "Advanced Field Source", "Source non-liquid fields"),
            ("NONE", "Clear", "Remove the FluidSim role"),
        ],
        default="SOURCE",
    )

    @classmethod
    def poll(cls, context):
        if not getattr(context, "selected_objects", None):
            cls.poll_message_set("Select at least one object to tag")
            return False
        return True

    def execute(self, context):
        return _execute_role_tag(self, context, self.role)


class _FLUIDSIM_OT_role_tag_base:
    role = "NONE"

    @classmethod
    def poll(cls, context):
        selected = getattr(context, "selected_objects", None) or ()
        if not selected:
            cls.poll_message_set("Select at least one object to tag")
            return False
        if cls.role == "COLLIDER" and not any(
                getattr(obj, "type", None) == "MESH" for obj in selected):
            cls.poll_message_set("Select at least one mesh object for a collider")
            return False
        return True

    def execute(self, context):
        return _execute_role_tag(self, context, self.role)


class FLUIDSIM_OT_tag_liquid_emitter(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_liquid_emitter"
    bl_label = "Tag Liquid Emitter"
    bl_options = {"REGISTER", "UNDO"}
    role = "SOURCE"


class FLUIDSIM_OT_tag_liquid_sink(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_liquid_sink"
    bl_label = "Tag Liquid Sink"
    bl_options = {"REGISTER", "UNDO"}
    role = "SINK"


class FLUIDSIM_OT_tag_collider(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_collider"
    bl_label = "Tag Collider"
    bl_options = {"REGISTER", "UNDO"}
    role = "COLLIDER"


class FLUIDSIM_OT_tag_force_region(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_force_region"
    bl_label = "Tag Force Region"
    bl_options = {"REGISTER", "UNDO"}
    role = "FORCE"


class FLUIDSIM_OT_tag_field_source(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_field_source"
    bl_label = "Tag Advanced Field Source"
    bl_options = {"REGISTER", "UNDO"}
    role = "FIELD"


class FLUIDSIM_OT_clear_object_role(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.clear_object_role"
    bl_label = "Clear FluidSim Role"
    bl_options = {"REGISTER", "UNDO"}
    role = "NONE"

    @classmethod
    def poll(cls, context):
        selected = getattr(context, "selected_objects", None) or ()
        if not selected:
            cls.poll_message_set("Select at least one tagged object")
            return False
        return any(getattr(getattr(obj, "fluidsim", None), "role", "NONE") != "NONE"
                   for obj in selected)


class FLUIDSIM_OT_inject_particles(bpy.types.Operator):
    """Inject active-object points once as a compatibility utility"""

    bl_idname = "fluidsim.inject_particles"
    bl_label = "Inject Active Object Points Once"

    @classmethod
    def poll(cls, context):
        if not _bridge().session().active:
            cls.poll_message_set("Initialize the solver before injecting points")
            return False
        obj = getattr(context, "active_object", None)
        if obj is None:
            cls.poll_message_set("Select an active mesh, point cloud, or particle-system object")
            return False
        if (getattr(obj, "type", None) not in {"MESH", "POINTCLOUD"}
                and not getattr(obj, "particle_systems", None)):
            cls.poll_message_set(
                "Active object must provide mesh vertices, point-cloud points, or a particle system")
            return False
        return True

    def execute(self, context):
        bridge = _bridge()
        sess = bridge.session()
        if not sess.active:
            self.report({"ERROR"}, "Solver not initialized")
            return {"CANCELLED"}
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}
        from .blender import sources as _sources

        n = _sources.inject_object_particles(context.scene, obj, sess)
        from .blender import viewport

        viewport.update_outputs(context.scene, sess)
        self.report({"INFO"}, f"Injected {n:,} particles from {obj.name}")
        return {"FINISHED"}


class FLUIDSIM_OT_run_validation(bpy.types.Operator):
    """Run the 13-test GPU validation suite (may take ~30s)"""

    bl_idname = "fluidsim.run_validation"
    bl_label = "Run Validation Suite"

    def execute(self, context):
        bridge = _bridge()
        sess = bridge.session()
        try:
            from .sim import validation

            results = validation.run_all(device=_configured_device())
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Validation failed to run: {exc}")
            return {"CANCELLED"}
        sess.validation_results = [r.as_dict() for r in results]
        passed = sum(1 for r in results if r.passed)
        msg = f"Validation: {passed}/{len(results)} passed"
        if passed == len(results):
            self.report({"INFO"}, msg)
        else:
            failed = [r.name for r in results if not r.passed]
            self.report({"WARNING"}, f"{msg} - failed: {', '.join(failed)}")
        context.scene.fluidsim.status = msg
        return {"FINISHED"}


class FLUIDSIM_OT_run_benchmark(bpy.types.Operator):
    """Run the deterministic benchmark scenario and store results"""

    bl_idname = "fluidsim.run_benchmark"
    bl_label = "Run Benchmark"

    save_baseline: BoolProperty(
        name="Save As Baseline", default=False,
        description="Store this run as the comparison baseline")

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _bridge()
        sess = bridge.session()
        try:
            from .diagnostics import benchmark as bench

            result = bench.run_benchmark(frames=fs.benchmark_frames)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Benchmark failed: {exc}")
            return {"CANCELLED"}
        data = result.as_dict()
        sess.benchmark_result = data
        baseline_path = bpy.path.abspath(fs.benchmark_baseline)
        if self.save_baseline:
            bench.save_benchmark(baseline_path, data)
            self.report({"INFO"}, f"Benchmark baseline saved: {baseline_path}")
        else:
            try:
                baseline = bench.load_benchmark(baseline_path)
                sess.benchmark_compare = bench.compare_benchmark(data, baseline)
                self.report(
                    {"INFO"},
                    f"Benchmark {sess.benchmark_compare['result']}: "
                    f"{data['timeStepAverage'] * 1000:.1f} ms/frame avg, "
                    f"{data['voxelsPerSecondAverage'] / 1e6:.1f}M voxels/s")
            except FileNotFoundError:
                sess.benchmark_compare = None
                self.report(
                    {"INFO"},
                    f"Benchmark complete (no baseline): "
                    f"{data['timeStepAverage'] * 1000:.1f} ms/frame avg, "
                    f"{data['voxelsPerSecondAverage'] / 1e6:.1f}M voxels/s")
        fs.status = f"Benchmark {data['result']}"
        return {"FINISHED"}


class FLUIDSIM_OT_write_report(bpy.types.Operator):
    """Write profiling / validation / benchmark data to a JSON report"""

    bl_idname = "fluidsim.write_report"
    bl_label = "Write Diagnostics Report"

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _bridge()
        sess = bridge.session()
        from .diagnostics import report as report_mod

        rep = report_mod.build_report(
            profiler=sess.profiler,
            frame_stats=sess.frame_stats,
            benchmark=sess.benchmark_result,
            validation=sess.validation_results,
            extra={"benchmark_comparison": sess.benchmark_compare}
            if sess.benchmark_compare else None,
        )
        path = bpy.path.abspath(fs.report_path)
        report_mod.write_report(path, rep)
        self.report({"INFO"}, f"Report written: {path}")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_add(bpy.types.Operator):
    """Add a custom Warp snippet slot"""

    bl_idname = "fluidsim.snippet_add"
    bl_label = "Add Snippet"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        item = fs.snippets.add()
        item.kernel_name = f"custom_{len(fs.snippets)}"
        text = bpy.data.texts.new(f"fluidsim_{item.kernel_name}.py")
        text.write(
            "# Warp snippet body. Available: solver fields bound as arguments,\n"
            "# index variables, world_position, voxel_size, time context.\n"
            "# Example (RunOver=VOXEL_LINEAR, argument: field temperature):\n"
            "# temperature[flat_voxel_index] = temperature[flat_voxel_index] * 0.99\n"
            "pass\n"
        )
        item.text_block = text
        fs.active_snippet = len(fs.snippets) - 1
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_remove(bpy.types.Operator):
    """Remove the active custom snippet"""

    bl_idname = "fluidsim.snippet_remove"
    bl_label = "Remove Snippet"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        if 0 <= fs.active_snippet < len(fs.snippets):
            fs.snippets.remove(fs.active_snippet)
            fs.active_snippet = min(fs.active_snippet, len(fs.snippets) - 1)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_argument_add(bpy.types.Operator):
    """Add an argument to the active snippet"""

    bl_idname = "fluidsim.snippet_argument_add"
    bl_label = "Add Argument"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        if 0 <= fs.active_snippet < len(fs.snippets):
            fs.snippets[fs.active_snippet].arguments.add()
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_argument_remove(bpy.types.Operator):
    """Remove the last argument from the active snippet"""

    bl_idname = "fluidsim.snippet_argument_remove"
    bl_label = "Remove Argument"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        if 0 <= fs.active_snippet < len(fs.snippets):
            snip = fs.snippets[fs.active_snippet]
            if len(snip.arguments):
                snip.arguments.remove(len(snip.arguments) - 1)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_apply(bpy.types.Operator):
    """Compile custom snippets and attach them to the running solver"""

    bl_idname = "fluidsim.snippet_apply"
    bl_label = "Apply Snippets"

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _bridge()
        sess = bridge.session()
        if not sess.active:
            self.report({"ERROR"}, "Solver not initialized")
            return {"CANCELLED"}
        for pg in fs.snippets:
            pg.last_error = ""
        # Static validation gate: only valid snippets may reach the scene.
        invalid = 0
        for pg in fs.snippets:
            if not pg.enabled:
                continue
            bridge.validate_snippet_pg(pg)
            if not pg.is_valid:
                invalid += 1
        errors = sess.solver.set_custom_snippets(bridge.build_snippets(scene))
        for name, err in errors:
            for pg in fs.snippets:
                if pg.kernel_name == name:
                    pg.last_error = str(err)[:256]
        if errors or invalid:
            self.report({"WARNING"},
                        f"{len(errors) + invalid} snippet(s) rejected "
                        f"({invalid} invalid, {len(errors)} compile errors); "
                        f"valid snippets are running")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{len(fs.snippets)} snippet(s) compiled")
        return {"FINISHED"}


def _active_snippet(context):
    fs = context.scene.fluidsim
    if 0 <= fs.active_snippet < len(fs.snippets):
        return fs.snippets[fs.active_snippet]
    return None


def _snippet_code(pg) -> str:
    return pg.text_block.as_string() if pg.text_block else ""


def _set_snippet_code(pg, code: str):
    if pg.text_block is None:
        pg.text_block = bpy.data.texts.new(f"fluidsim_{pg.kernel_name}.py")
    pg.text_block.clear()
    pg.text_block.write(code if code.endswith("\n") or not code else code + "\n")


def _append_snippet_code(pg, addition: str):
    code = _snippet_code(pg)
    if code and not code.endswith("\n"):
        code += "\n"
    _set_snippet_code(pg, code + addition.rstrip("\n") + "\n")


class FLUIDSIM_OT_snippet_select(bpy.types.Operator):
    """Make this snippet the active one (shown in the editor)"""

    bl_idname = "fluidsim.snippet_select"
    bl_label = "Select Snippet"

    index: IntProperty(default=0, min=0)

    def execute(self, context):
        fs = context.scene.fluidsim
        if self.index < len(fs.snippets):
            fs.active_snippet = self.index
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_load_preset(bpy.types.Operator):
    """Load the selected library snippet into a new stack slot"""

    bl_idname = "fluidsim.snippet_load_preset"
    bl_label = "Load Snippet"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .properties import user_snippet_dir
        from .sim import custom

        fs = context.scene.fluidsim
        selection = fs.snippet_library_selection
        if not selection or selection == "NONE":
            self.report({"ERROR"}, "No library snippet selected")
            return {"CANCELLED"}

        if selection.startswith("EX:"):
            preset = custom.preset_by_id(selection[3:])
            if preset is None:
                self.report({"ERROR"}, f"Unknown preset '{selection}'")
                return {"CANCELLED"}
            data = custom.preset_to_dict(
                preset.label, preset.description, preset.order,
                preset.run_over, preset.code, preset.arguments,
                preset.frame_start, preset.frame_end, preset.frame_step)
            base_name = preset.preset_id
        else:  # USR:<filename>
            import os

            path = os.path.join(user_snippet_dir(), selection[4:])
            found = [d for d in
                     custom.load_user_presets(user_snippet_dir())
                     if d.get("_path") == path]
            if not found:
                self.report({"ERROR"}, "Saved snippet not found on disk")
                return {"CANCELLED"}
            data = found[0]
            base_name = os.path.splitext(selection[4:])[0]

        pg = fs.snippets.add()
        pg.kernel_name = f"{base_name}_{len(fs.snippets)}"
        pg.description = data.get("description", "")
        pg.order = data.get("order", "BEFORE_FORCES")
        pg.run_over = data.get("run_over", "PARTICLES")
        pg.frame_start = int(data.get("frame_start", 0))
        pg.frame_end = int(data.get("frame_end", 0))
        pg.frame_step = int(data.get("frame_step", 1))
        for arg in data.get("arguments", ()):  # serialized snippet arguments
            apg = pg.arguments.add()
            if arg.get("mode") == "CONSTANT":
                apg.mode = "CONSTANT"
                apg.constant_name = arg.get("name", "value")
                apg.constant_type = arg.get("type", "FLOAT")
                apg.constant_value = float(arg.get("value", 0.0))
            else:
                apg.mode = "FIELD"
                apg.field = arg.get("field", "temperature")
        text = bpy.data.texts.new(f"fluidsim_{pg.kernel_name}.py")
        text.write(data.get("code", ""))
        pg.text_block = text
        fs.active_snippet = len(fs.snippets) - 1
        bridge = _bridge()
        bridge.validate_snippet_pg(pg)
        self.report({"INFO"}, f"Loaded '{data.get('label', base_name)}' - "
                              f"press Apply Snippets to run it")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_validate(bpy.types.Operator):
    """Statically validate the active snippet (must pass before it can run)"""

    bl_idname = "fluidsim.snippet_validate"
    bl_label = "Validate"

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        bridge = _bridge()
        issues = bridge.validate_snippet_pg(pg)
        errors = [i for i in issues if i.level == "ERROR"]
        warnings = [i for i in issues if i.level == "WARNING"]
        if errors:
            self.report({"WARNING"},
                        f"{len(errors)} error(s): {errors[0]}")
            return {"FINISHED"}
        if warnings:
            self.report({"INFO"}, f"Valid with {len(warnings)} warning(s)")
        else:
            self.report({"INFO"}, "Snippet is valid")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_autofix(bpy.types.Operator):
    """Auto-repair common snippet mistakes, then re-validate"""

    bl_idname = "fluidsim.snippet_autofix"
    bl_label = "Auto Fix"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .sim import custom

        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        code = _snippet_code(pg)
        fixed, fixes = custom.autofix_snippet(code)
        if fixed != code:
            _set_snippet_code(pg, fixed)
        bridge = _bridge()
        bridge.validate_snippet_pg(pg)
        if fixes:
            self.report({"INFO"}, f"Applied {len(fixes)} fix(es): "
                                  f"{'; '.join(fixes[:3])}")
        else:
            self.report({"INFO"}, "Nothing to fix")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_test(bpy.types.Operator):
    """Test the active snippet: static validation plus a Warp compile"""

    bl_idname = "fluidsim.snippet_test"
    bl_label = "Test"

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        bridge = _bridge()
        issues = bridge.validate_snippet_pg(pg)
        errors = [i for i in issues if i.level == "ERROR"]
        if errors:
            self.report({"WARNING"}, f"Static validation failed: {errors[0]}")
            return {"CANCELLED"}
        try:
            from .sim.custom import CompiledSnippet

            compiled = CompiledSnippet(bridge.snippet_from_pg(pg))
        except Exception as exc:  # warp unavailable etc.
            self.report({"WARNING"}, f"Warp compile unavailable: {exc}")
            return {"CANCELLED"}
        if compiled.error:
            pg.last_error = str(compiled.error)[:256]
            pg.is_valid = False
            self.report({"WARNING"}, f"Warp compile failed: {compiled.error}")
            return {"CANCELLED"}
        pg.last_error = ""
        self.report({"INFO"}, "Snippet compiled successfully")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_save_preset(bpy.types.Operator):
    """Save the active snippet to your snippet library for later reuse"""

    bl_idname = "fluidsim.snippet_save_preset"
    bl_label = "Save to Library"

    preset_name: StringProperty(name="Name", default="")

    def invoke(self, context, event):
        pg = _active_snippet(context)
        if pg is not None and not self.preset_name:
            self.preset_name = pg.kernel_name
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from .properties import user_snippet_dir
        from .sim import custom

        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        args = []
        for apg in pg.arguments:
            if apg.mode == "CONSTANT":
                args.append({"mode": "CONSTANT", "name": apg.constant_name,
                             "type": apg.constant_type,
                             "value": apg.constant_value})
            else:
                args.append({"mode": "FIELD", "field": apg.field})
        data = custom.preset_to_dict(
            self.preset_name or pg.kernel_name,
            pg.description or "User snippet",
            pg.order, pg.run_over, _snippet_code(pg), args,
            pg.frame_start, pg.frame_end, pg.frame_step)
        path = custom.save_user_preset(user_snippet_dir(create=True), data)
        self.report({"INFO"}, f"Saved: {path}")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_insert_shortcut(bpy.types.Operator):
    """Insert a code shortcut at the end of the active snippet"""

    bl_idname = "fluidsim.snippet_insert_shortcut"
    bl_label = "Insert Shortcut"
    bl_options = {"REGISTER", "UNDO"}

    shortcut_id: StringProperty()

    def execute(self, context):
        from .sim import custom

        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        shortcut = custom.shortcut_by_id(self.shortcut_id)
        if shortcut is None:
            self.report({"ERROR"}, f"Unknown shortcut '{self.shortcut_id}'")
            return {"CANCELLED"}
        _append_snippet_code(pg, shortcut.code)
        _bridge().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_shortcut_field(bpy.types.Operator):
    """Bind the chosen solver field as an argument and reference it in code"""

    bl_idname = "fluidsim.snippet_shortcut_field"
    bl_label = "Add Field"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        name = fs.shortcut_field
        if not any(a.mode == "FIELD" and a.field == name
                   for a in pg.arguments):
            apg = pg.arguments.add()
            apg.mode = "FIELD"
            apg.field = name
        index = ("particle_index" if pg.run_over == "PARTICLES"
                 else "flat_voxel_index")
        if name.startswith("particle_") and pg.run_over != "PARTICLES":
            self.report({"WARNING"},
                        f"'{name}' is a particle field; set Run Over to "
                        f"Particles to index it correctly")
        _append_snippet_code(pg, f"# field '{name}': read/write via "
                                 f"{name}[{index}]")
        _bridge().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_shortcut_constant(bpy.types.Operator):
    """Add a named constant argument and reference it in code"""

    bl_idname = "fluidsim.snippet_shortcut_constant"
    bl_label = "Add Constant"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        name = fs.shortcut_constant_name.strip()
        if not name.isidentifier():
            self.report({"ERROR"}, f"'{name}' is not a valid constant name")
            return {"CANCELLED"}
        existing = next((a for a in pg.arguments
                         if a.mode == "CONSTANT" and a.constant_name == name),
                        None)
        if existing is not None:
            existing.constant_value = fs.shortcut_constant_value
        else:
            apg = pg.arguments.add()
            apg.mode = "CONSTANT"
            apg.constant_name = name
            apg.constant_type = "FLOAT"
            apg.constant_value = fs.shortcut_constant_value
        _append_snippet_code(
            pg, f"# constant '{name}' = {fs.shortcut_constant_value:g} "
                f"(edit value in the Arguments list)")
        _bridge().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_edit_line(bpy.types.Operator):
    """Edit one line of the active snippet"""

    bl_idname = "fluidsim.snippet_edit_line"
    bl_label = "Edit Line"
    bl_options = {"REGISTER", "UNDO"}

    line_index: IntProperty(default=0, min=0)
    line_text: StringProperty(name="Line", default="")

    def invoke(self, context, event):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        lines = _snippet_code(pg).split("\n")
        if self.line_index < len(lines):
            self.line_text = lines[self.line_index]
        return context.window_manager.invoke_props_dialog(self, width=520)

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        lines = _snippet_code(pg).split("\n")
        while len(lines) <= self.line_index:
            lines.append("")
        lines[self.line_index] = self.line_text
        _set_snippet_code(pg, "\n".join(lines))
        _bridge().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_line_add(bpy.types.Operator):
    """Append a new line to the active snippet"""

    bl_idname = "fluidsim.snippet_line_add"
    bl_label = "Add Line"
    bl_options = {"REGISTER", "UNDO"}

    line_text: StringProperty(name="Line", default="")

    def invoke(self, context, event):
        self.line_text = ""
        return context.window_manager.invoke_props_dialog(self, width=520)

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        _append_snippet_code(pg, self.line_text)
        _bridge().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_line_delete(bpy.types.Operator):
    """Delete one line of the active snippet"""

    bl_idname = "fluidsim.snippet_line_delete"
    bl_label = "Delete Line"
    bl_options = {"REGISTER", "UNDO"}

    line_index: IntProperty(default=0, min=0)

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        lines = _snippet_code(pg).split("\n")
        if self.line_index < len(lines):
            lines.pop(self.line_index)
            _set_snippet_code(pg, "\n".join(lines))
            _bridge().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_export_frame(bpy.types.Operator):
    """Export the current simulation frame (surface + particles) to disk"""

    bl_idname = "fluidsim.export_frame"
    bl_label = "Export Current Frame"

    filepath: StringProperty(subtype="FILE_PATH", default="//fluidsim_frame")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        bridge = _bridge()
        sess = bridge.session()
        if not sess.active:
            self.report({"ERROR"}, "Solver not initialized")
            return {"CANCELLED"}
        from .io import export as export_mod

        base = bpy.path.abspath(self.filepath)
        paths = export_mod.export_frame(sess.solver, base)
        self.report({"INFO"}, f"Exported: {', '.join(paths)}")
        return {"FINISHED"}


class FLUIDSIM_OT_export_cache_sequence(bpy.types.Operator):
    """Bake the selected timeline range to FluidSim cache frames"""

    bl_idname = "fluidsim.export_cache_sequence"
    bl_label = "Export Cache Sequence"

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _bridge()
        from .io import cache

        try:
            start, end = cache.resolve_frame_range(
                scene,
                use_custom=fs.cache_use_custom_range,
                frame_start=fs.cache_frame_start,
                frame_end=fs.cache_frame_end,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            fs.status = f"Cache export failed: {exc}"
            return {"CANCELLED"}

        try:
            base = bridge.session().export_cache_sequence(scene, start, end)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache sequence export failed: {exc}")
            fs.status = f"Cache export failed: {exc}"
            return {"CANCELLED"}

        count = end - start + 1
        fs.status = f"Exported {count} cache frame(s) to {base}"
        self.report({"INFO"}, fs.status)
        return {"FINISHED"}


CLASSES = (
    FLUIDSIM_OT_initialize,
    FLUIDSIM_OT_release,
    FLUIDSIM_OT_clear_error,
    FLUIDSIM_OT_apply_safe_settings,
    FLUIDSIM_OT_reset,
    FLUIDSIM_OT_step,
    FLUIDSIM_OT_toggle_live,
    FLUIDSIM_OT_setup_scene,
    FLUIDSIM_OT_tag_object,
    FLUIDSIM_OT_tag_liquid_emitter,
    FLUIDSIM_OT_tag_liquid_sink,
    FLUIDSIM_OT_tag_collider,
    FLUIDSIM_OT_tag_force_region,
    FLUIDSIM_OT_tag_field_source,
    FLUIDSIM_OT_clear_object_role,
    FLUIDSIM_OT_inject_particles,
    FLUIDSIM_OT_run_validation,
    FLUIDSIM_OT_run_benchmark,
    FLUIDSIM_OT_write_report,
    FLUIDSIM_OT_snippet_add,
    FLUIDSIM_OT_snippet_remove,
    FLUIDSIM_OT_snippet_argument_add,
    FLUIDSIM_OT_snippet_argument_remove,
    FLUIDSIM_OT_snippet_apply,
    FLUIDSIM_OT_snippet_select,
    FLUIDSIM_OT_snippet_load_preset,
    FLUIDSIM_OT_snippet_validate,
    FLUIDSIM_OT_snippet_autofix,
    FLUIDSIM_OT_snippet_test,
    FLUIDSIM_OT_snippet_save_preset,
    FLUIDSIM_OT_snippet_insert_shortcut,
    FLUIDSIM_OT_snippet_shortcut_field,
    FLUIDSIM_OT_snippet_shortcut_constant,
    FLUIDSIM_OT_snippet_edit_line,
    FLUIDSIM_OT_snippet_line_add,
    FLUIDSIM_OT_snippet_line_delete,
    FLUIDSIM_OT_export_frame,
    FLUIDSIM_OT_export_cache_sequence,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
