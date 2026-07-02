"""Operators for the FluidSim Warp add-on."""

from __future__ import annotations

import json

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty


def _bridge():
    from .blender import scene_bridge

    return scene_bridge


class FLUIDSIM_OT_initialize(bpy.types.Operator):
    """Create (or recreate) the GPU solver from current scene settings"""

    bl_idname = "fluidsim.initialize"
    bl_label = "Initialize Solver"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        try:
            sess = _bridge().initialize_session(scene)
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
            bridge.sync_scene_to_solver(scene, sess)
            for _ in range(self.frames):
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

        # Domain box.
        mesh = bpy.data.meshes.new("FluidSim Domain")
        domain = bpy.data.objects.new("FluidSim Domain", mesh)
        scene.collection.objects.link(domain)
        import bmesh

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()
        domain.scale = (4.0, 4.0, 4.0)
        domain.location = (0.0, 0.0, 2.0)
        domain.display_type = "WIRE"
        domain.hide_render = True
        fs.domain_object = domain

        # Particle source box.
        src_mesh = bpy.data.meshes.new("FluidSim Source")
        src = bpy.data.objects.new("FluidSim Source", src_mesh)
        scene.collection.objects.link(src)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(src_mesh)
        bm.free()
        src.location = (-1.2, 0.0, 2.6)
        src.scale = (1.2, 3.2, 2.2)
        from .blender import sources as _sources

        _sources.tag_source(src, shape="BOX", mode="PARTICLES")

        # Sphere collider.
        col_mesh = bpy.data.meshes.new("FluidSim Collider")
        col = bpy.data.objects.new("FluidSim Collider", col_mesh)
        scene.collection.objects.link(col)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=0.5)
        bm.to_mesh(col_mesh)
        bm.free()
        col.location = (0.8, 0.0, 0.5)
        from .blender import colliders as _colliders

        _colliders.tag_collider(col)

        scene.frame_start = 1
        scene.frame_end = 250
        scene.frame_set(1)
        self.report({"INFO"}, "Demo scene created - Initialize Solver, then play")
        return {"FINISHED"}


class FLUIDSIM_OT_tag_object(bpy.types.Operator):
    """Tag the selected objects with a simulation role"""

    bl_idname = "fluidsim.tag_object"
    bl_label = "Tag Object"
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(
        name="Role",
        items=[("SOURCE", "Source", ""), ("COLLIDER", "Collider", ""),
               ("SINK", "Sink", ""), ("NONE", "Clear", "")],
        default="SOURCE",
    )

    def execute(self, context):
        from .blender import colliders as _colliders
        from .blender import sources as _sources

        count = 0
        for obj in context.selected_objects:
            if self.role == "SOURCE":
                _sources.tag_source(obj)
            elif self.role == "SINK":
                _sources.tag_sink(obj)
            elif self.role == "COLLIDER":
                if obj.type != "MESH":
                    continue
                _colliders.tag_collider(obj)
            else:
                obj.fluidsim.role = "NONE"
                obj.display_type = "TEXTURED"
                obj.hide_render = False
            count += 1
        self.report({"INFO"}, f"Tagged {count} object(s) as {self.role}")
        return {"FINISHED"}


class FLUIDSIM_OT_inject_particles(bpy.types.Operator):
    """Inject the active object's points/vertices as liquid particles now"""

    bl_idname = "fluidsim.inject_particles"
    bl_label = "Inject Object Particles"

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

            results = validation.run_all(device="cuda:0")
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

    def execute(self, context):
        fs = context.scene.fluidsim
        item = fs.snippets.add()
        item.kernel_name = f"custom_{len(fs.snippets)}"
        text = bpy.data.texts.new(f"fluidsim_{item.kernel_name}.py")
        text.write(
            "# Warp snippet body. Available: solver fields bound as arguments,\n"
            "# index variables, world_position, voxel_size, time context.\n"
            "# Example (RunOver=VOXEL_LINEAR, argument: field temperature RW):\n"
            "temperature[flat_voxel_index] = temperature[flat_voxel_index] * 0.99\n"
        )
        item.text_block = text
        fs.active_snippet = len(fs.snippets) - 1
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_remove(bpy.types.Operator):
    """Remove the active custom snippet"""

    bl_idname = "fluidsim.snippet_remove"
    bl_label = "Remove Snippet"

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

    def execute(self, context):
        fs = context.scene.fluidsim
        if 0 <= fs.active_snippet < len(fs.snippets):
            fs.snippets[fs.active_snippet].arguments.add()
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_argument_remove(bpy.types.Operator):
    """Remove the last argument from the active snippet"""

    bl_idname = "fluidsim.snippet_argument_remove"
    bl_label = "Remove Argument"

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
        errors = sess.solver.set_custom_snippets(bridge.build_snippets(scene))
        for name, err in errors:
            for pg in fs.snippets:
                if pg.kernel_name == name:
                    pg.last_error = str(err)[:256]
        if errors:
            self.report({"WARNING"},
                        f"{len(errors)} snippet(s) failed to compile")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{len(fs.snippets)} snippet(s) compiled")
        return {"FINISHED"}


class FLUIDSIM_OT_export_frame(bpy.types.Operator):
    """Export the current simulation frame (surface + particles) to disk"""

    bl_idname = "fluidsim.export_frame"
    bl_label = "Export Current Frame"

    filepath: StringProperty(subtype="FILE_PATH", default="//fluidsim_frame")

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


CLASSES = (
    FLUIDSIM_OT_initialize,
    FLUIDSIM_OT_release,
    FLUIDSIM_OT_reset,
    FLUIDSIM_OT_step,
    FLUIDSIM_OT_toggle_live,
    FLUIDSIM_OT_setup_scene,
    FLUIDSIM_OT_tag_object,
    FLUIDSIM_OT_inject_particles,
    FLUIDSIM_OT_run_validation,
    FLUIDSIM_OT_run_benchmark,
    FLUIDSIM_OT_write_report,
    FLUIDSIM_OT_snippet_add,
    FLUIDSIM_OT_snippet_remove,
    FLUIDSIM_OT_snippet_argument_add,
    FLUIDSIM_OT_snippet_argument_remove,
    FLUIDSIM_OT_snippet_apply,
    FLUIDSIM_OT_export_frame,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
