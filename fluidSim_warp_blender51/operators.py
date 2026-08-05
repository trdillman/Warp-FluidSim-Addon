"""Operators for the FluidSim Warp add-on."""

from __future__ import annotations

import json
import math
import re

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty


def _bridge():
    from .blender import scene_bridge

    return scene_bridge


def _configured_device():
    from .preferences import configured_device

    return configured_device()


def _scene_identity_token(scene):
    pointer = getattr(scene, "as_pointer", None)
    if callable(pointer):
        try:
            value = int(pointer())
            if value:
                return str(value)
        except (TypeError, ValueError, ReferenceError):
            pass
    return str(id(scene))


def _owned_capacity_plan(scene):
    bridge = _bridge()
    getter = getattr(bridge, "capacity_plan_for_scene", None)
    if callable(getter):
        return getter(scene)
    sess = bridge.session()
    getter = getattr(sess, "capacity_plan_for_scene", None)
    return getter(scene) if callable(getter) else getattr(
        sess, "capacity_plan", None)


def _owned_capacity_signature(scene, plan):
    bridge = _bridge()
    getter = getattr(bridge, "capacity_plan_signature_for_scene", None)
    if callable(getter):
        return str(getter(scene))
    return str(getattr(plan, "scene_signature", ""))


class _FLUIDSIM_OT_session_base:
    """Grey out operators that cannot do anything without a live solver."""

    @classmethod
    def poll(cls, context):
        if not _bridge().session().active:
            cls.poll_message_set("Initialize the solver first")
            return False
        return True


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
            plan = _owned_capacity_plan(scene)
            integrity_blocked = bool(plan and any(
                getattr(blocker, "code", "") != "MEMORY_ESTIMATE"
                for blocker in getattr(plan, "blockers", ())))
            if integrity_blocked:
                self.report(
                    {"WARNING"},
                    "Capacity adjustment required; review the FluidSim panel",
                )
                scene.fluidsim.status = (
                    "Capacity warning — approve a safe option below to run")
            else:
                self.report({"ERROR"}, f"Solver initialization failed: {exc}")
                scene.fluidsim.status = f"Init failed: {exc}"
            return {"CANCELLED"}
        dev = sess.solver.device
        cap = sess.solver.grid.capacity
        self.report({"INFO"},
                    f"Solver ready on {dev} ({cap} leaves, "
                    f"{sess.solver.particles.capacity:,} particle capacity)")
        return {"FINISHED"}


class FLUIDSIM_OT_release(_FLUIDSIM_OT_session_base, bpy.types.Operator):
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


def _capacity_suggestion(plan, key):
    return next((item for item in getattr(plan, "suggestions", ())
                 if getattr(item, "key", "") == key), None)


def _format_capacity_value(value):
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(f"{float(item):.4g}" for item in value) + ")"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _apply_domain_bounds(scene, minimum, maximum):
    domain = getattr(scene.fluidsim, "domain_object", None)
    if domain is None:
        raise RuntimeError("The scene has no domain object to resize")
    if getattr(domain, "parent", None) is not None:
        raise RuntimeError(
            "Automatic domain-bound adjustment requires an unparented domain")
    rotations = tuple(float(value) for value in domain.rotation_euler)
    delta_rotations = tuple(
        float(value) for value in getattr(
            domain, "delta_rotation_euler", (0.0, 0.0, 0.0)))
    angle_distance = lambda value: abs((value + math.pi) % (2.0 * math.pi) - math.pi)
    if any(angle_distance(value) > 1.0e-6
           for value in rotations + delta_rotations):
        raise RuntimeError(
            "Automatic domain-bound adjustment requires an axis-aligned domain")
    minimum = tuple(float(value) for value in minimum)
    maximum = tuple(float(value) for value in maximum)
    dimensions = tuple(maximum[i] - minimum[i] for i in range(3))
    if any(value <= 0.0 for value in dimensions):
        raise RuntimeError("Suggested domain bounds are not valid")
    domain.location = tuple((minimum[i] + maximum[i]) * 0.5 for i in range(3))
    domain.dimensions = dimensions
    if getattr(bpy.context, "view_layer", None) is not None:
        bpy.context.view_layer.update()


class FLUIDSIM_OT_apply_capacity_suggestion(bpy.types.Operator):
    """Confirm, apply, and retry one planner-verified capacity option."""

    bl_idname = "fluidsim.apply_capacity_suggestion"
    bl_label = "Approve Capacity Changes & Run"
    bl_options = {"REGISTER"}

    suggestion_key: StringProperty(default="", options={"SKIP_SAVE"})
    expected_scene_signature: StringProperty(default="", options={"SKIP_SAVE"})
    expected_scene_identity: StringProperty(
        default="", options={"HIDDEN", "SKIP_SAVE"})
    changes_json: StringProperty(default="{}", options={"HIDDEN", "SKIP_SAVE"})
    suggestion_label: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    estimated_bytes: StringProperty(default="0", options={"HIDDEN", "SKIP_SAVE"})
    budget_bytes: StringProperty(default="0", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        plan = _owned_capacity_plan(getattr(context, "scene", None))
        if not getattr(plan, "suggestions", ()):
            cls.poll_message_set("Initialize first to calculate safe capacity options")
            return False
        return True

    def invoke(self, context, _event):
        plan = _owned_capacity_plan(context.scene)
        suggestion = _capacity_suggestion(plan, self.suggestion_key)
        if suggestion is None:
            self.report({"WARNING"}, "Capacity option is stale; initialize again")
            return {"CANCELLED"}
        self.expected_scene_signature = _owned_capacity_signature(
            context.scene, plan)
        self.expected_scene_identity = _scene_identity_token(context.scene)
        self.changes_json = json.dumps(dict(suggestion.changes))
        self.suggestion_label = str(suggestion.label)
        self.estimated_bytes = str(int(suggestion.estimated_bytes))
        self.budget_bytes = str(int(suggestion.budget_bytes))
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Approve these exact scene changes?", icon="QUESTION")
        layout.label(text=self.suggestion_label)
        changes = json.loads(self.changes_json or "{}")
        settings = _bridge().build_settings(context.scene)
        for name, value in changes.items():
            old_value = getattr(settings, name, getattr(context.scene.fluidsim, name, None))
            layout.label(text=(
                f"{name}: {_format_capacity_value(old_value)} → "
                f"{_format_capacity_value(value)}"))
        layout.separator()
        layout.label(text=(
            f"Estimated peak {int(self.estimated_bytes) / 1024**3:.2f} GiB / "
            f"safe live budget {int(self.budget_bytes) / 1024**3:.2f} GiB"))
        layout.label(
            text="Estimate only; approval cannot guarantee CUDA allocation.",
            icon="INFO",
        )
        layout.label(text="Settings are unchanged until you confirm.")

    def execute(self, context):
        bridge = _bridge()
        scene = context.scene
        fs = scene.fluidsim
        if (self.expected_scene_identity
                and self.expected_scene_identity != _scene_identity_token(scene)):
            self.report(
                {"WARNING"},
                "Capacity option belongs to another scene; review this scene",
            )
            return {"CANCELLED"}
        if _owned_capacity_plan(scene) is None:
            self.report(
                {"WARNING"},
                "Capacity option is no longer owned by this scene",
            )
            return {"CANCELLED"}
        try:
            fresh_plan = bridge.refresh_capacity_plan(scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Capacity recheck failed: {exc}")
            return {"CANCELLED"}
        if (self.expected_scene_signature
                and _owned_capacity_signature(scene, fresh_plan)
                != self.expected_scene_signature):
            self.report(
                {"WARNING"},
                "Scene changed after the warning; initialize again for new options",
            )
            fs.status = "Capacity options changed — review the refreshed warning"
            return {"CANCELLED"}
        suggestion = _capacity_suggestion(fresh_plan, self.suggestion_key)
        if suggestion is None:
            self.report(
                {"WARNING"},
                "This option is no longer inside the live safety budget",
            )
            fs.status = "Capacity option is no longer safe — review alternatives"
            return {"CANCELLED"}

        changes = dict(suggestion.changes)
        domain_min = changes.pop("domain_min", None)
        domain_max = changes.pop("domain_max", None)
        old_values = {name: getattr(fs, name) for name in changes
                      if hasattr(fs, name)}
        domain = getattr(fs, "domain_object", None)
        old_domain = None if domain is None else (
            tuple(domain.location), tuple(domain.dimensions))
        changed = []
        try:
            for name, value in changes.items():
                if not hasattr(fs, name):
                    raise RuntimeError(
                        f"Capacity option cannot apply setting {name!r}")
                rna = getattr(fs, "bl_rna", None)
                properties = getattr(rna, "properties", {})
                rna_property = properties.get(name)
                if rna_property is not None and rna_property.type == "INT":
                    value = int(value)
                setattr(fs, name, value)
                changed.append(name)
            if domain_min is not None or domain_max is not None:
                settings = bridge.build_settings(scene)
                _apply_domain_bounds(
                    scene,
                    domain_min if domain_min is not None else settings.domain_min,
                    domain_max if domain_max is not None else settings.domain_max,
                )
                changed.append("domain bounds")
        except Exception as exc:  # noqa: BLE001
            for name, value in old_values.items():
                setattr(fs, name, value)
            if domain is not None and old_domain is not None:
                domain.location, domain.dimensions = old_domain
            self.report({"ERROR"}, f"Capacity option was not applied: {exc}")
            fs.status = f"Capacity option was not applied: {exc}"
            return {"CANCELLED"}

        bridge.clear_error(scene)
        fs.status = "Approved capacity settings: " + ", ".join(changed)
        try:
            sess = bridge.initialize_session(scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Approved retry stopped safely: {exc}")
            fs.status = f"Approved retry stopped safely: {exc}"
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Solver ready with approved option: {suggestion.label}",
        )
        fs.status = (
            f"Solver ready on {sess.solver.device}; approved {suggestion.label}")
        return {"FINISHED"}


class FLUIDSIM_OT_apply_safe_settings(bpy.types.Operator):
    """Compatibility entry point that routes into reviewed capacity options."""

    bl_idname = "fluidsim.apply_safe_settings"
    bl_label = "Review Capacity Options"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(getattr(getattr(context, "scene", None), "fluidsim", None))

    def execute(self, context):
        bridge = _bridge()
        scene = context.scene
        fs = scene.fluidsim
        try:
            plan = bridge.refresh_capacity_plan(scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Capacity inspection failed: {exc}")
            fs.status = f"Capacity inspection failed: {exc}"
            return {"CANCELLED"}
        if not tuple(getattr(plan, "suggestions", ())):
            self.report({"ERROR"}, "No verified capacity-only option is available")
            fs.status = "No verified capacity-only option is available"
            return {"CANCELLED"}
        # Never mutate semantic quality/capacity from a one-click retry. The
        # existing plan panel and apply_capacity_suggestion dialog own the one
        # exact-values confirmation and stale-signature revalidation flow.
        fs.status = "Capacity options ready — review and confirm one option"
        self.report({"WARNING"}, fs.status)
        return {"CANCELLED"}


class FLUIDSIM_OT_reset(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Reset the simulation to an empty state"""

    bl_idname = "fluidsim.reset"
    bl_label = "Reset Simulation"

    def execute(self, context):
        sess = _bridge().session()
        owns_scene = getattr(sess, "owns_scene", lambda _scene: sess.active)
        if not owns_scene(context.scene):
            self.report({"WARNING"}, "Solver not initialized for this scene")
            return {"CANCELLED"}
        _bridge().reset_simulation(context.scene)
        return {"FINISHED"}


class FLUIDSIM_OT_apply_pending_reinitialize(bpy.types.Operator):
    """Apply pending allocation/domain settings by rebuilding the solver"""

    bl_idname = "fluidsim.apply_pending_reinitialize"
    bl_label = "Apply Pending & Reinitialize"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        if getattr(scene, "fluidsim", None) is None:
            return False
        bridge = _bridge()
        if bridge.pending_presentation(scene)["action"] == "NONE":
            return False
        sess = bridge.session()
        return not sess.active or sess.owns_scene(scene)

    def execute(self, context):
        try:
            changed = _bridge().apply_pending_reinitialize(context.scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Pending reinitialize failed: {exc}")
            return {"CANCELLED"}
        if not changed:
            self.report({"WARNING"}, "No pending settings to apply")
            return {"CANCELLED"}
        self.report({"INFO"}, "Pending settings applied")
        return {"FINISHED"}


class FLUIDSIM_OT_apply_pending_reset(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Apply reset-tier pending settings without reallocating the solver"""

    bl_idname = "fluidsim.apply_pending_reset"
    bl_label = "Apply Pending & Reset"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        bridge = _bridge()
        sess = bridge.session()
        if not sess.active:
            cls.poll_message_set("Initialize the solver first")
            return False
        scene = getattr(context, "scene", None)
        if getattr(scene, "fluidsim", None) is None:
            cls.poll_message_set("No FluidSim scene is active")
            return False
        if bridge.pending_presentation(scene)["action"] != "RESET_PENDING":
            cls.poll_message_set("No reset-tier settings are pending")
            return False
        if not sess.owns_scene(scene):
            cls.poll_message_set("The active solver belongs to another scene")
            return False
        return True

    def execute(self, context):
        try:
            changed = _bridge().apply_pending_reset(context.scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Pending reset failed: {exc}")
            return {"CANCELLED"}
        if not changed:
            self.report({"WARNING"}, "No pending reset settings to apply")
            return {"CANCELLED"}
        self.report({"INFO"}, "Pending reset settings applied")
        return {"FINISHED"}


class FLUIDSIM_OT_revert_pending(bpy.types.Operator):
    """Restore the last successfully applied restart/reinitialize values"""

    bl_idname = "fluidsim.revert_pending"
    bl_label = "Revert Pending Values"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return (getattr(scene, "fluidsim", None) is not None
                and _bridge().pending_presentation(scene)["count"] > 0)

    def execute(self, context):
        try:
            reverted = _bridge().revert_pending_values(context.scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Pending revert failed: {exc}")
            return {"CANCELLED"}
        if not reverted:
            self.report({"WARNING"}, "No applied pending snapshot is available")
            return {"CANCELLED"}
        self.report({"INFO"}, "Pending values reverted")
        return {"FINISHED"}


class FLUIDSIM_OT_step(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Advance the simulation by a number of frames"""

    bl_idname = "fluidsim.step"
    bl_label = "Step Frame"

    frames: IntProperty(
        name="Frames", default=1, min=1, max=250,
        description="Set how many simulation frames this action advances; higher values perform more solver work before returning")

    def execute(self, context):
        scene = context.scene
        bridge = _bridge()
        sess = bridge.session()
        owns_scene = getattr(sess, "owns_scene", lambda _scene: sess.active)
        if not owns_scene(scene):
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
            sess = bridge.session()
            owns_scene = getattr(sess, "owns_scene", lambda _scene: sess.active)
            if not owns_scene(scene):
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
        objects = tuple(getattr(scene, "objects", ()))
        existing_domain = getattr(fs, "domain_object", None)
        existing_source = any(
            getattr(getattr(obj, "fluidsim", None), "role", "NONE") == "SOURCE"
            for obj in objects)
        existing_collider = any(
            getattr(getattr(obj, "fluidsim", None), "role", "NONE") == "COLLIDER"
            for obj in objects)
        if existing_domain is not None and existing_source and existing_collider:
            self.report(
                {"INFO"},
                "Quick Setup already exists; existing domain and inputs were preserved")
            return {"FINISHED"}

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
        description="Choose the FluidSim role assigned to every selected object; changing it changes whether the objects emit, collide, remove, force, source fields, or remain untagged",
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
        if not any(getattr(getattr(obj, "fluidsim", None), "role", "NONE") != "NONE"
                   for obj in selected):
            cls.poll_message_set("Select at least one object with a FluidSim role")
            return False
        return True


class FLUIDSIM_OT_inject_particles(bpy.types.Operator):
    """Append mesh vertices, point-cloud points, or the first particle system once; this compatibility utility is not a volume fill"""

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
        from .blender import viewport

        scene = context.scene
        try:
            n = _sources.inject_object_particles(scene, obj, sess)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Particle injection failed: {exc}")
            bridge.set_error(scene, f"Particle injection failed: {exc}")
            scene.fluidsim.status = f"Injection failed: {exc}"
            return {"CANCELLED"}
        result = getattr(sess.solver.sourcing, "last_injection_result", None)
        if result is not None and not result.admitted:
            self.report({"WARNING"}, (
                "Particle injection needs "
                f"{result.required:,} slots but only {result.available:,} "
                "are available. Increase Max Particles or enable automatic "
                "particle-capacity growth, then retry."
            ))
            return {"CANCELLED"}
        viewport.update_outputs(scene, sess)
        self.report({"INFO"}, f"Injected {n:,} particles from {obj.name}")
        return {"FINISHED"}


class FLUIDSIM_OT_run_validation(_FLUIDSIM_OT_session_base, bpy.types.Operator):
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


class FLUIDSIM_OT_run_benchmark(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Run the deterministic benchmark scenario and store results"""

    bl_idname = "fluidsim.run_benchmark"
    bl_label = "Run Benchmark"

    save_baseline: BoolProperty(
        name="Save As Baseline", default=False,
        description="Enable to store this benchmark run as the comparison baseline; disable to compare without replacing the existing baseline")

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


class FLUIDSIM_OT_write_report(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Write profiling / validation / benchmark data to a JSON report"""

    bl_idname = "fluidsim.write_report"
    bl_label = "Write Diagnostics Report"

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _bridge()
        sess = bridge.session()
        from .diagnostics import report as report_mod

        path = bpy.path.abspath(fs.report_path)
        try:
            rep = report_mod.build_report(
                profiler=sess.profiler,
                frame_stats=sess.frame_stats,
                benchmark=sess.benchmark_result,
                validation=sess.validation_results,
                extra={"benchmark_comparison": sess.benchmark_compare}
                if sess.benchmark_compare else None,
            )
            report_mod.write_report(path, rep)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Report write failed: {exc}")
            bridge.set_error(scene, f"Report write failed: {exc}")
            fs.status = f"Report write failed: {exc}"
            return {"CANCELLED"}
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


class FLUIDSIM_OT_snippet_apply(_FLUIDSIM_OT_session_base, bpy.types.Operator):
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

    preset_name: StringProperty(
        name="Name", default="",
        description="Set the name used for the saved snippet preset; changing it changes the library label and output filename")

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
    line_text: StringProperty(
        name="Line", default="",
        description="Edit the selected snippet source line; changing it replaces that line and triggers snippet validation")

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

    line_text: StringProperty(
        name="Line", default="",
        description="Set the source code appended as a new final line; changing it changes the code that is validated and compiled")

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


class FLUIDSIM_OT_export_frame(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Export the current simulation frame (surface + particles) to disk"""

    bl_idname = "fluidsim.export_frame"
    bl_label = "Export Current Frame"

    filepath: StringProperty(
        subtype="FILE_PATH", default="//fluidsim_frame",
        description="Set the output path prefix for the current surface and particle export; changing it writes the exported frame to a different location")

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

        scene = context.scene
        base = bpy.path.abspath(self.filepath)
        try:
            paths = export_mod.export_frame(sess.solver, base)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Frame export failed: {exc}")
            bridge.set_error(scene, f"Frame export failed: {exc}")
            scene.fluidsim.status = f"Frame export failed: {exc}"
            return {"CANCELLED"}
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
    FLUIDSIM_OT_apply_capacity_suggestion,
    FLUIDSIM_OT_apply_safe_settings,
    FLUIDSIM_OT_reset,
    FLUIDSIM_OT_apply_pending_reinitialize,
    FLUIDSIM_OT_apply_pending_reset,
    FLUIDSIM_OT_revert_pending,
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
