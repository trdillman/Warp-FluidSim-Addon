"""Operators for the FluidSim Warp add-on."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty


def _bridge():
    from .blender import scene_bridge

    return scene_bridge


def _pending():
    from .blender import pending_settings

    return pending_settings


def _snippets():
    from .blender import snippets

    return snippets


def _solver_settings():
    from .blender import solver_settings

    return solver_settings


def _configured_device():
    from .preferences import configured_device

    return configured_device()


def _runtime_poll(cls) -> bool:
    """Block GPU actions until external Warp and the selected CUDA GPU are ready."""
    from .runtime.warp_runtime import runtime_status_for_device

    ready, message = runtime_status_for_device(_configured_device())
    if not ready:
        cls.poll_message_set(message)
    return ready


def _mutation_poll(cls, context) -> bool:
    """Block solver mutations while Production Bake owns the scene."""
    from .blender import production_bake

    scene = getattr(context, "scene", None)
    if scene is not None and production_bake.bake_is_active(scene):
        cls.poll_message_set("Production Bake owns this scene")
        return False
    return _runtime_poll(cls)


def _resolved_cache_path(scene) -> Path:
    """Resolve the exact authored cache target used by mutation dialogs."""
    raw = str(bpy.path.abspath(scene.fluidsim.cache_path) or "").strip()
    if not raw:
        raise ValueError("The cache path is empty")
    return Path(raw).expanduser().resolve()


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
        if not _mutation_poll(cls, context):
            return False
        if not _bridge().session().active:
            cls.poll_message_set("Initialize the solver first")
            return False
        return True


class FLUIDSIM_OT_initialize(bpy.types.Operator):
    """Create (or recreate) the GPU solver from current scene settings"""

    bl_idname = "fluidsim.initialize"
    bl_label = "Initialize Solver"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _mutation_poll(cls, context)

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


class FLUIDSIM_OT_release(bpy.types.Operator):
    """Release the GPU solver and free device memory."""

    bl_idname = "fluidsim.release"
    bl_label = "Release Solver"

    @classmethod
    def poll(cls, context):
        from .blender import production_bake

        if production_bake.bake_is_active(getattr(context, "scene", None)):
            cls.poll_message_set("Production Bake owns this scene")
            return False
        sess = _bridge().session()
        if not sess.active:
            cls.poll_message_set("No solver session is active")
            return False
        owns_scene = getattr(sess, "owns_scene", None)
        if callable(owns_scene) and not owns_scene(getattr(context, "scene", None)):
            cls.poll_message_set("The active solver belongs to another scene")
            return False
        return True

    def execute(self, context):
        if not _bridge().release_session(context.scene):
            self.report({"WARNING"}, "Another scene owns the live solver")
            return {"CANCELLED"}
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
    def angle_distance(value):
        return abs((value + math.pi) % (2.0 * math.pi) - math.pi)
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
        settings = _solver_settings().build_settings(context.scene)
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
                # RNA clamps out-of-range assignments to the declared
                # [min, max] silently, so a write can succeed while storing a
                # different number than the one the dialog costed. Reporting
                # that as approved would break the budget promise this whole
                # flow exists to keep -- read back and fail into the rollback.
                stored = getattr(fs, name)
                if isinstance(value, float):
                    matches = math.isclose(stored, value, rel_tol=1e-6)
                else:
                    matches = stored == value
                if not matches:
                    raise RuntimeError(
                        f"setting {name!r} was stored as {stored!r} instead of "
                        f"{value!r}; it is outside the range this build allows")
                changed.append(name)
            if domain_min is not None or domain_max is not None:
                settings = _solver_settings().build_settings(scene)
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
        eligible, message = _pending().reinitialize_eligibility(scene)
        if not eligible and message:
            cls.poll_message_set(message)
        return eligible

    def execute(self, context):
        try:
            changed = _pending().apply_pending_reinitialize(context.scene)
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
        scene = getattr(context, "scene", None)
        eligible, message = _pending().reset_eligibility(scene)
        if not eligible and message:
            cls.poll_message_set(message)
        return eligible

    def execute(self, context):
        try:
            changed = _pending().apply_pending_reset(context.scene)
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
        eligible, message = _pending().revert_eligibility(scene)
        if not eligible and message:
            cls.poll_message_set(message)
        return eligible

    def execute(self, context):
        try:
            reverted = _pending().revert_pending_values(context.scene)
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

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

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

    @classmethod
    def poll(cls, context):
        fs = getattr(getattr(context, "scene", None), "fluidsim", None)
        if fs is None:
            cls.poll_message_set("The active scene has no FluidSim settings")
            return False
        from .blender import production_bake

        if production_bake.bake_is_active(context.scene):
            cls.poll_message_set("Production Bake owns this scene")
            return False
        if getattr(fs, "live", False):
            return True
        return _mutation_poll(cls, context)

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


class FLUIDSIM_OT_cache_playback_start(bpy.types.Operator):
    """Arm manifest/cache playback without exposing transport flags in the UI."""

    bl_idname = "fluidsim.cache_playback_start"
    bl_label = "Start Cache Playback"

    @classmethod
    def poll(cls, context):
        fs = getattr(getattr(context, "scene", None), "fluidsim", None)
        if fs is None:
            cls.poll_message_set("The active scene has no FluidSim settings")
            return False
        return not bool(getattr(fs, "cache_playback_enabled", False))

    def execute(self, context):
        fs = context.scene.fluidsim
        fs.cache_playback_enabled = True
        fs.live = True
        fs.status = "Cache Playback ready - press Timeline Play"
        return {"FINISHED"}


class FLUIDSIM_OT_cache_playback_stop(bpy.types.Operator):
    """Stop cache playback and return the scene to an idle timeline state."""

    bl_idname = "fluidsim.cache_playback_stop"
    bl_label = "Stop Cache Playback"

    @classmethod
    def poll(cls, context):
        fs = getattr(getattr(context, "scene", None), "fluidsim", None)
        if fs is None or not bool(getattr(fs, "cache_playback_enabled", False)):
            cls.poll_message_set("Cache playback is not active")
            return False
        return True

    def execute(self, context):
        fs = context.scene.fluidsim
        fs.live = False
        fs.cache_playback_enabled = False
        fs.status = "Cache Playback stopped"
        return {"FINISHED"}


class FLUIDSIM_OT_setup_scene(bpy.types.Operator):
    """Turn selected mesh geometry into a runnable FluidSim setup"""

    bl_idname = "fluidsim.setup_scene"
    bl_label = "Run Guided Setup"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Mode",
        items=(
            ("QUICK_LIQUID", "Quick Liquid", "Selected mesh becomes an initial-fill emitter"),
            ("QUICK_CONTAINER", "Quick Container", "Selected mesh becomes a collider with a height fill"),
            ("QUICK_VISCOUS", "Quick Viscous Liquid", "Selected mesh becomes a thick-liquid emitter"),
            ("ADD_COLLIDER", "Add Collider", "Tag selected meshes as colliders"),
            ("ADD_INFLOW", "Add Inflow", "Tag selected meshes as continuous inflows"),
            ("ADD_OUTFLOW", "Add Outflow", "Tag selected meshes as liquid sinks"),
        ),
        default="QUICK_LIQUID",
        options={"SKIP_SAVE"},
    )
    preset_id: StringProperty(default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        selected = getattr(context, "selected_objects", ())
        if not any(getattr(obj, "type", None) == "MESH" for obj in selected):
            cls.poll_message_set("Select at least one mesh object")
            return False
        return True

    def execute(self, context):
        from .blender.setup_wizard import setup_selected

        try:
            result = setup_selected(context, self.mode, self.preset_id)
        except Exception as exc:  # noqa: BLE001
            context.scene.fluidsim.status = f"Guided setup failed: {exc}"
            self.report({"ERROR"}, context.scene.fluidsim.status)
            return {"CANCELLED"}
        context.scene.fluidsim.status = result.message
        for warning in result.warnings[:3]:
            self.report({"WARNING"}, warning)
        self.report({"INFO"}, result.message)
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
        from .blender import output_publication
        from .blender import sources as _sources

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
        output_publication.update_outputs(scene, sess)
        self.report({"INFO"}, f"Injected {n:,} particles from {obj.name}")
        return {"FINISHED"}


class FLUIDSIM_OT_run_validation(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Run the GPU validation suite (may take ~30s)"""

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
            try:
                bench.save_benchmark(baseline_path, data)
            except (OSError, TypeError, ValueError) as exc:
                message = f"Benchmark baseline invalid: {baseline_path}: {exc}"
                fs.status = message
                self.report({"ERROR"}, message)
                return {"CANCELLED"}
            self.report({"INFO"}, f"Benchmark baseline saved: {baseline_path}")
        else:
            try:
                baseline = bench.load_benchmark(baseline_path)
                comparison = bench.compare_benchmark(data, baseline)
                if comparison.get("error"):
                    raise ValueError(comparison["error"])
                sess.benchmark_compare = comparison
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
            except (OSError, TypeError, ValueError) as exc:
                sess.benchmark_compare = None
                message = f"Benchmark baseline invalid: {baseline_path}: {exc}"
                fs.status = message
                self.report({"ERROR"}, message)
                return {"CANCELLED"}
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

        path = bpy.path.abspath(fs.report_path)
        try:
            from .io import cache as cache_mod

            settings_signature = ""
            if sess.active and getattr(sess, "solver", None) is not None:
                settings_signature = cache_mod.settings_record(
                    sess.solver.settings)["signature"]
            rep = report_mod.build_report(
                profiler=sess.profiler,
                frame_stats=sess.frame_stats,
                benchmark=sess.benchmark_result,
                validation=sess.validation_results,
                extra={"benchmark_comparison": sess.benchmark_compare}
                if sess.benchmark_compare else None,
                package_name=__package__,
                blender_version=".".join(str(value) for value in bpy.app.version[:3]),
                selected_device=_configured_device(),
                cache_path=bpy.path.abspath(fs.cache_path),
                scene_signature=str(getattr(sess, "scene_key", "")),
                settings_signature=settings_signature,
                effect_state=__import__(
                    __package__ + ".blender.effects", fromlist=["effect_report_state"]
                ).effect_report_state(scene),
            )
            report_mod.write_report(path, rep)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Report write failed: {exc}")
            bridge.set_error(scene, f"Report write failed: {exc}")
            fs.status = f"Report write failed: {exc}"
            return {"CANCELLED"}
        self.report({"INFO"}, f"Report written: {path}")
        return {"FINISHED"}


# deprecated compatibility adapter entry points forward to Custom Effects v2.
class FLUIDSIM_OT_snippet_add(bpy.types.Operator):
    """deprecated compatibility adapter that adds a Custom Effects v2 starter."""

    bl_idname = "fluidsim.snippet_add"
    bl_label = "Add Custom Effect"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .blender.effects import add_builtin_effect

        add_builtin_effect(context.scene, "fluidsim.vortex_force")
        self.report({"INFO"}, "Added Vortex Force in Custom Effects")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_remove(bpy.types.Operator):
    """deprecated compatibility adapter that removes the active Custom Effect."""

    bl_idname = "fluidsim.snippet_remove"
    bl_label = "Remove Custom Effect"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        result = bpy.ops.fluidsim.effect_remove()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


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
    """deprecated compatibility adapter that applies the Custom Effects v2 stack."""

    bl_idname = "fluidsim.snippet_apply"
    bl_label = "Compile & Apply Custom Effects"

    def execute(self, context):
        result = bpy.ops.fluidsim.effect_apply_stack()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


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
        _snippets().validate_snippet_pg(pg)
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
        issues = _snippets().validate_snippet_pg(pg)
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
        _snippets().validate_snippet_pg(pg)
        if fixes:
            self.report({"INFO"}, f"Applied {len(fixes)} fix(es): "
                                  f"{'; '.join(fixes[:3])}")
        else:
            self.report({"INFO"}, "Nothing to fix")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_test(bpy.types.Operator):
    """deprecated compatibility adapter that tests the active Custom Effects v2 draft."""

    bl_idname = "fluidsim.snippet_test"
    bl_label = "Test Custom Effect Draft"

    def execute(self, context):
        result = bpy.ops.fluidsim.effect_test_draft()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


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
        _snippets().validate_snippet_pg(pg)
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
        _snippets().validate_snippet_pg(pg)
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
        _snippets().validate_snippet_pg(pg)
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
        _snippets().validate_snippet_pg(pg)
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
        _snippets().validate_snippet_pg(pg)
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
            _snippets().validate_snippet_pg(pg)
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

    @classmethod
    def poll(cls, context):
        return _runtime_poll(cls)

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
            from .blender import cache_bake

            base = cache_bake.bake(scene, bridge.session(), start, end)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache sequence export failed: {exc}")
            fs.status = f"Cache export failed: {exc}"
            return {"CANCELLED"}

        count = end - start + 1
        fs.status = f"Exported {count} cache frame(s) to {base}"
        self.report({"INFO"}, fs.status)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Product presets and guided workflow
# ---------------------------------------------------------------------------


class FLUIDSIM_OT_apply_product_preset(bpy.types.Operator):
    """Apply a validated built-in or user product preset"""

    bl_idname = "fluidsim.apply_product_preset"
    bl_label = "Apply FluidSim Preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .presets import library

        identifier = str(context.scene.fluidsim.product_preset or "")
        try:
            preset = library.load_preset(identifier)
            result = library.apply_preset(context.scene, preset)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Preset could not be applied: {exc}")
            return {"CANCELLED"}
        context.scene.fluidsim.status = result.message
        self.report({"INFO"}, result.message)
        return {"FINISHED"}


class FLUIDSIM_OT_save_product_preset(bpy.types.Operator):
    """Capture allow-listed scene settings as an atomic user preset"""

    bl_idname = "fluidsim.save_product_preset"
    bl_label = "Save User Preset"
    bl_options = {"REGISTER"}

    preset_id: StringProperty(
        name="Identifier",
        default="my_water",
        description="Lowercase stable identifier using letters, numbers, and underscores",
    )
    preset_name: StringProperty(name="Name", default="My Water")
    description: StringProperty(name="Description", default="Custom FluidSim settings")
    expected_scale: StringProperty(name="Expected Scale", default="1 Blender unit = 1 metre")
    performance_cost: EnumProperty(
        name="Performance Cost",
        items=(("Low", "Low", ""), ("Medium", "Medium", ""),
               ("High", "High", ""), ("Very High", "Very High", "")),
        default="Medium",
    )
    overwrite: BoolProperty(name="Overwrite Existing", default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=520)

    def execute(self, context):
        from .presets import library

        try:
            preset = library.capture_scene_preset(
                context.scene,
                identifier=self.preset_id,
                name=self.preset_name,
                description=self.description,
                expected_scale=self.expected_scale,
                performance_cost=self.performance_cost,
            )
            path = library.save_user_preset(
                preset, overwrite=bool(self.overwrite)
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"User preset could not be saved: {exc}")
            return {"CANCELLED"}
        context.scene.fluidsim.status = f"Saved user preset: {path.name}"
        self.report({"INFO"}, context.scene.fluidsim.status)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Responsive production bake and cache operations
# ---------------------------------------------------------------------------

_ACTIVE_BAKE = None


def _update_bake_rna(scene, controller) -> None:
    fs = scene.fluidsim
    fs.bake_status = controller.status.value
    fs.bake_progress = float(controller.progress)
    fs.bake_message = str(controller.message)
    fs.status = str(controller.message)


class _FLUIDSIM_OT_production_bake_modal:
    resume_existing = False
    restart_existing = False
    _timer = None
    _controller = None

    @classmethod
    def poll(cls, context):
        global _ACTIVE_BAKE
        if _ACTIVE_BAKE is not None:
            cls.poll_message_set("A production bake is already active")
            return False
        return _runtime_poll(cls)

    def execute(self, context):
        global _ACTIVE_BAKE
        from .blender import production_bake
        from .blender.production_bake import BakeController, ProductionBakeBackend

        scene = context.scene
        fs = scene.fluidsim
        backend = ProductionBakeBackend(
            checkpoint_interval=int(fs.checkpoint_interval),
            compressed=bool(fs.cache_compress),
        )
        controller = BakeController(backend=backend)
        try:
            if self.resume_existing:
                controller.resume(scene, end=int(fs.bake_end))
            else:
                controller.start(
                    scene,
                    int(fs.bake_start),
                    int(fs.bake_end),
                    restart=bool(self.restart_existing),
                )
            controller.reserve_cache_path(backend.cache_path(scene))
            production_bake.register_active_bake(scene, controller)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Production bake could not start: {exc}")
            return {"CANCELLED"}
        self._controller = controller
        _ACTIVE_BAKE = controller
        _update_bake_rna(scene, controller)
        window_manager = context.window_manager
        self._timer = window_manager.event_timer_add(0.05, window=context.window)
        window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        global _ACTIVE_BAKE
        from .blender import production_bake
        from .blender.production_bake import BakeStatus

        controller = self._controller
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _ACTIVE_BAKE = None
        production_bake.clear_active_bake(context.scene, controller)
        _update_bake_rna(context.scene, controller)
        if controller.status is BakeStatus.COMPLETE:
            self.report({"INFO"}, controller.message)
            return {"FINISHED"}
        if controller.status is BakeStatus.CANCELLED:
            self.report({"WARNING"}, controller.message)
            return {"CANCELLED"}
        self.report({"ERROR"}, controller.error or controller.message)
        return {"CANCELLED"}

    def modal(self, context, event):
        from .blender.production_bake import BakeStatus

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._controller.request_cancel()
            _update_bake_rna(context.scene, self._controller)
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        self._controller.tick(context.scene)
        _update_bake_rna(context.scene, self._controller)
        if self._controller.status in {
            BakeStatus.COMPLETE, BakeStatus.CANCELLED, BakeStatus.FAILED
        }:
            return self._finish(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if self._controller is not None:
            self._controller.request_cancel()
            self._controller.tick(context.scene)
        return self._finish(context)


class FLUIDSIM_OT_bake_start(_FLUIDSIM_OT_production_bake_modal, bpy.types.Operator):
    """Start a fresh one-frame-transaction production bake"""

    bl_idname = "fluidsim.bake_start"
    bl_label = "Start Production Bake"
    bl_options = {"REGISTER"}
    resume_existing = False
    restart_existing: BoolProperty(
        name="Restart Existing Cache",
        default=False,
        options={"SKIP_SAVE"},
        description="Delete only FluidSim cache-v2 artifacts before baking",
    )
    expected_cache_path: StringProperty(
        default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        if not self.restart_existing:
            return self.execute(context)
        try:
            self.expected_cache_path = str(_resolved_cache_path(context.scene))
        except Exception as exc:
            self.report({"ERROR"}, f"Cache path could not be resolved: {exc}")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text="Replace Existing Production Cache?", icon="ERROR")
        layout.label(text=self.expected_cache_path, icon="FILE_FOLDER")
        layout.label(text="FluidSim will replace its manifest, frames, checkpoints, and logs.")
        layout.label(text="Unrelated files in the directory are not removed.", icon="INFO")

    def execute(self, context):
        if self.restart_existing and self.expected_cache_path:
            current = str(_resolved_cache_path(context.scene))
            if current != self.expected_cache_path:
                self.report({"ERROR"}, "Cache path changed after confirmation; review Restart again")
                return {"CANCELLED"}
        return super().execute(context)


class FLUIDSIM_OT_bake_resume(_FLUIDSIM_OT_production_bake_modal, bpy.types.Operator):
    """Resume from the last verified committed checkpoint"""

    bl_idname = "fluidsim.bake_resume"
    bl_label = "Resume Production Bake"
    bl_options = {"REGISTER"}
    resume_existing = True


class FLUIDSIM_OT_bake_cancel(bpy.types.Operator):
    """Request cancellation at the next manifest-committed frame boundary"""

    bl_idname = "fluidsim.bake_cancel"
    bl_label = "Cancel Production Bake"

    @classmethod
    def poll(cls, _context):
        if _ACTIVE_BAKE is None:
            cls.poll_message_set("No production bake is active")
            return False
        return True

    def execute(self, context):
        _ACTIVE_BAKE.request_cancel()
        _update_bake_rna(context.scene, _ACTIVE_BAKE)
        return {"FINISHED"}


class FLUIDSIM_OT_cache_integrity_scan(bpy.types.Operator):
    """Verify the manifest-authoritative cache and report orphan files"""

    bl_idname = "fluidsim.cache_integrity_scan"
    bl_label = "Scan Cache Integrity"

    def execute(self, context):
        from .blender.cache_manager import inspect_cache

        path = Path(bpy.path.abspath(context.scene.fluidsim.cache_path))
        inspection = inspect_cache(path)
        context.scene.fluidsim.status = inspection.message
        severity = {"INFO"} if inspection.healthy else {"ERROR"}
        detail = inspection.message
        if inspection.orphan_files:
            detail += f"; {len(inspection.orphan_files)} orphan(s)"
        self.report(severity, detail)
        return {"FINISHED"}


class FLUIDSIM_OT_cache_remove_orphans(bpy.types.Operator):
    """Remove only files not referenced by the committed cache-v2 manifest"""

    bl_idname = "fluidsim.cache_remove_orphans"
    bl_label = "Remove Cache Orphans"
    bl_options = {"REGISTER"}

    expected_cache_path: StringProperty(
        default="", options={"HIDDEN", "SKIP_SAVE"})
    expected_orphans_json: StringProperty(
        default="[]", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        from .blender.cache_manager import inspect_cache

        try:
            path = _resolved_cache_path(context.scene)
            inspection = inspect_cache(path)
        except Exception as exc:
            self.report({"ERROR"}, f"Cache could not be inspected: {exc}")
            return {"CANCELLED"}
        if not inspection.orphan_files:
            self.report({"INFO"}, "No orphan cache files were found")
            return {"CANCELLED"}
        self.expected_cache_path = str(path)
        self.expected_orphans_json = json.dumps(list(inspection.orphan_files))
        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, _context):
        layout = self.layout
        orphans = tuple(json.loads(self.expected_orphans_json or "[]"))
        layout.label(text=f"Remove {len(orphans)} uncommitted/orphan file(s)?", icon="ERROR")
        layout.label(text=self.expected_cache_path, icon="FILE_FOLDER")
        box = layout.box()
        for relative in orphans:
            box.label(text=str(relative), icon="FILE")

    def execute(self, context):
        from .blender.cache_manager import inspect_cache, remove_orphans

        path = _resolved_cache_path(context.scene)
        if self.expected_cache_path and str(path) != self.expected_cache_path:
            self.report({"ERROR"}, "Cache path changed after confirmation; inspect again")
            return {"CANCELLED"}
        expected = tuple(json.loads(self.expected_orphans_json or "[]"))
        current = tuple(inspect_cache(path).orphan_files)
        if expected and current != expected:
            self.report({"ERROR"}, "Cache orphan inventory changed after confirmation; inspect again")
            return {"CANCELLED"}
        removed = remove_orphans(path)
        self.report({"INFO"}, f"Removed {len(removed)} orphan cache file(s)")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_free(bpy.types.Operator):
    """Transactionally free the configured FluidSim cache"""

    bl_idname = "fluidsim.cache_free"
    bl_label = "Free Cache"
    bl_options = {"REGISTER"}

    expected_cache_path: StringProperty(
        default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        try:
            self.expected_cache_path = str(_resolved_cache_path(context.scene))
        except Exception as exc:
            self.report({"ERROR"}, f"Cache path could not be resolved: {exc}")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text="Free Entire Cache?", icon="ERROR")
        layout.label(text=self.expected_cache_path, icon="FILE_FOLDER")
        layout.label(text="The directory is transactionally renamed before deletion.")

    def execute(self, context):
        from .blender.cache_manager import free_cache

        path = _resolved_cache_path(context.scene)
        if self.expected_cache_path and str(path) != self.expected_cache_path:
            self.report({"ERROR"}, "Cache path changed after confirmation; review Free Cache again")
            return {"CANCELLED"}
        try:
            retained = free_cache(path)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache could not be freed: {exc}")
            return {"CANCELLED"}
        if retained is not None:
            self.report({"ERROR"}, f"Cache deletion failed; data remains at {retained}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Cache freed")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_move(bpy.types.Operator):
    """Copy, verify, publish, and then remove a cache"""

    bl_idname = "fluidsim.cache_move"
    bl_label = "Move Cache"
    bl_options = {"REGISTER"}

    destination: StringProperty(name="Destination", subtype="DIR_PATH")
    expected_source: StringProperty(
        default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        source = _resolved_cache_path(context.scene)
        self.expected_source = str(source)
        if not self.destination:
            self.destination = str(source.with_name(source.name + "_moved"))
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text=f"Source: {self.expected_source}", icon="FILE_FOLDER")
        layout.prop(self, "destination")
        layout.label(text="The copy is verified before the source is removed.", icon="INFO")

    def execute(self, context):
        from .blender.cache_manager import move_cache

        fs = context.scene.fluidsim
        source = _resolved_cache_path(context.scene)
        if self.expected_source and str(source) != self.expected_source:
            self.report({"ERROR"}, "Cache source changed after confirmation; review Move again")
            return {"CANCELLED"}
        destination = Path(bpy.path.abspath(self.destination)).expanduser().resolve()
        try:
            moved = move_cache(source, destination)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache move failed: {exc}")
            return {"CANCELLED"}
        fs.cache_path = str(moved)
        self.report({"INFO"}, f"Cache moved to {moved}")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_rename(bpy.types.Operator):
    """Rename only the cache directory leaf"""

    bl_idname = "fluidsim.cache_rename"
    bl_label = "Rename Cache"
    bl_options = {"REGISTER"}

    name: StringProperty(name="New Directory Name", default="fluidsim_cache")
    expected_source: StringProperty(
        default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        source = _resolved_cache_path(context.scene)
        self.expected_source = str(source)
        self.name = source.name
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text=f"Current: {self.expected_source}", icon="FILE_FOLDER")
        layout.prop(self, "name")

    def execute(self, context):
        from .blender.cache_manager import rename_cache

        fs = context.scene.fluidsim
        source = _resolved_cache_path(context.scene)
        if self.expected_source and str(source) != self.expected_source:
            self.report({"ERROR"}, "Cache source changed after confirmation; review Rename again")
            return {"CANCELLED"}
        try:
            renamed = rename_cache(source, self.name)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache rename failed: {exc}")
            return {"CANCELLED"}
        fs.cache_path = str(renamed)
        self.report({"INFO"}, f"Cache renamed to {renamed.name}")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_open_folder(bpy.types.Operator):
    """Open the configured cache folder without shell interpolation"""

    bl_idname = "fluidsim.cache_open_folder"
    bl_label = "Open Cache Folder"

    def execute(self, context):
        from .blender.cache_manager import open_cache_folder

        try:
            open_cache_folder(Path(bpy.path.abspath(context.scene.fluidsim.cache_path)))
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not open cache folder: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class FLUIDSIM_OT_use_active_as_domain(bpy.types.Operator):
    """Assign the active mesh as the scene's FluidSim domain bounds object."""

    bl_idname = "fluidsim.use_active_as_domain"
    bl_label = "Use Active Object as FluidSim Domain"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        scene = getattr(context, "scene", None)
        if scene is None or getattr(scene, "fluidsim", None) is None:
            return False
        if obj is None or getattr(obj, "type", None) != "MESH":
            cls.poll_message_set(
                "Select a mesh object to use as the FluidSim domain")
            return False
        if obj is getattr(scene.fluidsim, "domain_object", None):
            cls.poll_message_set(
                "The active object is already the FluidSim domain")
            return False
        from .blender.ui_common import active_object_role

        if active_object_role(context) != "NONE":
            cls.poll_message_set(
                "Clear the object's FluidSim input role before assigning it as the domain")
            return False
        return True

    def execute(self, context):
        obj = getattr(context, "active_object", None)
        scene = getattr(context, "scene", None)
        if (
            scene is None
            or getattr(scene, "fluidsim", None) is None
            or obj is None
            or getattr(obj, "type", None) != "MESH"
        ):
            self.report(
                {"ERROR"},
                "Select a mesh object to use as the FluidSim domain",
            )
            return {"CANCELLED"}
        from .blender.ui_common import active_object_role

        if active_object_role(context) != "NONE":
            self.report(
                {"ERROR"},
                "Clear the object's FluidSim input role before using it as the domain",
            )
            return {"CANCELLED"}
        scene.fluidsim.domain_object = obj
        scene.fluidsim.status = f"FluidSim domain assigned to {obj.name}"
        self.report({"INFO"}, scene.fluidsim.status)
        return {"FINISHED"}


class FLUIDSIM_OT_clear_domain_assignment(bpy.types.Operator):
    """Clear the selected scene domain reference without deleting the object."""

    bl_idname = "fluidsim.clear_domain_assignment"
    bl_label = "Remove FluidSim Domain Assignment"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        obj = getattr(context, "active_object", None)
        return bool(
            scene is not None
            and getattr(scene, "fluidsim", None) is not None
            and obj is not None
            and obj is getattr(scene.fluidsim, "domain_object", None)
        )

    def execute(self, context):
        scene = getattr(context, "scene", None)
        fs = getattr(scene, "fluidsim", None) if scene is not None else None
        obj = getattr(context, "active_object", None)
        if fs is None or obj is None or obj is not getattr(fs, "domain_object", None):
            self.report(
                {"ERROR"}, "The active object is no longer the FluidSim domain")
            return {"CANCELLED"}
        fs.domain_object = None
        fs.status = f"Removed FluidSim domain assignment from {obj.name}"
        self.report({"INFO"}, fs.status)
        return {"FINISHED"}


class FLUIDSIM_OT_remove_selected_output(bpy.types.Operator):
    """Remove one selected add-on-owned output after explicit confirmation."""

    bl_idname = "fluidsim.remove_selected_output"
    bl_label = "Remove Selected FluidSim Output"
    bl_options = {"REGISTER", "UNDO"}

    expected_object: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    expected_kind: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        from .blender import output_publication

        obj = getattr(context, "active_object", None)
        if obj is None:
            return False
        try:
            kind = obj.get(output_publication.OUTPUT_TAG)
            owned = bool(obj.get("fluidsim_owned"))
        except AttributeError:
            return False
        return bool(
            owned
            and kind
            in {
                output_publication.SURFACE_NAME,
                output_publication.PARTICLES_NAME,
                *output_publication.WHITEWATER_OUTPUT_NAMES,
            }
        )

    def invoke(self, context, event):
        from .blender import output_publication

        obj = getattr(context, "active_object", None)
        if obj is None:
            return {"CANCELLED"}
        self.expected_object = str(obj.name)
        self.expected_kind = str(obj.get(output_publication.OUTPUT_TAG, ""))
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from .blender import output_publication

        obj = getattr(context, "active_object", None)
        if obj is None:
            self.report({"ERROR"}, "No active FluidSim output")
            return {"CANCELLED"}
        current_kind = str(obj.get(output_publication.OUTPUT_TAG, ""))
        if (
            obj.name != self.expected_object
            or current_kind != self.expected_kind
            or not bool(obj.get("fluidsim_owned"))
        ):
            self.report(
                {"ERROR"},
                "Selected output changed after confirmation; review removal again",
            )
            return {"CANCELLED"}
        try:
            removed = output_publication.remove_output(
                context.scene,
                current_kind,
                expected_object=obj,
            )
        except Exception as exc:  # noqa: BLE001
            self.report(
                {"ERROR"}, f"Could not remove FluidSim output: {exc}")
            return {"CANCELLED"}
        if not removed:
            self.report(
                {"ERROR"}, "The selected FluidSim output no longer exists")
            return {"CANCELLED"}
        context.scene.fluidsim.status = (
            f"Removed {self.expected_object}; Create / Repair Outputs restores it"
        )
        self.report({"INFO"}, context.scene.fluidsim.status)
        return {"FINISHED"}


class FLUIDSIM_OT_install_render_nodes(bpy.types.Operator):
    """Install or explicitly refresh bundled FluidSim render node groups"""

    bl_idname = "fluidsim.install_render_nodes"
    bl_label = "Install FluidSim Render Nodes"
    bl_options = {"REGISTER", "UNDO"}

    force: BoolProperty(
        name="Update FluidSim-owned Modifiers",
        default=False,
        options={"SKIP_SAVE"},
        description=(
            "Repoint only FluidSim-owned modifiers to the bundled revision; "
            "user node groups and user modifiers are never overwritten"
        ),
    )

    def execute(self, context):
        from .blender import geometry_nodes, output_publication

        try:
            outputs = output_publication.ensure_output_shells(context.scene)
            result = geometry_nodes.install_render_nodes(
                context.scene,
                force=bool(self.force),
                output_objects={
                    str(obj.get(output_publication.OUTPUT_TAG, obj.name)): obj
                    for obj in outputs
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Render-node installation failed: {exc}")
            return {"CANCELLED"}
        for warning in result.warnings[:3]:
            self.report({"WARNING"}, warning)
        context.scene.fluidsim.status = result.message
        self.report({"INFO"}, result.message)
        return {"FINISHED"}


class FLUIDSIM_OT_copy_bake_command(bpy.types.Operator):
    """Copy a shell-safe headless bake command for the current scene"""

    bl_idname = "fluidsim.copy_bake_command"
    bl_label = "Copy Headless Bake Command"

    def execute(self, context):
        import os
        import shlex
        import subprocess

        blend_path = str(bpy.data.filepath or "")
        if not blend_path:
            self.report({"ERROR"}, "Save the .blend file before copying a bake command")
            return {"CANCELLED"}
        runner = Path(__file__).resolve().parent / "release" / "run_headless_bake.py"
        fs = context.scene.fluidsim
        args = [
            bpy.app.binary_path,
            "--background",
            blend_path,
            "--python",
            str(runner),
            "--",
            "--scene",
            context.scene.name,
            "--start",
            str(fs.bake_start),
            "--end",
            str(fs.bake_end),
            "--cache",
            str(Path(bpy.path.abspath(fs.cache_path)).resolve()),
            "--device",
            _configured_device(),
            "--resume",
            "auto",
            "--checkpoint-interval",
            str(fs.checkpoint_interval),
        ]
        command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
        context.window_manager.clipboard = command
        self.report({"INFO"}, "Headless production-bake command copied")
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
    FLUIDSIM_OT_cache_playback_start,
    FLUIDSIM_OT_cache_playback_stop,
    FLUIDSIM_OT_setup_scene,
    FLUIDSIM_OT_apply_product_preset,
    FLUIDSIM_OT_save_product_preset,
    FLUIDSIM_OT_bake_start,
    FLUIDSIM_OT_bake_resume,
    FLUIDSIM_OT_bake_cancel,
    FLUIDSIM_OT_cache_integrity_scan,
    FLUIDSIM_OT_cache_remove_orphans,
    FLUIDSIM_OT_cache_free,
    FLUIDSIM_OT_cache_move,
    FLUIDSIM_OT_cache_rename,
    FLUIDSIM_OT_cache_open_folder,
    FLUIDSIM_OT_copy_bake_command,
    FLUIDSIM_OT_use_active_as_domain,
    FLUIDSIM_OT_clear_domain_assignment,
    FLUIDSIM_OT_remove_selected_output,
    FLUIDSIM_OT_install_render_nodes,
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
