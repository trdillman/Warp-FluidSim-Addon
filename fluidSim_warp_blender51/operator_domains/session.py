"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


def _bridge():
    return _runtime._bridge()


def _configured_device():
    return _runtime._configured_device()


def _mutation_poll(cls, context):
    return _runtime._mutation_poll(cls, context)


def _owned_capacity_plan(scene):
    return _runtime._owned_capacity_plan(scene)


def _owned_capacity_signature(scene, plan):
    return _runtime._owned_capacity_signature(scene, plan)


def _pending():
    return _runtime._pending()


def _resolved_cache_path(scene):
    return _runtime._resolved_cache_path(scene)


def _runtime_poll(cls):
    return _runtime._runtime_poll(cls)


def _scene_identity_token(scene):
    return _runtime._scene_identity_token(scene)


def _solver_settings():
    return _runtime._solver_settings()


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
            integrity_blocked = bool(
                plan
                and any(
                    getattr(blocker, "code", "") != "MEMORY_ESTIMATE"
                    for blocker in getattr(plan, "blockers", ())
                )
            )
            if integrity_blocked:
                self.report(
                    {"WARNING"},
                    "Capacity adjustment required; review the FluidSim panel",
                )
                scene.fluidsim.status = "Capacity warning â€” approve a safe option below to run"
            else:
                self.report({"ERROR"}, f"Solver initialization failed: {exc}")
                scene.fluidsim.status = f"Init failed: {exc}"
            return {"CANCELLED"}
        dev = sess.solver.device
        cap = sess.solver.grid.capacity
        self.report(
            {"INFO"},
            f"Solver ready on {dev} ({cap} leaves, "
            f"{sess.solver.particles.capacity:,} particle capacity)",
        )
        return {"FINISHED"}


class FLUIDSIM_OT_release(bpy.types.Operator):
    """Release the GPU solver and free device memory."""

    bl_idname = "fluidsim.release"
    bl_label = "Release Solver"

    @classmethod
    def poll(cls, context):
        from ..blender import production_bake

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
    return next(
        (item for item in getattr(plan, "suggestions", ()) if getattr(item, "key", "") == key), None
    )


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
        raise RuntimeError("Automatic domain-bound adjustment requires an unparented domain")
    rotations = tuple(float(value) for value in domain.rotation_euler)
    delta_rotations = tuple(
        float(value) for value in getattr(domain, "delta_rotation_euler", (0.0, 0.0, 0.0))
    )

    def angle_distance(value):
        return abs((value + math.pi) % (2.0 * math.pi) - math.pi)

    if any(angle_distance(value) > 1.0e-6 for value in rotations + delta_rotations):
        raise RuntimeError("Automatic domain-bound adjustment requires an axis-aligned domain")
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
    expected_scene_identity: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
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
        self.expected_scene_signature = _owned_capacity_signature(context.scene, plan)
        self.expected_scene_identity = _scene_identity_token(context.scene)
        self.changes_json = json.dumps(dict(suggestion.changes))
        self.suggestion_label = str(suggestion.label)
        self.estimated_bytes = str(int(suggestion.estimated_bytes))
        self.budget_bytes = str(int(suggestion.budget_bytes))
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, context):
        from ..blender.scene_bridge import active_domain_settings

        layout = self.layout
        layout.label(text="Approve these exact scene changes?", icon="QUESTION")
        layout.label(text=self.suggestion_label)
        changes = json.loads(self.changes_json or "{}")
        domain_settings = active_domain_settings(context.scene)
        for name, value in changes.items():
            old_value = getattr(domain_settings, name, None)
            layout.label(
                text=(
                    f"{name}: {_format_capacity_value(old_value)} â†’ "
                    f"{_format_capacity_value(value)}"
                )
            )
        layout.separator()
        layout.label(
            text=(
                f"Estimated peak {int(self.estimated_bytes) / 1024**3:.2f} GiB / "
                f"safe live budget {int(self.budget_bytes) / 1024**3:.2f} GiB"
            )
        )
        layout.label(
            text="Estimate only; approval cannot guarantee CUDA allocation.",
            icon="INFO",
        )
        layout.label(text="Settings are unchanged until you confirm.")

    def execute(self, context):
        from ..blender.scene_bridge import active_domain_settings

        bridge = _bridge()
        scene = context.scene
        scene_fs = scene.fluidsim
        if self.expected_scene_identity and self.expected_scene_identity != _scene_identity_token(
            scene
        ):
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
        if (
            self.expected_scene_signature
            and _owned_capacity_signature(scene, fresh_plan) != self.expected_scene_signature
        ):
            self.report(
                {"WARNING"},
                "Scene changed after the warning; initialize again for new options",
            )
            scene_fs.status = "Capacity options changed â€” review the refreshed warning"
            return {"CANCELLED"}
        suggestion = _capacity_suggestion(fresh_plan, self.suggestion_key)
        if suggestion is None:
            self.report(
                {"WARNING"},
                "This option is no longer inside the live safety budget",
            )
            scene_fs.status = "Capacity option is no longer safe â€” review alternatives"
            return {"CANCELLED"}

        from ..blender.scene_bridge import active_domain_settings

        domain_settings = active_domain_settings(scene)
        if domain_settings is None:
            message = "Capacity option cannot apply without a designated domain"
            self.report({"ERROR"}, message)
            scene_fs.status = message
            return {"CANCELLED"}
        changes = dict(suggestion.changes)
        domain_min = changes.pop("domain_min", None)
        domain_max = changes.pop("domain_max", None)
        old_values = {
            name: getattr(domain_settings, name)
            for name in changes
            if hasattr(domain_settings, name)
        }
        domain = getattr(scene_fs, "domain_object", None)
        old_domain = None if domain is None else (tuple(domain.location), tuple(domain.dimensions))
        changed = []
        try:
            for name, value in changes.items():
                if not hasattr(domain_settings, name):
                    raise RuntimeError(f"Capacity option cannot apply setting {name!r}")
                rna = getattr(domain_settings, "bl_rna", None)
                properties = getattr(rna, "properties", {})
                rna_property = properties.get(name)
                if rna_property is not None and rna_property.type == "INT":
                    value = int(value)
                setattr(domain_settings, name, value)
                # RNA clamps out-of-range assignments to the declared
                # [min, max] silently, so a write can succeed while storing a
                # different number than the one the dialog costed. Reporting
                # that as approved would break the budget promise this whole
                # flow exists to keep -- read back and fail into the rollback.
                stored = getattr(domain_settings, name)
                if isinstance(value, float):
                    matches = math.isclose(stored, value, rel_tol=1e-6)
                else:
                    matches = stored == value
                if not matches:
                    raise RuntimeError(
                        f"setting {name!r} was stored as {stored!r} instead of "
                        f"{value!r}; it is outside the range this build allows"
                    )
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
                setattr(domain_settings, name, value)
            if domain is not None and old_domain is not None:
                domain.location, domain.dimensions = old_domain
            self.report({"ERROR"}, f"Capacity option was not applied: {exc}")
            scene_fs.status = f"Capacity option was not applied: {exc}"
            return {"CANCELLED"}

        bridge.clear_error(scene)
        scene_fs.status = "Approved capacity settings: " + ", ".join(changed)
        try:
            sess = bridge.initialize_session(scene)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Approved retry stopped safely: {exc}")
            scene_fs.status = f"Approved retry stopped safely: {exc}"
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Solver ready with approved option: {suggestion.label}",
        )
        scene_fs.status = f"Solver ready on {sess.solver.device}; approved {suggestion.label}"
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
        fs.status = "Capacity options ready â€” review and confirm one option"
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
        name="Frames",
        default=1,
        min=1,
        max=250,
        description="Set how many simulation frames this action advances; higher values perform more solver work before returning",
    )

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


def _animation_override(context):
    if bool(getattr(getattr(bpy, "app", None), "background", False)):
        raise RuntimeError("Preview needs a Blender window showing this scene")
    window = getattr(context, "window", None)
    if window is None or window.scene != context.scene:
        raise RuntimeError("Preview needs a Blender window showing this scene")
    for area in window.screen.areas:
        if area.type in {"VIEW_3D", "DOPESHEET_EDITOR", "GRAPH_EDITOR", "NLA_EDITOR", "SEQUENCE_EDITOR"}:
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                return dict(window=window, screen=window.screen, area=area, region=region)
    raise RuntimeError("Preview needs a 3D View or animation editor in this window")


def _stop_failed_playback(context):
    """Best-effort timeline cleanup, returning any actionable cleanup error."""
    window = getattr(context, "window", None)
    try:
        if window and window.scene == context.scene and window.screen.is_animation_playing:
            with context.temp_override(**_animation_override(context)):
                bpy.ops.screen.animation_cancel(restore_frame=False)
    except (RuntimeError, ReferenceError) as exc:
        return f"; timeline could not pause: {exc}"
    return ""


class _RestartConfirmation:
    def invoke(self, context, event):
        if _pending().changes_require_restart(context.scene):
            self.allow_restart = True
            self._confirmation_token = _pending().change_confirmation_token(context.scene)
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def draw(self, context):
        self.layout.label(text=f"Restart simulation from frame {context.scene.frame_start}?")
        self.layout.label(text="Current preview progress will be cleared.")

    def _validate_confirmation(self, scene):
        token = getattr(self, "_confirmation_token", None)
        if token is not None and token != _pending().change_confirmation_token(scene):
            raise RuntimeError("Scene or settings changed; review the restart again")


class _PreviewStart(_RestartConfirmation):
    """Prepare the current simulation and start Blender playback"""

    bl_idname = "fluidsim.preview_start"
    bl_label = "Preview"
    _cache_preview = False
    allow_restart: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _mutation_poll(cls, context)

    def execute(self, context):
        from ..blender import playback

        scene = context.scene
        try:
            self._validate_confirmation(scene)
            with context.temp_override(**_animation_override(context)):
                if not bpy.ops.screen.animation_play.poll():
                    raise RuntimeError("No editor can start preview playback")
                cache_preview = self._cache_preview
                if cache_preview:
                    playback.prepare_cache_preview(scene)
                else:
                    playback.prepare_live_preview(
                        scene, depsgraph=context.evaluated_depsgraph_get(),
                        allow_restart=getattr(self, "allow_restart", False))
                with playback.suppress_mode_transitions():
                    scene.fluidsim.cache_playback_enabled = cache_preview
                    scene.fluidsim.live = True
                if not context.window.screen.is_animation_playing:
                    if bpy.ops.screen.animation_play() != {"FINISHED"}:
                        raise RuntimeError("Preview was prepared but playback did not start")
                scene.fluidsim.status = "Playing preview"
            return {"FINISHED"}
        except Exception as exc:
            scene.fluidsim.live = False
            cleanup_error = _stop_failed_playback(context)
            scene.fluidsim.status = f"Preview could not start: {exc}{cleanup_error}"
            self.report({"ERROR"}, scene.fluidsim.status)
            return {"CANCELLED"}


class FLUIDSIM_OT_preview_start(_PreviewStart, bpy.types.Operator):
    bl_idname = "fluidsim.preview_start"
    bl_label = "Preview"
    allow_restart: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})


class FLUIDSIM_OT_cache_preview_start(_PreviewStart, bpy.types.Operator):
    """Validate the cached frame and start Blender playback"""

    bl_idname = "fluidsim.cache_preview_start"
    bl_label = "Preview Cache"
    _cache_preview = True

    @classmethod
    def poll(cls, context):
        from ..blender import production_bake

        scene = getattr(context, "scene", None)
        if getattr(scene, "fluidsim", None) is None:
            return False
        if production_bake.bake_is_active(scene):
            cls.poll_message_set("Production Bake owns this scene")
            return False
        return True

    def invoke(self, context, event):
        return self.execute(context)


class FLUIDSIM_OT_preview_pause(bpy.types.Operator):
    """Pause playback, keeping the current simulation and frame"""

    bl_idname = "fluidsim.preview_pause"
    bl_label = "Pause"

    def execute(self, context):
        try:
            with context.temp_override(**_animation_override(context)):
                if context.window.screen.is_animation_playing:
                    bpy.ops.screen.animation_cancel(restore_frame=False)
            context.scene.fluidsim.live = False
            context.scene.fluidsim.status = "Preview paused"
            return {"FINISHED"}
        except Exception as exc:
            context.scene.fluidsim.live = False
            self.report({"ERROR"}, f"Could not pause timeline: {exc}")
            return {"CANCELLED"}


class _ApplyChanges(_RestartConfirmation):
    """Apply current changes, restarting only with explicit consent"""

    bl_idname = "fluidsim.apply_changes"
    bl_label = "Apply Changes"
    _force_restart = False
    allow_restart: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _mutation_poll(cls, context)

    def execute(self, context):
        from ..blender import playback

        scene = context.scene
        window = getattr(context, "window", None)
        was_playing = bool(window and window.scene == scene and window.screen.is_animation_playing)
        try:
            self._validate_confirmation(scene)
            if was_playing:
                with context.temp_override(**_animation_override(context)):
                    bpy.ops.screen.animation_cancel(restore_frame=False)
            scene.fluidsim.live = False
            _pending().apply_changes(
                scene, allow_restart=getattr(self, "allow_restart", False),
                depsgraph=context.evaluated_depsgraph_get(),
                force_restart=self._force_restart)
            if was_playing:
                with context.temp_override(**_animation_override(context)):
                    with playback.suppress_mode_transitions():
                        scene.fluidsim.live = True
                    if bpy.ops.screen.animation_play() != {"FINISHED"}:
                        raise RuntimeError("Changes applied, but playback could not resume")
            scene.fluidsim.status = "Changes applied" if not was_playing else "Playing preview"
            return {"FINISHED"}
        except Exception as exc:
            scene.fluidsim.live = False
            cleanup_error = _stop_failed_playback(context)
            scene.fluidsim.status = f"Changes not completed: {exc}{cleanup_error}"
            self.report({"ERROR"}, scene.fluidsim.status)
            return {"CANCELLED"}


class FLUIDSIM_OT_apply_changes(_ApplyChanges, bpy.types.Operator):
    bl_idname = "fluidsim.apply_changes"
    bl_label = "Apply Changes"
    allow_restart: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})


class FLUIDSIM_OT_restart_simulation(_ApplyChanges, bpy.types.Operator):
    """Discard preview progress and restart at the scene start frame"""

    bl_idname = "fluidsim.restart_simulation"
    bl_label = "Restart Simulation"
    _force_restart = True
    allow_restart: BoolProperty(default=True, options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, event):
        return self.execute(context)


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
        from ..blender import production_bake

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
        from ..blender import playback

        fs = context.scene.fluidsim
        if not playback.set_cache_playback_enabled(context.scene, True):
            return {"CANCELLED"}
        with playback.suppress_mode_transitions():
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
