"""Capability-owned Blender operators."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from ..blender import capacity_approval
from ..blender.scene_access import active_domain_object, active_domain_settings, scene_identity
from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


class FLUIDSIM_OT_initialize(bpy.types.Operator):
    """Create (or recreate) the GPU solver from current scene settings"""

    bl_idname = "fluidsim.initialize"
    bl_label = "Initialize Solver"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _runtime._mutation_poll(cls, context)

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
            sess = _runtime._bridge().initialize_session(
                scene,
                ignore_memory_safeguard=self.ignore_memory_safeguard,
            )
        except Exception as exc:  # noqa: BLE001
            plan = _runtime._owned_capacity_plan(scene)
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
                scene.fluidsim.status = "Capacity warning — approve a safe option below to run"
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
        if not _runtime._bake_poll(cls, context):
            return False
        sess = _runtime._bridge().session()
        if not sess.active:
            cls.poll_message_set("No solver session is active")
            return False
        owns_scene = getattr(sess, "owns_scene", None)
        if callable(owns_scene) and not owns_scene(getattr(context, "scene", None)):
            cls.poll_message_set("The active solver belongs to another scene")
            return False
        return True

    def execute(self, context):
        if not _runtime._bridge().release_session(context.scene):
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
        bridge = _runtime._bridge()
        bridge.clear_error(context.scene)
        fs = context.scene.fluidsim
        fs.status = "Solver ready" if bridge.session().active else "Solver not initialized"
        return {"FINISHED"}


def _update_view_layer():
    if getattr(bpy.context, "view_layer", None) is not None:
        bpy.context.view_layer.update()


class FLUIDSIM_OT_apply_capacity_suggestion(bpy.types.Operator):
    """Confirm, apply, and retry one planner-verified capacity option."""

    bl_idname = "fluidsim.apply_capacity_suggestion"
    bl_label = "Approve Capacity Changes & Run"
    # UNDO (REVIEW-011): this writes domain settings and the domain transform.
    # Without an undo step, Ctrl+Z still reverts those writes, folded into the
    # previous step. With one, it reverts exactly this approval, and the
    # reinit signature (voxel_size, capacities, domain bounds) flags that the
    # live solver needs reinitializing.
    bl_options = {"REGISTER", "UNDO"}

    suggestion_key: StringProperty(default="", options={"SKIP_SAVE"})
    expected_scene_signature: StringProperty(default="", options={"SKIP_SAVE"})
    expected_scene_identity: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    changes_json: StringProperty(default="{}", options={"HIDDEN", "SKIP_SAVE"})
    suggestion_label: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    estimated_bytes: StringProperty(default="0", options={"HIDDEN", "SKIP_SAVE"})
    budget_bytes: StringProperty(default="0", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        if not _runtime._bake_poll(cls, context):
            return False
        plan = _runtime._owned_capacity_plan(getattr(context, "scene", None))
        if not getattr(plan, "suggestions", ()):
            cls.poll_message_set("Initialize first to calculate safe capacity options")
            return False
        return True

    def invoke(self, context, _event):
        plan = _runtime._owned_capacity_plan(context.scene)
        suggestion = capacity_approval.find_suggestion(plan, self.suggestion_key)
        if suggestion is None:
            self.report({"WARNING"}, "Capacity option is stale; initialize again")
            return {"CANCELLED"}
        self.expected_scene_signature = _runtime._owned_capacity_signature(context.scene, plan)
        self.expected_scene_identity = str(scene_identity(context.scene))
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
        domain_settings = active_domain_settings(context.scene)
        for name, value in changes.items():
            old_value = getattr(domain_settings, name, None)
            layout.label(
                text=(
                    f"{name}: {capacity_approval.format_value(old_value)} → "
                    f"{capacity_approval.format_value(value)}"
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
        bridge = _runtime._bridge()
        scene = context.scene
        scene_fs = scene.fluidsim
        if self.expected_scene_identity and self.expected_scene_identity != str(
            scene_identity(scene)
        ):
            self.report(
                {"WARNING"},
                "Capacity option belongs to another scene; review this scene",
            )
            return {"CANCELLED"}
        if _runtime._owned_capacity_plan(scene) is None:
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
            and _runtime._owned_capacity_signature(scene, fresh_plan) != self.expected_scene_signature
        ):
            self.report(
                {"WARNING"},
                "Scene changed after the warning; initialize again for new options",
            )
            scene_fs.status = "Capacity options changed — review the refreshed warning"
            return {"CANCELLED"}
        suggestion = capacity_approval.find_suggestion(fresh_plan, self.suggestion_key)
        if suggestion is None:
            self.report(
                {"WARNING"},
                "This option is no longer inside the live safety budget",
            )
            scene_fs.status = "Capacity option is no longer safe — review alternatives"
            return {"CANCELLED"}

        domain_settings = active_domain_settings(scene)
        if domain_settings is None:
            message = "Capacity option cannot apply without a designated domain"
            self.report({"ERROR"}, message)
            scene_fs.status = message
            return {"CANCELLED"}
        try:
            changed = capacity_approval.apply_suggestion(
                domain_settings,
                active_domain_object(scene),
                suggestion.changes,
                build_settings=lambda: _runtime._solver_settings().build_settings(scene),
                on_transform_changed=_update_view_layer,
            )
        except capacity_approval.CapacityApplyError as exc:
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
        if not _runtime._bake_poll(cls, context):
            return False
        return bool(getattr(getattr(context, "scene", None), "fluidsim", None))

    def execute(self, context):
        bridge = _runtime._bridge()
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
        sess = _runtime._bridge().session()
        owns_scene = getattr(sess, "owns_scene", lambda _scene: sess.active)
        if not owns_scene(context.scene):
            self.report({"WARNING"}, "Solver not initialized for this scene")
            return {"CANCELLED"}
        _runtime._bridge().reset_simulation(context.scene)
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
        eligible, message = _runtime._pending().reinitialize_eligibility(scene)
        if not eligible and message:
            cls.poll_message_set(message)
        return eligible

    def execute(self, context):
        try:
            changed = _runtime._pending().apply_pending_reinitialize(context.scene)
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
        eligible, message = _runtime._pending().reset_eligibility(scene)
        if not eligible and message:
            cls.poll_message_set(message)
        return eligible

    def execute(self, context):
        try:
            changed = _runtime._pending().apply_pending_reset(context.scene)
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
        eligible, message = _runtime._pending().revert_eligibility(scene)
        if not eligible and message:
            cls.poll_message_set(message)
        return eligible

    def execute(self, context):
        try:
            reverted = _runtime._pending().revert_pending_values(context.scene)
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
        bridge = _runtime._bridge()
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
        if _runtime._pending().changes_require_restart(context.scene):
            self.allow_restart = True
            self._confirmation_token = _runtime._pending().change_confirmation_token(context.scene)
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def draw(self, context):
        self.layout.label(text=f"Restart simulation from frame {context.scene.frame_start}?")
        self.layout.label(text="Current preview progress will be cleared.")

    def _validate_confirmation(self, scene):
        token = getattr(self, "_confirmation_token", None)
        if token is not None and token != _runtime._pending().change_confirmation_token(scene):
            raise RuntimeError("Scene or settings changed; review the restart again")


class _PreviewStart(_RestartConfirmation):
    """Prepare the current simulation and start Blender playback"""

    bl_idname = "fluidsim.preview_start"
    bl_label = "Play"
    _cache_preview = False
    allow_restart: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _runtime._mutation_poll(cls, context)

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
    bl_label = "Play"
    allow_restart: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})


class FLUIDSIM_OT_cache_preview_start(_PreviewStart, bpy.types.Operator):
    """Validate the cached frame and start Blender playback"""

    bl_idname = "fluidsim.cache_preview_start"
    bl_label = "Play Cache"
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
    """Pause playback, keeping the current simulation and frame.

    Disarming lets the paused timeline be scrubbed without stepping or
    resetting the solver; Play (or the spacebar) re-arms and resumes it.
    """

    bl_idname = "fluidsim.preview_pause"
    bl_label = "Pause"

    def execute(self, context):
        try:
            with context.temp_override(**_animation_override(context)):
                if context.window.screen.is_animation_playing:
                    bpy.ops.screen.animation_cancel(restore_frame=False)
            context.scene.fluidsim.live = False
            context.scene.fluidsim.status = "Paused - press Play to resume"
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
        return _runtime._mutation_poll(cls, context)

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
            _runtime._pending().apply_changes(
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
        # A click asks first when stepped preview frames would be lost; script
        # and EXEC_DEFAULT callers restart without a dialog, as before.
        if _preview_progress_at_risk(context.scene):
            self._confirmation_token = _runtime._pending().change_confirmation_token(
                context.scene)
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)


def _preview_progress_at_risk(scene) -> bool:
    """Whether this scene owns a solver that has stepped past its start."""
    pending = _runtime._pending()
    sess = _runtime._bridge().session()
    owns_scene = getattr(sess, "owns_scene", None)
    if not callable(owns_scene) or not owns_scene(scene):
        return False
    return not pending._is_disposable_start(sess)


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
        return _runtime._mutation_poll(cls, context)

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _runtime._bridge()
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
        # Cache playback releases the shared solver session.
        if not _runtime._bake_poll(cls, context):
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
