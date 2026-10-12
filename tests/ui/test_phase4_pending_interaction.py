"""Behavioral contracts for Phase-4 pending/reinitialize interaction."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from settings_doubles import full_settings, split_settings


def _bridge(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.context = SimpleNamespace()
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(frame_change_post=[]))
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    output_publication = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    output_publication.refresh_existing_output_settings = lambda *_args, **_kwargs: None
    output_publication.update_outputs = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication", output_publication)
    preferences = types.ModuleType("fluidSim_warp_blender51.preferences")
    preferences.bpy = bpy
    preferences.configured_device = lambda: getattr(
        getattr(
            getattr(bpy.context, "preferences", None), "addons", {}
        ).get("fluidSim_warp_blender51", SimpleNamespace(
            preferences=SimpleNamespace(device="cuda:0"))).preferences,
        "device", "cuda:0")
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.preferences", preferences)
    sys.modules.pop("fluidSim_warp_blender51.blender.pending_settings", None)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    bridge = importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")
    bridge.pending = importlib.import_module(
        "fluidSim_warp_blender51.blender.pending_settings")
    return bridge


def _scene(*, voxel_size=0.1):
    fs = full_settings(
        live=True, cache_playback_enabled=False, status="",
        domain_object=None, voxel_size=voxel_size, leaf_capacity=64,
        max_particles=1000, auto_grow_particles=True,
        particle_growth_failure_policy="DEFER",
        max_whitewater=200, profiling_enabled=False,
        profiling_sync=True, topology_expand=1, open_boundaries=False,
        domain_sdf_walls=True,
        gravity=(0.0, 0.0, -9.81), use_scene_gravity=False,
        flip_ratio=0.95, cfl=3.0,
    )
    return SimpleNamespace(
        fluidsim=split_settings(fs), objects=[], frame_start=1, frame_current=1,
        use_gravity=True, gravity=(0.0, 0.0, -9.81),
    )


def _activate(bridge, scene, *, frame=5):
    settings = SimpleNamespace(
        voxel_size=scene.fluidsim.domain_object.fluidsim_domain.voxel_size,
        leaf_capacity=scene.fluidsim.domain_object.fluidsim_domain.leaf_capacity,
        max_particles=scene.fluidsim.domain_object.fluidsim_domain.max_particles,
        auto_grow_particles=scene.fluidsim.domain_object.fluidsim_domain.auto_grow_particles,
        particle_growth_failure_policy=scene.fluidsim.domain_object.fluidsim_domain.particle_growth_failure_policy,
        max_whitewater=scene.fluidsim.domain_object.fluidsim_domain.max_whitewater,
        topology_expand=scene.fluidsim.domain_object.fluidsim_domain.topology_expand,
        open_boundaries=scene.fluidsim.domain_object.fluidsim_domain.open_boundaries,
        domain_sdf_walls=scene.fluidsim.domain_object.fluidsim_domain.domain_sdf_walls,
        flip_ratio=scene.fluidsim.domain_object.fluidsim_domain.flip_ratio,
        cfl=scene.fluidsim.domain_object.fluidsim_domain.cfl,
        gravity=scene.fluidsim.domain_object.fluidsim_domain.gravity,
    )
    solver = SimpleNamespace(frame=frame, settings=settings, particles=object())
    sess = bridge.session()
    sess.solver = solver
    sess.scene_identity = bridge.scene_identity(scene)
    sess.start_frame = scene.frame_start
    sess.last_frame = frame
    bridge.pending.capture_applied_state(scene, sess)
    return sess, solver


def test_particle_capacity_policy_updates_live_without_restarting_or_rewriting_runtime_capacity(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene)
    solver.particles = SimpleNamespace(capacity=4096)

    scene.fluidsim.domain_object.fluidsim_domain.max_particles = 2048
    scene.fluidsim.domain_object.fluidsim_domain.auto_grow_particles = False
    scene.fluidsim.domain_object.fluidsim_domain.particle_growth_failure_policy = "PAUSE"
    bridge.pending.update_particle_capacity_policy(scene)

    assert sess.solver is solver
    assert solver.particles.capacity == 4096
    assert solver.settings.max_particles == 1000
    assert solver.settings.auto_grow_particles is False
    assert solver.settings.particle_growth_failure_policy == "PAUSE"
    pending = bridge.pending.pending_presentation(scene)
    assert pending["action"] == "REINIT_PENDING"
    assert any("max_particles" in reason for reason in pending["reasons"])
    assert "active pool never shrinks automatically" in scene.fluidsim.status


def test_midrun_reinit_edit_is_pending_and_preserves_solver_state(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene)
    particles = solver.particles

    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.05
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.REBUILD)

    pending = bridge.pending.pending_presentation(scene)
    assert pending["action"] == "REINIT_PENDING"
    assert any("voxel_size" in reason for reason in pending["reasons"])
    assert sess.solver is solver
    assert sess.solver.particles is particles
    assert solver.settings.voxel_size == 0.1


def test_reset_projection_preserves_closed_domain_wall_invariant(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    scene.fluidsim.domain_object.fluidsim_domain.domain_sdf_walls = False
    _sess, solver = _activate(bridge, scene)

    bridge.pending.apply_reset_solver_settings(scene)

    assert solver.settings.open_boundaries is False
    assert solver.settings.domain_sdf_walls is True


def test_pending_state_is_isolated_between_scenes(monkeypatch):
    bridge = _bridge(monkeypatch)
    first = _scene()
    second = _scene()
    _activate(bridge, first)
    first.fluidsim.domain_object.fluidsim_domain.leaf_capacity = 128
    bridge.pending.request_scene_update(first, bridge.pending.SceneUpdate.REBUILD)

    # Switching the single live solver to another scene must not erase the
    # first scene's applied snapshot or pending reasons.
    _activate(bridge, second)
    second.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.2
    bridge.pending.request_scene_update(second, bridge.pending.SceneUpdate.REBUILD)

    first_pending = bridge.pending.pending_presentation(first)
    second_pending = bridge.pending.pending_presentation(second)
    assert first_pending["action"] == "REINIT_PENDING"
    assert first_pending["reasons"] == ("Reinitialize: leaf_capacity",)
    assert second_pending["action"] == "REINIT_PENDING"
    assert second_pending["reasons"] == ("Reinitialize: voxel_size",)


def test_rewind_reconstructs_with_pending_allocation_once(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, _solver = _activate(bridge, scene, frame=7)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.04
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.REBUILD)
    calls = []

    def initialize(actual_scene, **_kwargs):
        calls.append(actual_scene.fluidsim.domain_object.fluidsim_domain.voxel_size)
        sess.solver = SimpleNamespace(
            frame=0,
            settings=SimpleNamespace(voxel_size=actual_scene.fluidsim.domain_object.fluidsim_domain.voxel_size),
        )
        sess.scene_identity = bridge.scene_identity(actual_scene)
        bridge.pending.capture_applied_state(actual_scene, sess)
        return sess

    monkeypatch.setattr(sess, "initialize", initialize)
    scene.frame_current = scene.frame_start
    from fluidSim_warp_blender51.blender import playback
    playback.handle_frame_change(sess, scene, depsgraph=None)

    assert calls == [0.04]
    assert bridge.pending.pending_presentation(scene)["action"] == "NONE"
    assert sess.last_frame == scene.frame_start


def test_reset_lifecycle_pending_applies_exactly_once(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    source_pg = SimpleNamespace(
        role="SOURCE", point_source_representation="DIRECT_APPEND",
        mesh_evaluation_mode="FIRST_SIM_FRAME", emission_mode="INITIAL_FILL",
        auto_inject_seed=False, emit_rate=8.0,
        source_sampling_mode="LEGACY_ABSOLUTE", source_sampling_multiplier=1.0,
    )
    scene.objects = [SimpleNamespace(name="Source", fluidsim=split_settings(source_pg))]
    sess, _solver = _activate(bridge, scene, frame=4)
    source_pg.emission_mode = "CONTINUOUS_INFLOW"
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.RESET)
    resets = []

    def reset(actual_scene, **_kwargs):
        resets.append(source_pg.emission_mode)
        bridge.pending.capture_applied_state(actual_scene, sess, reset_only=True)

    monkeypatch.setattr(sess, "reset", reset)
    assert bridge.pending.apply_pending_on_rewind(scene, sess) is True
    assert bridge.pending.apply_pending_on_rewind(scene, sess) is False
    assert resets == ["CONTINUOUS_INFLOW"]
    assert bridge.pending.pending_presentation(scene)["action"] == "NONE"


def test_scripted_signature_change_is_detected_without_rna_callback(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, _solver = _activate(bridge, scene, frame=3)

    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.025
    bridge.pending.refresh_pending_state(scene, sess)

    assert bridge.pending.pending_presentation(scene)["action"] == "REINIT_PENDING"
    assert "Reinitialize: voxel_size" in bridge.pending.pending_presentation(scene)["reasons"]


def test_revert_pending_values_restores_applied_snapshot(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    _activate(bridge, scene, frame=3)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.025
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.REBUILD)

    assert bridge.pending.revert_pending_values(scene) is True
    assert scene.fluidsim.domain_object.fluidsim_domain.voxel_size == 0.1
    assert bridge.pending.pending_presentation(scene)["action"] == "NONE"


def test_failed_pending_apply_retains_action_and_reason(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, _solver = _activate(bridge, scene, frame=3)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.025
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.REBUILD)
    monkeypatch.setattr(
        sess, "initialize", lambda _scene, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("allocation failed")))

    try:
        bridge.pending.apply_pending_reinitialize(scene)
    except RuntimeError:
        pass

    pending = bridge.pending.pending_presentation(scene)
    assert pending["action"] == "REINIT_PENDING"
    assert pending["reasons"]
    assert pending["blocked_reason"] == "allocation failed"


def test_pending_publication_warning_updates_scene_and_session(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, _solver = _activate(bridge, scene)
    warning = "Render input synchronization failed: RuntimeError: offline"

    recorded = bridge.pending._record_publication_warning(
        scene,
        sess,
        SimpleNamespace(warning=warning),
    )

    assert recorded is True
    assert scene.fluidsim.status == f"Output published with warning: {warning}"
    assert sess.current_error(scene) == warning


def test_midrun_reset_edit_is_pending_without_resetting(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene, frame=6)
    scene.fluidsim.domain_object.fluidsim_domain.topology_expand = 3

    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.RESET)

    assert bridge.pending.pending_presentation(scene)["action"] == "RESET_PENDING"
    assert sess.solver is solver
    assert solver.settings.topology_expand == 1


def test_rewind_without_pending_resets_once(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, _solver = _activate(bridge, scene, frame=6)
    calls = []
    monkeypatch.setattr(
        sess, "reset", lambda actual, **_kwargs: calls.append(actual))
    scene.frame_current = scene.frame_start

    from fluidSim_warp_blender51.blender import playback
    playback.handle_frame_change(sess, scene, depsgraph=None)

    assert calls == [scene]
    assert sess.last_frame == scene.frame_start


def test_rewind_apply_failure_stops_playback_and_retains_solver(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene, frame=6)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.04
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.REBUILD)
    monkeypatch.setattr(sess, "initialize", lambda _scene: (_ for _ in ()).throw(
        RuntimeError("allocation failed")))
    scene.frame_current = scene.frame_start

    from fluidSim_warp_blender51.blender import playback
    playback.handle_frame_change(sess, scene, depsgraph=None)

    assert scene.fluidsim.live is False
    assert sess.solver is solver
    assert bridge.pending.pending_presentation(scene)["action"] == "REINIT_PENDING"
    assert "playback stopped" in scene.fluidsim.status


def test_prepare_step_reuses_depsgraph_signature_snapshot(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, _solver = _activate(bridge, scene, frame=3)
    calls = []
    original = bridge.pending._reset_signature

    def counted(actual_scene):
        calls.append(actual_scene)
        return original(actual_scene)

    monkeypatch.setattr(bridge.pending, "_reset_signature", counted)
    monkeypatch.setattr(
        bridge.pending, "apply_live_settings_safely", lambda *_args: True)
    bridge.pending._on_interaction_state_update(scene, object())
    sess.prepare_step(scene)
    sess.prepare_step(scene)

    assert calls == [scene]


def test_direct_step_refreshes_immediate_scripted_edit_without_depsgraph(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, _solver = _activate(bridge, scene, frame=3)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.025
    monkeypatch.setattr(sess, "step", lambda *_args, **_kwargs: "stepped")

    assert bridge.step_frame(scene, sess, write_cache=False) == "stepped"
    assert bridge.pending.pending_presentation(scene)["action"] == "REINIT_PENDING"


def test_scripted_edit_handler_refreshes_cached_pending_presentation(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    _activate(bridge, scene, frame=3)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.025

    bridge.pending._on_interaction_state_update(scene, object())

    assert bridge.pending.pending_presentation(scene)["action"] == "REINIT_PENDING"


def test_apply_operators_do_not_destroy_another_scenes_solver(monkeypatch):
    bridge = _bridge(monkeypatch)
    first = _scene()
    second = _scene()
    sess, first_solver = _activate(bridge, first, frame=4)
    _activate(bridge, second, frame=4)
    second.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.05
    bridge.pending.refresh_pending_state(second, sess)
    # Restore ownership to the first scene while keeping the second snapshot.
    sess.solver = first_solver
    sess.scene_identity = bridge.scene_identity(first)

    for apply in (bridge.pending.apply_pending_reset, bridge.pending.apply_pending_reinitialize):
        try:
            apply(second)
        except RuntimeError:
            pass
        else:
            raise AssertionError("cross-scene pending apply was not blocked")
    assert sess.solver is first_solver
    assert sess.owns_scene(first)


def test_apply_reinitialize_with_no_pending_is_a_noop(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene, frame=4)

    assert bridge.pending.apply_pending_reinitialize(scene) is False
    assert sess.solver is solver


def test_pending_presentation_owns_reset_and_reinitialize_copy(monkeypatch):
    bridge = _bridge(monkeypatch)

    reset_scene = SimpleNamespace(
        fluidsim_pending_action="RESET_PENDING",
        fluidsim_pending_count=1,
        fluidsim_pending_reasons="Reset: flip_ratio",
        fluidsim_pending_blocked="",
        fluidsim_pending_epoch=bridge.pending._presentation_epoch,
    )
    reset = bridge.pending.pending_presentation(reset_scene)
    assert reset["summary"] == "1 change(s) staged for reset"
    assert reset["reason_detail"] == "will apply after reset"
    assert reset["action_label"] == "Reset Now"
    assert reset["action_operator"] == "fluidsim.apply_pending_reset"

    reinitialize_scene = SimpleNamespace(
        fluidsim_pending_action="REINIT_PENDING",
        fluidsim_pending_count=2,
        fluidsim_pending_reasons=(
            "Reinitialize: voxel_size\nReinitialize: max_particles"),
        fluidsim_pending_blocked="",
        fluidsim_pending_epoch=bridge.pending._presentation_epoch,
    )
    reinitialize = bridge.pending.pending_presentation(reinitialize_scene)
    assert reinitialize["summary"] == (
        "2 change(s) staged to reinitialize the solver")
    assert reinitialize["reason_detail"] == (
        "will apply when the solver is reinitialized")
    assert reinitialize["action_label"] == "Reinitialize Now"
    assert reinitialize["action_operator"] == (
        "fluidsim.apply_pending_reinitialize")

    reset_copy = " ".join((
        reset["summary"], reset["reason_detail"], reset["action_label"]))
    reinitialize_copy = " ".join((
        reinitialize["summary"], reinitialize["reason_detail"],
        reinitialize["action_label"]))
    assert "restart" not in reset_copy.lower()
    assert "restart" not in reinitialize_copy.lower()


def test_revert_never_changes_device_preferences(monkeypatch):
    bridge = _bridge(monkeypatch)
    prefs = SimpleNamespace(device="cuda:0")
    bridge.bpy.context.preferences = SimpleNamespace(addons={
        "fluidSim_warp_blender51": SimpleNamespace(preferences=prefs),
    })
    from fluidSim_warp_blender51 import preferences
    monkeypatch.setattr(preferences, "bpy", bridge.bpy)
    scene = _scene()
    _activate(bridge, scene, frame=4)
    prefs.device = "cuda:1"
    bridge.pending.refresh_pending_state(scene, bridge.session())

    assert bridge.pending.revert_pending_values(scene) is False
    assert prefs.device == "cuda:1"
    assert "never changed automatically" in bridge.pending.pending_presentation(scene)[
        "blocked_reason"]


def test_reserve_edit_debounces_owned_plan_refresh_outside_rna_callback(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    other = _scene()
    sess, solver = _activate(bridge, scene)
    solver.particles = SimpleNamespace(capacity=4096)
    old_plan = SimpleNamespace(scene_signature="old")
    new_plan = SimpleNamespace(scene_signature="new")
    sess.set_capacity_plan(scene, old_plan, memory_bypassed=True)
    refreshed = []
    timer_callbacks = []
    bridge.bpy.app.timers = SimpleNamespace(register=lambda callback, **_kwargs: (
        timer_callbacks.append(callback)))

    def refresh(actual):
        refreshed.append(actual)
        return sess.set_capacity_plan(actual, new_plan)

    monkeypatch.setattr(sess, "refresh_capacity_plan", refresh)
    scene.fluidsim.domain_object.fluidsim_domain.max_particles = 2048
    bridge.pending.update_particle_capacity_policy(scene)
    # Another slider tick invalidates state but must reuse the same timer.
    scene.fluidsim.domain_object.fluidsim_domain.max_particles = 3072
    bridge.pending.update_particle_capacity_policy(scene)

    assert refreshed == []
    assert len(timer_callbacks) == 1
    assert sess.capacity_plan_for_scene(scene) is None
    assert sess.capacity_plan_for_scene(other) is None
    assert sess.capacity_plan_memory_bypassed is False
    assert solver.particles.capacity == 4096
    pending = bridge.pending.pending_presentation(scene)
    assert pending["action"] == "REINIT_PENDING"
    assert any("max_particles" in reason for reason in pending["reasons"])

    timer_callbacks[0]()

    assert refreshed == [scene]
    assert sess.capacity_plan_for_scene(scene) is new_plan


# --- B55: Revert restores the applied run, not the current selection ---------


class _ScaledMatrix:
    """``matrix_world`` of an unrotated object scaled uniformly by ``s``."""

    def __init__(self, s):
        self.s = float(s)

    def __matmul__(self, vector):
        x, y, z = (float(component) * self.s for component in vector)
        return SimpleNamespace(x=x, y=y, z=z)

    def __getitem__(self, row):
        return tuple((self.s if row == column else 0.0) if column < 3
                     else (1.0 if row == 3 else 0.0) for column in range(4))

    def copy(self):
        return _ScaledMatrix(self.s)


def _second_domain(scene, **changes):
    from settings_doubles import domain_object

    values = dict(vars(scene.fluidsim.domain_object.fluidsim_domain))
    values.update(changes)
    obj = domain_object(**values)
    obj.name = "Second Domain"
    return obj


def test_revert_after_switching_domains_restores_the_applied_domain_only(
        monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    applied = scene.fluidsim.domain_object
    _activate(bridge, scene, frame=3)
    applied.fluidsim_domain.leaf_capacity = 128  # a pending edit on A
    second = _second_domain(scene, leaf_capacity=999, particles_per_voxel=3.0,
                            voxel_size=0.3)
    untouched = dict(vars(second.fluidsim_domain))
    scene.fluidsim.domain_object = second
    bridge.pending.refresh_pending_state(scene, bridge.session())

    assert bridge.pending.revert_pending_values(scene) is True

    assert scene.fluidsim.domain_object is applied
    assert applied.fluidsim_domain.leaf_capacity == 64
    # The newly selected domain keeps every one of its own values.
    assert vars(second.fluidsim_domain) == untouched
    assert bridge.pending.pending_presentation(scene)["action"] == "NONE"


def test_revert_under_a_scale_change_restores_divisions_in_any_snapshot_order(
        monkeypatch):
    from fluidSim_warp_blender51.sim.quality_contract import (
        divisions_for_voxel_size, voxel_size_for_divisions)

    bridge = _bridge(monkeypatch)
    scene = _scene()
    obj = scene.fluidsim.domain_object
    obj.matrix_world = _ScaledMatrix(1.0)
    values = dict(vars(obj.fluidsim_domain))
    values.pop("voxel_size")

    class DerivedGroup(SimpleNamespace):
        """voxel_size derived from divisions over the domain's longest axis."""

        @property
        def voxel_size(self):
            return voxel_size_for_divisions(
                self.resolution_divisions, 4.0 * obj.matrix_world.s)

        @voxel_size.setter
        def voxel_size(self, value):
            self.resolution_divisions = divisions_for_voxel_size(
                value, 4.0 * obj.matrix_world.s)

    obj.fluidsim_domain = DerivedGroup(**values)
    obj.fluidsim_domain.resolution_divisions = 64
    sess, _solver = _activate(bridge, scene, frame=3)
    state = bridge.pending._interaction_state(scene)
    # The snapshot used to be a frozenset walk, so its order followed the
    # string hash seed; put divisions first, the order that used to lose.
    snapshot = dict(state.reinit_snapshot)
    state.reinit_snapshot = {
        "resolution_divisions": snapshot.pop("resolution_divisions"), **snapshot}

    obj.matrix_world = _ScaledMatrix(2.0)  # the user scales the domain 2x
    bridge.pending.refresh_pending_state(scene, sess)
    assert bridge.pending.pending_presentation(scene)["action"] == "REINIT_PENDING"

    assert bridge.pending.revert_pending_values(scene) is True

    assert obj.matrix_world.s == 1.0
    assert obj.fluidsim_domain.resolution_divisions == 64
    assert bridge.pending.pending_presentation(scene)["action"] == "NONE"


class _DeletableObject:
    """A bpy Object wrapper whose ID can be deleted (pyrna then raises)."""

    def __init__(self, name):
        self.name_ = name
        self.removed = False
        self.group = SimpleNamespace(
            role="NONE", point_source_representation="DIRECT_APPEND",
            mesh_evaluation_mode="FIRST_SIM_FRAME", emission_mode="INITIAL_FILL",
            auto_inject_seed=False, emit_rate=8.0,
            source_sampling_mode="LEGACY_ABSOLUTE", source_sampling_multiplier=1.0)

    def __getattr__(self, name):
        if self.__dict__["removed"]:
            raise ReferenceError("StructRNA of type Object has been removed")
        if name == "fluidsim":
            return self.__dict__["group"]
        if name == "name":
            return self.__dict__["name_"]
        raise AttributeError(name)

    def as_pointer(self):
        if self.removed:
            raise ReferenceError("StructRNA of type Object has been removed")
        return id(self)


def test_revert_skips_a_deleted_snapshot_object_and_restores_the_rest(
        monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    doomed = _DeletableObject("Camera")
    kept = _DeletableObject("Source")
    scene.objects = [doomed, kept]
    _activate(bridge, scene, frame=3)

    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.025
    kept.group.emission_mode = "CONTINUOUS_INFLOW"
    doomed.removed = True
    scene.objects = [kept]
    bridge.pending.refresh_pending_state(scene, bridge.session())

    bridge.pending.revert_pending_values(scene)  # must not raise

    assert scene.fluidsim.domain_object.fluidsim_domain.voxel_size == 0.1
    assert kept.group.emission_mode == "INITIAL_FILL"
    reasons = bridge.pending.pending_presentation(scene)["reasons"]
    assert not any(reason.startswith("Reinitialize") for reason in reasons)
    # The deleted camera never fed the simulation (B59), so nothing is left.
    assert bridge.pending.pending_presentation(scene)["action"] == "NONE"

# --- B59: only simulation participants enter the reset signature ------------


def _object(name, role="NONE", **overrides):
    values = dict(role=role, point_source_representation="DIRECT_APPEND",
                  mesh_evaluation_mode="FIRST_SIM_FRAME",
                  emission_mode="INITIAL_FILL", auto_inject_seed=False,
                  emit_rate=8.0, source_sampling_mode="LEGACY_ABSOLUTE",
                  source_sampling_multiplier=1.0)
    values.update(overrides)
    return SimpleNamespace(name=name, fluidsim=SimpleNamespace(**values))


def test_adding_a_non_participating_object_is_not_a_pending_reset(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    scene.objects = [_object("Source", "SOURCE")]
    sess, _solver = _activate(bridge, scene, frame=3)

    # A camera, a light, or an output object Output Preparation creates.
    scene.objects.append(_object("FluidSim Particles"))
    scene.objects.append(_object("Camera"))
    bridge.pending.refresh_pending_state(scene, sess)

    assert bridge.pending.pending_presentation(scene)["action"] == "NONE"


def test_participants_seeds_and_role_changes_still_make_reset_pending(
        monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    bystander = _object("Cube")
    scene.objects = [bystander]
    sess, _solver = _activate(bridge, scene, frame=3)

    scene.objects.append(_object("Seed", auto_inject_seed=True))
    bridge.pending.refresh_pending_state(scene, sess)
    assert bridge.pending.pending_presentation(scene)["action"] == "RESET_PENDING"

    scene.objects.pop()
    bystander.fluidsim.role = "COLLIDER"
    bridge.pending.refresh_pending_state(scene, sess)
    assert bridge.pending.pending_presentation(scene)["action"] == "RESET_PENDING"

    # Revert can still put a newly tagged object back to role NONE.
    assert bridge.pending.revert_pending_values(scene) is True
    assert bystander.fluidsim.role == "NONE"

# --- B60: saved pending badges do not outlive the session that wrote them ---


def test_pending_badges_saved_in_a_file_do_not_show_after_loading_it(
        monkeypatch):
    bridge = _bridge(monkeypatch)
    from fluidSim_warp_blender51.blender import playback
    monkeypatch.setattr(playback, "clear_cache_cursors", lambda: None)
    scene = _scene()
    sess, _solver = _activate(bridge, scene, frame=3)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.025
    bridge.pending.refresh_pending_state(scene, sess)
    assert bridge.pending.pending_presentation(scene)["action"] == "REINIT_PENDING"

    # Opening a file: load_pre drops the session and every applied snapshot;
    # the scene's custom properties come back with whatever was saved.
    bridge.pending._on_load_pre("saved.blend")

    presentation = bridge.pending.pending_presentation(scene)
    assert presentation["action"] == "NONE"
    assert presentation["count"] == 0
    assert presentation["reasons"] == ()
    assert bridge.pending.reinitialize_eligibility(scene)[0] is False
    assert bridge.pending.revert_eligibility(scene)[0] is False

# --- B58: live/sync edits re-check the narrow-band admission refusal ---------


def _narrow_band_run(bridge, monkeypatch):
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings

    scene = _scene()
    domain = scene.fluidsim.domain_object.fluidsim_domain
    domain.narrow_band_enabled = True
    domain.viscosity_enabled = True
    domain.viscosity_from_particles = False
    sess, solver = _activate(bridge, scene, frame=3)
    solver.settings = SolverSettings(
        device="cpu", narrow_band_enabled=True, viscosity_enabled=True,
        viscosity_from_particles=False, flip_ratio=0.95)
    errors = []
    monkeypatch.setattr(sess, "set_error", lambda _scene, message: errors.append(message))
    return scene, domain, sess, solver, errors


def test_a_live_edit_narrow_band_refuses_is_refused_and_not_applied(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene, domain, _sess, solver, errors = _narrow_band_run(bridge, monkeypatch)

    domain.flip_ratio = 0.5
    domain.viscosity_from_particles = True  # per-particle viscosity
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.LIVE)

    # The running narrow-band solver keeps its admitted, admissible values.
    assert solver.settings.viscosity_from_particles is False
    assert solver.settings.flip_ratio == 0.95
    assert "Narrow-Band Particles cannot run" in scene.fluidsim.status
    assert any("Viscosity from particles" in message for message in errors)

    # Removing the conflict lets the held live edits through.
    domain.viscosity_from_particles = False
    errors.clear()
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.LIVE)
    assert solver.settings.flip_ratio == 0.5
    assert errors == []


def test_a_sync_edit_adding_a_scalar_field_source_is_refused(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene, _domain, sess, _solver, errors = _narrow_band_run(bridge, monkeypatch)
    synced = []
    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: synced.append(True))
    field = _object("Heat", "FIELD")
    field.fluidsim.field_operation = "SCALAR"
    scene.objects = [field]

    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.SYNC)

    assert synced == []
    assert "Narrow-Band Particles cannot run" in scene.fluidsim.status
    assert any("Scalar / Temperature" in message for message in errors)

    field.fluidsim.field_operation = "PRESSURE"
    bridge.pending.request_scene_update(scene, bridge.pending.SceneUpdate.SYNC)
    assert synced == [True]