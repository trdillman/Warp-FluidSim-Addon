"""Behavioral contracts for Phase-4 pending/reinitialize interaction."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from settings_doubles import split_settings


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
    fs = SimpleNamespace(
        live=True, cache_playback_enabled=False, status="",
        domain_object=None, voxel_size=voxel_size, leaf_capacity=64,
        max_particles=1000, auto_grow_particles=True,
        particle_growth_failure_policy="DEFER",
        max_whitewater=200, profiling_enabled=False,
        profiling_sync=True, topology_expand=1, open_boundaries=False,
        domain_sdf_walls=True, spacetime_enabled=False,
        spacetime_reference_mass=0.0, spacetime_seed=1729,
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
        spacetime_enabled=scene.fluidsim.domain_object.fluidsim_domain.spacetime_enabled,
        spacetime_reference_mass=scene.fluidsim.domain_object.fluidsim_domain.spacetime_reference_mass,
        spacetime_seed=scene.fluidsim.domain_object.fluidsim_domain.spacetime_seed,
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
        auto_inject_seed=False,
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


def test_st_reference_calibration_cache_scales_distinct_density_without_recompute(
        monkeypatch):
    bridge = _bridge(monkeypatch)
    calls = []
    solver = SimpleNamespace(
        settings=SimpleNamespace(particles_per_voxel=8.0, spacetime_seed=9),
        _calibrate_st_reference_mass=lambda: calls.append(1) or 3.5,
    )

    assert bridge.pending._cached_st_reference_mass(solver) == 3.5
    solver.settings.particles_per_voxel = 16.0
    assert bridge.pending._cached_st_reference_mass(solver) == 7.0
    assert calls == [1]


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
    sess._set_capacity_plan(scene, old_plan, memory_bypassed=True)
    refreshed = []
    timer_callbacks = []
    bridge.bpy.app.timers = SimpleNamespace(register=lambda callback, **_kwargs: (
        timer_callbacks.append(callback)))

    def refresh(actual):
        refreshed.append(actual)
        return sess._set_capacity_plan(actual, new_plan)

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
