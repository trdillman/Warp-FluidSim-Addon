"""A running Production Bake owns the shared Solver Session (decision D2).

Regression for review items B18, B19 and B26:
- B18: Apply Preset / Apply Pending, Inject Object, Initialize in another
  scene and Load Demo could reset, mutate, replace or release the solver a
  bake was stepping, because the guard checked only the context scene while
  the session is a process singleton.
- B19: live edits made mid-bake reached the baking solver (through property
  callbacks and through the bake's own per-step live re-application) under
  the cache's original settings identity.
- B26 (decision D3): any right-click cancelled the bake and was swallowed.
"""
from __future__ import annotations

from types import SimpleNamespace

import test_operator_poll_and_guards as harness


def _operators(monkeypatch, baking_scene):
    harness._install_bpy_stub(monkeypatch)
    operators = harness._reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import production_bake

    monkeypatch.setattr(production_bake, "bake_is_active",
                        lambda scene: scene is baking_scene)
    monkeypatch.setattr(production_bake, "any_bake_active", lambda: True)
    monkeypatch.setattr(operators._operator_runtime, "_runtime_poll", lambda cls: True)
    monkeypatch.setattr(
        operators._operator_runtime, "_bridge",
        lambda: SimpleNamespace(session=lambda: SimpleNamespace(
            active=True, owns_scene=lambda scene: scene is baking_scene)))
    return operators


def test_session_operators_refuse_in_every_scene_while_a_bake_runs(monkeypatch):
    baking = SimpleNamespace(name="A", fluidsim=SimpleNamespace())
    other = SimpleNamespace(name="B", fluidsim=SimpleNamespace(live=False))
    operators = _operators(monkeypatch, baking)
    context = SimpleNamespace(scene=other, active_object=SimpleNamespace(type="MESH"),
                              selected_objects=())

    for name in ("FLUIDSIM_OT_initialize", "FLUIDSIM_OT_step",
                 "FLUIDSIM_OT_toggle_live", "FLUIDSIM_OT_inject_particles",
                 "FLUIDSIM_OT_apply_product_preset"):
        assert getattr(operators, name).poll(context) is False, name


def test_pending_actions_refuse_while_a_bake_runs(monkeypatch):
    baking = SimpleNamespace(name="A", fluidsim=SimpleNamespace())
    _operators(monkeypatch, baking)
    from fluidSim_warp_blender51.blender import pending_settings

    for check in (pending_settings.reset_eligibility,
                  pending_settings.reinitialize_eligibility,
                  pending_settings.revert_eligibility):
        eligible, message = check(baking)
        assert eligible is False
        assert "Production Bake" in message
    try:
        pending_settings.apply_pending_reinitialize(baking)
    except RuntimeError as exc:
        assert "Production Bake" in str(exc)
    else:
        raise AssertionError("apply_pending_reinitialize ran during a bake")


def test_live_edits_are_held_not_applied_to_the_baking_solver(monkeypatch):
    baking = SimpleNamespace(name="A", fluidsim=SimpleNamespace(status=""))
    _operators(monkeypatch, baking)
    from fluidSim_warp_blender51.blender import pending_settings

    applied = []
    sess = SimpleNamespace(owns_scene=lambda scene: scene is baking,
                           sync_scene=lambda *_a: applied.append("sync"))
    monkeypatch.setattr(pending_settings, "session", lambda: sess)
    monkeypatch.setattr(pending_settings, "apply_live_settings_safely",
                        lambda *_a: applied.append("live"))
    monkeypatch.setattr(pending_settings, "_tag_view3d_redraw", lambda: None)

    pending_settings.request_scene_update(baking, pending_settings.SceneUpdate.LIVE)
    pending_settings.request_scene_update(baking, pending_settings.SceneUpdate.SYNC)

    assert applied == []
    assert "Production Bake" in baking.fluidsim.status


def test_the_bake_step_does_not_reapply_live_settings(monkeypatch):
    harness._install_bpy_stub(monkeypatch)
    scene_bridge = harness._reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    from fluidSim_warp_blender51.blender import pending_settings

    applied = []
    monkeypatch.setattr(pending_settings, "refresh_pending_state", lambda *a, **k: None)
    monkeypatch.setattr(pending_settings, "apply_live_settings_safely",
                        lambda *_a: applied.append("live") or True)
    sess = scene_bridge.Session.__new__(scene_bridge.Session)
    monkeypatch.setattr(scene_bridge.Session, "owns_scene", lambda self, scene: True)

    assert sess.prepare_step(object(), apply_live=False) is True
    assert applied == []
    sess.prepare_step(object())
    assert applied == ["live"]


def test_right_click_passes_through_and_only_esc_cancels(monkeypatch, tmp_path):
    import test_artist_bake_safety as bake_harness

    fake_bpy, cache_operators = bake_harness._install_bpy_stub(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake

    # This bake is registered process-wide; keep it out of later tests.
    monkeypatch.setattr(production_bake, "_ACTIVE_BAKES", {})
    bake_harness._ModalController.instances.clear()
    monkeypatch.setattr(production_bake, "BakeController", bake_harness._ModalController)
    monkeypatch.setattr(production_bake, "ProductionBakeBackend", bake_harness._ModalBackend)
    scene = bake_harness._Scene(404, tmp_path / "cache")
    fake_bpy.data.scenes[:] = [scene]
    context, _timers = bake_harness._modal_context(scene)
    operator = cache_operators.FLUIDSIM_OT_bake_start()
    operator.restart_existing = False
    assert operator.execute(context) == {"RUNNING_MODAL"}
    controller = bake_harness._ModalController.instances[-1]

    right = operator.modal(context, SimpleNamespace(type="RIGHTMOUSE", value="PRESS"))
    assert right == {"PASS_THROUGH"}
    assert controller.cancel_requests == 0

    operator.modal(context, SimpleNamespace(type="ESC", value="PRESS"))
    assert controller.cancel_requests == 1
    finished = operator.modal(context, SimpleNamespace(type="TIMER", value="NOTHING"))
    assert finished == {"CANCELLED"}
    assert not production_bake.any_bake_active()
