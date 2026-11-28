"""Completed solver clocks precede optional work and cannot follow new identities."""

from dataclasses import FrozenInstanceError
import sys
from types import SimpleNamespace

import pytest

from test_surface_publication import live_surface  # noqa: F401 - shared publication boundary


@pytest.fixture
def completed_session(monkeypatch, live_surface):  # noqa: F811 - imported pytest fixture
    from test_scene_bridge_settings import _install_scene_bridge_stubs
    from test_surface_publication import output as fixture_publication
    from fluidSim_warp_blender51 import blender

    # Sibling runtime tests reload publication to isolate their Blender doubles.
    # Bind Session and per-test imports to the exact module patched by this
    # shared live_surface fixture, restoring both import paths after the test.
    monkeypatch.setitem(sys.modules, fixture_publication.__name__, fixture_publication)
    monkeypatch.setattr(blender, "output_publication", fixture_publication)
    bridge = _install_scene_bridge_stubs(monkeypatch)
    scene, old, obj, settings = live_surface
    session = bridge.Session()
    session.solver = old.solver
    session.surface_epoch = old.surface_epoch
    session.solved_timeline_frame = 18
    session.bind_scene(bridge.scene_identity(scene), "clock-fixture")
    session.solver.frame = 5
    session.solver.liquid_state_revision = 29
    session.solver.sim_time = 5 / 24
    now = [10.0]
    monkeypatch.setattr(bridge, "monotonic", lambda: now[0], raising=False)
    stats = SimpleNamespace(
        active_particles=3, active_voxels=4, whitewater_particles=0,
        step_time=0.01, substeps=1, cfl_substeps_clamped=False, as_dict=lambda: {})

    def step():
        session.solver.frame += 1
        session.solver.liquid_state_revision += 1
        session.solver.sim_time = session.solver.frame / 24
        return stats

    session.solver.step_frame = step
    return bridge, scene, session, obj, settings, now


def completion(session, scene):
    query = getattr(session, "current_solver_completion", None)
    return query(scene) if callable(query) else None


def test_clock_precedes_delayed_optional_publication(completed_session, monkeypatch):
    from fluidSim_warp_blender51.blender import gpu_particle_owner, output_publication
    from fluidSim_warp_blender51.blender.surface_publication import current_surface_publication

    _, scene, session, _, _, now = completed_session
    assert completion(session, scene) is None
    original = output_publication.prepare_live_payload

    def delayed(*args, **kwargs):
        now[0] = 90.0
        return original(*args, **kwargs)

    monkeypatch.setattr(output_publication, "prepare_live_payload", delayed)
    monkeypatch.setattr(gpu_particle_owner, "publish_scene", lambda *_args: None)
    session.step(scene, write_cache=False)
    clock = completion(session, scene)
    assert clock is not None, "The completed solver state has no authoritative clock"
    assert clock.ready_at == 10.0
    assert (clock.solver_frame, clock.state_revision, clock.simulation_time,
            clock.source_timeline_frame, clock.epoch) == (6, 30, 0.25, 19, 43)
    assert current_surface_publication(scene).state_ready_at == 10.0
    with pytest.raises(FrozenInstanceError):
        clock.ready_at = 90.0


def test_no_output_step_still_records_clock_and_same_frame_solve_replaces_it(completed_session):
    _, scene, session, _, _, now = completed_session
    session.step(scene, write_cache=False, publish_outputs=False)
    first = completion(session, scene)
    assert first is not None
    now[0] = 12.0
    session.step(scene, write_cache=False, publish_outputs=False)
    second = completion(session, scene)
    assert second is not None and second != first
    assert (first.ready_at, first.solver_frame, first.source_timeline_frame) == (10.0, 6, 19)
    assert (second.ready_at, second.solver_frame, second.source_timeline_frame) == (12.0, 7, 19)


@pytest.mark.parametrize("field,value", [
    ("frame", 7), ("liquid_state_revision", 31), ("sim_time", 0.5),
])
def test_changed_solver_facts_cannot_reuse_clock(completed_session, field, value):
    _, scene, session, _, _, _ = completed_session
    session.step(scene, write_cache=False, publish_outputs=False)
    assert completion(session, scene) is not None
    setattr(session.solver, field, value)
    assert completion(session, scene) is None


@pytest.mark.parametrize("change", ["solver", "epoch", "timeline", "scene", "domain"])
def test_owner_cutovers_cannot_reuse_clock(completed_session, change):
    _, scene, session, _, _, _ = completed_session
    session.step(scene, write_cache=False, publish_outputs=False)
    assert completion(session, scene) is not None
    if change == "solver":
        session.solver = SimpleNamespace(**vars(session.solver))
    elif change == "epoch":
        session.surface_epoch += 1
    elif change == "timeline":
        session.solved_timeline_frame = 20
    elif change == "scene":
        scene = SimpleNamespace(frame_current=19, fluidsim=scene.fluidsim)
    else:
        scene.fluidsim.domain_object.session_uid += 1
    assert completion(session, scene) is None


def test_failed_solve_invalidates_clock_even_if_solver_numbers_unchanged(completed_session):
    _, scene, session, _, _, _ = completed_session
    session.step(scene, write_cache=False, publish_outputs=False)
    assert completion(session, scene) is not None

    def fail():
        raise RuntimeError("injected solve failure")

    session.solver.step_frame = fail
    with pytest.raises(RuntimeError, match="injected solve failure"):
        session.step(scene, write_cache=False, publish_outputs=False)
    assert completion(session, scene) is None


def test_optional_output_failure_keeps_true_completion_fact(completed_session, monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.blender.surface_publication import current_surface_publication

    _, scene, session, _, _, now = completed_session

    def fail(*_args, **_kwargs):
        now[0] = 500.0
        raise RuntimeError("publication failed after completion")

    monkeypatch.setattr(output_publication, "prepare_live_payload", fail)
    with pytest.raises(RuntimeError, match="publication failed"):
        session.step(scene, write_cache=False)
    assert completion(session, scene).ready_at == 10.0
    assert current_surface_publication(scene) is None


def test_clock_query_is_read_only_and_timeline_cursor_does_not_relabel_it(completed_session):
    _, scene, session, _, _, now = completed_session
    session.step(scene, write_cache=False, publish_outputs=False)
    expected = completion(session, scene)
    assert expected is not None

    def forbidden():
        raise AssertionError("Clock query read or advanced solver")

    session.solver.step_frame = session.solver.host_frame = forbidden
    scene.frame_current = 999
    now[0] = 999.0
    assert completion(session, scene) is expected
    assert expected.source_timeline_frame == 19 and expected.ready_at == 10.0


def test_publication_rebuild_cadence_and_sync_retry_preserve_source_clock(completed_session, monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication as output
    from fluidSim_warp_blender51.blender.surface_publication import (
        current_surface_publication, get_surface_state,
    )

    _, scene, session, _, _, now = completed_session
    session.step(scene, write_cache=False, publish_outputs=False)
    output.update_outputs(scene, session)
    initial = current_surface_publication(scene)
    assert getattr(initial, "state_ready_at", None) == 10.0
    now[0] = 12.0
    session.step(scene, write_cache=False, publish_outputs=False)
    output.update_outputs(scene, session)  # Frame 7 skips cadence 3.
    assert current_surface_publication(scene) == initial
    with monkeypatch.context() as pending:
        pending.setattr(output, "_sync_render_inputs", lambda _scene:
                        output.RenderInputSyncResult(False, exception=RuntimeError("sync delay")))
        output.update_outputs(scene, session, force_surface=True)
    staged = get_surface_state(scene).pending
    assert staged.state_ready_at == 12.0
    now[0] = 500.0
    output.refresh_existing_output_settings(scene)
    assert current_surface_publication(scene) == staged
    output.update_outputs(scene, session, force_surface=True)
    rebuilt = current_surface_publication(scene)
    assert rebuilt.generation > staged.generation and rebuilt.state_ready_at == 12.0


def test_unowned_direct_step_and_cache_publication_have_no_solver_clock(completed_session, monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication as output, rigid_publication
    from fluidSim_warp_blender51.blender.surface_publication import current_surface_publication
    from fluidSim_warp_blender51.io.cache import VerifiedCacheFrame
    import numpy as np

    _, scene, session, _, _, _ = completed_session
    session.solver.step_frame()  # No authoritative Session completion boundary.
    output.update_outputs(scene, session)
    assert getattr(current_surface_publication(scene), "state_ready_at", "missing") is None
    frame = VerifiedCacheFrame(
        {"frame": np.asarray(4), "solver_frame": np.asarray(11),
         "simulation_time": np.asarray(0.75)}, cache_identity=("cache", "1" * 64))
    monkeypatch.setattr(output, "prepare_cache_payload", lambda scene, _frame:
                        output.prepare_live_payload(scene, None, True))
    monkeypatch.setattr(rigid_publication, "publish_cache_frame", lambda *_args: None)
    output.update_outputs_from_cache_frame(scene, frame)
    assert current_surface_publication(scene).state_ready_at is None


def test_replacement_and_release_clear_completed_receipt(completed_session, monkeypatch):
    from fluidSim_warp_blender51.sim import memory_safety

    _, scene, session, _, _, _ = completed_session
    session.step(scene, write_cache=False, publish_outputs=False)
    assert completion(session, scene) is not None
    old_solver = session.solver
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda *_args: None)
    session.install_solver(
        SimpleNamespace(**vars(old_solver)), profiler=None, device="cpu",
        scene_inputs=session.scene_inputs, solved_timeline_frame=19)
    assert completion(session, scene) is None
    # Restoring the old object cannot restore the superseded receipt.
    session.solver = old_solver
    session.surface_epoch = 43
    assert completion(session, scene) is None
    session.step(scene, write_cache=False, publish_outputs=False)
    assert completion(session, scene) is not None
    session._clear_runtime()
    assert completion(session, scene) is None
