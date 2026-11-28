"""Exercise real host publication outcomes and independent channel commits."""

from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender import output_publication as output


@pytest.fixture
def live_surface(monkeypatch):
    from fluidSim_warp_blender51 import blender
    from fluidSim_warp_blender51.blender import liquid_display, rigid_publication

    monkeypatch.setattr(blender, "materials", SimpleNamespace(ensure_surface_material=None),
                        raising=False)
    scene = SimpleNamespace(frame_current=19, fluidsim=SimpleNamespace(
        domain_object=SimpleNamespace(session_uid=101, fluidsim_domain=SimpleNamespace(
            live_surface_publish_cadence=3))))
    solver = SimpleNamespace(frame=6, liquid_state_revision=30, sim_time=0.25,
                             surface=SimpleNamespace(mesh_key=("materialized",)))
    solver.host_frame = lambda: SimpleNamespace(
        should_publish_surface=lambda cadence: solver.frame % cadence == 0)
    session = SimpleNamespace(active=True, solver=solver, surface_epoch=43,
                              solved_timeline_frame=19)
    mesh = SimpleNamespace(session_uid=103, vertices=[1, 2, 3], users=1)
    obj = SimpleNamespace(session_uid=102, type="MESH", data=mesh,
                          get=lambda _key, default: default)
    verts = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    settings = SimpleNamespace(visible=True, verts=verts, tris=np.asarray([[0, 1, 2]]))

    def prepare(_scene, _host, publish_surface):
        return output.PublicationPayload(
            voxel_size=0.1, publish_surface=publish_surface,
            surface_visible=settings.visible, surface_attribute_mode="NONE",
            surface_verts=settings.verts, surface_tris=settings.tris,
            surface_attributes={}, particles_visible=False,
            particle_attribute_mode="NONE", particle_radius=0.01,
            particles=np.empty((0, 3)), particle_attributes={},
            whitewater_visible=False, whitewater_output_mode="COMBINED",
            whitewater_attribute_mode="NONE", whitewater_radius=0.01,
            whitewater=np.empty((0, 3)), whitewater_attributes={},
        )

    def swap(_obj, vertices, _triangles, *_args):
        mesh.vertices = list(vertices)

    monkeypatch.setattr(output, "_require_bpy", lambda: None)
    monkeypatch.setattr(output, "output_object", lambda _scene, kind, **_kwargs:
                        obj if kind == output.SURFACE_NAME else None)
    monkeypatch.setattr(output, "_ensure_object", lambda *_args: obj)
    monkeypatch.setattr(output, "_ensure_material", lambda *_args: None)
    monkeypatch.setattr(output, "prepare_live_payload", prepare)
    monkeypatch.setattr(output, "_swap_mesh", swap)
    monkeypatch.setattr(output, "_sync_render_inputs", lambda _scene:
                        output.RenderInputSyncResult(True))
    monkeypatch.setattr(liquid_display, "apply_viewport", lambda _scene: None)
    monkeypatch.setattr(rigid_publication, "publish_live", lambda *_args: None)
    # Supply the real refresh path's read-only presentation inputs.
    monkeypatch.setattr(liquid_display, "_viewport_geometry", lambda *_args: (True, False))
    monkeypatch.setattr(output, "_effective_voxel_size", lambda *_args: 0.1)
    scene.fluidsim.show_surface = True
    scene.fluidsim.domain_object.fluidsim_domain.surface_enabled = True
    return scene, session, obj, settings


def test_live_cadence_preserves_actual_source_and_same_frame_rebuild_advances(live_surface):
    scene, session, obj, _ = live_surface
    output.update_outputs(scene, session)
    from fluidSim_warp_blender51.blender.surface_publication import (
        SurfaceSource,
        SurfaceStatus,
        current_surface_publication,
        get_surface_state,
    )

    initial = current_surface_publication(scene)
    assert initial.source == SurfaceSource.LIVE
    assert (initial.timeline_frame, initial.solver_frame, initial.state_revision,
            initial.simulation_time, initial.epoch) == (19, 6, 30, 0.25, 43)
    assert (initial.object_token, initial.mesh_token, initial.build_key) == (
        obj.session_uid, obj.data.session_uid, ("materialized",))

    scene.frame_current = 20
    session.solver.frame = 7
    session.solver.liquid_state_revision += 1
    output.update_outputs(scene, session)
    assert get_surface_state(scene).status == SurfaceStatus.UNCHANGED
    assert current_surface_publication(scene) == initial

    output.update_outputs(scene, session, force_surface=True)
    rebuilt = current_surface_publication(scene)
    assert rebuilt.generation > initial.generation
    assert (rebuilt.timeline_frame, rebuilt.solver_frame) == (20, 7)
    output.update_outputs(scene, session, force_surface=True)
    assert current_surface_publication(scene).generation > rebuilt.generation


def test_late_rigid_failure_quarantines_through_skip_and_sync_retry(live_surface, monkeypatch):
    from fluidSim_warp_blender51.blender import rigid_publication

    scene, session, _, _ = live_surface
    output.update_outputs(scene, session)
    from fluidSim_warp_blender51.blender.surface_publication import (
        SurfaceStatus,
        current_surface_publication,
        get_surface_state,
    )

    previous = current_surface_publication(scene)
    with monkeypatch.context() as fault:
        def fail(*_args):
            raise RuntimeError("late rigid failure")
        fault.setattr(rigid_publication, "publish_live", fail)
        with pytest.raises(RuntimeError, match="late rigid failure"):
            output.update_outputs(scene, session, force_surface=True)

    damaged = get_surface_state(scene)
    assert damaged.status == SurfaceStatus.ERROR and damaged.quarantined
    assert damaged.committed == previous
    assert current_surface_publication(scene) is None

    session.solver.frame = 7
    output.update_outputs(scene, session)
    assert get_surface_state(scene).quarantined
    output.refresh_existing_output_settings(scene)
    assert current_surface_publication(scene) is None

    output.update_outputs(scene, session, force_surface=True)
    assert current_surface_publication(scene).generation > previous.generation
    assert not get_surface_state(scene).quarantined


def test_extraction_error_preserves_untouched_committed_geometry(live_surface, monkeypatch):
    scene, session, _, _ = live_surface
    output.update_outputs(scene, session)
    from fluidSim_warp_blender51.blender.surface_publication import (
        SurfaceStatus,
        current_surface_publication,
        get_surface_state,
    )

    prior = current_surface_publication(scene)
    def fail(*_args, **_kwargs):
        raise ValueError("extraction failed before publication")
    monkeypatch.setattr(output, "prepare_live_payload", fail)
    with pytest.raises(ValueError, match="before publication"):
        output.update_outputs(scene, session, force_surface=True)
    assert get_surface_state(scene).status == SurfaceStatus.ERROR
    assert not get_surface_state(scene).quarantined
    assert current_surface_publication(scene) == prior


def test_sync_retry_commits_exact_pending_generation_and_valid_empty(live_surface, monkeypatch):
    scene, session, _, settings = live_surface
    output.update_outputs(scene, session)
    from fluidSim_warp_blender51.blender.surface_publication import (
        SurfaceStatus,
        current_surface_publication,
        get_surface_state,
    )

    prior = current_surface_publication(scene)
    settings.verts = np.empty((0, 3))
    settings.tris = np.empty((0, 3))
    with monkeypatch.context() as warning:
        warning.setattr(output, "_sync_render_inputs", lambda _scene:
                        output.RenderInputSyncResult(False, exception=RuntimeError("sync refused")))
        result = output.update_outputs(scene, session, force_surface=True)
    assert not result.synchronized
    staged = get_surface_state(scene)
    assert staged.status == SurfaceStatus.PENDING
    assert staged.committed == prior
    assert staged.pending.status == SurfaceStatus.EMPTY
    assert not staged.quarantined
    assert current_surface_publication(scene) is None

    scene.frame_current = 99
    output.refresh_existing_output_settings(scene)
    assert current_surface_publication(scene) == staged.pending
    assert current_surface_publication(scene).timeline_frame == 19
    assert current_surface_publication(scene).triangle_count == 0
    assert get_surface_state(scene).status == SurfaceStatus.EMPTY


def test_epoch_cutover_rejects_previous_stamp_with_reused_frame(live_surface):
    scene, session, _, _ = live_surface
    output.update_outputs(scene, session)
    from fluidSim_warp_blender51.blender.surface_publication import (
        current_surface_publication,
        invalidate_surfaces,
    )

    prior = current_surface_publication(scene)
    invalidate_surfaces(scene)
    assert current_surface_publication(scene) is None

    session.surface_epoch += 1
    output.update_outputs(scene, session, force_surface=True)
    latest = current_surface_publication(scene)
    assert (latest.timeline_frame, latest.solver_frame) == (
        prior.timeline_frame, prior.solver_frame)
    assert latest.epoch != prior.epoch


def test_validated_cache_transition_does_not_relabel_live_commit(live_surface, monkeypatch):
    from fluidSim_warp_blender51.blender import rigid_publication
    from fluidSim_warp_blender51.io import cache

    scene, session, _, _ = live_surface
    output.update_outputs(scene, session)
    from fluidSim_warp_blender51.blender.surface_publication import (
        SurfaceSource,
        current_surface_publication,
    )

    live = current_surface_publication(scene)
    frame = cache.VerifiedCacheFrame(
        {"frame": np.asarray(4), "solver_frame": np.asarray(11),
         "simulation_time": np.asarray(0.75)}, cache_identity=("cache", "1" * 64),
    )
    monkeypatch.setattr(output, "prepare_cache_payload", lambda scene, _frame:
                        output.prepare_live_payload(scene, None, True))
    monkeypatch.setattr(rigid_publication, "publish_cache_frame", lambda *_args: None)
    output.update_outputs_from_cache_frame(scene, frame)
    stamped = current_surface_publication(scene)
    assert stamped.source == SurfaceSource.CACHE
    assert stamped.epoch != live.epoch
    assert stamped.cache_identity == frame.cache_identity
    assert (stamped.timeline_frame, stamped.source_timeline_frame, stamped.cache_frame,
            stamped.solver_frame, stamped.simulation_time) == (19, 4, 4, 11, 0.75)

    output.update_outputs_from_cache_frame(scene, dict(frame))
    assert current_surface_publication(scene) is None


def test_gpu_commit_is_independent_of_quarantined_evaluated_channel(live_surface, monkeypatch):
    scene, session, _, _ = live_surface
    output.update_outputs(scene, session)
    from fluidSim_warp_blender51.blender.surface_publication import (
        SurfaceChannel,
        SurfaceSource,
        current_surface_publication,
        surface_attempt,
    )

    evaluated = current_surface_publication(scene)
    with surface_attempt(scene, SurfaceChannel.GPU_FINAL, source=SurfaceSource.LIVE,
                         epoch=evaluated.epoch, solver_frame=evaluated.solver_frame) as attempt:
        attempt.record(vertex_count=3, triangle_count=1, visible=True)
        attempt.succeed(synchronized=True)
    gpu = current_surface_publication(scene, SurfaceChannel.GPU_FINAL)
    assert gpu.channel == SurfaceChannel.GPU_FINAL and gpu.generation > 0

    def fail(*_args):
        raise RuntimeError("host mesh changed then failed")
    monkeypatch.setattr(output, "_swap_mesh", fail)
    with pytest.raises(RuntimeError, match="host mesh changed"):
        output.update_outputs(scene, session, force_surface=True)
    assert current_surface_publication(scene) is None
    assert current_surface_publication(scene, SurfaceChannel.GPU_FINAL) == gpu



def test_sync_cannot_commit_a_candidate_after_the_output_mesh_was_replaced(live_surface, monkeypatch):
    scene, session, obj, _ = live_surface
    from fluidSim_warp_blender51.blender.surface_publication import (
        current_surface_publication,
        get_surface_state,
    )

    with monkeypatch.context() as warning:
        warning.setattr(output, "_sync_render_inputs", lambda _scene:
                        output.RenderInputSyncResult(False, exception=RuntimeError("sync refused")))
        output.update_outputs(scene, session, force_surface=True)
    pending = get_surface_state(scene).pending
    assert pending is not None

    obj.data = SimpleNamespace(session_uid=104, vertices=[1, 2, 3])
    session.solver.frame = 7
    output.update_outputs(scene, session)  # sync during an ordinary cadence skip
    assert current_surface_publication(scene) is None


def test_publication_generation_is_never_reused_after_scene_invalidation(live_surface):
    scene, session, _, _ = live_surface
    from fluidSim_warp_blender51.blender.surface_publication import (
        current_surface_publication,
        invalidate_surfaces,
    )

    output.update_outputs(scene, session)
    old = current_surface_publication(scene)
    invalidate_surfaces(scene)
    output.update_outputs(scene, session)
    after = current_surface_publication(scene)
    assert after.epoch == old.epoch and after.timeline_frame == old.timeline_frame
    assert after.generation != old.generation


def test_scene_subscriber_observes_completed_identity_and_is_weak(live_surface):
    from gc import collect
    from weakref import ref

    from fluidSim_warp_blender51.blender.surface_publication import subscribe_surfaces

    scene, session, _, _ = live_surface
    seen = []

    class Observer:
        def accept(self, channel, state):
            seen.append((channel, state.status, state.eligible.generation))

    observer = Observer()
    observe = ref(observer)
    unsubscribe = subscribe_surfaces(scene, observer.accept)
    output.update_outputs(scene, session)
    assert seen == [("EVALUATED_MESH", "READY", seen[0][2])]
    del observer
    collect()
    assert observe() is None
    output.update_outputs(scene, session)
    assert len(seen) == 1
    unsubscribe()



def test_force_publication_keeps_solved_timeline_after_bookkeeping_restore(live_surface):
    from fluidSim_warp_blender51.blender.surface_publication import (
        current_surface_publication,
    )

    scene, session, _, _ = live_surface
    output.update_outputs(scene, session)
    scene.frame_current = 7
    session.last_frame = 7
    output.update_outputs(scene, session, force_surface=True)
    stamp = current_surface_publication(scene)
    assert stamp.timeline_frame == 7
    assert getattr(stamp, "source_timeline_frame", None) == 19
