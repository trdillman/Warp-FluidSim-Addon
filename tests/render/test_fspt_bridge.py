"""Render consumers cannot turn mutable or retired output into a live image."""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import surface_publication as publication


def bridge():
    name = "fluidSim_warp_blender51.blender.fspt_bridge"
    assert importlib.util.find_spec(name), "FSPT publication consumer is missing"
    return importlib.import_module(name)


@pytest.fixture
def accepted(monkeypatch):
    domain = SimpleNamespace(session_uid=201, name="Domain")
    mesh = SimpleNamespace(session_uid=203)
    surface = SimpleNamespace(session_uid=202, name="Surface", data=mesh,
                              hide_render=False, visible_get=lambda **_kw: True)
    scene = SimpleNamespace(session_uid=200, frame_current=7,
                            objects={"Domain": domain, "Surface": surface},
                            fluidsim=SimpleNamespace(domain_object=domain,
                                output_collection_name="Output", cache_playback_enabled=False))
    with publication.surface_attempt(scene, source=publication.SurfaceSource.LIVE,
                                     epoch=25, timeline_frame=7, solver_frame=6,
                                     state_revision=60, simulation_time=0.25,
                                     source_timeline_frame=7) as attempt:
        attempt.mark_mutation()
        attempt.record(vertex_count=3, triangle_count=1, visible=True, obj=surface)
        attempt.succeed(synchronized=True)
    b = bridge()
    monkeypatch.setattr(b, "_production_bake_active", lambda: False, raising=False)
    monkeypatch.setattr(b, "_surface_object", lambda _scene: surface)
    sess = SimpleNamespace(surface_epoch=25, owns_scene=lambda owner: owner is scene,
                           solved_timeline_frame=7, solver=SimpleNamespace(
                               frame=6, liquid_state_revision=60, sim_time=0.25,
                               surface=SimpleNamespace(mesh_key=None)))
    monkeypatch.setattr(b, "_session", lambda: sess)
    yield b, scene, surface
    publication.invalidate_surfaces(scene)


def test_static_scene_does_not_require_a_solver(monkeypatch):
    b = bridge()
    monkeypatch.setattr(b, "_surface_object", lambda _scene: None)
    assert b.capture_publications(SimpleNamespace(session_uid=400), for_viewport=True) == ()


def _advance_live(scene, surface, *, solver_frame=7, state_revision=61,
                  simulation_time=0.30, synchronized=True, triangle_count=1):
    with publication.surface_attempt(scene, source=publication.SurfaceSource.LIVE,
                                     epoch=25, timeline_frame=8, solver_frame=solver_frame,
                                     state_revision=state_revision, simulation_time=simulation_time,
                                     source_timeline_frame=8) as attempt:
        attempt.mark_mutation()
        attempt.record(vertex_count=3, triangle_count=triangle_count, visible=True, obj=surface)
        attempt.succeed(synchronized=synchronized)


def test_owned_viewport_image_can_follow_a_later_live_publication_for_at_most_250_ms(accepted):
    b, scene, surface = accepted
    old = b.capture_publications(scene, for_viewport=True)
    _advance_live(scene, surface)
    result = b.validate_viewport_publications(scene, old, clock=lambda: old[0].published_at + 0.2)
    assert result["status"] == "RETAINED" and result["publication_age_seconds"] == pytest.approx(0.2)
    assert result["retention_limit_seconds"] == 0.25
    assert "not authoritative solver-ready" in result["age_clock"]
    with pytest.raises(b.SurfaceNotReadyError, match="age"):
        b.validate_viewport_publications(scene, old, clock=lambda: old[0].published_at + 0.2501)
    with pytest.raises(b.SurfaceNotReadyError):
        b.validate_publications(scene, old, for_viewport=True)


def test_paused_current_image_remains_valid_without_being_claimed_recent(accepted):
    b, scene, _ = accepted
    current = b.capture_publications(scene, for_viewport=True)
    result = b.validate_viewport_publications(scene, current, clock=lambda: current[0].published_at + 600)
    assert result["status"] == "CURRENT" and result["publication_age_seconds"] == pytest.approx(600)


@pytest.mark.parametrize("field,value", [("solver_frame", 6), ("state_revision", 60),
                                         ("simulation_time", 0.25), ("triangle_count", 0)])
def test_owned_history_rejects_same_state_rebuilds_and_cleared_geometry(accepted, field, value):
    b, scene, surface = accepted
    old = b.capture_publications(scene, for_viewport=True)
    _advance_live(scene, surface, **{field: value})
    with pytest.raises(b.SurfaceNotReadyError, match="compatible|advancing"):
        b.validate_viewport_publications(scene, old, clock=lambda: old[0].published_at + 0.1)


def test_owned_history_refuses_pending_sync_and_mutation_quarantine(accepted):
    b, scene, surface = accepted
    old = b.capture_publications(scene, for_viewport=True)
    _advance_live(scene, surface, synchronized=False)
    with pytest.raises(b.SurfaceNotReadyError):
        b.validate_viewport_publications(scene, old, clock=lambda: old[0].published_at + 0.1)
    with publication.surface_attempt(scene, source=publication.SurfaceSource.LIVE,
                                     epoch=25, timeline_frame=9) as attempt:
        attempt.mark_mutation()
        attempt.fail(RuntimeError("partial write"))
    with pytest.raises(b.SurfaceNotReadyError):
        b.validate_viewport_publications(scene, old, clock=lambda: old[0].published_at + 0.1)


def test_capture_preserves_published_state_when_timeline_moves(accepted):
    b, scene, _ = accepted
    original = b.capture_publications(scene, for_viewport=True)
    scene.frame_current = 99
    assert b.capture_publications(scene, for_viewport=True) == original
    assert original[0].timeline_frame == 7
    assert original[0].solver_frame == 6


def test_unknown_or_retired_surface_is_not_a_static_mesh(accepted):
    b, scene, _ = accepted
    publication.invalidate_surfaces(scene)
    with pytest.raises(b.SurfaceNotReadyError, match="publication|surface"):
        b.capture_publications(scene, for_viewport=True)


def test_partial_mutation_quarantine_blocks_capture_and_old_presentation(accepted):
    b, scene, _ = accepted
    old = b.capture_publications(scene, for_viewport=True)
    with (
        pytest.raises(RuntimeError, match="mutation"),
        publication.surface_attempt(scene, source=publication.SurfaceSource.LIVE,
                                    epoch=25, timeline_frame=8) as attempt,
    ):
        attempt.mark_mutation()
        raise RuntimeError("mutation failed")
    with pytest.raises(b.SurfaceNotReadyError):
        b.validate_publications(scene, old, for_viewport=True)


def test_owner_mesh_and_scene_membership_are_required(accepted):
    b, scene, surface = accepted
    surface.data = SimpleNamespace(session_uid=999)
    with pytest.raises(b.SurfaceNotReadyError, match="mesh|output"):
        b.capture_publications(scene, for_viewport=True)
    surface.data = SimpleNamespace(session_uid=203)
    scene.objects.pop("Surface")
    with pytest.raises(b.SurfaceNotReadyError, match="scene|owner"):
        b.capture_publications(scene, for_viewport=True)


def test_source_mode_change_rejects_old_pixels_even_at_same_frame(accepted):
    b, scene, _ = accepted
    captured = b.capture_publications(scene, for_viewport=True)
    scene.fluidsim.cache_playback_enabled = True
    with pytest.raises(b.SurfaceNotReadyError, match="source|CACHE"):
        b.validate_publications(scene, captured, for_viewport=True)


def test_replacement_between_capture_boundaries_is_rejected(accepted):
    b, scene, surface = accepted
    captured = b.capture_publications(scene, for_viewport=True)
    with publication.surface_attempt(scene, source=publication.SurfaceSource.LIVE,
                                     epoch=25, timeline_frame=7, solver_frame=6) as attempt:
        attempt.mark_mutation()
        attempt.record(vertex_count=3, triangle_count=1, visible=True, obj=surface)
        attempt.succeed(synchronized=True)
    with pytest.raises(b.SurfaceNotReadyError, match="changed|generation"):
        b.validate_publications(scene, captured, for_viewport=True)


def test_final_render_refuses_a_different_actual_state(accepted):
    b, scene, _ = accepted
    scene.frame_current = 8
    with pytest.raises(b.SurfaceNotReadyError, match="frame|state"):
        b.capture_publications(scene, for_viewport=False, exact_frame=True)


def test_frame_metadata_uses_submitted_tuple_not_scene_mutation(accepted):
    b, scene, _ = accepted
    from fluidSim_warp_blender51.fspt.contracts import RenderFrame
    frame = RenderFrame(None, ("radiance",), 2, "cuda:0", "test", "test",
                        scene_frame=7, publications=b.capture_publications(scene, for_viewport=True))
    scene.frame_current = 100
    result = b.frame_info(frame, 32, 16)
    assert result["scene_frame"] == 7
    assert result["sources"][0]["solver_frame"] == 6
    assert result["sources"][0]["generation"] == frame.publications[0].generation
    assert result["sources"][0]["source"] == "LIVE"


def _empty_publication(scene, surface):
    with publication.surface_attempt(scene, source=publication.SurfaceSource.LIVE,
                                     epoch=25, timeline_frame=7, solver_frame=6) as attempt:
        attempt.mark_mutation()
        attempt.record(vertex_count=0, triangle_count=0, visible=True, obj=surface)
        attempt.succeed(synchronized=True)


def test_evaluated_capture_carries_empty_surface_provenance(accepted):
    from fluidSim_warp_blender51.blender.fspt_scene import extract_scene
    b, scene, surface = accepted
    scene.world = None
    _empty_publication(scene, surface)
    graph = SimpleNamespace(scene=scene, view_layer=SimpleNamespace(material_override=None),
                            object_instances=[])
    result = extract_scene(graph, for_viewport=True)
    assert result.publications == b.capture_publications(scene, for_viewport=True)


def test_evaluated_capture_rejects_publication_changed_during_extraction(accepted):
    from fluidSim_warp_blender51.blender.fspt_scene import extract_scene
    b, scene, surface = accepted
    scene.world = None
    _empty_publication(scene, surface)

    class ReplacingInstances:
        def __iter__(self):
            _empty_publication(scene, surface)
            return iter(())

    graph = SimpleNamespace(scene=scene, view_layer=SimpleNamespace(material_override=None),
                            object_instances=ReplacingInstances())
    with pytest.raises(b.SurfaceNotReadyError, match="changed|generation"):
        extract_scene(graph, for_viewport=True)


def test_final_rejects_same_timeline_with_newer_solver_revision(accepted):
    b, scene, _ = accepted
    assert b.capture_publications(scene, for_viewport=False, exact_frame=True)
    b._session().solver.liquid_state_revision = 61
    with pytest.raises(b.SurfaceNotReadyError, match="state|revision"):
        b.capture_publications(scene, for_viewport=False, exact_frame=True)


def test_final_rejects_same_state_with_new_surface_build(accepted):
    b, scene, _ = accepted
    b._session().solver.surface.mesh_key = ("rebuilt", 2)
    with pytest.raises(b.SurfaceNotReadyError, match="surface|recipe|build"):
        b.capture_publications(scene, for_viewport=False, exact_frame=True)


def test_selected_cache_root_must_match_validated_publication(accepted, tmp_path):
    b, scene, surface = accepted
    root_a = str(tmp_path / "A")
    scene.fluidsim.cache_path = str(tmp_path / "B")
    scene.fluidsim.cache_playback_enabled = True
    with publication.surface_attempt(scene, source=publication.SurfaceSource.CACHE,
                                     timeline_frame=7, cache_frame=7,
                                     source_timeline_frame=7,
                                     cache_identity=(root_a, "1" * 64)) as attempt:
        attempt.mark_mutation()
        attempt.record(vertex_count=3, triangle_count=1, visible=True, obj=surface)
        attempt.succeed(synchronized=True)
    with pytest.raises(b.SurfaceNotReadyError, match="cache|sequence"):
        b.capture_publications(scene, for_viewport=False, exact_frame=True)


def test_nonempty_fluid_missing_from_evaluation_is_not_certified(accepted):
    from fluidSim_warp_blender51.blender.fspt_scene import extract_scene
    b, scene, _ = accepted
    scene.world = None
    graph = SimpleNamespace(scene=scene, view_layer=SimpleNamespace(material_override=None),
                            object_instances=[])
    with pytest.raises(b.SurfaceNotReadyError, match="evaluated|captured"):
        extract_scene(graph, for_viewport=True)


def test_target_view_layer_exclusion_omits_fluid_provenance(accepted):
    from fluidSim_warp_blender51.blender.fspt_scene import extract_scene
    _, scene, _ = accepted
    scene.world = None
    layer = SimpleNamespace(material_override=None, objects={})
    graph = SimpleNamespace(scene=scene, view_layer=layer, object_instances=[])
    assert extract_scene(graph, for_viewport=True).publications == ()


def test_final_admission_refreshes_already_solved_surface_without_stepping(accepted, monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication
    b, scene, surface = accepted
    before = publication.current_surface_publication(scene)
    solver = b._session().solver
    saved_state = (solver.frame, solver.liquid_state_revision, solver.sim_time)

    def publish(owner, session, *, force_surface):
        assert owner is scene and session.solver is solver and force_surface
        with publication.surface_attempt(scene, source=publication.SurfaceSource.LIVE,
                                         epoch=25, timeline_frame=7, solver_frame=6,
                                         source_timeline_frame=7, state_revision=60,
                                         simulation_time=0.25) as attempt:
            attempt.mark_mutation()
            attempt.record(vertex_count=3, triangle_count=1, visible=True, obj=surface)
            attempt.succeed(synchronized=True)

    monkeypatch.setattr(output_publication, "update_outputs", publish)
    graph = SimpleNamespace(scene=scene, view_layer=SimpleNamespace(), update=lambda: None)
    actual = b.prepare_final_publication(graph)
    assert actual[0].generation > before.generation
    assert (solver.frame, solver.liquid_state_revision, solver.sim_time) == saved_state


def test_static_final_with_disabled_domain_never_requires_solver(monkeypatch):
    b = bridge()
    scene = SimpleNamespace(session_uid=801, frame_current=7,
                            fluidsim=SimpleNamespace(domain_object=object(), show_surface=False))
    monkeypatch.setattr(b, "_surface_object", lambda _scene: None)
    def unexpected_solver():
        raise AssertionError("disabled fluid must not request a solver")
    monkeypatch.setattr(b, "_session", unexpected_solver)
    graph = SimpleNamespace(scene=scene, view_layer=SimpleNamespace())
    assert b.prepare_final_publication(graph) == ()


def test_final_fluid_preparation_refuses_while_bake_owns_state(accepted, monkeypatch):
    b, scene, _ = accepted
    monkeypatch.setattr(b, "_production_bake_active", lambda: True, raising=False)
    before = publication.current_surface_publication(scene)
    graph = SimpleNamespace(scene=scene, view_layer=SimpleNamespace(), update=lambda: None)
    with pytest.raises(b.SurfaceNotReadyError, match="Bake|bake"):
        b.prepare_final_publication(graph)
    assert publication.current_surface_publication(scene) == before


@pytest.mark.parametrize("for_viewport", [True, False])
def test_designated_domain_with_none_role_is_not_render_geometry(monkeypatch, for_viewport):
    """Domain ownership is a Scene pointer, not an Object role enum value."""
    import numpy as np

    from fluidSim_warp_blender51.blender.fspt_scene import extract_scene

    class Rows:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __len__(self):
            return len(self.values)

        def foreach_get(self, name, target):
            values = np.zeros(len(self.values), dtype=np.int32) if name == "material_index" else self.values
            target[:] = values.ravel()

    class MeshObject(SimpleNamespace):
        def get(self, key):
            return None

    mesh = SimpleNamespace(vertices=Rows([[2, 0, 0], [3, 0, 0], [2, 1, 0]]),
                           loop_triangles=Rows([[0, 1, 2]]), materials=[],
                           calc_loop_triangles=lambda: None)
    geometry = SimpleNamespace(pointcloud=None, curves=None, volume=None,
                               grease_pencil=None, instance_references=lambda: ())
    extracted = []

    def obj(name, uid):
        item = MeshObject(name=name, session_uid=uid, type="MESH", hide_render=False,
                          fluidsim=SimpleNamespace(role="NONE"), material_slots=[],
                          visible_get=lambda **kw: True, evaluated_geometry=lambda: geometry,
                          to_mesh_clear=lambda: None)
        item.original = item
        item.to_mesh = lambda **kw: (extracted.append(name), mesh)[1]
        return item

    domain, ordinary = obj("Designated Domain", 1301), obj("Ordinary mesh", 1302)
    scene = SimpleNamespace(session_uid=1300, frame_current=1, world=None,
                            fluidsim=SimpleNamespace(domain_object=domain, show_surface=False))
    monkeypatch.setattr(bridge(), "_surface_object", lambda _scene: None)
    graph = SimpleNamespace(scene=scene, view_layer=SimpleNamespace(material_override=None),
        object_instances=[SimpleNamespace(object=item, is_instance=False, show_self=True,
                                          matrix_world=np.eye(4)) for item in (domain, ordinary)])
    snapshot = extract_scene(graph, for_viewport=for_viewport)
    assert extracted == ["Ordinary mesh"]
    np.testing.assert_array_equal(snapshot.vertices, mesh.vertices.values)
    assert len(snapshot.triangles) == 1
