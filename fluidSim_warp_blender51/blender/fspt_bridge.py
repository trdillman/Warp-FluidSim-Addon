"""Read-only eligibility boundary between published surfaces and FSPT.

Blender callbacks own capture. A descriptor grants no ownership of a mutable
mesh; consumers copy it and revalidate before accepting or presenting work.
"""
import os
from dataclasses import asdict

from . import surface_publication as publication
from .scene_access import active_domain_object


class SurfaceNotReadyError(RuntimeError):
    pass


def _original(scene):
    return getattr(scene, "original", scene)


def _surface_object(scene):
    if not hasattr(scene, "fluidsim"):
        return None
    from . import output_publication

    return output_publication.output_object(scene, output_publication.SURFACE_NAME, ensure=False)


def _session():
    from .scene_bridge import session

    return session()


def _production_bake_active():
    from .production_bake import any_bake_active

    return any_bake_active()


def _belongs(scene, obj):
    if obj is None:
        return True
    objects = getattr(scene, "objects", ())
    getter = getattr(objects, "get", None)
    if callable(getter):
        return publication.id_token(getter(obj.name)) == publication.id_token(obj)
    return any(publication.id_token(item) == publication.id_token(obj) for item in objects)


def _visible(obj, for_viewport, view_layer=None):
    if obj is None:
        return False
    if view_layer is not None and hasattr(view_layer, "objects"):
        if publication.id_token(view_layer.objects.get(obj.name)) != publication.id_token(obj):
            return False
    if for_viewport:
        return bool(obj.visible_get(**({"view_layer": view_layer} if view_layer is not None else {})))
    return not bool(obj.hide_render)


def _cache_root(scene):
    path = str(getattr(scene.fluidsim, "cache_path", ""))
    if path.startswith("//"):
        import bpy

        path = bpy.path.abspath(path)
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def capture_publications(scene, *, for_viewport, exact_frame=False, view_layer=None):
    """Return immutable eligible facts without extracting or creating output IDs."""
    scene = _original(scene)
    obj = _surface_object(scene)
    state = publication.get_surface_state(scene)
    stamp = state.eligible
    if obj is not None and not _visible(obj, for_viewport, view_layer):
        return ()
    if obj is None and stamp is None:
        fs = getattr(scene, "fluidsim", None)
        if active_domain_object(scene) is not None and getattr(fs, "show_surface", False):
            raise SurfaceNotReadyError("Waiting for a committed fluid surface publication")
        return ()
    if stamp is None:
        raise SurfaceNotReadyError(state.reason or "Fluid surface has no eligible publication")
    if stamp.scene_token != publication.id_token(scene):
        raise SurfaceNotReadyError("Fluid surface belongs to a different scene")
    if not _belongs(scene, obj) or not _belongs(scene, active_domain_object(scene)):
        raise SurfaceNotReadyError("Fluid surface owner is not in this scene")
    if (stamp.object_token != publication.id_token(obj)
            or stamp.mesh_token != publication.id_token(getattr(obj, "data", None))):
        raise SurfaceNotReadyError("Fluid output object or mesh was replaced")
    fs = getattr(scene, "fluidsim", None)
    expected = (publication.SurfaceSource.CACHE if getattr(fs, "cache_playback_enabled", False)
                else publication.SurfaceSource.LIVE)
    if stamp.source != expected:
        raise SurfaceNotReadyError(f"Fluid source changed to {expected}; waiting for its publication")
    if expected == publication.SurfaceSource.LIVE:
        sess = _session()
        if not sess.owns_scene(scene) or getattr(sess, "surface_epoch", None) != stamp.epoch:
            raise SurfaceNotReadyError("Fluid surface belongs to a retired solver epoch")
        if exact_frame:
            solver = sess.solver
            if (stamp.solver_frame != getattr(solver, "frame", None)
                    or stamp.state_revision != getattr(solver, "liquid_state_revision", None)
                    or stamp.simulation_time != getattr(solver, "sim_time", None)):
                raise SurfaceNotReadyError("Final render requires the current completed solver state revision")
            if stamp.build_key != getattr(getattr(solver, "surface", None), "mesh_key", None):
                raise SurfaceNotReadyError("Final render requires the current completed surface build")
    elif stamp.cache_identity is None or stamp.cache_identity[0] != _cache_root(scene):
        raise SurfaceNotReadyError("Selected cache sequence differs from the validated surface publication")
    if exact_frame and getattr(stamp, "source_timeline_frame", None) != int(scene.frame_current):
        raise SurfaceNotReadyError("Final render requires the exact solved or validated cache frame")
    return (stamp,)


def validate_publications(scene, captured, *, for_viewport, exact_frame=False, view_layer=None):
    current = capture_publications(scene, for_viewport=for_viewport, exact_frame=exact_frame,
                                   view_layer=view_layer)
    if tuple(captured) != current:
        raise SurfaceNotReadyError("Fluid publication generation changed during capture or rendering")


def prepare_final_publication(depsgraph):
    """Force only an already solved requested state through its existing owner."""
    scene = _original(depsgraph.scene)
    layer = depsgraph.view_layer
    obj = _surface_object(scene)
    if obj is None and not getattr(getattr(scene, "fluidsim", None), "show_surface", False):
        return ()
    if obj is None and active_domain_object(scene) is None:
        return ()
    if obj is not None and not _visible(obj, False, layer):
        return ()
    if _production_bake_active():
        raise SurfaceNotReadyError("Production Bake owns fluid state; final surface preparation is unavailable")
    if getattr(getattr(scene, "fluidsim", None), "cache_playback_enabled", False):
        if obj is not None and not _visible(obj, False, layer):
            return ()
        from . import playback

        # Revalidate once at final-job admission, including an in-place rebake.
        # The immutable captured version is then pinned for this render job.
        playback.play_cache_frame(_session(), scene, frame=int(scene.frame_current))
        depsgraph.update()
        return capture_publications(scene, for_viewport=False, exact_frame=True, view_layer=layer)
    obj = _surface_object(scene)
    if obj is None and active_domain_object(scene) is None:
        return ()
    if obj is not None and not _visible(obj, False, layer):
        return ()
    sess = _session()
    if (not sess.owns_scene(scene)
            or getattr(sess, "solved_timeline_frame", None) != int(scene.frame_current)):
        raise SurfaceNotReadyError("Final render requires an already solved exact timeline state")
    from . import output_publication

    # A final job pays one owner-controlled refresh so paused surface recipes,
    # waves and required render inputs cannot hide behind an old equal key.
    output_publication.update_outputs(scene, sess, force_surface=True)
    depsgraph.update()
    return capture_publications(scene, for_viewport=False, exact_frame=True, view_layer=layer)


def frame_info(frame, width, height):
    """Describe the submitted job; never read mutable scene/timeline labels."""
    return {
        "status": "COMPLETE", "key": list(frame.key), "samples": frame.samples,
        "device": frame.device, "backend": frame.backend, "transport": frame.transport,
        "width": width, "height": height, "scene_frame": frame.scene_frame,
        "sources": [asdict(stamp) for stamp in frame.publications],
        "geometry_digest": frame.geometry_digest,
        "statistics": dict(frame.statistics),
    }
