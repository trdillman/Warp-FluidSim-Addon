"""Render output integration.

Simulation outputs are real datablocks, so they render directly. This module
keeps render visibility coherent and steps the solver during animation
renders (render frame-change fires the same handler as playback when the
simulation is live).
"""

from __future__ import annotations

import bpy

from . import viewport

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)


def ensure_render_visibility(scene):
    """Make simulation outputs render-visible and source/collider helpers not."""
    fs = scene.fluidsim
    for name, show in (
        (viewport.SURFACE_NAME, fs.show_surface),
        (viewport.PARTICLES_NAME, fs.show_particles),
        (viewport.WHITEWATER_NAME, fs.show_whitewater),
    ):
        obj = viewport.output_object(scene, name)
        if obj is not None:
            obj.hide_render = not show
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is not None and pg.role in {"SOURCE", "SINK", "FORCE"}:
            obj.hide_render = True


def configure_cycles_points(scene):
    """Ensure point clouds render as spheres in Cycles."""
    for name in (viewport.PARTICLES_NAME, viewport.WHITEWATER_NAME):
        obj = viewport.output_object(scene, name)
        if obj is None or obj.type != "POINTCLOUD":
            continue
        # Cycles renders point clouds natively; nothing more needed.
        # EEVEE ignores point clouds without geometry nodes as of 5.1.


@persistent
def _on_render_init(scene, *_args):
    # Must be @persistent: Blender frees non-persistent app handlers on file
    # load, so without it render visibility silently stops being enforced
    # after the user opens any .blend while the add-on stays enabled.
    #
    # This module owns render_init outright -- unregister_handlers() below drops
    # every add-on callback on that list -- so the solver hoist rides along here
    # rather than registering a second handler that this module would strip.
    from . import scene_bridge

    scene_bridge.ensure_session_before_frames(scene, "render")
    ensure_render_visibility(scene)


def register_handlers():
    unregister_handlers()
    bpy.app.handlers.render_init.append(_on_render_init)


def unregister_handlers():
    handlers = bpy.app.handlers.render_init
    for fn in list(handlers):
        if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51"):
            handlers.remove(fn)
