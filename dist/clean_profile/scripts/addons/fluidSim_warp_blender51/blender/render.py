"""Render output integration.

Simulation outputs are real datablocks, so they render directly. This module
keeps render visibility coherent and steps the solver during animation
renders (render frame-change fires the same handler as playback when the
simulation is live).
"""

from __future__ import annotations

import bpy

from . import viewport


def ensure_render_visibility(scene):
    """Make simulation outputs render-visible and source/collider helpers not."""
    fs = scene.fluidsim
    for name, show in (
        (viewport.SURFACE_NAME, fs.show_surface),
        (viewport.PARTICLES_NAME, fs.show_particles),
        (viewport.WHITEWATER_NAME, fs.show_whitewater),
    ):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = not show
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is not None and pg.role in {"SOURCE", "SINK", "FORCE"}:
            obj.hide_render = True


def configure_cycles_points(scene):
    """Ensure point clouds render as spheres in Cycles."""
    for name in (viewport.PARTICLES_NAME, viewport.WHITEWATER_NAME):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "POINTCLOUD":
            continue
        # Cycles renders point clouds natively; nothing more needed.
        # EEVEE ignores point clouds without geometry nodes as of 5.1.


def _on_render_init(scene, *_args):
    ensure_render_visibility(scene)


def register_handlers():
    unregister_handlers()
    bpy.app.handlers.render_init.append(_on_render_init)


def unregister_handlers():
    handlers = bpy.app.handlers.render_init
    for fn in list(handlers):
        if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51"):
            handlers.remove(fn)
