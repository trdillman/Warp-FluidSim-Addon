"""Scene-owned liquid viewport representation policy."""

_SYNCING = False
MODES = ("PREVIEW", "SURFACE", "OFF")
_HELD = 0
_PENDING = set()


def cycle(settings):
    mode = settings.liquid_display_mode
    settings.liquid_display_mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
    return settings.liquid_display_mode


def mode_changed(settings, context):
    global _SYNCING
    if _SYNCING:
        return
    _SYNCING = True
    try:
        settings.liquid_preview_enabled = settings.liquid_display_mode == "PREVIEW"
        from .liquid_preview import handler
        from .pending_settings import SceneUpdate, request_scene_update
        handler.sync_registration()
        handler.tag_redraw()
        scene = getattr(settings, "id_data", None) or getattr(context, "scene", None)
        if scene is not None:
            request_scene_update(scene, SceneUpdate.OUTPUT)
    finally:
        _SYNCING = False


def preview_changed(settings, context):
    if _SYNCING:
        return
    if hasattr(settings, "liquid_display_mode"):
        settings.liquid_display_mode = "PREVIEW" if settings.liquid_preview_enabled else "SURFACE"
    else:
        from .liquid_preview import handler
        handler.sync_registration()
        handler.tag_redraw()


def viewport_families(scene, surface_enabled):
    """Viewport geometry needs, independent of render/export visibility."""
    from .liquid_preview import handler
    mode = getattr(scene.fluidsim, "liquid_display_mode", None)
    ordinary = mode == "SURFACE" or (mode == "PREVIEW" and handler.failure() is not None)
    return bool(ordinary and surface_enabled), bool(ordinary and not surface_enabled)


def hold_viewport():
    """Let an exporter own output visibility until ``release_viewport``.

    Export un-hides the objects it samples and runs frame updates; the display
    policy must not re-hide them from per-frame publication in the middle of
    those updates (Blender 5.2.0 crashes in the depsgraph).
    """
    global _HELD
    _HELD += 1


def release_viewport():
    global _HELD
    _HELD = max(0, _HELD - 1)


def apply_viewport(scene, *, deferred=True):
    if _HELD:
        # Outputs can be created while an exporter owns visibility. Remember
        # the scene so its current policy is restored after the final hold.
        _defer_view_layer_write(scene)
        return
    from . import output_publication as output
    mode = getattr(getattr(scene, "fluidsim", None), "liquid_display_mode", None)
    if mode is None:
        return
    fs = scene.fluidsim
    ds = output._domain_settings(scene)
    show_surface, show_particles = viewport_families(
        scene, getattr(ds, "surface_enabled", False))
    surface = output.output_object(scene, output.SURFACE_NAME)
    if show_surface and (surface is None or not len(getattr(getattr(surface, "data", None), "vertices", (1,)))):
        show_surface, show_particles = False, True
    changes = []
    for name, hidden in ((output.SURFACE_NAME, not show_surface),
                         (output.PARTICLES_NAME, not show_particles)):
        obj = output.output_object(scene, name)
        if obj is not None:
            # Render/export visibility stays authored; viewport visibility has
            # its own policy and must also clear the global viewport hide flag.
            render_visible = (getattr(fs, "show_surface", False)
                              and getattr(ds, "surface_enabled", False)
                              if name == output.SURFACE_NAME else
                              getattr(fs, "show_particles", False))
            output._set_hidden(obj, not render_visible, viewport=False)
            if not hidden:
                output._set_hidden(obj, False, render=False)
            # hide_set re-tags the depsgraph even for an equal value, so only
            # flags that actually change are written.
            layers = getattr(scene, "view_layers", ())
            if layers:
                for layer in layers:
                    if obj.name in layer.objects and obj.hide_get(view_layer=layer) != hidden:
                        changes.append((obj, layer, hidden))
            elif obj.hide_get() != hidden:
                changes.append((obj, None, hidden))
    if not changes:
        return
    if deferred and _defer_view_layer_write(scene):
        return
    for obj, layer, hidden in changes:
        if layer is None:
            obj.hide_set(hidden)
        else:
            obj.hide_set(hidden, view_layer=layer)


def _defer_view_layer_write(scene):
    """Queue the view-layer write for a timer; False when it must run now.

    Per-frame publication runs inside the frame-change handler, i.e. during the
    depsgraph update. Changing a base's hide flag there (typically when the
    output objects were just created) crashed Blender 5.2.0 in
    deg_check_base_in_depsgraph, so the write waits until the update is over.
    """
    name = getattr(scene, "name", None)
    try:
        import bpy
        timers = bpy.app.timers
        if not isinstance(name, str):
            return False
        _PENDING.add(name)
        if not timers.is_registered(_flush_pending):
            timers.register(_flush_pending, first_interval=0.0)
        return True
    except Exception:  # noqa: BLE001 - no usable timer: write immediately
        return False


def _flush_pending():
    import bpy

    if _HELD:
        # Keep both the timer and its requests alive across nested exports.
        return 0.01
    names = set(_PENDING)
    _PENDING.clear()
    for scene in bpy.data.scenes:
        if scene.name in names:
            apply_viewport(scene, deferred=False)
    return None


def restore_scene(scene):
    """Migrate old files and restore the saved viewport mode on load."""
    global _SYNCING
    settings = getattr(scene, "fluidsim", None)
    if settings is None or not hasattr(settings, "liquid_display_mode"):
        return
    _SYNCING = True
    try:
        is_set = getattr(settings, "is_property_set", lambda _name: False)
        if not is_set("liquid_display_mode"):
            from .scene_access import active_domain_settings
            ds = active_domain_settings(scene)
            particles_only = (getattr(settings, "show_particles", False)
                              and not (getattr(settings, "show_surface", False)
                                       and getattr(ds, "surface_enabled", False)))
            settings.liquid_display_mode = "PREVIEW" if settings.liquid_preview_enabled or particles_only else "SURFACE"
        settings.liquid_preview_enabled = settings.liquid_display_mode == "PREVIEW"
        apply_viewport(scene)
    finally:
        _SYNCING = False


def show_surface_fallback(scene):
    """Restore available geometry; refresh missing outputs outside drawing."""
    apply_viewport(scene)
    import bpy
    if not bpy.app.timers.is_registered(_refresh_fallback_outputs):
        bpy.app.timers.register(_refresh_fallback_outputs, first_interval=0.0)


def _refresh_fallback_outputs():
    import bpy

    from .pending_settings import SceneUpdate, request_scene_update
    for scene in bpy.data.scenes:
        if getattr(getattr(scene, "fluidsim", None), "liquid_preview_enabled", False):
            request_scene_update(scene, SceneUpdate.OUTPUT)
    return None
