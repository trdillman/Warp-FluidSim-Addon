"""Scene-owned liquid viewport representation policy."""

_SYNCING = False
MODES = ("PREVIEW", "SURFACE", "OFF")


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


def apply_viewport(scene):
    from . import output_publication as output
    mode = getattr(getattr(scene, "fluidsim", None), "liquid_display_mode", None)
    if mode is None:
        return
    from .liquid_preview import handler
    show_surface = mode == "SURFACE" or (mode == "PREVIEW" and handler.failure() is not None)
    for name, hidden in ((output.SURFACE_NAME, not show_surface),
                         (output.PARTICLES_NAME, True)):
        obj = output.output_object(scene, name)
        if obj is not None:
            layers = getattr(scene, "view_layers", ())
            if layers:
                for layer in layers:
                    if obj.name in layer.objects:
                        obj.hide_set(hidden, view_layer=layer)
            else:
                obj.hide_set(hidden)


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
            settings.liquid_display_mode = "PREVIEW" if settings.liquid_preview_enabled else "SURFACE"
        settings.liquid_preview_enabled = settings.liquid_display_mode == "PREVIEW"
        apply_viewport(scene)
    finally:
        _SYNCING = False


def show_surface_fallback(scene):
    """Keep a generated surface visible when the preview renderer fails."""
    from . import output_publication as output
    obj = output.output_object(scene, output.SURFACE_NAME)
    if obj is None:
        return
    layers = getattr(scene, "view_layers", ())
    if layers:
        for layer in layers:
            if obj.name in layer.objects:
                obj.hide_set(False, view_layer=layer)
    else:
        obj.hide_set(False)
