"""Shared Blender UI contracts for FluidSim Properties and viewport surfaces."""

from __future__ import annotations

import bpy


class FluidSimScenePanel:
    """Base for FluidSim panels in Blender Scene Properties."""

    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return scene is not None and getattr(scene, "fluidsim", None) is not None


class FluidSimObjectPanel:
    """Base for FluidSim panels in Blender Object Properties."""

    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        obj = getattr(context, "object", None)
        return (
            scene is not None
            and getattr(scene, "fluidsim", None) is not None
            and obj is not None
            and getattr(obj, "fluidsim", None) is not None
        )


class FluidSimView3DPanel:
    """Base for the View3D sidebar surfaces: the status HUD launcher and the Live Tuning shelf."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FluidSim"


_ACTIVITY_ICONS = {
    "IDLE": "PAUSE",
    "LIVE_PREVIEW": "PLAY",
    "CACHE_PLAYBACK": "PLAY",
    "PRODUCTION_BAKE": "RENDER_ANIMATION",
}


def activity_ui_state(scene) -> dict[str, object]:
    """Return the shared derived activity plus current-scene ownership."""
    from .playback import activity_snapshot
    from .scene_bridge import session

    owns_scene = bool(session().owns_scene(scene))
    result = dict(activity_snapshot(scene, owns_scene=owns_scene))
    result["icon"] = _ACTIVITY_ICONS.get(str(result.get("id", "IDLE")), "INFO")
    return result


def pending_ui_state(scene) -> dict[str, object]:
    """Expose the existing pending-settings presentation without duplicating policy."""
    from .pending_settings import pending_presentation

    return dict(pending_presentation(scene))


def active_object_role(context) -> str:
    """Return the canonical role for the active object, or ``NONE``."""
    obj = getattr(context, "object", None)
    settings = getattr(obj, "fluidsim", None) if obj is not None else None
    return str(getattr(settings, "role", "NONE") or "NONE")


def canonical_region_kind(obj) -> str:
    """Return the role-owned canonical region kind for ``obj``."""
    settings = getattr(obj, "fluidsim", None) if obj is not None else None
    if settings is None:
        return ""
    role = str(getattr(settings, "role", "NONE") or "NONE")
    if role == "SOURCE":
        return str(getattr(settings, "source_region_kind", "ANALYTIC") or "ANALYTIC")
    if role == "SINK":
        return str(getattr(settings, "sink_region_kind", "ANALYTIC") or "ANALYTIC")
    if role == "FIELD":
        return str(getattr(settings, "field_region_kind", "ANALYTIC") or "ANALYTIC")
    if role == "FORCE":
        return "ANALYTIC"
    return ""


# (severity key) -> (Blender icon, text prefix, use alert row)
_SEVERITY_PRESENTATION = {
    "OK": ("CHECKMARK", "", False),
    "INFO": ("INFO", "", False),
    "WARNING": ("ERROR", "Warning: ", False),   # ERROR = Blender's warning-triangle glyph
    "ERROR": ("CANCEL", "Error: ", True),        # circle-slash + red alert row = strictly heavier
}


def draw_status_message(layout, text: str, severity: str = "INFO") -> None:
    """Draw one status row; severity is shape + text (+ color), never icon-alone."""
    icon, prefix, alert = _SEVERITY_PRESENTATION.get(
        str(severity).upper(), _SEVERITY_PRESENTATION["INFO"]
    )
    row = layout.row(align=True)
    if alert:
        row.alert = True
    row.label(text=f"{prefix}{text}", icon=icon)


def draw_property(layout, data, name: str, *, text: str | None = None, **kwargs) -> bool:
    """Draw an RNA property when present; return whether it was drawn."""
    if data is None or not hasattr(data, name):
        return False
    if text is None:
        layout.prop(data, name, **kwargs)
    else:
        layout.prop(data, name, text=text, **kwargs)
    return True


def draw_property_grid(layout, data, names: tuple[str, ...], *, columns: int = 2):
    """Draw a compact Blender-native grid and skip unavailable compatibility fields."""
    grid = layout.grid_flow(
        row_major=True,
        columns=max(1, int(columns)),
        even_columns=True,
        even_rows=False,
        align=True,
    )
    for name in names:
        draw_property(grid, data, name)
    return grid
