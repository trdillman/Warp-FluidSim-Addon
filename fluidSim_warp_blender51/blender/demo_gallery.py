"""Demo Gallery for FluidSim canonical scenes in the 3D Viewport N-panel.

Provides discovery, inspection, and safe scene loading of the 6 canonical
simulations (Dam Break, Crown Splash, Paddle Stir, Chromatic Vortex, Density
Ladder, Water Wheel).
Loading always appends the demo into a New Scene, preserving active user
work and initial empty scenes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy

from .scene_access import active_domain_settings
from .scene_bridge import release_session
from .ui_common import FluidSimView3DPanel

_types = getattr(bpy, "types", None)
_Operator = getattr(_types, "Operator", object) if _types else object
_Panel = getattr(_types, "Panel", object) if _types else object

_props = getattr(bpy, "props", None)
_StringProp = getattr(_props, "StringProperty", lambda **kw: None) if _props else (lambda **kw: None)

# Canonical demo catalog metadata matching DEMO_SCENE_CONTRACT.md and demos/MANIFEST.md
DEMO_CATALOG: dict[str, dict[str, Any]] = {
    "dam_break": {
        "title": "Dam Break (Rocky Cove)",
        "scene_title": "Dam Break (Demo)",
        "blend_filename": "dam_break.blend",
        "preview_filename": "dam_break_preview.png",
        "icon": "MOD_FLUIDSIM",
        "subtitle": "Turbulent Cove Flow & Obstacles",
        "description": "Sudden reservoir release onto rocky cove terrain with apron and whitewater.",
        "tags": ("256k Particles", "0.03m Voxels", "Whitewater", "Obstacles"),
        "key_feature": "Martin & Moses front benchmark with cove terrain and jetty colliders.",
    },
    "crown_splash": {
        "title": "Crown Splash",
        "scene_title": "Crown Splash (Demo)",
        "blend_filename": "crown_splash.blend",
        "preview_filename": "crown_splash_preview.png",
        "icon": "WATER",
        "subtitle": "High-Speed Droplet Impact",
        "description": "Worthington crown formation from a droplet impacting a shallow liquid pool.",
        "tags": ("259k Particles", "0.01m Voxels", "High Churn Whitewater", "Fine Detail"),
        "key_feature": "Surface tension sheet dynamics and prompt whitewater emission.",
    },
    "paddle_stir": {
        "title": "Paddle Stir",
        "scene_title": "Paddle Stir (Demo)",
        "blend_filename": "paddle_stir.blend",
        "preview_filename": None,
        "icon": "FORCE_VORTEX",
        "subtitle": "Moving Collider Velocity Coupling",
        "description": "Keyframed rotating 4-blade paddle in a cylindrical tank driving liquid momentum.",
        "tags": ("158k Particles", "0.012m Voxels", "Moving Collider", "Velocity Coupling"),
        "key_feature": "Rotational boundary momentum transfer and obstacle wall containment.",
    },
    "chromatic_vortex": {
        "title": "Chromatic Vortex",
        "scene_title": "Chromatic Vortex (Demo)",
        "blend_filename": "chromatic_vortex.blend",
        "preview_filename": None,
        "icon": "COLOR",
        "subtitle": "Multi-Species Color Mixing",
        "description": "Dual dye streams (red & blue) mixing inside a glass vessel under a vortex force.",
        "tags": ("Color Mixing", "Vortex Force", "Glass Vessel Collider", "Dye Advection"),
        "key_feature": "Advected dye concentration field mixing with GPU color blending.",
    },
    "buoyancy_ladder": {
        "title": "Density Ladder",
        "scene_title": "Density Ladder (Demo)",
        "blend_filename": "buoyancy_ladder.blend",
        "preview_filename": "buoyancy_ladder_preview.png",
        "icon": "SORTSIZE",
        "subtitle": "Two-Way Rigid Coupling: Buoyancy",
        "description": "Four blocks of 250, 500, 750 and 1500 kg/m3 drop into a pool; the liquid floats the light ones at their own depth and sinks the heavy one.",
        "tags": ("Dynamic Colliders", "Buoyancy", "Pose Keyframes", "0.01m Voxels"),
        "key_feature": "Each block's draft comes from its density, solved by the liquid.",
    },
    "water_wheel": {
        "title": "Water Wheel",
        "scene_title": "Water Wheel (Demo)",
        "blend_filename": "water_wheel.blend",
        "preview_filename": "water_wheel_preview.png",
        "icon": "MOD_SCREW",
        "subtitle": "Two-Way Rigid Coupling: Locked Axes",
        "description": "A falling stream turns a six-paddle wheel pinned to its axle; a floor drain keeps the water moving.",
        "tags": ("Dynamic Collider", "Axis Locks", "Inflow + Sink", "0.01m Voxels"),
        "key_feature": "Lock Axes turn a free body into a hinge the liquid drives.",
    },
}

_preview_collection = None


def get_preview_collection():
    """Return the cached preview collection, or None if previews are unavailable."""
    global _preview_collection
    if _preview_collection is None:
        try:
            import bpy.utils.previews

            _preview_collection = bpy.utils.previews.new()
        except Exception:
            _preview_collection = None
    return _preview_collection


def cleanup_preview_icons():
    """Unload custom preview icon collection."""
    global _preview_collection
    if _preview_collection is not None:
        try:
            import bpy.utils.previews

            bpy.utils.previews.remove(_preview_collection)
        except Exception:
            pass
        _preview_collection = None


_ADDON_DIR = Path(__file__).resolve().parents[1]

# Where demo files are looked for, in priority order.
DEMO_SEARCH_DIRS: tuple[Path, ...] = (
    _ADDON_DIR.parent / "demos",  # repository checkout (development workflow)
    _ADDON_DIR / "resources" / "demos",  # bundled with an installed extension
    _ADDON_DIR / "demos",  # add-on sibling folder
)


def _find_demo_asset(filename: str | None) -> Path | None:
    """Return the first existing ``filename`` across :data:`DEMO_SEARCH_DIRS`."""
    if not filename:
        return None
    for directory in DEMO_SEARCH_DIRS:
        path = directory / filename
        if path.is_file():
            return path
    return None


def resolve_demo_blend_path(demo_id: str) -> Path | None:
    """Resolve the filesystem path to the canonical .blend file for demo_id."""
    meta = DEMO_CATALOG.get(demo_id)
    return _find_demo_asset(meta["blend_filename"]) if meta else None


def resolve_demo_preview_path(demo_id: str) -> Path | None:
    """Resolve the path to a PNG preview thumbnail for demo_id, if present."""
    meta = DEMO_CATALOG.get(demo_id)
    return _find_demo_asset(meta.get("preview_filename")) if meta else None


def demo_render_filepath(demo_id: str) -> str:
    """Blend-relative render output for an opened demo scene."""
    return f"//fluidsim_demo_renders/{demo_id}_"


def available_demo_ids() -> tuple[str, ...]:
    """Catalog ids whose .blend file exists on disk, in catalog order."""
    return tuple(d for d in DEMO_CATALOG if resolve_demo_blend_path(d) is not None)


def demo_tooltip(demo_id: str) -> str:
    """Hover text for a demo's load button: what it shows and what loading does."""
    meta = DEMO_CATALOG.get(demo_id)
    if not meta:
        return FLUIDSIM_OT_load_demo.bl_description
    return "\n".join(
        (
            meta["description"],
            meta["key_feature"],
            " · ".join(meta["tags"]),
            "Opens in a new scene; your current scene is kept.",
        )
    )


def is_scene_empty(scene: bpy.types.Scene | None) -> bool:
    """Return True if `scene` is an unmodified empty or default startup scene.

    An empty scene has no active FluidSim domain, and either has 0 objects
    or only contains standard default startup objects (Cube, Light, Camera)
    without animation, custom modifiers, or FluidSim roles.
    """
    if scene is None:
        return True

    if active_domain_settings(scene) is not None:
        return False

    objects = list(getattr(scene, "objects", ()))
    if len(objects) == 0:
        return True

    default_names = {"Cube", "Light", "Camera"}
    if len(objects) <= 3 and all(obj.name in default_names for obj in objects):
        for obj in objects:
            if getattr(obj, "animation_data", None) and obj.animation_data.action:
                return False
            if getattr(obj, "modifiers", None) and len(obj.modifiers) > 0:
                return False
            fs = getattr(obj, "fluidsim", None)
            if fs is not None and getattr(fs, "role", "NONE") != "NONE":
                return False
            if any(str(k).startswith("fluidsim") for k in obj.keys()):
                return False
        return True

    return False


def _unique_scene_name(desired_name: str, existing_scene=None) -> str:
    """Return a unique scene name starting with desired_name."""
    scenes = bpy.data.scenes
    if desired_name not in scenes or scenes[desired_name] == existing_scene:
        return desired_name
    idx = 2
    while f"{desired_name} {idx}" in scenes:
        idx += 1
    return f"{desired_name} {idx}"


class FLUIDSIM_OT_load_demo(_Operator):
    """Load a canonical FluidSim demo scene into a New Scene."""

    bl_idname = "fluidsim.load_demo"
    bl_label = "Load Demo Scene"
    bl_description = "Load this demo simulation into Blender as a New Scene"
    bl_options = {"REGISTER", "UNDO"}

    demo_id: _StringProp(
        name="Demo ID",
        description="Identifier of the demo to load",
        default="dam_break",
    )

    @classmethod
    def description(cls, context, properties):
        return demo_tooltip(getattr(properties, "demo_id", ""))

    @classmethod
    def poll(cls, context):
        # Loading releases the shared solver session, which a running bake
        # is still stepping (owner decision D2).
        from . import production_bake

        if production_bake.any_bake_active():
            cls.poll_message_set("A Production Bake is running; load demos after it ends")
            return False
        return True

    def execute(self, context):
        demo = DEMO_CATALOG.get(self.demo_id)
        if not demo:
            self.report({"ERROR"}, f"Unknown demo '{self.demo_id}'")
            return {"CANCELLED"}

        blend_path = resolve_demo_blend_path(self.demo_id)
        if blend_path is None or not blend_path.is_file():
            self.report(
                {"ERROR"},
                f"Demo blend file '{demo['blend_filename']}' not found on disk",
            )
            return {"CANCELLED"}

        current_scene = getattr(context, "scene", None)
        was_empty = is_scene_empty(current_scene)

        # Release active GPU solver session on the old scene before switching
        if current_scene is not None:
            try:
                release_session(current_scene)
            except Exception:
                pass

        existing_scene_names = {s.name for s in bpy.data.scenes}

        # Append the scene and its dependencies as local datablocks
        try:
            with bpy.data.libraries.load(str(blend_path), link=False) as (
                data_from,
                data_to,
            ):
                data_to.scenes = data_from.scenes
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to load demo library: {exc}")
            return {"CANCELLED"}

        # Find the newly imported scene
        imported_scene = None
        for scn in data_to.scenes:
            if scn is not None:
                imported_scene = scn
                break

        if imported_scene is None:
            new_scenes = [
                s for s in bpy.data.scenes if s.name not in existing_scene_names
            ]
            if new_scenes:
                imported_scene = new_scenes[0]

        if imported_scene is None:
            self.report({"ERROR"}, f"Could not import scene from {blend_path.name}")
            return {"CANCELLED"}

        # Rename to clean descriptive title (e.g. "Dam Break (Demo)")
        desired_title = demo.get("scene_title", demo["title"])
        imported_scene.name = _unique_scene_name(desired_title, imported_scene)
        _migrate_appended_scene(imported_scene)

        # Activate the new scene in the active window
        if hasattr(context, "window") and context.window is not None:
            context.window.scene = imported_scene

        # Ensure timeline starts at frame_start
        imported_scene.frame_current = imported_scene.frame_start
        # The demo files carry the absolute output path of the machine that
        # built them; render next to the artist's own .blend instead.
        imported_scene.render.filepath = demo_render_filepath(self.demo_id)

        scene_note = (
            "initial empty scene preserved"
            if was_empty
            else "previous scene preserved"
        )
        self.report(
            {"INFO"},
            f"Opened {demo['title']} in new scene '{imported_scene.name}' ({scene_note})",
        )
        return {"FINISHED"}


def _migrate_appended_scene(scene) -> None:
    """Run the pre-3.0 domain migration on an appended scene.

    ``load_post`` migrates files that are opened, not data appended from a
    library, so a demo saved before Resolution Divisions would otherwise come in
    at the default 64 divisions instead of its authored voxel size.
    """
    from .file_compat import migrate_domain_resolution

    migrate_domain_resolution(objects=tuple(scene.objects), scenes=(scene,))


def _demo_icon_kwargs(demo_id: str, pcoll) -> dict[str, Any]:
    """Thumbnail icon when a preview image is loaded, else the catalog's built-in icon."""
    if pcoll is not None and demo_id in pcoll:
        return {"icon_value": pcoll[demo_id].icon_id}
    return {"icon": DEMO_CATALOG[demo_id].get("icon", "MOD_FLUIDSIM")}


class FLUIDSIM_PT_demo_gallery(FluidSimView3DPanel, _Panel):
    """Collapsible demo shelf in the FluidSim 3D View sidebar.

    One button per demo keeps it short enough to sit above the workflow stages;
    the full description lives in each button's tooltip. The shelf hides itself
    when no demo files are on disk, so an install without them shows no dead UI.
    """

    bl_label = "Demo Gallery"
    bl_idname = "FLUIDSIM_PT_demo_gallery"
    bl_order = 8
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        if scene is None or getattr(scene, "fluidsim", None) is None:
            return False
        return bool(available_demo_ids())

    def draw(self, context):
        layout = self.layout
        pcoll = get_preview_collection()
        available = set(available_demo_ids())

        hint = layout.row()
        hint.active = False
        hint.label(text="Each demo opens in a new scene.", icon="INFO")

        for demo_id, meta in DEMO_CATALOG.items():
            column = layout.column(align=True)
            button = column.row(align=True)
            button.scale_y = 1.4
            if demo_id in available:
                op = button.operator(
                    "fluidsim.load_demo",
                    text=meta["title"],
                    **_demo_icon_kwargs(demo_id, pcoll),
                )
                op.demo_id = demo_id
            else:
                button.enabled = False
                button.label(text=f"{meta['title']} (file missing)", icon="ERROR")

            subtitle = column.row()
            subtitle.active = False
            subtitle.scale_y = 0.8
            subtitle.label(text=meta["subtitle"])


PANEL_CLASSES = (
    FLUIDSIM_OT_load_demo,
    FLUIDSIM_PT_demo_gallery,
)


def register():
    for cls in PANEL_CLASSES:
        bpy.utils.register_class(cls)

    # Load preview icons if images exist
    pcoll = get_preview_collection()
    if pcoll is not None:
        for demo_id in DEMO_CATALOG:
            preview_path = resolve_demo_preview_path(demo_id)
            if preview_path and preview_path.is_file() and demo_id not in pcoll:
                try:
                    pcoll.load(demo_id, str(preview_path), "IMAGE")
                except Exception:
                    pass


def unregister():
    from ..unregister_guard import unregister_classes

    unregister_classes(PANEL_CLASSES)
    cleanup_preview_icons()
