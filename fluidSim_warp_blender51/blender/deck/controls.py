"""Every Liquid Deck control, declared as a thin twin of an existing target.

A control holds no settings state. It names either an operator the panels also
call (``kind="operator"``) or a property the panels also draw
(``kind="property"``, with the group that declares it). Running a control calls
that operator or writes that property, so Pending Settings classification, undo
and file compatibility are exactly the panel's. ``tests/ui/test_deck_parity.py``
checks each target exists and has a panel twin.

One target belongs to another 4.0 lane and may be absent: ``collider_dynamic``
(rigid coupling). It is marked ``optional``; the Deck draws it disabled until the
target exists.

Pure Python: importable without ``bpy``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from ...presets.schema import SCENE_OWNED_SETTINGS


@dataclass(frozen=True)
class Control:
    kind: str  # "operator" | "property"
    target: str  # operator id, or property name
    owner: str  # "" for operators; "scene" | "domain" | "object" for properties
    label: str
    icon: str
    tooltip: str
    optional: bool = False
    properties: tuple = ()  # operator keyword arguments, as (name, value) pairs
    call_context: str = "INVOKE_DEFAULT"


def operator(target, label, icon, why, *, optional=False, call_context="INVOKE_DEFAULT",
             **properties) -> Control:
    return Control("operator", target, "", label, icon, f"{why} ({target})", optional,
                   tuple(sorted(properties.items())), call_context)


def setting(target, label, icon, why) -> Control:
    """A scene or domain setting; the declaring group is the one the panel draws."""
    owner = "scene" if target in SCENE_OWNED_SETTINGS else "domain"
    return Control("property", target, owner, label, icon, f"{why} ({target})")


def object_setting(target, label, icon, why, *, optional=False) -> Control:
    return Control("property", target, "object", label, icon, f"{why} ({target})", optional)


# Deck bar -------------------------------------------------------------------

PLAY = operator("screen.animation_play", "Play", "PLAY",
                "Play the timeline; the simulation runs live")
PAUSE = operator("fluidsim.preview_pause", "Pause", "PAUSE", "Pause live playback")
CANCEL_BAKE = operator("fluidsim.bake_cancel", "Cancel Bake", "CANCEL",
                       "Cancel the running Production Bake")
STEP = operator("fluidsim.step", "Step", "FRAME_NEXT", "Advance the simulation one frame",
                call_context="EXEC_DEFAULT", frames=1)
RESTART = operator("fluidsim.restart_simulation", "Restart", "FILE_REFRESH",
                   "Restart the simulation from the first frame")
APPLY = operator("fluidsim.apply_changes", "Apply", "CHECKMARK",
                 "Apply the pending changes with a reset or re-initialize")
REVERT = operator("fluidsim.revert_pending", "Revert", "LOOP_BACK",
                  "Revert the pending changes to the applied values")
LIQUID_PREVIEW = setting("liquid_preview_enabled", "Liquid Preview", "SHADING_RENDERED",
                         "Toggle the Liquid Preview look in the 3D View")

BAR_CONTROLS = (PLAY, PAUSE, CANCEL_BAKE, STEP, RESTART, APPLY, REVERT, LIQUID_PREVIEW)
PENDING_ACTIONS = (APPLY, REVERT)


def transport_controls(activity: str, playing: bool, owns_scene: bool) -> tuple[Control, ...]:
    """The transport buttons for this state, mirroring ``panels.draw_transport``.

    Step is offered where the Scene Properties panel offers Step One Frame.
    """
    if activity == "PRODUCTION_BAKE":
        first = CANCEL_BAKE
    elif playing:
        first = PAUSE
    else:
        first = PLAY
    result = [first]
    if owns_scene and activity not in {"CACHE_PLAYBACK", "PRODUCTION_BAKE"}:
        result.extend((STEP, RESTART))
    return tuple(result)


# Radial wheel: six slots per workflow stage ---------------------------------

STAGES = ("SETUP", "SIMULATE", "LOOK", "BAKE")
STAGE_LABELS = {"SETUP": "Setup", "SIMULATE": "Simulate", "LOOK": "Look", "BAKE": "Bake"}

STAGE_SLOTS = OrderedDict((
    ("SETUP", (
        operator("fluidsim.setup_scene", "Set Up Scene", "SCENE_DATA",
                 "Create a liquid domain and emitter in this scene"),
        operator("fluidsim.tag_liquid_emitter", "Emitter", "MOD_FLUIDSIM",
                 "Make the selected object a liquid emitter"),
        operator("fluidsim.tag_collider", "Collider", "MOD_PHYSICS",
                 "Make the selected object a collider"),
        operator("fluidsim.use_active_as_domain", "Domain", "CUBE",
                 "Use the active object as the liquid domain"),
        operator("fluidsim.apply_product_preset", "Preset", "PRESET",
                 "Apply the selected preset to this scene"),
        operator("fluidsim.load_demo", "Demo", "FILE_BLEND",
                 "Load the Dam Break demo into a new scene", demo_id="dam_break"),
    )),
    ("SIMULATE", (
        PLAY,
        STEP,
        RESTART,
        APPLY,
        setting("viscosity_enabled", "Viscosity", "MOD_SMOOTH",
                "Turn viscosity on or off"),
        setting("surface_tension_enabled", "Surface Tension", "MOD_FLUIDSIM",
                "Turn surface tension on or off"),
    )),
    ("LOOK", (
        LIQUID_PREVIEW,
        setting("surface_enabled", "Surface", "MESH_DATA", "Build the liquid surface mesh"),
        setting("whitewater_enabled", "Whitewater", "FORCE_TURBULENCE",
                "Simulate foam, bubbles and spray"),
        setting("show_whitewater", "Show Whitewater", "HIDE_OFF",
                "Show the whitewater particles in the viewport"),
        operator("fluidsim.install_render_nodes", "Materials", "MATERIAL",
                 "Install the liquid render materials and nodes"),
        setting("show_overlay", "Overlay", "OVERLAY", "Show the simulation stats overlay"),
    )),
    ("BAKE", (
        operator("fluidsim.bake_start", "Bake", "RENDER_ANIMATION",
                 "Start a Production Bake of the frame range"),
        operator("fluidsim.bake_resume", "Resume", "PLAY",
                 "Resume the Production Bake from its last checkpoint"),
        operator("fluidsim.cache_preview_start", "Play Cache", "FILE_MOVIE",
                 "Play the baked cache on the timeline"),
        operator("fluidsim.export_alembic", "Alembic", "EXPORT",
                 "Export the baked liquid as Alembic"),
        operator("fluidsim.export_vdb", "VDB", "EXPORT", "Export the baked liquid as OpenVDB"),
        operator("fluidsim.cache_free", "Free Cache", "TRASH",
                 "Delete the baked cache files"),
    )),
))


def next_stage(stage: str) -> str:
    try:
        index = STAGES.index(stage)
    except ValueError:
        return STAGES[0]
    return STAGES[(index + 1) % len(STAGES)]


# Gizmos ---------------------------------------------------------------------

EMITTER_VELOCITY = object_setting("initial_velocity", "Velocity", "ORIENTATION_NORMAL",
                                  "Drag to set the emitted liquid velocity")
EMITTER_MULTIPLIER = object_setting("source_sampling_multiplier", "Fill", "MOD_PARTICLES",
                                    "Turn to set the source fill multiplier")
EMITTER_LEGACY_RATE = object_setting("emit_rate", "Fill", "MOD_PARTICLES",
                                     "Turn to set the legacy target particles per voxel")
COLLIDER_FRICTION = object_setting("collider_friction", "Friction", "MOD_PHYSICS",
                                   "Turn to set the collider friction")
COLLIDER_DYNAMIC = object_setting("collider_dynamic", "Dynamic", "PHYSICS",
                                  "Toggle two-way rigid coupling for this collider",
                                  optional=True)

GIZMO_CONTROLS = (EMITTER_VELOCITY, EMITTER_MULTIPLIER, EMITTER_LEGACY_RATE,
                  COLLIDER_FRICTION, COLLIDER_DYNAMIC)


def all_controls() -> tuple[Control, ...]:
    """Every Deck control, each target once."""
    seen = {}
    for control in (*BAR_CONTROLS, *(c for slots in STAGE_SLOTS.values() for c in slots),
                    *GIZMO_CONTROLS):
        seen.setdefault((control.kind, control.target, control.owner), control)
    return tuple(seen.values())


# Running a control ------------------------------------------------------------

def operator_available(idname: str) -> bool:
    """Whether ``idname`` is a registered operator (``bpy.ops`` returns a stub for any name)."""
    import bpy

    module, _, name = idname.partition(".")
    try:
        getattr(getattr(bpy.ops, module), name).get_rna_type()
    except (AttributeError, KeyError, RuntimeError):
        return False
    return True


def property_owner(context, control):
    """The settings group ``control`` writes, or ``None`` when there is none."""
    if control.owner == "object":
        obj = getattr(context, "active_object", None)
        return getattr(obj, "fluidsim", None)
    scene = getattr(context, "scene", None)
    if scene is None:
        return None
    from ..pending_settings import declaring_group

    return declaring_group(scene, control.target)


def control_enabled(context, control) -> bool:
    if control.kind == "operator":
        return operator_available(control.target)
    group = property_owner(context, control)
    return group is not None and hasattr(group, control.target)


def control_active(context, control) -> bool:
    """Whether a boolean property control is currently on (``False`` for operators)."""
    if control.kind != "property":
        return False
    group = property_owner(context, control)
    return bool(getattr(group, control.target, False)) if group is not None else False


def run_control(context, control):
    """Call the control's operator, or toggle its boolean property, as the panel would."""
    if control.kind == "operator":
        if not operator_available(control.target):
            return {"CANCELLED"}
        import bpy

        module, _, name = control.target.partition(".")
        op = getattr(getattr(bpy.ops, module), name)
        if not op.poll():
            return {"CANCELLED"}
        return op(control.call_context, **dict(control.properties))
    group = property_owner(context, control)
    if group is None or not isinstance(getattr(group, control.target, None), bool):
        return {"CANCELLED"}
    setattr(group, control.target, not getattr(group, control.target))
    scene = getattr(context, "scene", None)
    if scene is not None:
        from . import bar

        bar.note_deck_write(scene, control.target)
    return {"FINISHED"}
