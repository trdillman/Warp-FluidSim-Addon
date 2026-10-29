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
    # The tooltip is plain language only: no operator or property names.
    return Control("operator", target, "", label, icon, why, optional,
                   tuple(sorted(properties.items())), call_context)


def setting(target, label, icon, why) -> Control:
    """A scene or domain setting; the declaring group is the one the panel draws."""
    owner = "scene" if target in SCENE_OWNED_SETTINGS else "domain"
    return Control("property", target, owner, label, icon, why)


def object_setting(target, label, icon, why, *, optional=False) -> Control:
    return Control("property", target, "object", label, icon, why, optional)


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

SEARCH = operator("fluidsim.command_palette", "Search", "VIEWZOOM",
                  "Search every FluidSim command, preset, demo and setting (Ctrl+K)")
WHEEL = operator("fluidsim.radial_wheel", "Wheel", "PROP_CON",
                 "Open the wheel of actions for the current stage (Alt+Right-click)")
BAKE = operator("fluidsim.bake_start", "Bake", "RENDER_ANIMATION",
                "Start a Production Bake of the frame range")
# What the Bake slot becomes, following the panel's Production Bake actions
# (``ui_scene.draw_cache_bake``): progress and Cancel while this scene bakes,
# Resume or Restart when the cache already holds frames.
BAKE_PROGRESS = operator("fluidsim.bake_cancel", "Cancel Bake", "RENDER_ANIMATION",
                         "The Production Bake is running; click to cancel it at the next "
                         "saved frame")
BAKE_RESUME = operator("fluidsim.bake_resume", "Resume", "PLAY",
                       "Resume the Production Bake from its last checkpoint")
BAKE_RESTART = operator("fluidsim.bake_start", "Restart Bake…", "FILE_REFRESH",
                        "Replace the existing cache and bake again (asks first)",
                        restart_existing=True)
# The Deck's own surfaces: they open the Deck's palette and wheel, so no panel twins them.
DECK_OPERATORS = frozenset({SEARCH.target, WHEEL.target})

# Right-hand group of the bar, after the pending chip.
TOOL_CONTROLS = (LIQUID_PREVIEW, SEARCH, WHEEL, BAKE)
BAR_CONTROLS = (PLAY, PAUSE, CANCEL_BAKE, STEP, RESTART, APPLY, REVERT, *TOOL_CONTROLS)
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


def deck_state(context) -> tuple[str, bool, bool]:
    """``(activity, playing, owns_scene)`` for the context scene, as the bar reads them."""
    from ..ui_common import activity_ui_state

    state = activity_ui_state(getattr(context, "scene", None))
    playing = bool(getattr(getattr(context, "screen", None), "is_animation_playing", False))
    return str(state.get("id", "IDLE")), playing, bool(state.get("owns_scene"))


def scene_is_baking(scene) -> bool:
    """Whether ``scene`` itself has a running Production Bake."""
    from .. import production_bake

    return any(baking is scene or baking == scene
               for baking, _controller in production_bake.active_bakes())


def _bake_action(scene) -> str:
    """``START``, ``RESUME`` or ``RESTART``: the panel's Production Bake action for the cache."""
    try:
        import bpy

        from ..ui_cache_state import production_bake_action

        raw = str(bpy.path.abspath(scene.fluidsim.cache_path) or "").strip()
        return production_bake_action(raw or None).action
    except Exception:  # noqa: BLE001 - presentation only; the operator re-checks
        return "START"


def bake_control(scene) -> Control:
    """The Bake slot for ``scene``: progress and Cancel, Resume, Restart, or Bake."""
    if scene_is_baking(scene):
        return BAKE_PROGRESS
    action = _bake_action(scene)
    if action == "RESUME":
        return BAKE_RESUME
    if action == "RESTART":
        return BAKE_RESTART
    return BAKE


def resolve(context, control) -> Control:
    """The control a slot stands for right now.

    Play follows the bar's transport (Cancel Bake while baking, Pause while
    playing), and Bake follows the cache (see :func:`bake_control`), so the wheel
    and the bar offer the same action in every state.
    """
    if control is PLAY:
        return transport_controls(*deck_state(context))[0]
    if control is BAKE:
        return bake_control(getattr(context, "scene", None))
    return control


def _state_reason(context, control) -> str:
    """Why ``control`` should not run in the current state, beyond its operator's poll."""
    target = control.target
    if target not in {STEP.target, APPLY.target, "fluidsim.install_render_nodes"}:
        return ""
    scene = getattr(context, "scene", None)
    if target == APPLY.target:
        from ..pending_settings import pending_presentation

        if str(pending_presentation(scene).get("action", "NONE")) == "NONE":
            return "No changes are waiting to be applied"
        return ""
    activity, _playing, owns_scene = deck_state(context)
    if target == STEP.target:
        # The bar offers Step only where the panel offers Step One Frame.
        if activity == "CACHE_PLAYBACK":
            return "Cache playback is on; switch to Live Preview to step the simulation"
        if not owns_scene:
            from ..scene_bridge import session

            if getattr(session(), "active", False):
                return "Another scene owns the solver; release it from that scene first"
        return ""
    from ..scene_access import active_domain_object

    if active_domain_object(scene) is None:
        return _MISSING_GROUP["domain"]
    if activity == "PRODUCTION_BAKE":
        return "A Production Bake is running; install the materials when it ends"
    return ""


# Radial wheel: six slots per workflow stage ---------------------------------

STAGES = ("SETUP", "SIMULATE", "LOOK", "BAKE")
STAGE_LABELS = {"SETUP": "Setup", "SIMULATE": "Simulate", "LOOK": "Look", "BAKE": "Bake"}

STAGE_SLOTS = OrderedDict((
    ("SETUP", (
        operator("fluidsim.setup_scene", "Set Up Scene", "SCENE_DATA",
                 "Fit a liquid domain around the selected mesh and tag the objects' roles"),
        operator("fluidsim.tag_liquid_emitter", "Emitter", "MOD_FLUIDSIM",
                 "Make the selected object a liquid emitter"),
        operator("fluidsim.tag_collider", "Collider", "MOD_PHYSICS",
                 "Make the selected object a collider"),
        operator("fluidsim.use_active_as_domain", "Domain", "CUBE",
                 "Use the active object as the liquid domain"),
        # Preset and Demo open the palette filtered to that list, so the wheel picks one.
        operator("fluidsim.command_palette", "Preset", "PRESET",
                 "Choose a preset and apply it to this scene", kinds="preset"),
        operator("fluidsim.command_palette", "Demo", "FILE_BLEND",
                 "Choose a demo and load it into a new scene", kinds="demo"),
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
        BAKE,
        BAKE_RESUME,
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
    """Whether the control's target exists (it may still be unable to run right now)."""
    if control.kind == "operator":
        return operator_available(control.target)
    group = property_owner(context, control)
    return group is not None and hasattr(group, control.target)


_MISSING_GROUP = {
    "object": "Select a FluidSim object first",
    "domain": "Assign a FluidSim domain first (Setup > Domain or Set Up Scene)",
    "scene": "No scene is open",
}


class _PollProbe:
    """Stands in for an operator class so its ``poll`` can be called to read its message."""

    def __init__(self, cls):
        self._cls = cls
        self.message = ""

    def __getattr__(self, name):
        return getattr(self._cls, name)

    def poll_message_set(self, message, *args):
        self.message = message(*args) if callable(message) else str(message)


def poll_reason(context, idname: str) -> str:
    """The message the operator's ``poll`` gives for failing, or ``""`` when it gives none."""
    import bpy

    module, _, name = idname.partition(".")
    try:
        identifier = getattr(getattr(bpy.ops, module), name).get_rna_type().identifier
        cls = getattr(bpy.types, identifier, None)
        poll = getattr(getattr(cls, "poll", None), "__func__", None)
        if poll is None:
            return ""
        probe = _PollProbe(cls)
        poll(probe, context)
        return probe.message
    except Exception:  # noqa: BLE001 - only the message is wanted; poll() decides
        return ""


def control_status(context, control) -> tuple[bool, str]:
    """``(can run now, why not)`` for drawing a control and explaining a refused click."""
    if control.kind == "operator":
        if not operator_available(control.target):
            return False, "Not available in this build"
        reason = _state_reason(context, control)
        if reason:
            return False, reason
        import bpy

        module, _, name = control.target.partition(".")
        if getattr(getattr(bpy.ops, module), name).poll():
            return True, ""
        return False, (poll_reason(context, control.target)
                       or "Not available in the current state")
    group = property_owner(context, control)
    if group is None:
        return False, _MISSING_GROUP.get(control.owner, "Not available")
    if not hasattr(group, control.target):
        return False, "Not available on this object"
    return True, ""


def call_operator(idname: str, call_context="INVOKE_DEFAULT", **properties):
    """``(result, message)``: run ``idname`` if it can run, else say why it did not."""
    import bpy

    module, _, name = idname.partition(".")
    op = getattr(getattr(bpy.ops, module), name)
    if not op.poll():
        return {"CANCELLED"}, (poll_reason(bpy.context, idname)
                               or "Not available in the current state")
    try:
        return op(call_context, **properties), ""
    except RuntimeError as exc:
        return {"CANCELLED"}, str(exc)


def control_active(context, control) -> bool:
    """Whether a boolean property control is currently on (``False`` for operators)."""
    if control.kind != "property":
        return False
    group = property_owner(context, control)
    return bool(getattr(group, control.target, False)) if group is not None else False


# The warning ``fluidsim.toggle_liquid_preview`` gives; the Deck toggle says the same.
NO_PUBLISHED_PARTICLES = ("Liquid Preview is on but no particles are published yet; "
                          "play the simulation in a particle view")


def _published_particles() -> bool:
    try:
        from .. import output_publication

        record = output_publication.latest_published_points()
    except Exception:  # noqa: BLE001 - only decides whether to warn
        return True
    return record is not None and getattr(record, "particle_count", 0) != 0


def run_control(context, control, report=None):
    """Call the control's operator, or toggle its boolean property, as the panel would.

    ``report(kind, message)`` (an operator's ``report``) receives what the panel
    would say: why an operator call failed, or the Liquid Preview warning.
    """
    if control.kind == "operator":
        if not operator_available(control.target):
            return {"CANCELLED"}
        result, message = call_operator(control.target, control.call_context,
                                        **dict(control.properties))
        if message and report is not None:
            report({"WARNING"}, f"{control.label}: {message}")
        return result
    group = property_owner(context, control)
    if group is None or not isinstance(getattr(group, control.target, None), bool):
        return {"CANCELLED"}
    value = not getattr(group, control.target)
    setattr(group, control.target, value)
    scene = getattr(context, "scene", None)
    if scene is not None:
        from . import bar

        bar.note_deck_write(scene, control.target, group)
    if (control is LIQUID_PREVIEW and value and report is not None
            and not _published_particles()):
        report({"WARNING"}, NO_PUBLISHED_PARTICLES)
    return {"FINISHED"}
