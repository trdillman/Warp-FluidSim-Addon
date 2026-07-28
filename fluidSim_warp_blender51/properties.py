"""Blender property groups for the FluidSim Warp add-on.

Scene-level settings drive the solver; object-level settings tag scene
objects as sources / colliders / sinks. Property definitions import only
pure enum and snippet metadata so registration never touches Warp/CUDA.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .sim import custom
from .sim.enums import (
    BlendMode,
    CustomKernelConstantType,
    CustomKernelOrder,
    CustomKernelRunOver,
    LimitBehavior,
    MeshEvaluationMode,
    MeshSinkRegionMode,
    ParticleSampleType,
    ProjectNonDivergentSolver,
    RelaxMode,
    SdfFilterType,
    SurfaceResolutionMode,
    SurfaceUpdateCadence,
    SourceHeightBackMode,
    SourceGeometryMode,
    SourceShapeForceType,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceEmissionMode,
    SourceShapeType,
    TraceMode,
    ViscositySolver,
    enum_items,
)


def _items(enum_cls, descriptions=None):
    return enum_items(enum_cls, descriptions)


LIQUID_VIEW_ITEMS = (
    ("SURFACE", "Surface", "Show the marching-cubes liquid surface"),
    ("PARTICLES", "Particles", "Show liquid as particles only"),
    ("SURFACE_PARTICLES", "Surface + Particles", "Show both surface mesh and liquid particles"),
)


LIQUID_ATTR_ITEMS = (
    ("NONE", "Material", "Use the assigned material color"),
    ("HEIGHT", "Height", "Color by world-space height"),
    ("SPEED", "Speed", "Color by liquid velocity magnitude"),
    ("TEMPERATURE", "Temperature", "Color by particle temperature attribute"),
    ("VISCOSITY", "Viscosity", "Color by particle viscosity attribute"),
    ("DENSITY", "Density", "Color by particle density attribute"),
)


PARTICLE_ATTR_ITEMS = LIQUID_ATTR_ITEMS + (
    ("AGE", "Age", "Color by particle age"),
    ("LIFE", "Life", "Color by particle life span"),
    ("TIME_OFFSET", "ST Time Offset", "Color by beta ST-FLIP time residual"),
)


WHITEWATER_ATTR_ITEMS = (
    ("NONE", "Material", "Use the assigned material color"),
    ("HEIGHT", "Height", "Color by world-space height"),
    ("SPEED", "Speed", "Color by whitewater velocity magnitude"),
    ("AGE", "Age", "Color by whitewater age"),
    ("LIFE", "Life", "Color by whitewater life span"),
)


SURFACE_SMOOTH_ITEMS = (
    ("BOX", "Box", "Use the existing box smoothing filter"),
    ("MEAN_CURVATURE_FLOW", "Mean Curvature", "Use the existing mean-curvature smoothing filter"),
)


_LIQUID_VIEW_SYNCING = False


def _apply_liquid_view_mode(self, context):
    global _LIQUID_VIEW_SYNCING
    if _LIQUID_VIEW_SYNCING:
        return
    _LIQUID_VIEW_SYNCING = True
    try:
        mode = self.liquid_view_mode
        self.show_surface = mode in {"SURFACE", "SURFACE_PARTICLES"}
        self.show_particles = mode in {"PARTICLES", "SURFACE_PARTICLES"}
    finally:
        _LIQUID_VIEW_SYNCING = False
    _request_output_refresh(self, context)


def _sync_liquid_view_mode(self, context):
    global _LIQUID_VIEW_SYNCING
    if _LIQUID_VIEW_SYNCING:
        return
    _LIQUID_VIEW_SYNCING = True
    try:
        if self.show_surface and self.show_particles:
            self.liquid_view_mode = "SURFACE_PARTICLES"
        elif self.show_surface:
            self.liquid_view_mode = "SURFACE"
        elif self.show_particles:
            self.liquid_view_mode = "PARTICLES"
    finally:
        _LIQUID_VIEW_SYNCING = False
    _request_output_refresh(self, context)


def _request_scene_update(context, action):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        return
    try:
        from .blender import scene_bridge

        scene_bridge.request_scene_update(scene, action)
    except Exception as exc:  # noqa: BLE001 - property callbacks must be safe
        try:
            scene.fluidsim.status = f"Settings changed; auto-update failed: {exc}"
        except Exception:  # noqa: BLE001
            pass


def _request_rebuild(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.REBUILD)


def _request_reset(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.RESET)


def _request_sync(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.SYNC)


def _request_output_refresh(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.OUTPUT)


def _request_surface_refresh(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.SURFACE)


def _request_whitewater_refresh(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.WHITEWATER)


def _on_live_changed(self, context):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        return
    try:
        from .blender import scene_bridge

        scene_bridge.set_live_enabled(scene, bool(self.live))
    except Exception as exc:  # noqa: BLE001
        self.status = f"Live toggle failed: {exc}"


def _on_cache_playback_changed(self, context):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        return
    try:
        from .blender import scene_bridge

        scene_bridge.set_cache_playback_enabled(
            scene, bool(self.cache_playback_enabled))
    except Exception as exc:  # noqa: BLE001
        self.status = f"Cache playback toggle failed: {exc}"


# ---------------------------------------------------------------------------
# Snippet library dropdown (examples + user-saved presets)
# ---------------------------------------------------------------------------


def user_snippet_dir(create=False) -> str:
    """Directory where user snippet presets are saved as JSON."""
    import os

    try:
        base = bpy.utils.user_resource("CONFIG", path="fluidsim_snippets",
                                       create=create)
    except Exception:  # headless/stub environments
        base = os.path.join(os.path.expanduser("~"), ".fluidsim_snippets")
        if create:
            os.makedirs(base, exist_ok=True)
    return base


# Blender requires enum item strings to stay referenced from Python.
_SNIPPET_LIBRARY_ITEMS = []

_LIBRARY_CATEGORY_ORDER = ("Impressive", "Creative", "Learning", "Utility")


def _snippet_library_items(self, context):
    global _SNIPPET_LIBRARY_ITEMS
    import os

    # Dynamic EnumProperty defaults to item index 0. A separator/header cannot
    # occupy that slot: Blender then resolves the fresh-scene value to "" and
    # logs an RNA warning. Keep an explicit non-action item first.
    items = [("NONE", "Select a snippet...", "Choose a snippet to load")]
    by_cat = {}
    for preset in custom.EXAMPLE_PRESETS:
        by_cat.setdefault(preset.category, []).append(preset)
    for cat in _LIBRARY_CATEGORY_ORDER:
        presets = by_cat.pop(cat, None)
        if not presets:
            continue
        items.append(("", cat, ""))
        for p in presets:
            items.append((f"EX:{p.preset_id}", p.label, p.description))
    for cat in sorted(by_cat):  # any future categories
        items.append(("", cat, ""))
        for p in by_cat[cat]:
            items.append((f"EX:{p.preset_id}", p.label, p.description))
    user_presets = custom.load_user_presets(user_snippet_dir())
    if user_presets:
        items.append(("", "My Snippets", ""))
        for data in user_presets:
            ident = "USR:" + os.path.basename(data["_path"])
            items.append((ident, data["label"],
                          data.get("description", "User snippet")))
    _SNIPPET_LIBRARY_ITEMS = items
    return items


_SHORTCUT_FIELD_ITEMS = [
    (name, name, f"Solver field '{name}'") for name in custom.KNOWN_FIELDS
]


# ---------------------------------------------------------------------------
# Custom snippet argument / snippet property groups
# ---------------------------------------------------------------------------


class FLUIDSIM_PG_snippet_argument(bpy.types.PropertyGroup):
    mode: EnumProperty(
        name="Mode",
        items=[("FIELD", "Field", "Bind a solver field"),
               ("CONSTANT", "Constant", "Bind a constant value")],
        default="FIELD",
    )
    field: StringProperty(
        name="Field",
        description="Solver field name (e.g. velocity_u, temperature, pressure)",
        default="temperature",
    )
    constant_type: EnumProperty(
        name="Type", items=_items(CustomKernelConstantType), default="FLOAT")
    constant_name: StringProperty(name="Name", default="value")
    constant_value: FloatProperty(name="Value", default=0.0)


class FLUIDSIM_PG_snippet(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=True)
    kernel_name: StringProperty(name="Kernel Name", default="my_kernel")
    description: StringProperty(name="Description", default="")
    order: EnumProperty(name="Order", items=_items(CustomKernelOrder),
                        default="BEFORE_FORCES")
    run_over: EnumProperty(name="Run Over", items=_items(CustomKernelRunOver),
                           default="VOXEL_LINEAR")
    text_block: PointerProperty(
        name="Code", type=bpy.types.Text,
        description="Text datablock containing the Warp snippet body",
    )
    arguments: CollectionProperty(type=FLUIDSIM_PG_snippet_argument)
    active_argument: IntProperty(default=0)
    last_error: StringProperty(name="Last Error", default="")
    # Validation gate: a snippet only reaches the solver when its current
    # source passed static validation (is_valid + matching hash).
    is_valid: BoolProperty(name="Valid", default=False)
    validated_hash: StringProperty(default="")
    validation_report: StringProperty(name="Validation Report", default="")
    # Layering window: combine snippets at different times / frame strides.
    frame_start: IntProperty(
        name="Start Frame", default=0, min=0,
        description="First frame this snippet runs on (0 = from the start)")
    frame_end: IntProperty(
        name="End Frame", default=0, min=0,
        description="Last frame this snippet runs on (0 = never stops)")
    frame_step: IntProperty(
        name="Frame Step", default=1, min=1, max=250,
        description="Run this snippet every Nth frame")


# ---------------------------------------------------------------------------
# Object-level settings (sources / colliders / sinks)
# ---------------------------------------------------------------------------


def _mesh_object_poll(_settings, obj):
    return getattr(obj, "type", None) == "MESH"


def _point_input_poll(_settings, obj):
    return (getattr(obj, "type", None) in {"MESH", "POINTCLOUD"}
            or bool(getattr(obj, "particle_systems", None)))


def _volume_input_poll(_settings, obj):
    return getattr(obj, "type", None) == "VOLUME"


class FLUIDSIM_PG_object(bpy.types.PropertyGroup):
    role: EnumProperty(
        name="Role",
        items=[
            ("NONE", "None", "Not part of the simulation"),
            ("SOURCE", "Liquid Emitter", "Emit liquid"),
            ("COLLIDER", "Collider", "Obstacle (mesh SDF)"),
            ("SINK", "Liquid Sink", "Remove liquid"),
            ("FORCE", "Force Region", "Apply an analytic force to liquid"),
            ("FIELD", "Advanced Field Source",
             "Source pressure, scalar, vector, or staggered velocity fields without emitting liquid"),
        ],
        default="NONE",
        update=_request_reset,
    )
    # Source shape settings.
    shape: EnumProperty(
        name="Shape", items=_items(SourceShapeShape), default="BOX",
        update=_request_sync,
        description=(
            "Analytic emission shape with exact local Euclidean distance "
            "and the object transform applied"
        ))
    use_mesh_shape: BoolProperty(
        name="Use Mesh", default=False,
        update=_request_sync,
        description="Legacy compatibility flag; use Geometry = Mesh")
    source_ui_schema_version: IntProperty(
        name="Source UI Schema", default=0, min=0,
        description="Non-destructive presentation schema marker")
    source_region_kind: EnumProperty(
        name="Emission Region",
        items=[
            ("ANALYTIC", "Analytic Volume", "Use an analytic signed-distance region"),
            ("MESH_SOLID", "Mesh Solid Volume", "Fill a validated watertight mesh interior"),
            ("MESH_SHELL", "Mesh Surface Shell", "Emit from an explicit triangle shell"),
            ("POINTS", "Points / Particles", "Use owned point or particle input"),
            ("VOLUME_GRID", "Volume Grid", "Import named OpenVDB grids from a Blender Volume"),
        ],
        default="ANALYTIC", update=_request_sync)
    sink_region_kind: EnumProperty(
        name="Removal Region",
        items=[
            ("ANALYTIC", "Analytic Volume", "Remove liquid in an analytic region"),
            ("MESH_SOLID", "Mesh Solid Interior", "Remove liquid inside a watertight mesh"),
            ("MESH_SHELL", "Mesh Surface Shell", "Remove liquid in a mesh shell"),
            ("VOLUME_GRID", "Volume Grid Mask", "Remove liquid using an imported mask or SDF"),
        ],
        default="ANALYTIC", update=_request_sync)
    point_source_representation: EnumProperty(
        name="Point Representation",
        items=[
            ("DIRECT_APPEND", "Direct Append Once", "Append the owned input points once; not a volume fill"),
            ("RASTERIZED_REGION", "Rasterized Region", "Rasterize points into source leaves and top up occupancy"),
        ],
        default="DIRECT_APPEND", update=_request_reset)
    point_input_object: PointerProperty(
        name="Point Input", type=bpy.types.Object, poll=_point_input_poll,
        update=_request_sync)
    point_rasterize_radius: FloatProperty(
        name="Rasterize Radius", default=0.1, min=0.0, subtype="DISTANCE",
        update=_request_sync)
    volume_input_object: PointerProperty(
        name="Volume Input", type=bpy.types.Object, poll=_volume_input_poll,
        update=_request_sync)
    volume_grid_type: EnumProperty(
        name="Volume Region", items=[
            ("FOG", "Fog / Mask", "Use a fog grid above the threshold"),
            ("SDF", "Signed Distance", "Use the inside of an SDF grid"),
        ], default="FOG", update=_request_sync)
    volume_mask_grid: StringProperty(
        name="Fog / Mask Grid", default="density", update=_request_sync)
    volume_sdf_grid: StringProperty(
        name="SDF Grid", default="surface", update=_request_sync)
    volume_scalar_grid: StringProperty(
        name="Scalar Grid", default="", update=_request_sync)
    volume_vector_grid: StringProperty(
        name="Vector Grid", default="", update=_request_sync)
    volume_vector_x_grid: StringProperty(
        name="Vector X Grid", default="", update=_request_sync)
    volume_vector_y_grid: StringProperty(
        name="Vector Y Grid", default="", update=_request_sync)
    volume_vector_z_grid: StringProperty(
        name="Vector Z Grid", default="", update=_request_sync)
    volume_fog_threshold: FloatProperty(
        name="Fog Threshold", default=0.0, update=_request_sync)
    field_operation: EnumProperty(
        name="Target Field", items=[
            ("PRESSURE", "Pressure", "Source pressure before projection"),
            ("SCALAR", "Scalar / Temperature", "Source a scalar field after P2G"),
            ("VECTOR", "Vector Velocity", "Source a centered vector field after P2G"),
            ("STAGGERED_VELOCITY", "Staggered Velocity", "Source independent MAC face components"),
        ], default="PRESSURE", update=_request_sync)
    field_region_kind: EnumProperty(
        name="Field Region", items=[
            ("ANALYTIC", "Analytic Region", "Use this object's analytic region"),
            ("VOLUME_GRID", "Volume Grid", "Use named values from a Blender Volume"),
        ], default="ANALYTIC", update=_request_sync)
    field_value_source: EnumProperty(
        name="Payload Source", items=[
            ("CONSTANT", "Constant", "Use the constant value below"),
            ("VOLUME_GRID", "Volume Grid", "Sample the named scalar or vector grid"),
        ], default="CONSTANT", update=_request_sync)
    field_scalar_value: FloatProperty(
        name="Field Value", default=1.0, update=_request_sync)
    field_vector_value: FloatVectorProperty(
        name="Field Vector", size=3, default=(0.0, 0.0, 0.0),
        subtype="VELOCITY", update=_request_sync)
    field_amount: FloatProperty(
        name="Blend Amount", default=1.0, min=0.0, update=_request_sync)
    geometry_mode: EnumProperty(
        name="Region",
        items=[
            ("ANALYTIC", "Analytic",
             "Use one of the six analytic shapes",
             SourceGeometryMode.ANALYTIC.value),
            ("MESH", "Mesh",
             "Use evaluated mesh triangles",
             SourceGeometryMode.MESH.value),
        ],
        default=SourceGeometryMode.ANALYTIC.name,
        update=_request_sync,
    )
    mesh_object: PointerProperty(
        name="Mesh Object",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
        update=_request_sync,
        description="Optional mesh object; empty uses this tagged object's mesh",
    )
    mesh_evaluation_mode: EnumProperty(
        name="Mesh Evaluation",
        items=_items(MeshEvaluationMode, {
            "FIRST_SIM_FRAME": (
                "Capture evaluated geometry on the first simulated frame and retain it until reset"),
            "EVERY_SIM_FRAME": (
                "Reevaluate modifiers, transforms, animation, and topology before every simulation frame"),
        }),
        default=MeshEvaluationMode.FIRST_SIM_FRAME.name,
        update=_request_reset,
    )
    mesh_surface_thickness: FloatProperty(
        name="Surface Band",
        default=0.0,
        min=0.0,
        subtype="DISTANCE",
        update=_request_sync,
        description="Triangle shell half-width; zero uses the voxel-resolution bound",
    )
    sink_region_mode: EnumProperty(
        name="Sink Region",
        items=[
            ("SOLID", "Mesh Solid Interior",
             "Remove particles inside a validated watertight mesh",
             MeshSinkRegionMode.SOLID.value),
            ("SURFACE", "Mesh Surface Shell",
             "Remove particles only in the triangle shell",
             MeshSinkRegionMode.SURFACE.value),
        ],
        default=MeshSinkRegionMode.SOLID.name,
        update=_request_sync,
    )
    mesh_error: StringProperty(name="Mesh Error", default="")
    source_mode: EnumProperty(
        name="Mode", items=_items(SourceShapeSourceMode), default="PARTICLES",
        update=_request_sync,
        description=(
            "Compatibility solver representation. PARTICLES emits liquid; "
            "FIELD is the original pressure-influence mode and does not "
            "create liquid (pressure sourcing is not yet exposed in this port)"))
    emit_rate: FloatProperty(
        name="Target Particles / Voxel", default=8.0, min=0.0, soft_max=32.0,
        update=_request_sync,
        description=(
            "Target particle occupancy in each source voxel; this is not a "
            "volume-per-second emission rate"))
    emission_mode: EnumProperty(
        name="Emission",
        items=_items(SourceEmissionMode, {
            "INITIAL_FILL": (
                "Fill the source region once on the first enabled step "
                "after Initialize/Reset"),
            "CONTINUOUS_INFLOW": (
                "Replenish the source region every substep while Enabled"),
        }),
        default=SourceEmissionMode.CONTINUOUS_INFLOW.name,
        update=_request_reset,
        description="Source emission lifecycle")
    initial_velocity: FloatVectorProperty(
        name="Initial Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY",
        update=_request_sync)
    use_object_velocity: BoolProperty(
        name="Inherit Object Velocity", default=True,
        update=_request_sync,
        description="Add velocity from animated object motion")
    blend_mode: EnumProperty(name="Blend", items=_items(BlendMode), default="REPLACE",
                             update=_request_sync)
    feather: FloatProperty(name="Feather", default=0.0, min=0.0,
                           update=_request_sync)
    temperature: FloatProperty(name="Temperature", default=0.0,
                               update=_request_sync)
    viscosity: FloatProperty(name="Viscosity", default=0.0, min=0.0,
                             update=_request_sync)
    density: FloatProperty(
        name="Density Attribute", default=1.0, min=0.0,
        description="Particle density attribute; not physical liquid mass",
        update=_request_sync)
    use_lifespan: BoolProperty(name="Use Lifespan", default=False,
                               update=_request_sync)
    life_min: FloatProperty(name="Life Min", default=2.0, min=0.0,
                            update=_request_sync)
    life_max: FloatProperty(name="Life Max", default=4.0, min=0.0,
                            update=_request_sync)
    force_type: EnumProperty(
        name="Force Type", items=_items(SourceShapeForceType), default="OFF",
        update=_request_sync)
    force_strength: FloatProperty(name="Force Strength", default=1.0,
                                  update=_request_sync)
    force_direction: FloatVectorProperty(
        name="Force Direction", size=3, default=(0.0, 0.0, 1.0),
        update=_request_sync)
    # Collider settings.
    collider_friction: FloatProperty(
        name="Friction", default=0.0, min=0.0, max=1.0,
        description="Tangential velocity loss for collisions with this collider only",
        update=_request_sync)
    collider_restitution: FloatProperty(
        name="Restitution", default=0.0, min=0.0, max=1.0,
        description="Bounce for collisions with this collider only",
        update=_request_sync)
    transfer_velocity: BoolProperty(
        name="Transfer Velocity", default=True,
        update=_request_sync,
        description="Push liquid with animated collider motion")
    enabled: BoolProperty(name="Enabled", default=True, update=_request_sync)
    source_status_severity: StringProperty(name="Source Status Severity", default="")
    source_status_code: StringProperty(name="Source Status Code", default="")
    source_status_message: StringProperty(name="Source Status", default="")
    source_status_emitted: IntProperty(name="Last Emitted", default=0, min=0)
    source_status_capacity_limited: BoolProperty(name="Capacity Limited", default=False)
    auto_inject_seed: BoolProperty(
        name="Auto Inject Seed",
        default=False,
        update=_request_reset,
        description=(
            "Inject this object's points when the solver is initialized or "
            "reset; use for playable seed-based demo scenes"
        ),
    )


# ---------------------------------------------------------------------------
# Scene-level settings
# ---------------------------------------------------------------------------


class FLUIDSIM_PG_settings(bpy.types.PropertyGroup):
    # -- Simulation -----------------------------------------------------------
    live: BoolProperty(
        name="Live Simulation", default=False,
        update=_on_live_changed,
        description="Step the GPU solver during animation playback")
    max_frame_catchup: IntProperty(
        name="Max Frame Catch-up", default=8, min=1, max=120,
        description=(
            "Largest forward timeline jump the live solver will step through "
            "during one frame change. Blender drops frames during playback, so "
            "a single step per frame change would silently run the sim in slow "
            "motion. Beyond this bound the solver reports a desync instead of "
            "stalling Blender on a long scrub"))
    domain_object: PointerProperty(
        name="Domain", type=bpy.types.Object,
        update=_request_rebuild,
        description="Object whose bounding box defines the simulation domain")
    time_scale: FloatProperty(name="Time Scale", default=1.0, min=0.0, soft_max=4.0,
                              update=_request_reset)
    cfl: FloatProperty(name="CFL", default=1.0, min=0.1, soft_max=5.0,
                       update=_request_reset)
    min_substeps: IntProperty(name="Min Substeps", default=1, min=1, max=16,
                              update=_request_reset)
    max_substeps: IntProperty(name="Max Substeps", default=4, min=1, max=32,
                              update=_request_reset)
    gravity: FloatVectorProperty(
        name="Gravity", size=3, default=(0.0, 0.0, -9.81), subtype="ACCELERATION",
        update=_request_reset)
    use_scene_gravity: BoolProperty(name="Use Scene Gravity", default=True,
                                    update=_request_reset)

    # -- Domain / sparse grid -------------------------------------------------
    voxel_size: FloatProperty(
        name="Voxel Size", default=0.05, min=0.005, soft_max=0.5, precision=4,
        update=_request_rebuild)
    leaf_capacity: IntProperty(
        name="Leaf Capacity", default=8192, min=64, max=262144,
        update=_request_rebuild,
        description="Maximum allocated 8x8x8 leaves (sparse tiled leaf grid)")
    topology_expand: IntProperty(
        name="Topology Expand", default=1, min=0, max=4,
        update=_request_reset,
        description="Leaf dilation iterations around active fluid")
    open_boundaries: BoolProperty(
        name="Open Boundaries", default=False,
        update=_request_reset,
        description="Kill particles leaving the domain instead of clamping")
    domain_sdf_walls: BoolProperty(
        name="Domain Collision Walls", default=True,
        update=_request_reset,
        description="Build collision SDF walls around the domain bounds "
                     "(recommended for closed domains; without this the "
                     "pressure solve treats sparse-grid padding beyond the "
                     "domain box as open space)")

    # -- Particles -------------------------------------------------------------
    max_particles: IntProperty(
        name="Max Particles", default=2_000_000, min=1024, max=50_000_000,
        update=_request_rebuild)
    particles_per_voxel: IntProperty(name="Particles / Voxel", default=8, min=1, max=64,
                                     update=_request_reset)
    sample_type: EnumProperty(
        name="Transfer", items=_items(ParticleSampleType), default="FLIP",
        update=_request_reset)
    flip_ratio: FloatProperty(name="FLIP Ratio", default=0.97, min=0.0, max=1.0,
                              update=_request_reset)
    trace_mode: EnumProperty(name="Trace", items=_items(TraceMode), default="RK3",
                             update=_request_reset)
    relax_enabled: BoolProperty(name="Relax Particles", default=True,
                                update=_request_reset)
    relax_strength: FloatProperty(name="Relax Strength", default=0.5, min=0.0, max=1.0,
                                  update=_request_reset)
    relax_mode: EnumProperty(name="Relax Mode", items=_items(RelaxMode),
                             default="LOCAL_VOXEL", update=_request_reset)
    max_speed: FloatProperty(
        name="Max Speed", default=0.0, min=0.0,
        update=_request_reset,
        description="Speed limit in m/s (0 disables)")
    limit_behavior: EnumProperty(
        name="Limit Behavior", items=_items(LimitBehavior), default="MAINTAIN",
        update=_request_reset)
    limit_percent: FloatProperty(name="Limit Percent", default=0.5, min=0.0, max=1.0,
                                 update=_request_reset)
    # -- Global forces ---------------------------------------------------------
    directional_force: FloatVectorProperty(
        name="Directional Acceleration", size=3, default=(0.0, 0.0, 0.0),
        subtype="ACCELERATION", update=_request_reset,
        description="Constant acceleration applied to every valid velocity face")
    omnidirectional_force_center: FloatVectorProperty(
        name="Center", size=3, default=(0.0, 0.0, 0.0),
        update=_request_reset,
        description="World-space center of the radial global force")
    omnidirectional_force_strength: FloatProperty(
        name="Radial Acceleration", default=0.0, update=_request_reset,
        description="Positive repels; negative attracts")
    omnidirectional_force_radius: FloatProperty(
        name="Radius", default=1.0, min=0.001, subtype="DISTANCE",
        update=_request_reset,
        description="Falloff radius of the radial global force")
    force_diffusion: FloatProperty(
        name="Velocity Diffusion", default=0.0, min=0.0, max=1.0,
        update=_request_reset,
        description="Blend the MAC velocity toward its six-neighbour average")
    dissipation_enabled: BoolProperty(
        name="Dissipate Particle Attribute", default=False,
        update=_request_reset)
    dissipation_attribute: EnumProperty(
        name="Attribute",
        items=[
            ("temperature", "Temperature", "Dissipate particle temperature"),
            ("viscosity", "Viscosity", "Dissipate particle viscosity"),
            ("density", "Density", "Dissipate particle density"),
        ],
        default="temperature", update=_request_reset)
    dissipation_rate: FloatProperty(
        name="Rate", default=0.0, min=0.0, subtype="FACTOR",
        update=_request_reset,
        description="Exponential decay rate per second")
    dissipation_target: FloatProperty(
        name="Target", default=0.0, update=_request_reset,
        description="Value approached by the selected particle attribute")
    attraction_enabled: BoolProperty(
        name="Attract to Collision SDF", default=False,
        update=_request_reset,
        description="Accelerate liquid toward colliders and closed domain walls")
    attraction_strength: FloatProperty(
        name="Attraction Strength", default=0.0, update=_request_reset)
    attraction_max_distance: FloatProperty(
        name="Attraction Distance", default=1.0, min=0.001,
        subtype="DISTANCE", update=_request_reset,
        description="Maximum distance from collision SDF surfaces affected by attraction")
    count_stiffness: FloatProperty(
        name="Count Stiffness", default=0.0, min=0.0, soft_max=4.0,
        update=_request_reset,
        description="Anti-clumping pressure from particle counts")

    # -- Spatiotemporal FLIP (beta) -------------------------------------------
    spacetime_enabled: BoolProperty(
        name="Enable ST-FLIP (Beta)", default=False, update=_request_reset,
        description="Experimental spatiotemporal FLIP path for large time steps")
    spacetime_jitter_strength: FloatProperty(
        name="Jitter", default=1.0, min=0.0, max=2.0, update=_request_reset)
    spacetime_adaptive_jitter: BoolProperty(
        name="Adaptive Jitter", default=True, update=_request_reset)
    spacetime_phase_eta: FloatProperty(
        name="Phase Eta", default=0.5, min=0.01, soft_max=2.0, update=_request_reset)
    spacetime_phase_threshold: FloatProperty(
        name="Phase Threshold", default=0.5, min=0.0, max=1.0,
        update=_request_reset)
    spacetime_reference_mass: FloatProperty(
        name="Reference Mass", default=0.0, min=0.0, soft_max=64.0,
        update=_request_reset)
    spacetime_variable_projection: BoolProperty(
        name="Variable Projection (Beta)", default=True, update=_request_reset)
    spacetime_liquid_density: FloatProperty(
        name="Liquid Density", default=1.0, min=1.0e-6, soft_max=10.0,
        update=_request_reset)
    spacetime_gas_density: FloatProperty(
        name="Gas Density", default=0.0, min=0.0, soft_max=1.0,
        update=_request_reset)
    spacetime_min_density: FloatProperty(
        name="Min Density", default=0.05, min=1.0e-6, soft_max=1.0,
        update=_request_reset)
    spacetime_local_cfl: FloatProperty(
        name="Local CFL", default=1.0, min=0.0, soft_max=4.0,
        update=_request_reset)
    spacetime_max_local_substeps: IntProperty(
        name="Max Local Substeps", default=64, min=1, max=256,
        update=_request_reset)
    spacetime_resync_surface: BoolProperty(
        name="Resync Surface", default=True, update=_request_surface_refresh)
    spacetime_seed: IntProperty(
        name="Seed", default=1729, min=0, max=2147483647, update=_request_reset)

    # -- Pressure ---------------------------------------------------------------
    pressure_solver: EnumProperty(
        name="Solver", items=_items(ProjectNonDivergentSolver), default="PCG",
        update=_request_reset)
    pressure_iterations: IntProperty(name="Iterations", default=200, min=1, max=4000,
                                     update=_request_reset)
    pressure_tolerance: FloatProperty(
        name="Tolerance", default=1.0e-5, min=0.0, precision=8,
        update=_request_reset)
    pressure_threshold: FloatProperty(
        name="Threshold", default=0.0, min=0.0, max=1.0,
        update=_request_reset,
        description="Relative pressure threshold (removes tiny pressures)")

    # -- Viscosity ---------------------------------------------------------------
    viscosity_enabled: BoolProperty(name="Enable Viscosity", default=False,
                                    update=_request_reset)
    viscosity_solver: EnumProperty(
        name="Solver", items=_items(ViscositySolver), default="PCG",
        update=_request_reset)
    viscosity_constant: FloatProperty(
        name="Viscosity", default=0.05, min=0.0, soft_max=100.0,
        update=_request_reset)
    viscosity_from_particles: BoolProperty(
        name="From Particles", default=False,
        update=_request_reset,
        description="Use per-particle viscosity attribute instead of the constant")
    viscosity_iterations: IntProperty(name="Iterations", default=40, min=1, max=1000,
                                      update=_request_reset)
    viscosity_tolerance: FloatProperty(
        name="Tolerance", default=1.0e-4, min=0.0, precision=8,
        update=_request_reset)

    # -- Surface -------------------------------------------------------------------
    surface_enabled: BoolProperty(name="Generate Surface", default=True,
                                  update=_request_surface_refresh)
    particle_radius_scale: FloatProperty(
        name="Particle Radius", default=1.2, min=0.5, soft_max=4.0,
        update=_request_surface_refresh)
    surface_splat_radius: IntProperty(
        name="Splat Radius", default=2, min=1, max=16,
        update=_request_surface_refresh,
        description="Minimum particle SDF splat support in surface-grid voxels")
    surface_resolution_mode: EnumProperty(
        name="Resolution Mode", items=_items(SurfaceResolutionMode),
        default="SIMULATION", update=_request_surface_refresh)
    surface_resolution_multiplier: FloatProperty(
        name="Resolution Multiplier", default=2.0, min=0.25, soft_max=4.0, max=8.0,
        update=_request_surface_refresh,
        description="Render surface density relative to simulation; 2 means half voxel size")
    surface_voxel_size: FloatProperty(
        name="Surface Voxel Size", default=0.0, min=0.0, soft_max=1.0,
        precision=5, update=_request_surface_refresh)
    surface_leaf_capacity_scale: FloatProperty(
        name="Leaf Capacity Scale", default=8.0, min=1.0, soft_max=16.0, max=128.0,
        update=_request_surface_refresh)
    surface_max_dense_dim: IntProperty(
        name="Max Dense Dim", default=512, min=32, max=2048,
        update=_request_surface_refresh)
    surface_update_cadence: EnumProperty(
        name="Update Cadence", items=_items(SurfaceUpdateCadence),
        default="FRAME", update=_request_surface_refresh)
    live_surface_publish_cadence: IntProperty(
        name="Live Surface Every", default=4, min=1, max=8,
        update=_request_surface_refresh,
        description=("Publish the Blender viewport surface every N live frames; "
                     "physics, cache writes, and exports still evaluate every frame"))
    surface_build_lowres_sdf: BoolProperty(
        name="Keep Sim Surface Field", default=True,
        update=_request_surface_refresh)
    surface_filter_type: EnumProperty(
        name="Filter", items=_items(SdfFilterType), default="BOX",
        update=_request_surface_refresh)
    surface_filter_iterations: IntProperty(name="Filter Iterations",
                                           default=0, min=0, max=16,
                                           update=_request_surface_refresh)
    surface_filter_strength: FloatProperty(name="Filter Strength",
                                           default=0.7, min=0.0, max=1.0,
                                           update=_request_surface_refresh)
    surface_dilate_iterations: IntProperty(name="Dilate Iterations",
                                           default=0, min=0, max=16,
                                           update=_request_surface_refresh)
    surface_dilate_strength: FloatProperty(name="Dilate Strength",
                                           default=1.0, min=0.0, max=1.0,
                                           update=_request_surface_refresh)
    surface_erode_iterations: IntProperty(name="Erode Iterations",
                                          default=0, min=0, max=16,
                                          update=_request_surface_refresh)
    surface_erode_strength: FloatProperty(name="Erode Strength",
                                          default=1.0, min=0.0, max=1.0,
                                          update=_request_surface_refresh)
    surface_smooth_type: EnumProperty(name="Smooth Type",
                                      items=SURFACE_SMOOTH_ITEMS,
                                      default="BOX",
                                      update=_request_surface_refresh)
    surface_smooth_iterations: IntProperty(name="Smooth Iterations",
                                           default=1, min=0, max=16,
                                           update=_request_surface_refresh)
    surface_smooth_strength: FloatProperty(name="Smooth Strength",
                                           default=0.7, min=0.0, max=1.0,
                                           update=_request_surface_refresh)

    # -- Whitewater -------------------------------------------------------------
    whitewater_enabled: BoolProperty(name="Enable Whitewater", default=False,
                                     update=_request_whitewater_refresh)
    max_whitewater: IntProperty(
        name="Max Whitewater", default=1_000_000, min=1024, max=20_000_000,
        update=_request_rebuild)
    ww_churn_decay: FloatProperty(name="Churn Decay", default=2.0, min=0.0,
                                  update=_request_reset)
    ww_churn_gain: FloatProperty(name="Churn Gain", default=1.0, min=0.0,
                                 update=_request_reset)
    ww_churn_threshold: FloatProperty(name="Churn Threshold", default=0.5, min=0.0,
                                      update=_request_reset)
    ww_emission_rate: FloatProperty(name="Emission Rate", default=2.0, min=0.0,
                                    update=_request_reset)
    ww_min_speed: FloatProperty(name="Min Speed", default=0.5, min=0.0,
                                update=_request_reset)
    ww_life_min: FloatProperty(name="Life Min", default=1.0, min=0.0,
                               update=_request_reset)
    ww_life_max: FloatProperty(name="Life Max", default=3.0, min=0.0,
                               update=_request_reset)
    ww_bubble_drag: FloatProperty(name="Bubble Drag", default=4.0, min=0.0,
                                  update=_request_reset)
    ww_spray_drag: FloatProperty(name="Spray Drag", default=0.2, min=0.0,
                                 update=_request_reset)

    # -- Height source ------------------------------------------------------------
    height_enabled: BoolProperty(name="Height Fill", default=False,
                                 update=_request_reset,
                                 description="Fill the domain below a height")
    height_base: FloatProperty(name="Base Height", default=0.3,
                               update=_request_reset)
    height_back_mode: EnumProperty(
        name="Back Mode", items=_items(SourceHeightBackMode), default="FLAT",
        update=_request_reset)
    height_profile_image: PointerProperty(
        name="Height Profile", type=bpy.types.Image, update=_request_reset,
        description="Grayscale image sampled across the domain as height offset")
    height_profile_scale: FloatProperty(
        name="Profile Scale", default=1.0, update=_request_reset)
    height_emission_mode: EnumProperty(
        name="Fill Lifecycle", items=_items(SourceEmissionMode),
        default=SourceEmissionMode.CONTINUOUS_INFLOW.name,
        update=_request_reset)
    height_error: StringProperty(name="Height Fill Error", default="")
    height_velocity: FloatVectorProperty(name="Velocity", size=3,
                                         default=(0.0, 0.0, 0.0),
                                         update=_request_reset)
    height_emit_particles: BoolProperty(name="Emit Particles", default=True,
                                        update=_request_reset)

    # -- Output ---------------------------------------------------------------------
    liquid_view_mode: EnumProperty(
        name="Liquid View",
        items=LIQUID_VIEW_ITEMS,
        default="SURFACE",
        update=_apply_liquid_view_mode,
    )
    show_surface: BoolProperty(
        name="Surface Mesh", default=True, update=_sync_liquid_view_mode)
    show_particles: BoolProperty(
        name="Particles", default=False, update=_sync_liquid_view_mode)
    show_whitewater: BoolProperty(name="Whitewater", default=True,
                                  update=_request_output_refresh)
    particle_display_fraction: FloatProperty(
        name="Display Fraction", default=1.0, min=0.01, max=1.0,
        update=_request_output_refresh,
        description="Fraction of particles shown in the viewport")
    particle_display_scale: FloatProperty(
        name="Display Scale", default=1.0, min=0.05, soft_max=4.0,
        update=_request_output_refresh,
        description="Viewport point radius scale for liquid particles")
    surface_attribute_mode: EnumProperty(
        name="Surface Attribute",
        items=LIQUID_ATTR_ITEMS,
        default="NONE",
        update=_request_output_refresh,
        description="Color the liquid surface by a simulation attribute",
    )
    particle_attribute_mode: EnumProperty(
        name="Particle Attribute",
        items=PARTICLE_ATTR_ITEMS,
        default="NONE",
        update=_request_output_refresh,
        description="Color liquid particles by a simulation attribute",
    )
    whitewater_attribute_mode: EnumProperty(
        name="Whitewater Attribute",
        items=WHITEWATER_ATTR_ITEMS,
        default="NONE",
        update=_request_output_refresh,
        description="Color whitewater particles by a simulation attribute",
    )
    whitewater_display_scale: FloatProperty(
        name="Whitewater Scale", default=1.0, min=0.05, soft_max=4.0,
        update=_request_output_refresh,
        description="Viewport point radius scale for whitewater particles")
    surface_material: PointerProperty(name="Surface Material",
                                      type=bpy.types.Material,
                                      update=_request_output_refresh)
    particle_material: PointerProperty(name="Particle Material",
                                       type=bpy.types.Material,
                                       update=_request_output_refresh)
    whitewater_material: PointerProperty(name="Whitewater Material",
                                         type=bpy.types.Material,
                                         update=_request_output_refresh)
    output_collection_name: StringProperty(
        name="Output Collection", default="FluidSim Output",
        update=_request_output_refresh)

    # -- Custom snippets ----------------------------------------------------------
    snippets: CollectionProperty(type=FLUIDSIM_PG_snippet)
    active_snippet: IntProperty(default=0)
    snippet_library_selection: EnumProperty(
        name="Library",
        description="Example and saved snippets that can be loaded into the stack",
        items=_snippet_library_items,
    )
    snippet_editor_lines: IntProperty(
        name="Editor Lines", default=24, min=4, max=200,
        description="Maximum source lines shown in the inline snippet editor")
    # Shortcut inputs (Houdini-wrangle-style quick insertion).
    shortcut_field: EnumProperty(
        name="Field", items=_SHORTCUT_FIELD_ITEMS, default="temperature",
        description="Solver field to bind as an argument and reference in code")
    shortcut_constant_name: StringProperty(
        name="Name", default="strength",
        description="Name of the constant to add to the snippet")
    shortcut_constant_value: FloatProperty(
        name="Value", default=1.0,
        description="Value of the constant to add to the snippet")

    # -- Diagnostics ---------------------------------------------------------------
    profiling_enabled: BoolProperty(name="Enable Profiling", default=False,
                                    update=_request_rebuild)
    profiling_sync: BoolProperty(
        name="Accurate GPU Timing", default=True,
        update=_request_rebuild,
        description="Synchronize CUDA around each phase for exact timings")
    show_overlay: BoolProperty(name="Viewport Overlay", default=True,
                               update=_request_output_refresh)
    report_path: StringProperty(
        name="Report Path", subtype="FILE_PATH", default="//fluidsim_report.json")
    benchmark_frames: IntProperty(name="Benchmark Frames", default=24, min=4, max=500)
    benchmark_baseline: StringProperty(
        name="Baseline", subtype="FILE_PATH", default="//fluidsim_benchmark.json")

    # -- Cache (optional, supplemental only) -----------------------------------------
    cache_enabled: BoolProperty(
        name="Enable Cache", default=False,
        description="Optionally record frames to disk during playback")
    cache_path: StringProperty(name="Cache Path", subtype="DIR_PATH",
                               default="//fluidsim_cache/")
    cache_compress: BoolProperty(
        name="Compress Cache", default=False,
        description="zlib-compress cache frames (smaller files, much slower "
                    "writes during playback)")
    cache_playback_enabled: BoolProperty(
        name="Cache Playback",
        default=False,
        update=_on_cache_playback_changed,
        description="Use cached frames as the timeline source during playback")
    cache_use_custom_range: BoolProperty(
        name="Custom Cache Range",
        default=False,
        description="Override the scene frame range for cache export/playback")
    cache_frame_start: IntProperty(
        name="Start",
        default=1,
        min=0,
        description="First frame for custom cache sequence operations")
    cache_frame_end: IntProperty(
        name="End",
        default=250,
        min=0,
        description="Last frame for custom cache sequence operations")

    # -- Status mirrors (read-only UI feedback) ----------------------------------------
    status: StringProperty(name="Status", default="Solver not initialized")
    stat_particles: IntProperty(default=0)
    stat_voxels: IntProperty(default=0)
    stat_whitewater: IntProperty(default=0)
    stat_step_ms: FloatProperty(default=0.0)
    stat_substeps: IntProperty(default=0)


CLASSES = (
    FLUIDSIM_PG_snippet_argument,
    FLUIDSIM_PG_snippet,
    FLUIDSIM_PG_object,
    FLUIDSIM_PG_settings,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fluidsim = PointerProperty(type=FLUIDSIM_PG_settings)
    bpy.types.Object.fluidsim = PointerProperty(type=FLUIDSIM_PG_object)


def unregister():
    if hasattr(bpy.types.Object, "fluidsim"):
        del bpy.types.Object.fluidsim
    if hasattr(bpy.types.Scene, "fluidsim"):
        del bpy.types.Scene.fluidsim
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            # Already unregistered (e.g. a partially-rolled-back register()):
            # unregister() must remain safe to call on a partial state.
            pass
