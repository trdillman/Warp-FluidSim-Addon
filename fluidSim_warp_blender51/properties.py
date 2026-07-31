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
    ("BOX", "Box", "Average nearby SDF samples for broad, inexpensive smoothing that can soften volume"),
    ("MEAN_CURVATURE_FLOW", "Mean Curvature", "Move the SDF along estimated curvature for rounder features with better shape preservation"),
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


def _request_live(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.LIVE)


def _request_sync(self, context):
    from .blender.scene_bridge import SceneUpdate

    _request_scene_update(context, SceneUpdate.SYNC)


def _request_particle_capacity(self, context):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        return
    try:
        from .blender import scene_bridge

        scene_bridge.update_particle_capacity_policy(scene)
    except Exception as exc:  # noqa: BLE001 - property callbacks must be safe
        scene.fluidsim.status = f"Particle capacity setting failed: {exc}"


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
        items=[("FIELD", "Field", "Bind a live solver field so the kernel reads or writes field data"),
               ("CONSTANT", "Constant", "Bind one fixed numeric value for every kernel invocation")],
        default="FIELD",
        description=(
            "Select whether this argument binds changing solver data or a fixed value; "
            "changing modes replaces the argument controls shown below"),
    )
    field: StringProperty(
        name="Field",
        description="Set the solver field this argument reads or writes (for example velocity_u, temperature, or pressure); changing it redirects the snippet to different simulation data",
        default="temperature",
    )
    constant_type: EnumProperty(
        name="Type", items=_items(CustomKernelConstantType), default="FLOAT")
    constant_name: StringProperty(
        name="Name", default="value",
        description="Set the identifier used for this constant in the snippet; changing it requires matching code references")
    constant_value: FloatProperty(
        name="Value", default=0.0,
        description="Set the numeric value supplied to every kernel invocation; higher or lower values affect the snippet wherever this constant is used")


class FLUIDSIM_PG_snippet(bpy.types.PropertyGroup):
    enabled: BoolProperty(
        name="Enabled", default=True,
        description="Enable to run this snippet in its selected solver phase; disable to keep it in the stack without executing it")
    kernel_name: StringProperty(
        name="Kernel Name", default="my_kernel",
        description="Set the generated kernel identifier; changing it renames the compiled snippet entry point")
    description: StringProperty(name="Description", default="")
    order: EnumProperty(
        name="Order", items=_items(CustomKernelOrder),
        default="BEFORE_FORCES",
        description="Choose the solver phase in which this snippet runs; changing it changes which simulation data the snippet sees and affects")
    run_over: EnumProperty(
        name="Run Over", items=_items(CustomKernelRunOver),
        default="VOXEL_LINEAR",
        description="Choose whether the kernel runs over sparse tiles, active voxels, or particles; changing it changes invocation count and available indexing")
    text_block: PointerProperty(
        name="Code", type=bpy.types.Text,
        description="Select the Text datablock containing the Warp snippet body; changing it replaces the code compiled and run by this stack entry",
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
            "Choose the analytic emission shape; changing it replaces the exact local Euclidean distance test, "
            "with the object's transform applied"
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
        default="ANALYTIC", update=_request_sync,
        description="Choose the geometry representation used to emit liquid; changing it replaces the source inputs and membership test")
    sink_region_kind: EnumProperty(
        name="Removal Region",
        items=[
            ("ANALYTIC", "Analytic Volume", "Remove liquid in an analytic region"),
            ("MESH_SOLID", "Mesh Solid Interior", "Remove liquid inside a watertight mesh"),
            ("MESH_SHELL", "Mesh Surface Shell", "Remove liquid in a mesh shell"),
            ("VOLUME_GRID", "Volume Grid Mask", "Remove liquid using an imported mask or SDF"),
        ],
        default="ANALYTIC", update=_request_sync,
        description="Choose the geometry representation used to remove liquid; changing it replaces the sink inputs and membership test")
    point_source_representation: EnumProperty(
        name="Point Representation",
        items=[
            ("DIRECT_APPEND", "Direct Append Once", "Append the owned input points once; not a volume fill"),
            ("RASTERIZED_REGION", "Rasterized Region", "Rasterize points into source leaves and top up occupancy"),
        ],
        default="DIRECT_APPEND", update=_request_reset,
        description="Choose whether owned points are appended once or rasterized into a refillable source region; changing it resets source lifecycle state")
    point_input_object: PointerProperty(
        name="Point Input", type=bpy.types.Object, poll=_point_input_poll,
        update=_request_sync,
        description="Select the mesh, point cloud, or particle-system object that supplies source points; changing it replaces the emitted point set")
    point_rasterize_radius: FloatProperty(
        name="Rasterize Radius", default=0.1, min=0.0, subtype="DISTANCE",
        update=_request_sync,
        description="Set the world-space radius used to turn each input point into a source region; higher values create broader, more connected emission and lower values create tighter regions")
    volume_input_object: PointerProperty(
        name="Volume Input", type=bpy.types.Object, poll=_volume_input_poll,
        update=_request_sync,
        description="Select the Blender Volume object that supplies named OpenVDB grids; changing it replaces the sampled volume data")
    volume_grid_type: EnumProperty(
        name="Volume Region", items=[
            ("FOG", "Fog / Mask", "Use a fog grid above the threshold"),
            ("SDF", "Signed Distance", "Use the inside of an SDF grid"),
        ], default="FOG", update=_request_sync,
        description="Choose whether volume membership comes from values above a fog threshold or negative signed distances; changing it changes how the selected grid is interpreted")
    volume_mask_grid: StringProperty(
        name="Fog / Mask Grid", default="density", update=_request_sync,
        description="Set the fog or mask grid name to sample; changing it selects different OpenVDB data and an invalid name prevents this volume source from compiling")
    volume_sdf_grid: StringProperty(
        name="SDF Grid", default="surface", update=_request_sync,
        description="Set the signed-distance grid name whose negative region is treated as inside; changing it selects different OpenVDB data")
    volume_scalar_grid: StringProperty(
        name="Scalar Grid", default="", update=_request_sync,
        description="Set the scalar OpenVDB grid sampled for a field source; changing the name selects a different payload and an invalid name fails closed")
    volume_vector_grid: StringProperty(
        name="Vector Grid", default="", update=_request_sync,
        description="Set the packed vector OpenVDB grid sampled for velocity; changing the name selects a different vector payload")
    volume_vector_x_grid: StringProperty(
        name="Vector X Grid", default="", update=_request_sync,
        description="Set the OpenVDB grid used for the X velocity component; changing it replaces the sourced X component")
    volume_vector_y_grid: StringProperty(
        name="Vector Y Grid", default="", update=_request_sync,
        description="Set the OpenVDB grid used for the Y velocity component; changing it replaces the sourced Y component")
    volume_vector_z_grid: StringProperty(
        name="Vector Z Grid", default="", update=_request_sync,
        description="Set the OpenVDB grid used for the Z velocity component; changing it replaces the sourced Z component")
    volume_fog_threshold: FloatProperty(
        name="Fog Threshold", default=0.0, update=_request_sync,
        description="Set the minimum fog or mask value included in the region; higher values shrink the source to denser areas and lower values include fainter areas")
    field_operation: EnumProperty(
        name="Target Field", items=[
            ("PRESSURE", "Pressure", "Source pressure before projection"),
            ("SCALAR", "Scalar / Temperature", "Source a scalar field after P2G"),
            ("VECTOR", "Vector Velocity", "Source a centered vector field after P2G"),
            ("STAGGERED_VELOCITY", "Staggered Velocity", "Source independent MAC face components"),
        ], default="PRESSURE", update=_request_sync,
        description="Choose which solver field receives this source; changing it changes the payload controls and the solver phase where the value is applied")
    field_region_kind: EnumProperty(
        name="Field Region", items=[
            ("ANALYTIC", "Analytic Region", "Use this object's analytic region"),
            ("VOLUME_GRID", "Volume Grid", "Use named values from a Blender Volume"),
        ], default="ANALYTIC", update=_request_sync,
        description="Choose whether the field source uses this object's analytic shape or a Volume grid; changing it replaces the region controls")
    field_value_source: EnumProperty(
        name="Payload Source", items=[
            ("CONSTANT", "Constant", "Use the constant value below"),
            ("VOLUME_GRID", "Volume Grid", "Sample the named scalar or vector grid"),
        ], default="CONSTANT", update=_request_sync,
        description="Choose a fixed value or values sampled from a Volume grid; changing it switches which payload controls are used")
    field_scalar_value: FloatProperty(
        name="Field Value", default=1.0, update=_request_sync,
        description="Set the constant scalar or pressure payload; higher and lower values source correspondingly larger and smaller field values")
    field_vector_value: FloatVectorProperty(
        name="Field Vector", size=3, default=(0.0, 0.0, 0.0),
        subtype="VELOCITY", update=_request_sync,
        description="Set the constant vector payload; changing a component changes the sourced velocity magnitude and direction on that axis")
    field_amount: FloatProperty(
        name="Blend Amount", default=1.0, min=0.0, update=_request_sync,
        description="Scale the field-source effect; higher values apply more of the payload, zero applies none, and values above one can amplify it")
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
        description="Choose an exact analytic shape or evaluated mesh triangles for the region; changing it replaces the geometry controls and membership method",
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
        description="Choose whether evaluated mesh geometry is captured once or refreshed every simulation frame; refreshing tracks animation but costs more CPU time",
    )
    mesh_surface_thickness: FloatProperty(
        name="Surface Band",
        default=0.0,
        min=0.0,
        subtype="DISTANCE",
        update=_request_sync,
        description="Set triangle-shell half-width in world units; higher values create a thicker emission or removal band, while zero uses the voxel-resolution bound",
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
        description="Choose whether a mesh sink removes liquid from its validated interior or only near its triangle surface; changing it changes the removal region",
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
            "Set target particle occupancy in each source voxel, not volume per second. "
            "Higher values create denser source liquid at greater capacity cost, while zero emits no particles"))
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
        description="Choose whether this source fills once after initialization or replenishes every substep; changing it resets the source lifecycle")
    initial_velocity: FloatVectorProperty(
        name="Initial Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY",
        update=_request_sync,
        description="Set the velocity assigned to emitted liquid; changing a component changes initial speed and direction on that axis")
    use_object_velocity: BoolProperty(
        name="Inherit Object Velocity", default=True,
        update=_request_sync,
        description="Enable to add velocity from animated object motion to emitted liquid; disable to use only Initial Velocity")
    blend_mode: EnumProperty(
        name="Blend", items=_items(BlendMode), default="REPLACE",
        update=_request_sync,
        description="Choose how source values combine with existing solver values; changing the mode can add, replace, clamp, pull toward, or blend the target")
    feather: FloatProperty(name="Feather", default=0.0, min=0.0,
                           update=_request_sync)
    temperature: FloatProperty(
        name="Temperature", default=0.0, update=_request_sync,
        description="Set the temperature attribute on emitted particles; higher or lower values change attribute-driven visualization and field operations")
    viscosity: FloatProperty(
        name="Viscosity", default=0.0, min=0.0, update=_request_sync,
        description="Set the viscosity attribute on emitted particles; higher values make those particles contribute thicker behavior when particle viscosity is enabled")
    density: FloatProperty(
        name="Density Attribute", default=1.0, min=0.0,
        description="Set the density attribute on emitted particles, not physical liquid mass; higher or lower values affect density-based visualization and ST-FLIP weighting",
        update=_request_sync)
    use_lifespan: BoolProperty(
        name="Use Lifespan", default=False, update=_request_sync,
        description="Enable to remove emitted particles after a randomized lifetime; disable to keep them until another sink or boundary removes them")
    life_min: FloatProperty(
        name="Life Min", default=2.0, min=0.0, update=_request_sync,
        description="Set the shortest emitted-particle lifetime in seconds; higher values keep every particle alive longer and lower values allow earlier removal")
    life_max: FloatProperty(
        name="Life Max", default=4.0, min=0.0, update=_request_sync,
        description="Set the longest emitted-particle lifetime in seconds; higher values allow some particles to persist longer and lower values shorten the range")
    force_type: EnumProperty(
        name="Force Type", items=_items(SourceShapeForceType), default="OFF",
        update=_request_sync,
        description="Choose the force pattern applied inside this region; changing it changes how strength and direction are interpreted")
    force_strength: FloatProperty(
        name="Force Strength", default=1.0, update=_request_sync,
        description="Scale the regional acceleration; larger absolute values produce stronger motion, zero disables the effect, and negative values reverse supported force directions")
    force_direction: FloatVectorProperty(
        name="Force Direction", size=3, default=(0.0, 0.0, 1.0),
        update=_request_sync,
        description="Set the directional-force vector or spin axis; changing components changes the force orientation and non-unit length can also scale directional magnitude")
    # Collider settings.
    collider_friction: FloatProperty(
        name="Friction", default=0.0, min=0.0, max=1.0,
        description="Set tangential velocity loss at this collider; higher values make liquid stick and slide less, while zero preserves tangential motion",
        update=_request_sync)
    collider_restitution: FloatProperty(
        name="Restitution", default=0.0, min=0.0, max=1.0,
        description="Set bounce at this collider; higher values return more normal velocity after impact, while zero produces no bounce",
        update=_request_sync)
    transfer_velocity: BoolProperty(
        name="Transfer Velocity", default=True,
        update=_request_sync,
        description="Enable to push liquid with animated collider motion; disable to keep collisions stationary even when the object moves")
    enabled: BoolProperty(
        name="Enabled", default=True, update=_request_sync,
        description="Enable this object to participate in simulation with its assigned role; disable to ignore it without deleting its settings")
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
            "Enable to inject this object's points whenever the solver initializes or resets, which supports playable seed-based scenes; "
            "disable to emit only through the normal source lifecycle"
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
        description="Enable to step the GPU solver as the timeline plays; disable to stop live stepping while keeping the current state")
    max_frame_catchup: IntProperty(
        name="Max Frame Catch-up", default=8, min=1, max=120,
        description=(
            "Set the largest forward timeline jump stepped during one frame change. "
            "Higher values recover more dropped playback frames but can stall Blender longer; lower values report desync sooner instead of silently running in slow motion"))
    domain_object: PointerProperty(
        name="Domain", type=bpy.types.Object,
        update=_request_rebuild,
        description="Select the object whose bounding box defines the simulation domain; changing it rebuilds grid bounds and capacity requirements")
    time_scale: FloatProperty(
        name="Time Scale", default=1.0, min=0.0, soft_max=4.0,
        update=_request_live,
        description="Scale simulated time per timeline frame; higher values advance liquid faster and may require more substeps, lower values create slow motion, and zero pauses physics")
    cfl: FloatProperty(
        name="CFL", default=1.0, min=0.1, soft_max=5.0,
        update=_request_live,
        description="Set the target Courant number used to size substeps; lower values use smaller, more stable steps at higher cost, while higher values use fewer steps with greater instability risk")
    min_substeps: IntProperty(
        name="Min Substeps", default=1, min=1, max=16,
        update=_request_live,
        description="Set the minimum solver steps taken per frame even when motion is slow; higher values improve temporal resolution but increase compute time")
    max_substeps: IntProperty(
        name="Max Substeps", default=4, min=1, max=32,
        update=_request_live,
        description="Cap adaptive solver steps per frame; higher values better satisfy CFL during fast motion but cost more, while lower values can clamp the time step and reduce accuracy")
    gravity: FloatVectorProperty(
        name="Gravity", size=3, default=(0.0, 0.0, -9.81), subtype="ACCELERATION",
        update=_request_live,
        description="Set custom acceleration by axis when Scene Gravity is off; larger component magnitudes pull liquid more strongly in their signed directions")
    use_scene_gravity: BoolProperty(
        name="Use Scene Gravity", default=True, update=_request_live,
        description="Enable to use the scene gravity vector; disable to use the custom Gravity value below")

    # -- Domain / sparse grid -------------------------------------------------
    voxel_size: FloatProperty(
        name="Voxel Size", default=0.05, min=0.005, soft_max=0.5, precision=4,
        update=_request_rebuild,
        description="Set simulation-grid cell size in world units; smaller values capture finer detail but allocate more leaves, particles, VRAM, and compute, while larger values simulate more coarsely and cheaply")
    leaf_capacity: IntProperty(
        name="Leaf Capacity", default=8192, min=64, max=262144,
        update=_request_rebuild,
        description="Set the maximum allocated 8x8x8 sparse leaves; higher values allow a larger active liquid region but reserve more VRAM, while lower values save memory and can exhaust capacity sooner")
    topology_expand: IntProperty(
        name="Topology Expand", default=1, min=0, max=4,
        update=_request_reset,
        description="Set sparse-leaf dilation passes around active fluid; higher values allocate a wider safety band and more VRAM, while zero keeps only directly activated topology")
    open_boundaries: BoolProperty(
        name="Open Boundaries", default=False,
        update=_request_reset,
        description="Enable to remove particles that leave the domain; disable to clamp them at domain limits instead")
    domain_sdf_walls: BoolProperty(
        name="Domain Collision Walls", default=True,
        update=_request_reset,
        description="Build collision SDF walls around the domain bounds "
                     "(recommended for closed domains; without this the "
                     "pressure solve treats sparse-grid padding beyond the "
                     "domain box as open space)")

    # -- Particles -------------------------------------------------------------
    max_particles: IntProperty(
        name="Initial Particle Reserve", default=2_000_000, min=1024,
        max=50_000_000, update=_request_particle_capacity,
        description=(
            "Set the particle capacity reserved when a solver starts. With Auto Grow enabled, "
            "the live GPU pool can grow transactionally without rewriting this authored value; "
            "an active pool never shrinks automatically"
        ))
    auto_grow_particles: BoolProperty(
        name="Auto Grow Particle Pool", default=True,
        update=_request_particle_capacity,
        description=(
            "Grow GPU particle and bin buffers transactionally when an admitted source needs more capacity; "
            "the existing solver remains authoritative if the live VRAM guard or allocation refuses growth"
        ))
    particle_growth_failure_policy: EnumProperty(
        name="If Growth Cannot Proceed",
        items=(
            ("DEFER", "Continue and Defer Emission",
             "Keep playback running, leave the source armed, and retry its whole batch later"),
            ("PAUSE", "Pause Live, Preserve Solver",
             "Turn Live off after the current safe step while preserving the initialized solver and all particles"),
        ),
        default="DEFER", update=_request_particle_capacity,
        description="Choose how a source behaves when guarded particle growth cannot allocate safely")
    capacity_show_alternatives: BoolProperty(
        name="Show Capacity Alternatives", default=False,
        options={"SKIP_SAVE"},
        description="Expand lower-priority resolution and domain alternatives")
    particles_per_voxel: IntProperty(
        name="Particles / Voxel", default=8, min=1, max=64,
        update=_request_live,
        description="Set the target initial particle density in occupied cells; higher values sample liquid more densely and cost more memory and transfer work, while lower values are lighter but noisier")
    sample_type: EnumProperty(
        name="Transfer", items=_items(ParticleSampleType), default="FLIP",
        update=_request_live,
        description="Choose the particle/grid velocity-transfer method; PIC is smoother and more dissipative, while FLIP preserves motion and can be noisier")
    flip_ratio: FloatProperty(
        name="FLIP Ratio", default=0.97, min=0.0, max=1.0,
        update=_request_live,
        description="Blend PIC and FLIP velocity updates; increase toward 1 for more energetic, less dissipative motion, or decrease toward 0 for smoother, more damped motion")
    trace_mode: EnumProperty(
        name="Trace", items=_items(TraceMode), default="RK3",
        update=_request_live,
        description="Choose the trajectory integrator used for particle advection; higher-order modes improve curved-path accuracy but require more velocity samples")
    relax_enabled: BoolProperty(
        name="Relax Particles", default=True, update=_request_live,
        description="Enable to redistribute particles away from clumps each step; disable to preserve raw advection at lower cost but allow uneven sampling")
    relax_strength: FloatProperty(
        name="Relax Strength", default=0.5, min=0.0, max=1.0,
        update=_request_live,
        description="Scale particle redistribution; higher values separate clumps more aggressively, lower values preserve more original positions, and zero has no relaxation effect")
    relax_mode: EnumProperty(
        name="Relax Mode", items=_items(RelaxMode),
        default="LOCAL_VOXEL", update=_request_live,
        description="Choose whether relaxation compares particles only in their voxel or also in neighboring voxels; neighbor search is smoother but costs more")
    max_speed: FloatProperty(
        name="Max Speed", default=0.0, min=0.0,
        update=_request_live,
        description="Set the particle speed limit in meters per second; lower positive values clamp motion more strongly, higher values allow faster motion, and zero disables limiting")
    limit_behavior: EnumProperty(
        name="Limit Behavior", items=_items(LimitBehavior), default="MAINTAIN",
        update=_request_live,
        description="Choose how velocities above Max Speed are handled; changing the mode preserves, scales, or acceleration-limits excess motion")
    limit_percent: FloatProperty(
        name="Limit Percent", default=0.5, min=0.0, max=1.0,
        update=_request_live,
        description="Set the fraction of speed above Max Speed removed each step; higher values damp excess speed more strongly, while lower values retain more of it")
    # -- Global forces ---------------------------------------------------------
    directional_force: FloatVectorProperty(
        name="Directional Acceleration", size=3, default=(0.0, 0.0, 0.0),
        subtype="ACCELERATION", update=_request_live,
        description="Set constant acceleration applied to every valid velocity face; larger component magnitudes drive stronger motion in their signed axis directions")
    omnidirectional_force_center: FloatVectorProperty(
        name="Center", size=3, default=(0.0, 0.0, 0.0),
        update=_request_live,
        description="Set the world-space origin of the radial global force; changing a component moves the attraction or repulsion center along that axis")
    omnidirectional_force_strength: FloatProperty(
        name="Radial Acceleration", default=0.0, update=_request_live,
        description="Set radial acceleration; larger positive values repel more strongly, larger negative magnitudes attract more strongly, and zero disables the force")
    omnidirectional_force_radius: FloatProperty(
        name="Radius", default=1.0, min=0.001, subtype="DISTANCE",
        update=_request_live,
        description="Set the radial force's falloff distance; higher values affect a larger region more gradually, while lower values concentrate the force near its center")
    force_diffusion: FloatProperty(
        name="Velocity Diffusion", default=0.0, min=0.0, max=1.0,
        update=_request_live,
        description="Blend MAC velocity toward its six-neighbor average; higher values smooth and damp velocity more strongly, while zero leaves velocity unchanged")
    dissipation_enabled: BoolProperty(
        name="Dissipate Particle Attribute", default=False,
        update=_request_live,
        description="Enable to exponentially move the selected particle attribute toward Target over time; disable to leave it unchanged by dissipation")
    dissipation_attribute: EnumProperty(
        name="Attribute",
        items=[
            ("temperature", "Temperature", "Decay particle temperature toward the target value"),
            ("viscosity", "Viscosity", "Decay particle viscosity toward the target value"),
            ("density", "Density", "Decay particle density toward the target value"),
        ],
        default="temperature", update=_request_live,
        description="Choose which particle attribute Rate moves toward Target; changing it leaves the other attributes unaffected")
    dissipation_rate: FloatProperty(
        name="Rate", default=0.0, min=0.0, subtype="FACTOR",
        update=_request_live,
        description="Set exponential attribute decay per second; higher values approach Target faster, while zero prevents decay")
    dissipation_target: FloatProperty(
        name="Target", default=0.0, update=_request_live,
        description="Set the value approached by the selected particle attribute; raising or lowering it changes the endpoint, while Rate controls how quickly particles get there")
    attraction_enabled: BoolProperty(
        name="Attract to Collision SDF", default=False,
        update=_request_live,
        description="Enable to accelerate liquid toward colliders and closed domain walls; disable to apply no collision-surface attraction")
    attraction_strength: FloatProperty(
        name="Attraction Strength", default=0.0, update=_request_live,
        description="Set acceleration toward collision surfaces; higher positive values pull liquid toward surfaces more strongly, zero adds no attraction, and negative values repel")
    attraction_max_distance: FloatProperty(
        name="Attraction Distance", default=1.0, min=0.001,
        subtype="DISTANCE", update=_request_live,
        description="Set the maximum distance from collision surfaces affected by attraction; higher values pull from farther away, while lower values confine the effect near surfaces")
    count_stiffness: FloatProperty(
        name="Count Stiffness", default=0.0, min=0.0, soft_max=4.0,
        update=_request_live,
        description="Scale anti-clumping pressure derived from particle counts; higher values separate crowded particles more strongly, while zero disables this correction")

    # -- Spatiotemporal FLIP (beta) -------------------------------------------
    spacetime_enabled: BoolProperty(
        name="Enable ST-FLIP (Beta)", default=False, update=_request_reset,
        description="Enable the experimental spatiotemporal FLIP path for large time steps; disable to use the standard FLIP path; changing this resets the solver")
    spacetime_jitter_strength: FloatProperty(
        name="Jitter", default=1.0, min=0.0, max=2.0, update=_request_live,
        description="Scale ST-FLIP temporal sampling jitter; higher values spread samples more strongly, lower values reduce variation, and zero disables jitter")
    spacetime_adaptive_jitter: BoolProperty(
        name="Adaptive Jitter", default=True, update=_request_live,
        description="Enable to reduce or expand jitter from local flow conditions; disable to use the fixed Jitter value everywhere")
    spacetime_phase_eta: FloatProperty(
        name="Phase Eta", default=0.5, min=0.01, soft_max=2.0, update=_request_live,
        description="Set the ST-FLIP phase-correction response; higher values correct phase differences more aggressively, while lower values preserve more temporal variation")
    spacetime_phase_threshold: FloatProperty(
        name="Phase Threshold", default=0.5, min=0.0, max=1.0,
        update=_request_live,
        description="Set the phase difference that triggers correction; higher values correct only larger mismatches, while lower values correct more particles")
    spacetime_reference_mass: FloatProperty(
        name="Reference Mass", default=0.0, min=0.0, soft_max=64.0,
        update=_request_reset,
        description="Override the ST-FLIP reference particle mass; higher values make each reference sample heavier, while zero derives mass from liquid density and voxel sampling")
    spacetime_variable_projection: BoolProperty(
        name="Variable Projection (Beta)", default=True, update=_request_live,
        description="Enable density-weighted variable-coefficient pressure projection for ST-FLIP; disable to use the standard constant-coefficient projection")
    spacetime_liquid_density: FloatProperty(
        name="Liquid Density", default=1.0, min=1.0e-6, soft_max=10.0,
        update=_request_live,
        description="Set liquid density used by variable projection; higher values increase liquid inertia relative to gas, while lower values reduce the contrast")
    spacetime_gas_density: FloatProperty(
        name="Gas Density", default=0.0, min=0.0, soft_max=1.0,
        update=_request_live,
        description="Set background gas density for variable projection; higher values strengthen gas coupling, while zero treats gas as negligible")
    spacetime_min_density: FloatProperty(
        name="Min Density", default=0.05, min=1.0e-6, soft_max=1.0,
        update=_request_live,
        description="Clamp the minimum density used by variable projection; higher values improve conditioning but blur density contrast, while lower values preserve contrast with greater numerical risk")
    spacetime_local_cfl: FloatProperty(
        name="Local CFL", default=1.0, min=0.0, soft_max=4.0,
        update=_request_live,
        description="Set the local ST-FLIP travel limit per substep; lower values create more local substeps for accuracy, while higher values permit larger, cheaper steps")
    spacetime_max_local_substeps: IntProperty(
        name="Max Local Substeps", default=64, min=1, max=256,
        update=_request_live,
        description="Cap ST-FLIP local temporal subdivisions; higher values handle faster motion more accurately at greater cost, while lower values can clamp local integration")
    spacetime_resync_surface: BoolProperty(
        name="Resync Surface", default=True, update=_request_surface_refresh,
        description="Enable to rebuild surface state after ST-FLIP temporal resynchronization; disable to save work but allow visible surface lag")
    spacetime_seed: IntProperty(
        name="Seed", default=1729, min=0, max=2147483647, update=_request_reset,
        description="Set the deterministic ST-FLIP jitter sequence; changing it produces a different sampling pattern without changing overall strength")

    # -- Pressure ---------------------------------------------------------------
    pressure_solver: EnumProperty(
        name="Solver", items=_items(ProjectNonDivergentSolver), default="PCG",
        update=_request_live,
        description="Choose the linear solver for incompressibility; PCG usually converges faster, while Jacobi is simpler but often needs more iterations")
    pressure_iterations: IntProperty(
        name="Iterations", default=200, min=1, max=4000,
        update=_request_live,
        description="Cap pressure-solver iterations; higher values can remove more divergence at greater cost, while lower values finish faster but may leave compressibility")
    pressure_tolerance: FloatProperty(
        name="Tolerance", default=1.0e-5, min=0.0, precision=8,
        update=_request_live,
        description="Set the pressure residual required for early convergence; lower values demand a more accurate solve and may take longer, while higher values stop sooner")
    pressure_threshold: FloatProperty(
        name="Threshold", default=0.0, min=0.0, max=1.0,
        update=_request_live,
        description="Remove solved pressures below this fraction of the frame's pressure range; higher values suppress more small pressure detail, while zero keeps all solved values")

    # -- Viscosity ---------------------------------------------------------------
    viscosity_enabled: BoolProperty(
        name="Enable Viscosity", default=False, update=_request_live,
        description="Enable the implicit viscosity solve to make liquid resist shear; disable it for inviscid motion and lower compute cost")
    viscosity_solver: EnumProperty(
        name="Solver", items=_items(ViscositySolver), default="PCG",
        update=_request_live,
        description="Choose the linear solver for viscosity diffusion; PCG usually converges faster, while Jacobi is simpler but often needs more iterations")
    viscosity_constant: FloatProperty(
        name="Viscosity", default=0.05, min=0.0, soft_max=100.0,
        update=_request_live,
        description="Set constant kinematic viscosity when From Particles is off; higher values produce thicker, slower-spreading liquid, while zero adds no viscous resistance")
    viscosity_from_particles: BoolProperty(
        name="From Particles", default=False,
        update=_request_live,
        description="Enable to derive viscosity from each particle attribute; disable to apply the same constant Viscosity value to all liquid")
    viscosity_iterations: IntProperty(
        name="Iterations", default=40, min=1, max=1000,
        update=_request_live,
        description="Cap viscosity-solver iterations; higher values can converge thick fluids more accurately at greater cost, while lower values finish faster with more residual error")
    viscosity_tolerance: FloatProperty(
        name="Tolerance", default=1.0e-4, min=0.0, precision=8,
        update=_request_live,
        description="Set the viscosity residual required for early convergence; lower values demand a tighter solve and may take longer, while higher values stop sooner")

    # -- Surface -------------------------------------------------------------------
    surface_enabled: BoolProperty(
        name="Generate Surface", default=True,
        update=_request_surface_refresh,
        description="Enable to build a marching-cubes mesh from liquid particles; disable to skip surface extraction and its GPU, CPU, and viewport cost")
    particle_radius_scale: FloatProperty(
        name="Particle Radius", default=1.2, min=0.5, soft_max=4.0,
        update=_request_surface_refresh,
        description="Scale each particle's sphere radius when building the surface SDF; higher values join gaps and create a fuller, thicker surface, while lower values reveal smaller features and can create holes")
    surface_splat_radius: IntProperty(
        name="Splat Radius", default=2, min=1, max=16,
        update=_request_surface_refresh,
        description=(
            "Set the minimum voxel neighborhood stamped by each particle into the surface SDF. "
            "Higher values widen and smooth particle influence but increase GPU work; lower values are sharper and cheaper. "
            "Particle size may require a larger effective radius than this minimum"))
    surface_resolution_mode: EnumProperty(
        name="Resolution Mode", items=_items(SurfaceResolutionMode),
        default="SIMULATION", update=_request_surface_refresh,
        description="Choose whether surface resolution follows the simulation grid, a multiplier, or an explicit voxel size; changing it can rebuild the surface grid and alter detail, VRAM, and extraction time")
    surface_resolution_multiplier: FloatProperty(
        name="Resolution Multiplier", default=2.0, min=0.25, soft_max=4.0, max=8.0,
        update=_request_surface_refresh,
        description="Scale surface-grid density relative to the simulation grid; higher values use smaller surface voxels for finer detail with more VRAM and extraction cost, while lower values are coarser and cheaper; 2 means half-size voxels")
    surface_voxel_size: FloatProperty(
        name="Surface Voxel Size", default=0.0, min=0.0, soft_max=1.0,
        precision=5, update=_request_surface_refresh,
        description="Set explicit surface-grid cell size in Custom mode; smaller values produce finer meshes with more VRAM and extraction work, while larger values are coarser and cheaper; zero falls back to simulation voxel size")
    surface_leaf_capacity_scale: FloatProperty(
        name="Leaf Capacity Scale", default=8.0, min=1.0, soft_max=16.0, max=128.0,
        update=_request_surface_refresh,
        description=(
            "Multiply the simulation leaf budget for a separate render-surface sparse grid. "
            "Higher values cover more fine-resolution liquid but reserve more VRAM; lower values save memory but may exhaust surface leaves. "
            "This matters when surface and simulation voxel sizes differ"))
    surface_max_dense_dim: IntProperty(
        name="Max Dense Dim", default=512, min=32, max=2048,
        update=_request_surface_refresh,
        description=(
            "Cap the marching-cubes dense extraction span on any axis, in surface voxels. "
            "Higher values allow larger active surfaces but can sharply increase VRAM and extraction time; lower values fail surface extraction sooner when the liquid span exceeds the cap"))
    surface_update_cadence: EnumProperty(
        name="Update Cadence", items=_items(SurfaceUpdateCadence),
        default="FRAME", update=_request_surface_refresh,
        description="Choose when the expensive render surface rebuilds; more frequent updates track substeps better but cost more, while manual updates are cheapest and can appear stale")
    live_surface_publish_cadence: IntProperty(
        name="Live Surface Every", default=4, min=1, max=8,
        update=_request_surface_refresh,
        description=("Set how often the Blender viewport mesh publishes during live playback; "
                     "higher values publish less often for better responsiveness, while 1 publishes every frame; physics, cache, and exports still evaluate every frame"))
    surface_build_lowres_sdf: BoolProperty(
        name="Keep Sim Surface Field", default=True,
        update=_request_surface_refresh,
        description="Enable to retain the simulation-resolution SDF alongside a separate render surface; whitewater always rebuilds this solver dependency even when this render-only preference is disabled")
    surface_filter_type: EnumProperty(
        name="Filter", items=_items(SdfFilterType), default="BOX",
        update=_request_surface_refresh,
        description="Choose the SDF operation used by Filter Iterations; changing it switches between expansion, contraction, box smoothing, and curvature smoothing")
    surface_filter_iterations: IntProperty(
        name="Filter Iterations", default=0, min=0, max=16,
        update=_request_surface_refresh,
        description="Set passes of the selected SDF filter; higher values strengthen and widen the effect but cost more, while zero skips this stage")
    surface_filter_strength: FloatProperty(
        name="Filter Strength", default=0.7, min=0.0, max=1.0,
        update=_request_surface_refresh,
        description="Blend each general filter pass with the original SDF; higher values apply the selected filter more strongly, while zero leaves the field unchanged")
    surface_dilate_iterations: IntProperty(
        name="Dilate Iterations", default=0, min=0, max=16,
        update=_request_surface_refresh,
        description="Set dedicated surface-expansion passes; higher values thicken and connect the liquid surface at greater cost, while zero skips dilation")
    surface_dilate_strength: FloatProperty(
        name="Dilate Strength", default=1.0, min=0.0, max=1.0,
        update=_request_surface_refresh,
        description="Scale each dilation pass; higher values expand the surface more per pass, while zero prevents expansion")
    surface_erode_iterations: IntProperty(
        name="Erode Iterations", default=0, min=0, max=16,
        update=_request_surface_refresh,
        description="Set dedicated surface-contraction passes; higher values thin the liquid and remove small features at greater cost, while zero skips erosion")
    surface_erode_strength: FloatProperty(
        name="Erode Strength", default=1.0, min=0.0, max=1.0,
        update=_request_surface_refresh,
        description="Scale each erosion pass; higher values contract the surface more per pass, while zero prevents contraction")
    surface_smooth_type: EnumProperty(
        name="Smooth Type", items=SURFACE_SMOOTH_ITEMS,
        default="BOX", update=_request_surface_refresh,
        description="Choose box averaging or curvature-flow smoothing; changing it alters how details are softened and how strongly volume is preserved")
    surface_smooth_iterations: IntProperty(
        name="Smooth Iterations", default=1, min=0, max=16,
        update=_request_surface_refresh,
        description="Set dedicated smoothing passes; higher values remove more small-scale detail and cost more, while zero leaves the post-filter surface unsmoothed")
    surface_smooth_strength: FloatProperty(
        name="Smooth Strength", default=0.7, min=0.0, max=1.0,
        update=_request_surface_refresh,
        description="Blend each smoothing pass with the current SDF; higher values soften the surface more, while zero prevents smoothing")

    # -- Whitewater -------------------------------------------------------------
    whitewater_enabled: BoolProperty(
        name="Enable Whitewater", default=False,
        update=_request_whitewater_refresh,
        description="Enable secondary bubble and spray particles from liquid churn; disable to skip their emission, simulation, memory, and display cost")
    max_whitewater: IntProperty(
        name="Max Whitewater", default=1_000_000, min=1024, max=20_000_000,
        update=_request_rebuild,
        description="Set the GPU capacity for whitewater particles; higher values allow more bubbles and spray but reserve more VRAM, while lower values cap emission sooner")
    ww_churn_decay: FloatProperty(
        name="Churn Decay", default=2.0, min=0.0, update=_request_live,
        description="Set how quickly stored churn fades per second; higher values shorten emission after motion subsides, while zero preserves churn")
    ww_churn_gain: FloatProperty(
        name="Churn Gain", default=1.0, min=0.0, update=_request_live,
        description="Scale churn generated by turbulent liquid motion; higher values create stronger whitewater triggers, while zero prevents new churn")
    ww_churn_threshold: FloatProperty(
        name="Churn Threshold", default=0.5, min=0.0, update=_request_live,
        description="Set the churn level required for whitewater emission; higher values restrict emission to stronger turbulence, while lower values emit more readily")
    ww_emission_rate: FloatProperty(
        name="Emission Rate", default=2.0, min=0.0, update=_request_live,
        description="Scale whitewater particles spawned from eligible liquid; higher values produce denser bubbles and spray at greater cost, while zero emits none")
    ww_min_speed: FloatProperty(
        name="Min Speed", default=0.5, min=0.0, update=_request_live,
        description="Set the minimum liquid speed eligible for whitewater emission; higher values restrict emission to faster motion, while lower values include calmer flow")
    ww_life_min: FloatProperty(
        name="Life Min", default=1.0, min=0.0, update=_request_live,
        description="Set the shortest whitewater lifetime in seconds; higher values keep every whitewater particle visible longer and increase occupancy")
    ww_life_max: FloatProperty(
        name="Life Max", default=3.0, min=0.0, update=_request_live,
        description="Set the longest whitewater lifetime in seconds; higher values allow some particles to persist longer and increase occupancy")
    ww_bubble_drag: FloatProperty(
        name="Bubble Drag", default=4.0, min=0.0, update=_request_live,
        description="Set how quickly bubbles follow surrounding liquid velocity; higher values couple bubbles more strongly, while zero lets them retain their own motion")
    ww_spray_drag: FloatProperty(
        name="Spray Drag", default=0.2, min=0.0, update=_request_live,
        description="Set aerodynamic damping on spray velocity; higher values slow spray more quickly, while zero preserves ballistic motion")

    # -- Height source ------------------------------------------------------------
    height_enabled: BoolProperty(
        name="Height Fill", default=False,
        update=_request_reset,
        description="Enable to fill the domain below the configured height when the solver resets; disable to omit this height source")
    height_base: FloatProperty(
        name="Base Height", default=0.3, update=_request_reset,
        description="Set the world-space base fill level; higher values fill more of the domain vertically, while lower values fill less")
    height_back_mode: EnumProperty(
        name="Back Mode", items=_items(SourceHeightBackMode), default="FLAT",
        update=_request_reset,
        description="Choose how the height source continues behind its profile; changing it switches between infinite, flat, and image-profiled back surfaces")
    height_profile_image: PointerProperty(
        name="Height Profile", type=bpy.types.Image, update=_request_reset,
        description="Select a grayscale image sampled across the domain as a height offset; changing it replaces the fill profile, while an empty image produces no image variation")
    height_profile_scale: FloatProperty(
        name="Profile Scale", default=1.0, update=_request_reset,
        description="Scale grayscale height offsets from the profile image; larger absolute values create taller variation, zero flattens the profile, and negative values invert it")
    height_emission_mode: EnumProperty(
        name="Fill Lifecycle", items=_items(SourceEmissionMode),
        default=SourceEmissionMode.INITIAL_FILL.name,
        update=_request_reset,
        description="Choose whether the height region fills once after initialization or is replenished continuously; changing it resets lifecycle state")
    height_error: StringProperty(name="Height Fill Error", default="")
    height_velocity: FloatVectorProperty(
        name="Velocity", size=3, default=(0.0, 0.0, 0.0),
        update=_request_reset,
        description="Set initial velocity for height-fill particles; changing components changes their starting speed and direction")
    height_emit_particles: BoolProperty(
        name="Emit Particles", default=True, update=_request_reset,
        description="Enable to seed liquid particles in the height region; disable to keep the height configuration without creating particles")

    # -- Output ---------------------------------------------------------------------
    liquid_view_mode: EnumProperty(
        name="Liquid View",
        items=LIQUID_VIEW_ITEMS,
        default="SURFACE",
        update=_apply_liquid_view_mode,
        description="Choose whether the viewport shows the liquid surface, particles, or both; changing it updates the two visibility toggles below",
    )
    show_surface: BoolProperty(
        name="Surface Mesh", default=True, update=_sync_liquid_view_mode,
        description="Enable to display and publish the generated liquid mesh; disable to hide the mesh while simulation continues")
    show_particles: BoolProperty(
        name="Particles", default=False, update=_sync_liquid_view_mode,
        description="Enable to display liquid particles in the viewport; disable to hide them while simulation continues")
    show_whitewater: BoolProperty(
        name="Whitewater", default=True, update=_request_output_refresh,
        description="Enable to display simulated whitewater particles; disable to hide them without disabling whitewater simulation")
    particle_display_fraction: FloatProperty(
        name="Display Fraction", default=1.0, min=0.01, max=1.0,
        update=_request_output_refresh,
        description="Set the fraction of liquid particles shown in the viewport; higher values show a denser, more representative cloud at greater display cost, while lower values improve responsiveness")
    particle_display_scale: FloatProperty(
        name="Display Scale", default=1.0, min=0.05, soft_max=4.0,
        update=_request_output_refresh,
        description="Scale liquid-particle viewport point radius; higher values look larger and denser, while lower values reveal separation without changing simulation particles")
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
        description="Scale whitewater viewport point radius; higher values make bubbles and spray look larger, while lower values reveal separation without changing simulation")
    surface_material: PointerProperty(
        name="Surface Material", type=bpy.types.Material,
        update=_request_output_refresh,
        description="Select the material assigned to generated liquid surfaces; changing it updates current and future output meshes")
    particle_material: PointerProperty(
        name="Particle Material", type=bpy.types.Material,
        update=_request_output_refresh,
        description="Select the material used for liquid-particle display; changing it updates the viewport particle output")
    whitewater_material: PointerProperty(
        name="Whitewater Material", type=bpy.types.Material,
        update=_request_output_refresh,
        description="Select the material used for whitewater display; changing it updates bubble and spray output")
    output_collection_name: StringProperty(
        name="Output Collection", default="FluidSim Output",
        update=_request_output_refresh,
        description="Set the Blender collection name that owns generated outputs; changing it moves or creates outputs under the new collection")

    # -- Custom snippets ----------------------------------------------------------
    snippets: CollectionProperty(type=FLUIDSIM_PG_snippet)
    active_snippet: IntProperty(default=0)
    snippet_library_selection: EnumProperty(
        name="Library",
        description="Select an example or saved snippet to load into the stack; changing the selection changes which preset the load action will insert",
        items=_snippet_library_items,
    )
    snippet_editor_lines: IntProperty(
        name="Editor Lines", default=24, min=4, max=200,
        description="Set the maximum source lines shown in the inline snippet editor; higher values show more code but make the panel taller, while lower values keep it compact")
    # Shortcut inputs (Houdini-wrangle-style quick insertion).
    shortcut_field: EnumProperty(
        name="Field", items=_SHORTCUT_FIELD_ITEMS, default="temperature",
        description="Choose the solver field that the shortcut action binds and references in code; changing it redirects the inserted snippet line")
    shortcut_constant_name: StringProperty(
        name="Name", default="strength",
        description="Set the identifier for the constant inserted into the snippet; changing it changes the generated argument and code reference")
    shortcut_constant_value: FloatProperty(
        name="Value", default=1.0,
        description="Set the numeric value for the inserted constant; higher or lower values affect the generated snippet wherever that argument is used")

    # -- Diagnostics ---------------------------------------------------------------
    profiling_enabled: BoolProperty(
        name="Enable Profiling", default=False, update=_request_rebuild,
        description="Enable per-phase solver timing collection for diagnostics; disable to avoid synchronization and measurement overhead")
    profiling_sync: BoolProperty(
        name="Accurate GPU Timing", default=True,
        update=_request_rebuild,
        description="Enable CUDA synchronization around each phase for accurate timings at higher profiling cost; disable for lower overhead but less precise asynchronous measurements")
    show_overlay: BoolProperty(
        name="Viewport Overlay", default=True,
        update=_request_output_refresh,
        description="Enable to show FluidSim status and statistics in the viewport; disable to hide the overlay without stopping simulation")
    report_path: StringProperty(
        name="Report Path", subtype="FILE_PATH", default="//fluidsim_report.json",
        description="Set where validation and profiling JSON is written; changing it sends the next report to a different file")
    benchmark_frames: IntProperty(
        name="Benchmark Frames", default=24, min=4, max=500,
        description="Set the number of simulation frames measured by a benchmark; higher values improve averaging but take longer, while lower values finish faster with noisier results")
    benchmark_baseline: StringProperty(
        name="Baseline", subtype="FILE_PATH", default="//fluidsim_benchmark.json",
        description="Select the JSON baseline used for performance comparison; changing it changes which recorded timings determine regressions")

    # -- Cache (optional, supplemental only) -----------------------------------------
    cache_enabled: BoolProperty(
        name="Enable Cache", default=False,
        description="Enable to record simulated frames to Cache Path during playback; disable to run without writing cache files")
    cache_path: StringProperty(
        name="Cache Path", subtype="DIR_PATH", default="//fluidsim_cache/",
        description="Set the directory used to write and read frame caches; changing it switches the cache sequence available for recording and playback")
    cache_compress: BoolProperty(
        name="Compress Cache", default=False,
        description="zlib-compress cache frames (smaller files, much slower "
                    "writes during playback)")
    cache_playback_enabled: BoolProperty(
        name="Cache Playback",
        default=False,
        update=_on_cache_playback_changed,
        description="Enable to read cached frames instead of stepping live simulation during playback; disable to return the timeline to the live solver")
    cache_use_custom_range: BoolProperty(
        name="Custom Cache Range",
        default=False,
        description="Enable to use the custom Start and End frames for cache operations; disable to use the scene frame range")
    cache_frame_start: IntProperty(
        name="Start",
        default=1,
        min=0,
        description="Set the first frame included in custom cache operations; higher values skip more early frames and values above End produce an invalid range")
    cache_frame_end: IntProperty(
        name="End",
        default=250,
        min=0,
        description="Set the last frame included in custom cache operations; higher values extend the cache sequence and values below Start produce an invalid range")

    # -- Status mirrors (read-only UI feedback) ----------------------------------------
    status: StringProperty(name="Status", default="Solver not initialized")
    stat_particles: IntProperty(default=0)
    stat_voxels: IntProperty(default=0)
    stat_whitewater: IntProperty(default=0)
    stat_whitewater_required: IntProperty(default=0)
    stat_whitewater_admitted: IntProperty(default=0)
    stat_whitewater_dropped: IntProperty(default=0)
    stat_whitewater_policy: StringProperty(default="FIXED_CAPACITY_DETERMINISTIC_DROPS")
    stat_whitewater_status: StringProperty(default="OK")
    stat_step_ms: FloatProperty(default=0.0)
    stat_substeps: IntProperty(default=0)
    stat_cfl_clamped: BoolProperty(default=False)


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
