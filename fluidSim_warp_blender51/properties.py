"""Blender property groups for the FluidSim Warp add-on.

Domain-level settings (on the domain object) drive the solver; scene-level
settings hold interface, playback, cache and output state; object-level
settings tag scene objects as sources / colliders / sinks. Property definitions import only
pure enum and snippet metadata so registration never touches Warp/CUDA.
"""

from __future__ import annotations


import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .sim import constants as _C
from .sim.enums import (
    BlendMode,
    LimitBehavior,
    MeshEvaluationMode,
    MeshSinkRegionMode,
    ParticleSampleType,
    ProjectNonDivergentSolver,
    RelaxMode,
    SdfFilterType,
    SurfaceResolutionMode,
    SurfaceUpdateCadence,
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
    ("HIDDEN", "Hidden", "Hide both the liquid surface and liquid particles"),
    ("SURFACE", "Surface", "Show the marching-cubes liquid surface"),
    ("PARTICLES", "Particles", "Show liquid as particles only"),
    ("SURFACE_PARTICLES", "Surface + Particles", "Show both surface mesh and liquid particles"),
)


LIQUID_ATTR_ITEMS = (
    ("NONE", "Material", "Use the assigned material color"),
    ("COLOR", "Source Color", "Display the persistent source colors and their mixed result"),
    ("HEIGHT", "Height", "Color by world-space height"),
    ("SPEED", "Speed", "Color by liquid velocity magnitude"),
    ("TEMPERATURE", "Temperature", "Color by particle temperature attribute"),
    ("VISCOSITY", "Viscosity", "Color by particle viscosity attribute"),
    ("DENSITY", "Density", "Color by particle density attribute"),
)


PARTICLE_ATTR_ITEMS = LIQUID_ATTR_ITEMS + (
    ("AGE", "Age", "Color by particle age"),
    ("LIFE", "Life", "Color by particle life span"),
    ("TIME_OFFSET", "Time Offset", "Color by the per-particle time offset (always zero)"),
)


WHITEWATER_ATTR_ITEMS = (
    ("NONE", "Material", "Use the assigned material color"),
    ("HEIGHT", "Height", "Color by world-space height"),
    ("SPEED", "Speed", "Color by whitewater velocity magnitude"),
    ("AGE", "Age", "Color by whitewater age"),
    ("LIFE", "Life", "Color by whitewater life span"),
    ("TYPE", "Type", "Color by persistent foam, bubble, and spray type"),
)


WHITEWATER_OUTPUT_ITEMS = (
    ("COMBINED", "Combined", "Publish one fast whitewater point pool with a type attribute"),
    ("SEPARATE", "Separate", "Publish independently materialized foam, bubble, and spray outputs"),
    ("BOTH", "Combined + Separate", "Publish the combined preview pool and all three typed outputs"),
)


# Saved files store the item number (0, 1), not SdfFilterType's value; the
# solver converts by identifier, so these numbers must never change.
SURFACE_SMOOTH_ITEMS = (
    ("BOX", "Box", "Average nearby SDF samples for broad, inexpensive smoothing that can soften volume", 0),
    ("MEAN_CURVATURE_FLOW", "Mean Curvature", "Move the SDF along estimated curvature for rounder features with better shape preservation", 1),
)

# SourceHeightBackMode without INFINITE, which the solver cannot run. The
# numbers are SourceHeightBackMode values and are stored in saved files; a
# 1.x file that saved INFINITE (0) reads back as "" and is rejected with a
# message by the height-source gather.
HEIGHT_BACK_MODE_ITEMS = (
    ("FLAT", "Flat", "Close the height region with a flat back surface at Base Height", 1),
    ("PROFILE", "Profile", "Shape the back surface from the selected grayscale Height Profile", 2),
)


# Two-tier UI disclosure (design D2). These are presentation-only settings:
# they change which panels and controls the add-on shows, never solver state.
UI_MODE_ITEMS = (
    ("BASIC", "Basic", "Show only the everyday controls needed to author and run a liquid; hide expert solver internals"),
    ("ADVANCED", "Advanced", "Reveal the full set of solver, surfacing, whitewater, and export controls for expert tuning"),
)
QUALITY_ITEMS = (
    ("DRAFT", "Draft", "Fastest, lowest-fidelity preview for iterating on staging and motion"),
    ("PREVIEW", "Preview", "Balanced fidelity and speed for day-to-day look development"),
    ("FINAL", "Final", "Highest fidelity for production renders at the greatest compute cost"),
    ("CUSTOM", "Custom", "Fidelity is governed by the individual solver settings rather than a quality preset"),
)
# Basic-tier headline for the 33 per-attribute export toggles (design D2, spec §8).
# Presentation-only for now: the per-attribute inventory in Advanced remains the
# source of truth; wiring this preset to those toggles is deferred to the Chunk 3+
# data-model work (it affects cache identity, so it must land with the migration).
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
        else:
            self.liquid_view_mode = "HIDDEN"
    finally:
        _LIQUID_VIEW_SYNCING = False
    _request_output_refresh(self, context)


def _request_scene_update(context, action):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        return
    try:
        from .blender import pending_settings

        pending_settings.request_scene_update(scene, action)
    except Exception as exc:  # noqa: BLE001 - property callbacks must be safe
        try:
            scene.fluidsim.status = f"Settings changed; auto-update failed: {exc}"
        except Exception:  # noqa: BLE001
            pass


def _request_rebuild(self, context):
    from .blender.pending_settings import SceneUpdate

    _request_scene_update(context, SceneUpdate.REBUILD)


def _request_reset(self, context):
    from .blender.pending_settings import SceneUpdate

    _request_scene_update(context, SceneUpdate.RESET)


def _request_live(self, context):
    from .blender.pending_settings import SceneUpdate

    _request_scene_update(context, SceneUpdate.LIVE)


def _request_sync(self, context):
    from .blender.pending_settings import SceneUpdate

    _request_scene_update(context, SceneUpdate.SYNC)


def _update_field_region_kind(self, context):
    """Keep field payload controls reachable for the selected coverage mode."""
    if self.field_region_kind != "VOLUME_GRID" and self.field_value_source != "CONSTANT":
        self.field_value_source = "CONSTANT"
    _request_sync(self, context)


def _update_field_value_source(self, context):
    """Volume payloads require volume-grid coverage and its mapping panel."""
    if self.field_value_source == "VOLUME_GRID" and self.field_region_kind != "VOLUME_GRID":
        self.field_region_kind = "VOLUME_GRID"
    _request_sync(self, context)


def _request_particle_capacity(self, context):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        return
    try:
        from .blender import pending_settings

        pending_settings.update_particle_capacity_policy(scene)
    except Exception as exc:  # noqa: BLE001 - property callbacks must be safe
        scene.fluidsim.status = f"Particle capacity setting failed: {exc}"


def _request_output_refresh(self, context):
    from .blender.pending_settings import SceneUpdate

    _request_scene_update(context, SceneUpdate.OUTPUT)


def _request_surface_refresh(self, context):
    from .blender.pending_settings import SceneUpdate

    _request_scene_update(context, SceneUpdate.SURFACE)


def _request_whitewater_refresh(self, context):
    from .blender.pending_settings import SceneUpdate

    _request_scene_update(context, SceneUpdate.WHITEWATER)


def _scene_for_mode_callback(self, context):
    """Resolve the Scene that owns this scene-level property group."""
    try:
        owner = getattr(self, "id_data", None)
    except ReferenceError:
        owner = None
    scene_type = getattr(bpy.types, "Scene", None)
    if scene_type is not None:
        try:
            if isinstance(owner, scene_type):
                return owner
        except TypeError:
            pass
    return getattr(context, "scene", None) if context is not None else None


def _on_live_changed(self, context):
    scene = _scene_for_mode_callback(self, context)
    if scene is None:
        return
    try:
        from .blender import playback

        playback.set_live_enabled(scene, bool(self.live))
    except Exception as exc:  # noqa: BLE001
        self.status = f"Live toggle failed: {exc}"


def _on_cache_playback_changed(self, context):
    scene = _scene_for_mode_callback(self, context)
    if scene is None:
        return
    try:
        from .blender import playback

        playback.set_cache_playback_enabled(
            scene, bool(self.cache_playback_enabled))
    except Exception as exc:  # noqa: BLE001
        self.status = f"Cache playback toggle failed: {exc}"


# Blender retains pointers to dynamic EnumProperty strings; keep one module
# cache so the returned items stay alive.
_PRODUCT_PRESET_ITEMS = []


def _product_preset_items(_self, _context):
    global _PRODUCT_PRESET_ITEMS
    try:
        from .presets.library import discover_presets

        records = discover_presets()
        items = [
            (
                f"{record.source}:{record.preset.id}",
                record.preset.name,
                f"{record.preset.description} Cost: {record.preset.performance_cost}. "
                f"Scale: {record.preset.expected_scale}.",
            )
            for record in records
        ]
    except Exception as exc:  # invalid user data must not break RNA drawing
        items = [("BUILTIN:preview_water", "Preview Water", f"Preset scan failed: {exc}")]
    _PRODUCT_PRESET_ITEMS = items or [
        ("BUILTIN:preview_water", "Preview Water", "Fast live-preview defaults")
    ]
    return _PRODUCT_PRESET_ITEMS


# Object-level settings (sources / colliders / sinks)


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
        description="Set how this object participates in the simulation: emitting liquid, colliding with it, removing it, applying a force, or sourcing an advanced field",
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
    source_sampling_mode: EnumProperty(
        name="Source Fill Target",
        items=[
            ("INHERIT", "Inherit Simulation Sampling",
             "Use the global Liquid Sampling target"),
            ("MULTIPLIER", "Sampling Multiplier",
             "Multiply global Liquid Sampling without changing spatial resolution"),
            ("LEGACY_ABSOLUTE", "Legacy Absolute",
             "Preserve the source's saved Target Particles / Voxel value"),
        ],
        default="LEGACY_ABSOLUTE", update=_request_reset,
        description="Control source fill occupancy, not solver spatial resolution")
    source_sampling_multiplier: FloatProperty(
        name="Fill Multiplier", default=1.0, min=0.0, max=8.0,
        update=_request_reset,
        description="Multiply global Liquid Sampling for this source; zero disables liquid emission")
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
        description="Choose which solver field receives this non-liquid source; field operations write solver data and never emit liquid, and changing the target changes the payload controls and solver phase")
    field_region_kind: EnumProperty(
        name="Field Region", items=[
            ("ANALYTIC", "Analytic Region", "Use this object's analytic region"),
            ("VOLUME_GRID", "Volume Grid", "Use named values from a Blender Volume"),
        ], default="ANALYTIC", update=_update_field_region_kind,
        description="Choose whether the field source uses this object's analytic shape or a Volume grid; changing it replaces the region controls")
    field_value_source: EnumProperty(
        name="Payload Source", items=[
            ("CONSTANT", "Constant", "Use the constant value below"),
            ("VOLUME_GRID", "Volume Grid", "Sample the named scalar or vector grid"),
        ], default="CONSTANT", update=_update_field_value_source,
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
            "FIELD creates no liquid and instead applies the Field Operation "
            "(pressure by default) inside the region"))
    emit_rate: FloatProperty(
        name="Legacy Target Particles / Voxel", default=8.0, min=0.0,
        max=float(_C.MAX_PARTICLES_PER_VOXEL), soft_max=32.0,
        update=_request_reset,
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
                           subtype="DISTANCE", update=_request_sync)
    temperature: FloatProperty(
        name="Temperature", default=0.0, update=_request_sync,
        description="Set the temperature attribute on emitted particles; higher or lower values change attribute-driven visualization and field operations")
    viscosity: FloatProperty(
        name="Viscosity", default=0.0, min=0.0, update=_request_sync,
        description="Set the viscosity attribute on emitted particles; higher values make those particles contribute thicker behavior when particle viscosity is enabled")
    density: FloatProperty(
        name="Density Attribute", default=1.0, min=0.0,
        description="Set the density attribute on emitted particles, not physical liquid mass; higher or lower values affect density-based visualization",
        update=_request_sync)
    source_color: FloatVectorProperty(
        name="Source Color", size=4, subtype="COLOR",
        default=(0.1, 0.45, 1.0, 1.0), min=0.0, max=1.0,
        update=_request_sync,
        description="Set the persistent color assigned to particles emitted by this source")
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


# Scene-level settings


_OTHER_GROUP_FIELDS = {}


def _reject_other_group_write(own, other, name, where):
    """Raise when ``name`` is declared only on the other half of the split.

    Blender lets Python set any attribute on a PropertyGroup instance, so a
    1.x-style ``scene.fluidsim.voxel_size = 0.05`` would otherwise land on the
    Python wrapper, read back as if it worked, and never reach the solver.
    """
    fields = _OTHER_GROUP_FIELDS.get(own)
    if fields is None:
        fields = _OTHER_GROUP_FIELDS[own] = frozenset(
            set(getattr(other, "__annotations__", {}))
            - set(getattr(own, "__annotations__", {})))
    if name in fields:
        raise AttributeError(
            f"{name!r} is not a {own.__name__} setting; set it on {where}")


class FLUIDSIM_PG_scene(bpy.types.PropertyGroup):
    """Scene-scoped FluidSim state, bound to ``Scene.fluidsim``.

    Interface mode, the Quality tier, the domain designation, playback and
    cache plumbing, viewport/output presentation, and the read-only status
    mirrors. Solver settings live on :class:`FLUIDSIM_PG_domain`. The two
    classes are disjoint, so reading a domain setting off ``scene.fluidsim``
    raises instead of returning a stale value. The class that declares a
    setting owns it. ``presets.schema.SCENE_OWNED_SETTINGS`` is the bpy-free
    copy of this class's field set, and
    tests/runtime/test_pending_settings_registry.py pins the two equal. Writing
    a domain setting here raises too (see ``_reject_other_group_write``).
    """

    def __setattr__(self, name, value):
        _reject_other_group_write(
            FLUIDSIM_PG_scene, FLUIDSIM_PG_domain, name,
            "scene.fluidsim.domain_object.fluidsim_domain")
        super().__setattr__(name, value)

    # UI disclosure (presentation-only; never touches solver state)
    ui_mode: EnumProperty(
        name="Interface", items=UI_MODE_ITEMS, default="BASIC",
        description="Choose how much of the interface is shown: Basic hides expert solver internals, Advanced reveals the full control set")
    quality: EnumProperty(
        name="Quality", items=QUALITY_ITEMS, default="PREVIEW",
        update=_request_rebuild,
        description="Select a fidelity target; Draft/Preview/Final trade speed against accuracy, and Custom defers entirely to the individual solver settings")

    # Simulation
    live: BoolProperty(
        name="Live Simulation", default=False,
        update=_on_live_changed,
        description="Enable to step the GPU solver as the timeline plays; disable to stop live stepping while keeping the current state")
    playback_advanced_expanded: BoolProperty(
        name="Advanced Playback", default=False, options={"SKIP_SAVE"},
        description="Show or hide expert controls for recovering timeline frames dropped during live playback")
    max_frame_catchup: IntProperty(
        name="Maximum Playback Recovery Frames", default=8, min=1, max=120,
        description=(
            "Set the largest forward timeline gap the live solver may recover synchronously. "
            "Higher values preserve more dropped playback frames but can stall Blender longer; lower values stop sooner and report desynchronization instead of silently running the simulation in slow motion"))
    domain_object: PointerProperty(
        name="Domain", type=bpy.types.Object,
        update=_request_rebuild,
        description="Select the object whose bounding box defines the simulation domain; changing it rebuilds grid bounds and capacity requirements")

    capacity_show_alternatives: BoolProperty(
        name="Show Capacity Alternatives", default=False,
        options={"SKIP_SAVE"},
        description="Expand lower-priority resolution and domain alternatives")
    surface_advanced_expanded: BoolProperty(
        name="Advanced Surface", default=False, options={"SKIP_SAVE"},
        description="Show or hide sparse-capacity and simulation-field retention controls")

    # Viewport display / output
    liquid_view_mode: EnumProperty(
        name="Liquid View",
        items=LIQUID_VIEW_ITEMS,
        default="SURFACE",
        update=_apply_liquid_view_mode,
        description="Choose whether the viewport hides liquid output or shows the liquid surface, particles, or both; changing it updates the two visibility toggles below",
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
    whitewater_output_mode: EnumProperty(
        name="Whitewater Outputs",
        items=WHITEWATER_OUTPUT_ITEMS,
        default="COMBINED",
        update=_request_output_refresh,
        description="Choose the fast combined pool, separate typed outputs, or both",
    )
    whitewater_display_scale: FloatProperty(
        name="Whitewater Scale", default=1.0, min=0.05, soft_max=4.0,
        update=_request_output_refresh,
        description="Scale whitewater viewport point radius; higher values make bubbles and spray look larger, while lower values reveal separation without changing simulation")
    whitewater_display_fraction: FloatProperty(
        name="Display Fraction", default=1.0, min=0.01, max=1.0,
        update=_request_output_refresh,
        description="Set the fraction of whitewater points shown in the viewport; higher values show a denser, more representative cloud at greater display cost, while lower values improve responsiveness")
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
    foam_material: PointerProperty(
        name="Foam Material", type=bpy.types.Material,
        update=_request_output_refresh,
        description="Select the material assigned to the separate foam output")
    bubble_material: PointerProperty(
        name="Bubble Material", type=bpy.types.Material,
        update=_request_output_refresh,
        description="Select the material assigned to the separate bubble output")
    spray_material: PointerProperty(
        name="Spray Material", type=bpy.types.Material,
        update=_request_output_refresh,
        description="Select the material assigned to the separate spray output")
    output_collection_name: StringProperty(
        name="Output Collection", default="FluidSim Output",
        update=_request_output_refresh,
        description="Set the Blender collection name that owns generated outputs; changing it moves or creates outputs under the new collection")

    # Guided setup / product presets
    setup_mode: EnumProperty(
        name="Setup Mode",
        items=(
            ("QUICK_LIQUID", "Quick Liquid", "Selected mesh becomes an initial-fill liquid emitter"),
            ("QUICK_CONTAINER", "Quick Container", "Selected mesh becomes a collider/container with a height fill"),
            ("QUICK_VISCOUS", "Quick Viscous Liquid", "Selected mesh becomes a thick-liquid emitter"),
            ("ADD_COLLIDER", "Add Collider", "Tag selected meshes as colliders"),
            ("ADD_INFLOW", "Add Inflow", "Tag selected meshes as continuous liquid inflows"),
            ("ADD_OUTFLOW", "Add Outflow", "Tag selected meshes as liquid sinks"),
        ),
        default="QUICK_LIQUID",
        description="Choose the selected-object setup transaction",
    )
    product_preset: EnumProperty(
        name="Preset",
        items=_product_preset_items,
        description="Apply a validated built-in or user FluidSim preset",
    )
    active_preset_id: StringProperty(
        name="Active Preset",
        default="",
        description="Identifier of the last successfully applied product preset",
    )

    # Production bake
    cache_format: EnumProperty(
        name="Cache Format",
        # The "V2" identifier is kept so saved .blend files still resolve; the
        # bake writes cache format v3.
        items=(("V2", "Resumable Cache", "Transactional frame cache with deterministic restart checkpoints"),),
        default="V2",
        description="Choose the on-disk cache layout used for production bakes",
    )
    bake_start: IntProperty(
        name="Bake Start", default=1, min=-1048574, max=1048574,
        description="First production-bake frame",
    )
    bake_end: IntProperty(
        name="Bake End", default=250, min=-1048574, max=1048574,
        description="Last production-bake frame",
    )
    checkpoint_interval: IntProperty(
        name="Checkpoint Interval", default=1, min=1, max=100,
        description="Checkpoint cadence; production bakes commit one restart checkpoint per frame",
    )
    bake_status: EnumProperty(
        name="Bake Status",
        items=tuple((value, value.title(), "") for value in (
            "IDLE", "PREFLIGHT", "INITIALIZING", "RUNNING", "CANCELLING",
            "CANCELLED", "COMPLETE", "FAILED",
        )),
        default="IDLE",
        options={"SKIP_SAVE"},
        description="Latest measured production-bake lifecycle state",
    )
    bake_progress: FloatProperty(
        name="Bake Progress", default=0.0, min=0.0, max=1.0,
        subtype="FACTOR", options={"SKIP_SAVE"},
        description="Latest measured fraction of the production bake completed",
    )
    bake_message: StringProperty(
        name="Bake Message", default="", options={"SKIP_SAVE"},
        description="Most recent status or error message reported by the production bake",
    )
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

    # Cache (optional, supplemental only)
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

    # Status mirrors (read-only UI feedback)
    status: StringProperty(
        name="Status", default="Solver not initialized",
        description="Latest measured solver lifecycle status reported for this domain")
    stat_particles: IntProperty(
        default=0,
        description="Latest measured count of active liquid particles in the solver")
    stat_voxels: IntProperty(
        default=0,
        description="Latest measured count of active sparse-grid voxels in the solver")
    stat_whitewater: IntProperty(
        default=0,
        description="Latest measured count of active whitewater (foam/bubble/spray) particles")
    stat_whitewater_foam: IntProperty(
        default=0,
        description="Latest measured count of active foam particles")
    stat_whitewater_bubbles: IntProperty(
        default=0,
        description="Latest measured count of active bubble particles")
    stat_whitewater_spray: IntProperty(
        default=0,
        description="Latest measured count of active spray particles")
    stat_whitewater_invalid_types: IntProperty(
        default=0,
        description="Latest measured count of whitewater particles with an unrecognized type classification")
    stat_whitewater_required: IntProperty(
        default=0,
        description="Latest measured count of whitewater particles the solver requested to spawn this step")
    stat_whitewater_admitted: IntProperty(
        default=0,
        description="Latest measured count of requested whitewater particles actually admitted within capacity")
    stat_whitewater_dropped: IntProperty(
        default=0,
        description="Latest measured count of requested whitewater particles dropped because capacity was exceeded")
    stat_whitewater_policy: StringProperty(
        default="FIXED_CAPACITY_DETERMINISTIC_DROPS",
        description="Name of the policy currently governing whitewater admission when capacity is exceeded")
    stat_whitewater_status: StringProperty(
        default="OK",
        description="Latest measured whitewater admission health status")
    stat_step_ms: FloatProperty(
        default=0.0,
        description="Most recent solver step time in milliseconds")
    stat_substeps: IntProperty(
        default=0,
        description="Latest measured number of substeps taken on the most recent frame")
    stat_cfl_clamped: BoolProperty(
        default=False,
        description="Latest measured indicator that the CFL-driven substep count was clamped by Max Substeps")
    stat_capillary_dt: FloatProperty(
        default=0.0,
        description="Latest measured stable timestep required by surface-tension capillary wave stability")
    stat_capillary_substeps_required: IntProperty(
        default=1,
        description="Latest measured number of substeps required to satisfy capillary wave stability")
    stat_capillary_clamped: BoolProperty(
        default=False,
        description="Latest measured indicator that the capillary-required substep count was clamped by Max Substeps")
    stat_substep_limiter: StringProperty(
        default="MINIMUM",
        description="Name of the constraint (CFL, capillary, or minimum) that selected the most recent substep count")
    stat_surface_tension_active_voxels: IntProperty(
        default=0,
        description="Latest measured count of voxels with active surface-tension forces")
    stat_surface_tension_max_curvature: FloatProperty(
        default=0.0,
        description="Latest measured maximum surface curvature used by surface tension")
    stat_surface_tension_max_acceleration: FloatProperty(
        default=0.0,
        description="Latest measured maximum acceleration applied by surface tension")
    stat_surface_tension_temporary_bytes: IntProperty(
        default=0,
        description="Latest measured temporary VRAM, in bytes, allocated for surface-tension computation")
    stat_surface_tension_submission_ms: FloatProperty(
        default=0.0,
        description="Most recent GPU submission time in milliseconds for surface-tension computation")
    stat_surface_tension_status: StringProperty(
        default="DISABLED",
        description="Latest measured surface-tension computation health status")
    stat_surface_tension_message: StringProperty(
        default="",
        description="Most recent status or error message reported by surface-tension computation")
    stat_native_solver_message: StringProperty(
        default="",
        description="Why a pressure or viscosity solve fell back from CUDA graphs to a slower host loop, if it did")


# Domain-level settings


def _domain_longest_axis(group) -> float:
    """Longest world-space AABB edge of the domain object that owns ``group``.

    Mirrors ``blender.solver_settings.domain_bounds`` (``matrix_world`` applied to
    ``bound_box``) and falls back to the default domain box with no object.
    """
    from .blender.solver_settings import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN

    obj = getattr(group, "id_data", None)
    corners = getattr(obj, "bound_box", None)
    matrix = getattr(obj, "matrix_world", None)
    if corners is None or matrix is None:
        return max(hi - lo for lo, hi in zip(DEFAULT_DOMAIN_MIN, DEFAULT_DOMAIN_MAX))
    from mathutils import Vector

    world = [matrix @ Vector(corner) for corner in corners]
    return max(
        max(getattr(c, axis) for c in world) - min(getattr(c, axis) for c in world)
        for axis in ("x", "y", "z"))


def _get_voxel_size(self) -> float:
    from .sim.quality_contract import voxel_size_for_divisions

    return voxel_size_for_divisions(
        self.resolution_divisions, _domain_longest_axis(self))


def _set_voxel_size(self, value) -> None:
    from .sim.quality_contract import divisions_for_voxel_size

    self.resolution_divisions = divisions_for_voxel_size(
        value, _domain_longest_axis(self))


_MIN_VOXEL_SIZE = 1.0e-4


class FLUIDSIM_PG_domain(bpy.types.PropertyGroup):
    """Per-domain solver settings, bound to ``Object.fluidsim_domain``.

    Everything the solver consumes: resolution, particles, forces, pressure,
    viscosity, surface tension, surface/whitewater generation, height fill and
    the production attribute inventory. Read these through
    ``scene_access.active_domain_settings(scene)``, never off ``scene.fluidsim``.
    """

    def __setattr__(self, name, value):
        _reject_other_group_write(
            FLUIDSIM_PG_domain, FLUIDSIM_PG_scene, name, "scene.fluidsim")
        super().__setattr__(name, value)

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

    # Domain / sparse grid
    resolution_divisions: IntProperty(
        name="Resolution Divisions", default=64, min=8, soft_min=16, soft_max=512,
        update=_request_rebuild,
        description="Set the number of simulation-grid cells along the longest domain axis; higher values capture finer detail but allocate more leaves, particles, VRAM, and compute, while resizing the domain changes the cell size and not the cell count")
    voxel_size: FloatProperty(
        name="Voxel Size", min=_MIN_VOXEL_SIZE, precision=4, subtype="DISTANCE",
        get=_get_voxel_size, set=_set_voxel_size,
        description="Effective simulation-grid cell size in world units, derived as the longest domain axis divided by Resolution Divisions; writing it sets the divisions that give the nearest cell size")
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

    # Particles
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
    particles_per_voxel: IntProperty(
        name="Liquid Sampling", default=8, min=0, max=_C.MAX_PARTICLES_PER_VOXEL,
        update=_request_reset,
        description="Set authoritative liquid rest density and inherited source occupancy; changing it requires Reset, while zero disables inherited liquid emission")
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
        name="Max Speed", default=0.0, min=0.0, unit="VELOCITY",
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

    # Global forces
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
        name="Rate", default=0.0, min=0.0,
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

    # Pressure
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
    # Default is JACOBI, not the spec's MULTIGRID: the multigrid speed targets are
    # unmeasured and the demo hash must hold until the beta re-baseline. See
    # docs/specs/3.0/SPEC-multigrid-pressure.md, "Settings".
    pressure_preconditioner: EnumProperty(
        name="Preconditioner",
        items=[
            ("JACOBI", "Jacobi", "Use the diagonal preconditioner; the cheapest per iteration, but it needs more iterations on large or finely resolved domains"),
            ("MULTIGRID", "Multigrid", "Use a multigrid V-cycle preconditioner; each iteration costs more but far fewer are needed on large or finely resolved domains"),
        ],
        default="JACOBI", update=_request_live,
        description="Choose the preconditioner for the PCG pressure solver; Multigrid can converge in fewer iterations on large domains, while Jacobi is cheaper per iteration. Used only by the PCG solver")
    pressure_threshold: FloatProperty(
        name="Threshold", default=0.0, min=0.0, max=1.0,
        update=_request_live,
        description="Remove solved pressures below this fraction of the frame's pressure range; higher values suppress more small pressure detail, while zero keeps all solved values")

    # Viscosity
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
        name="Iterations", default=_C.DEFAULT_VISCOSITY_ITERATIONS,
        min=1, max=1000,
        update=_request_live,
        description="Cap viscosity-solver iterations; higher values can converge thick fluids more accurately at greater cost, while lower values finish faster with more residual error")
    viscosity_tolerance: FloatProperty(
        name="Tolerance", default=1.0e-4, min=0.0, precision=8,
        update=_request_live,
        description="Set the viscosity residual required for early convergence; lower values demand a tighter solve and may take longer, while higher values stop sooner")

    # Surface tension
    surface_tension_enabled: BoolProperty(
        name="Enable Surface Tension", default=False, update=_request_reset,
        description="Enable continuum-surface-force capillarity from the liquid SDF; changing allocation state resets the solver")
    surface_tension_coefficient: FloatProperty(
        name="Coefficient", default=0.072, min=0.0, soft_max=1.0,
        precision=5, update=_request_live,
        description="Set surface-tension coefficient in newtons per metre; water near room temperature is approximately 0.072 N/m, higher values form tighter droplets, and zero applies no force")
    surface_tension_density: FloatProperty(
        name="Liquid Density", default=1000.0, min=1.0e-6,
        soft_min=1.0, soft_max=5000.0, update=_request_live,
        description="Set liquid density in kilograms per cubic metre for capillary acceleration and timestep stability; higher density reduces surface-tension acceleration")
    surface_tension_band_voxels: FloatProperty(
        name="Force Band", default=1.5, min=0.5, max=8.0,
        update=_request_live,
        description="Set the half-width of the compact surface-force band in simulation voxels; wider bands are smoother but spread the force over more cells")
    surface_tension_safety: FloatProperty(
        name="Timestep Safety", default=0.5, min=0.05, max=1.0,
        update=_request_live,
        description="Scale the explicit capillary timestep limit; lower values request more conservative substeps and higher values reduce cost with less margin")
    surface_tension_smoothing_iterations: IntProperty(
        name="Normal Smoothing", default=1, min=0, max=8,
        update=_request_live,
        description="Set normalized 7-point smoothing passes for SDF normals; more passes reduce curvature noise but soften small capillary features")

    # Clear dye color mixing
    species_enabled: BoolProperty(
        name="Enable Color Mixing", default=False, update=_request_reset,
        description="Mix the existing source colors on the GPU; changing color-channel storage resets the solver")
    species_mixing_rate: FloatProperty(
        name="Mixing Strength", default=8.0, min=0.0, soft_max=40.0,
        update=_request_live,
        description="Control how quickly nearby liquid particle colors approach their local mixture; zero preserves source colors")

    # Surface
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
        precision=5, subtype="DISTANCE", update=_request_surface_refresh,
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
        name="Live Surface Every", default=1, min=1, max=8,
        update=_request_surface_refresh,
        description=("Set how often the Blender viewport mesh publishes during live playback; "
                     "higher values publish less often for better responsiveness, while 1 publishes every frame; physics, cache, and exports still evaluate every frame"))
    surface_build_lowres_sdf: BoolProperty(
        name="Keep Sim Surface Field", default=False,
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

    # Whitewater
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
    ww_surface_band: FloatProperty(
        name="Emission Surface Band", default=2.0, min=0.05, soft_max=8.0,
        update=_request_live,
        description="Set the liquid-SDF emission band in voxels; higher values admit candidates farther from the interface")
    ww_min_speed: FloatProperty(
        name="Min Speed", default=0.5, min=0.0, update=_request_live,
        description="Set the minimum liquid speed eligible for whitewater emission; higher values restrict emission to faster motion, while lower values include calmer flow")
    ww_life_min: FloatProperty(
        name="Life Min", default=1.0, min=0.0, update=_request_live,
        description="Set the shortest whitewater lifetime in seconds; higher values keep every whitewater particle visible longer and increase occupancy")
    ww_life_max: FloatProperty(
        name="Life Max", default=3.0, min=0.0, update=_request_live,
        description="Set the longest whitewater lifetime in seconds; higher values allow some particles to persist longer and increase occupancy")
    ww_foam_band: FloatProperty(
        name="Foam Band", default=1.0, min=0.05, soft_max=8.0,
        update=_request_live,
        description="Set the near-surface foam band in voxels; deeper particles classify as bubbles and sufficiently elevated fast particles classify as spray")
    ww_spray_min_speed: FloatProperty(
        name="Spray Speed", default=0.5, min=0.0, update=_request_live,
        description="Set the minimum speed required to enter spray above the foam band")
    ww_type_hysteresis: FloatProperty(
        name="Type Hysteresis", default=0.25, min=0.0, max=0.95,
        update=_request_live,
        description="Separate foam/bubble/spray enter and exit thresholds to prevent rapid type flicker")
    ww_foam_advection: FloatProperty(
        name="Foam Fluid Advection", default=1.0, min=0.0, max=1.0,
        update=_request_live,
        description="Blend foam velocity toward the surrounding liquid each substep; one follows the liquid fully")
    ww_foam_preservation: FloatProperty(
        name="Foam Preservation", default=0.0, min=0.0, max=1.0,
        update=_request_live,
        description="Slow foam aging in dense whitewater regions when density influence is enabled")
    ww_foam_density_influence: FloatProperty(
        name="Foam Density Influence", default=0.0, min=0.0, soft_max=1.0,
        update=_request_live,
        description="Scale how strongly local whitewater density contributes to foam preservation")
    ww_foam_life_multiplier: FloatProperty(
        name="Foam Life", default=1.0, min=0.01, soft_max=5.0,
        update=_request_live,
        description="Multiply base lifetime while particles are classified as foam")
    ww_bubble_drag: FloatProperty(
        name="Bubble Drag", default=4.0, min=0.0, update=_request_live,
        description="Set how quickly bubbles follow surrounding liquid velocity; higher values couple bubbles more strongly, while zero lets them retain their own motion")
    ww_bubble_buoyancy: FloatProperty(
        name="Bubble Buoyancy", default=1.0, min=0.0, soft_max=4.0,
        update=_request_live,
        description="Scale acceleration opposite gravity for particles classified as bubbles")
    ww_bubble_life_multiplier: FloatProperty(
        name="Bubble Life", default=1.0, min=0.01, soft_max=5.0,
        update=_request_live,
        description="Multiply base lifetime while particles are classified as bubbles")
    ww_spray_drag: FloatProperty(
        name="Spray Drag", default=0.2, min=0.0, update=_request_live,
        description="Set aerodynamic damping on spray velocity; higher values slow spray more quickly, while zero preserves ballistic motion")
    ww_spray_gravity_scale: FloatProperty(
        name="Spray Gravity", default=1.0, min=0.0, soft_max=4.0,
        update=_request_live,
        description="Scale scene gravity for particles classified as spray")
    ww_spray_life_multiplier: FloatProperty(
        name="Spray Life", default=1.0, min=0.01, soft_max=5.0,
        update=_request_live,
        description="Multiply base lifetime while particles are classified as spray")

    # Height source
    height_enabled: BoolProperty(
        name="Height Fill", default=False,
        update=_request_reset,
        description="Enable to fill the domain below the configured height when the solver resets; disable to omit this height source")
    height_base: FloatProperty(
        name="Base Height", default=0.3, subtype="DISTANCE",
        update=_request_reset,
        description="Set the world-space base fill level; higher values fill more of the domain vertically, while lower values fill less")
    height_back_mode: EnumProperty(
        name="Back Mode", items=HEIGHT_BACK_MODE_ITEMS, default="FLAT",
        update=_request_reset,
        description="Choose how the height source closes behind its profile; changing it switches between a flat and an image-profiled back surface")
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
    height_error: StringProperty(
        name="Height Fill Error", default="",
        description="Most recent error reported while filling the height source, if any")
    height_velocity: FloatVectorProperty(
        name="Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY",
        update=_request_reset,
        description="Set initial velocity for height-fill particles; changing components changes their starting speed and direction")
    height_emit_particles: BoolProperty(
        name="Emit Particles", default=True, update=_request_reset,
        description="Enable to seed liquid particles in the height region; disable to keep the height configuration without creating particles")

    # Production render attributes
    # Inventory changes are reset-tier because color/source/UID channels are
    # allocated only when selected and cache signatures include every toggle.
    # The master switch gates the whole inventory; off skips the per-frame
    # attribute readback, transfer, publication and cache bytes.
    export_attributes: BoolProperty(
        name="Export Attributes", default=False, update=_request_reset,
        description=(
            "Enable to publish and cache the per-point attributes selected in Advanced "
            "(velocity, age, motion blur and so on); disable for faster playback and "
            "baking and smaller caches. Whitewater type is always kept"))
    surface_attribute_velocity: BoolProperty(
        name="Velocity", default=True, update=_request_reset,
        description="Publish world-units-per-second liquid velocity on surface vertices")
    surface_attribute_speed: BoolProperty(
        name="Speed", default=False, update=_request_reset,
        description="Publish scalar liquid speed on surface vertices")
    surface_attribute_vorticity: BoolProperty(
        name="Vorticity", default=False, update=_request_reset,
        description="Sample projected-grid curl onto surface vertices")
    surface_attribute_age: BoolProperty(
        name="Age", default=False, update=_request_reset,
        description="Transfer nearest liquid-particle age to surface vertices")
    surface_attribute_lifetime: BoolProperty(
        name="Lifetime", default=False, update=_request_reset,
        description="Transfer nearest liquid-particle lifetime to surface vertices")
    surface_attribute_color: BoolProperty(
        name="Color", default=False, update=_request_reset,
        description="Allocate source color and transfer it to surface vertices")
    surface_attribute_source_id: BoolProperty(
        name="Source ID", default=False, update=_request_reset,
        description="Allocate deterministic source identity and transfer it to the surface")
    surface_attribute_viscosity: BoolProperty(
        name="Viscosity", default=False, update=_request_reset,
        description="Transfer liquid-particle viscosity to surface vertices")
    surface_attribute_density: BoolProperty(
        name="Density", default=False, update=_request_reset,
        description="Transfer liquid-particle density to surface vertices")
    surface_attribute_motion: BoolProperty(
        name="Motion Blur", default=True, update=_request_reset,
        description="Publish Blender's conventional per-frame velocity attribute")
    particles_attribute_velocity: BoolProperty(
        name="Velocity", default=False, update=_request_reset,
        description="Publish liquid-particle velocity")
    particles_attribute_speed: BoolProperty(
        name="Speed", default=False, update=_request_reset,
        description="Publish liquid-particle speed")
    particles_attribute_vorticity: BoolProperty(
        name="Vorticity", default=False, update=_request_reset,
        description="Sample projected-grid curl at liquid particles")
    particles_attribute_age: BoolProperty(
        name="Age", default=True, update=_request_reset,
        description="Publish liquid-particle age")
    particles_attribute_lifetime: BoolProperty(
        name="Lifetime", default=True, update=_request_reset,
        description="Publish liquid-particle lifetime")
    particles_attribute_color: BoolProperty(
        name="Color", default=False, update=_request_reset,
        description="Allocate and publish source color per liquid particle")
    particles_attribute_source_id: BoolProperty(
        name="Source ID", default=False, update=_request_reset,
        description="Allocate and publish deterministic source identity")
    particles_attribute_uid: BoolProperty(
        name="UID", default=False, update=_request_reset,
        description="Allocate and publish a monotonic deterministic particle identifier")
    particles_attribute_viscosity: BoolProperty(
        name="Viscosity", default=False, update=_request_reset,
        description="Publish liquid-particle viscosity")
    particles_attribute_density: BoolProperty(
        name="Density", default=False, update=_request_reset,
        description="Publish liquid-particle density")
    whitewater_attribute_velocity: BoolProperty(
        name="Velocity", default=False, update=_request_reset,
        description="Publish whitewater velocity")
    whitewater_attribute_speed: BoolProperty(
        name="Speed", default=False, update=_request_reset,
        description="Publish whitewater speed")
    whitewater_attribute_age: BoolProperty(
        name="Age", default=True, update=_request_reset,
        description="Publish whitewater age")
    whitewater_attribute_lifetime: BoolProperty(
        name="Lifetime", default=True, update=_request_reset,
        description="Publish whitewater lifetime")
    whitewater_attribute_type: BoolProperty(
        name="Type", default=True, options={"HIDDEN"},
        description="Persistent foam, bubble, and spray type; always published")

    # Diagnostics
    profiling_enabled: BoolProperty(
        name="Enable Profiling", default=False, update=_request_rebuild,
        description="Enable per-phase solver timing collection for diagnostics; disable to avoid synchronization and measurement overhead")
    profiling_sync: BoolProperty(
        name="Accurate GPU Timing", default=True,
        update=_request_rebuild,
        description="Enable CUDA synchronization around each phase for accurate timings at higher profiling cost; disable for lower overhead but less precise asynchronous measurements")


CLASSES = (
    FLUIDSIM_PG_object,
    FLUIDSIM_PG_scene,
    FLUIDSIM_PG_domain,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fluidsim = PointerProperty(type=FLUIDSIM_PG_scene)
    bpy.types.Object.fluidsim = PointerProperty(type=FLUIDSIM_PG_object)
    # Per-object home for domain settings; the solver reads only this group.
    bpy.types.Object.fluidsim_domain = PointerProperty(type=FLUIDSIM_PG_domain)


def unregister():
    if hasattr(bpy.types.Object, "fluidsim_domain"):
        del bpy.types.Object.fluidsim_domain
    if hasattr(bpy.types.Object, "fluidsim"):
        del bpy.types.Object.fluidsim
    if hasattr(bpy.types.Scene, "fluidsim"):
        del bpy.types.Scene.fluidsim
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not unregister {cls.__name__}: {exc!r}")
