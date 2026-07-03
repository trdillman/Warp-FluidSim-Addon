"""Blender property groups for the FluidSim Warp add-on.

Scene-level settings drive the solver; object-level settings tag scene
objects as sources / colliders / sinks. Property definitions import only
``sim.enums`` (pure python) so registration never touches Warp/CUDA.
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

from .sim.enums import (
    BlendMode,
    CustomKernelConstantType,
    CustomKernelOrder,
    CustomKernelRunOver,
    LimitBehavior,
    MultiGrid,
    ParticleSampleType,
    ProjectNonDivergentSolver,
    RelaxMode,
    SdfFilterType,
    SourceHeightBackMode,
    SourceShapeForceType,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
    TraceMode,
    ViscositySolver,
    enum_items,
)


def _items(enum_cls):
    return enum_items(enum_cls)


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
)


WHITEWATER_ATTR_ITEMS = (
    ("NONE", "Material", "Use the assigned material color"),
    ("HEIGHT", "Height", "Color by world-space height"),
    ("SPEED", "Speed", "Color by whitewater velocity magnitude"),
    ("AGE", "Age", "Color by whitewater age"),
    ("LIFE", "Life", "Color by whitewater life span"),
)


def _apply_liquid_view_mode(self, _context):
    mode = self.liquid_view_mode
    self.show_surface = mode in {"SURFACE", "SURFACE_PARTICLES"}
    self.show_particles = mode in {"PARTICLES", "SURFACE_PARTICLES"}


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


# ---------------------------------------------------------------------------
# Object-level settings (sources / colliders / sinks)
# ---------------------------------------------------------------------------


class FLUIDSIM_PG_object(bpy.types.PropertyGroup):
    role: EnumProperty(
        name="Role",
        items=[
            ("NONE", "None", "Not part of the simulation"),
            ("SOURCE", "Source", "Emit liquid"),
            ("COLLIDER", "Collider", "Obstacle (mesh SDF)"),
            ("SINK", "Sink", "Remove liquid"),
            ("FORCE", "Force", "Force region"),
        ],
        default="NONE",
    )
    # Source shape settings.
    shape: EnumProperty(
        name="Shape", items=_items(SourceShapeShape), default="BOX",
        description="Analytic emission shape (object transform applies)")
    use_mesh_shape: BoolProperty(
        name="Use Mesh", default=False,
        description="Sample the object's mesh volume (SDF) instead of an analytic shape")
    source_mode: EnumProperty(
        name="Mode", items=_items(SourceShapeSourceMode), default="PARTICLES")
    emit_rate: FloatProperty(
        name="Emit Rate", default=8.0, min=0.0, soft_max=32.0,
        description="Target particles per voxel inside the source")
    initial_velocity: FloatVectorProperty(
        name="Initial Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY")
    use_object_velocity: BoolProperty(
        name="Inherit Object Velocity", default=True,
        description="Add velocity from animated object motion")
    blend_mode: EnumProperty(name="Blend", items=_items(BlendMode), default="REPLACE")
    feather: FloatProperty(name="Feather", default=0.0, min=0.0)
    temperature: FloatProperty(name="Temperature", default=0.0)
    viscosity: FloatProperty(name="Viscosity", default=0.0, min=0.0)
    density: FloatProperty(name="Density", default=1.0, min=0.0)
    use_lifespan: BoolProperty(name="Use Lifespan", default=False)
    life_min: FloatProperty(name="Life Min", default=2.0, min=0.0)
    life_max: FloatProperty(name="Life Max", default=4.0, min=0.0)
    force_type: EnumProperty(
        name="Force Type", items=_items(SourceShapeForceType), default="OFF")
    force_strength: FloatProperty(name="Force Strength", default=1.0)
    force_direction: FloatVectorProperty(
        name="Force Direction", size=3, default=(0.0, 0.0, 1.0))
    # Collider settings.
    collider_friction: FloatProperty(name="Friction", default=0.0, min=0.0, max=1.0)
    collider_restitution: FloatProperty(name="Restitution", default=0.0, min=0.0, max=1.0)
    transfer_velocity: BoolProperty(
        name="Transfer Velocity", default=True,
        description="Push liquid with animated collider motion")
    enabled: BoolProperty(name="Enabled", default=True)


# ---------------------------------------------------------------------------
# Scene-level settings
# ---------------------------------------------------------------------------


class FLUIDSIM_PG_settings(bpy.types.PropertyGroup):
    # -- Simulation -----------------------------------------------------------
    live: BoolProperty(
        name="Live Simulation", default=False,
        description="Step the GPU solver during animation playback")
    domain_object: PointerProperty(
        name="Domain", type=bpy.types.Object,
        description="Object whose bounding box defines the simulation domain")
    time_scale: FloatProperty(name="Time Scale", default=1.0, min=0.0, soft_max=4.0)
    cfl: FloatProperty(name="CFL", default=1.0, min=0.1, soft_max=5.0)
    min_substeps: IntProperty(name="Min Substeps", default=1, min=1, max=16)
    max_substeps: IntProperty(name="Max Substeps", default=4, min=1, max=32)
    gravity: FloatVectorProperty(
        name="Gravity", size=3, default=(0.0, 0.0, -9.81), subtype="ACCELERATION")
    use_scene_gravity: BoolProperty(name="Use Scene Gravity", default=True)

    # -- Domain / sparse grid -------------------------------------------------
    voxel_size: FloatProperty(
        name="Voxel Size", default=0.05, min=0.005, soft_max=0.5, precision=4)
    leaf_capacity: IntProperty(
        name="Leaf Capacity", default=8192, min=64, max=262144,
        description="Maximum allocated 8x8x8 leaves (sparse octree)")
    topology_expand: IntProperty(
        name="Topology Expand", default=1, min=0, max=4,
        description="Leaf dilation iterations around active fluid")
    open_boundaries: BoolProperty(
        name="Open Boundaries", default=False,
        description="Kill particles leaving the domain instead of clamping")

    # -- Particles -------------------------------------------------------------
    max_particles: IntProperty(
        name="Max Particles", default=2_000_000, min=1024, max=50_000_000)
    particles_per_voxel: IntProperty(name="Particles / Voxel", default=8, min=1, max=64)
    sample_type: EnumProperty(
        name="Transfer", items=_items(ParticleSampleType), default="FLIP")
    flip_ratio: FloatProperty(name="FLIP Ratio", default=0.97, min=0.0, max=1.0)
    trace_mode: EnumProperty(name="Trace", items=_items(TraceMode), default="RK3")
    relax_enabled: BoolProperty(name="Relax Particles", default=True)
    relax_strength: FloatProperty(name="Relax Strength", default=0.5, min=0.0, max=1.0)
    relax_mode: EnumProperty(name="Relax Mode", items=_items(RelaxMode),
                             default="NEIGHBOR_VOXELS")
    max_speed: FloatProperty(
        name="Max Speed", default=0.0, min=0.0,
        description="Speed limit in m/s (0 disables)")
    limit_behavior: EnumProperty(
        name="Limit Behavior", items=_items(LimitBehavior), default="MAINTAIN")
    limit_percent: FloatProperty(name="Limit Percent", default=0.5, min=0.0, max=1.0)
    count_stiffness: FloatProperty(
        name="Count Stiffness", default=0.0, min=0.0, soft_max=4.0,
        description="Anti-clumping pressure from particle counts")

    # -- Collision ---------------------------------------------------------------
    restitution: FloatProperty(name="Restitution", default=0.0, min=0.0, max=1.0)
    friction: FloatProperty(name="Friction", default=0.0, min=0.0, max=1.0)

    # -- Pressure ---------------------------------------------------------------
    pressure_solver: EnumProperty(
        name="Solver", items=_items(ProjectNonDivergentSolver), default="PCG")
    pressure_iterations: IntProperty(name="Iterations", default=200, min=1, max=4000)
    pressure_tolerance: FloatProperty(
        name="Tolerance", default=1.0e-5, min=0.0, precision=8)
    multigrid: EnumProperty(name="Multigrid", items=_items(MultiGrid), default="OFF")
    pressure_threshold: FloatProperty(
        name="Threshold", default=0.0, min=0.0, max=1.0,
        description="Relative pressure threshold (removes tiny pressures)")

    # -- Viscosity ---------------------------------------------------------------
    viscosity_enabled: BoolProperty(name="Enable Viscosity", default=False)
    viscosity_solver: EnumProperty(
        name="Solver", items=_items(ViscositySolver), default="PCG")
    viscosity_constant: FloatProperty(
        name="Viscosity", default=0.05, min=0.0, soft_max=100.0)
    viscosity_from_particles: BoolProperty(
        name="From Particles", default=False,
        description="Use per-particle viscosity attribute instead of the constant")
    viscosity_iterations: IntProperty(name="Iterations", default=40, min=1, max=1000)
    viscosity_tolerance: FloatProperty(
        name="Tolerance", default=1.0e-4, min=0.0, precision=8)

    # -- Surface -------------------------------------------------------------------
    surface_enabled: BoolProperty(name="Generate Surface", default=True)
    particle_radius_scale: FloatProperty(
        name="Particle Radius", default=1.2, min=0.5, soft_max=4.0)
    surface_filter_type: EnumProperty(
        name="Filter", items=_items(SdfFilterType), default="BOX")
    surface_filter_iterations: IntProperty(name="Filter Iterations",
                                           default=1, min=0, max=16)
    surface_filter_strength: FloatProperty(name="Filter Strength",
                                           default=0.7, min=0.0, max=1.0)

    # -- Whitewater -------------------------------------------------------------
    whitewater_enabled: BoolProperty(name="Enable Whitewater", default=False)
    max_whitewater: IntProperty(
        name="Max Whitewater", default=1_000_000, min=1024, max=20_000_000)
    ww_churn_decay: FloatProperty(name="Churn Decay", default=2.0, min=0.0)
    ww_churn_gain: FloatProperty(name="Churn Gain", default=1.0, min=0.0)
    ww_churn_threshold: FloatProperty(name="Churn Threshold", default=0.5, min=0.0)
    ww_emission_rate: FloatProperty(name="Emission Rate", default=2.0, min=0.0)
    ww_min_speed: FloatProperty(name="Min Speed", default=0.5, min=0.0)
    ww_life_min: FloatProperty(name="Life Min", default=1.0, min=0.0)
    ww_life_max: FloatProperty(name="Life Max", default=3.0, min=0.0)
    ww_bubble_drag: FloatProperty(name="Bubble Drag", default=4.0, min=0.0)
    ww_spray_drag: FloatProperty(name="Spray Drag", default=0.2, min=0.0)

    # -- Height source ------------------------------------------------------------
    height_enabled: BoolProperty(name="Height Fill", default=False,
                                 description="Fill the domain below a height")
    height_base: FloatProperty(name="Base Height", default=0.3)
    height_back_mode: EnumProperty(
        name="Back Mode", items=_items(SourceHeightBackMode), default="FLAT")
    height_velocity: FloatVectorProperty(name="Velocity", size=3,
                                         default=(0.0, 0.0, 0.0))
    height_emit_particles: BoolProperty(name="Emit Particles", default=True)

    # -- Output ---------------------------------------------------------------------
    liquid_view_mode: EnumProperty(
        name="Liquid View",
        items=LIQUID_VIEW_ITEMS,
        default="SURFACE",
        update=_apply_liquid_view_mode,
    )
    show_surface: BoolProperty(name="Surface Mesh", default=True)
    show_particles: BoolProperty(name="Particles", default=False)
    show_whitewater: BoolProperty(name="Whitewater", default=True)
    particle_display_fraction: FloatProperty(
        name="Display Fraction", default=1.0, min=0.01, max=1.0,
        description="Fraction of particles shown in the viewport")
    surface_attribute_mode: EnumProperty(
        name="Surface Attribute",
        items=LIQUID_ATTR_ITEMS,
        default="NONE",
        description="Color the liquid surface by a simulation attribute",
    )
    particle_attribute_mode: EnumProperty(
        name="Particle Attribute",
        items=PARTICLE_ATTR_ITEMS,
        default="NONE",
        description="Color liquid particles by a simulation attribute",
    )
    whitewater_attribute_mode: EnumProperty(
        name="Whitewater Attribute",
        items=WHITEWATER_ATTR_ITEMS,
        default="NONE",
        description="Color whitewater particles by a simulation attribute",
    )
    surface_material: PointerProperty(name="Surface Material",
                                      type=bpy.types.Material)
    whitewater_material: PointerProperty(name="Whitewater Material",
                                         type=bpy.types.Material)
    output_collection_name: StringProperty(
        name="Output Collection", default="FluidSim Output")

    # -- Custom snippets ----------------------------------------------------------
    snippets: CollectionProperty(type=FLUIDSIM_PG_snippet)
    active_snippet: IntProperty(default=0)

    # -- Diagnostics ---------------------------------------------------------------
    profiling_enabled: BoolProperty(name="Enable Profiling", default=False)
    profiling_sync: BoolProperty(
        name="Accurate GPU Timing", default=True,
        description="Synchronize CUDA around each phase for exact timings")
    show_overlay: BoolProperty(name="Viewport Overlay", default=True)
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
    del bpy.types.Object.fluidsim
    del bpy.types.Scene.fluidsim
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
