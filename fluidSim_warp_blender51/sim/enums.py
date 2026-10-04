"""Solver enums ported from the original FluidSim enum tables.

Python IntEnum mirrors of the reference enums that the port uses. Values
must stay identical to the original tables (see git history) for fidelity.
No bpy imports here.
"""

from enum import IntEnum


class ParticleSampleType(IntEnum):
    """FLUIDSIM::Parameters::ParticleSampleType"""
    PIC = 0
    FLIP = 1


class ViscositySolver(IntEnum):
    """FLUIDSIM::Parameters::ViscositySolver"""
    JACOBI = 0
    PCG = 1


class ProjectNonDivergentSolver(IntEnum):
    """FLUIDSIM::Parameters::ProjectNonDivergentSolver"""
    JACOBI = 0
    PCG = 1


class TraceMode(IntEnum):
    """FLUIDSIM::Parameters::TraceMode / Enum_FluidSim_ParametersTraceMode"""
    NONE = 0
    SINGLE_SAMPLE = 1
    MID_POINT = 2
    MODIFIED_MID_POINT = 3
    RK3 = 4


class RelaxMode(IntEnum):
    """FLUIDSIM::Parameters::RelaxMode"""
    LOCAL_VOXEL = 0
    NEIGHBOR_VOXELS = 1


class SdfFilterType(IntEnum):
    """FLUIDSIM::Parameters::SdfFilterType (source spelling preserved)"""
    DIALATE = 0
    ERODE = 1
    BOX = 2
    MEAN_CURVATURE_FLOW = 3


class SurfaceResolutionMode(IntEnum):
    """Render/viewport surface grid resolution policy."""
    SIMULATION = 0      # use the simulation voxel size and legacy grid
    MULTIPLIER = 1      # voxel_size / surface_resolution_multiplier
    CUSTOM = 2          # explicit surface_voxel_size


class SurfaceUpdateCadence(IntEnum):
    """How often the expensive render surface is rebuilt."""
    SUBSTEP = 0         # legacy behavior: rebuild inside every solver substep
    FRAME = 1           # recommended: rebuild once at the end of each display frame
    MANUAL = 2          # no automatic rebuild; surface_mesh() can force one


class LimitBehavior(IntEnum):
    """CORE::Parameters::LimitBehavior"""
    MAINTAIN = 0
    PERCENT = 1
    ACCELERATION = 2


class Completion(IntEnum):
    """COMPUTE::Completion"""
    PASS = 0
    COMMIT = 1
    FINISH = 2


class BlendMode(IntEnum):
    """Enum_Core_ParametersBlendMode"""
    ADD = 0
    REPLACE = 1
    MAXIMUM = 2
    PULL = 3
    BLEND = 4


class SourceShapeType(IntEnum):
    """Enum_Core_SourceShape_Type"""
    SOURCE = 0
    COLLISION = 1
    SINK = 2
    ACTIVATION = 3
    CAMERA_FRUSTUM = 4


class SourceShapeSourceMode(IntEnum):
    """Enum_Core_SourceShape_SourceMode"""
    FIELD = 0
    PARTICLES = 1


class SourceEmissionMode(IntEnum):
    """User-facing emission lifecycle, separate from the solver-representation
    SourceShapeSourceMode (FIELD/PARTICLES).

    INITIAL_FILL: emit the source region once on the first enabled solver
        step after Initialize or Reset, then re-arm only on Initialize/Reset.
    CONTINUOUS_INFLOW: replenish the source region to its target particle
        occupancy every solver substep while Enabled.
    """
    INITIAL_FILL = 0
    CONTINUOUS_INFLOW = 1


class SourceGeometryMode(IntEnum):
    """Blender source geometry contract."""
    ANALYTIC = 0
    MESH = 1


class MeshEvaluationMode(IntEnum):
    """Evaluated-mesh capture cadence."""
    FIRST_SIM_FRAME = 0
    EVERY_SIM_FRAME = 1


class MeshSinkRegionMode(IntEnum):
    """Mesh sink membership policy."""
    SOLID = 0
    SURFACE = 1


class SourceOperation(IntEnum):
    """Operation performed by a normalized Phase 2 source plan."""
    LIQUID = 0
    SINK = 1
    FORCE = 2
    COLLISION = 3
    ACTIVATION = 4
    PRESSURE = 5
    SCALAR_FIELD = 6
    VECTOR_FIELD = 7
    STAGGERED_VELOCITY = 8


class SourceRegionKind(IntEnum):
    """Geometry/input representation, independent of source operation."""
    ANALYTIC_SDF = 0
    MESH_SOLID = 1
    MESH_SHELL = 2
    POINTS = 3
    VOLUME_GRID = 4
    HEIGHT = 5


class SourceLifecycle(IntEnum):
    """Runtime source lifecycle."""
    ONCE = 0
    TOP_UP = 1
    DIRECT_APPEND = 2
    EVERY_FRAME = 3


class SourceEvaluationCadence(IntEnum):
    """Geometry/payload capture cadence."""
    STATIC = 0
    FIRST_SIM_FRAME = 1
    EVERY_SIM_FRAME = 2


class SourcePayloadKind(IntEnum):
    """Payload transported by a normalized source plan."""
    NONE = 0
    LIQUID_ATTRIBUTES = 1
    PRESSURE = 2
    SCALAR = 3
    VECTOR = 4
    STAGGERED_VECTOR = 5
    MASK = 6


class MeshSourceRegion(IntEnum):
    """Truthful mesh-emitter region; never silently falls back."""
    SOLID_VOLUME = 0
    SURFACE_SHELL = 1


class SourceShapeShape(IntEnum):
    """Enum_Core_SourceShape_Shape"""
    SPHERE = 0
    CYLINDER = 1
    BOX = 2
    PYRAMID = 3
    CONE = 4
    PLANE = 5


class SourceShapeForceType(IntEnum):
    """Enum_Core_SourceShape_ForceType"""
    OFF = 0
    DIRECTIONAL = 1
    OMNIDIRECTIONAL = 2
    SPIN = 3
    RADIAL = 4


class BlenderForceFieldType(IntEnum):
    """Native Blender field kinds understood by the GPU proxy kernel."""

    WIND = 1
    FORCE = 2
    VORTEX = 3
    MAGNETIC = 4
    HARMONIC = 5
    CHARGE = 6
    LENNARD_JONES = 7
    TEXTURE = 8
    CURVE_GUIDE = 9
    BOID = 10
    TURBULENCE = 11
    DRAG = 12
    FLUID_FLOW = 13


class BlenderForceFieldShape(IntEnum):
    POINT = 0
    LINE = 1
    PLANE = 2
    SURFACE = 3
    EVERY_POINT = 4


class BlenderTextureForceMode(IntEnum):
    RGB = 0
    GRADIENT = 1
    CURL = 2


class SourceHeightBackMode(IntEnum):
    """Enum_Core_ParametersSourceHeight_BackMode"""
    INFINITE = 0
    FLAT = 1
    PROFILE = 2


class RampToScaleFieldCombine(IntEnum):
    """Enum_Core_ParametersRampToScaleFieldCombine"""
    MULTIPLY = 0
    MAX = 1
    MIN = 2


# Blender EnumProperty item helpers

# User-facing effects for enums exposed by the add-on UI.  Blender displays
# these strings while users hover dropdown choices, so repeating the label is
# not enough: each entry states what selecting that value changes.
_ENUM_DESCRIPTIONS = {
    ParticleSampleType: {
        "PIC": "Use direct grid velocity on particles for smoother, more stable, but more dissipative motion",
        "FLIP": "Apply grid velocity changes to particles to preserve energy and detail, with more noise",
    },
    ViscositySolver: {
        "JACOBI": "Use simple fixed-point viscosity iterations; predictable but usually slower to converge",
        "PCG": "Use preconditioned conjugate gradients for faster viscosity convergence on most scenes",
    },
    ProjectNonDivergentSolver: {
        "JACOBI": "Use simple fixed-point pressure iterations; predictable but usually slower to converge",
        "PCG": "Use preconditioned conjugate gradients for faster pressure convergence on most scenes",
    },
    TraceMode: {
        "NONE": "Move particles from their current velocity without sampling a traced grid trajectory",
        "SINGLE_SAMPLE": "Sample grid velocity once for the cheapest first-order trajectory",
        "MID_POINT": "Sample at the trajectory midpoint for better curved-path accuracy at extra cost",
        "MODIFIED_MID_POINT": "Blend particle and midpoint grid velocity for a more stable midpoint trajectory",
        "RK3": "Use three-stage Runge-Kutta tracing for the best available path accuracy at the highest sampling cost",
    },
    RelaxMode: {
        "LOCAL_VOXEL": "Redistribute particles using neighbors in the same voxel for lower cost and more local changes",
        "NEIGHBOR_VOXELS": "Include adjacent voxels for smoother cross-cell spacing at higher search cost",
    },
    SdfFilterType: {
        "DIALATE": "Expand negative SDF regions to thicken and connect the extracted liquid surface",
        "ERODE": "Contract negative SDF regions to thin the surface and remove small features",
        "BOX": "Average nearby SDF samples to soften small surface detail",
        "MEAN_CURVATURE_FLOW": "Smooth by estimated curvature for rounder features with less box-filter character",
    },
    SurfaceResolutionMode: {
        "SIMULATION": "Use the simulation voxel size and shared grid for the lowest additional memory cost",
        "MULTIPLIER": "Divide simulation voxel size by Resolution Multiplier for independently finer or coarser surfaces",
        "CUSTOM": "Use the explicit Surface Voxel Size for direct control of render-surface resolution",
    },
    SurfaceUpdateCadence: {
        "SUBSTEP": "Rebuild after every solver substep for the freshest surface at the highest cost",
        "FRAME": "Rebuild once per display frame for responsive playback with current frame-level results",
        "MANUAL": "Do not rebuild automatically; keep the previous mesh until a caller explicitly requests a surface",
    },
    LimitBehavior: {
        "MAINTAIN": "Hard-clamp velocity magnitude to Max Speed",
        "PERCENT": "Remove the Limit Percent fraction of speed above Max Speed each step",
        "ACCELERATION": "Limit how quickly velocity can be reduced toward Max Speed rather than clamping immediately",
    },
    BlendMode: {
        "ADD": "Add the source payload to the existing target value",
        "REPLACE": "Overwrite the target with the source payload",
        "MAXIMUM": "Keep whichever value is larger between the target and source payload",
        "PULL": "Move the existing target toward the source payload instead of replacing it immediately",
        "BLEND": "Interpolate the existing target and source payload using Blend Amount",
    },
    SourceShapeSourceMode: {
        "FIELD": "Apply the Field Operation (pressure by default) inside the region without creating liquid",
        "PARTICLES": "Emit liquid particles into the selected source region",
    },
    SourceEmissionMode: {
        "INITIAL_FILL": "Fill the source region once on the first enabled step after initialization or reset",
        "CONTINUOUS_INFLOW": "Replenish the source region every solver substep while the source remains enabled",
    },
    SourceShapeShape: {
        "SPHERE": "Use the object's transformed sphere interior as the analytic region",
        "CYLINDER": "Use the object's transformed cylinder interior as the analytic region",
        "BOX": "Use the object's transformed box interior as the analytic region",
        "PYRAMID": "Use the object's transformed pyramid interior as the analytic region",
        "CONE": "Use the object's transformed cone interior as the analytic region",
        "PLANE": "Use the object's transformed plane half-space as the analytic region",
    },
    SourceShapeForceType: {
        "OFF": "Apply no source-shape force inside the region",
        "DIRECTIONAL": "Accelerate liquid along Force Direction inside the region",
        "OMNIDIRECTIONAL": "Accelerate liquid radially from the region center in every direction",
        "SPIN": "Rotate liquid around the Force Direction axis inside the region",
        "RADIAL": "Accelerate liquid along the analytic shape's local radial direction",
    },
    SourceHeightBackMode: {
        "INFINITE": "Extend the height region backward without a finite back surface",
        "FLAT": "Close the height region with a flat back surface at Base Height",
        "PROFILE": "Shape the back surface from the selected grayscale Height Profile",
    },
}


def enum_items(enum_cls, descriptions=None):
    """Build Blender EnumProperty items with user-facing selection effects."""
    merged_descriptions = dict(_ENUM_DESCRIPTIONS.get(enum_cls, {}))
    if descriptions:
        merged_descriptions.update(descriptions)
    items = []
    for member in enum_cls:
        label = member.name.replace("_", " ").title()
        description = merged_descriptions.get(
            member.name,
            f"Select {label} for this setting",
        )
        items.append((member.name, label, description, member.value))
    return items
