"""Solver enums ported from the synthesized FluidSim enum tables.

Python IntEnum mirrors of the reference `REQUIRED_ENUMS`. Values must stay
identical to the synthesized tables for fidelity. No bpy imports here.
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


class MultiGrid(IntEnum):
    """FLUIDSIM::Parameters::MultiGrid"""
    OFF = 0
    V_CYCLE = 1
    HALF_V_CYCLE = 2


class MultiGridResolutionDecomposition(IntEnum):
    """FLUIDSIM::Parameters::MultiGridResolutionDecomposition"""
    FULL = 0
    HALF = 1
    SINGLE = 2


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


class PressureDownloadProperty(IntEnum):
    """FLUIDSIM::Parameters::PressureDownloadProperty"""
    PRESSURE = 0
    PRESSURE_GRADIENT = 1
    DIVERGENCE = 2


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


class ParticleAttributeDataType(IntEnum):
    """FLUIDSIM::Parameters::ParticleAttributeDataType"""
    INT = 0
    FLOAT = 1
    HALF = 2


class FieldPrecision(IntEnum):
    """CORE::Parameters::FieldPrecision"""
    NONE = -1
    BIT16 = 0
    BIT32 = 1


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


class ComputeAPI(IntEnum):
    """COMPUTE::API"""
    NONE = -2
    AUTO = -1
    OPENCL = 0
    METAL = 1
    CUDA = 2


class DeviceType(IntEnum):
    """COMPUTE::Device::Type"""
    NONE = -1
    CPU = 0
    GPU = 1
    UNKNOWN = 2


class CustomKernelType(IntEnum):
    """CORE::Solver::CustomOpenClType (ported to Warp snippets)"""
    KERNEL = 0
    FUNCTION = 1


class CustomKernelOrder(IntEnum):
    """CORE::Solver::CustomOpenClOrder"""
    FIRST_STEP = 0
    BEFORE_SOURCING = 1
    BEFORE_FORCES = 2
    BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION = 3
    FINAL_STEP = 4


class CustomKernelRunOver(IntEnum):
    """CORE::Solver::CustomOpenClRunOver"""
    VOXEL_TILE = 0
    VOXEL_LINEAR = 1
    PARTICLES = 2


class CustomKernelArgumentMode(IntEnum):
    """CORE::Solver::CustomOpenClArgumentMode"""
    FIELD = 0
    CONSTANT = 1


class CustomKernelFieldPermission(IntEnum):
    """CORE::Solver::CustomOpenClArgumentFieldPermissions"""
    WRITE = 0
    READ = 1
    READ_WRITE = 2


class CustomKernelConstantType(IntEnum):
    """CORE::Solver::CustomOpenClArgumentConstantType"""
    FLOAT = 0
    INT = 1


class TimeStepType(IntEnum):
    """Enum_Core_ParametersTimeStepType"""
    OFF = 0
    FRAME_STEP = 1
    TIME_STEP = 2
    SOLVER_FRAME_STEP = 3
    SOLVER_TIME_STEP = 4


class SourceVolumeType(IntEnum):
    """Enum_Core_ParametersSourceVolumeType"""
    NORMAL = 0
    COLLISION = 1
    SINK = 2


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


class SourceShapeNoiseSpace(IntEnum):
    """Enum_Core_SourceShape_NoiseSpace"""
    WORLD = 0
    OBJECT = 1


class SourceHeightBackMode(IntEnum):
    """Enum_Core_ParametersSourceHeight_BackMode"""
    INFINITE = 0
    FLAT = 1
    PROFILE = 2


class RampToScaleFieldType(IntEnum):
    """Enum_Core_ParametersRampToScaleFieldType"""
    OFF = 0
    PARTICLE_ATTRIBUTE = 1
    FIELD = 2
    SOURCE_VOLUME_MASK = 3
    SPEED = 4
    X_AXIS = 5
    Y_AXIS = 6
    Z_AXIS = 7


class RampToScaleFieldCombine(IntEnum):
    """Enum_Core_ParametersRampToScaleFieldCombine"""
    MULTIPLY = 0
    MAX = 1
    MIN = 2


class GravityFieldsCombine(IntEnum):
    """Enum_Core_ParametersForcesGravityFieldsCombine"""
    ADD = 0
    MINIMUM = 1
    MAXIMUM = 2
    AVERAGE = 3


class TurbulenceType(IntEnum):
    """Enum_FluidSim_Parameters_ForcesTurbulenceType"""
    FAST_PERLIN = 0
    PERLIN = 1


# Blender EnumProperty item helpers -----------------------------------------

def enum_items(enum_cls, descriptions=None):
    """Build Blender EnumProperty items from an IntEnum class."""
    descriptions = descriptions or {}
    items = []
    for member in enum_cls:
        label = member.name.replace("_", " ").title()
        items.append((member.name, label, descriptions.get(member.name, label), member.value))
    return items
