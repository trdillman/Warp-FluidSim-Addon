"""FluidSim Warp port constants.

Voxel-type bits and solver constants mirror the synthesized reference
(`Bit_Core_VoxelType`, `Const_Core`, `Const_FluidSim`).
No bpy imports allowed in this module.
"""

from ..version import BUILD_LABEL, VERSION_STRING

TOOL_NAME = "fluidSim"
TOOL_NAME_LABEL = "FluidSim"
TOOL_VERSION = VERSION_STRING
TOOL_BUILD = BUILD_LABEL
TOOL_PROFILER = "fluidSim_profiler::1.0"
TOOL_PROFILER_LABEL = "FluidSim Profiler 1.0"
BLENDER_TARGET = "5.1+"
BACKEND_TARGET = "NVIDIA Warp / CUDA"

# Voxel type bit masks (Bit_Core_VoxelType)
VOXEL_NONE = 0
VOXEL_COLLISION = 1
VOXEL_SINK = 2
VOXEL_FLUID = 4
VOXEL_SOURCE_REGION_TRACKING = 8
VOXEL_X = 16
VOXEL_Y = 32
VOXEL_Z = 64
VOXEL_SDF = 128
VOXEL_GRADIENT = 256
VOXEL_UNUSED_1 = 512
VOXEL_COLLISION_X = 1024
VOXEL_COLLISION_Y = 2048
VOXEL_COLLISION_Z = 4096
VOXEL_ACTIVE = 8192
VOXEL_COLOR = 16384

REDUCE_CHUNK = 4

# Const_FluidSim (0 == derive at runtime)
PARTICLE_SEPARATION_SIZE = 0.0
MAX_PARTICLE_COUNT = 0

# Sparse tiled leaf grid: leaves are 8x8x8 voxel tiles in a flat leaf table.
LEAF_DIM = 8
LEAF_SHIFT = 3          # log2(LEAF_DIM)
LEAF_MASK = LEAF_DIM - 1
LEAF_VOLUME = LEAF_DIM ** 3  # 512 voxels per leaf

# Default allocation policy
DEFAULT_MAX_PARTICLES = 2_000_000
DEFAULT_MAX_WHITEWATER = 1_000_000
DEFAULT_LEAF_CAPACITY = 8192          # 8192 leaves * 512 voxels = 4.19M voxels
DEFAULT_PARTICLES_PER_VOXEL = 8
# Hard ceiling on liquid occupancy: the RNA max of both authored density
# settings, every host-side clamp, and the emission kernel's per-voxel loop.
MAX_PARTICLES_PER_VOXEL = 64
DEFAULT_RELAX_BIN_SAMPLE_LIMIT = 32

# Numerical defaults
DEFAULT_VOXEL_SIZE = 0.05
DEFAULT_CFL = 1.0
DEFAULT_FLIP_RATIO = 0.97
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
DEFAULT_PRESSURE_ITERATIONS = 200
DEFAULT_PRESSURE_TOLERANCE = 1.0e-5
# Measured on dam_break at DEFAULT_VISCOSITY_TOLERANCE with the ceiling
# lifted to 256: u converges in 35-41 iterations, v in 34-39, w in
# 41-46. The previous default of 40 sat below w's *minimum*, so w was
# truncated on every solve and reported nothing about it. This is a
# ceiling, not a workload -- the PCG loop is a device-side
# `wp.capture_while` on the residual test, so a solve that converges at
# 42 costs 42 iterations regardless of what the ceiling allows.
DEFAULT_VISCOSITY_ITERATIONS = 64
DEFAULT_VISCOSITY_TOLERANCE = 1.0e-4

# Timer tracks (profiler contract): compute_0..15 phase assignment
TIMER_PHASE_SLOTS = {
    "sourcing": 0,
    "indexing": 1,
    "p2g": 2,
    "forces": 3,
    "viscosity": 4,
    "pressure": 5,
    "g2p": 6,
    "advection": 7,
    "collision": 8,
    "sink": 9,
    "relax": 10,
    "surface": 11,
    "whitewater": 12,
    "custom": 13,
    "lifecycle": 14,
    "misc": 15,
}
