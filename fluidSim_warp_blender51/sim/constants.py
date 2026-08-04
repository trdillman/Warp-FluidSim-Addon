"""FluidSim Warp port constants.

Voxel-type bits and solver constants mirror the synthesized reference
(`Bit_Core_VoxelType`, `Const_Core`, `Const_FluidSim`).
No bpy imports allowed in this module.
"""

TOOL_NAME = "fluidSim"
TOOL_NAME_LABEL = "FluidSim"
TOOL_VERSION = "1.0.240-beta"
TOOL_BUILD = "fluidSim1.0.240-collider-material-response-stflip-beta"
TOOL_PROFILER = "fluidSim_profiler::1.0"
TOOL_PROFILER_LABEL = "FluidSim Profiler 1.0"
BLENDER_TARGET = "5.1"
BACKEND_TARGET = "NVIDIA Warp / CUDA"

# ---------------------------------------------------------------------------
# Voxel type bit masks (Bit_Core_VoxelType)
# ---------------------------------------------------------------------------
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

# Const_Core
REDUCE_CHUNK = 4

# Const_FluidSim (0 == derive at runtime)
PARTICLE_SEPARATION_SIZE = 0.0
MAX_PARTICLE_COUNT = 0

# ---------------------------------------------------------------------------
# Sparse tiled leaf grid: leaves are 8x8x8 voxel tiles in a flat leaf table.
# ---------------------------------------------------------------------------
LEAF_DIM = 8
LEAF_SHIFT = 3          # log2(LEAF_DIM)
LEAF_MASK = LEAF_DIM - 1
LEAF_VOLUME = LEAF_DIM ** 3  # 512 voxels per leaf

# Default allocation policy
DEFAULT_MAX_PARTICLES = 2_000_000
DEFAULT_MAX_WHITEWATER = 1_000_000
DEFAULT_LEAF_CAPACITY = 8192          # 8192 leaves * 512 voxels = 4.19M voxels
DEFAULT_PARTICLES_PER_VOXEL = 8
DEFAULT_RELAX_BIN_SAMPLE_LIMIT = 32

# Numerical defaults
DEFAULT_VOXEL_SIZE = 0.05
DEFAULT_CFL = 1.0
DEFAULT_FLIP_RATIO = 0.97
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
DEFAULT_PRESSURE_ITERATIONS = 200
DEFAULT_PRESSURE_TOLERANCE = 1.0e-5
DEFAULT_VISCOSITY_ITERATIONS = 40
DEFAULT_VISCOSITY_TOLERANCE = 1.0e-4

# Spatiotemporal FLIP (Braun et al., ACM TOG 45(4) Art. 76).
#
# ST_EPS_RHO is Eq. (15)'s eps_rho: beta_f = alpha_f / max(rho(phi_f), eps_rho).
# The paper gives no value.  Reconstruction, relative to rho_liquid = 1: the
# max() is a divide-by-zero and PPE-conditioning guard, and the resulting
# condition number scales as rho_liquid / eps_rho.  A value near 0.05 makes that
# ratio 20 and turns the guard into a physical cap that stiffens the free
# surface Eq. (17)'s rho_g -> 0 limit exists to free; a value near 1e-6 makes it
# 1e6 and diagonally-preconditioned PCG stalls at the paper's 1e-4 tolerance.
# 1e-4 sits four orders below liquid (physically negligible) and four orders
# above the float32 reciprocal risk.  st_face_beta keeps its own absolute
# max(eps_rho, 1e-8) as the true zero guard.
ST_EPS_RHO = 1.0e-4

# ST_EPS_MASS is Eq. (9)'s eps_m: a face is under-sampled, and its u* = P_f/m_f
# meaningless, when m_f <= eps_m.  The paper quotes 1e-9; this port uses 1e-8
# because that is the threshold k_normalize_mac already applies when it forms
# u* = P/m.  The invariant that matters is that the velocity-extrapolation
# validity mask and the divide gate agree -- a face valid to one and invalid to
# the other either keeps an amplified garbage velocity or gets flooded over.
ST_EPS_MASS = 1.0e-8

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
