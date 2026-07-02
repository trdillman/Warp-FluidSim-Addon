"""Custom kernel/user-extension contract synthesized from the original UI wrapper.

The Blender port should expose an equivalent capability for advanced users,
implemented in Warp/Python rather than OpenCL C.
"""

from dataclasses import dataclass, field
from enum import Enum

class SnippetType(Enum):
    KERNEL = "kernel"
    FUNCTION = "function"

class SnippetOrder(Enum):
    FIRST_STEP = "first_step"
    BEFORE_SOURCING = "before_sourcing"
    BEFORE_FORCES = "before_forces"
    BEFORE_PROJECT_AND_ADVECTION = "before_project_non_divergent_and_advection"
    FINAL_STEP = "final_step"

class RunOver(Enum):
    VOXEL_TILE = "voxel_tile"
    VOXEL_LINEAR = "voxel_linear"
    PARTICLES = "particles"

class ArgumentMode(Enum):
    FIELD = "field"
    CONSTANT = "constant"

class FieldPermission(Enum):
    WRITE = "write"
    READ = "read"
    READ_WRITE = "read_write"

class ConstantType(Enum):
    FLOAT = "float"
    INT = "int"

@dataclass(frozen=True)
class CustomArgumentSpec:
    mode: ArgumentMode
    field: str | None = None
    field_permission: FieldPermission | None = None
    constant_type: ConstantType | None = None
    constant_name: str | None = None
    constant_value: float | int | None = None

@dataclass
class CustomSnippetSpec:
    enabled: bool
    snippet_type: SnippetType
    order: SnippetOrder
    run_over: RunOver
    kernel_name: str
    code: str
    arguments: list[CustomArgumentSpec] = field(default_factory=list)

FIELD_EXPANSION_RULES = {
    "vector_like_name_substrings": ["velocity", "position"],
    "vector_like_suffixes": ["x", "y", "z"],
    "color_like_name_substrings": ["color"],
    "color_like_suffixes": ["r", "g", "b"],
    "voxel_type_field": "voxelType",
    "particle_position_field": "particlePosition",
    "default_scalar_type": "field_t",
}

RUN_OVER_CONTEXT = {
    RunOver.VOXEL_TILE: {
        "thread_domain": "leaf-local voxel tile plus leaf index",
        "available_indices": ["local_x", "global_y", "global_z", "leaf_index", "flat_voxel_index", "world_position"],
        "collision_skip": "skip if voxelType marks collision and voxelType is part of snippet args",
    },
    RunOver.VOXEL_LINEAR: {
        "thread_domain": "flat voxel index",
        "available_indices": ["flat_voxel_index"],
        "collision_skip": "skip if voxelType marks collision and voxelType is part of snippet args",
    },
    RunOver.PARTICLES: {
        "thread_domain": "flat particle index",
        "available_indices": ["particle_index"],
        "collision_skip": "not implicit",
    },
}

STANDARD_TIME_CONTEXT = ["frame", "time", "real_time_step", "solver_time_step", "time_scale", "substeps"]

PORTING_DECISION = (
    "Implement snippets as controlled Warp/Python extension points. Avoid raw arbitrary GPU code injection unless a safe compilation "
    "strategy is deliberately added. Preserve field/constant binding semantics and run-over domains."
)
