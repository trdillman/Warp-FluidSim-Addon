"""Project and source metadata synthesized from FluidSim references."""

TOOL_METADATA = {'TOOL_NAME': 'fluidSim', 'TOOL_NAME_PRODUCT': 'fluidSim', 'TOOL_NAME_LABEL': 'FluidSim', 'TOOL_NAME_MODULE': 'FLUIDSIM', 'TOOL_NAME_ACRONYM': 'PS', 'TOOL_VERSION': '1.0.226', 'TOOL_VERSION_MAJOR_REVISION': '1.0', 'TOOL_VERSION_MAJOR': '1', 'TOOL_VERSION_REVISION': '0', 'TOOL_VERSION_BUILD': '226', 'TOOL_BUILD': 'fluidSim1.0.226', 'TOOL_RELEASE': 'fluidSim1.0', 'TOOL_MODULE': 'FLUIDSIM_1_0', 'TOOL_RELEASE_TIME': '1778290748', 'TOOL_PROFILER': 'fluidSim_profiler::1.0', 'TOOL_PROFILER_LABEL': 'FluidSim Profiler 1.0'}

SOURCE_FACTS = {'canonical_solver_source': 'fluidSim.cl', 'uploaded_alias': 'liquid.cl', 'line_count': 20640, 'byte_count': 772693, 'kernel_count': 188, 'core_kernel_count': 80, 'fluid_kernel_count': 108, 'local_memory_declaration_count': 38, 'barrier_count': 47, 'atomic_reference_count': 66, 'half_token_count': 37, 'field_precision_macro': '__CORE_16BIT_FIELDS__', 'source_namespace_aliases': {'paradigm': 'fluidSim'}, 'known_audit_items': ['coreConvertFieldStaggeredVectorToMagnitude: verify Y/Z component reads and the Z half-scale assignment during porting.', 'core custom-kernel parameter extraction in the Houdini wrapper returns after the first snippet; decide whether to preserve or correct for multiple snippets.', 'Replace Houdini hou/PySide node UI, hscript commands, and license operations with Blender 5.1 equivalents or omit where non-functional.']}

BLENDER_TARGET = "5.1"
BACKEND_TARGET = "NVIDIA Warp / CUDA"
CLEAN_ROOM_NOTE = (
    "No original .cl or Houdini Python files are embedded in this package; "
    "only synthesized requirements, manifests, and contracts are present."
)
