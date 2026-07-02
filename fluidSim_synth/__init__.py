"""Synthesized FluidSim-to-Warp reference package.

This package intentionally contains no original OpenCL or Houdini source files.
It contains distilled contracts, inventories, and target add-on requirements
derived from the provided implementation artifacts.
"""

from .fluid_meta import TOOL_METADATA, SOURCE_FACTS
from .kernel_manifest import KERNELS, KERNEL_GROUP_COUNTS
from .enums import REQUIRED_ENUMS
