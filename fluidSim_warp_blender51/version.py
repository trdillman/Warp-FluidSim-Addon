"""Canonical FluidSim release identity."""

VERSION = (4, 0, 0)
VERSION_STRING = ".".join(str(part) for part in VERSION)
EXTENSION_ID = "fluidsim_warp"
BUILD_CHANNEL = "stable"
BUILD_LABEL = f"FluidSim Warp {VERSION_STRING}-{BUILD_CHANNEL}"
BLENDER_MIN_VERSION = (5, 2, 0)
VALIDATION_HOST_BLENDER_VERSION = (5, 2, 0)
CERTIFIED_BLENDER_VERSION = VALIDATION_HOST_BLENDER_VERSION
CERTIFIED_WARP_VERSION = "1.18.0"
CERTIFIED_WARP_DISTRIBUTION = "1.18.0+cu13"
# Oldest Warp the add-on accepts. Older-than-certified and newer 1.x releases
# load with a warning.
MIN_WARP_VERSION = "1.17.0"
SUPPORTED_PLATFORMS = ("windows-x64", "linux-x64")
# Platforms with a recorded GPU gate run. Linux is packaged but not yet certified.
VALIDATED_RUNTIME_PLATFORMS = ("windows-x64",)
CERTIFIED_PLATFORMS = VALIDATED_RUNTIME_PLATFORMS
# The certified Warp wheels are CUDA 13 builds. NVIDIA's R580 driver series is
# the first with the CUDA 13.0 driver API, reported by cuDriverGetVersion as
# 13000; older drivers keep working with Warp 1.17.0.
CERTIFIED_DRIVER_MIN = (13, 0)
CERTIFIED_DRIVER_LABEL = "NVIDIA R580-series"
