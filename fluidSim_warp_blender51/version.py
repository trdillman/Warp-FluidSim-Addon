"""Canonical FluidSim release identity."""

VERSION = (3, 2, 0)
VERSION_STRING = ".".join(str(part) for part in VERSION)
EXTENSION_ID = "fluidsim_warp"
BUILD_CHANNEL = "stable"
BUILD_LABEL = f"FluidSim Warp {VERSION_STRING}-{BUILD_CHANNEL}"
BLENDER_MIN_VERSION = (5, 2, 0)
VALIDATION_HOST_BLENDER_VERSION = (5, 2, 0)
CERTIFIED_BLENDER_VERSION = VALIDATION_HOST_BLENDER_VERSION
CERTIFIED_WARP_VERSION = "1.17.0"
CERTIFIED_WARP_DISTRIBUTION = "1.17.0+cu12"
# Oldest Warp the add-on accepts; newer 1.x releases load with a warning.
MIN_WARP_VERSION = "1.17.0"
SUPPORTED_PLATFORMS = ("windows-x64", "linux-x64")
# Platforms with a recorded GPU gate run. Linux is packaged but not yet certified.
VALIDATED_RUNTIME_PLATFORMS = ("windows-x64",)
CERTIFIED_PLATFORMS = VALIDATED_RUNTIME_PLATFORMS
