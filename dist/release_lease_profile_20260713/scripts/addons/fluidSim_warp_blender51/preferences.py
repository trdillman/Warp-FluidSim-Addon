"""Add-on preferences: device selection and runtime information."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty


def _probe_warp():
    """Report Warp/CUDA availability without initializing a solver."""
    try:
        import warp as wp

        wp.init()
        if wp.is_cuda_available():
            dev = wp.get_device("cuda:0")
            return True, f"Warp {wp.config.version} - {dev.name}"
        return False, f"Warp {wp.config.version} - no CUDA device"
    except Exception as exc:  # noqa: BLE001
        return False, f"Warp unavailable: {exc}"


class FLUIDSIM_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    device: StringProperty(
        name="Device", default="cuda:0",
        description="Warp CUDA device identifier")
    verbose: BoolProperty(
        name="Verbose Kernel Loading", default=False,
        description="Print Warp module load timings to the console")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "device")
        layout.prop(self, "verbose")
        ok, msg = _probe_warp()
        layout.label(text=msg, icon="CHECKMARK" if ok else "ERROR")
        layout.label(
            text="Solver runs GPU-only through NVIDIA Warp/CUDA; "
                 "first initialization compiles kernels (cached afterwards).")


def register():
    bpy.utils.register_class(FLUIDSIM_AP_preferences)


def unregister():
    bpy.utils.unregister_class(FLUIDSIM_AP_preferences)
