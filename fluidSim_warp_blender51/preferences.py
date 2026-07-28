"""Add-on preferences: device selection and runtime information."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty


_WARP_PROBE_RESULT = None


def configured_device():
    """The device the user selected, or the CUDA default.

    Sole owner of that lookup. Two callers previously open-coded it and a
    third ignored it entirely, so the preferences panel could report a
    different device than the solver was running on.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        if prefs.device:
            return prefs.device
    except (KeyError, AttributeError):
        pass
    return "cuda:0"


def _probe_warp(*, force=False):
    """Report Warp/CUDA availability, cached across preference redraws."""
    global _WARP_PROBE_RESULT
    if _WARP_PROBE_RESULT is not None and not force:
        return _WARP_PROBE_RESULT
    try:
        import warp as wp

        wp.init()
        if wp.is_cuda_available():
            # Probing cuda:0 while the solver runs on the selected device made
            # this panel's checkmark report a card that was never in use.
            dev = wp.get_device(configured_device())
            result = (True, f"Warp {wp.config.version} - {dev.name}")
        else:
            result = (False, f"Warp {wp.config.version} - no CUDA device")
    except Exception as exc:  # noqa: BLE001
        result = (False, f"Warp unavailable: {exc}")
    _WARP_PROBE_RESULT = result
    return result


class FLUIDSIM_OT_redetect_device(bpy.types.Operator):
    """Explicitly refresh the cached Warp/CUDA device status."""

    bl_idname = "fluidsim.redetect_device"
    bl_label = "Re-detect Device"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        ok, message = _probe_warp(force=True)
        self.report({"INFO" if ok else "WARNING"}, message)
        return {"FINISHED"}


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
        row = layout.row(align=True)
        row.label(text=msg, icon="CHECKMARK" if ok else "ERROR")
        row.operator("fluidsim.redetect_device", text="", icon="FILE_REFRESH")
        layout.label(
            text="Solver runs GPU-only through NVIDIA Warp/CUDA; "
                 "first initialization compiles kernels (cached afterwards).")


CLASSES = (FLUIDSIM_OT_redetect_device, FLUIDSIM_AP_preferences)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
