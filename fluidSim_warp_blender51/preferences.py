"""Add-on preferences: Warp installation and CUDA device selection."""

from __future__ import annotations

import os

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty

from .runtime.warp_runtime import (
    cuda_device_items,
    probe_runtime,
    runtime_status_for_device,
)
from .version import CERTIFIED_WARP_VERSION
from .warp_installer import (
    foreign_warp_block_reason,
    get_installed_warp_info,
    install_warp,
    owned_install_status,
)

_DEVICE_ITEMS_CACHE: list[tuple[str, str, str]] = []


def _device_items(_self, _context):
    """Keep dynamic enum strings alive for Blender's RNA callback."""
    global _DEVICE_ITEMS_CACHE
    _DEVICE_ITEMS_CACHE = cuda_device_items()
    return _DEVICE_ITEMS_CACHE


def _preferences(context=None):
    context = bpy.context if context is None else context
    try:
        return context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return None


def configured_device() -> str:
    """Return a detected selected device, falling back to the first CUDA GPU."""
    probe = probe_runtime()
    aliases = tuple(device.alias for device in probe.cuda_devices)
    prefs = _preferences()
    selected = str(getattr(prefs, "device", "") or "")
    if selected in aliases:
        return selected
    return aliases[0] if aliases else ""


def set_configured_device(device: str) -> bool:
    """Store one selected CUDA alias through the sole preferences owner."""
    prefs = _preferences()
    if prefs is None:
        return False
    prefs.device = str(device)
    return True


def effect_authoring_enabled(context=None) -> bool:
    """Whether the developer 'custom effect authoring' preference is enabled.

    Gates the Warp-code authoring workspace only; the artist-facing Custom
    Effects stack stays available regardless.  Defaults to ``False`` when
    preferences are unavailable (add-on not registered, or a stub context in
    tests), so the developer surface is opt-in and never leaks on by accident.
    """
    prefs = _preferences(context)
    return bool(getattr(prefs, "enable_effect_authoring", False))


def _apply_developer_override(prefs) -> None:
    if os.environ.get("FLUIDSIM_ALLOW_EXTERNAL_WARP") != "1":
        return
    path = str(getattr(prefs, "developer_warp_path", "") or "").strip()
    if not path:
        return
    from .warp_discovery import enable_developer_override

    enable_developer_override(path)


def _refresh_runtime(prefs):
    _apply_developer_override(prefs)
    probe = probe_runtime(force=True)
    aliases = tuple(device.alias for device in probe.cuda_devices)
    if prefs is not None and aliases and getattr(prefs, "device", "") not in aliases:
        prefs.device = aliases[0]
    selected = configured_device()
    return probe, runtime_status_for_device(selected, probe=probe)


class FLUIDSIM_OT_install_warp(bpy.types.Operator):
    """Download and install the certified NVIDIA Warp runtime from PyPI."""

    bl_idname = "fluidsim.install_warp"
    bl_label = "Install Warp"
    bl_description = (
        "Download and install the certified NVIDIA Warp runtime. "
        "Requires an internet connection"
    )
    bl_options = {"INTERNAL"}

    force_reinstall: BoolProperty(
        name="Force Reinstall",
        default=False,
        description="Reinstall even if already installed",
    )

    def execute(self, context):
        def progress(msg: str) -> None:
            self.report({"INFO"}, msg)

        result = install_warp(
            force_reinstall=self.force_reinstall,
            progress_callback=progress,
        )

        if result.success:
            self.report({"INFO"}, result.message)
            if not result.already_installed and not self.force_reinstall:
                prefs = _preferences(context)
                _refresh_runtime(prefs)
        else:
            self.report({"ERROR"}, result.message)

        return {"FINISHED"} if result.success else {"CANCELLED"}

    def invoke(self, context, _event):
        return self.execute(context)


class FLUIDSIM_OT_refresh_runtime_probe(bpy.types.Operator):
    """Refresh Warp import state and the CUDA device list."""

    bl_idname = "fluidsim.refresh_runtime_probe"
    bl_label = "Refresh Runtime Detection"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs = _preferences(context)
        _probe, (ready, message) = _refresh_runtime(prefs)
        self.report({"INFO" if ready else "WARNING"}, message)
        return {"FINISHED"}


class FLUIDSIM_OT_redetect_device(bpy.types.Operator):
    """Compatibility alias for the pre-1.0.244 refresh operator."""

    bl_idname = "fluidsim.redetect_device"
    bl_label = "Re-detect Device"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs = _preferences(context)
        _probe, (ready, message) = _refresh_runtime(prefs)
        self.report({"INFO" if ready else "WARNING"}, message)
        return {"FINISHED"}


class FLUIDSIM_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    device: EnumProperty(
        name="CUDA Device",
        items=_device_items,
        description="CUDA device used for FluidSim allocations and kernel launches",
    )
    verbose: BoolProperty(
        name="Verbose Kernel Loading",
        default=False,
        description=(
            "Enable detailed Warp module-load timings in the console for diagnosis; "
            "disable for quieter startup output"
        ),
    )
    enable_effect_authoring: BoolProperty(
        name="Enable custom effect authoring (developer)",
        default=False,
        description=(
            "Show the developer Warp-code authoring workspace and per-session edit "
            "trust for Custom Effects. Enable this only to author kernels; artists "
            "use the effect stack, which stays available whether this is on or off"
        ),
    )
    developer_warp_path: StringProperty(
        name="External Warp Module Directory",
        subtype="DIR_PATH",
        default="",
        description=(
            "Trusted local-development override used only when the process sets "
            "FLUIDSIM_ALLOW_EXTERNAL_WARP=1"
        ),
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Runtime", icon="GPU")

        blocked = foreign_warp_block_reason()
        foreign = bool(blocked) and (
            "does not own" in blocked or "user site-packages" in blocked
        )
        if foreign:
            row = box.row()
            row.alert = True
            row.label(text="Remove the other Warp copy first", icon="ERROR")
            box.label(text=blocked)
            row = box.row()
            row.scale_y = 1.5
            row.operator("fluidsim.install_warp", text="Install Warp", icon="IMPORT")
        else:
            warp_installed, warp_info = get_installed_warp_info()
            status = owned_install_status()
            if not warp_installed or status is None:
                row = box.row()
                row.alert = True
                row.label(text="Warp is not installed", icon="ERROR")
                row = box.row()
                row.scale_y = 1.5
                row.operator("fluidsim.install_warp", text="Install Warp", icon="IMPORT")
                box.label(
                    text=(
                        f"Downloads certified NVIDIA Warp {CERTIFIED_WARP_VERSION} "
                        "(CUDA 12) from PyPI."
                    )
                )
            else:
                box.label(text=warp_info)
                box.prop(self, "device")
                probe = probe_runtime()
                selected = configured_device()
                ready, message = runtime_status_for_device(selected, probe=probe)
                row = box.row(align=True)
                row.label(text=message, icon="CHECKMARK" if ready else "ERROR")
                row.operator("fluidsim.refresh_runtime_probe", text="", icon="FILE_REFRESH")
                if probe.module_file:
                    box.label(text=f"Warp: {probe.warp_version} · {probe.module_file}")
                if status == "certified" and not ready and probe.last_error:
                    row = box.row()
                    row.alert = True
                    repair = row.operator(
                        "fluidsim.install_warp",
                        text="Repair Warp",
                        icon="FILE_REFRESH",
                    )
                    repair.force_reinstall = True
                if status == "unsupported":
                    row = box.row()
                    row.alert = True
                    row.operator(
                        "fluidsim.install_warp",
                        text="Install Certified Warp",
                        icon="IMPORT",
                    )

        authoring = layout.box()
        authoring.label(text="Custom Effects", icon="MODIFIER")
        authoring.prop(self, "enable_effect_authoring")
        authoring.label(
            text="Off: artists use the effect stack. On: the Warp-code workspace and per-session edit trust appear.",
            icon="INFO",
        )

        if os.environ.get("FLUIDSIM_ALLOW_EXTERNAL_WARP") == "1":
            developer = layout.box()
            developer.label(text="Developer Runtime Override", icon="CONSOLE")
            developer.prop(self, "developer_warp_path")
            developer.label(text="Trusted local-development override; normal releases ignore this path.")


CLASSES = (
    FLUIDSIM_OT_install_warp,
    FLUIDSIM_OT_refresh_runtime_probe,
    FLUIDSIM_OT_redetect_device,
    FLUIDSIM_AP_preferences,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
