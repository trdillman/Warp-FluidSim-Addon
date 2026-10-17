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
from .warp_discovery import developer_override_active
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
    if probe.available:
        from .runtime.prewarm_timer import register_timer

        register_timer()
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


class FLUIDSIM_OT_install_warp_from_file(bpy.types.Operator):
    """Install the certified NVIDIA Warp wheel from a file on this computer."""

    bl_idname = "fluidsim.install_warp_from_file"
    bl_label = "Install Warp from File"
    bl_description = (
        "Install the certified NVIDIA Warp wheel from a .whl file you downloaded "
        "yourself. FluidSim checks the file's size and SHA-256 before installing. "
        "No internet connection needed"
    )
    bl_options = {"INTERNAL"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.whl", options={"HIDDEN"})

    def execute(self, context):
        if not self.filepath:
            self.report({"ERROR"}, "Choose the Warp .whl file")
            return {"CANCELLED"}
        result = install_warp(
            wheel_file=bpy.path.abspath(self.filepath),
            progress_callback=lambda msg: self.report({"INFO"}, msg),
        )
        if result.success:
            self.report({"INFO"}, result.message)
            if not result.already_installed:
                _refresh_runtime(_preferences(context))
            return {"FINISHED"}
        self.report({"ERROR"}, result.message)
        return {"CANCELLED"}

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def _draw_install_buttons(box, text: str = "Install Warp") -> None:
    row = box.row(align=True)
    row.scale_y = 1.5
    row.operator("fluidsim.install_warp", text=text, icon="IMPORT")
    row.operator("fluidsim.install_warp_from_file", text="Install from File", icon="FILEBROWSER")


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
    deck_mode: EnumProperty(
        name="Liquid Deck",
        items=(
            ("AUTO", "Auto",
             "Show the Liquid Deck in new files and files saved by FluidSim 4.0 or later; "
             "files saved by 3.x open without it"),
            ("ON", "On", "Show the Liquid Deck wherever the scene's Liquid Deck toggle is on"),
            ("OFF", "Off", "Never show the Liquid Deck; the panels work as before"),
        ),
        default="ON",
        description=(
            "Choose when the in-viewport Liquid Deck (bar, palette, wheel and handles) "
            "appears; Off hides it everywhere without changing any simulation setting"
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

        active_override = developer_override_active(
            str(getattr(self, "developer_warp_path", "") or "")
        )
        blocked = foreign_warp_block_reason()
        foreign = not active_override and bool(blocked) and (
            "does not own" in blocked or "user site-packages" in blocked
        )
        if foreign:
            row = box.row()
            row.alert = True
            row.label(text="Remove the other Warp copy first", icon="ERROR")
            box.label(text=blocked)
            _draw_install_buttons(box)
        else:
            warp_installed, warp_info = get_installed_warp_info()
            status = owned_install_status()
            if not active_override and (not warp_installed or status is None):
                row = box.row()
                row.alert = True
                row.label(text="Warp is not installed", icon="ERROR")
                _draw_install_buttons(box)
                box.label(
                    text=(
                        f"Downloads certified NVIDIA Warp {CERTIFIED_WARP_VERSION} "
                        "(CUDA 13; needs an NVIDIA R580 or newer driver) from PyPI."
                    )
                )
                box.label(
                    text=(
                        "On a slow or offline connection, download the warp-lang "
                        f"{CERTIFIED_WARP_VERSION} wheel from pypi.org and use Install from File."
                    )
                )
            else:
                if active_override:
                    box.label(text="External Warp (authorized developer override)")
                    status = None  # External code is never an owned repair target.
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
                    _draw_install_buttons(box, "Install Certified Warp")

        authoring = layout.box()
        authoring.label(text="Custom Effects", icon="MODIFIER")
        authoring.prop(self, "enable_effect_authoring")
        authoring.label(
            text="Off: artists use the effect stack. On: the Warp-code workspace and per-session edit trust appear.",
            icon="INFO",
        )

        deck = layout.box()
        deck.label(text="Liquid Deck", icon="WINDOW")
        deck.prop(self, "deck_mode", expand=True)
        deck.label(
            text="Command palette: Ctrl+K. Radial wheel: Alt+Right Mouse (3D View).",
            icon="INFO",
        )

        if os.environ.get("FLUIDSIM_ALLOW_EXTERNAL_WARP") == "1":
            developer = layout.box()
            developer.label(text="Developer Runtime Override", icon="CONSOLE")
            developer.prop(self, "developer_warp_path")
            developer.label(text="Trusted local-development override; normal releases ignore this path.")


CLASSES = (
    FLUIDSIM_OT_install_warp,
    FLUIDSIM_OT_install_warp_from_file,
    FLUIDSIM_OT_refresh_runtime_probe,
    FLUIDSIM_OT_redetect_device,
    FLUIDSIM_AP_preferences,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    from .unregister_guard import unregister_classes

    unregister_classes(CLASSES)
