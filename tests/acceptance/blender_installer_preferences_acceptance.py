"""Validate installed Preferences icons and native draw before Warp/CUDA setup.

Run in foreground on certified Blender with -- --addon-module bl_ext.<repo>.<id>.
--diagnostic-background checks the actual RNA enums only; it never certifies UI.
Use private absolute BLENDER_USER_{CONFIG,SCRIPTS,EXTENSIONS,DATAFILES} paths.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path

import bpy

_INSTALL_OPERATORS = {"fluidsim.install_warp", "fluidsim.install_warp_from_file"}


class _ObservedLayout:
    """Forward native layout calls and record only completed operator creation."""

    def __init__(self, layout, operators):
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_operators", operators)

    def __getattr__(self, name):
        return getattr(self._layout, name)

    def __setattr__(self, name, value):
        setattr(self._layout, name, value)

    def row(self, *args, **kwargs):
        return _ObservedLayout(self._layout.row(*args, **kwargs), self._operators)

    def operator(self, identifier, *args, **kwargs):
        properties = self._layout.operator(identifier, *args, **kwargs)
        self._operators.add(identifier)
        return properties


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
parser.add_argument("--diagnostic-background", action="store_true")
parser.add_argument("--json-out", default=str(ROOT / "artifacts/installer/preferences.json"))
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
report = {"blender": bpy.app.version_string, "certified_runtime": bpy.app.version[:3] == (5, 2, 0),
          "diagnostic": args.diagnostic_background, "draws": [], "passed": False}
output = Path(args.json_out)
output.parent.mkdir(parents=True, exist_ok=True)


def save():
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


try:
    sys.path.insert(0, str(ROOT))
    # Importing the product module is intentionally permitted before installation.
    preferences = importlib.import_module(args.addon_module + ".preferences")
    icons = {item.identifier for item in
             bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items}
    package_root = Path(preferences.__file__).parent
    literals = set()
    for source in package_root.rglob("*.py"):
        if "tests" in source.relative_to(package_root).parts:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg != "icon":
                continue
            # Include both branches of conditional literal icons.
            values = [node.value]
            while values:
                value = values.pop()
                if isinstance(value, ast.IfExp):
                    values.extend((value.body, value.orelse))
                elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                    literals.add(value.value)
    report["literal_icons"] = sorted(literals)
    report["invalid_icons"] = sorted(literals - icons)
    assert not report["invalid_icons"], report["invalid_icons"]
    assert "SYSTEM" in icons
    if args.diagnostic_background:
        report["passed"] = True
        save()
        print("INSTALLER-PREFERENCES-RNA-DIAGNOSTIC-PASS")
    else:
        assert not bpy.app.background, "Native Preferences draw requires foreground Blender"
        assert report["certified_runtime"], "Certified foreground acceptance requires Blender 5.2.0"
        assert args.addon_module.startswith("bl_ext."), "Install the extension before this lane"
        import addon_utils
        addon_utils.enable(args.addon_module, default_set=False)
        assert args.addon_module in bpy.context.preferences.addons
        assert not preferences.get_installed_warp_info()[0], "Start the no-Warp lane with empty owned storage"
        original_draw = preferences.FLUIDSIM_AP_preferences.draw
        original_buttons = preferences._draw_install_buttons
        state = {"controls": False, "operators": set(), "error": "", "ticks": 0}

        def buttons(box, text="Install Warp"):
            operators = set()
            original_buttons(_ObservedLayout(box, operators), text)
            state["operators"].update(operators)
            state["controls"] = _INSTALL_OPERATORS <= state["operators"]

        def draw(self, context):
            try:
                state["controls"] = False
                state["operators"] = set()
                original_draw(self, context)
                report["draws"].append({"controls": state["controls"],
                                        "operators": sorted(state["operators"])})
            except Exception as exc:
                state["error"] = repr(exc)
                raise

        preferences._draw_install_buttons = buttons
        preferences.FLUIDSIM_AP_preferences.draw = draw
        bpy.context.preferences.active_section = "ADDONS"
        bpy.ops.screen.userpref_show("INVOKE_DEFAULT")
        bpy.ops.preferences.addon_expand(module=args.addon_module)

        def finish():
            state["ticks"] += 1
            if state["error"] or state["controls"] or state["ticks"] >= 40:
                report["error"] = state["error"]
                report["passed"] = bool(state["controls"] and report["draws"] and not state["error"])
                save()
                print("INSTALLER-PREFERENCES-ACCEPTANCE-" + ("PASS" if report["passed"] else "FAIL"))
                bpy.ops.wm.quit_blender()
                return None
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
            return 0.25

        bpy.app.timers.register(finish, first_interval=0.25)
except Exception as exc:
    report["error"] = repr(exc)
    save()
    raise
