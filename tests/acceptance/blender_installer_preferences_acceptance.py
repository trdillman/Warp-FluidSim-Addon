"""Validate installed Preferences icons and native draw before Warp/CUDA setup.

Run all scenarios through run_installer_preferences_matrix.py using a built
extension and the exact locked local host wheel; each scenario gets a private
profile. Native execution requires certified foreground Blender 5.2.0.
--diagnostic-background checks the actual RNA enums only; it never certifies UI.
Use private absolute BLENDER_USER_{CONFIG,SCRIPTS,EXTENSIONS,DATAFILES} paths.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
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

    def box(self, *args, **kwargs):
        return _ObservedLayout(self._layout.box(*args, **kwargs), self._operators)

    def column(self, *args, **kwargs):
        return _ObservedLayout(self._layout.column(*args, **kwargs), self._operators)

    def split(self, *args, **kwargs):
        return _ObservedLayout(self._layout.split(*args, **kwargs), self._operators)

    def prop(self, owner, *args, **kwargs):
        return self._layout.prop(getattr(owner, "_native_owner", owner), *args, **kwargs)

    def operator(self, identifier, *args, **kwargs):
        properties = self._layout.operator(identifier, *args, **kwargs)
        self._operators.add(identifier)
        return properties


class _PreferencesDrawProxy:
    """Observe the actual native layout while RNA props retain their true owner."""

    def __init__(self, owner, operators):
        self._native_owner = owner
        self.layout = _ObservedLayout(owner.layout, operators)

    def __getattr__(self, name):
        return getattr(self._native_owner, name)


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
parser.add_argument("--diagnostic-background", action="store_true")
parser.add_argument("--scenario", default="no-warp")
parser.add_argument("--profile-root", type=Path)
parser.add_argument("--wheel", type=Path)
parser.add_argument("--json-out", default=str(ROOT / "artifacts/installer/preferences.json"))
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
report = {"blender": bpy.app.version_string, "certified_runtime": bpy.app.version[:3] == (5, 2, 0),
          "diagnostic": args.diagnostic_background, "scenario": args.scenario,
          "draws": [], "checks": {}, "actions": [], "passed": False}
output = Path(args.json_out)
output.parent.mkdir(parents=True, exist_ok=True)


def save():
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


try:
    sys.path.insert(0, str(ROOT))
    if not args.diagnostic_background:
        from run_installer_preferences_matrix import SCENARIOS, require_owned_profile
        assert args.scenario in SCENARIOS
        assert args.profile_root is not None and args.wheel is not None, "Use the isolated matrix runner"
        require_owned_profile(args.profile_root, os.environ)
        report["checks"]["private_profile"] = True
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
        from _installer_preferences_scenarios import (
            inventory,
            prepare_scenario,
            run_actions,
        )
        from run_installer_preferences_matrix import validate_scenario_report

        installer = importlib.import_module(args.addon_module + ".warp_installer")
        extension_root = Path(preferences.__file__).resolve().parent
        assert extension_root.is_relative_to(args.profile_root.resolve() / "repository"), extension_root
        report["checks"]["installed_extension"] = True
        assert not bpy.app.online_access, "Matrix must run with --offline-mode"
        report["checks"]["network_off"] = True
        owned, foreign = prepare_scenario(installer, args.scenario, args.wheel, args.profile_root)
        report["fixture"] = {"owned": str(owned), "foreign": str(foreign), "wheel": str(args.wheel)}
        addon_utils.enable(args.addon_module, default_set=False)
        assert args.addon_module in bpy.context.preferences.addons
        foreign_reason = installer.foreign_warp_block_reason()
        if args.scenario in {"good-owned", "damaged-owned"}:
            initial_state = installer.owned_install_status() == "certified" and not foreign_reason
        elif args.scenario == "foreign-preloaded":
            initial_state = bool(foreign_reason and "which FluidSim does not own" in foreign_reason)
        else:
            initial_state = not installer.is_certified_warp_installed() and not foreign_reason
        report["checks"]["scenario_state"] = initial_state
        assert report["checks"]["scenario_state"]
        required = ({"fluidsim.refresh_runtime_probe"} if args.scenario == "good-owned" else
                    {"fluidsim.refresh_runtime_probe", "fluidsim.install_warp"}
                    if args.scenario == "damaged-owned" else _INSTALL_OPERATORS)
        original_draw = preferences.FLUIDSIM_AP_preferences.draw
        original_buttons = preferences._draw_install_buttons
        state = {"controls": False, "operators": set(), "error": "", "ticks": 0, "stage": "draw"}

        def buttons(box, text="Install Warp"):
            operators = set()
            original_buttons(_ObservedLayout(box, operators), text)
            state["operators"].update(operators)
            state["controls"] = _INSTALL_OPERATORS <= state["operators"]

        def draw(self, context):
            try:
                state["controls"] = False
                state["operators"] = set()
                original_draw(_PreferencesDrawProxy(self, state["operators"]), context)
                report["draws"].append({"controls": required <= state["operators"],
                                        "operators": sorted(state["operators"])})
            except Exception as exc:
                state["error"] = repr(exc)
                raise

        preferences._draw_install_buttons = buttons
        preferences.FLUIDSIM_AP_preferences.draw = draw
        bpy.context.preferences.active_section = "ADDONS"
        bpy.ops.screen.userpref_show("INVOKE_DEFAULT")
        bpy.ops.preferences.addon_expand(module=args.addon_module)

        def complete(error=""):
            report["error"] = error
            report["passed"] = not error
            if not error:
                try:
                    validate_scenario_report(report, args.scenario)
                except (ValueError, TypeError) as exc:
                    report["passed"] = False
                    report["error"] = repr(exc)
            save()
            print("INSTALLER-PREFERENCES-ACCEPTANCE-" + ("PASS" if report["passed"] else "FAIL"))
            bpy.ops.wm.quit_blender()

        def finish():
            state["ticks"] += 1
            try:
                if state["error"]:
                    raise AssertionError(state["error"])
                if state["ticks"] >= 80:
                    raise AssertionError("Native draw/action timeout: " + state["stage"])
                if state["stage"] == "draw" and any(
                        required <= set(item["operators"]) for item in report["draws"]):
                    if args.scenario == "cancel":
                        state["before_cancel"] = inventory(owned)
                        result = bpy.ops.fluidsim.install_warp_from_file("INVOKE_DEFAULT")
                        assert "RUNNING_MODAL" in result, result
                        state["stage"] = "picker"
                    else:
                        checks, actions = run_actions(
                            bpy, installer, preferences, args.scenario, args.wheel, owned, foreign)
                        report["checks"].update(checks)
                        report["actions"].extend(actions)
                        loaded = sys.modules.get("warp")
                        report["loaded_warp"] = {"path": getattr(loaded, "__file__", None),
                                                 "version": getattr(loaded, "__version__", None)}
                        if args.scenario in {"offline-local", "repeated-clicks"}:
                            state["actions_draw_count"] = len(report["draws"])
                            state["stage"] = "post-install"
                        else:
                            complete()
                            return None
                elif state["stage"] == "post-install":
                    newer = report["draws"][state["actions_draw_count"]:]
                    if any("fluidsim.refresh_runtime_probe" in item["operators"] for item in newer):
                        report["checks"]["post_install_draw"] = True
                        complete()
                        return None
                elif state["stage"] == "picker":
                    for window in bpy.context.window_manager.windows:
                        for area in window.screen.areas:
                            if area.type == "FILE_BROWSER":
                                with bpy.context.temp_override(window=window, area=area):
                                    result = bpy.ops.file.cancel()
                                assert "FINISHED" in result, result
                                report["checks"]["picker_cancelled"] = True
                                report["actions"].append({"operator": "file.cancel", "result": sorted(result)})
                                state["stage"] = "cancelled"
                                return 0.25
                elif (state["stage"] == "cancelled"
                      and not any(area.type == "FILE_BROWSER"
                                  for window in bpy.context.window_manager.windows for area in window.screen.areas)):
                    report["checks"]["files_preserved"] = inventory(owned) == state["before_cancel"]
                    assert report["checks"]["files_preserved"]
                    complete()
                    return None
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        area.tag_redraw()
                return 0.25
            except Exception as exc:  # noqa: BLE001 - preserve timer failures in the report
                complete(repr(exc))
                return None

        bpy.app.timers.register(finish, first_interval=0.25)
except Exception as exc:
    report["error"] = repr(exc)
    save()
    raise
