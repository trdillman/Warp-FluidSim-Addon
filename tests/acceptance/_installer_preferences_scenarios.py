"""Profile-owned fixtures and real operator assertions for native matrix runs."""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path


def inventory(root):
    root = Path(root)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def prepare_scenario(installer, scenario, wheel, profile):
    """Mutate only this invocation's explicitly owned fixture directory."""
    info = installer.get_wheel_info_for_platform()
    if info is None or not installer._wheel_matches_lock(Path(wheel), info):
        raise ValueError("Scenario needs the exact locked host Warp wheel")
    owned = installer._default_install_target(create=False).resolve()
    if not owned.is_relative_to(Path(profile).resolve()) or owned.exists():
        raise ValueError("Scenario requires empty owned extension storage in its private profile")
    if "warp" in sys.modules:
        raise ValueError("Fresh scenario must start before Warp is imported")
    foreign = Path(profile) / "foreign-fixture"
    if scenario in {"good-owned", "damaged-owned"}:
        installer._extract_wheel(Path(wheel), owned)
        if scenario == "damaged-owned":
            (owned / "warp/__init__.py").write_text(
                'raise RuntimeError("INSTALLER_MATRIX_DAMAGED_OWNED_FIXTURE")\n', encoding="utf-8")
        installer.ensure_installed_warp_on_path()
    elif scenario == "foreign-preloaded":
        package = foreign / "warp"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            '__version__ = "1.19.0"\n', encoding="utf-8")
        sys.path.insert(0, str(foreign))
        importlib.import_module("warp")
    return owned, foreign


def invoke_operator(bpy, identifier, *, expected, message_contains="", **kwargs):
    """Real bpy.ops; ERROR-report cancellation can raise in Blender's wrapper."""
    namespace, name = identifier.split(".", 1)
    try:
        result = getattr(getattr(bpy.ops, namespace), name)(**kwargs)
    except RuntimeError as exc:
        if expected != "CANCELLED" or not message_contains or message_contains not in str(exc):
            raise
        return {"operator": identifier, "result": "CANCELLED", "message": str(exc)}
    if expected not in result:
        raise AssertionError(f"{identifier}: expected {expected}, got {result}")
    return {"operator": identifier, "result": expected, "message": ""}


def run_actions(bpy, installer, preferences, scenario, wheel, owned, foreign):
    """Use real installed product operators; no installer/runtime patching."""
    checks = {}
    actions = []
    before = inventory(owned)
    if scenario == "good-owned":
        installer.ensure_installed_warp_on_path()
        loaded = importlib.import_module("warp")
        checks["owned_loaded"] = Path(loaded.__file__).resolve().is_relative_to(owned)
        for _ in range(2):
            actions.append(invoke_operator(bpy, "fluidsim.install_warp", expected="FINISHED"))
        checks["repeated_noop"] = inventory(owned) == before
    elif scenario == "damaged-owned":
        probe = preferences.probe_runtime(force=True)
        checks["owned_failure"] = "INSTALLER_MATRIX_DAMAGED_OWNED_FIXTURE" in str(probe.last_error)
        actions.append(invoke_operator(bpy, "fluidsim.install_warp", expected="CANCELLED", message_contains="Online access required", force_reinstall=True))
        checks["repair_offline_refused"] = inventory(owned) == before
    elif scenario == "foreign-preloaded":
        foreign_before = inventory(foreign)
        actions.append(invoke_operator(bpy, "fluidsim.install_warp", expected="CANCELLED",
                                       message_contains="which FluidSim does not own"))
        actions.append(invoke_operator(bpy, "fluidsim.install_warp_from_file", expected="CANCELLED", message_contains="which FluidSim does not own", filepath=str(wheel)))
        checks["install_refused"] = not owned.exists() and bool(installer.foreign_warp_block_reason())
        checks["foreign_preserved"] = (inventory(foreign) == foreign_before and
                                       Path(sys.modules["warp"].__file__).resolve().is_relative_to(foreign))
    elif scenario in {"offline-local", "repeated-clicks"}:
        actions.append(invoke_operator(bpy, "fluidsim.install_warp_from_file", expected="FINISHED", filepath=str(wheel)))
        checks["local_install"] = installer.is_certified_warp_installed()
        loaded = importlib.import_module("warp")
        checks["owned_loaded"] = Path(loaded.__file__).resolve().is_relative_to(owned)
        if scenario == "repeated-clicks":
            installed = inventory(owned)
            for _ in range(3):
                actions.append(invoke_operator(bpy, "fluidsim.install_warp", expected="FINISHED"))
            checks["repeated_noop"] = inventory(owned) == installed
    elif scenario == "no-warp":
        actions.append(invoke_operator(bpy, "fluidsim.install_warp", expected="CANCELLED",
                                       message_contains="Online access required"))
        checks["online_refused"] = not owned.exists() and not installer.is_certified_warp_installed()
    else:
        raise ValueError(f"Cancellation is handled asynchronously, unknown action scenario {scenario}")
    if not all(checks.values()):
        raise AssertionError(f"Scenario assertions failed: {checks}")
    return checks, actions
