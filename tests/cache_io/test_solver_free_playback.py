"""Solver-free playback: the extension registers, reads a cache and publishes a frame
without importing Warp (3.0 interchange spec, section C).

Three pins:

* the add-on's registration table reaches Warp through no top-level import
  (static, transitive);
* with ``warp`` masked out, cache playback reads the fixture cache and hands the
  frame to output publication (run in a subprocess so the mask cannot leak);
* the prewarm timer does nothing when the runtime probe says CUDA is unavailable.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "fluidSim_warp_blender51"
PACKAGE = REPO / PACKAGE_NAME
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from tiny_cache import SETTINGS_SIGNATURE, build_tiny_cache  # noqa: E402


# --------------------------------------------------- static import closure ----

def _module_path(dotted: str) -> Path | None:
    """File for a module name below the package, or None when it is not ours."""
    parts = dotted.split(".")
    if parts[0] != PACKAGE_NAME:
        return None
    base = PACKAGE.joinpath(*parts[1:])
    if base.is_dir():
        return base / "__init__.py"
    candidate = base.with_suffix(".py")
    return candidate if candidate.is_file() else None


def _top_level_imports(path: Path, dotted: str) -> tuple[set[str], bool]:
    """``(package modules imported at import time, imports warp at import time)``.

    Walks module-level statements (``if``/``try``/``with``/class bodies included,
    function bodies excluded, ``if TYPE_CHECKING`` skipped).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = dotted if path.name == "__init__.py" else dotted.rpartition(".")[0]
    found: set[str] = set()
    imports_warp = False

    def resolve(level: int, module: str | None) -> str:
        if level == 0:
            return module or ""
        base = package.split(".")
        base = base[: len(base) - (level - 1)]
        return ".".join(base + ([module] if module else []))

    def walk(statements):
        nonlocal imports_warp
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "warp":
                        imports_warp = True
                    found.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                target = resolve(node.level, node.module)
                if target.split(".")[0] == "warp":
                    imports_warp = True
                found.add(target)
                for alias in node.names:
                    found.add(f"{target}.{alias.name}")
            elif isinstance(node, ast.If):
                if "TYPE_CHECKING" not in ast.unparse(node.test):
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, (ast.Try, ast.With, ast.ClassDef)):
                for field in ("body", "orelse", "finalbody"):
                    walk(getattr(node, field, []))
                for handler in getattr(node, "handlers", []):
                    walk(handler.body)

    walk(tree.body)
    return found, imports_warp


def _import_closure(roots):
    """Map every package module imported (transitively) at import time to
    whether it imports Warp itself."""
    seen: dict[str, bool] = {}
    queue = list(roots)
    while queue:
        dotted = queue.pop()
        if dotted in seen:
            continue
        path = _module_path(dotted)
        if path is None:
            continue
        # Importing a.b.c also imports the parents' __init__ modules.
        parts = dotted.split(".")
        for end in range(1, len(parts)):
            queue.append(".".join(parts[:end]))
        imported, imports_warp = _top_level_imports(path, dotted)
        seen[dotted] = imports_warp
        queue.extend(imported)
    return seen


def _registered_modules():
    """Modules of every registering step in ``registration.STEPS``."""
    tree = ast.parse((PACKAGE / "registration.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Step"
                and len(node.args) >= 3
                and isinstance(node.args[2], ast.Constant)
                and node.args[2].value is not None):
            modules.append(f"{PACKAGE_NAME}.{node.args[1].value}")
    return modules


def test_registration_table_reaches_warp_through_no_top_level_import():
    modules = _registered_modules()
    assert f"{PACKAGE_NAME}.blender.interchange" in modules
    assert f"{PACKAGE_NAME}.blender.playback" in modules
    closure = _import_closure([PACKAGE_NAME, f"{PACKAGE_NAME}.registration", *modules])
    offenders = sorted(name for name, imports_warp in closure.items() if imports_warp)
    assert not offenders, f"importing Warp at register time: {offenders}"
    assert len(closure) > 20, "the closure walk found almost nothing"
    assert f"{PACKAGE_NAME}.sim.solver" not in closure


# ------------------------------------------------ known gap, pinned loudly ----

def test_playback_signature_check_does_not_import_warp():
    source = (PACKAGE / "blender" / "solver_settings.py").read_text(encoding="utf-8")
    build = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "build_settings")
    imported = [
        (node.module or "") if isinstance(node, ast.ImportFrom)
        else ",".join(alias.name for alias in node.names)
        for node in ast.walk(build)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        name.split(".")[-1] == "solver" or name.split(".")[0] == "warp"
        for name in imported), imported


# --------------------------------------- masked-import playback (subprocess) ----

_MASKED_SCRIPT = r'''
import importlib
import importlib.abc
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


class BlockWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ImportError("warp is masked")
        return None


sys.meta_path.insert(0, BlockWarp())

bpy = MagicMock()
bpy.app.handlers.persistent = lambda function: function
bpy.path.abspath = lambda value: value
sys.modules["bpy"] = bpy
sys.modules["mathutils"] = MagicMock()

cache_dir, package, signature = sys.argv[1:4]
cache = importlib.import_module(package + ".io.cache")
playback = importlib.import_module(package + ".blender.playback")
output_publication = importlib.import_module(package + ".blender.output_publication")
cache_identity = importlib.import_module(package + ".blender.cache_identity")

published = []
output_publication.update_outputs_from_cache_frame = (
    lambda scene, frame_data: published.append(
        [int(frame_data["frame"]), len(frame_data["surface_verts"])])
    or SimpleNamespace(warning=""))
# The signature check needs real scene settings (and today Warp): see the xfail test.
cache_identity.settings_signature = lambda scene: signature


class Session:
    errors = []

    def set_error(self, scene, text):
        self.errors.append(text)

    def clear_error(self, scene):
        pass


fluidsim = SimpleNamespace(
    live=True, cache_playback_enabled=True, cache_use_custom_range=False,
    cache_path=cache_dir, status="")
scene = SimpleNamespace(frame_current=2, fluidsim=fluidsim)
playback.handle_frame_change(Session(), scene, None)

print(json.dumps({
    "published": published,
    "errors": Session.errors,
    "status": fluidsim.status,
    "warp_loaded": [n for n in sys.modules if n == "warp" or n.startswith("warp.")],
}))
'''


def test_cache_frame_plays_with_warp_masked_out(tmp_path):
    base = build_tiny_cache(tmp_path / "cache", frame_count=3)
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONNOUSERSITE="1")
    done = subprocess.run(
        [sys.executable, "-c", _MASKED_SCRIPT, str(base), PACKAGE_NAME,
         SETTINGS_SIGNATURE],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["warp_loaded"] == []
    assert result["errors"] == [], result["status"]

    from fluidSim_warp_blender51.io import cache

    expected = len(cache.read_frame(str(base), 2)["surface_verts"])
    assert result["published"] == [[2, expected]]
    assert result["status"].startswith("Cache frame 2")


# ------------------------------------------------------------- prewarm gate ----

def _prewarm_module(monkeypatch, available):
    from fluidSim_warp_blender51.runtime import prewarm_timer, warp_runtime

    calls = []
    monkeypatch.setattr(
        warp_runtime, "probe_runtime",
        lambda force=False: types.SimpleNamespace(available=available))
    monkeypatch.setattr(
        warp_runtime, "prewarm_simulation_kernels",
        lambda device=None: calls.append(device) or True)
    return prewarm_timer, calls


def test_prewarm_is_skipped_when_the_probe_says_cuda_is_unavailable(monkeypatch):
    prewarm_timer, calls = _prewarm_module(monkeypatch, available=False)
    assert prewarm_timer._background_prewarm() is None
    assert calls == []


def test_prewarm_runs_when_the_probe_succeeds(monkeypatch):
    prewarm_timer, calls = _prewarm_module(monkeypatch, available=True)
    assert prewarm_timer._background_prewarm() is None
    assert calls == [None]
