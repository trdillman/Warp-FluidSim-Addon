"""Cache and export operators report failures instead of raising or losing the cache path."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class _Operator:
    reports: list

    @classmethod
    def poll_message_set(cls, _message):
        pass

    def report(self, kinds, message):
        self.reports.append((set(kinds), message))


def _load(monkeypatch, *, filepath="C:/saved.blend"):
    bpy = types.ModuleType("bpy")
    bpy.path = SimpleNamespace(abspath=lambda value: value)
    bpy.data = SimpleNamespace(filepath=filepath, scenes=None)
    bpy.types = SimpleNamespace(Operator=_Operator)
    props = types.ModuleType("bpy.props")
    for name in ("BoolProperty", "EnumProperty", "IntProperty", "StringProperty",
                 "FloatProperty", "PointerProperty", "CollectionProperty"):
        setattr(props, name, lambda **_kwargs: None)
    bpy.props = props
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    package = importlib.import_module("fluidSim_warp_blender51.operator_domains")
    for name in ("cache", "common", "export"):
        monkeypatch.delitem(
            sys.modules, f"fluidSim_warp_blender51.operator_domains.{name}", raising=False)
        # ``from . import common`` prefers the package attribute over sys.modules.
        monkeypatch.delattr(package, name, raising=False)
    cache_ops = importlib.import_module("fluidSim_warp_blender51.operator_domains.cache")
    export_ops = importlib.import_module("fluidSim_warp_blender51.operator_domains.export")
    return cache_ops, export_ops


def _operator(cls, **attributes):
    operator = cls()
    operator.reports = []
    for name, value in attributes.items():
        setattr(operator, name, value)
    return operator


def _context(cache_path):
    scene = SimpleNamespace(fluidsim=SimpleNamespace(cache_path=str(cache_path), status=""))
    return SimpleNamespace(scene=scene)


def test_a_move_whose_source_cleanup_failed_still_repoints_the_scene(monkeypatch, tmp_path):
    cache_ops, _ = _load(monkeypatch)
    from fluidSim_warp_blender51.blender import cache_manager

    source = tmp_path / "cache"
    destination = tmp_path / "cache_moved"
    source.mkdir()

    def move_then_fail(_source, target):
        # The verified copy is published and the source is gone; only the
        # leftover cleanup raises, exactly as move_cache does on a locked file.
        source.rmdir()
        target.mkdir()
        raise RuntimeError("cache copied successfully, but source cleanup remains at x")

    monkeypatch.setattr(cache_manager, "move_cache", move_then_fail)
    context = _context(source)
    operator = _operator(
        cache_ops.FLUIDSIM_OT_cache_move,
        destination=str(destination), expected_source=str(source.resolve()))

    assert operator.execute(context) == {"FINISHED"}
    assert context.scene.fluidsim.cache_path == str(destination.resolve())
    assert operator.reports[-1][0] == {"WARNING"}


def test_a_move_that_fails_before_publishing_leaves_the_cache_path(monkeypatch, tmp_path):
    cache_ops, _ = _load(monkeypatch)
    from fluidSim_warp_blender51.blender import cache_manager

    source = tmp_path / "cache"
    source.mkdir()

    def refuse(_source, _target):
        raise FileExistsError("destination exists")

    monkeypatch.setattr(cache_manager, "move_cache", refuse)
    context = _context(source)
    operator = _operator(
        cache_ops.FLUIDSIM_OT_cache_move,
        destination=str(tmp_path / "other"), expected_source=str(source.resolve()))

    assert operator.execute(context) == {"CANCELLED"}
    assert context.scene.fluidsim.cache_path == str(source)


@pytest.mark.parametrize("name", [
    "FLUIDSIM_OT_cache_integrity_scan",
    "FLUIDSIM_OT_cache_open_folder",
    "FLUIDSIM_OT_cache_remove_orphans",
    "FLUIDSIM_OT_cache_free",
    "FLUIDSIM_OT_cache_move",
    "FLUIDSIM_OT_cache_rename",
])
def test_an_unusable_cache_path_is_reported_not_raised(monkeypatch, name):
    # Unsaved .blend plus a relative path: resolving it raises ValueError.
    cache_ops, _ = _load(monkeypatch, filepath="")
    context = _context("//fluidsim_cache/")
    operator = _operator(
        getattr(cache_ops, name), expected_cache_path="", expected_source="",
        expected_orphans_json="[]", destination="", name="x")

    assert operator.execute(context) == {"CANCELLED"}
    assert operator.reports and operator.reports[-1][0] == {"ERROR"}


def test_a_scan_resolves_the_cache_path_the_poll_checked(monkeypatch, tmp_path):
    cache_ops, _ = _load(monkeypatch)
    from fluidSim_warp_blender51.blender import cache_manager

    home = tmp_path / "home"
    (home / "fluid").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    seen = []

    def inspect(path):
        seen.append(path)
        return SimpleNamespace(healthy=True, message="ok", orphan_files=())

    monkeypatch.setattr(cache_manager, "inspect_cache", inspect)
    operator = _operator(cache_ops.FLUIDSIM_OT_cache_integrity_scan)

    assert operator.execute(_context("~/fluid")) == {"FINISHED"}
    assert seen == [(home / "fluid").resolve()]  # not the literal "~/fluid"


def test_export_refuses_an_unsaved_file_with_a_relative_cache_path(monkeypatch):
    _, export_ops = _load(monkeypatch, filepath="")
    context = _context("//fluidsim_cache/")
    operator = _operator(export_ops.FLUIDSIM_OT_export_cache_sequence)

    assert operator.execute(context) == {"CANCELLED"}
    assert "Save the .blend file first" in context.scene.fluidsim.status


def test_preset_comparison_tolerates_float32_storage():
    """Re-applying a preset onto its own values must not count float32 rounding as a change."""
    import numpy as np

    from fluidSim_warp_blender51.presets.library import _same_setting_value

    stored = float(np.float32(0.1))  # what Blender reads back for 0.1
    assert stored != 0.1
    assert _same_setting_value(stored, 0.1)
    assert not _same_setting_value(stored, 0.2)
    assert _same_setting_value("FLIP", "FLIP")
    assert not _same_setting_value("FLIP", "APIC")
