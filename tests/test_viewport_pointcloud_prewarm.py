"""Regression test for B6's point-cloud half: the first live write after
``ensure_outputs`` must not force ``bpy.data.pointclouds.new``/``.remove()``
from inside ``frame_change_post`` (see viewport._new_prewarmed_pointcloud).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeAttributes(dict):
    def new(self, name, type_, domain):
        attr = SimpleNamespace(name=name, type=type_, domain=domain)
        self[name] = attr
        return attr


class _FakePointCloud:
    """Minimal bpy.types.PointCloud stand-in covering what _needs_realloc reads."""

    def __init__(self, name):
        self.name = name
        self.users = 1
        self._points = []
        self.attributes = _FakeAttributes()

    def resize(self, count):
        self._points = [SimpleNamespace() for _ in range(count)]

    @property
    def points(self):
        return self._points


def _import_viewport(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.data = SimpleNamespace(
        pointclouds=SimpleNamespace(new=lambda name: _FakePointCloud(name)))
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop("fluidSim_warp_blender51.blender.viewport", None)
    return importlib.import_module("fluidSim_warp_blender51.blender.viewport")


def test_prewarmed_pointcloud_has_schema_and_capacity(monkeypatch):
    viewport = _import_viewport(monkeypatch)

    pc = viewport._new_prewarmed_pointcloud("FluidSim Particles")

    assert pc.attributes.get("position") is not None
    assert pc.attributes.get("radius") is not None
    assert pc.attributes.get(viewport.COLOR_ATTR) is None
    assert len(pc.points) == viewport._MIN_POINT_CAPACITY


def test_prewarmed_pointcloud_first_live_write_does_not_realloc(monkeypatch):
    viewport = _import_viewport(monkeypatch)
    pc = viewport._new_prewarmed_pointcloud("FluidSim Particles")

    # A typical first frame writes far fewer points than the headroom
    # capacity -- this is exactly the write ensure_outputs pre-warms for.
    assert viewport._needs_realloc(pc, 100, wants_color=False) is False


def test_bare_pointcloud_without_prewarm_still_needs_realloc(monkeypatch):
    """Baseline: an un-prewarmed cloud (the pre-patch state) does realloc.

    Pins the behaviour _new_prewarmed_pointcloud is closing over -- if this
    ever starts returning False, _needs_realloc changed underneath the fix
    and the prewarm test above would stop proving anything.
    """
    viewport = _import_viewport(monkeypatch)
    pc = viewport.bpy.data.pointclouds.new("FluidSim Particles")

    assert viewport._needs_realloc(pc, 100, wants_color=False) is True
