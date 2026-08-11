"""Regression test for B6's point-cloud half: the first live write after
``ensure_outputs`` must not force ``bpy.data.pointclouds.new``/``.remove()``
from inside ``frame_change_post`` (see output_publication._new_prewarmed_pointcloud).
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
    sys.modules.pop("fluidSim_warp_blender51.blender.output_publication", None)
    return importlib.import_module("fluidSim_warp_blender51.blender.output_publication")


def test_prewarmed_pointcloud_has_schema_and_capacity(monkeypatch):
    output_publication = _import_viewport(monkeypatch)

    pc = output_publication._new_prewarmed_pointcloud("FluidSim Particles")

    assert pc.attributes.get("position") is not None
    assert pc.attributes.get("radius") is not None
    assert pc.attributes.get(output_publication.COLOR_ATTR) is None
    assert len(pc.points) == output_publication._MIN_POINT_CAPACITY


def test_prewarmed_pointcloud_first_live_write_does_not_realloc(monkeypatch):
    output_publication = _import_viewport(monkeypatch)
    pc = output_publication._new_prewarmed_pointcloud("FluidSim Particles")

    # A typical first frame writes far fewer points than the headroom
    # capacity -- this is exactly the write ensure_outputs pre-warms for.
    assert output_publication._needs_realloc(pc, 100, wants_color=False) is False


def test_bare_pointcloud_without_prewarm_still_needs_realloc(monkeypatch):
    """Baseline: an un-prewarmed cloud (the pre-patch state) does realloc.

    Pins the behaviour _new_prewarmed_pointcloud is closing over -- if this
    ever starts returning False, _needs_realloc changed underneath the fix
    and the prewarm test above would stop proving anything.
    """
    output_publication = _import_viewport(monkeypatch)
    pc = output_publication.bpy.data.pointclouds.new("FluidSim Particles")

    assert output_publication._needs_realloc(pc, 100, wants_color=False) is True


def test_structural_replacements_are_counted_by_existing_fallback_reason(monkeypatch):
    import numpy as np

    output_publication = _import_viewport(monkeypatch)

    class Values:
        def foreach_set(self, _name, _values):
            pass

    class Attribute:
        def __init__(self):
            self.data = Values()

    class Attributes(dict):
        def new(self, name, _type, _domain):
            attr = Attribute()
            self[name] = attr
            return attr

    class PointCloud:
        def __init__(self, capacity, *, color=False):
            self.users = 1
            self.points = [None] * capacity
            self.attributes = Attributes(
                position=Attribute(), radius=Attribute())
            if color:
                self.attributes[output_publication.COLOR_ATTR] = Attribute()
            self.materials = []

        def resize(self, count):
            self.points = [None] * count

        def update_tag(self):
            pass

    class Object(dict):
        type = "POINTCLOUD"
        name = "counted"

        def __init__(self, cloud):
            super().__init__()
            self.data = cloud

    output_publication.bpy.data.pointclouds = SimpleNamespace(
        new=lambda _name: PointCloud(0), remove=lambda _cloud: None)
    output_publication.bpy.types = SimpleNamespace(
        Mesh=type("Mesh", (), {}), PointCloud=PointCloud)
    output_publication.structural_replacement_counts(reset=True)

    output_publication._swap_points(
        Object(PointCloud(1024)), np.zeros((1025, 3), np.float32), 0.1)
    output_publication._swap_points(
        Object(PointCloud(4096)), np.zeros((1000, 3), np.float32), 0.1)
    output_publication._swap_points(
        Object(PointCloud(1024)), np.zeros((100, 3), np.float32), 0.1,
        colors=np.ones((100, 4), np.float32))

    counts = output_publication.structural_replacement_counts()
    assert counts["growth"] == 1
    assert counts["shrink"] == 1
    assert counts["schema"] == 1
    assert counts["ownership"] == counts["forced"] == 0
