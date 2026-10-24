"""Cache-management operators refuse when there is no cache or a bake owns it.

Review finding, 2026-10-24: the cache operators had no ``poll``, so "Free
Entire Cache..." was offered (Deck wheel, Properties Manage, sidebar Storage)
with no cache on disk and while a Production Bake was writing the cache. The
Deck wheel shows the poll message as the reason a control is unavailable.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

DESTRUCTIVE = (
    "FLUIDSIM_OT_cache_free",
    "FLUIDSIM_OT_cache_move",
    "FLUIDSIM_OT_cache_rename",
    "FLUIDSIM_OT_cache_remove_orphans",
)
READ_ONLY = ("FLUIDSIM_OT_cache_integrity_scan", "FLUIDSIM_OT_cache_open_folder")
NO_CACHE = "No cache to manage"


class _Operator:
    poll_message = None

    @classmethod
    def poll_message_set(cls, message):
        cls.poll_message = message


def _load(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.data = SimpleNamespace(filepath="C:/saved.blend", scenes=None)
    fake_bpy.types = SimpleNamespace(Operator=_Operator)
    fake_props = types.ModuleType("bpy.props")
    for name in ("BoolProperty", "EnumProperty", "IntProperty", "StringProperty",
                 "FloatProperty", "PointerProperty", "CollectionProperty"):
        setattr(fake_props, name, lambda **_kwargs: None)
    fake_bpy.props = fake_props
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_props)
    for name in ("cache", "common"):
        monkeypatch.delitem(
            sys.modules, f"fluidSim_warp_blender51.operator_domains.{name}", raising=False)
    cache_ops = importlib.import_module("fluidSim_warp_blender51.operator_domains.cache")
    from fluidSim_warp_blender51.blender import production_bake

    monkeypatch.setattr(production_bake, "_ACTIVE_BAKES", {})
    return cache_ops, production_bake


class _Scene:
    def __init__(self, token, cache_path):
        self._token = token
        self.fluidsim = SimpleNamespace(cache_path=str(cache_path), status="")

    def as_pointer(self):
        return self._token


def _poll(cache_ops, class_name, scene):
    cls = getattr(cache_ops, class_name)
    cls.poll_message = None
    return cls.poll(SimpleNamespace(scene=scene)), cls.poll_message


def _bake_in(production_bake, scene):
    controller = SimpleNamespace(
        status=production_bake.BakeStatus.RUNNING, _reserved_cache_path=None,
        backend=SimpleNamespace(base_dir=None))
    production_bake.register_active_bake(scene, controller)


def _cache(tmp_path):
    frames = tmp_path / "cache" / "frames"
    frames.mkdir(parents=True)
    (frames / "frame_000001.npz").write_bytes(b"")
    return tmp_path / "cache"


@pytest.mark.parametrize("class_name", DESTRUCTIVE + READ_ONLY)
@pytest.mark.parametrize("layout", ["missing", "empty", "unrelated"])
def test_no_cache_disables_every_cache_operator(monkeypatch, tmp_path, class_name, layout):
    cache_ops, _production_bake = _load(monkeypatch)
    path = tmp_path / "cache"
    if layout != "missing":
        path.mkdir()
    if layout == "unrelated":
        (path / "notes.txt").write_text("not a cache", encoding="utf-8")

    assert _poll(cache_ops, class_name, _Scene(1, path)) == (False, NO_CACHE)


@pytest.mark.parametrize("class_name", DESTRUCTIVE + READ_ONLY)
def test_unresolvable_cache_path_disables_every_cache_operator(monkeypatch, class_name):
    cache_ops, _production_bake = _load(monkeypatch)

    assert _poll(cache_ops, class_name, _Scene(1, "")) == (False, NO_CACHE)


@pytest.mark.parametrize("class_name", DESTRUCTIVE + READ_ONLY)
def test_an_existing_cache_enables_every_cache_operator(monkeypatch, tmp_path, class_name):
    cache_ops, _production_bake = _load(monkeypatch)

    assert _poll(cache_ops, class_name, _Scene(1, _cache(tmp_path))) == (True, None)


@pytest.mark.parametrize("class_name", DESTRUCTIVE)
def test_a_bake_in_this_scene_disables_destructive_cache_operators(
        monkeypatch, tmp_path, class_name):
    cache_ops, production_bake = _load(monkeypatch)
    scene = _Scene(1, _cache(tmp_path))
    _bake_in(production_bake, scene)

    assert _poll(cache_ops, class_name, scene) == (False, "Production Bake owns this scene")


@pytest.mark.parametrize("class_name", DESTRUCTIVE)
def test_a_bake_in_another_scene_disables_destructive_cache_operators(
        monkeypatch, tmp_path, class_name):
    cache_ops, production_bake = _load(monkeypatch)
    _bake_in(production_bake, _Scene(2, tmp_path / "other"))

    assert _poll(cache_ops, class_name, _Scene(1, _cache(tmp_path))) == (
        False, "A Production Bake is running in another scene")


@pytest.mark.parametrize("class_name", READ_ONLY)
def test_read_only_cache_operators_stay_available_during_a_bake(
        monkeypatch, tmp_path, class_name):
    cache_ops, production_bake = _load(monkeypatch)
    scene = _Scene(1, _cache(tmp_path))
    _bake_in(production_bake, scene)

    assert _poll(cache_ops, class_name, scene) == (True, None)
