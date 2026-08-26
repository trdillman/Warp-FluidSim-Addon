"""Dependency-direction contracts for low-level cache IO."""

from __future__ import annotations

import importlib
import importlib.abc
import sys

import numpy as np


class _RejectBlenderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("fluidSim_warp_blender51.blender"):
            raise AssertionError(f"io.cache imported Blender adapter {fullname}")
        return None


def test_io_dependency_cache_import_and_hashing_need_no_blender_modules(monkeypatch):
    """Pure record hashing must remain usable when Blender adapters are absent."""
    package = importlib.import_module("fluidSim_warp_blender51")
    for name in tuple(sys.modules):
        if name.startswith("fluidSim_warp_blender51.blender"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delattr(package, "blender", raising=False)
    monkeypatch.delitem(
        sys.modules, "fluidSim_warp_blender51.io.cache", raising=False)
    reject_blender = _RejectBlenderImports()
    sys.meta_path.insert(0, reject_blender)
    try:
        cache = importlib.import_module("fluidSim_warp_blender51.io.cache")
        settings = {
            "device": "cuda:0",
            "flip_ratio": 0.95,
            "voxel_size": 0.1,
        }
        scene_objects = ({
            "name": "Source",
            "properties": {
                "emit_rate": 2.5,
                "role": "SOURCE",
                "source_status_message": "ignored",
            },
            "matrix_world": (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            "vertex_count": 2,
        },)
        source_records = ({
            "identity": "source-fixed",
            "name_full": "Source",
            "library": "",
            "role": "SOURCE",
            "properties": {"emit_rate": 2.5, "role": "SOURCE"},
            "matrix_world": scene_objects[0]["matrix_world"],
            "geometry": {
                "kind": "MESH",
                "vertex_count": 2,
                "vertex_bytes": np.array(
                    [[1, 2, 3], [4, 5, 6]], dtype=np.float32
                ).tobytes(),
                "polygon_count": 1,
            },
        },)

        assert cache.scene_key_from_records(settings, scene_objects) == (
            "f66de3ed6fb1e4f4b669b86d393fabe0")
        assert cache.source_signature_from_records(source_records) == (
            "a096363f31671213e0e1b0720033059b3e3cae8d1257c81cbd554ad5498eb217")
    finally:
        sys.meta_path.remove(reject_blender)
