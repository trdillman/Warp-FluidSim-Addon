"""Blender acceptance: a .vdb from io/vdb_writer.py loads as a Volume.

Run (headless, Blender 5.2):
  blender -b --factory-startup --python-use-system-env \
      --python tests/acceptance/blender_vdb_writer_acceptance.py -- [out_dir]
Prints VDB_WRITER_OK on success.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = Path(argv[0] if argv else ROOT / "build" / "vdb_acceptance")
    out.mkdir(parents=True, exist_ok=True)
    writer = _load("vdb_writer_under_test",
                   "fluidSim_warp_blender51/io/vdb_writer.py")
    sphere = _load("vdb_sphere_under_test", "tests/support/vdb_sphere.py")
    radius, voxel = 0.5, 0.05
    values, coords, _ = sphere.sphere_leaves(radius, voxel)
    path = out / "sphere.vdb"
    writer.write_level_set_vdb(str(path), values, coords, voxel)
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(coords)} leaves)")

    vol = bpy.data.volumes.new("vdb_sphere")
    vol.filepath = str(path)
    ok = vol.grids.load()
    print(f"grids.load() -> {ok}; error_message={vol.grids.error_message!r}")
    names = [g.name for g in vol.grids]
    print(f"grids={names} is_loaded={vol.grids.is_loaded}")
    assert "surface" in names, f"grid 'surface' missing: {names}"
    grid = vol.grids["surface"]
    print(f"surface: data_type={grid.data_type} channels={grid.channels} "
          f"matrix_object={[list(r) for r in grid.matrix_object]}")
    assert grid.data_type == "FLOAT", grid.data_type
    assert vol.grids.is_loaded

    obj = bpy.data.objects.new("vdb_sphere", vol)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("v2m", "VOLUME_TO_MESH")
    mod.grid_name = "surface"
    mod.threshold = 0.0
    mod.resolution_mode = "GRID"
    mod.use_smooth_shade = True
    mesh_obj = obj
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    mesh = ev.to_mesh()
    nverts = len(mesh.vertices)
    print(f"volume-to-mesh verts={nverts}")
    assert nverts > 0, "Volume to Mesh produced no vertices"
    co = np.empty(nverts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(axis=0), co.max(axis=0)
    err = max(np.abs(lo + radius).max(), np.abs(hi - radius).max()) / voxel
    print(f"bbox min={lo.tolist()} max={hi.tolist()} error={err:.3f} voxels")
    ev.to_mesh_clear()
    assert err <= 1.0, f"bbox error {err:.3f} voxels > 1"
    print("VDB_WRITER_OK")


try:
    main()
except BaseException:
    traceback.print_exc()
    print("VDB_WRITER_FAIL")
    sys.exit(1)
