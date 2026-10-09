"""Blender acceptance: a .vdb from io/vdb_writer.py loads as a Volume.

Writes sphere level sets (one centred, one off-centre with a non-zero world
origin so an axis swap or a wrong transform shows up), loads each as a Volume,
meshes it with a Volume to Mesh modifier and checks the bounding box.

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
RADIUS, VOXEL = 0.5, 0.05


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_case(writer, sphere, out, tag, center, origin):
    values, coords, _ = sphere.sphere_leaves(
        RADIUS, VOXEL, center=center, origin=origin)
    path = out / f"{tag}.vdb"
    writer.write_level_set_vdb(str(path), values, coords, VOXEL, origin=origin)
    print(f"[{tag}] wrote {path} ({path.stat().st_size} bytes, {len(coords)} leaves)")

    vol = bpy.data.volumes.new(tag)
    vol.filepath = str(path)
    ok = vol.grids.load()
    print(f"[{tag}] grids.load() -> {ok}; error_message={vol.grids.error_message!r}")
    names = [g.name for g in vol.grids]
    print(f"[{tag}] grids={names} is_loaded={vol.grids.is_loaded}")
    assert "surface" in names, f"grid 'surface' missing: {names}"
    grid = vol.grids["surface"]
    print(f"[{tag}] surface: data_type={grid.data_type} channels={grid.channels} "
          f"matrix_object={[list(r) for r in grid.matrix_object]}")
    assert grid.data_type == "FLOAT", grid.data_type
    assert vol.grids.is_loaded

    obj = bpy.data.objects.new(tag, vol)
    bpy.context.scene.collection.objects.link(obj)
    mesh_obj = bpy.data.objects.new(tag + "_mesh", bpy.data.meshes.new(tag))
    bpy.context.scene.collection.objects.link(mesh_obj)
    mod = mesh_obj.modifiers.new("v2m", "VOLUME_TO_MESH")
    mod.object = obj
    mod.grid_name = "surface"
    mod.threshold = 0.0
    mod.resolution_mode = "GRID"
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = mesh_obj.evaluated_get(dg)
    mesh = ev.to_mesh()
    nverts = len(mesh.vertices)
    print(f"[{tag}] volume-to-mesh verts={nverts}")
    assert nverts > 0, "Volume to Mesh produced no vertices"
    co = np.empty(nverts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(axis=0), co.max(axis=0)
    c = np.asarray(center)
    err = max(np.abs(lo - (c - RADIUS)).max(), np.abs(hi - (c + RADIUS)).max()) / VOXEL
    print(f"[{tag}] bbox min={lo.tolist()} max={hi.tolist()} error={err:.3f} voxels")
    ev.to_mesh_clear()
    assert err <= 1.0, f"[{tag}] bbox error {err:.3f} voxels > 1"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = Path(argv[0] if argv else ROOT / "build" / "vdb_acceptance")
    out.mkdir(parents=True, exist_ok=True)
    writer = _load("vdb_writer_under_test",
                   "fluidSim_warp_blender51/io/vdb_writer.py")
    sphere = _load("vdb_sphere_under_test", "tests/support/vdb_sphere.py")
    run_case(writer, sphere, out, "sphere", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    run_case(writer, sphere, out, "offset", (0.3, -0.2, 0.1), (-1.0, -1.5, -0.25))
    print("VDB_WRITER_OK")


try:
    main()
except BaseException:
    traceback.print_exc()
    print("VDB_WRITER_FAIL")
    sys.exit(1)
